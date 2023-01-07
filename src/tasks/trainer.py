
import torch
import os
import tqdm
import subprocess
import logging
import copy


from torch.cuda.amp import GradScaler, autocast

class Trainer:
    def __init__(self):
        pass

    def run_single_process(self, task):
        """ Sanity check a few things before actual training

        """
        # build model, optm, and load checkpoint
        model = task.build_model()
        if model is not None:
            task.model = model
        if not self.testing:
            self.optimizers = task.configure_optimizers()
            self.first_epoch = True

        checkpoint, _ = get_last_checkpoint(self.work_dir, self.resume_from_checkpoint)
        if checkpoint is not None:
            self.restore_weights(checkpoint)
            if not self.testing:
                self.restore_opt_state(checkpoint)
        elif self.on_gpu:
            task.cuda(self.root_gpu)
        del checkpoint

        # clear cache after restore
        if self.on_gpu:
            torch.cuda.empty_cache()

        # link up experiment object
        task_ref = self.get_task_ref()
        task_ref.trainer = self
        task_ref.testing = self.testing
        if self.proc_rank == 0:
            task_ref.build_tensorboard(save_dir=self.work_dir, name='lightning_logs', version='lastest')
        else:
            os.makedirs('tmp', exist_ok=True)
            task_ref.build_tensorboard(save_dir='tmp', name='tb_tmp', version='lastest')
        self.logger = task_ref.logger
        try:
            if self.testing:
                self.run_evaluation(test=True)
            else:
                self.train()
        except KeyboardInterrupt as e:
            task_ref.on_keyboard_interrupt()

    ####################
    # valid and test
    ####################
    def run_evaluation(self, test=False):
        eval_results = self.evaluate(self.task, test, tqdm_desc='Valid' if not test else 'test')
        if eval_results is not None and 'tb_log' in eval_results:
            tb_log_output = eval_results['tb_log']
            self.log_metrics_to_tb(tb_log_output)
        if self.proc_rank == 0 and not test:
            self.save_checkpoint(epoch=self.current_epoch, logs=eval_results)

    def evaluate(self, task, test=False, tqdm_desc='Valid', max_batches=None):
        # enable eval mode
        task.zero_grad()
        task.eval()
        torch.set_grad_enabled(False)

        task_ref = self.get_task_ref()
        if test:
            ret = task_ref.test_start()
            if ret == 'EXIT':
                return

        outputs = []
        dataloader = task_ref.test_dataloader() if test else task_ref.val_dataloader()
        pbar = tqdm.tqdm(dataloader, desc=tqdm_desc, total=max_batches, dynamic_ncols=True, unit='step',
                         disable=self.root_gpu > 0)
        for batch_idx, batch in enumerate(pbar):
            if batch is None:  # pragma: no cover
                continue
            # stop short when on fast_dev_run (sets max_batch=1)
            if max_batches is not None and batch_idx >= max_batches:
                break

            # make dataloader_idx arg in validation_step optional
            if self.on_gpu:
                batch = move_to_cuda(batch, self.root_gpu)
            args = [batch, batch_idx]
            if test:
                output = task_ref.test_step(*args)
            else:
                output = task_ref.validation_step(*args)
            # track outputs for collation
            outputs.append(output)
        # give model a chance to do something with the outputs (and method defined)
        if test:
            eval_results = task_ref.test_end(outputs)
        else:
            eval_results = task_ref.validation_end(outputs)
        # enable train mode again
        task.train()
        torch.set_grad_enabled(True)
        return eval_results
    
    ####################
    # train
    ####################
    def train(self):
        task_ref = self.get_task_ref()
        task_ref.on_train_start()
        if self.num_sanity_val_steps > 0:
            # run tiny validation (if validation defined) to make sure program won't crash during val
            self.evaluate(self.task, False, 'Sanity Val', max_batches=self.num_sanity_val_steps)
        # clear cache before training
        if self.on_gpu:
            torch.cuda.empty_cache()
        dataloader = task_ref.train_dataloader()
        epoch = self.current_epoch
        # run all epochs
        while True:
            # update training progress in trainer and model
            task_ref.current_epoch = epoch
            self.current_epoch = epoch
            # total batches includes multiple val checks
            self.batch_loss_value = 0  # accumulated grads
            # before epoch hook
            task_ref.on_epoch_start()

            # run epoch
            train_pbar = tqdm.tqdm(dataloader, initial=self.global_step, total=float('inf'),
                                   dynamic_ncols=True, unit='step', disable=self.root_gpu > 0)
            for batch_idx, batch in enumerate(train_pbar):
                pbar_metrics, tb_metrics = self.run_training_batch(batch_idx, batch)
                train_pbar.set_postfix(**pbar_metrics)
                should_check_val = (self.global_step % self.val_check_interval == 0
                                    and not self.first_epoch)
                if should_check_val:
                    self.run_evaluation()
                self.first_epoch = False
                # when metrics should be logged
                if (self.global_step + 1) % self.tb_log_interval == 0:
                    # logs user requested information to logger
                    self.log_metrics_to_tb(tb_metrics)

                self.global_step += 1
                task_ref.global_step = self.global_step
                if self.global_step > self.max_updates:
                    print("| Training end..")
                    break
            # epoch end hook
            task_ref.on_epoch_end()
            epoch += 1
            if self.global_step > self.max_updates:
                break
        task_ref.on_train_end()

    def run_training_batch(self, batch_idx, batch):
        if batch is None:
            return {}
        all_progress_bar_metrics = []
        all_log_metrics = []
        task_ref = self.get_task_ref()
        for opt_idx, optimizer in enumerate(self.optimizers):
            if optimizer is None:
                continue
            # make sure only the gradients of the current optimizer's paramaters are calculated
            # in the training step to prevent dangling gradients in multiple-optimizer setup.
            if len(self.optimizers) > 1:
                for param in task_ref.parameters():
                    param.requires_grad = False
                for group in optimizer.param_groups:
                    for param in group['params']:
                        param.requires_grad = True

            # forward pass
            with autocast(enabled=self.amp):
                if self.on_gpu:
                    batch = move_to_cuda(copy.copy(batch), self.root_gpu)
                args = [batch, batch_idx, opt_idx]
                output = task_ref.training_step(*args)
                loss = output['loss']
                if loss is None:
                    continue
                progress_bar_metrics = output['progress_bar']
                log_metrics = output['tb_log']
                # accumulate loss
                loss = loss / self.accumulate_grad_batches

            # backward pass
            if loss.requires_grad:
                if self.amp:
                    self.amp_scalar.scale(loss).backward()
                else:
                    loss.backward()

            # track progress bar metrics
            all_log_metrics.append(log_metrics)
            all_progress_bar_metrics.append(progress_bar_metrics)

            if loss is None:
                continue

            # nan grads
            if self.print_nan_grads:
                has_nan_grad = False
                for name, param in task_ref.named_parameters():
                    if (param.grad is not None) and torch.isnan(param.grad.float()).any():
                        print("| NaN params: ", name, param, param.grad)
                        has_nan_grad = True
                if has_nan_grad:
                    exit(0)

            # gradient update with accumulated gradients
            if (self.global_step + 1) % self.accumulate_grad_batches == 0:
                task_ref.on_before_optimization(opt_idx)
                if self.amp:
                    self.amp_scalar.step(optimizer)
                    self.amp_scalar.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()
                task_ref.on_after_optimization(self.current_epoch, batch_idx, optimizer, opt_idx)

        # collapse all metrics into one dict
        all_progress_bar_metrics = {k: v for d in all_progress_bar_metrics for k, v in d.items()}
        all_log_metrics = {k: v for d in all_log_metrics for k, v in d.items()}
        return all_progress_bar_metrics, all_log_metrics

    ####################
    # load and save checkpoint
    ####################
    def restore_weights(self, checkpoint):
        # load model state
        task_ref = self.get_task_ref()

        if len([k for k in checkpoint['state_dict'].keys() if '.' in k]) > 0:
            task_ref.load_state_dict(checkpoint['state_dict'])
        else:
            for k, v in checkpoint['state_dict'].items():
                getattr(task_ref, k).load_state_dict(v)

        if self.on_gpu:
            task_ref.cuda(self.root_gpu)
        # load training state (affects trainer only)
        self.best_val_results = checkpoint['checkpoint_callback_best']
        self.global_step = checkpoint['global_step']
        self.current_epoch = checkpoint['epoch']
        task_ref.global_step = self.global_step

    def restore_opt_state(self, checkpoint):
        if self.testing:
            return
        # restore the optimizers
        optimizer_states = checkpoint['optimizer_states']
        for optimizer, opt_state in zip(self.optimizers, optimizer_states):
            if optimizer is None:
                return
            try:
                optimizer.load_state_dict(opt_state)
                # move optimizer to GPU 1 weight at a time
                if self.on_gpu:
                    for state in optimizer.state.values():
                        for k, v in state.items():
                            if isinstance(v, torch.Tensor):
                                state[k] = v.cuda(self.root_gpu)
            except ValueError:
                print("| WARMING: optimizer parameters not match !!!")
        did_restore = True
        return did_restore

    def save_checkpoint(self, epoch, logs=None):
        monitor_op = np.less
        ckpt_path = f'{self.work_dir}/model_ckpt_steps_{self.global_step}.ckpt'
        logging.info(f'Epoch {epoch:05d}@{self.global_step}: saving model to {ckpt_path}')
        self._atomic_save(ckpt_path)
        for old_ckpt in get_all_ckpts(self.work_dir)[self.num_ckpt_keep:]:
            subprocess.check_call(f'rm -rf "{old_ckpt}"', shell=True)
            logging.info(f'Delete ckpt: {os.path.basename(old_ckpt)}')
        current = None
        if logs is not None and self.monitor_key in logs:
            current = logs[self.monitor_key]
        if current is not None and self.save_best:
            if monitor_op(current, self.best_val_results):
                best_filepath = f'{self.work_dir}/model_ckpt_best.pt'
                self.best_val_results = current
                logging.info(
                    f'Epoch {epoch:05d}@{self.global_step}: {self.monitor_key} reached {current:0.5f}. '
                    f'Saving model to {best_filepath}')
                self._atomic_save(best_filepath)

    def _atomic_save(self, filepath):
        checkpoint = self.dump_checkpoint()
        tmp_path = str(filepath) + ".part"
        torch.save(checkpoint, tmp_path, _use_new_zipfile_serialization=False)
        os.replace(tmp_path, filepath)

    def dump_checkpoint(self):
        checkpoint = {'epoch': self.current_epoch, 'global_step': self.global_step,
                      'checkpoint_callback_best': self.best_val_results}
        # save optimizers
        optimizer_states = []
        for i, optimizer in enumerate(self.optimizers):
            if optimizer is not None:
                optimizer_states.append(optimizer.state_dict())

        checkpoint['optimizer_states'] = optimizer_states
        task_ref = self.get_task_ref()
        checkpoint['state_dict'] = {
            k: v.state_dict() for k, v in task_ref.named_children() if len(list(v.parameters())) > 0}
        return checkpoint

         ####################
    # utils
    ####################
    def get_task_ref(self):
 #       from tasks.base_task import BaseTask
 #       task: BaseTask = self.task.module if isinstance(self.task, DDP) else self.task
        task = self.task
        return task

    def log_metrics_to_tb(self, metrics, step=None):
        """Logs the metric dict passed in.

        :param metrics:
        """
        # added metrics by Lightning for convenience
        metrics['epoch'] = self.current_epoch

        # turn all tensors to scalars
        scalar_metrics = self.metrics_to_scalars(metrics)

        step = step if step is not None else self.global_step
        # log actual metrics
        if self.proc_rank == 0:
            self.log_metrics(self.logger, scalar_metrics, step=step)

    @staticmethod
    def log_metrics(logger, metrics, step=None):
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            logger.add_scalar(k, v, step)

    def metrics_to_scalars(self, metrics):
        new_metrics = {}
        for k, v in metrics.items():
            if isinstance(v, torch.Tensor):
                v = v.item()

            if type(v) is dict:
                v = self.metrics_to_scalars(v)

            new_metrics[k] = v

        return new_metrics
