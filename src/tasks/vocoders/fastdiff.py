import torch
import torch.nn as nn
from pytorch_lightning import LightningModule
import os

from tasks.vocoders.base_vocoder import register_vocoder

from modules.FastDiff.FastDiff_model import FastDiffModel

@register_vocoder
class FastDiff(LightningModule):
    def __init__(self, config):
        super(FastDiff, self).__init__()

        self.lr = config["learning_rate"]
        self.b1 = config["adam_b1"]
        self.b2 = config["adam_b2"]
        self.weight_decay = config.get("weight_decay", 0)

        self.filter_length = config["filter_length"]
        self.hop_length = config["hop_length"]
        self.win_length = config["window_length"]
        self.n_mel_channels = config["n_mel_channels"]
        self.sampling_rate = config["sample_rate"]
        self.mel_fmin = config["mel_fmin"]
        self.mel_fmax = config["mel_fmax"]

        self.model = FastDiffModel(config)
        
        self.loss_fn = nn.MSELoss()

        self.beta_0 = config["beta_0"]
        self.beta_T = config["beta_T"]
        self.T = config["T"]
        noise_schedule = torch.linspace(self.beta_0, self.beta_T, self.T, device=self.device)
        self.compute_hyperparams_given_schedule(noise_schedule)

        

        


    def forward(self, batch):
        mels, _ = batch
        #audio_length = (mels.shape[-1] - 1) * self.hop_length + self.win_length
        audio_length = mels.size(-1) * self.hop_length
        ones = torch.ones(mels.size(0), 1, device=self.device)

        x = torch.normal(0., 1., size=(mels.size(0), 1, audio_length,), device=self.device)

        for n in range(self.T - 1, -1, -1):
                diffusion_steps = self.steps_infer[n] * ones
                epsilon_theta = self.model((x, mels, diffusion_steps))
                # if ddim:
                #     alpha_next = self.alpha[n] / (1 - self.beta[n]).sqrt()
                #     c1 = alpha_next / self.alpha[n]
                #     c2 = -(1 - self.alpha[n] ** 2.).sqrt() * c1
                #     c3 = (1 - alpha_next ** 2.).sqrt()
                #     x = c1 * x + c2 * epsilon_theta + c3 * epsilon_theta  # std_normal(size)
                # else:
                #     x -= self.beta[n] / torch.sqrt(1 - self.alpha[n] ** 2.) * epsilon_theta
                #     x /= torch.sqrt(1 - self.beta[n])
                #     if n > 0:
                #         x = x + sigma_infer[n] * std_normal(size)
                x -= self.beta[n] / torch.sqrt(1 - self.alpha[n] ** 2.) * epsilon_theta
                x /= torch.sqrt(1 - self.beta[n])
                if n > 0:
                    x = x + self.sigma[n] * torch.normal(0.0, 1.0, size=(audio_length,), device=self.device)
        
        return x
    
    def training_step(self, batch, batch_idx):
        loss = 1000 * self.theta_timestep_loss(batch)
        
        self.log("train_loss", loss)
        return loss

    
    def validation_step(self, batch, batch_idx):
        loss = 1000 * self.theta_timestep_loss(batch)
        
        self.log("val_loss", loss)
        return loss

    def test_step(self, batch, batch_idx):
        mels, _ = batch
        loss = 1000 * self.theta_timestep_loss(batch)

        if batch_idx < 3:
            print("{}".format(batch_idx))
            wavs = self.forward(mels)
        else:
            wavs = None

        
        returnable = {"loss": loss, "gen_wavs" : wavs}

        self.log("test_loss", loss)
        return returnable

    def test_epoch_end(self, outputs) -> None:
        previous = super().test_epoch_end(outputs)
        i = 0
        test_path = os.path.join(self.ckpt_dir, "Step {} wavs".format(self.global_step))
        os.makedirs(test_path, exist_ok=True)
        
        os.mkdir

        for output in output:
            if output["gen_wavs"] is not None:
                wavs = output["gen_wavs"]
                for j in range(wavs.size(0)):
                    torch.save(wavs[j,:,:], os.path.join(test_path, "{}.pt".format(i)))
                    i += 1

        return previous


    def configure_optimizers(self):
        self.opt = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.lr, betas=(self.b1, self.b2), weight_decay=self.weight_decay)
        return [self.opt], []

    def theta_timestep_loss(self, batch, reverse=False):
        """
        Compute the training loss for learning theta
        Parameters:
        batch (tuple, shape=(2,)):          training data in tuple form (mel_spectrograms, audios)
                                        mel_spectrograms: torch.tensor, shape is batchsize followed by each mel_spectrogram shape
                                        audios: torch.tensor, shape=(batchsize, 1, length of audio)
        Returns:
        theta loss
        """

        mel_spectrogram, audio = batch
        B, C, L = audio.shape  # B is batchsize, C=1, L is audio length
        ts = torch.randint(self.T, size=(B, 1, 1), device=self.device)  # randomly sample steps from 1~T
        z = torch.normal(0, 1, size=audio.shape, device=self.device)
        delta = (1 - self.alpha[ts] ** 2.).sqrt()
        alpha_cur = self.alpha[ts]
        noisy_audio = alpha_cur * audio + delta * z  # compute x_t from q(x_t|x_0)
        epsilon_theta = self.model((noisy_audio, mel_spectrogram, ts.view(B, 1)))

        z = z[:,:,:epsilon_theta.size(2)]

        if reverse:
            x0 = (noisy_audio - delta * epsilon_theta) / alpha_cur
            return self.loss_fn(epsilon_theta, z), x0

        return self.loss_fn(epsilon_theta, z)
    
    def set_schedule(self, beta):
        self.compute_hyperparams_given_schedule(beta)

    
    def compute_hyperparams_given_schedule(self, beta):
        """
        Compute diffusion process hyperparameters
        Parameters:
        beta (tensor):  beta schedule
        Returns:
        a dictionary of diffusion hyperparameters including:
            T (int), beta/alpha/sigma (torch.tensor on cpu, shape=(T, ))
        T   hese cpu tensors are changed to cuda tensors on each individual gpu
        """

        self.T = len(beta)
        alpha = 1 - beta
        sigma = beta + 0
        for t in range(1, self.T):
            alpha[t] *= alpha[t - 1]  # \alpha^2_t = \prod_{s=1}^t (1-\beta_s)
            sigma[t] *= (1 - alpha[t - 1]) / (1 - alpha[t])  # \sigma^2_t = \beta_t * (1-\alpha_{t-1}) / (1-\alpha_t)
        alpha = torch.sqrt(alpha)
        sigma = torch.sqrt(sigma)

        if not hasattr(self, "alpha"):
            self.register_buffer("alpha", alpha)
            self.register_buffer('beta', beta)
            self.register_buffer('sigma', sigma)
            self.register_buffer('noise_schedule', beta)
        else:
            self.alpha = alpha
            self.beta = beta
            self.sigma = sigma
            self.noise_schedule = beta

        
        # Mapping noise scales to time steps
        steps_infer = []
        for n in range(self.T):
            step = map_noise_scale_to_time_step(self.alpha[n], self.alpha)
            if step >= 0:
                steps_infer.append(step)
        self.steps_infer = torch.FloatTensor(steps_infer, device=self.device)

    def training_epoch_end(self, training_step_outputs):
        self.log("global_step", torch.tensor([self.global_step]).float().item())  # there's probably a better way to convert ints to float32s, but I odn't know it
        return super().training_epoch_end(training_step_outputs)

    

    # def sampling_given_noise_schedule(
    #         net,
    #         size,
    #         diffusion_hyperparams,
    #         inference_noise_schedule,
    #         condition=None,
    #         ddim=False,
    #         return_sequence=False):
    #     """
    #     Perform the complete sampling step according to p(x_0|x_T) = \prod_{t=1}^T p_{\theta}(x_{t-1}|x_t)
    #     Parameters:
    #     net (torch network):            the wavenet models
    #     size (tuple):                   size of tensor to be generated,
    #                                     usually is (number of audios to generate, channels=1, length of audio)
    #     diffusion_hyperparams (dict):   dictionary of diffusion hyperparameters returned by calc_diffusion_hyperparams
    #                                     note, the tensors need to be cuda tensors
    #     condition (torch.tensor):       ground truth mel spectrogram read from disk
    #                                     None if used for unconditional generation
    #     Returns:
    #     the generated audio(s) in torch.tensor, shape=size
    #     """

    #     _dh = diffusion_hyperparams
    #     T, alpha = _dh["T"], _dh["alpha"]
    #     assert len(alpha) == T
    #     assert len(size) == 3

    #     N = len(inference_noise_schedule)
    #     beta_infer = inference_noise_schedule
    #     alpha_infer = 1 - beta_infer
    #     sigma_infer = beta_infer + 0
    #     for n in range(1, N):
    #         alpha_infer[n] *= alpha_infer[n - 1]
    #         sigma_infer[n] *= (1 - alpha_infer[n - 1]) / (1 - alpha_infer[n])
    #     alpha_infer = torch.sqrt(alpha_infer)
    #     sigma_infer = torch.sqrt(sigma_infer)

    #     # Mapping noise scales to time steps
    #     steps_infer = []
    #     for n in range(N):
    #         step = map_noise_scale_to_time_step(alpha_infer[n], alpha)
    #         if step >= 0:
    #             steps_infer.append(step)
    #     print(steps_infer, flush=True)
    #     steps_infer = torch.FloatTensor(steps_infer)

    #     # N may change since alpha_infer can be out of the range of alpha
    #     N = len(steps_infer)

    #     print('begin sampling, total number of reverse steps = %s' % N)

    #     x = std_normal(size)
    #     if return_sequence:
    #         x_ = copy.deepcopy(x)
    #         xs = [x_]
    #     with torch.no_grad():
    #         for n in range(N - 1, -1, -1):
    #             diffusion_steps = (steps_infer[n] * torch.ones((size[0], 1))).cuda()
    #             epsilon_theta = net((x, condition, diffusion_steps,))
    #             if ddim:
    #                 alpha_next = alpha_infer[n] / (1 - beta_infer[n]).sqrt()
    #                 c1 = alpha_next / alpha_infer[n]
    #                 c2 = -(1 - alpha_infer[n] ** 2.).sqrt() * c1
    #                 c3 = (1 - alpha_next ** 2.).sqrt()
    #                 x = c1 * x + c2 * epsilon_theta + c3 * epsilon_theta  # std_normal(size)
    #             else:
    #                 x -= beta_infer[n] / torch.sqrt(1 - alpha_infer[n] ** 2.) * epsilon_theta
    #                 x /= torch.sqrt(1 - beta_infer[n])
    #                 if n > 0:
    #                     x = x + sigma_infer[n] * std_normal(size)
    #             if return_sequence:
    #                 x_ = copy.deepcopy(x)
    #                 xs.append(x_)
    #     if return_sequence:
    #         return xs
    #     return x


def map_noise_scale_to_time_step(alpha_infer, alpha):
    if alpha_infer < alpha[-1]:
        return len(alpha) - 1
    if alpha_infer > alpha[0]:
        return 0
    for t in range(len(alpha) - 1):
        if alpha[t+1] <= alpha_infer <= alpha[t]:
             step_diff = alpha[t] - alpha_infer
             step_diff /= alpha[t] - alpha[t+1]
             return t + step_diff.item()
    return -1