from utils.set_configs import set_config
from tasks import get_task


def run_task(config):
    task = get_task(config)
    task_cls = task(config)
    runnable_task = task_cls(config)
    runnable_task.start()


if __name__ == '__main__':
    config = set_config()
    run_task(config)
