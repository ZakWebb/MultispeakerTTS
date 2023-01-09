# I somehow need to be able to pass the output from one task to another
class BaseTask(object):
    def __init__(self) -> None:
        super(BaseTask, self).__init__()

    @classmethod
    def start(cls):
        raise NotImplementedError