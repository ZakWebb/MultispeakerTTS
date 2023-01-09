

class BaseConversion(object):
    def __init__(self, config):
        super(BaseConversion, self).__init__()

    def convert(self, input):
        raise NotImplementedError