

class BasePreprocess():
    def __init__(self, config):
        self.data_folder = config["data_folder"]
        self.output_folder = config["output_folder"]
        self.sample_rate = config["sample_rate"]
    
    def build_folders(self):
        raise NotImplemented

    def build_files(self):
        raise NotImplemented