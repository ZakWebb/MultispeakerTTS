import preprocess

_preprocessor_dict = {}


def main(config):
    preprocessor = _preprocessor_dict[config["dataset"]]

    if preprocessor == None:
        raise 