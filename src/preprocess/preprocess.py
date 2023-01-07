from preprocess import BasePreprocess, get_preprocessor_cls


def main(config):
    preprocessor_cls = get_preprocessor_cls(config)

    assert preprocessor_cls != None, "Dataset preprocessor for " + config["dataset"] + " not implemented"

    preprocessor = preprocessor_cls(config)

    preprocessor.build_folders()
    preprocessor.build_files()