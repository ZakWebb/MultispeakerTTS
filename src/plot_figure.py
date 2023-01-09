from utils.set_configs import set_config
from utils.draw_mel import spec_to_figure

from matplotlib import pyplot as plt
import os
import torch



def run_task(config):
    data = torch.load(config["mel_spec"])
    fig = spec_to_figure(data)
    plt.show(block=True)



if __name__ == '__main__':
    config = set_config()
    config["mel_spec"] = os.path.join(config["data_folder"],"train","mels","LJ001-0003.mel")
    run_task(config)