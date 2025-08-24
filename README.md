# MultispeakerTTS

This is an attempt to make a multispeaker TTS system with the ability to create random voices.  I'm still catching up to the state of the art, so things are currently in flux.  The current plan is to implement a version of CosyVoice 1.

I'll admit, I like the idea of also using an Acoustic encoder in addition to a semantic encoder, similar to the Higgs Audio stuff.

I really like DiTAR, in which they combine generative modeling and an autoregressive model.  I might instead try to implement something like this instead of CosyVoice.  I don't think the methods are completely incompatible, but I think it would be easier to modify DiTAR to add CosyVoice stuff than the other way around.  The only problem is that it doesn't look like there's a version of it online.  Oh well, that's kind of what I want.



Github References:

- [pytorch-kaldi-neural-speaker-embeddings](https://github.com/jefflai108/pytorch-kaldi-neural-speaker-embeddings)
- [soobinseo/Transformer-TTS](https://github.com/soobinseo/Transformer-TTS)
- [xcmyz/FastSpeech](https://github.com/xcmyz/FastSpeech)
- [NVIDIA/waveglow](https://github.com/NVIDIA/waveglow)
- [KevinMIN95/StyleSpeech](https://github.com/KevinMIN95/StyleSpeech)
- [jik876/hifi-gan](https://github.com/jik876/hifi-gan)
- [chatterbox](https://github.com/resemble-ai/chatterbox)
- [CosyVoice](https://github.com/FunAudioLLM/CosyVoice)
- [S3Tokenizer](https://github.com/xingchensong/S3Tokenizer)
- [FunASR](https://github.com/modelscope/FunASR)
- [FunCodec](https://github.com/modelscope/FunCodec)
- [Matcha-TTS](https://github.com/shivammehta25/Matcha-TTS)
- [AcademicCodec](https://github.com/yangdongchao/AcademiCodec)
- [StyleTTS2](https://github.com/yl4579/StyleTTS2)
- [Higgs Audio v2](https://github.com/boson-ai/higgs-audio)


Paper References:

- [WaveGlow: A Flow-based Generative Network for Speech Synthesis](https://arxiv.org/abs/1811.00002)
- [Neural Speech Synthesis with Transformer Network Paper](https://arxiv.org/abs/1809.08895)
- [FastSpeech: Fast, Robust and Controllable Text to Speech](https://arxiv.org/abs/1905.09263)
- [HiFi-GAN: Generative Adversarial Networks for Efficient and High Fidelity Speech Synthesis](https://arxiv.org/abs/2010.05646)
- [FunAudioLLM: Voice Understanding and Generation Foundation Models for Natural Interaction Between Humans and LLMs](https://arxiv.org/abs/2407.04051v2)
- [HiFi-Codec: Group-residual Vector quantization for High Fidelity Audio Codec](https://arxiv.org/abs/2305.02765)
- [StyleTTS 2: Towards Human-Level Text-to-Speech through Style Diffusion and Adversarial Training with Large Speech Language Models](https://arxiv.org/abs/2306.07691)
- [BASE TTS: Lessons from building a billion-parameter Text-to-Speech model on 100K hours of data](https://arxiv.org/abs/2402.08093)
- [Better speech synthesis through scaling](https://arxiv.org/abs/2305.07243)
- [Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers](https://arxiv.org/abs/2301.02111)
- [DiTAR: Diffusion Transformer Autoregressive Modeling for Speech Generation](https://arxiv.org/abs/2502.03930)
- [Clip-TTS: Contrastive Text-content and Mel-spectrogram, A High-Huality Text-to-Speech Method based on Contextual Semantic Understanding](https://arxiv.org/abs/2502.18889v1)


______________________________________________________________________

<div align="center">

# Your Project Name

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Paper](http://img.shields.io/badge/paper-arxiv.1001.2234-B31B1B.svg)](https://www.nature.com/articles/nature14539)
[![Conference](http://img.shields.io/badge/AnyConference-year-4b44ce.svg)](https://papers.nips.cc/paper/2020)

</div>

## Description

What it does

## Installation

#### Pip

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# [OPTIONAL] create conda environment
conda create -n myenv python=3.9
conda activate myenv

# install pytorch according to instructions
# https://pytorch.org/get-started/

# install requirements
pip install -r requirements.txt
```

#### Conda

```bash
# clone project
git clone https://github.com/YourGithubName/your-repo-name
cd your-repo-name

# create conda environment and install dependencies
conda env create -f environment.yaml -n myenv

# activate conda environment
conda activate myenv
```

## How to run

Train model with default configuration

```bash
# train on CPU
python src/train.py trainer=cpu

# train on GPU
python src/train.py trainer=gpu
```

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
python src/train.py experiment=experiment_name.yaml
```

You can override any parameter from command line like this

```bash
python src/train.py trainer.max_epochs=20 data.batch_size=64
```

I copied the main template from Lightning-Hydra-Template