import torch
import torch.nn as nn

from modules.FastSpeech2.Layers import Encoder, Decoder
from modules.FastSpeech2.Variance_Adaptor import VarianceAdaptor


class FastSpeech2_model(nn.Module):
    def __init__(self, config) -> None:
        super(FastSpeech2_model, self).__init__()
        self.encoder = Encoder(config)
        self.variance_adaptor = VarianceAdaptor(config)
        self.decoder = Decoder(config)

    def forward(self, phonemes, phoneme_mask, d_target=None, p_target=None, e_target=None):#src_seq, src_len, max_src_len=None, d_target=None, p_target=None, e_target=None):
        # Initialize Masks
        #src_mask = get_mask_from_lenths(src_len, max_src_len)
        # Encoding
        encoder_output, src_embedded = self.encoder(phonemes, phoneme_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, mel_len, mel_mask = self.variance_adaptor(
            encoder_output, phoneme_mask, d_target, p_target, e_target)
        
        mel_prediction = self.decoder(acoustic_adaptor_output, mel_mask)

        return mel_prediction, src_embedded, d_prediction, mel_mask, mel_len

    def inference(self, phonemes, phoneme_mask, src_len=None, max_src_len=None, return_attn=False):

        # Encoding
        encoder_output, src_embedded, enc_slf_attn = self.encoder(phonemes, phoneme_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, mel_len, mel_mask = self.variance_adaptor(encoder_output, phoneme_mask)

        # Deocoding
        mel_output, dec_slf_attn = self.decoder(acoustic_adaptor_output, mel_mask)

        if return_attn:
            return enc_slf_attn, dec_slf_attn

        return mel_output, src_embedded, d_prediction, mel_mask, mel_len
