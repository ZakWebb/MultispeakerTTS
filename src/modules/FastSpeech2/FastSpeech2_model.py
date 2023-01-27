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

    def forward(self, batch, d_target=None, p_target=None, e_target=None):#src_seq, src_len, max_src_len=None, d_target=None, p_target=None, e_target=None):
        phonemes, phoneme_mask, _, _ = batch
        # Initialize Masks
        #src_mask = get_mask_from_lenths(src_len, max_src_len)

        phoneme_bool_mask = phoneme_mask == 0



        # Encoding
        encoder_output, src_embedded = self.encoder(phonemes, phoneme_bool_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_preciction, mel_len, mel_mask = self.variance_adaptor(
            encoder_output, phoneme_bool_mask, d_target, p_target, e_target)
        
        mel_prediction = self.decoder(acoustic_adaptor_output, mel_mask)

        return mel_prediction, src_embedded, d_prediction, p_prediction, e_preciction, phoneme_bool_mask, mel_mask, mel_len

    def inference(self, batch, src_len=None, max_src_len=None, return_attn=False):

        phonemes, phoneme_mask, _, _ = batch
        style_vector = None
        
        phoneme_bool_mask = phoneme_mask == 0
        
        # Encoding
        encoder_output, src_embedded, enc_slf_attn = self.encoder(phonemes, style_vector, phoneme_bool_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, \
                mel_len, mel_mask = self.variance_adaptor(encoder_output, phoneme_bool_mask)

        # Deocoding
        mel_output, dec_slf_attn = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)

        if return_attn:
            return enc_slf_attn, dec_slf_attn

        return mel_output, src_embedded, d_prediction, p_prediction, e_prediction, phoneme_bool_mask, mel_mask, mel_len
