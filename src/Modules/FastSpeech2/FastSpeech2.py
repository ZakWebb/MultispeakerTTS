import torch
import torch.nn as nn

from .Layers import Encoder, Decoder, VarianceAdaptor


class FastSpeech2(nn.Module):
    def __init__(self, config) -> None:
        super(FastSpeech2, self).__init__()
        self.encoder = Encoder(config)
        self.variance_adapter = VarianceAdaptor(config)
        self.decoder = Decoder(config)

    def forward(self, src_seq, src_len, max_src_len=None, d_target=None, p_target=None, e_target=None):
        # Initialize Masks
        src_mask = get_mask_from_lenths(src_len, max_src_len)

        # Encoding
        encoder_output, src_embedded = self.encoder(src_seq, src_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_preciction, mel_len, mel_mask = self.variance_adapter(
            encoder_output, src_mask, d_target, p_target, e_target)
        
        mel_prediction = self.decoder(acoustic_adaptor_output, mel_mask)

        return mel_prediction, src_embedded, d_prediction, p_prediction, e_preciction, src_mask, mel_mask, mel_len

    def inference(self, style_vector, src_seq, src_len=None, max_src_len=None, return_attn=False):
        src_mask = get_mask_from_lengths(src_len, max_src_len)
        
        # Encoding
        encoder_output, src_embedded, enc_slf_attn = self.encoder(src_seq, style_vector, src_mask)

        # Variance Adaptor
        acoustic_adaptor_output, d_prediction, p_prediction, e_prediction, \
                mel_len, mel_mask = self.variance_adaptor(encoder_output, src_mask)

        # Deocoding
        mel_output, dec_slf_attn = self.decoder(acoustic_adaptor_output, style_vector, mel_mask)

        if return_attn:
            return enc_slf_attn, dec_slf_attn

        return mel_output, src_embedded, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len
