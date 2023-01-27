import torch
import torch.nn as nn
import numpy as np


from modules.Common.Modules import Mish, ConvNorm, LinearNorm, get_sinusoid_encoding_table
from modules.Common.SubLayers import MultiHeadAttention, PositionwiseFeedForward, StyleAdaptiveLayerNorm

from data_gen.symbols import symbols, PAD

class Encoder(nn.Module):
    def __init__(self, config, n_src_vocab = len(symbols)):
        super(Encoder, self).__init__()

        ## Define Config and constants

        
        self.max_seq_len = config["max_seq_len"]
        self.n_layers = config["encoder_layer"]
        self.d_model = config["encoder_hidden"]
        self.n_head = config["encoder_head"]
        self.d_k = config["encoder_hidden"] // self.n_head
        self.d_v = config["encoder_hidden"] // self.n_head
        self.d_inner = config["fft_conv1d_filter_size"]
        self.fft_conv1d_kernel_size = config["fft_conv1d_kernel_size"]
        self.d_out = config["decoder_hidden"]
        self.style_dim = config["style_vector_dim"]
        self.dropout = config["dropout"]


        self.src_word_embedding = nn.Embedding(n_src_vocab, self.d_model, padding_idx = PAD)

        self.prenet = PreNet(self.d_model, self.d_model, self.dropout)

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)
        
        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.dropout) for _ in range(self.n_layers)])
        
        self.fc_out = nn.Linear(self.d_model, self.d_out)


    def forward(self, src_seq, mask, return_attns=False):
        batch_size, max_len = src_seq.shape[0], src_seq.shape[1]
        
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # word embedding
        src_embedded = self.src_word_emb(src_seq)
        # prenet
        src_seq = self.prenet(src_embedded, mask)
        # position encoding
        if src_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(src_seq.shape[1], self.d_model)[:src_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        enc_output = src_seq + position_embedded
        # fft blocks
        slf_attn = []
        for enc_layer in self.layer_stack:
            enc_output, enc_slf_attn = enc_layer(
                enc_output, 
                mask=mask, 
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(enc_slf_attn)
        # last fc
        enc_output = self.fc_out(enc_output)
        return enc_output, src_embedded, slf_attn

class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()

        # Include config stuff

        self.prenet = nn.Sequential(
            nn.Linear(self.d_model, self.d_model//2),
            Mish(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model//2, self.d_model)
        )

        n_position = self.max_seq_len + 1
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, self.d_model).unsqueeze(0), requires_grad = False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            self.d_model, self.d_inner, self.n_head, self.d_k, self.d_v, 
            self.fft_conv1d_kernel_size, self.dropout) for _ in range(self.n_layers)])
        
        self.fc_out = nn.Linear(self.d_model, self.d_out)
    
    def forward(self, enc_seq, mask):
        batch_size, max_len = enc_seq.shape[0], enc_seq.shape[1]
        # -- Prepare masks
        slf_attn_mask = mask.unsqueeze(1).expand(-1, max_len, -1)

        # -- Forward
        # prenet
        dec_embedded = self.prenet(enc_seq)
        # poistion encoding
        if enc_seq.shape[1] > self.max_seq_len:
            position_embedded = get_sinusoid_encoding_table(enc_seq.shape[1], self.d_model)[:enc_seq.shape[1], :].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            position_embedded = self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)
        dec_output = dec_embedded + position_embedded
        # fft blocks
        slf_attn = []
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask)
            slf_attn.append(dec_slf_attn)
        # last fc
        dec_output = self.fc_out(dec_output)
        return dec_output, slf_attn

class PreNet(nn.Module):
    ''' Prenet '''
    def __init__(self, hidden_dim, out_dim, dropout):
        super(PreNet, self).__init__()

        self.convs = nn.Sequential(
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
            ConvNorm(hidden_dim, hidden_dim, kernel_size=3),
            Mish(),
            nn.Dropout(dropout),
        )
        self.fc = LinearNorm(hidden_dim, out_dim)

    def forward(self, input, mask=None):
        residual = input
        # convs
        output = input.transpose(1,2)
        output = self.convs(output)
        output = output.transpose(1,2)
        # fc & residual
        output = self.fc(output) + residual

        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)
        return output

class PostNet(nn.Module):
    def __init__(self, config):
        super(PostNet, self).__init__()

    def forward():
        raise NotImplemented


class FFTBlock(nn.Module):
    ''' FFT Block '''
    def __init__(self, d_model,d_inner,
                    n_head, d_k, d_v, fft_conv1d_kernel_size, style_dim, dropout):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.saln_0 = StyleAdaptiveLayerNorm(d_model, style_dim)

        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, fft_conv1d_kernel_size, dropout=dropout)
        self.saln_1 = StyleAdaptiveLayerNorm(d_model, style_dim)

    def forward(self, input, style_vector, mask=None, slf_attn_mask=None):
        # multi-head self attn
        slf_attn_output, slf_attn = self.slf_attn(input, mask=slf_attn_mask)
        slf_attn_output = self.saln_0(slf_attn_output, style_vector)
        if mask is not None:
            slf_attn_output = slf_attn_output.masked_fill(mask.unsqueeze(-1), 0)

        # position wise FF
        output = self.pos_ffn(slf_attn_output)
        output = self.saln_1(output, style_vector)
        if mask is not None:
            output = output.masked_fill(mask.unsqueeze(-1), 0)

        return output, slf_attn