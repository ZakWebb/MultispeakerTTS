import torch
import torch.nn as nn
import numpy as np


from ..Common.Modules import Mish, ConvNorm, LinearNorm, get_sinusoid_encoding_table


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        ## Define Config and constants

        self.src_word_embedding = nn.Embedding(n_src_vocab, d_word_vec, padding_idx = Constants.PAD)

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

