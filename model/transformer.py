"""
All the lines are forked from https://github.com/hyunwoongko/transformer
"""

import torch
from torch import nn

from Transformer.model.encoder import Encoder
from Transformer.model.decoder import Decoder

class Transformer(nn.Module):

    def __init__(self, src_pad_idx,
                 trg_pad_idx,
                 trg_sos_idx,
                 enc_voc_size,
                 dec_voc_size,
                 d_model,
                 n_head,
                 max_len,
                 ffn_hidden,
                 n_layers,
                 dropout_rate,
                 device):
        super(Transformer, self).__init__()
        self.src_pad_idx = src_pad_idx
        self.trg_pad_idx = trg_pad_idx
        self.trg_sos_idx = trg_sos_idx
        self.device = device

        self.encoder = Encoder(enc_voc_size=enc_voc_size,
                               max_len=max_len,
                               d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               dropout_rate=dropout_rate,
                               device=device)

        self.decoder = Decoder(dec_voc_size=dec_voc_size,
                               max_len=max_len,
                               d_model=d_model,
                               ffn_hidden=ffn_hidden,
                               n_head=n_head,
                               n_layers=n_layers,
                               dropout_rate=dropout_rate,
                               device=device)

    def forward(self, src, trg):

        enc_src = self.encoder(src)
        output = self.decoder(trg, enc_src)

        return output





















