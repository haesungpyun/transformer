"""
All the lines are forked from https://github.com/hyunwoongko/transformer
"""

import torch
from torch import nn

from Transformer.blocks.decoder_layer import DecoderLayer
from Transformer.embedding.transformer_embedding import TransformerEmbedding

class Decoder(nn.Module):

    def __init__(self, dec_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, dropout_rate, device):
        super(Decoder, self).__init__()
        self.emb = TransformerEmbedding(vocab_size=dec_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        dropout_rate=dropout_rate,
                                        device=device)

        self.layers= nn.ModuleList([DecoderLayer(d_model=d_model,
                                                 ffn_hidden=ffn_hidden,
                                                 n_head=n_head,
                                                 dropout_rate=dropout_rate,)
                                    for _ in range(n_layers)])

        self.linear = nn.Linear(d_model, dec_voc_size)
    def forward(self, trg, enc_src, trg_mask, src_mask):
        trg = self.emb(trg)

        for layer in self.layers:
            trg = layer(trg, enc_src, trg_mask, src_mask)

        # pass to LM head
        output = self.linear(trg)

        return output