"""
All the lines are forked from https://github.com/hyunwoongko/transformer
"""

from torch import nn

from Transformer.blocks.encoder_layer import EncoderLayer
from Transformer.embedding.transformer_embedding import TransformerEmbedding

class Encoder(nn.Module):

    def __init__(self, enc_voc_size, max_len, d_model, ffn_hidden, n_head, n_layers, dropout_rate, device):
        super(Encoder, self).__init__()
        self.emb = TransformerEmbedding(vocab_size=enc_voc_size,
                                        d_model=d_model,
                                        max_len=max_len,
                                        dropout_rate=dropout_rate,
                                        device=device)

        self.layers = nn.ModuleList([EncoderLayer(d_model=d_model,
                                                  ffn_hidden=ffn_hidden,
                                                  n_head=n_head,
                                                  dropout_rate=dropout_rate)
                                     for _ in range(n_layers)])

    def forward(self, x, src_mask):
        x = self.emb(x)

        for layer in self.layers:
            x = layer(x, src_mask)

        return x
