"""
All the lines are forked from https://github.com/hyunwoongko/transformer
"""

from torch import nn

from Transformer.layers.layer_norm import LayerNorm
from Transformer.layers.multi_head_attention import MultiHeadAttention
from Transformer.layers.position_wise_feed_forward import PositionWiseFeedForward

class DecoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, dropout_rate):
        super(DecoderLayer, self).__init__()
        self.self_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.enc_dec_attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=dropout_rate)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_hidden=ffn_hidden, dropout_rate=dropout_rate)
        self.norm3 = LayerNorm(d_model=d_model)
        self.dropout3 = nn.Dropout(p=dropout_rate)

    def forward(self, dec, enc, t_mask, s_mask):
        # 1. compute self attention
        _x = dec.copy()
        x = self.self_attention(q=dec, k=dec, v=dec, mask=t_mask)

        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        if enc is not None:
            # 3. compute attention between encoder and decoder
            _x = x
            x = self.enc_dec_attention(q=x, k=enc, v=enc, mask=s_mask)

            # 4. add and norm
            x = self.norm2(x + _x)
            x = self.dropout2(x)

        # 5. positionwise feed forward network
        _x = x.copy()
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm3(x + _x)
        x = self.dropout3(x)

        return x