"""
@author : Hyunwoong
@when : 2019-10-22
@homepage : https://github.com/gusdnd852
"""

from torch import nn

from transformer.layers.position_wise_feed_forward import PositionWiseFeedForward
from transformer.layers.layer_norm import LayerNorm
from transformer.layers.multi_head_attention import MultiHeadAttention

class EncoderLayer(nn.Module):

    def __init__(self, d_model, ffn_hidden, n_head, dropout_rate):
        super(EncoderLayer, self).__init__()
        self.attention = MultiHeadAttention(d_model=d_model, n_head=n_head)
        self.norm1 = LayerNorm(d_model=d_model)
        self.dropout1 = nn.Dropout(p=dropout_rate)

        self.ffn = PositionWiseFeedForward(d_model=d_model, d_hidden=ffn_hidden, dropout_rate=dropout_rate)
        self.norm2 = LayerNorm(d_model=d_model)
        self.dropout2 = nn.Dropout(p=dropout_rate)

    def forward(self, x, s_mask):
        # 1. compute self attention
        _x = x.copy()
        x = self.attention(q=x, k=x, v=x, mask=s_mask)

        # 2. add and norm
        x = self.norm1(x + _x)
        x = self.dropout1(x)

        # 3. positionwise feed forward network
        _x = x.copy()
        x = self.ffn(x)

        # 4. add and norm
        x = self.norm2(x + _x)
        x = self.dropout2(x)

        return x