"""
All the lines are forked from https://github.com/hyunwoongko/transformer
"""

import torch.nn as nn

from Transformer.embedding.token_embedding import TokenEmbedding
from Transformer.embedding.positional_encoding import PostionalEncoding


class TransformerEmbedding(nn.Module):
    """
    token embedding + positional encoding(sinusoid)
    positional encoding can give positional information to network
    """

    def __init__(self, vocab_size, d_model, max_len, dropout_rate, device):
        """
        class for word embedding that include positional information
        :param vocab_size: size of vocabulary
        :param d_model: dimensions of model
        :param max_len: maximum sencentce length in vocab
        """
        super(TransformerEmbedding, self).__init__()
        self.tok_emb = TokenEmbedding(vocab_size, d_model)
        self.pos_emb = PostionalEncoding(d_model, max_len, device)
        self.dropout = nn.Dropout(p=dropout_rate)

    def forward(self, x):
        tok_emb = self.tok_emb(x)
        pos_emb = self.pos_emb(x)
        return self.dropout(tok_emb + pos_emb)