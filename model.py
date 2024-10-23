import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy

from tokenizer  import tokenizer

class DecoderLM(nn.Module):
    'A decoder-only Transformer model for language modeling. '

    def __init__(
        self, 
        n_vocab, 
        embedding_dim, 
        hidden_dim, 
        max_seq_len, 
        num_layers, 
        num_heads, 
        dropout
    ):
        super(DecoderLM, self).__init__()
        self.embedding = nn.Embedding(n_vocab, embedding_dim)
        self.pos_embedding = nn.Embedding(max_seq_len, embedding_dim) # positional embedding
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(
                nn.TransformerDecoderLayer(embedding_dim, 
                                           num_heads, 
                                           hidden_dim, 
                                           dropout,
                                           activation=F.gelu) # use GeLU activation in GPT-2
            )
        self.fc = nn.Linear(embedding_dim, n_vocab)
    
    def forward(self, x, mask):
        """
        x: (batch_size, seq_len)
        mask: (seq_len, seq_len)
        output: (batch_size, seq_len, n_vocab)
        """
        x = self.embedding(x) + self.pos_embedding(torch.arange(x.size(1)).to(x.device))
        for layer in self.layers:
            x = layer(x, mask)
        x = self.fc(x)
        return x
    