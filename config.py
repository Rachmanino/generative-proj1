import torch
from tokenizer import tokenizer

n_vocab = tokenizer.vocab_size # 4497
embedding_dim = 256
hidden_dim = 512
max_seq_len = 256
num_layers = 2
num_heads = 2
dropout = 0.2

batch_size = 32
lr = 0.001
num_epochs = 10

# print(n_vocab)
