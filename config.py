from tokenizer import tokenizer
import torch

"""The config file of the training process."""
n_epoches = 1000
warmup_steps = 10
batch_size = 256
gradient_accumulation_steps = 4
lr = 2e-3
weight_decay = 0.1
max_seq_length = 256
adam_beta1 = 0.9
adam_beta2 = 0.99
lr_scheduler_type = "linear"


"""The config file of the model."""
n_layer = 4
n_head = 4
n_vocab = tokenizer.vocab_size
embedding_dim = 128
hidden_dim = 512
p = 0.1
n_positions = 256

"""Others"""
device = "cuda" if torch.cuda.is_available() else "cpu" # 暂时单卡训练
vocab_size = tokenizer.vocab_size+10



