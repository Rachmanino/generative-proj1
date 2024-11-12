from tokenizer import tokenizer
import torch

"""The config file of the training process."""
n_epoches = 100
warmup_steps = 10
batch_size = 16
gradient_accumulation_steps = 4
lr = 1e-4
weight_decay = 0.1
max_seq_length = 256
adam_beta1 = 0.9
adam_beta2 = 0.99
lr_scheduler_type = "linear"


"""The config file of the model."""
n_layer = 8
n_head = 4
n_vocab = tokenizer.vocab_size
embedding_dim = 512
hidden_dim = 1024
p = 0.1
n_positions = 256

"""Others"""
device = "cuda" if torch.cuda.is_available() else "cpu" # 暂时单卡训练
vocab_size = tokenizer.vocab_size+10



