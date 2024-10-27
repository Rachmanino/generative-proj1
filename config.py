from tokenizer import tokenizer
import torch

"""The config file of the training process."""
n_epoches = 10
warmup_steps = 0
batch_size = 32
gradient_accumulation_steps = 1
lr = 1e-3
weight_decay = 0
max_seq_length = 256
warmup_steps = 10
weight_decay = 0
adam_beta1 = 0.9
adam_beta2 = 0.999
lr_scheduler_type = "linear"


"""The config file of the model."""
n_layer = 2
n_head = 2
n_vocab = tokenizer.vocab_size
embedding_dim = 512
hidden_dim = 256
p = 0.1
n_positions = 256

"""Others"""
device = "cuda" if torch.cuda.is_available() else "cpu" # 暂时单卡训练
vocab_size = tokenizer.vocab_size+10



