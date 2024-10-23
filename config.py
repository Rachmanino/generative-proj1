from tokenizer import tokenizer

"""The config file of the training process."""
n_epoches = 100
warmup_steps = 10
batch_size = 32
gradient_accumulation_steps = 1
lr = 1e-3
weight_decay = 0
max_seq_length = 512
warmup_steps = 10
weight_decay = 0
adam_beta1 = 0.9
adam_beta2 = 0.999
lr_scheduler_type = "linear"


"""The config file of the model."""
n_layer = 2
n_head = 2
n_vocab = tokenizer.vocab_size
n_embd = 256
p = 0.1
n_positions = 512



