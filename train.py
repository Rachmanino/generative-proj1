import torch
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer

from model import DecoderLM
from tokenizer import tokenizer
from mydataset import MyTrainDataset
import config

#TODO: prepare the dataset
dataset = MyTrainDataset()

model = DecoderLM(
    n_vocab=config.n_vocab,
    embedding_dim=config.embedding_dim,
    hidden_dim=config.hidden_dim,
    max_seq_len=config.max_seq_length,
    num_layers=config.n_layer,
    num_heads=config.n_head,
    dropout=config.p,
)

training_args = SFTConfig( #TODO: check the arguments carefully
    output_dir="output",
    do_train=True,
    do_eval=False,
    per_device_train_batch_size=config.batch_size,
    per_device_eval_batch_size=config.batch_size,
    gradient_accumulation_steps=config.gradient_accumulation_steps,
    max_seq_length=config.max_seq_length,
    learning_rate=config.lr,
    weight_decay=config.weight_decay,
    adam_beta1=config.adam_beta1,
    adam_beta2=config.adam_beta2,
    num_train_epochs=config.n_epoches,
    lr_scheduler_type=config.lr_scheduler_type,
    warmup_steps=config.warmup_steps,
    bf16=torch.cuda.is_bf16_supported()
)

trainer = SFTTrainer(
    model = model,
    args = training_args,
    train_dataset = dataset,
    tokenizer=tokenizer,
)
trainer.train()




