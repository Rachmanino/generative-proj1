import torch
from torch import nn
from datasets import Dataset, load_dataset, load_from_disk
from trl import SFTConfig, SFTTrainer
from transformers import BertTokenizer, DataCollatorForLanguageModeling, TextDataset
from datasets import Dataset,DatasetDict
from transformers import GPT2Config
from transformers import GPT2LMHeadModel
from model import DecoderLM
from tokenizer import tokenizer
import config

from utils import *

dataset = load_from_disk("data/prepared_dataset")

# model = DecoderLM(
#     embedding_dim=config.embedding_dim,
#     n_head=config.n_head,
#     n_layer=config.n_layer,
#     hidden_dim=config.hidden_dim,
#     p=config.p,
#     vocab_size=config.vocab_size,
#     max_seq_length=config.max_seq_length
# )
gpt2config_vanilla_ntp=GPT2Config(
    vocab_size=config.vocab_size,
    n_positions=512,
    n_embd=512,
    n_layer=10,
    n_head=8,
    bos_token_id=4498,#cls
    eos_token_id=4497,#sep
)
model=GPT2LMHeadModel(gpt2config_vanilla_ntp)

training_args = SFTConfig( #TODO: check the arguments carefully
    output_dir="output",
    do_train=True,
    do_eval=True,
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
    bf16=torch.cuda.is_bf16_supported(),
    logging_dir = 'logs',
    logging_steps=50, # log to the stdout every x steps
    dataset_text_field='text',
    save_steps=5000 # save the model every x steps
)

train_dataset=Dataset.from_json('data/train.json')
test_dataset=Dataset.from_json('data/test.json')
dataset_dict=DatasetDict({
    'train':train_dataset,
    'test':test_dataset
})
# print(dataset_dict['train'])
trainer = SFTTrainer(
    model = model.to(config.device),
    args = training_args,
    train_dataset = dataset_dict['train'],
    tokenizer = tokenizer,
    preprocess_logits_for_metrics=nn.CrossEntropyLoss()
)

if __name__ == "__main__":
    print(f'#Parameters: {count_params(model)}')
    print('Start training...')
    
    trainer.train()