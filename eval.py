import torch
from torch import nn
import transformers
from datasets import load_dataset, load_from_disk
from safetensors.torch import load_model
from train import model
from tokenizer import tokenizer

import config
import json
from utils import *
# model_path = f'output/checkpoint-2500/model.safetensors'
# load_model(model, model_path, device=config.device)
load_last_ckpt('output', model, config.device)
with open('data/test.json','r',encoding='UTF-8') as f:
    dataset = json.load(f)
dataset = [p['text'] for p in dataset]
# print(dataset[:3])
encodings = tokenizer('\n\n'.join(dataset), return_tensors='pt')
# print(encodings['input_ids'].shape)

import torch
from tqdm import tqdm

max_length = config.n_positions
stride = 256
seq_len = encodings.input_ids.size(1)
print(seq_len)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(config.device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        logits=outputs.logits
        logits=logits[:,:-1,:].contiguous()
        target_ids=target_ids[:,1:].contiguous()
        # neg_log_likelihood = outputs[0]
        neg_log_likelihood = nn.functional.cross_entropy(logits.view(-1,config.vocab_size),target_ids.view(-1),reduction='mean')


    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(f'Perplexity: {ppl.item()}')
