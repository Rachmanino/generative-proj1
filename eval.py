import torch
from torch import nn
import transformers
from datasets import load_dataset, load_from_disk
from safetensors.torch import load_model
from train import model
from tokenizer import tokenizer

import config
import json
model_path = f'output/checkpoint-2890/model.safetensors'
load_model(model, model_path, device=config.device)
with open('data/test.json') as f:
    dataset = json.load(f)
dataset = [p['text'] for p in dataset]
encodings = tokenizer('\n\n'.join(dataset), return_tensors='pt')

import torch
from tqdm import tqdm

max_length = config.n_positions
stride = 512
seq_len = encodings.input_ids.size(1)

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
        neg_log_likelihood = outputs[0]

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(f'Perplexity: {ppl.item()}')
