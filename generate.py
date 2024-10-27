
from safetensors.torch import load_model
import torch
from torch import nn
import transformers
from train import model # use DecoderLM from train.py
from tokenizer import tokenizer

import config
import os

from utils import *

load_last_ckpt('output', model, config.device)
prompt ="令狐冲"
seq_len = 100

seq = prompt
for i in range(len(prompt), seq_len):
    with torch.no_grad():
        encodings = tokenizer(seq, return_tensors='pt')

        input_ids = encodings.input_ids.to(config.device)

        target_ids = input_ids.clone()
        _,output = model(input_ids,labels = target_ids)
        output_ids = torch.argmax(output,dim=-1).squeeze()
        print(input_ids,output_ids)
        tokens=tokenizer.convert_ids_to_tokens(output_ids)
        print(tokens)
        seq += tokens[i]

print(seq)


