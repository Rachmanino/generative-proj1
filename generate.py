
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
prompt ="段誉那小子可还活在世上。"
seq_len = 300

seq = prompt
# input_ids=tokenizer.encode(prompt,return_tensors='pt').to(config.device)[:,:-1]
# print(input_ids)
# output=model.generate(input_ids,)
# output=output.squeeze()
# output=tokenizer.convert_ids_to_tokens(output)
# print(output)
for i in range(len(prompt), seq_len):
    with torch.no_grad():
        if len(seq)>100:
            seq_input=seq[-100:]
        else:
            seq_input=seq
        encodings = tokenizer(seq, return_tensors='pt')

        input_ids = encodings.input_ids.to(config.device)

        target_ids = input_ids.clone()
        logits = model(input_ids).logits
        # print(logits.shape)
        output_ids = torch.argmax(logits,dim=-1).squeeze()
        # print(input_ids,output_ids)
        tokens=tokenizer.convert_ids_to_tokens(output_ids)
        # print(tokens)
        # if tokens[-2]!='[SEP]':
        seq += tokens[-2]

print(seq)


