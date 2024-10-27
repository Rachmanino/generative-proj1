
from safetensors.torch import load_model
import torch
from torch import nn
import transformers
from train import model # use DecoderLM from train.py
from tokenizer import tokenizer

import config
import os

def load_last_ckpt(model_dir_path: str, 
                   model: nn.Module, 
                   device: str):
    """
    Load the last checkpoint of the model.
    Args:
        model_dir_path: str, the path of the model directory.
        model: nn.Module, the model to load the checkpoint.
        device: str, the device to load the model.
    """
    ckpt_list = os.listdir(model_dir_path)
    ckpt_list = [int(ckpt.split('-')[1]) for ckpt in ckpt_list if ckpt.startswith('checkpoint')]
    if ckpt_list == []:
        raise FileNotFoundError(f"No checkpoint found in {model_dir_path}!")
    ckpt_list.sort()
    last_ckpt = ckpt_list[-1]
    ckpt_path = os.path.join(model_dir_path, f'checkpoint-{last_ckpt}/model.safetensors')
    load_model(model, ckpt_path, device=device)

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


