
from datasets import Dataset,DatasetDict
from safetensors.torch import load_model
import torch
from torch import nn
import transformers
from datasets import load_dataset, load_from_disk
from safetensors.torch import load_model
from train import model
from tokenizer import tokenizer

import config
import json
# def test(input:torch.Tensor):
#     print(torch.ones_like(input))


# input_ids=torch.tensor([2, 619, 3323, 1250, 2845, 3267, 1, 2627, 3])
# position_ids = torch.cumsum(torch.ones_like(input_ids), dim=0) - 1
# print(position_ids)

# examples={'input_ids': [2, 619, 3323, 1250, 2845, 3267, 1, 2627, 3],
#     'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
# concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
# print(concatenated_examples)

# train_dataset=Dataset.from_json('data/train.json')
# test_dataset=Dataset.from_json('data/test.json')
# dataset_dict=DatasetDict({
#     'train':train_dataset,
#     'test':test_dataset
# })
# print(dataset_dict['train'])
# data_train=[{'text':example['text']}for example in data]
# print(data_train[:3])

model_path = f'output/checkpoint-8500/model.safetensors'
load_model(model, model_path, device=config.device)
sequence="令狐冲"


for i in range(3,100):
    with torch.no_grad():
        encodings=tokenizer(sequence,return_tensors='pt')

        input_ids=encodings.input_ids.to(config.device)

        target_ids = input_ids.clone()
        _,output=model(input_ids,labels=target_ids)
        output_ids=torch.argmax(output,dim=-1).squeeze()
        print(input_ids,output_ids)
        tokens=tokenizer.convert_ids_to_tokens(output_ids)
        print(tokens)
        sequence+=tokens[i]

print(sequence)



