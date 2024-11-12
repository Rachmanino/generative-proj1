'''This file is modified from the origina load_tokenizer.py'''

# Example code to load the Tokenizer.
# You can also create your only tokenizer with vocab.txt.
from transformers import BertTokenizer
import torch
tokenizer = BertTokenizer(vocab_file='data/vocab.txt')

if __name__ == '__main__':
    # Example Usage
    # text = ["生成模型基础！","你好"]
    # # tokenizer.encode_plus
    # print(tokenizer(text,padding='longest',truncation=True,max_length=12)['input_ids'])
    # output:
    # {'input_ids': [2, 619, 3323, 1250, 2845, 3267, 1, 2627, 3],
    # 'token_type_ids': [0, 0, 0, 0, 0, 0, 0, 0, 0], 'attention_mask': [1, 1, 1, 1, 1, 1, 1, 1, 1]}
    # REMARK: 'token_type_ids' is useless for auto-regressive models like GPT2, please remove it if you are using GPT2.
    print(tokenizer.cls_token_id,tokenizer.sep_token_id)
