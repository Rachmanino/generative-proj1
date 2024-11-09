
from datasets import Dataset,DatasetDict
from safetensors.torch import load_model
import torch
from torch import nn
import transformers
from datasets import load_dataset, load_from_disk
from safetensors.torch import load_model
from train import model
from tokenizer import tokenizer
from transformers import AutoConfig, AutoModelForCausalLM

from queue import PriorityQueue
import operator

import config
import json
import jsonlines

model_path = f'output/checkpoint-882/model.safetensors'
load_model(model, model_path, device=config.device)
decoder = model
test=["马夫人尖声叫道：“马大元，你来捏死我好了，我就是看不惯你这副脓包样子！半点大事也担当不起的胆小鬼！",
      "又过好一会，忽然间听到一阵嗡嗡声音。木婉清一惊，叫道：“啊哟！毒发了，我耳朵中有怪声。”钟灵：“我也有。”巴天石却道：“这不是耳中怪声，好象是有一大群蜜蜂飞来。”果然嗡嗡之声越来越响，似有千千万万蜜蜂从四面八方飞来。"]

window = 69  #用后window个token作为输入
beam_width = 10 
topk = 100  # 生成topk句后终止（出现[SEP]视为生成一句）
Qsize = 10000 # 队列中最大总句数


class BeamSearchNode(object):
    def __init__(self, previousNode, wordId, logProb, length):
        '''
        :param hiddenstate:
        :param previousNode:
        :param wordId:
        :param logProb:
        :param length:
        '''
        self.prevNode = previousNode
        self.words = wordId       #储存到该字为止的所有index
        self.logp = logProb    
        self.leng = length

    def eval(self, alpha=1.0):
        reward = 0
        # Add here a function for shaping a reward

        return self.logp / float(self.leng - 1 + 1e-6) + alpha * reward





def beam_decode(sequence):


    encodings=tokenizer(sequence,return_tensors='pt')
    input_ids=encodings.input_ids.to(config.device)

    # Number of sentence to generate
    endnodes = []
    number_required = min((topk + 1), topk - len(endnodes))

    # starting node -  previous node, word id, logp, length
    node = BeamSearchNode(None, input_ids[...,:-1], 0, len(sequence)+1)
    nodes = PriorityQueue()

    # start the queue
    nodes.put((-node.eval(), node))
    qsize = 1

    # start beam search
    while True:
        # give up when decoding takes too long
        # TODO:delete some and continue?
        if qsize > Qsize: 
            break

        # fetch the best node
        score, n = nodes.get()
        decoder_input = n.words
        if n.leng >= 256:
            continue

        if n.words[0,-1] == 4497 and n.prevNode != None:    #4497:[SEP]
            endnodes.append((score, n))
            # print("dingdong")
            # if we reached maximum # of sentences required
            if len(endnodes) >= number_required:
                break
            else:
                continue

        # decode for one step using decoder
        if decoder_input.shape[1] > window :
            _, decoder_output = decoder(decoder_input[:,-window:], labels=decoder_input[:,-window:].clone())
        else:
            _, decoder_output = decoder(decoder_input, labels=decoder_input.clone())
        decoder_output = decoder_output[:,-1,]   
        # 将output维度从[1, n.leng, len(vocab_list)]变为[1, len(vocab_list)]

        # log_prov, indexes维度为 [batch_size, beam_width] = [1, beam_width]
        log_prob, indexes = torch.topk(decoder_output, beam_width, dim=1)
        nextnodes = []

        for new_k in range(beam_width):
            # decoded_t: [1,1],通过view(1,-1)将数字tensor变为维度为[1,1]的tensor
            decoded_t = indexes[0][new_k].view(1, -1)
            # log_p, int
            log_p = log_prob[0][new_k].item() # item()将tensor数字变为int
            node = BeamSearchNode(n, torch.cat((n.words, decoded_t), 1), n.logp + log_p, n.leng + 1)
            score = -node.eval()
            nextnodes.append((score, node))

        # put them into queue
        for i in range(len(nextnodes)):
            score, nn = nextnodes[i]
            nodes.put((score, nn))
            # increase qsize
        qsize += len(nextnodes) - 1

    # choose nbest paths, back trace them
    if len(endnodes) == 0:
        endnodes = [nodes.get() for _ in range(topk)]

    # 对已终结的句子排序，排序方式可进一步改善
    maxnode = max(endnodes, key=lambda x:x[1].leng)
    return [maxnode[1].words, maxnode[1].logp]



with torch.no_grad():
    for seq in test:
        output, logp = beam_decode(seq)
        tokens=tokenizer.convert_ids_to_tokens(output[0])
        print("".join(tokens), logp)
