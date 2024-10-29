
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

model_path = f'output/checkpoint-882/model.safetensors'
load_model(model, model_path, device=config.device)
sequence="又过好"





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


decoder = model


def beam_decode():

    beam_width = 10
    topk = 100  # how many sentence do you want to generate
    # decoded_batch = []

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
        if qsize > 5000: 
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
    utterances = []
    for score, n in sorted(endnodes, key=lambda x:x[1].leng):
        # utterance = []
        # utterance.append(n.wordid)
        # # back trace
        # while n.prevNode != None:
        #     n = n.prevNode
        #     utterance.append(n.wordid)

        # utterance = utterance[::-1]
        utterances.append(n.words)
    return utterances



with torch.no_grad():
    output = beam_decode()[-1][0]
    print(output)
    tokens=tokenizer.convert_ids_to_tokens(output)
    print("".join(tokens))
