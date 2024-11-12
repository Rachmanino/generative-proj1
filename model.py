import torch
from torch import nn
import torch.nn.functional as F
import os
import numpy
from typing import Optional
import config

class DecoderLM(nn.Module):
    def __init__(self, embedding_dim, n_head, n_layer, hidden_dim, p, vocab_size, max_seq_length):
        super(DecoderLM, self).__init__()
        self.embedding_dim = embedding_dim
        self.n_head = n_head
        self.n_layer = n_layer
        self.hidden_dim = hidden_dim
        self.p = p
        self.vocab_size = vocab_size
        self.max_seq_length = max_seq_length

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.positional_encoding = nn.Embedding(self.max_seq_length, self.embedding_dim)

        self.decoder = nn.TransformerDecoder(
            decoder_layer=nn.TransformerDecoderLayer(
                d_model=self.embedding_dim,
                nhead=self.n_head,
                dim_feedforward=self.hidden_dim,
                dropout=self.p,
                activation='gelu',
                batch_first=True,
                
            ),
            num_layers=self.n_layer,
            norm=nn.LayerNorm(self.embedding_dim)
        )
        self.decoderlayer=nn.TransformerDecoderLayer(
            d_model=self.embedding_dim,
            nhead=self.n_head,
            dim_feedforward=self.hidden_dim,
            dropout=self.p,
            activation='gelu',
            batch_first=True,
        )
        self.out_fc = nn.Linear(self.embedding_dim, self.vocab_size)

    
    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.FloatTensor] = None,
                token_type_ids: Optional[torch.Tensor] = None, # useless in our task
                labels: Optional[torch.Tensor] = None,):
        """
        The forward process of the model.
        Args:
            input_ids: torch.Tensor, shape (batch_size, seq_length)
            attention_mask: torch.Tensor, shape (batch_size, seq_length)
            token_type_ids: torch.Tensor, shape (batch_size, seq_length)
            labels: torch.Tensor, shape (batch_size, seq_length)
        Returns:
            loss: torch.Tensor, shape (1,)
            output: torch.Tensor, shape (batch_size, seq_length, vocab_size)
        """
        assert input_ids.shape[1] <= self.max_seq_length, f"input_ids.shape[1] should be less than or equal to {self.max_seq_length}"
        input_ids=input_ids[:,:-1].contiguous()
        labels=labels[:,1:].contiguous()

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        position_ids = torch.cumsum(torch.ones_like(input_ids), dim=1) - 1
        # print(input_ids,labels)
        padding_mask = (input_ids == 0).bool().to(config.device)
        padding_mask = padding_mask.float().masked_fill(padding_mask, float('-inf'))
        attn_mask=nn.Transformer.generate_square_subsequent_mask(input_ids.shape[1],config.device)
        x = self.embedding(input_ids) + self.positional_encoding(position_ids) # (B, S, D)
        x = self.decoder.forward(tgt=x, 
                                 memory=x,  
                                 tgt_mask=attn_mask,
                                 memory_mask=attn_mask,
                                 tgt_key_padding_mask=padding_mask, 
                                 memory_key_padding_mask=padding_mask,)
                                #  tgt_is_causal=True,
                                #  memory_is_causal=True) # (B, S, D)
        output = self.out_fc(x) # (B, S, V)
        # print(output.shape, labels.shape)
        # labels=labels[:,1:]
        # labels=torch.cat((labels,(4497*torch.ones(labels.shape[0],1).int()).to(config.device)),dim=1)

        loss = F.cross_entropy(output.view(-1, self.vocab_size), labels.view(-1), reduction='mean')
        return loss, output


