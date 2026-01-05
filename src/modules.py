import torch 

import torch.nn as nn

from pathlib import Path


class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, num_heads, context_length, qkv_bias = False):
        super().__init__()

        self.d_in = d_in 
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.Query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.register_buffer(
            'mask', torch.triu(torch.ones((context_length, context_length)), diagonal = 1)
        )


    def forward(self, x):
        batch_size, num_tokens, emb_size = x.shape
        
        Q = self.Query(x)
        K = self.Key(x)
        V = self.Value(x)

        Q = Q.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        K = K.view(batch_size, num_tokens, self.num_heads, self.head_dim)
        V = V.view(batch_size, num_tokens, self.num_heads, self.head_dim)

        Q = Q.transpose(1, 2) 
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        att_scores = V @ K.transpose(2, 3)
        attention_mask = self.mask[ : num_tokens, : num_tokens]
        att_scores.masked_fill_(attention_mask.bool(), -torch.inf)

        context_vec = att_scores @ V
        context_vec = torch.softmax(context_vec / self.head_dim ** 0.5, dim = -1)
        context_vec = self.out_proj(context_vec.contiguous().view(batch_size, num_tokens, emb_size))
        
        return context_vec