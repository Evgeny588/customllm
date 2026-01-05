import torch 

import torch.nn as nn

from pathlib import Path


class MultiheadAttention(nn.Module):
    def __init__(self, d_in, d_out, dropout, num_heads, context_length, qkv_bias = False):
        super().__init__()

        assert (d_out % num_heads) == 0, 'd_out must be divisible by num_heads' 
     
        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads
        self.Query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.Value = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
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
        attention_mask = self.mask.bool()[ : num_tokens, : num_tokens]
        att_scores.masked_fill_(attention_mask, -torch.inf)

        context_vec = att_scores @ V
        context_vec = torch.softmax(context_vec / self.head_dim ** 0.5, dim = -1)
        context_vec = self.dropout(context_vec).transpose(1, 2)
        context_vec = self.out_proj(context_vec.contiguous().view(batch_size, num_tokens, emb_size))
        
        return context_vec
    


class LayerNorm(nn.Module):
    def __init__(self, emb_dim):
        super().__init__()
        self.eps = 1e-5
        self.scale = nn.Parameter(torch.ones(emb_dim))
        self.shift = nn.Parameter(torch.zeros(emb_dim))

    
    def forward(self, x):
        mean = x.mean(dim = -1, keepdim = True)
        var = x.var(dim = -1, keepdim = True, unbiased = False)
        norm_x = (x - mean) / torch.sqrt(var + self.eps)
        
        return self.scale * norm_x + self.shift




class GELU(nn.Module):
    def __init__(self):
        super().__init__()


    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(
            torch.sqrt(torch.tensor(2.0 / torch.pi)) * 
            (x + 0.044715 * torch.pow(x, 3))
        ))
    


class FeedForward(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(cfg['emb_dim'], 4 * cfg['emb_dim']),
            GELU(),
            nn.Linear(4 * cfg['emb_dim'], cfg['emb_dim'])
        )


    def forward(self, x):
        return self.layers(x)
    


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.att = MultiheadAttention(
            d_in= cfg['emb_dim'],
            d_out= cfg['emb_dim'],
            context_length= cfg['context_length'],
            num_heads= cfg['n_heads'],
            dropout= cfg['drop_rate'],
            qkv_bias= cfg['qkv_bias']
        )
        self.ff = FeedForward(cfg)
        self.norm1 = LayerNorm(cfg['emb_dim'])
        self.norm2 = LayerNorm(cfg['emb_dim'])
        self.drop_shortcut = nn.Dropout(cfg['drop_rate'])


    def forward(self, x):

        shortcut = x
        x = self.norm1(x)
        x = self.att(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        shortcut = x
        x = self.ff(x)
        x = self.drop_shortcut(x)
        x = x + shortcut

        return x
    



class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg['vocab_size'], cfg['emb_dim'])
        self.pos_emb = nn.Embedding(cfg['context_length'], cfg['emb_dim'])
        self.drop_emb = nn.Dropout(cfg['drop_rate'])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg['n_layers'])]
        )

        self.final_norm = LayerNorm(cfg['emb_dim'])
        self.out_head = nn.Linear(cfg['emb_dim'], cfg['vocab_size'], bias = False)


    def forward(self, in_idx):
        batch_size, seq_len = in_idx.shape
        tok_embeds = self.tok_emb(in_idx)
        pos_embeds = self.pos_emb(
            torch.arange(seq_len, device = in_idx.device, dtype = torch.long)
        )
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)
        x = self.trf_blocks(x)
        x = self.final_norm(x)
        logits = self.out_head(x)

        return logits
    

