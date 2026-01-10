import torch 
import logging

import torch.nn as nn
import torch.nn.functional as F

from tqdm import tqdm
from pathlib import Path
from torch.utils.data import Dataset



small_config = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 768,
        'n_heads': 12,
        'n_layers': 12,
        'drop_rate': 0.1,
        'qkv_bias': False
        }

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
        
        att_scores = Q @ K.transpose(2, 3)
        attention_mask = self.mask.bool()[ : num_tokens, : num_tokens]
        att_scores.masked_fill_(attention_mask, -torch.inf)
        att_scores = torch.softmax(
            att_scores / self.head_dim ** 0.5, dim = -1 
        )
        att_scores = self.dropout(att_scores)

        context_vec = (att_scores @ V).transpose(1, 2)
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
        x = self.norm2(x)
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
    


class EarlyStopping:
    def __init__(self, delta, num_cycles, mode = 'loss', logging = True):
        self.delta = delta
        self.mode = mode
        self.logging = logging
        self.num_cycles = num_cycles
        self.flag = False
        self.best_value = float('inf') if mode == 'loss' else -float('inf')
        self.counter = 0

    def __call__(self, metric_or_loss):

        if self.mode == 'loss':
            if metric_or_loss < self.best_value - self.delta:
                self.best_value = metric_or_loss
                self.counter = 0
            else:
                self.counter += 1
        elif self.mode == 'metric':
            if metric_or_loss > self.best_value + self.delta:
                self.best_value = metric_or_loss
                self.counter = 0
            else:
                self.counter += 1

        if self.logging:
            print(f'--------ES counter: {self.counter}/{self.num_cycles}-------')

        if self.counter >= self.num_cycles:
            print('--------Early stop--------')
            self.flag = True
            return True
        return False
    


class TextDataset(Dataset):
    def __init__(self, text_data, window, tokenizer, slice = 1):
        self.tokens = tokenizer.encode(text_data) 
        self.window = window
        self.slice = slice

    def __len__(self):
        return len(self.tokens) - self.window - self.slice 
    

    def __getitem__(self, idx):
        start_pos = idx
        end_pos = idx + self.window

        non_sliced = self.tokens[start_pos: end_pos]
        sliced = self.tokens[start_pos + self.slice: end_pos + self.slice]

        return torch.tensor(non_sliced, dtype = torch.long), torch.tensor(sliced, dtype = torch.long)
    


class Saver:
    def __init__(self, model, optimizer, path, patience = 0.01):
        self.model = model
        self.path = path
        self.patience = patience
        self.optimizer = optimizer
        self.best_loss = float('inf')


    def __call__(self,current_loss, train_loss = None):
        if current_loss <= self.best_loss - self.patience:
            torch.save(
                {'states_dicts': {
                    'model': self.model.state_dict(),
                    'optimizer': self.optimizer.state_dict()
                },
                 'metrics': {
                     'train_loss': train_loss,
                     'val_loss': current_loss
                 }
                 }, self.path
                 
            )



def train_cycle(model, optimizer, train_loader, device, vocab_size):
    model.train()
    train_batches = 0
    train_loss = 0.0

    # Train cycle
    for inputs, targets in tqdm(train_loader, desc = 'Train:', leave = False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward
        logits = model(inputs)
        logits = logits.view(-1, vocab_size)
        targets = targets.view(-1, )
        loss = F.cross_entropy(
            input = logits,
            target = targets
        )
        
        # Backward
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_batches += 1
        train_loss += loss.item() 
    return train_loss / train_batches


def validation_cycle(model, val_loader, device, vocab_size): 
    model.eval()
    eval_batches = 0
    eval_loss = 0.0

    # Validation cycle
    for inputs, targets in tqdm(val_loader, desc = 'Validation:', leave = False):
        inputs = inputs.to(device)
        targets = targets.to(device)

        # Forward
        with torch.no_grad():
            logits = model(inputs)
            logits = logits.view(-1, vocab_size)
            targets = targets.view(-1, )
            loss = F.cross_entropy(
                input = logits,
                target = targets
            )
        
        eval_batches += 1
        eval_loss += loss.item()
    return eval_loss / eval_batches

        
