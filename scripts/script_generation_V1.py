import torch
import sys
import logging

from pathlib import Path
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tiktoken import get_encoding
from torch import nn
from src.modules import GPTModel


small_config = {
        'vocab_size': 50257,
        'context_length': 1024,
        'emb_dim': 768,
        'n_heads': 12,
        'n_layers': 12,
        'drop_rate': 0.1,
        'qkv_bias': False
        }

def generation(prompt, configured_model, max_length, tokenizer):
        configured_model.eval()
        tokenized_prompt = tokenizer.encode(prompt)
        tokenized_prompt = torch.tensor(tokenized_prompt,
                                         dtype = torch.long).unsqueeze(0)
        
        for gen_step in range(max_length): 
            logits = configured_model(tokenized_prompt) 
            word_idx = torch.argmax(logits, dim = -1) 
            new_token = word_idx[0][-1]
            new_token = new_token.unsqueeze(0).unsqueeze(0)
            tokenized_prompt = torch.cat([tokenized_prompt, new_token], dim = 1)

        print(tokenizer.decode(tokenized_prompt.squeeze().tolist()))

def main():
    model = GPTModel(config)
    tokenizer = get_encoding('gpt2')
    prompt = input('Your prompt: ')
    generation(prompt, model, 15, tokenizer)

def main2():
    tokenizer = get_encoding('gpt2')     
    print(tokenizer.decode([0]))


if __name__ == '__main__':
    main2()