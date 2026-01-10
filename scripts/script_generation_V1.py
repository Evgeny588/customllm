import torch
import sys
import logging

from pathlib import Path
from warnings import filterwarnings
filterwarnings('ignore')
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

from tiktoken import get_encoding
from torch import nn
from src.modules import GPTModel, small_config

device = 'cuda' if torch.cuda.is_available() else 'cpu'

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
    model = GPTModel(small_config)
    weights = torch.load(project_root / 'checkpoints/best_model.pth', map_location = device)['states_dicts']['model']

    model.load_state_dict(weights)
    tokenizer = get_encoding('gpt2')
    prompt = input('Your prompt: ')
    generation(prompt, model, 15, tokenizer)

def main_2():
    tokenizer = get_encoding('gpt2')     
    print(tokenizer.decode([0]))


if __name__ == '__main__':
    main()