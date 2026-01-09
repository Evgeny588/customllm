import torch
import sys
import logging
 
from pathlib import Path
from torch import nn
from tiktoken import get_encoding
from torch.utils.data import Dataset, DataLoader

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
logging.basicConfig(level = logging.INFO)
from src.modules import GPTModel, small_config


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3
tokenizer = get_encoding('gpt2')


class TextDataset(Dataset):
    def __init__(self, text_data, window, tokenizer, slice = 1):
        self.tokens = tokenizer.encode(text_data) 
        self.window = window
        self.slice = slice

    def __len__(self):
        return len(self.tokens) - self.window
    

    def __getitem__(self, idx):
        start_pos = idx
        end_pos = idx + self.window

        non_sliced = self.tokens[start_pos: end_pos]
        sliced = self.tokens[start_pos + self.slice: end_pos + self.slice]

        return torch.tensor(non_sliced, dtype = torch.long), torch.tensor(sliced, dtype = torch.long)
    


def main():

    with open(root_path / 'data/text2.txt', mode = 'r') as file:
        text = file.read()


    dataset = TextDataset(text, 5, tokenizer)
    loader = DataLoader(
        dataset = dataset,
        batch_size = 5,
        shuffle = False
    )
    
    model = GPTModel(small_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    criterion = # Надо изучить по книге
    
    
    

if __name__ == '__main__':
    main()
