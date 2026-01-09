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
from src.modules import GPTModel


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-3
tokenizer = get_encoding('gpt2')


# class TextDataset(Dataset):
#     def __init__(self, data, window, tokenizer, step = 1):
#         self.data = data
#         self.window = window
#         self.step = step
#         self.start_index = 0
#         self.tokenizer = tokenizer
#         self.list_of_data = data.split(' ')
#         logging.info('Dataset initialized!')

#     def __len__(self):
#         return len(self.list_of_data)
    

#     def __getitem__(self, index): 
#         text = self.list_of_data[index: self.window]
#         target = self.list_of_data[index + self.window + 1]
#         self.start_index += self.step

#         return torch.tensor(self.tokenizer.encode(str(text)), dtype = torch.long), torch.tensor(self.tokenizer.encode(str(target)), dtype = torch.long)
    

#     def reset_index(self):
#         self.start_index = 0


# def main():
#     with open(Path(root_path) / 'data/text.txt', mode = 'r') as file:
#         text = file.read()

#     data = TextDataset(text, 5, tokenizer)
#     loader = DataLoader(data, 1, shuffle = False)

#     for batch, label in loader:
#         logging.info((batch, label))
#         logging.info((tokenizer.decode(batch[0].tolist()), tokenizer.decode(label[0].tolist())))
#         break



# if __name__ == '__main__':
#     main()



class TextDataset(Dataset):
    def __init__(self, text, window_size, tokenizer):
        self.text = text.lower()
        self.window = window_size
        self.tokenizer = tokenizer
        self.splitted_text = self.text.split(' ')
        logging.info('Dataset initialized.') 
        
    def __len__(self):
        return len(self.splitted_text)


    def __getitem__(self, idx):
       logging.info(f'Index: {idx}')
       non_sliced = self.splitted_text[idx: idx + self.window]
       sliced = self.splitted_text[idx + 1: idx + 1 + self.window]
       logging.debug('Slicing sucsessful')

       tokenized_non_sliced = tokenizer.encode(' '.join(non_sliced))
       tokenized_sliced = tokenizer.encode(' '.join(sliced)) 
       logging.debug('Tokenizing sucsessful')

       return torch.tensor(tokenized_non_sliced, dtype = torch.long), torch.tensor(tokenized_sliced, dtype = torch.long)


def main():
    with open(root_path / 'data/text.txt', mode = 'r') as file:
        text = file.read()   

    data = TextDataset(text, 5, tokenizer)

    for i, (b, t) in enumerate(data):
        logging.info(f'Batch: {b}, Target: {t}')
        decoded_b = tokenizer.decode(b.tolist())
        decoded_t = tokenizer.decode(t.tolist())
        logging.info(f'Decoded batch: {decoded_b}')
        logging.info(f'Decoded target: {decoded_t}')
        logging.info('--' * 30)
        
        if i >= 5:
            break


if __name__ == '__main__':
    print("Token 9542:", tokenizer.decode([9542]))
    print("Token 32683:", tokenizer.decode([32683])) 