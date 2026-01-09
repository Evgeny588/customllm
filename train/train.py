import torch
import sys
import logging
import torch.nn.functional as F
from pathlib import Path
from torch import nn
from tiktoken import get_encoding
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
logging.basicConfig(level = logging.INFO)
from src.modules import GPTModel, small_config

EPOCHS = 5
DEVICE = 'cpu'                # 'cuda' if torch.cuda.is_available() else 'cpu'
LR = 1e-4
tokenizer = get_encoding('gpt2')


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
    


def main():

    with open(root_path / 'data/text2.txt', mode = 'r') as file:
        text = file.read()


    dataset = TextDataset(text, 6, tokenizer)
    loader = DataLoader(
        dataset = dataset,
        batch_size = 10,
        shuffle = False
    )
    
    model = GPTModel(small_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)


    epoch_loss, epoch_acc = [], []
    for epoch in tqdm(range(EPOCHS)):
        
        current_loss = 0.0
        counter_batch = 0
        model.train()

        for inputs, targets in tqdm(loader):
           
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(inputs)
            logits = logits.view(-1, small_config['vocab_size'])
            targets = targets.view(-1, )

            loss = F.cross_entropy(input= logits, target= targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logging.debug(f'Batch_loss = {loss.item(): .4f}')

            counter_batch += 1 
            current_loss += loss.item()
            
        loss_on_epoch = current_loss / counter_batch
        epoch_loss.append(loss_on_epoch)

        logging.info(f'Loss on {epoch} epoch = {loss_on_epoch: .4f}')



if __name__ == '__main__':
    main()
