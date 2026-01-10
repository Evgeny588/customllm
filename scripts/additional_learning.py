import torch
import logging
import sys

from torch import nn
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from tiktoken import get_encoding
from tqdm import tqdm
from src.modules import GPTModel, small_config, TextDataset, EarlyStopping
from src.modules import train_cycle, validation_cycle, Saver
from torch.utils.data import DataLoader


data_path = Path(root_path / 'data/')
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 20
BATCH_SIZE = 16
LR = 1e-2
pin_memory = torch.cuda.is_available()
num_workers = 2 if pin_memory else 0


def main():
    config = torch.load(root_path / 'checkpoints/best_model.pth')   
    tokenizer = get_encoding('gpt2')
    early_stop = EarlyStopping(1e-1, 4)
     

    model = GPTModel(small_config).to(device)
    model.load_state_dict(config['states_dicts']['model'])

    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    optimizer.load_state_dict(config['states_dicts']['optimizer'])
    saver = Saver(
        model,
        optimizer,
        root_path / 'checkpoints/additional_model.pth',
    )
    with open(data_path / 'train.txt') as file:
        train_text = file.read()
    with open(data_path / 'eval.txt') as file:
        eval_text = file.read()
    train_dataset = TextDataset(train_text, 10, tokenizer)
    eval_dataset = TextDataset(eval_text, 10, tokenizer)

    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        pin_memory = pin_memory,
        num_workers = num_workers
    )
    eval_loader = DataLoader(
        dataset = eval_dataset,
        batch_size = BATCH_SIZE,
        pin_memory = pin_memory,
        num_workers = num_workers
    )


    # Cycle
    for epoch in tqdm(range(EPOCHS), desc = f'Epoch:'):
        train_loss = train_cycle(
            model = model,
            optimizer = optimizer,
            train_loader = train_loader,
            device = device,
            vocab_size = small_config['vocab_size']
        )
        logging.info(f'Train loss = {train_loss: .4f}')
        validation_loss = validation_cycle(
            model = model,
            val_loader = eval_loader,
            device = device,
            vocab_size = small_config['vocab_size']
        )
        logging.info(f'Validation loss = {validation_loss: .4f}')

        saver(validation_loss, train_loss)
        if early_stop(validation_loss):
            break


if __name__ == '__main__':
    main() 