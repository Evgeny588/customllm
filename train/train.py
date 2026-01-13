import torch
import sys
import logging
import time
import torch.nn.functional as F

from pathlib import Path
from torch import nn
from tiktoken import get_encoding
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))

from set_logs import setup_logging
from src.modules import GPTModel, small_config, EarlyStopping, TextDataset

setup_logging()
logging = logging.getLogger(Path(__file__).stem)
logging.debug('---------Train script---------')

EPOCHS = 20
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 16
LR = 1e-4
tokenizer = get_encoding('gpt2')
pin_memory = torch.cuda.is_available()
num_workers = 2 if pin_memory else 0

    


def main():

    with open(root_path / 'data/text2.txt', mode = 'r') as file:
        text = file.read()


    train_dataset = TextDataset(text, 6, tokenizer)
    train_loader = DataLoader(
        dataset = train_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        pin_memory= pin_memory,
        num_workers= num_workers
    )
    logging.debug('Train dataset and loader was initialized')

    eval_dataset = TextDataset(text, 6, tokenizer)
    eval_loader = DataLoader(
        dataset = eval_dataset,
        batch_size = BATCH_SIZE,
        shuffle = False,
        pin_memory= pin_memory,
        num_workers= num_workers
    )
    logging.debug('Validation dataset and loader was initialized')

    model = GPTModel(small_config).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LR)
    stopper = EarlyStopping(0.01, 3)

    best_eval_loss = 0.0
    train_epoch_loss, eval_epoch_loss = [], []
    logging.info(f'Start train on {DEVICE}')
    
    for epoch in tqdm(range(EPOCHS), desc = 'Epoch cycle'):
        
        logging.info(f'-------------Epoch {epoch}-------------')
        train_loss = 0.0
        counter_train_batch = 0
        model.train()
        # Forward and backward
        for inputs, targets in tqdm(train_loader, desc = 'Train batch cycle'):
           
            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            logits = model(inputs)
            logits = logits.view(-1, small_config['vocab_size'])
            targets = targets.view(-1, )

            loss = F.cross_entropy(input= logits, target= targets)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            logging.debug(f'Batch_train_loss = {loss.item(): .4f}')

            counter_train_batch += 1 
            train_loss += loss.item()
            
        train_loss_on_epoch = train_loss / counter_train_batch
        train_epoch_loss.append(train_loss_on_epoch)
        logging.info(f'Train loss on {epoch} epoch = {train_loss_on_epoch: .4f}')

        # Evaluate
        model.eval()
        eval_loss = 0.0
        counter_eval_batch = 0

        for inputs, targets in tqdm(eval_loader, desc = 'Eval batch cycle'):

            inputs = inputs.to(DEVICE)
            targets = targets.to(DEVICE)

            with torch.no_grad():
                logits = model(inputs)
                logits = logits.view(-1, small_config['vocab_size'])
                targets = targets.view(-1, )

                loss = F.cross_entropy(input= logits, target= targets)

            logging.debug(f'Batch validation loss = {loss.item(): .4f}')
            
            counter_eval_batch += 1
            eval_loss += loss.item()
        
        eval_loss_on_epoch = eval_loss / counter_eval_batch
        eval_epoch_loss.append(eval_loss_on_epoch)
        logging.info(f'Validation loss on {epoch} epoch = {eval_loss_on_epoch: .4f}')

        # Stopper
        if stopper(eval_loss_on_epoch):
            break
        # Saver
        if eval_loss >= best_eval_loss:
            best_eval_loss = eval_loss

            torch.save({
                'metadata': {
                    'time': time.localtime(),
                    'epoch': epoch,
                    'learning_rate': LR,
                    'batch_size': BATCH_SIZE
                },
                'states_dicts': {
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict()
                },
                'losses': {
                    'train': train_loss_on_epoch,
                    'val': eval_loss_on_epoch
                }
            },
            root_path / 'checkpoints/best_model.pth')







if __name__ == '__main__':
    main()
