import torch
import logging
import sys
import argparse
import warnings

from torch import nn
from pathlib import Path
root_path = Path(__file__).resolve().parent.parent
sys.path.append(str(root_path))
from tiktoken import get_encoding
from tqdm import tqdm
from src.modules import GPTModel, small_config, TextDataset, EarlyStopping
from src.modules import train_cycle, validation_cycle, Saver
from torch.utils.data import DataLoader

logging.basicConfig(
    level = logging.DEBUG
)

def parse_args():
    parser = argparse.ArgumentParser(
        description = 'Дообучение модели'
    )

    parser.add_argument(
        '--epochs',
        type = int,
        default = 3,
        help = 'Количество эпох обучения'
    )

    parser.add_argument(
        '--lr',
        type = float,
        default = 0.01,
        help = 'Скорость обучения'
    )

    parser.add_argument(
        '--batch_size',
        type = int,
        default = 8,
        help = 'Batch size'
    )

    parser.add_argument(
        '--device',
        type = str,
        default = 'cpu',
        help = 'Устройство на котором обучается модель (Процессор или видеокарта)'
    )
    parser.add_argument(
        '--warnings',
        type = str,
        default = 'ignore',
        help = 'Игнорировать предупреждения или нет (default or ignore)'
    )
    return parser.parse_args()


args = parse_args()
warnings.filterwarnings(args.warnings)

data_path = Path(root_path / 'data/')
device = args.device
EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
LR = args.lr
pin_memory = torch.cuda.is_available()
num_workers = 2 if pin_memory else 0

logging.info(f'Устройство: {device}')
logging.debug(f'Epochs: {EPOCHS}')
logging.debug(f'Batch_size: {BATCH_SIZE}')
logging.debug(f'Learning rate: {LR}')

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
    for epoch in tqdm(range(EPOCHS), desc = 'Epoch:'):
        logging.info(f'===============Epoch {epoch}===============')
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