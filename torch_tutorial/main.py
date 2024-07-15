import sys
from os.path import dirname, abspath
from pprint import pprint

import torch

BASE_DIR = dirname(dirname(abspath(__file__)))
sys.path.append(BASE_DIR)

from torch_tutorial.dataloader import init_dataloader
from torch_tutorial.parse import init_parser_args
from torch_tutorial.train import train, init_optimizer
from torch_tutorial.test import test
from torch_tutorial.utils import get_device, fix_seed
from torch_tutorial.model import LinearRegression
from torch_tutorial.config import LOG_PATH, print_path
from torch_tutorial.logger import Logger

if __name__ == '__main__':
    args = init_parser_args()
    pprint(args)
    print_path()
    if not LOG_PATH.exists():
        LOG_PATH.mkdir(parents=True)
    fix_seed(args.seed)
    dataloader = init_dataloader()
    # To accelerate operations in the neural network, we move it to the GPU if available.
    device = get_device(device=args.device, cpu=args.cpu)
    model = LinearRegression()
    criterion, optimizer, scheduler = init_optimizer(model=model, lr=args.lr, weight_decay=args.weight_decay)

    # The training process is conducted over several iterations (epochs).
    # During each epoch, the model learns parameters to make better predictions.
    # We print the model’s accuracy and loss at each epoch.
    # We’d like to see the accuracy increase and the loss decrease with every epoch.
    for epoch in range(args.epochs):
        loss = train(dataloader, model, criterion, optimizer, scheduler, device)
        test_loss, accuracy = test(dataloader, model, criterion, device)
        if epoch and (epoch + 1) % (args.epochs / 10) == 0:
            print(f"epoch: {epoch + 1:4}  loss: {loss} test_loss: {test_loss} accuracy: {accuracy}")
            Logger.insert_row(epoch=epoch + 1, train_loss=loss, test_loss=test_loss, accuracy=accuracy, args=args, model=model)

    # A common way to save a model is to serialize the internal state dictionary (containing the model parameters).
    MODEL_PATH = LOG_PATH / "model.pth"
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Save PyTorch Model to {MODEL_PATH}")
