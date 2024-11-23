import os
from pathlib import Path

import torch

from model import LinearRegression


def inference(model_folder: Path):
    """
    The process for loading a model includes re-creating the model structure and loading the state dictionary into it.
    """
    model = LinearRegression()
    model.load_state_dict(torch.load(Path(model_folder) / "model.pth", weights_only=True))

    points = [torch.tensor([i], dtype=torch.float) for i in range(1, 6)]

    model.eval()
    with torch.no_grad():
        for x in points:
            predict = model(x)
            print(f"X: {x.item()} Y: {x.item() * 2 + 3} Predict: {round(predict.item(), 4)}")


if __name__ == '__main__':
    from config import ROOT_PATH
    from os import listdir

    checkpoint_folder = ROOT_PATH / "log"
    folders = listdir(checkpoint_folder)
    latest_folder = max(folders, key=lambda x: os.path.getmtime(os.path.join(checkpoint_folder, x)))
    model_folder = checkpoint_folder / latest_folder
    print(f"Loading Checkpoint from: {model_folder}")
    inference(model_folder=model_folder)
