import json
from typing import Tuple

import torch
from torch.utils.data import Dataset

from config import DATASET_PATH


class LinearDataset(Dataset):
    """
    PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset.
    Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.
    """

    def __init__(self):
        with open(DATASET_PATH / "data.json", "r") as file:
            data = json.load(file)
        self.x = [torch.tensor(i.get("x"), dtype=torch.float) for i in data]
        self.y = [torch.tensor(i.get("y"), dtype=torch.float) for i in data]
        # self.x = [torch.tensor(i, dtype=torch.float) for i in range(100)]
        # self.y = [i * 2 + 3 for i in self.x]

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.x)

    def __repr__(self) -> str:
        information = "\nDataset " + self.__class__.__name__ + "Information:\n"
        information += f"Number of datapoints: {self.__len__()}\n"
        information += f"X: {self.x[:5]}\n"
        information += f"Y: {self.y[:5]}\n"
        return information


if __name__ == '__main__':
    dataset = LinearDataset()
    print(dataset)
