from torch.utils.data import DataLoader

from dataset import LinearDataset


def init_dataloader():
    """
    PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset.
    Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.
    """
    dataset = LinearDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True, num_workers=0)
    return dataloader


if __name__ == '__main__':
    dataloader = init_dataloader()
    for X, y in dataloader:
        print(f"X : {X}")
        print(f"Y : {y}")
        print(f"Shape of X : {X.size()}")
        print(f"Shape of Y: {y.shape} {y.dtype}")
        break
