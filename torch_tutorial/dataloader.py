from torch.utils.data import DataLoader

from dataset import init_dataset

__all__ = ["init_dataloader"]


def init_dataloader():
    """
    PyTorch has two primitives to work with data: torch.utils.data.DataLoader and torch.utils.data.Dataset.
    Dataset stores the samples and their corresponding labels, and DataLoader wraps an iterable around the Dataset.
    """
    train_dataset, test_dataset = init_dataset()
    train_dataloader = DataLoader(
        dataset=train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    test_dataloader = DataLoader(
        dataset=test_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        drop_last=True
    )
    return train_dataloader, test_dataloader


if __name__ == '__main__':
    train_dataloader, test_dataloader = init_dataloader()
    for X, y in train_dataloader:
        print(f"X : {X}")
        print(f"Y : {y}")
        print(f"Shape of X : {X.size()}")
        print(f"Shape of Y: {y.shape} {y.dtype}")
        break
