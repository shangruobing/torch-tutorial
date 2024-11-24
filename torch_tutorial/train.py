import torch
import torch.nn as nn

__all__ = ["init_optimizer", "train"]


def init_optimizer(model, lr=0.01, weight_decay=0.001):
    """
    To train a model, we need a loss function, an optimizer and a scheduler.
    """
    criterion = nn.MSELoss(reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer,
                                                           T_max=80,
                                                           eta_min=0.000001)
    return criterion, optimizer, scheduler


def train(dataloader, model, criterion, optimizer, scheduler, device):
    """
    In a single training loop, the model makes predictions on the training dataset (fed to it in batches),
    and back propagates the prediction error to adjust the model’s parameters.
    """
    model.to(device)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        optimizer.zero_grad()
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        predict = model(X)
        loss = criterion(predict, y)
        # Backpropagation
        loss.backward()
        optimizer.step()
        scheduler.step()
        return round(loss.item(), 4)


if __name__ == '__main__':
    from model import LinearRegression

    model = LinearRegression()
    criterion, optimizer, scheduler = init_optimizer(model)
