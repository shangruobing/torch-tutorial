import torch


def test(dataloader, model, criterion, device):
    """
    We also check the model’s performance against the test dataset to ensure it is learning.
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            predict = model(X)
            test_loss += criterion(predict, y).item()
            correct += (torch.floor(predict) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    accuracy = correct / size
    return round(test_loss, 4), accuracy
