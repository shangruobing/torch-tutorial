import torch.nn as nn

__all__ = ["LinearRegression"]


class LinearRegression(nn.Module):
    """
    To define a neural network in PyTorch, we create a class that inherits from nn.Module.
    We define the layers of the network in the __init__ function and specify how data will pass through the network in the forward function.
    """

    def __init__(self):
        super(LinearRegression, self).__init__()
        self.linear = nn.Sequential(
            nn.Linear(in_features=1, out_features=16),
            nn.Linear(in_features=16, out_features=8),
            nn.Linear(in_features=8, out_features=1),
        )

    def forward(self, x):
        out = self.linear(x)
        return out


if __name__ == '__main__':
    model = LinearRegression()
    print(model)
