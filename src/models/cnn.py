import torch.nn as nn


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(1, 4, 3, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(4, 8, 3, padding=1)
        self.relu2 = nn.ReLU()

        self.pool = nn.MaxPool2d(2, 2)

        self.fc = nn.Linear(8*7*7, 10)

    def forward(self, x, return_features=False):
        h1 = self.relu1(self.conv1(x))  # 4 * 28 * 28
        h2 = self.pool(h1)               # 4 * 14 * 14
        h3 = self.relu2(self.conv2(h2))  # 8 * 14 * 14
        h4 = self.pool(h3)               # 8 * 7 * 7
        h5 = h4.view(x.size(0), -1)
        out = self.fc(h5)

        if return_features:
            return out, h1, h2, h3, h4, h5
        return out
