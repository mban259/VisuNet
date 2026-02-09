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

    def forward(self, x):
        x = self.relu1(self.conv1(x))  # 4 * 28 * 28
        x = self.pool(x)               # 4 * 14 * 14
        x = self.relu2(self.conv2(x))  # 8 * 14 * 14
        x = self.pool(x)               # 8 * 7 * 7
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
