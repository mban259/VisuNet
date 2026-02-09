import torch
from torch import nn, optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import SimpleCNN
from pathlib import Path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    train_ds = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True

    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    epochs = 10
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss:.4f}")

    Path("data").mkdir(exist_ok=True)
    torch.save(model.state_dict(), "data/mnist_cnn.pth")
    print("Model saved to data/mnist_cnn.pth")


if __name__ == "__main__":
    main()
