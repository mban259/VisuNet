import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from models.cnn import SimpleCNN


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    test_ds = datasets.MNIST(
        root="data",
        train=False,
        download=True,
        transform=transform
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=256,
        shuffle=False
    )

    model = SimpleCNN().to(device)
    model.load_state_dict(
        torch.load("data/mnist_cnn.pth", map_location=device)
    )
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            preds = logits.argmax(dim=1)

            correct += (preds == y).sum().item()
            total += y.size(0)

    accuracy = correct / total * 100.0

    print(f"Total samples : {total}")
    print(f"Correct       : {correct}")
    print(f"Accuracy      : {accuracy:.2f}%")


if __name__ == "__main__":
    main()
