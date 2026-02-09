import torch
from torchvision import datasets, transforms
from models.cnn import SimpleCNN


def main():
    device = torch.device(
        "cuda") if torch.cuda.is_available() else torch.device("cpu")
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

    img, label = test_ds[0]

    model = SimpleCNN().to(device)
    model.load_state_dict(
        torch.load("data/mnist_cnn.pth", map_location=device)
    )
    model.eval()

    activations = {}

    def hook(name):
        def fn(_, __, output):
            activations[name] = output.detach().cpu()
        return fn
    model.conv1.register_forward_hook(hook("conv1"))
    model.conv2.register_forward_hook(hook("conv2"))

    with torch.no_grad():
        logits = model(img.unsqueeze(0).to(device))
        pred = logits.argmax(dim=1).item()

        print("GT   :", label)
        print("Pred :", pred)

    for k, v in activations.items():
        print(f"{k} shape:", v.shape)

if __name__ == "__main__":
    main()
