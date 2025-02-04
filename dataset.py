import torch
import torchvision
import torchvision.transforms as transforms


def load_dataset(dataset, batch_size=4, transform=None):
    transform = (
        transform
        if transform is not None
        else transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
        )
    )

    if dataset == "FashionMNIST":  # 1x28x28
        training_set = torchvision.datasets.FashionMNIST(
            "./data", train=True, transform=transform, download=True
        )
        validation_set = torchvision.datasets.FashionMNIST(
            "./data", train=False, transform=transform, download=True
        )
    elif dataset == "SVHN":  # 3x32x32
        training_set = torchvision.datasets.SVHN(
            "./data", split="train", transform=transform, download=True
        )
        validation_set = torchvision.datasets.SVHN(
            "./data", split="train", transform=transform, download=True
        )
    elif dataset == "CIFAR10":  # 3x32x32
        training_set = torchvision.datasets.CIFAR10(
            "./data", train=True, transform=transform, download=True
        )
        validation_set = torchvision.datasets.CIFAR10(
            "./data", train=False, transform=transform, download=True
        )
    else:
        raise Exception("Dataset not available")

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=False
    )

    # Get image dimensions
    dataiter = iter(training_loader)
    images, _ = next(dataiter)
    img_dim = list(images.size())

    return training_loader, validation_loader, img_dim
