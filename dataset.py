import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import v2


def load_dataset(dataset, batch_size=4, augmentation=False):
    if augmentation:
        transform_train = v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Pad(4, 0, "constant"),
                v2.RandomHorizontalFlip(p=0.5),
                v2.RandomResizedCrop(size=(32, 32)),
                v2.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ]
        )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ]
    )

    if dataset == "FashionMNIST":  # 1x28x28
        training_set = torchvision.datasets.FashionMNIST(
            "./data", train=True, transform=transform_train, download=True
        )
        validation_set = torchvision.datasets.FashionMNIST(
            "./data", train=False, transform=transform_test, download=True
        )
        img_dim = list([batch_size, 1, 28, 28])
    elif dataset == "SVHN":  # 3x32x32
        training_set = torchvision.datasets.SVHN(
            "./data", split="train", transform=transform_train, download=True
        )
        validation_set = torchvision.datasets.SVHN(
            "./data", split="train", transform=transform_test, download=True
        )
        img_dim = list([batch_size, 3, 32, 32])
    elif dataset == "CIFAR10":  # 3x32x32
        training_set = torchvision.datasets.CIFAR10(
            "./data", train=True, transform=transform_train, download=True
        )
        validation_set = torchvision.datasets.CIFAR10(
            "./data", train=False, transform=transform_test, download=True
        )
        img_dim = list([batch_size, 3, 32, 32])
    else:
        raise Exception("Dataset not available")

    training_loader = torch.utils.data.DataLoader(
        training_set, batch_size=batch_size, shuffle=True
    )
    validation_loader = torch.utils.data.DataLoader(
        validation_set, batch_size=batch_size, shuffle=False
    )

    return training_loader, validation_loader, img_dim
