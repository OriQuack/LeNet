import argparse
import time
import torch
from torch.utils.tensorboard import SummaryWriter

from dataset import load_dataset
from utils import load_model
from run import train_one_epoch
from run import test_one_epoch


def train_one_dataset(params, training_loader, validation_loader, writer):
    model = load_model(params)

    # Optimizer
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=params.lr,
        momentum=params.momentum,
        weight_decay=params.weight_decay,
    )

    # Learning rate scheduler 없음

    for epoch in range(params.epochs):
        # Train
        last_loss = train_one_epoch(model, training_loader, optimizer, epoch, writer)

        # Validate
        avg_vloss = test_one_epoch(model, validation_loader, optimizer, epoch, writer)

        print("LOSS train {} valid {}".format(last_loss, avg_vloss))

        if writer is not None:
            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": last_loss, "Validation": avg_vloss},
                epoch + 1,
            )

            writer.flush()
        # epoch += 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model description")

    # Model
    parser.add_argument("--model", type=str, default="LeNet")

    # Mode
    # parser.add_argument("--train", type=bool, default=True)
    # parser.add_argument("--test", type=bool, default=False)

    # Hyperparameters
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--weight_decay", type=float, default=0.0)

    # Dataset
    parser.add_argument("--dataset", type=str, default="CIFAR10")

    # Tensorboard
    parser.add_argument("--tensorboard", type=bool, default=True)
    parser.add_argument("--save_name", type=str, default="default")

    # GPU in Apple Silicon
    parser.add_argument("--device", type=str, default="cpu")

    params = parser.parse_args()

    # Use gpu
    if params.device == "gpu" and torch.backends.mps.is_available():
        torch.set_default_device("mps")
        print("Training on MPS...")
    else:
        torch.device("cpu")

    # Load dataset
    training_loader, validation_loader, img_dim = load_dataset(
        params.dataset, batch_size=params.batch_size
    )
    params.img_dim = img_dim

    # Init tensorboard
    if params.tensorboard:
        writer = SummaryWriter(
            "runs/{}_{}_{}".format(
                params.dataset, params.model, params.save_name, time.time()
            )
        )
    else:
        writer = None

    # Train model
    train_one_dataset(params, training_loader, validation_loader, writer)
