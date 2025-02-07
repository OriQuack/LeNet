import os
import torch
from tqdm import tqdm


def train_one_epoch(net, params, training_loader, optimizer, epoch_index, tb_writer):
    device = torch.device(os.environ["TORCH_DEVICE"])

    running_loss = 0.0
    last_loss = 0.0
    running_accuracy = 0.0
    last_accuracy = 0.0

    for i, data in tqdm(
        enumerate(training_loader),
        desc="train",
        unit="batch",
        total=len(training_loader),
    ):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)

        optimizer.zero_grad()

        loss, outputs = net(inputs, labels)

        loss.backward()

        optimizer.step()

        running_accuracy += torch.sum((labels == outputs)) / outputs.shape[0]
        running_loss += loss.item()
        if i % 10 == 9:
            last_accuracy = running_accuracy / 10
            last_loss = running_loss / 10
            tb_x = epoch_index * len(training_loader) + i + 1
            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                tb_writer.add_scalar("Accuracy/train", last_accuracy, tb_x)
            running_accuracy = 0.0
            running_loss = 0.0

    return last_loss, last_accuracy


def test_one_epoch(net, params, test_loader, optimizer, epoch_index, tb_writer):
    device = torch.device(os.environ["TORCH_DEVICE"])

    running_loss = 0.0
    running_accuracy = 0.0

    net.eval()

    with torch.no_grad():
        for i, data in tqdm(
            enumerate(test_loader), desc="test", unit="batch", total=len(test_loader)
        ):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)

            loss, outputs = net(inputs, labels)
            running_loss += loss
            running_accuracy += torch.sum((labels == outputs)) / outputs.shape[0]

    avg_loss = running_loss / (i + 1)
    avg_accuracy = running_accuracy / (i + 1)

    return avg_loss, avg_accuracy
