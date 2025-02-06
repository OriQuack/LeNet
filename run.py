import torch

def train_one_epoch(net, training_loader, optimizer, epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()

        loss, outputs = net(inputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 10
            tb_x = epoch_index * len(training_loader) + i + 1
            if tb_writer is not None:
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


def test_one_epoch(net, test_loader, optimizer, epoch_index, tb_writer):
    running_loss = 0.0

    net.eval()

    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inputs, labels = data
            loss, voutputs = net(inputs, labels)
            running_loss += loss

    avg_loss = running_loss / (i + 1)

    return avg_loss