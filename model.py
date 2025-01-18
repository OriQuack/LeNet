import torch
from torch.utils.tensorboard import SummaryWriter


# MODEL
class LeNet(torch.nn.Module):
    def __init__(self, input_dim=[4, 1, 32, 32]):
        super(LeNet, self).__init__()

        # LOSS
        self.loss_fn = torch.nn.CrossEntropyLoss()

        # NET
        self.conv1 = torch.nn.Conv2d(input_dim[1], 6, 5, stride=1)
        self.conv2 = torch.nn.Conv2d(6, 16, 5, stride=1)

        self.pool = torch.nn.MaxPool2d(2)

        self.relu = torch.nn.ReLU()

        dim = ((input_dim[2] - 5 + 1) // 2 - 5 + 1) // 2

        self.fc1 = torch.nn.Linear(16 * dim * dim, 120)
        self.fc2 = torch.nn.Linear(120, 84)
        self.fc3 = torch.nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, self.num_flat_features(x))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

    def train_one_epoch(self, training_loader, optimizer, epoch_index, tb_writer):
        running_loss = 0.0
        last_loss = 0.0

        for i, data in enumerate(training_loader):
            inputs, labels = data

            optimizer.zero_grad()

            outputs = self(inputs)

            loss = self.loss_fn(outputs, labels)
            loss.backward()

            optimizer.step()

            running_loss += loss.item()
            if i % 1000 == 999:
                last_loss = running_loss / 1000
                tb_x = epoch_index * len(training_loader) + i + 1
                tb_writer.add_scalar("Loss/train", last_loss, tb_x)
                running_loss = 0.0

        return last_loss

    def train_model(
        self,
        epochs,
        training_loader,
        validation_loader,
        lr=0.001,
        momentum=0.9,
        writer: SummaryWriter = None,
    ):
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, momentum=momentum)

        for epoch in range(epochs):
            self.train(True)
            avg_loss = self.train_one_epoch(training_loader, optimizer, epoch, writer)

            running_vloss = 0.0

            self.eval()

            with torch.no_grad():
                for i, vdata in enumerate(validation_loader):
                    vinputs, vlabels = vdata
                    voutputs = self(vinputs)
                    vloss = self.loss_fn(voutputs, vlabels)
                    running_vloss += vloss

            avg_vloss = running_vloss / (i + 1)
            print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

            writer.add_scalars(
                "Training vs. Validation Loss",
                {"Training": avg_loss, "Validation": avg_vloss},
                epoch + 1,
            )

            writer.flush()

            epoch += 1
