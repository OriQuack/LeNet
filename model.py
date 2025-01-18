import torch
import torchvision
import torchvision.transforms as transforms

from torch.utils.tensorboard import SummaryWriter
from datetime import datetime


# DATASETS
DATASET = "SVHN"

transform = transforms.Compose(
    [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
)

if DATASET == "FashionMNIST":
    training_set = torchvision.datasets.FashionMNIST(
        "./data", train=True, transform=transform, download=True
    )
    validation_set = torchvision.datasets.FashionMNIST(
        "./data", train=False, transform=transform, download=True
    )
elif DATASET == "SVHN":
    training_set = torchvision.datasets.SVHN(
        "./data", split="train", transform=transform, download=True
    )
    validation_set = torchvision.datasets.SVHN(
        "./data", split="train", transform=transform, download=True
    )
elif DATASET == "CIFAR10":
    training_set = torchvision.datasets.CIFAR10(
        "./data", train=True, transform=transform, download=True
    )
    validation_set = torchvision.datasets.CIFAR10(
        "./data", train=False, transform=transform, download=True
    )

training_loader = torch.utils.data.DataLoader(training_set, batch_size=4, shuffle=True)
validation_set = torch.utils.data.DataLoader(
    validation_set, batch_size=4, shuffle=False
)

dataiter = iter(training_loader)
images, labels = next(dataiter)

print(images.size())

img_dim = images.size()
img_grid = torchvision.utils.make_grid(images)


# MODEL
import torch.functional as F


class LeNet(torch.nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()

        self.conv1 = torch.nn.Conv2d(img_dim[1], 6, 5)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.pool = torch.nn.MaxPool2d(2)

        self.relu = torch.nn.ReLU()

        self.fc1 = torch.nn.Linear(16 * 5 * 5, 120)  # 왜 28일땐 4, 32일땐 5?
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


model = LeNet()


# LOSS
loss_fn = torch.nn.CrossEntropyLoss()

optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)


# TRAINING
def train_one_epoch(epoch_index, tb_writer):
    running_loss = 0.0
    last_loss = 0.0

    for i, data in enumerate(training_loader):
        inputs, labels = data

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()
        if i % 1000 == 999:
            last_loss = running_loss / 1000
            tb_x = epoch_index * len(training_loader) + i + 1
            tb_writer.add_scalar("Loss/train", last_loss, tb_x)
            running_loss = 0.0

    return last_loss


# MAIN
# timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
# writer = SummaryWriter("runs/fashion_trainer_{}".format(timestamp))
writer = SummaryWriter("runs/{}_1".format(DATASET))
writer.add_image("Samples {}".format(DATASET), img_grid)

epoch_number = 0

EPOCHS = 10

best_vloss = 1000000.0

for epoch in range(EPOCHS):
    model.train(True)
    avg_loss = train_one_epoch(epoch_number, writer)

    running_vloss = 0.0

    model.eval()

    with torch.no_grad():
        for i, vdata in enumerate(validation_set):
            vinputs, vlabels = vdata
            voutputs = model(vinputs)
            vloss = loss_fn(voutputs, vlabels)
            running_vloss += vloss

    avg_vloss = running_vloss / (i + 1)
    print("LOSS train {} valid {}".format(avg_loss, avg_vloss))

    writer.add_scalars(
        "Training vs. Validation Loss",
        {"Training": avg_loss, "Validation": avg_vloss},
        epoch_number + 1,
    )

    writer.flush()

    epoch_number += 1
