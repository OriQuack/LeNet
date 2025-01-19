import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from dataset import load_dataset
from model import LeNet

# SVHN, FashionMNIST, CIFAR10
DATASET = "CIFAR10"

# Get dataset loaders
training_loader, validation_loader = load_dataset(DATASET, batch_size=1)

# Get image dimensions
dataiter = iter(training_loader)
images, labels = next(dataiter)
img_dim = list(images.size())

# Init tensorboard
writer = SummaryWriter("runs/{}_7".format(DATASET))

# Add images
img_grid = torchvision.utils.make_grid(images)
writer.add_image("Samples {}".format(DATASET), img_grid)

# Run test
model = LeNet(img_dim)
model.train_model(20, training_loader, validation_loader, lr=0.00001, momentum=0.9, writer=writer)
