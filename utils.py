import os
import torch
from models.LeNet import LeNet
from models.ResNet import ResNet
from models.PreActResNet import PreActResNet


def load_model(params):
    device = torch.device(os.environ["TORCH_DEVICE"])

    if params.model == "LeNet":
        model = LeNet(params.img_dim).to(device)
    elif params.model == "ResNet":
        model = ResNet(params.img_dim).to(device)
    elif params.model == "PreActResNet":
        model = PreActResNet(params.img_dim).to(device)
    else:
        raise Exception("Model not available")

    return model
