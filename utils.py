import os
import torch
from models.LeNet import LeNet
from models.ResNet import ResNet
from models.PreActResNet import PreActResNet


def try_makedir(path):
    if not os.path.isdir(path):
        try:
            os.makedirs(path)
        except FileExistsError:
            pass


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

    if params.load:
        checkpoint = torch.load(f"results/{params.file_dir}/params{params.load_epoch}")
        model.load_state_dict(checkpoint["model_state_dict"])

    return model
