import os
import torch
import torch.optim as optim

from models.LeNet import LeNet
from models.ResNet import ResNet
from models.PreActResNet import PreActResNet
from models.StochasticDepth import StochasticDepth
from models.DenseNet import DenseNet
from models.FractalNet import FractalNet
from models.VisionTransformer import VisionTransformer
from models.SwinTransformer import SwinTransformer


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
    elif params.model == "StochasticDepth":
        model = StochasticDepth(params.img_dim).to(device)
    elif params.model == "DenseNet":
        model = DenseNet(params.img_dim).to(device)
    elif params.model == "FractalNet":
        model = FractalNet(params.img_dim).to(device)
    elif params.model == "VisionTransformer":
        model = VisionTransformer(params.img_dim).to(device)
    elif params.model == "SwinTransformer":
        model = SwinTransformer(params.img_dim).to(device)
    else:
        raise Exception("Model not available")

    if params.load:
        checkpoint = torch.load(f"results/{params.file_dir}/params{params.load_epoch}")
        model.load_state_dict(checkpoint["model_state_dict"])

    return model


def get_optimizer(params, model):
    if params.optimizer == "SGD":
        optimizer = optim.SGD(
            model.parameters(),
            lr=params.lr,
            momentum=params.momentum,
            weight_decay=params.weight_decay,
        )
    elif params.optimizer == "Adam":
        optimizer = optim.Adam(
            model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
        )
    elif params.optimizer == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=params.lr,
            weight_decay=params.weight_decay,
        )
    return optimizer


def get_scheduler(params, optimizer):
    schedulers = []
    # Warmup scheduler
    if params.warmup != 0:
        warmup_sched = optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.0, end_factor=1.0, total_iters=params.warmup
        )
        schedulers.append(warmup_sched)

    # Default scheduler
    if params.scheduler == "None":
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[], gamma=1)
    elif params.scheduler == "MultiStep":
        scheduler = optim.lr_scheduler.MultiStepLR(
            optimizer, milestones=[40, 60], gamma=0.1
        )
    elif params.scheduler == "Cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(params.epochs - params.warmup)
        )
    schedulers.append(scheduler)

    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=schedulers, milestones=[params.warmup]
    )
