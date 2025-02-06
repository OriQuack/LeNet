from models.LeNet import LeNet
from models.ResNet import ResNet
from models.PreActResNet import PreActResNet


def load_model(params):
    if params.model == "LeNet":
        model = LeNet(params.img_dim)
    elif params.model == "ResNet":
        model = ResNet(params.img_dim)
    elif params.model == "PreActResNet":
        model = PreActResNet(params.img_dim)
    else:
        raise Exception("Model not available")
    
    return model
    