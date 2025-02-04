from models.LeNet import LeNet
from models.ResNet import ResNet


def load_model(params):
    if params.model == "LeNet":
        model = LeNet(params.img_dim)
    if params.model == "ResNet":
        model = ResNet(params.img_dim)
    else:
        raise Exception("Model not available")
    
    return model
    