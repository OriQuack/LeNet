from model import LeNet

def load_model(params):
    if params.model == "LeNet":
        model = LeNet(params.img_dim)
    else:
        raise Exception("Model not available")
    
    return model
    