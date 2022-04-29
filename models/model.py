from models.wrap_mobilenet import *
from models.wrap_resnet import *

def get_model(config="resnet"):
    if "resnet" in config.backbone:
        model = get_resnet(config=config)
    elif "mobilenet" in config.backbone:
        model = get_mobilenet(config)
    return model