from wrap_mobilenet import *
from wrap_resnet import *

def get_model(config="resnet"):
    if "resnet" in config.backbone:
        model = get_resnet(config=config)
    elif "mobilenet" in config.backbone:
        model = get_mobilenet()
    return model