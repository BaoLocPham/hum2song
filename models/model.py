from models.wrap_mobilenet import *
from models.wrap_resnet import *
from models.wrap_vgg import *
from models.wrap_alexnet import *

def get_model(config="resnet"):
    if "resnet" in config.backbone:
        model = get_resnet(config=config)
    elif "mobilenet" in config.backbone:
        model = get_mobilenet(config)
    elif "vgg" in config.backbone:
        model = get_vgg(config)
    elif "alexnet" in config.backbone:
        model = get_alexnet(config)
    return model