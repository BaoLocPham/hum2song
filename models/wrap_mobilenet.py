from models.mobilenet import *

def get_mobilenet(config):
    """
    Get mobile based on config.py
    """
    model = None
    if config.backbone == 'mobilenetv2':
        model = mobilenet_v2()
    return model