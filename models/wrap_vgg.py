from models.vgg import *

def get_vgg(config):
    """
    Get mobile based on config.py
    """
    model = None
    if config.backbone == 'vgg11':
        model = vgg11()
    elif config.backbone == 'vgg11_bn':
        model = vgg11_bn()
    return model