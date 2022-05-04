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
    elif config.backbone == 'vgg13':
        model = vgg13()
    elif config.backbone == 'vgg13_bn':
        model = vgg13_bn()
    elif config.backbone == 'vgg16':
        model = vgg16()
    elif config.backbone == 'vgg16_bn':
        model = vgg16_bn()
    elif config.backbone == 'vgg19':
        model = vgg19()
    elif config.backbone == 'vgg19_bn':
        model = vgg19_bn()
    return model