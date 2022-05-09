from models.alexnet import *

def get_alexnet(config):
    """
    Get alexnet based on config.py
    """
    model = None
    if config.backbone == 'alexnet':
        model = alexnet()
    return model