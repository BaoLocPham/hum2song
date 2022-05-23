from models.resnet import *

class WrapModule(nn.Module):
    def __init__(self, block, layers, use_se=True, **kwargs):
        super(WrapModule, self).__init__()
        self.module = ResNetFace(block, layers, use_se, **kwargs)

    def _make_layer(self, block, planes, blocks, stride=1):
        return self.module._make_layer(block, planes, blocks, stride=1)

    def forward(self, x):
        return self.module(x)

def get_wrap_resnet(config):
    """
    Get wrap resnet based on config.py
    """
    model = None
    if config.backbone == 'resnet18':
        model = wrap_resnet_face18(use_se=config.use_se)
    elif config.backbone == 'resnet34':
        model = wrap_resnet_face34(use_se=config.use_se)
    elif config.backbone == 'resnet50':
        model = wrap_resnet_face50(use_se=config.use_se)
    # elif config.backbone == 'resnet101':
    #     model =  wrap_resnet_face101(use_se=config.use_se)
    return model

def get_resnet(config):
    """
    Get resnet based on config.py
    """
    model = None
    if config.backbone == 'resnet18':
        model = resnet_face18(use_se=config.use_se)
    elif config.backbone == 'resnet34':
        model = resnet_face34(use_se=config.use_se)
    elif config.backbone == 'resnet50':
        model = resnet_face50(use_se=config.use_se)
    elif config.backbone == 'resnet101':
        model = resnet_face101(use_se=config.use_se)
    elif config.backbone == 'resnet152':
        model = resnet_face152(use_se=config.use_se)
    return model

# def resnet_face18(use_se=True, **kwargs):
#     model = ResNetFace(IRBlock, [2, 3, 4, 3], use_se=use_se, **kwargs)
#     return model


# def wrap_resnet_face18(use_se=True, **kwargs):
#     model = WrapModule(IRBlock, [2, 3, 4, 3], use_se=use_se, **kwargs)
#     return model

# def resnet_face34(use_se=True, **kwargs):
#     model = ResNetFace(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
#     return model

# def wrap_resnet_face34(use_se=True, **kwargs):
#     model = WrapModule(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
#     return model

# def resnet_face34(use_se=True, **kwargs):
#     model = ResNetFace(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
#     return model

# def wrap_resnet_face34(use_se=True, **kwargs):
#     model = WrapModule(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
#     return model

# def resnet_face50(use_se=True, **kwargs):
#     model = ResNetFace(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
#     return model

# def wrap_resnet_face50(use_se=True, **kwargs):
#     model = WrapModule(IRBlock, [3, 4, 6, 3], use_se=use_se, **kwargs)
#     return model