from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from network.unet import UNet
import torch.nn as nn

def build_backbone(cfg):
    name = cfg.pop('name')
    if name == 'Resnet':
        depth = cfg.pop('depth')
        model = eval(f"resnet{depth}")
    else:
        assert False
    return model(**cfg)

def build_decoder(cfg):
    name = cfg.pop('name')
    if name == 'UNet':
        model = UNet
    else:
        assert False
    return model(cfg)

def build_loss(cfg):
    name = cfg.pop('name')
    if name == 'CrossEntropyLoss':
        model = nn.CrossEntropyLoss
    else:
        assert False
    return model(**cfg)