from math import log
from numpy.lib.arraysetops import isin
from torch.nn.modules import loss
from network.resnet import resnet18, resnet34, resnet50, resnet101, resnet152
from network.vgg import vgg11, vgg13, vgg16, vgg19
from network.resnetw import ResNetW
from network.unet import UNet, UNetComplex
from network.unet_multilayer import UNetMultilayer, UNetComplexMulti
from network.unet_heatmap import UNetHeatmap
from network.upernet import UPerNet
import torch.nn as nn
import torch.nn.functional as F
import torch


def build_backbone(cfg):
    name = cfg.pop('name')
    if name == 'Resnet':
        depth = cfg.pop('depth')
        model = eval(f"resnet{depth}")
    elif name == 'VGG':
        depth = cfg.pop('depth')
        model = eval(f"vgg{depth}")
    elif name=="Resnet-ws":
        model = ResNetW
    else:
        assert False
    return model(**cfg)


def build_decoder(cfg, **kwarg):
    name = cfg.pop('name')
    if name == 'UNet':
        model = UNet
    elif name == 'UNet-MultiLayer':
        model = UNetMultilayer
    elif name == 'UNet-Complex':
        model = UNetComplex
    elif name == 'UNet-Complex-MultiLayer':
        model = UNetComplexMulti
    elif name == 'UNet-Heatmap':
        model = UNetHeatmap
    else:
        assert False
    return model(cfg, **kwarg)


def build_decoder(cfg, **kwarg):
    name = cfg.pop('name')
    if name == 'UNet':
        model = UNet
    elif name == 'UNet-MultiLayer':
        model = UNetMultilayer
    elif name == 'UNet-Complex':
        model = UNetComplex
    elif name == 'UNet-Complex-MultiLayer':
        model = UNetComplexMulti
    elif name == 'UNet-Heatmap':
        model = UNetHeatmap
    elif name == 'UPerNet':
        model = UPerNet
    else:
        assert False
    return model(cfg, **kwarg)

class DICELoss:
    def __call__(self, logits, label, weights=[1.0]):
        if isinstance(logits, list) or isinstance(logits, tuple):
            loss = 0
            for logit, weight in zip(logits, weights):
                if label.shape[2] == logit.shape[2]:
                    suitlabel = label
                else:
                    suitlabel = F.interpolate(label.unsqueeze(1).float(), size=logit.shape[2:], mode='nearest').squeeze(1).long()
                    # import pdb; pdb.set_trace()
                loss += weight*self(logit, suitlabel)
            return loss
        pred = torch.softmax(logits, dim=1)[:, 1, :, :].flatten(1)
        label = label.flatten(1)
        inter = (pred * label).sum(dim=1)
        union = pred.sum(dim=1)+label.sum(dim=1)
        dices = inter*2/union
        return 1-dices.mean()

class CrossEntropyLoss:
    def __call__(self, logits, label):
        if isinstance(logits, list) or isinstance(logits, tuple):
            loss = 0
            for logit in logits:
                if label.shape[2] == logit.shape[2]:
                    suitlabel = label
                else:
                    suitlabel = F.interpolate(label, size=(logit.shape[2], logit.shape[3]), mode='nearest')
            loss += self(logit, suitlabel)
            return loss
        assert isinstance(logits, torch.Tensor)
        return F.cross_entropy(logits, label)


class LossContainer:
    def __init__(self, losses, keys, names, weights) -> None:
        self.losses = losses
        self.keys = keys
        self.names = names
        self.weights = weights

    def __call__(self, **kwarg):
        loss = 0
        losses = dict()
        for i in range(len(self.losses)):
            ls = self.losses[i](**kwarg)
            loss += self.weights[i] * ls
            losses[self.names[i]] = ls.item()
        return loss, losses

class TVLoss:
    def __call__(self, logits):
        score = torch.softmax(logits, dim=1)[:, 1, :, :]


def build_loss(cfgs):
    losses = list()
    keys = list()
    names = list()
    weights = list()

    for d in cfgs:
        typ = d.pop('type')
        name = d.pop('name', typ)
        weight = d.pop('weight', 1.0)

        if typ == 'CrossEntropyLoss':
            loss = CrossEntropyLoss()
            key = ['logits', 'label']
        elif typ == 'DICELoss':
            loss = DICELoss()
            key = ['logits', 'label']
        elif typ == 'TVLoss':
            loss = TVLoss()
            key = ['logits']
        else:
            assert False
        
        losses.append(loss)
        keys.append(key)
        names.append(name)
        weights.append(weight)

    return LossContainer(losses, keys, names, weights)
