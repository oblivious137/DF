from os import removedirs
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.builder import build_backbone, build_decoder, build_loss
from utils import calc_DICE

class Detector(nn.Module):
    def __init__(self, cfg) -> None:
        super(Detector, self).__init__()
        self.backbone = build_backbone(cfg.backbone)
        loss = build_loss(cfg.loss)
        self.decoder = build_decoder(cfg.decoder, loss=loss)
    
    def forward(self, image, **karg):
        features = self.backbone(image)
        ret = self.decoder(features=features, image=image, **karg)
        if self.training:
            ret['pred'] = self.decoder.get_orishape(ret, (image.shape[3:1:-1],))
            DICE = calc_DICE(ret['pred'], karg['label'])
            ret['DICE'] = DICE
        else:
            ret['pred'] = self.decoder.get_orishape(ret, karg['ori_shape'])
        return ret

def build_discriminator():
    layers = list()
    inc = 64
    layers.append(nn.Conv2d(4, inc, 7, padding=3, stride=2))
    layers.append(nn.ReLU(True))
    for i in range(4):
        layers.append(nn.Conv2d(inc, inc*2, 3, padding=1, bias=False, stride=2))
        layers.append(nn.ReLU(True))
        layers.append(nn.Conv2d(inc*2, inc*2, 3, padding=1, bias=False))
        layers.append(nn.ReLU(True))
        inc = inc*2
    layers.append(nn.AdaptiveAvgPool2d(1))
    layers.append(nn.Flatten(1))
    layers.append(nn.Linear(inc, 1))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)
    

class GANDetector(nn.Module):
    def __init__(self, cfg) -> None:
        super(GANDetector, self).__init__()
        self.backbone = build_backbone(cfg.backbone)
        loss = build_loss(cfg.loss)
        self.decoder = build_decoder(cfg.decoder, loss=loss)
        self.discriminator = build_discriminator()
    
    def forward(self, image, **karg):
        features = self.backbone(image)
        ret = self.decoder(features=features, image=image, **karg)
        if self.training:
            ret['pred'] = self.decoder.get_orishape(ret, (image.shape[3:1:-1],))
            assert ret['prob'].requires_grad
            DICE = calc_DICE(ret['pred'], karg['label'])
            ret['DICE'] = DICE
            real = torch.cat((image, karg['label'].unsqueeze(1)), dim=1)
            fake = torch.cat((image, ret['prob'][:,1].unsqueeze(1)), dim=1)
            fake_score = self.discriminator(fake)
            dloss = 1-F.mse_loss(fake_score, torch.zeros_like(fake_score))
            ret['losses']['GAN Loss'] = dloss
            ret['loss'] += dloss
            
            real_score = self.discriminator(real)
            fake_score = self.discriminator(fake.detach())
            dloss = 2*F.mse_loss(fake_score, torch.zeros_like(fake_score)) + F.mse_loss(real_score, torch.ones_like(fake_score))
            ret['losses']['Discriminator Loss'] = dloss
            ret['loss'] += dloss
        else:
            ret['pred'] = self.decoder.get_orishape(ret, karg['ori_shape'])
        return ret


def build_detector(cfg):
    typ = cfg.pop('type', 'Detector')
    if typ == 'Detector':
        return Detector(cfg)
    elif typ == 'GANDetector':
        return GANDetector(cfg)