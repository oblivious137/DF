import torch
import torch.nn as nn
import torch.nn.functional as F
from network.builder import build_backbone, build_decoder, build_loss
from utils import calc_DICE

class Detector(nn.Module):
    def __init__(self, cfg) -> None:
        super(Detector, self).__init__()
        self.backbone = build_backbone(cfg.backbone)
        self.decoder = build_decoder(cfg.decoder)
        self.loss = build_loss(cfg.loss)
        self.cvt_back = cfg.cvt_back
    
    def forward(self, image, label, **karg):
        outputs = self.backbone(image)
        pred = self.decoder(outputs)
        if self.training:
            loss = self.loss(pred, label)
            DICE = calc_DICE(torch.argmax(pred.detach(), dim=1), label)
            return loss, DICE
        else:
            return self.get_pred(pred, karg['ori_shape'])

    def get_pred(self, pred, ori_shape):
        bs = pred.shape[0]
        assert bs == 1
        pred = F.upsample_bilinear(pred.detach().cpu(), ori_shape[0])
        pred = torch.argmax(pred, dim=1)
        return pred

