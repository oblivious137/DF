from math import log
from pickle import dump
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.batchrenorm import BatchRenorm2d
from mmcv.ops import roi_align

class UPerNet(nn.Module):
    def __init__(self, cfg, loss=None) -> None:
        super(UPerNet, self).__init__()
        self.loss = loss
        assert len(cfg.strides) == len(cfg.feature_channel)
        fpn_in = list()
        for i in range(1, len(cfg.strides)):
            in_channel = cfg.feature_channel[i]
            fpn_in.append(nn.Sequential(nn.Conv2d(in_channel, cfg.fpn_channel, 1, bias=False),
                                        BatchRenorm2d(cfg.fpn_channel),
                                        nn.ReLU()))
        self.fpn_in = nn.ModuleList(fpn_in)
        
        fpn_out = list()
        for i in range(1, len(cfg.strides)):
            fpn_out.append(nn.Sequential(nn.Conv2d(cfg.fpn_channel, cfg.fpn_channel, 3, padding=1),
                                         BatchRenorm2d(cfg.fpn_channel),
                                         nn.ReLU()))
        self.fpn_out = nn.ModuleList(fpn_out)
        
        conv5_trans = list()
        self.conv5_pooling = cfg.conv5_pooling
        for _ in cfg.conv5_pooling:
            conv5_trans.append(nn.Sequential(nn.Conv2d(cfg.feature_channel[0], cfg.conv5_channel, 1, bias=False),
                                             BatchRenorm2d(cfg.conv5_channel),
                                             nn.ReLU()))
        self.conv5_trans = nn.ModuleList(conv5_trans)
        self.conv5_fuse = nn.Sequential(nn.Conv2d(cfg.feature_channel[0] + len(cfg.conv5_pooling) * cfg.conv5_channel, cfg.fpn_channel, 3, padding=1),
                                        BatchRenorm2d(cfg.fpn_channel),
                                        nn.ReLU())
        
        self.fpn_fuse = nn.Sequential(nn.Conv2d(len(cfg.feature_channel) * cfg.fpn_channel, cfg.fpn_channel, 3, padding=1),
                                      BatchRenorm2d(cfg.fpn_channel),
                                      nn.ReLU())
        
        self.final = nn.Conv2d(cfg.fpn_channel, 2, 7, padding=3)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, image, features, **kwarg):
        assert len(features) == len(self.fpn_in) + 1
        features = features[::-1]
        for i in range(1, len(features)):
            assert features[i].shape[1] <= features[i-1].shape[1]
        
        conv5 = features.pop(0)

        conv5_trans = list()
        roi = list() # fake rois, just used for pooling
        for i in range(image.shape[0]): # batch size
            roi.append(torch.Tensor([i, 0, 0, conv5.shape[3], conv5.shape[2]]).view(1, -1)) # b, x0, y0, x1, y1
        roi = torch.cat(roi, dim=0).type_as(conv5)
        for scale, trans in zip(self.conv5_pooling, self.conv5_trans):
            f = F.interpolate(roi_align(conv5, roi, scale), (conv5.shape[2], conv5.shape[3]), mode='bilinear', align_corners=False)
            conv5_trans.append(trans(f))
        conv5_trans.append(conv5)
        f = self.conv5_fuse(torch.cat(conv5_trans, dim=1))
        fuse_list = list()
        output_size = features[-1].size()[2:]
        fuse_list.append(F.interpolate(f, size=output_size, mode='bilinear', align_corners=False))
        for i in range(len(features)):
            conv_x = features[i]
            conv_x = self.fpn_in[i](conv_x)
            f = F.interpolate(f, size=conv_x.size()[2:], mode='bilinear', align_corners=False)
            f = conv_x + f
            f = self.fpn_out[i](f)
            fuse_list.append(F.interpolate(f, size=output_size, mode='bilinear', align_corners=False))
        f = self.fpn_fuse(torch.cat(fuse_list, dim=1))

        logits = self.final(f)

        logits = F.interpolate(logits, size=image.shape[2:], mode='bilinear', align_corners=False)

        ret = dict(prob=torch.softmax(logits, dim=1))
        if self.training:
            loss, losses = self.loss(logits=logits, label=kwarg['label'])
            ret['loss'] = loss
            ret['losses'] = losses
        return ret
    
    def get_orishape(self, ret, ori_shape):
        prob = ret['prob']
        bs = prob.shape[0]
        assert bs == 1
        prob = F.interpolate(prob, (ori_shape[0][1], ori_shape[0][0]), mode='bilinear', align_corners=False)
        pred = torch.argmax(prob, dim=1)
        ret['prob'] = prob
        return pred.long()

