from math import log
from pickle import dump
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.batchrenorm import BatchRenorm2d

class UNet(nn.Module):
    def __init__(self, cfg, loss=None) -> None:
        super(UNet, self).__init__()
        self.loss = loss
        self.upsamplers = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        assert len(cfg.strides) == len(cfg.feature_channel)
        for i in range(len(cfg.strides)):
            cur = cfg.feature_channel[i]
            inc = cur if i==0 else (cur*2)
            outc = cfg.feature_channel[i+1] if i+1<len(cfg.feature_channel) else cfg.out_channel
            scale_factor = (cfg.strides[i]//cfg.strides[i+1]) if i+1<len(cfg.strides) else cfg.strides[i]
            self.skipconvs.append(nn.Sequential(
                nn.Conv2d(inc, cur, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(cur),
                nn.ReLU(inplace=True),
            ))
            self.upsamplers.append(nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                nn.Conv2d(cur, outc, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(outc),
                nn.ReLU(inplace=True)
            ))
        self.final = nn.Sequential(
            nn.Conv2d(outc+3, outc, 3, padding=1),
            BatchRenorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc//2, 3, padding=1),
            BatchRenorm2d(outc//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc//2, 2, 1),
        )
        # self.final = nn.Conv2d(outc, 2, 7, padding=3)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, image, features, **kwarg):
        assert len(features) == len(self.skipconvs)
        features = features[::-1]
        x = 0
        for i in range(len(features)):
            if i == 0:
                x = self.skipconvs[i](features[i])
            else:
                x = self.skipconvs[i](torch.cat((x, features[i]), dim=1))
                assert features[i].shape[1] <= features[i-1].shape[1]
            x = self.upsamplers[i](x)
        x = self.final(torch.cat((image, x), dim=1))
        # x = self.final(x)
        logits=x
        ret = dict(prob=torch.softmax(logits, dim=1))
        if self.training:
            loss, losses = self.loss(logits=x, label=kwarg['label'])
            ret['loss'] = loss
            ret['losses'] = losses
        return ret
    
    def get_orishape(self, ret, ori_shape):
        prob = ret['prob']
        bs = prob.shape[0]
        assert bs == 1
        prob = F.interpolate(prob, (ori_shape[0, 1], ori_shape[0, 0]), mode='bilinear')
        pred = torch.argmax(prob, dim=1)
        ret['prob'] = prob
        return pred.long()



class UNetComplex(nn.Module):
    def __init__(self, cfg, loss=None) -> None:
        super(UNetComplex, self).__init__()
        self.loss = loss
        self.upsamplers = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        assert len(cfg.strides) == len(cfg.feature_channel)
        for i in range(len(cfg.strides)):
            cur = cfg.feature_channel[i]
            inc = cur if i==0 else (cur*2)
            outc = cfg.feature_channel[i+1] if i+1<len(cfg.feature_channel) else cfg.out_channel
            scale_factor = (cfg.strides[i]//cfg.strides[i+1]) if i+1<len(cfg.strides) else cfg.strides[i]
            self.skipconvs.append(nn.Sequential(
                nn.Conv2d(cur, cur, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(cur),
                nn.ReLU(inplace=True),
                nn.Conv2d(cur, cur, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(cur),
                nn.ReLU(inplace=True),
            ))
            self.upsamplers.append(nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                nn.Conv2d(cur, outc, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(outc),
                nn.ReLU(inplace=True),
                nn.Conv2d(outc, outc, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(outc),
                nn.ReLU(inplace=True)
            ))
        self.final = nn.Sequential(
            nn.Conv2d(outc+3, outc, 3, padding=1, bias=False),
            BatchRenorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, outc//2, 3, padding=1, bias=False),
            BatchRenorm2d(outc//2),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc//2, 2, 7, padding=3),
        )
        # self.final = nn.Conv2d(outc, 2, 7, padding=3)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, image, features, **kwarg):
        assert len(features) == len(self.skipconvs)
        features = features[::-1]
        x = 0
        for i in range(len(features)):
            if i == 0:
                x = self.skipconvs[i](features[i])
            else:
                # x = self.skipconvs[i](torch.cat((x, features[i]), dim=1))
                x = x + self.skipconvs[i](features[i])
                assert features[i].shape[1] <= features[i-1].shape[1]
            x = self.upsamplers[i](x)
        x = self.final(torch.cat((image, x), dim=1))
        # x = self.final(x)
        logits=x
        ret = dict(prob=torch.softmax(logits, dim=1))
        if self.training:
            loss, losses = self.loss(logits=x, label=kwarg['label'])
            ret['loss'] = loss
            ret['losses'] = losses
        return ret
    
    def get_orishape(self, ret, ori_shape):
        prob = ret['prob']
        bs = prob.shape[0]
        assert bs == 1
        prob = F.interpolate(prob, (ori_shape[0, 1], ori_shape[0, 0]), mode='bilinear')
        pred = torch.argmax(prob, dim=1)
        ret['prob'] = prob
        return pred