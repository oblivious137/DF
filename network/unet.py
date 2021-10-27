import torch
import torch.nn as nn
from torch.nn.modules import padding
from network.batchrenorm import BatchRenorm2d

class UNet(nn.Module):
    def __init__(self, cfg) -> None:
        super(UNet, self).__init__()
        self.upsamplers = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        assert len(cfg.strides) == len(cfg.feature_channel)
        for i in range(len(cfg.strides)):
            inc = cfg.feature_channel[i]
            outc = cfg.feature_channel[i+1] if i+1<len(cfg.feature_channel) else cfg.out_channel
            scale_factor = (cfg.strides[i]//cfg.strides[i+1]) if i+1<len(cfg.strides) else cfg.strides[i]
            self.skipconvs.append(nn.Sequential(
                nn.Conv2d(inc, inc, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(inc),
                nn.ReLU(inplace=True)
            ))
            self.upsamplers.append(nn.Sequential(
                nn.UpsamplingBilinear2d(scale_factor=scale_factor),
                nn.Conv2d(inc, outc, 3, stride=1, padding=1, bias=False),
                BatchRenorm2d(outc),
                nn.ReLU(inplace=True)
            ))
        self.final = nn.Conv2d(outc, 2, 7, padding=3)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    
    def forward(self, features):
        assert len(features) == len(self.skipconvs)
        x = 0
        for i in range(len(features)):
            x = self.skipconvs[i](features[i]) + x
            x = self.upsamplers[i](x)
        x = self.final(x)
        return x
