from math import exp, pi, sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.batchrenorm import BatchRenorm2d


class Heatmaper(nn.Module):
    def __init__(self, inc, output, pool_size=(7, 7)) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(pool_size),
                                nn.Flatten(1),
                                nn.Linear(inc * pool_size[0] * pool_size[1], 512),
                                nn.ReLU(), nn.Dropout(), nn.Linear(512, 2), nn.Sigmoid())
        self.output = output

    def forward(self, feature):
        h, w = feature.shape[2:]
        pred = self.output(feature)
        pred = torch.softmax(pred, dim=1)[:, 1, :, :]
        pred = pred / pred.reshape(-1,h*w).sum(dim=-1).reshape(-1,1,1)
        ha = torch.arange(0, h).reshape(1, h).to(pred.device)
        wa = torch.arange(0, w).reshape(1, w).to(pred.device)
        y = (ha * pred.sum(dim=2)).sum(dim=1, keepdim=True)
        x = (wa * pred.sum(dim=1)).sum(dim=1, keepdim=True)
        ha = ha - y
        wa = wa - x
        scale = torch.Tensor([[h/7, w/7]]).to(x.device)
        sigma = (self.fc(feature) * 99 + 1) * scale
        ha = (ha*ha / sigma[:,0] / sigma[:,0]).exp() / sigma[:,0]
        wa = (wa*wa / sigma[:,1] / sigma[:,1]).exp() / sigma[:,1]
        heatmap = ha.reshape(-1, h, 1) * wa.reshape(-1, 1, w) * 10
        logit = torch.stack((1-heatmap, heatmap), dim=1)
        return logit


class UNetHeatmap(nn.Module):
    def __init__(self, cfg, loss=None) -> None:
        super(UNetHeatmap, self).__init__()
        self.loss = loss
        self.upsamplers = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.outputs = nn.ModuleList()
        self.count = -1
        assert len(cfg.strides) == len(cfg.feature_channel)
        for i in range(len(cfg.strides)):
            cur = cfg.feature_channel[i]
            inc = cur if i == 0 else (cur * 2 + 2)
            outc = cfg.feature_channel[i+1] if i + \
                1 < len(cfg.feature_channel) else cfg.out_channel
            scale_factor = (
                cfg.strides[i]//cfg.strides[i+1]) if i+1 < len(cfg.strides) else cfg.strides[i]
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
            if i != 0:
                self.outputs.append(nn.Sequential(
                    nn.Conv2d(cur, 2, 3, padding=1),
                ))
            else:
                self.outputs.append(nn.Sequential(
                    Heatmaper(cur, nn.Conv2d(cur, 2, 3, padding=1)),
                ))
        self.final = nn.Sequential(
            nn.Conv2d(outc+3, outc, 3, padding=1, bias=False),
            BatchRenorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, 2, 3, padding=1)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode="fan_out", nonlinearity="relu")

    def forward(self, image, features, **kwarg):
        assert len(features) == len(self.skipconvs)
        features = features[::-1]
        x = 0
        logits = list()
        probs = list()
        outi = 0
        for i in range(len(features)):
            if i == 0:
                x = self.skipconvs[i](features[i])
            else:
                outi = F.interpolate(outi, size=x.shape[2:])
                x = self.skipconvs[i](torch.cat((x, features[i], outi), dim=1))
                assert features[i].shape[1] <= features[i-1].shape[1]
            out = self.outputs[i](x)
            outi = outi + out
            logits.append(outi)
            probs.append(torch.softmax(outi, dim=1))
            x = self.upsamplers[i](x)
        x = self.final(torch.cat((image, x), dim=1))
        outi = F.interpolate(outi, size=x.shape[2:])
        outi = x+outi
        logits.append(outi)
        probs.append(torch.softmax(outi, dim=1))
        ret = dict(probs=probs)
        if self.training:
            self.count += image.shape[0]
            loss, losses = self.loss(
                logits=logits, label=kwarg['label'], weights=self.generate_weights(self.count))
            ret['loss'] = loss
            ret['losses'] = losses
        return ret

    def generate_weights(self, count):
        num = len(self.skipconvs)+1
        return [1/num] * num

    def get_orishape(self, ret, ori_shape):
        probs = ret['probs']
        bs = probs[0].shape[0]
        assert bs == 1
        prob = F.interpolate(
                    probs[-1], (ori_shape[0][1], ori_shape[0][0]), mode='bilinear')
        pred = torch.argmax(prob, dim=1)
        prob = prob/torch.sum(prob, dim=1, keepdim=True)
        ret['prob'] = prob
        return pred

