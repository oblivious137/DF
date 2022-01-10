from math import exp, pi, sqrt
import torch
import torch.nn as nn
import torch.nn.functional as F
from network.batchrenorm import BatchRenorm2d

class UNetMultilayer(nn.Module):
    def __init__(self, cfg, loss=None) -> None:
        super(UNetMultilayer, self).__init__()
        self.loss = loss
        self.upsamplers = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.outputs = nn.ModuleList()
        self.count = -1
        assert len(cfg.strides) == len(cfg.feature_channel)
        for i in range(len(cfg.strides)):
            cur = cfg.feature_channel[i]
            inc = cur if i==0 else (cur * 2)
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
            self.outputs.append(nn.Sequential(
                nn.Conv2d(cur, cfg.out_channel, 3, padding=1, bias=False),
                BatchRenorm2d(cfg.out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(cfg.out_channel, 2, 3, padding=1),
            ))
        self.final = nn.Sequential(
            nn.Conv2d(outc+3, outc, 3, padding=1, bias=False),
            BatchRenorm2d(outc),
            nn.ReLU(inplace=True),
            nn.Conv2d(outc, 2, 3, padding=1)
        )
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
    

    def generate_weights(self, count):
        def _normalP(x, mu, sigma):
            return exp(-(x-mu)**2/2/(sigma**2))/sqrt(2*pi)/sigma
        num = len(self.skipconvs)+1
        return [1/num] * num
        weights = list()
        center = float(count)/2000-0.3
        center = min(center, num+2.5)
        for i in range(num):
            weights.append(_normalP(i, mu=center, sigma=1.5))
        s = sum(weights)
        for i in range(num):
            weights[i] = weights[i] / s
        return weights

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
                x = self.skipconvs[i](torch.cat((x, features[i]), dim=1))
                # x = self.skipconvs[i](x + features[i])
                assert features[i].shape[1] <= features[i-1].shape[1]
            out = self.outputs[i](x)
            # if isinstance(outi, torch.Tensor):
            #     outi = F.upsample_bilinear(outi, out.shape[2:])
            # outi = outi + out
            outi=out
            logits.append(outi)
            probs.append(torch.softmax(outi, dim=1))
            x = self.upsamplers[i](x)
        # x = self.final(x)
        x = self.final(torch.cat((image, x), dim=1))
        # if isinstance(outi, torch.Tensor):
        #     outi = F.upsample_bilinear(outi, x.shape[2:])
        # outi = x+outi
        outi=x
        logits.append(outi)
        probs.append(torch.softmax(outi, dim=1))
        ret = dict(probs=probs)
        if self.training:
            self.count += image.shape[0]
            loss, losses = self.loss(logits=logits, label=kwarg['label'], weights=self.generate_weights(self.count))
            ret['loss'] = loss
            ret['losses'] = losses
        return ret
    
    def get_orishape(self, ret, ori_shape):
        probs = ret['probs']
        bs = probs[0].shape[0]
        assert bs == 1
        prob = 0
        if self.count >= 0:
            weights = self.generate_weights(self.count)
        else:
            # test only mode
            weights = self.generate_weights(28000)
        for weight, p in zip(weights, probs):
            prob = prob + weight*F.interpolate(p, (ori_shape[0][1], ori_shape[0][0]), mode='bilinear')
        pred = torch.argmax(prob, dim=1)
        prob = prob/torch.sum(prob, dim=1, keepdim=True)
        ret['prob'] = prob
        return pred





class UNetComplexMulti(nn.Module):
    def __init__(self, cfg, loss=None) -> None:
        super(UNetComplexMulti, self).__init__()
        self.loss = loss
        self.upsamplers = nn.ModuleList()
        self.skipconvs = nn.ModuleList()
        self.outputs = nn.ModuleList()
        self.count = -1
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
            self.outputs.append(nn.Sequential(
                nn.Conv2d(cur, cfg.out_channel, 3, padding=1, bias=False),
                BatchRenorm2d(cfg.out_channel),
                nn.ReLU(inplace=True),
                nn.Conv2d(cfg.out_channel, 2, 7, padding=3),
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

    def generate_weights(self, count):
        def _normalP(x, mu, sigma):
            return exp(-(x-mu)**2/2/(sigma**2))/sqrt(2*pi)/sigma
        num = len(self.skipconvs)+1
        # weights = [(2**i) for i in range(num)]
        weights = [1 for i in range(num)]
        s = sum(weights)
        weights = list(map(lambda x: x/s, weights))
        return weights
    
    def forward(self, image, features, **kwarg):
        assert len(features) == len(self.skipconvs)
        features = features[::-1]
        x = 0
        logits=list()
        probs = list()
        for i in range(len(features)):
            if i == 0:
                x = self.skipconvs[i](features[i])
            else:
                x = x + self.skipconvs[i](features[i])
                assert features[i].shape[1] <= features[i-1].shape[1]
            outi = self.outputs[i](x)
            logits.append(outi)
            probs.append(torch.softmax(outi, dim=1))
            x = self.upsamplers[i](x)
        x = self.final(torch.cat((image, x), dim=1))
        logits.append(x)
        ret = dict(probs=probs)
        if self.training:
            self.count += image.shape[0]
            loss, losses = self.loss(logits=logits, label=kwarg['label'], weights=self.generate_weights(self.count))
            ret['loss'] = loss
            ret['losses'] = losses
        return ret
    
    def get_orishape(self, ret, ori_shape):
        probs = ret['probs']
        bs = probs[0].shape[0]
        assert bs == 1
        prob = 0
        if self.count >= 0:
            weights = self.generate_weights(self.count)
        else:
            # test only mode
            weights = self.generate_weights(28000)
        for weight, p in zip(weights, probs):
            prob = prob + weight*F.interpolate(p, (ori_shape[0][1], ori_shape[0][0]), mode='bilinear')
        pred = torch.argmax(prob, dim=1)
        prob = prob/torch.sum(prob, dim=1, keepdim=True)
        ret['prob'] = prob
        return pred