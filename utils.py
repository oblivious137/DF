import torch
import torch.nn
import numpy as np
import pydensecrf.densecrf as dcrf
import pydensecrf.utils as utils

def calc_DICE_tensor(preds, labels):
    bs = 1
    if len(preds.shape) == 3:
        bs = preds.shape[0]
    preds = preds.reshape((bs, -1))
    labels = labels.reshape((bs, -1))
    inter = 2*(preds * labels).sum(axis=1)
    union = preds.sum(dim=1)+labels.sum(dim=1)
    return (inter/union).mean().item()
    

def calc_DICE(preds, labels):
    if isinstance(preds, torch.Tensor):
        return calc_DICE_tensor(preds, labels)
    bs = 1
    if len(preds.shape) == 3:
        bs = preds.shape[0]
    preds = preds.reshape((bs, -1))
    labels = labels.reshape((bs, -1))
    inter = 2*(preds * labels).sum(axis=1)
    union = preds.sum(axis=1)+labels.sum(axis=1)
    return (inter/union).mean()

class DenseCRF(object):
    def __init__(self, iter_max, pos_w, pos_xy_std, bi_w, bi_xy_std, bi_rgb_std):
        self.iter_max = iter_max
        self.pos_w = pos_w
        self.pos_xy_std = pos_xy_std
        self.bi_w = bi_w
        self.bi_xy_std = bi_xy_std
        self.bi_rgb_std = bi_rgb_std

    def __call__(self, image, probmap):
        C, H, W = probmap.shape

        U = utils.unary_from_softmax(probmap)
        U = np.ascontiguousarray(U)

        image = np.ascontiguousarray(image)

        d = dcrf.DenseCRF2D(W, H, C)
        d.setUnaryEnergy(U)
        d.addPairwiseGaussian(sxy=self.pos_xy_std, compat=self.pos_w)
        d.addPairwiseBilateral(
            sxy=self.bi_xy_std, srgb=self.bi_rgb_std, rgbim=image, compat=self.bi_w
        )
        Q = d.inference(self.iter_max)
        Q = np.array(Q).reshape((C, H, W))

        return Q