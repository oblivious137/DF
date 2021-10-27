import torch
import torch.nn
import numpy as np

def calc_DICE(preds, labels):
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()
        labels = labels.cpu().numpy()
    bs = 1
    if len(preds.shape) == 3:
        bs = preds.shape[0]
    preds = preds.reshape((bs, -1))
    labels = labels.reshape((bs, -1))
    inter = 2*(preds * labels).sum(axis=1)
    union = preds.sum(axis=1)+labels.sum(axis=1)
    return (inter/union).mean()

