import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from dataset.transforms import build_transforms
import numpy as np
import imageio
from imgaug.augmentables.segmaps import SegmentationMapsOnImage

class UIDataset(Dataset):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.root = cfg.root
        self.imageroot = os.path.join(self.root, "img")
        self.maskroot = os.path.join(self.root, "mask")
        self.load_mask = cfg.load_mask
        with open(os.path.join(self.root, cfg.split_file)) as f:
            img_ids = f.readlines()
            img_ids = list(filter(lambda x: len(x)>0, map(lambda x: x.strip(), img_ids)))
        self.imgids = img_ids
        self.transforms = build_transforms(cfg.transforms)
    
    def __len__(self):
        return len(self.imgids)
    
    def __getitem__(self, index):
        img = imageio.imread(os.path.join(self.imageroot, self.imgids[index]))
        img_info = dict()
        img_info["filename"] = self.imgids[index]
        img_info["ori_shape"] = (img.shape[1], img.shape[0])
        img_info["shape"] = (img.shape[1], img.shape[0])
        img_info["padding"] = (0, 0)
        img_info["image"] = img
        if self.load_mask:
            mask = (imageio.imread(os.path.join(self.maskroot, self.imgids[index]))>0).astype(np.uint8)
            mask = SegmentationMapsOnImage(mask, shape=img_info["image"].shape)
            img_info["label"] = mask
            img_info["ori_label"] = mask.deepcopy()
        img_info = self.transforms(img_info)
        return img_info


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import sys
    sys.path.append(r"D:\subject\DataFountain")
    from configs.dataset import trainset
    ds = UIDataset(trainset)
    for i in range(len(ds)):
        print(f"{i+1}th image, {ds.imgids[i]}")
        data = ds.__getitem__(i)
        plt.figure()
        plt.subplot(1,2,1)
        
        img = data["image"].permute((1,2,0)).numpy()
        img = img - img.min()
        img = img/img.max()
        img = (img * 255).astype(np.uint8)
        mask = SegmentationMapsOnImage(data["label"].numpy().astype(np.int32), shape=img.shape)
        plt.imshow(mask.draw_on_image(img, alpha=0.2)[0])

        img = imageio.imread(os.path.join(ds.imageroot, ds.imgids[i]))
        mask = (imageio.imread(os.path.join(ds.maskroot, ds.imgids[i]))>0)
        mask = SegmentationMapsOnImage(mask, shape=img.shape)
        plt.subplot(1,2,2)
        plt.imshow(mask.draw_on_image(img, alpha=0.2)[0])
        plt.show()