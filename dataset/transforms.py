from math import ceil
from matplotlib.pyplot import sca
from numpy.lib.arraysetops import isin
from numpy.lib.function_base import select
import torch
import numpy as np
import imgaug.augmenters as iaa

class RandomSelect:
    def __init__(self, transforms=None, select_range=None):
        if select_range is None:
            self.select_range = (0, len(transforms))
        else:
            if isinstance(select_range, int):
                self.select_range = (0, select_range)
            else:
                self.select_range = select_range
        self.transforms = list()
        for T in transforms:
            self.transforms.append(get_transform(T))
    
    def __call__(self, img_info):
        num = int(torch.randint(self.select_range[0], self.select_range[1]+1, (1,)))
        if num > 0:
            np.random.shuffle(list(range(len(self.transforms))))
            for i in range(num):
                img_info = self.transforms[i](img_info)
        return img_info

class Compose:
    def __init__(self, transforms=None, img_mean=None, img_std=None):
        self.transforms = list()
        for T in transforms:
            self.transforms.append(get_transform(T))
        if img_mean is not None:
            self.mean = torch.tensor(img_mean).reshape((1, 1, 3))
            self.std = torch.tensor(img_std).reshape((1, 1, 3))
            if self.mean.max()>1.5:
                self.mean = self.mean.float()/255
                self.std = self.std.float()/255
                input(f"Your std and mean is divided by 255: std={self.std}, mean={self.mean}")
    
    def __call__(self, img_info):
        for T in self.transforms:
            img_info = T(img_info)
        img = torch.from_numpy(np.ascontiguousarray(img_info['image'])).float()/255
        if hasattr(self, "mean"):
            img_info['image'] = ((img-self.mean)/self.std).permute((2,0,1))
            if 'label' in img_info:
                img_info['label'] = torch.from_numpy(np.ascontiguousarray(img_info['label'].get_arr())).long()
            if 'ori_label' in img_info:
                img_info['ori_label'] = torch.from_numpy(np.ascontiguousarray(img_info['ori_label'].get_arr())).long()
        if 'shape' in img_info:
            img_info['shape'] = torch.tensor(img_info['shape'])
        if 'ori_shape' in img_info:
            img_info['ori_shape'] = torch.tensor(img_info['ori_shape'])
        if 'padding' in img_info:
            img_info['padding'] = torch.tensor(img_info['padding'])
        return img_info

class Resize:
    def __init__(self, shapes=[(256, 256)], base_size=1, pad_mode='resize'):
        self.shapes = shapes
        self.base_size = base_size
        self.pad_mode = pad_mode
    
    def __call__(self, img_info):
        idx = int(torch.randint(0, len(self.shapes)-1, (1,)))
        
        aw, ah = self.shapes[idx]
        w, h = img_info["shape"]
        scale = min(aw/w, ah/h)
        aw, ah = (round(w*scale), round(h*scale))

        if self.pad_mode == 'resize':
            aw = round(aw/self.base_size) * self.base_size
            ah = round(ah/self.base_size) * self.base_size

        aug = iaa.Resize({"height": ah, "width": aw})
        if "label" in img_info:
            img_info["image"], img_info["label"] = aug(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = aug(image=img_info["image"])
        
        if self.pad_mode == 'zeropad':
            w, h = (aw, ah)
            aw = ceil(aw/self.base_size) * self.base_size
            ah = ceil(ah/self.base_size) * self.base_size
            pw, ph = (aw - w, ah - h)
            if pw>0 or ph>0:
                img_info["image"] = iaa.size.pad(img_info["image"], right=pw, bottom=ph)
                if "label" in img_info:
                    img_info["label"] = img_info["label"].pad(right=pw, bottom=ph)
        img_info["shape"] = (aw, ah)

        return img_info
        
class Crop:
    def __init__(self, ratio=0.1):
        if isinstance(ratio, float):
            ratio = (0, ratio)
        self.crop = iaa.Crop(percent=ratio)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.crop(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.crop(image=img_info["image"])
        
        return img_info

class Fliplr:
    def __init__(self, ratio=0.5):
        self.flip = iaa.Fliplr(ratio)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.flip(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.flip(image=img_info["image"])
        
        return img_info
 
class Flipud:
    def __init__(self, ratio=0.5):
        self.flip = iaa.Flipud(ratio)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.flip(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.flip(image=img_info["image"])
        
        return img_info

class Rotate:
    def __init__(self, angle_range=(-5, 5)):
        if isinstance(angle_range, int):
            angle_range = (-angle_range, angle_range)
        self.rotate = iaa.Rotate(angle_range)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.rotate(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.rotate(image=img_info["image"])
        
        return img_info

class ShearX:
    def __init__(self, shear_range=(-5, 5)):
        if isinstance(shear_range, int):
            shear_range = (-shear_range, shear_range)
        self.shear = iaa.ShearX(shear_range)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.shear(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.shear(image=img_info["image"])
        
        return img_info
        
class ShearY:
    def __init__(self, shear_range=(-5, 5)):
        if isinstance(shear_range, int):
            shear_range = (-shear_range, shear_range)
        self.shear = iaa.ShearY(shear_range)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.shear(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.shear(image=img_info["image"])
        
        return img_info

class ScaleX:
    def __init__(self, scale=(0.8, 1.2)):
        if isinstance(scale, float):
            scale = (1-scale, 1+scale)
        self.scale = iaa.ScaleX(scale)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.scale(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.scale(image=img_info["image"])
        
        return img_info

class ScaleY:
    def __init__(self, scale=(0.8, 1.2)):
        if isinstance(scale, float):
            scale = (1-scale, 1+scale)
        self.scale = iaa.ScaleY(scale)
    
    def __call__(self, img_info):
        if "label" in img_info:
            img_info["image"], img_info["label"] = self.scale(image=img_info["image"], segmentation_maps=img_info["label"])
        else:
            img_info["image"] = self.scale(image=img_info["image"])
        
        return img_info

class GaussianBlur:
    def __init__(self, sigma=(0.0, 3.0)):
        self.blur = iaa.GaussianBlur(sigma)
    
    def __call__(self, img_info):
        img_info["image"] = self.blur(image=img_info["image"])
        return img_info

class BilateralBlur:
    def __init__(self, d=(3, 10), sigma_color=(10, 250), sigma_space=(10, 250)):
        self.blur = iaa.BilateralBlur(d=d, sigma_color=sigma_color, sigma_space=sigma_space)
    
    def __call__(self, img_info):
        img_info["image"] = self.blur(image=img_info["image"])
        return img_info

class GaussianNoise:
    def __init__(self, loc=0, scale=(0, 15), per_channel=False):
        self.noise = iaa.AdditiveGaussianNoise(loc=loc, scale=scale, per_channel=per_channel)
    
    def __call__(self, img_info):
        img_info["image"] = self.noise(image=img_info["image"])
        return img_info

class LaplaceNoise:
    def __init__(self, loc=0, scale=(0, 15), per_channel=False):
        self.noise = iaa.AdditiveLaplaceNoise(loc=loc, scale=scale)
    
    def __call__(self, img_info):
        img_info["image"] = self.noise(image=img_info["image"])
        return img_info

class PepperNoise:
    def __init__(self, p=(0.02, 0.05), size_percent = (0.9, 1)):
        self.noise = iaa.CoarsePepper(p=p, size_percent=size_percent)
    
    def __call__(self, img_info):
        img_info["image"] = self.noise(image=img_info["image"])
        return img_info

class SaltNoise:
    def __init__(self, p=(0.02, 0.05), size_percent = (0.9, 1)):
        self.noise = iaa.CoarseSalt(p=p, size_percent=size_percent)
    
    def __call__(self, img_info):
        img_info["image"] = self.noise(image=img_info["image"])
        return img_info
        
class GammaContrast:
    def __init__(self, gamma=(0.7, 1.7)):
        self.contrast = iaa.GammaContrast(gamma=gamma)
    
    def __call__(self, img_info):
        img_info["image"] = self.contrast(image=img_info["image"])
        return img_info
        
class LogContrast:
    def __init__(self, gain=(0.4, 1.6)):
        self.contrast = iaa.LogContrast(gain=gain)
    
    def __call__(self, img_info):
        img_info["image"] = self.contrast(image=img_info["image"])
        return img_info
        


def get_transform(cfg):
    transform = eval(cfg.pop('name'))
    return transform(**cfg)

build_transforms = get_transform