from configs.config import _CFG
from configs.dataset_base import *

trainset.transforms.update(_CFG(
        style="caffe",
        img_std = [1.0, 1.0, 1.0],
        img_mean = [102.9801, 115.9465, 122.7717],
    )
)



valset.transforms.update(_CFG(
        style="caffe",
        img_std = [1.0, 1.0, 1.0],
        img_mean = [102.9801, 115.9465, 122.7717],
    )
)


testset.transforms.update(_CFG(
        style="caffe",
        img_std = [1.0, 1.0, 1.0],
        img_mean = [102.9801, 115.9465, 122.7717],
    )
)