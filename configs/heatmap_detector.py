from configs.config import _CFG
from configs.dataset_pytorch import *

config=_CFG(
    trainset=trainset,
    valset=valset,
    testset=testset,
    detector = _CFG(
        backbone=_CFG(
            name="Resnet",
            depth=50,
            pretrained=True,
            output_layers=[0, 1, 2, 3, 4],
            frozen=4,
        ),
        decoder=_CFG(
            name="UNet-Heatmap",
            strides=[32, 16, 8, 4, 2],
            feature_channel=[2048, 1024, 512, 256, 64],
            out_channel=32
        ),
        loss=[
            dict(type="DICELoss"),
        ]
    ),

    solver = _CFG(
        max_epoch=160,
        batch_size=1,
        print_freq=50,
        save_freq=10,
        checkpoint_dir="checkpoints/UNet-Heatmap",
        optimizer=_CFG(name="SGD", lr=0.01, momentum=0.96, weight_decay=0.0005,
                    paramwise_cfg=dict(bias_decay_mult=0., bias_lr_mult=2.)),
        scheduler=_CFG(
            policy='step',
            step=[55, 130],
            gamma=0.1,
        ),
        # CRF=_CFG(
        #     ITER_MAX = 10,
        #     POS_W = 3,
        #     POS_XY_STD = 3,
        #     BI_W = 4,
        #     BI_XY_STD = 67,
        #     BI_RGB_STD = 3,
        # )
    )
)
