from configs.config import _CFG
from configs.dataset_caffe import trainset, valset, testset

config=_CFG(
    trainset=trainset,
    valset=valset,
    testset=testset,
    detector = _CFG(
        backbone=_CFG(
            name="Resnet-ws",
            depth=50,
            pretrained = "pretrained/resnetw50.pth",
            out_indices=(0, 1, 2, 4),
            num_stages=4,
            strides=(1, 2, 2, 2),
            dilations=(1, 1, 1, 1),
            frozen_stages=4,
            norm_cfg=dict(type='BN', requires_grad=False),
            norm_eval=True,
            style='pytorch',
            small_kernel=True,
            avg2max=True,
            dilated_conv=True,
            weak_backward=False,
        ),
        decoder=_CFG(
            name="UNet-MultiLayer",
            strides=[8, 8, 4, 2],  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
            feature_channel=[2048, 512, 256, 64],
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
        checkpoint_dir="checkpoints/UNet-DICE-Resnetws",
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
        #     POS_XY_STD = 1,
        #     BI_W = 4,
        #     BI_XY_STD = 67,
        #     BI_RGB_STD = 3,
        # )
    )
)
