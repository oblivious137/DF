from configs.config import _CFG
from configs.dataset import trainset, valset

detector = _CFG(
    backbone = _CFG(
        name = "Resnet",
        depth = 50,
        pretrained = True,
        output_layers = [0, 1, 2, 3, 4],
        frozen = 4,
    ),
    decoder = _CFG(
        name = "UNet",
        strides = [32, 16, 8, 4, 2], # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        feature_channel = [2048, 1024, 512, 256, 64],
        out_channel = 32
    ),
    loss = _CFG(
        name = "CrossEntropyLoss"
    )
)

solver = _CFG(
    max_epoch = 10,
    batch_size = 1,
    print_freq = 20,
    save_freq = 2,
    checkpoint_dir = "checkpoints/UNet",
    optimizer = _CFG(name="SGD", lr = 0.001, momentum = 0.9, weight_decay=0.0005,
                     paramwise_cfg=dict(bias_decay_mult=0., bias_lr_mult=2.)),
    scheduler = _CFG(
        policy='step',
        step=[7]
    )
)