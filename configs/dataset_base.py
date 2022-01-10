from configs.config import _CFG

trainset = _CFG(
    root = r"D:\subject\DataFountain\train_dataset",
    split_file = "train.txt",
    load_mask = True,
    train = True,
    transforms = _CFG(
        name = "Compose",
        style="pytorch",
        img_std = [0.229, 0.224, 0.225],
        img_mean = [0.485, 0.456, 0.406],
        transforms = [
            _CFG(
                name = "Resize",
                shapes = [(1000, 240), (1000, 280), (1000, 320), (1000, 360), (1000, 400), (1000, 440), (1000, 480), (1000, 520)],
                base_size = 32,
                pad_mode='resize',
            ),
            _CFG(
                name = "Crop",
                ratio = 0.2
            ),
            _CFG(
                name = "Fliplr",
                ratio = 0.5
            ),
            _CFG(
                name = "Flipud",
                ratio = 0.5
            ),
            _CFG(
                # shape transform
                name = "RandomSelect",
                transforms = [
                    _CFG(
                        name = "Rotate",
                        angle_range = (-25, 25)
                    ),
                    _CFG(
                        name = "ShearX",
                        shear_range = (-15, 15)
                    ),
                    _CFG(
                        name = "ShearY",
                        shear_range = (-15, 15)
                    ),
                    _CFG(
                        name = "ScaleX",
                        scale = (0.8, 1.2)
                    ),
                    _CFG(
                        name = "ScaleY",
                        scale = (0.8, 1.2)
                    ),
                    _CFG(
                        name = "PiecewiseAffine",
                        scale = (0.01, 0.05),
                        nb_rows = (2, 6),
                        nb_cols = (2, 6),
                    ),
                ],
                select_range = 3,
            ),
            _CFG(
                # blur
                name = "RandomSelect",
                transforms = [
                    _CFG(
                        name = "GaussianBlur",
                    ),
                    _CFG(
                        name = "BilateralBlur",
                    )
                ],
                select_range = 1
            ),
            _CFG(
                # Noise
                name = "RandomSelect",
                transforms = [
                    _CFG(
                        name = "GaussianNoise",
                        loc=0, scale=(0, 15)
                    ),
                    _CFG(
                        name = "PepperNoise",
                        p=(0.01, 0.03), size_percent = (0.95, 1)
                    ),
                    _CFG(
                        name = "SaltNoise",
                        p=(0.01, 0.03), size_percent = (0.95, 1)
                    )
                ],
                select_range = 2
            ),
            _CFG(
                # Contrast
                name = "RandomSelect",
                transforms = [
                    _CFG(
                        name = "GammaContrast",
                        gamma=(0.7, 1.7)
                    ),
                    _CFG(
                        name = "LogContrast",
                        gain=(0.4, 1.6)
                    ),
                ],
                select_range = 1
            )
        ],
    )
)



valset = _CFG(
    root = r"D:\subject\DataFountain\train_dataset",
    split_file = "valid.txt",
    load_mask = True,
    train = False,
    transforms = _CFG(
        name = "Compose",
        img_std = [0.229, 0.224, 0.225],
        img_mean = [0.485, 0.456, 0.406],
        transforms = [
            _CFG(
                name = "Resize",
                shapes = [(1000, 400)],
                base_size = 32,
            )
        ]
    )
)

testset = _CFG(
    root = r"D:\subject\DataFountain\test_dataset_A",
    split_file = "test.txt",
    load_mask = False,
    train = False,
    transforms = _CFG(
        name = "Compose",
        img_std = [0.229, 0.224, 0.225],
        img_mean = [0.485, 0.456, 0.406],
        transforms = [
            _CFG(
                name = "Resize",
                shapes = [(1000, 400)],
                base_size = 32,
            )
        ]
    )
)