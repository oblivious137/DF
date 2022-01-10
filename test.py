from network.detector import build_detector
from dataset.UIDataset import UIDataset
from runner import Runner
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='the config file')
parser.add_argument('--load_from', type=str, help='the weight file')
args = parser.parse_args()
config_file = args.config

config_file = os.path.relpath(config_file)
config_file = os.path.splitext(config_file)[0]
config_file = '.'.join(os.path.split(config_file))

cfg = None
exec(f"from {config_file} import config as cfg")

testset = UIDataset(cfg.testset)

model = build_detector(cfg.detector)

solver = Runner(model=model, valset=testset, test_only=True, load_from=args.load_from, cfg=cfg)

import torch
with torch.no_grad():
    solver.run()