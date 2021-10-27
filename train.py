from network.detector import Detector
from dataset.UIDataset import UIDataset
from runner import Runner
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument('--config', type=str, help='the config file')
args = parser.parse_args()
config_file = args.config

config_file = os.path.splitext(config_file)[0]
config_file = config_file.replace('/', '.')
config_file = config_file.replace('\\', '.')

cfg = None
exec(f"import {config_file} as cfg")

valset = UIDataset(cfg.valset)
trainset = UIDataset(cfg.trainset)

model = Detector(cfg.detector)

solver = Runner(model=model, trainset=trainset, valset=valset, cfg=cfg.solver)

solver.run()