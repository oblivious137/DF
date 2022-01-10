from genericpath import exists
import os
import torch
from torch.serialization import save
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import calc_DICE
import time
import shutil
import pickle
import numpy as np
from utils import DenseCRF
from imageio import imread


def build_optimizer(model, cfg):
    name = cfg.pop('name')
    weights = list()
    biases = list()
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if 'bias' in n:
            biases.append(p)
        else:
            weights.append(p)
    biases = dict(params=biases)
    weights = dict(params=weights)
    weight_decay = cfg.get('weight_decay', 0.0)
    lr = cfg.lr
    if 'paramwise_cfg' in cfg:
        paramwise_cfg = cfg.pop('paramwise_cfg')
        biases['lr'] = lr * paramwise_cfg.get('bias_lr_mult', 1.0)
        biases['weight_decay'] = weight_decay * paramwise_cfg.get('bias_decay_mult', 1.0)
    
    if name == 'SGD':
        return torch.optim.SGD([weights, biases], **cfg)


def build_scheduler(optim, cfg, start_epoch=1):
    name = cfg.pop('policy')
    if name == 'step':
        optim = torch.optim.lr_scheduler.MultiStepLR(optim, milestones=cfg.step, gamma=cfg.gamma)
    else:
        assert False
    
    if start_epoch != 1:
        optim.step(start_epoch-1)
    
    return optim

class Runner:
    def __init__(self, model=None, trainset=None, valset=None, cfg=None, test_only=False, load_from=None) -> None:
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.test_only = test_only
        self.cfg = cfg.solver
        self.rootdir = self.cfg.checkpoint_dir
        self.count = 0
        if isinstance(load_from, str):
            self.model.load_state_dict(torch.load(load_from))
        
        print(self.model)
        print(self.trainset)
        print(self.valset)

        if test_only:
            return
        self.max_epoch = self.cfg.max_epoch
        self.start_epoch = self.cfg.get('start_epoch', 1)
        self.count += (self.start_epoch-1) * len(self.trainset)
        self.optim = build_optimizer(model, self.cfg.optimizer)
        self.sched = build_scheduler(self.optim, self.cfg.scheduler, self.start_epoch)
        print(self.cfg)
        print(self.optim)
        print(self.sched)
        self.print_freq = self.cfg.print_freq
        self.save_freq = self.cfg.save_freq
        log_dir = os.path.join(self.rootdir, "log")
        if os.path.exists(log_dir):
            shutil.rmtree(log_dir)
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(self.rootdir, "cfg.pkl"), "wb") as f:
            pickle.dump(dict(**cfg), f)
        self.writer = SummaryWriter(log_dir)

    def train(self, epoch_th, warmup=False):
        self.model.train()
        dataloader = DataLoader(self.trainset, batch_size=self.cfg.batch_size, shuffle=True)
        for i, data in enumerate(dataloader):
            for k,v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
            if not warmup:
                self.count += data['image'].shape[0]
            
            self.optim.zero_grad()
            output = self.model(**data)
            loss = output['loss']
            loss.backward()
            self.optim.step()

            dice = output['DICE']
            if not warmup:
                for k, v in output['losses'].items():
                    self.writer.add_scalar(f"Loss/{k}", v, global_step=self.count)
                self.writer.add_scalar("Dice/train", dice, global_step=self.count)
                if i % self.print_freq == 0:
                    print(f"[{i}/{epoch_th}] time: {time.asctime(time.localtime(time.time()))}")
                    self.writer.add_images("Detect/train", self.visualize(data['image'], output['pred'], data['label']), global_step=self.count)
            del output

    def visualize(self, image, pred, label):
        image = image.detach().cpu()
        image = image-image.min()
        image = image/image.max()
        label = label.cpu()
        pred = pred.cpu()
        image[:, 0, :, :] = image[:, 0, :, :] * 0.6 + pred.float() * 0.4
        image[:, 1, :, :] = image[:, 1, :, :] * 0.6 + label.float() * 0.4
        return image

    def eval(self, epoch_th, save_res=False):
        self.model.eval()
        dataloader = DataLoader(self.valset, batch_size=1, shuffle=False)
        dices = list()
        # res = list()
        for i, data in enumerate(dataloader):
            for k,v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
            output = self.model(**data)
            pred = output['pred']
            dice = calc_DICE(pred, data['ori_label'])
            dices.append(dice)
            # res.append(pred)
            self.writer.add_images("Detect/test", self.visualize(data['ori_image'], output['pred'], data['ori_label']), global_step=(epoch_th-1)*len(dataloader)+i)
        dice = sum(dices)/len(dices)
        self.writer.add_scalar("Dice/test", dice, global_step=epoch_th)
        print(f"[epoch {epoch_th}] Average DICE:", dice)
        # if save_res:
        #     torch.save(res, os.path.join(self.rootdir, "res.pth"))

    def test(self):
        self.model.eval()
        dataloader = DataLoader(self.valset, batch_size=1, shuffle=False)
        res = list()
        if 'CRF' in self.cfg:
            postprocessor = DenseCRF(self.cfg.CRF.ITER_MAX, self.cfg.CRF.POS_W, self.cfg.CRF.POS_XY_STD, self.cfg.CRF.BI_W, self.cfg.CRF.BI_XY_STD, self.cfg.CRF.BI_RGB_STD)
        for data in dataloader:
            for k,v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
            output = self.model(**data)
            if 'CRF' in self.cfg:
                prob = output['prob'][0].cpu().numpy()
                image = imread(os.path.join('test_dataset_A', 'img', str(data['filename'][0])))
                prob = postprocessor(image, prob)
                pred = np.argmax(prob, axis=0)
            else:
                pred = output['pred'].cpu().numpy()
            res.append(pred)
        with open(os.path.join(self.rootdir, "res.pkl"), "wb") as f:
            pickle.dump(res, f)

    def run(self):
        self.model.cuda()
        if self.test_only:
            self.test()
        else:
            if 'warmup' in self.cfg:
                for param in self.optim.param_groups:
                    param['lr'] *= 1e-6
                print("warmuping...")
                for i in range(self.cfg.warmup):
                    self.train(None, True)
                    print(f"warmup {i+1}")
                print("warmup finish")
                for param in self.optim.param_groups:
                    param['lr'] *= 1e6
            for epoch in range(self.start_epoch, self.max_epoch+1):
                print(f"epoch #{epoch} start, lr={self.sched.get_last_lr()}")
                self.train(epoch)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.rootdir, f"epoch_{epoch}.pth"))
                self.eval(epoch)
                self.sched.step()
            torch.save(self.model.state_dict(), os.path.join(self.rootdir, f"epoch_final.pth"))
    