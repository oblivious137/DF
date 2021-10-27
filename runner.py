import os
import torch
from torch.serialization import save
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import calc_DICE
import time


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


def build_scheduler(optim, cfg):
    name = cfg.pop('policy')
    if name == 'step':
        return torch.optim.lr_scheduler.MultiStepLR(optim, cfg.step)

class Runner:
    def __init__(self, model=None, trainset=None, valset=None, cfg=None, eval_only=False) -> None:
        self.model = model
        self.trainset = trainset
        self.valset = valset
        self.cfg = cfg
        self.eval_only = eval_only
        self.count = 0
        self.rootdir = cfg.checkpoint_dir
        
        print(self.model)
        print(self.trainset)
        print(self.valset)
        print(self.cfg)

        if self.cfg is None:
            return
        self.max_epoch = cfg.max_epoch
        self.optim = build_optimizer(model, cfg.optimizer)
        self.sched = build_scheduler(self.optim, cfg.scheduler)
        print(self.optim)
        print(self.sched)
        self.print_freq = cfg.print_freq
        log_dir = os.path.join(self.rootdir, "log")
        if os.path.exists(log_dir):
            os.removedirs()
        os.makedirs(log_dir)
        self.writer = SummaryWriter(log_dir)

    def train(self, epoch_th):
        self.model.train()
        dataloader = DataLoader(self.trainset, batch_size=self.cfg.batch_size, shuffle=True)
        for i, data in enumerate(dataloader):
            for k,v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
            self.count += data['image'].shape[0]
            
            self.optim.zero_grad()
            loss, dice = self.model(**data)
            loss.backward()
            self.optim.step()

            self.writer.add_scalar("Loss", loss.item(), global_step=self.count)
            self.writer.add_scalar("Dice/train", dice, global_step=self.count)
            if i % self.print_freq == 0:
                print(f"[{i}]/[{epoch_th}] time:{time.asctime(time.localtime(time.time()))}")

    def eval(self, epoch_th, save_res=False):
        self.model.eval()
        dataloader = DataLoader(self.valset, batch_size=1, shuffle=False)
        dices = list()
        res = list()
        for i, data in enumerate(dataloader):
            for k,v in data.items():
                if isinstance(v, torch.Tensor):
                    data[k] = v.cuda()
            pred = self.model(**data)
            dice = calc_DICE(pred, data['ori_label'])
            dices.append(dice)
            res.append(pred)
        dice = sum(dices)/len(dices)
        if hasattr(self, "writer"):
            self.writer.add_scalar("Dice/test", dice, global_step=epoch_th)
        print(f"[epoch {epoch_th}] Average DICE:", dice)
        if save_res:
            torch.save(res, os.path.join(self.rootdir, "res.pth"))

    def run(self):
        self.model.cuda()
        if self.eval_only:
            self.eval(0, True)
        else:
            for epoch in range(1, self.max_epoch+1):
                self.train(epoch)
                if epoch % self.save_freq == 0:
                    torch.save(self.model.state_dict(), os.path.join(self.rootdir, f"epoch_{epoch}.pth"))
                self.eval(epoch)
            torch.save(self.model.state_dict(), os.path.join(self.rootdir, f"epoch_final.pth"))
    