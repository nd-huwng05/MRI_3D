import numpy as np
import scipy.ndimage as ndimage
import torch
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
import os
import time
import random
import scipy.ndimage as ndimage
from torch.utils.tensorboard import SummaryWriter

# model_urls
# load_backbone(backbone='wide_resnet50_2',edc=False)
# CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler)
# TBWrapper
# RescaleSegmentor
# setup_result_path(result_path, model_dir,sub_dir, run_name)


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth'
}


def load_backbone(backbone='wide_resnet50_2', edc=False):
    if backbone == 'resnet18':
        model = models.resnet18(weights=None)
    elif backbone == 'resnet50':
        model = models.resnet50(weights=None)
    elif backbone == 'wide_resnet50_2':
        model = models.wide_resnet50_2(weights=None)

    if edc:
        backbone_path = './../results/MRI_EDC/best_encoder.pth'
        state_dict = torch.load(backbone_path)
    else:
        state_dict_path = model_urls[backbone]
        state_dict = torch.hub.load_state_dict_from_url(state_dict_path, progress=True)
    model.load_state_dict(state_dict)
    return model


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup, max_iters):
        self.warmup = warmup
        self.max_num_iters = max_iters
        super().__init__(optimizer)

    def get_lr(self):
        lr_factor = self.get_lr_factor(epoch=self.last_epoch)
        return [base_lr * lr_factor for base_lr in self.base_lrs]

    def get_lr_factor(self, epoch):
        lr_factor = 0.5 * (1 + np.cos(np.pi * epoch / self.max_num_iters))
        if epoch <= self.warmup:
            lr_factor *= epoch * 1.0 / self.warmup
        return lr_factor


class TBWrapper:
    def __init__(self, log_dir):
        self.g_iter = 0
        self.logger = SummaryWriter(log_dir=log_dir)

    def step(self):
        self.g_iter += 1


class RescaleSegmentor:
    def __init__(self, device, target_size=[192, 192]):
        self.device = device
        self.target_size = target_size
        self.smoothing = 4

    def convert_to_segmentation(self, patch_scores):
        with torch.no_grad():
            if isinstance(patch_scores, np.ndarray):
                patch_scores = torch.from_numpy(patch_scores)
            _scores = patch_scores.to(self.device)
            _scores = _scores.unsqueeze(1)  # [bs,1,h,w]
            _scores = F.interpolate(
                _scores, size=self.target_size, mode='bilinear', align_corners=False
            )
            _scores = _scores.squeeze(1)
            _scores = _scores.cpu().numpy()
            return torch.from_numpy(np.stack([
                ndimage.gaussian_filter(patch_score, sigma=self.smoothing)
                for patch_score in _scores])).to(self.device)
            # return torch.from_numpy(_scores).to(self.device)
            # return  torch.from_numpy(np.stack([
            # ndimage.median_filter(patch_score, size=3)
            # for patch_score in _scores])).to(self.device)


def setup_result_path(result_path, model_dir, sub_dir, run_name):
    # date = time.strftime("-%Y-%m-%d", time.localtime(time.time()))
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    model_path = os.path.join(result_path, model_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    dir_path = os.path.join(model_path, sub_dir)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    run_path = os.path.join(dir_path, run_name)
    if not os.path.exists(run_path):
        os.makedirs(run_path)

    tensorboard_path = os.path.join(run_path, 'tensorboard')
    if not os.path.exists(tensorboard_path):
        os.makedirs(tensorboard_path)

    savedModels_path = os.path.join(run_path, 'savedModels')
    if not os.path.exists(savedModels_path):
        os.makedirs(savedModels_path)

    log_path = os.path.join(run_path, 'log.txt')
    if not os.path.exists(log_path):
        with open(log_path, 'w') as f:
            f.write('')

    return tensorboard_path, savedModels_path, log_path