import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torchsummary import summary as tc_summary

from .networks import create_model
from .base_solver import BaseSolver


class SRGANModel(BaseSolver):
    def __init__(self):
        super(SRGANModel, self).__init__()
        self.results = {'training_d_loss': [],
                        'training_g_loss': [],
                        'val_loss': [],
                        'real_score': [],
                        'fake_score': [],
                        'psnr': [],
                        'ssim': []}

