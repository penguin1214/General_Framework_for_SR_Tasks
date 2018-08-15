import os
import torch
import torch.nn as nn
from utils.tf_logger import Logger as TFLogger


class BaseSolver(object):
    def __init__(self, opt):
        self.opt = opt
        self.scale = opt['scale']
        self.save_dir = opt['path']['models']
        self.is_train = opt['is_train']
        self.use_gpu = torch.cuda.is_available()
        self.Tensor = torch.cuda.FloatTensor if self.use_gpu else torch.FloatTensor
        # log
        self.checkpoint_dir = opt['path']['epoch']
        self.log_dir = opt['path']['log']
        self.results_dir = opt['path']['results']
        self.vis_dir = opt['path']['vis']
        self.test_sr_dir = os.path.join(self.results_dir, 'sr')
        self.tf_logger = TFLogger(self.log_dir)
        self.log_step = opt['train']['log_step']
        self.val_step = opt['train']['val_step']
        self.training_loss = 0.0
        self.val_loss = 0.0
        self.best_prec = 0.0
        self.skip_threshold = opt['train']['skip_threshold']
        self.last_epoch_loss = 1e8    # for skip threshold
        self.use_curriculum = False

        # test
        if not self.is_train:
            self.model_pth = opt['train']['resume_path']

    def name(self):
        return 'BaseSolver'

    def feed_data(self, batch):
        pass

    def summary(self, input_size):
        """print network summary"""
        pass

    def train_step(self):
        pass

    def validate(self, val_crop, crop_size):
        pass

    def test(self, use_chop):
        pass

    def _exact_crop_forward(self, upscale, crop_size):
        pass

    def _overlap_crop_forward(self, upscale):
        pass

    def save(self, epoch, is_best):
        pass

    def load(self):
        pass

    def current_loss(self):
        pass

    def current_visual(self):
        pass

    def current_learning_rate(self):
        pass

    def update_learning_rate(self, epoch):
        pass

    def tf_log(self, epoch):
        pass

    # TODO:
    # def save_network(self):
    # def load_network(self):
