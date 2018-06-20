import os
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from utils.torchsummary import summary as tc_summary

from .networks import create_model
from .base_solver import BaseSolver


class SRModel(BaseSolver):
    def __init__(self, opt):
        super(SRModel, self).__init__(opt)
        self.train_opt = opt['train']

        self.LR = self.Tensor()
        self.HR = self.Tensor()
        self.var_LR = None
        self.var_HR = None

        self.results = {'training_loss': [],
                        'val_loss': [],
                        'psnr': [],
                        'ssim': []}


        if opt['mode'] == 'sr':
            self.model = create_model(opt)
        else:
            assert 'Invalid opt.mode [%s] for SRModel class!'
        self.load()

        if self.is_train:
            self.model.train()

            loss_type = self.train_opt['pixel_criterion']
            if loss_type == 'l1':
                self.criterion_pix = nn.L1Loss()
            elif loss_type == 'l2':
                self.criterion_pix = nn.MSELoss()
            else:
                raise NotImplementedError('[ERROR] Loss type [%s] is not implemented!'%loss_type)

            if self.use_gpu:
                self.criterion_pix = self.criterion_pix.cuda()
            self.criterion_pix_weight = self.train_opt['pixel_weight']

            weight_decay = self.train_opt['weight_decay_G'] if self.train_opt['weight_decay_G'] else 0
            optim_params = []
            for k, v in self.model.named_parameters():
                if v.requires_grad:
                    optim_params.append(v)
                else:
                    print('[WARNING] Parameters [%s] will not be optimized!'%k)
                self.optimizer = optim.Adam(optim_params, lr=self.train_opt['lr_G'], weight_decay=weight_decay)

            if self.train_opt['lr_scheme'].lower() == 'multisteplr':
                self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, self.train_opt['lr_steps'], self.train_opt['lr_gamma'])
            else:
                raise NotImplementedError('[ERROR] Only MultiStepLR scheme is supported!')

            self.log_dict = OrderedDict()
            print('[Model Initialized]')

    def name(self):
        return 'SRModel'

    def feed_data(self, batch):
        input, target = batch['LR'], batch['HR']
        self.LR.resize_(input.size()).copy_(input)
        self.var_LR = Variable(self.LR)
        self.HR.resize_(target.size()).copy_(target)
        self.var_HR = Variable(self.HR)

    def summary(self, input_size):
        print('========================= Model Summary ========================')
        print(self.model)
        print('================================================================')
        print('Input Size: %s'%str(input_size))
        tc_summary(self.model, input_size)
        print('================================================================')

    def train_step(self):
        self.optimizer.zero_grad()
        self.SR = self.model(self.var_LR)
        loss_pix = self.criterion_pix_weight*self.criterion_pix(self.SR, self.var_HR)
        loss_pix.backward()
        self.optimizer.step()

        loss_step = loss_pix

        return loss_step

    def test(self):
        self.model.eval()
        self.SR = self.model(self.var_LR)
        loss_pix = self.criterion_pix_weight*self.criterion_pix(self.SR, self.var_HR)
        self.model.train()
        return loss_pix

    def save(self, epoch, is_best):
        filename = os.path.join(self.checkpoint_dir,'checkpoint.pth')
        print('[Saving checkpoint to %s ...]'%filename)
        state = {
            'epoch': epoch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'best_prec': self.best_prec,
            'results': self.results
        }
        torch.save(state, filename)
        if is_best:
            torch.save(state, os.path.join(self.checkpoint_dir, 'best_checkpoint.pth'))
        print(['=> Done.'])

    def load(self):
        pass

    def current_loss(self):
        pass

    def get_current_visual(self, need_HR=True):
        out_dict = OrderedDict()
        out_dict['LR'] = self.var_LR.data[0].float().cpu()
        out_dict['SR'] = self.SR.data[0].float().cpu()
        if need_HR:
            out_dict['HR'] = self.var_HR.data[0].float().cpu()
        return out_dict

    def current_learning_rate(self):
        return self.optimizer.param_groups[0]['lr']

    def update_learning_rate(self):
        self.scheduler.step()

    def tf_log(self, epoch):
        print('[Logging...]')
        info = {
            'training_loss': self.training_loss,
            'val_loss': self.val_loss
        }
        for key, value in info.items():
            self.tf_logger.scalar_summary(key, value, epoch)

        # 2. params (histogram summary)
        for key, value in self.model.named_parameters():
            key = key.replace('.', '/')
            self.tf_logger.histo_summary(key, value.data.cpu().numpy(), epoch)
            self.tf_logger.histo_summary(key + '/grad', value.grad.data.cpu().numpy(), epoch)
        print('=> Done.')

    # def save_network(self):
    # def load_network(self):
