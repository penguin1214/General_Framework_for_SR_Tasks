import os.path
import torch.utils.data as data
from data import common

class LRHRDataset(data.Dataset):
    '''
    Read LR and HR image pair.
    If only HR image is provided, generate LR image on-the-fly.
    The pair is ensured by 'sorted' function, so please check the name convention.
    '''

    def name(self):
        return 'LRHRDataset'

    def __init__(self, opt):
        super(LRHRDataset, self).__init__()
        self.opt = opt
        self.paths_LR = None
        self.paths_HR = None
        self.LR_env = None # environment for lmdb
        self.HR_env = None

        # read image list from subset list txt
        if opt['subset_file'] is not None and opt['phase'] == 'train':
            with open(opt['subset_file']) as f:
                self.paths_HR = sorted([os.path.join(opt['dataroot_HR'], line.rstrip('\n')) \
                        for line in f])
            if opt['dataroot_LR'] is not None:
                raise NotImplementedError('Now subset only supports generating LR on-the-fly.')
        # read image list from lmdb or image files
        self.HR_env, self.paths_HR = common.get_image_paths(opt['data_type'], opt['dataroot_HR'])
        self.LR_env, self.paths_LR = common.get_image_paths(opt['data_type'], opt['dataroot_LR'])

        assert self.paths_HR, 'Error: HR paths are empty.'
        if self.paths_LR and self.paths_HR:
            assert len(self.paths_LR) == len(self.paths_HR), \
                'HR and LR datasets have different number of images - {}, {}.'.format(\
                len(self.paths_LR), len(self.paths_HR))

        self.random_scale_list = None

    def __getitem__(self, index):
        HR_path, LR_path = None, None
        scale = self.opt['scale']
        HR_size = self.opt['HR_size']

        # get HR image
        HR_path = self.paths_HR[index]
        img_HR = common.read_img(self.HR_env, HR_path)

        # get LR image
        if self.paths_LR:
            LR_path = self.paths_LR[index]
            img_LR = common.read_img(self.LR_env, LR_path)
        else:
            raise NotImplementedError('Low resolution images do not exist')

        img_LR, img_HR = self._get_patch(img_LR, img_HR, scale, HR_size, self.opt['phase'])
        # channel conversion
        if self.opt['color']:
            img_LR, img_HR = common.channel_convert(img_HR.shape[2], [img_LR, img_HR], self.opt['color'])

        # HWC to CHW, BGR to RGB, numpy to tensor

        tensor_LR, tensor_HR = common.np2Tensor([img_LR, img_HR])

        if LR_path is None:
            LR_path = HR_path

        return {'LR': tensor_LR, 'HR': tensor_HR, 'LR_path': LR_path, 'HR_path': HR_path}

    def _get_patch(self, lr, hr, scale, patch_size, phase_str):
        if phase_str == 'train':
            lr, hr = common.get_patch(
                lr, hr, patch_size, scale)
            lr, hr = common.augment([lr, hr], self.opt['use_flip'], self.opt['use_rot'])
            lr = common.add_noise(lr, self.opt['noise'])
        else:
            hr = common.modcrop(hr, scale)

        return lr, hr

    def __len__(self):
        return len(self.paths_HR)
