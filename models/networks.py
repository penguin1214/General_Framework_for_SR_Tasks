﻿import torch
import torch.nn as nn
from torch.nn import init
import functools

####################
# initialize
####################

def weights_init_normal(m, std=0.02):
    classname = m.__class__.__name__
    print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.normal_(m.weight.data, 0.0, std)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, std)
        init.constant_(m.bias.data, 0.0)

def weights_init_kaiming(m, scale=1):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        print('initializing [%s] ...' % classname)
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
        m.weight.data *= scale
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.constant_(m.weight.data, 1.0)
        m.weight.data *= scale
        init.constant_(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    print('initializing [%s] ...' % classname)
    if classname.find('Conv') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        init.orthogonal_(m.weight.data, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm2d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def init_weights(net, init_type='kaiming', scale=1, std=0.02):
    # scale for 'kaiming', std for 'normal'.
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        weights_init_normal_ = functools.partial(weights_init_normal, std=std)
        net.apply(weights_init_normal_)
    elif init_type == 'kaiming':
        weights_init_kaiming_ = functools.partial(weights_init_kaiming, scale=scale)
        net.apply(weights_init_kaiming_)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

####################
# define network
####################
def create_model(opt):
    if opt['mode'] == 'sr' or opt['mode'] == 'sr_curriculum':
        netG = define_G(opt['networks']['G'])
        return netG
    elif opt['mode'] == 'srgan':
        netG = define_G(opt['networks']['G'])
        netD = define_D(opt['networks']['D'])
        return netG, netD
    else:
        raise NotImplementedError("The mode [%s] of networks is not recognized." % opt['mode'])

# Generator
def define_G(opt):
    which_model = opt['which_model'].lower()
    if which_model == 'sr_resnet_torch':
        import models.modules.archs as Arch
        netG = Arch.SRResNet(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], \
            num_blocks=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=opt['norm_type'], mode=opt['mode'])
    elif which_model == 'sr_resnet':
        import models.modules.archs as Arch
        netG = Arch.SRResNet(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], \
            num_blocks=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=opt['norm_type'], mode=opt['mode'],\
            upsample_mode='upconv')
    elif which_model == 'dbpn':
        import models.modules.archs as Arch
        # TODO: Need residual or only process on Y channel
        netG = Arch.DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                             num_features=opt['num_features'], \
                             bp_stages=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=None,
                             mode=opt['mode'])
    elif which_model == 'd-dbpn':
        import models.modules.archs as Arch
        netG = Arch.D_DBPN(in_channels=opt['in_channels'], out_channels=opt['out_channels'],
                         num_features=opt['num_features'], \
                         bp_stages=opt['num_blocks'], upscale_factor=opt['scale'], norm_type=None,
                         mode=opt['mode'])
    elif which_model == 'drrn':
        import models.modules.drrn_arch as DRRNArch
        netG = DRRNArch.DRRN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_recurs=opt['num_recurs'])

    elif which_model.find('rcan') >= 0:
        import models.modules.rcan as RCANArch
        netG = RCANArch.RCAN(in_channels=opt['in_channels'], out_channels=opt['out_channels'], num_features=opt['num_features'], num_groups = opt['num_groups'], num_blocks = opt['num_blocks'], upscale_factor=opt['scale'])
    else:
        raise NotImplementedError("Network [%s] is not recognized." % which_model)

    if torch.cuda.is_available():
        netG = nn.DataParallel(netG).cuda()

    return netG

# Discriminator
def define_D(opt):
    # TODO: modify key value
    which_model = opt['which_model']

    if which_model == 'discriminaotr_vgg_128':
        import models.modules.archs as Arch
        netD = Arch.Discriminaotr_VGG_128(in_channels=opt['in_channels'], base_nf=opt['nf'], \
            norm_type=opt['norm_type'], mode=opt['mode'] ,act_type=opt['act_type'])
    elif which_model == 'discriminaotr_vgg_32':
        import models.modules.archs as Arch
        netD = Arch.Discriminaotr_VGG_32(in_channels=opt['in_channels'], base_nf=opt['nf'], \
            norm_type=opt['norm_type'], mode=opt['mode'] ,act_type=opt['act_type'])
    elif which_model == 'discriminaotr_vgg_32_y':
        import models.modules.archs as Arch
        netD = Arch.Discriminaotr_VGG_32_Y(in_channels=opt['in_channels'], base_nf=opt['nf'], \
            norm_type=opt['norm_type'], mode=opt['mode'] ,act_type=opt['act_type'])
    else:
        raise NotImplementedError('Discriminator model [%s] is not recognized' % which_model)

    # init_weights(netD, init_type='kaiming', scale=1)
    if torch.cuda.is_available():
        netD = nn.DataParallel(netD).cuda()
    return netD


def define_F(opt, use_bn=False):
    import models.modules.archs as Arch
    gpu_ids = opt['gpu_ids']
    tensor = torch.cuda.FloatTensor if gpu_ids else torch.FloatTensor
    # pytorch pretrained VGG19-54, before ReLU.
    if use_bn:
        feature_layer = 49
    else:
        feature_layer = 34
    netF = Arch.VGGFeatureExtractor(feature_layer=feature_layer, use_bn=use_bn, \
        use_input_norm=True, tensor=tensor)
    if gpu_ids:
        netF = nn.DataParallel(netF).cuda()
    netF.eval()  # No need to train
    return netF
