import math
import torch
import torch.nn as nn

# TODO: carefully use *
import models.modules.blocks as B

class SRResNet(nn.Module):
    # TODO: upscale_factor=3
    def __init__(self,in_channels, out_channels, num_features, num_blocks, upscale_factor=4, norm_type='bn', act_type='relu', mode='NAC', upsample_mode='upconv'):
        super(SRResNet, self).__init__()

        feature_extract = B.ConvBlock(in_channels, num_features, kernel_size=9, norm_type=None, act_type='prelu')
        res_blocks = [B.ResBlock(num_features, num_features, num_features, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode) for _ in range(num_blocks)]
        conv_lr = B.ConvBlock(num_features, num_features, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode)

        if upsample_mode == 'upconv':
            upsample_block = B.UpsampleConvBlock(upscale_factor=upscale_factor, in_channels=num_features, out_channels=num_features,\
                                               kernel_size=3, stride=1)
        elif upsample_mode == 'pixelshuffle':
            upsample_block = B.PixelShuffleBlock()
        else:
            raise NotImplementedError('Upsample mode [%s] is not supported!'%upsample_mode)

        conv_hr = B.ConvBlock(num_features, out_channels, kernel_size=9, norm_type=None, act_type=None)

        # TODO: dense connection
        # TODO: Notice: We must unpack the residual blocks using ‘*’ before building a nn.Sequential
        self.network = B.sequential(feature_extract, B.ShortcutBlock(B.sequential(*res_blocks, conv_lr)), upsample_block, conv_hr)

    def forward(self, x):
        # TODO: if batch size is 1, should unsqueeze(0), otherwise unexpected stride size error!
        return self.network(x)

class DBPN(nn.Module):
    def __init__(self,in_channels, out_channels, num_features, bp_stages, upscale_factor=4, norm_type=None, act_type='prelu', mode='NAC', upsample_mode='upconv'):
        super(DBPN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            projection_filter = 12

        feature_extract_1 = B.ConvBlock(in_channels, 128, kernel_size=3, norm_type=norm_type, act_type=act_type)
        feature_extract_2 = B.ConvBlock(128, num_features, kernel_size=1, norm_type=norm_type, act_type=act_type)

        bp_units = []
        for _ in range(bp_stages-1):
            bp_units.extend([B.UpprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                                padding=padding, norm_type=norm_type, act_type=act_type),
                            B.DownprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                                  padding=padding, norm_type=norm_type, act_type=act_type)])

        last_bp_unit = B.UpprojBlock(num_features, num_features, projection_filter, stride=stride, valid_padding=False,
                                           padding=padding, norm_type=norm_type, act_type=act_type)
        conv_hr = B.ConvBlock(num_features, out_channels, kernel_size=1, norm_type=None, act_type=None)

        self.network = B.sequential(feature_extract_1, feature_extract_2, *bp_units, last_bp_unit, conv_hr)

    def forward(self, x):
        return self.network(x)


class D_DBPN(nn.Module):
    def __init__(self,in_channels, out_channels, num_features, bp_stages, upscale_factor=4, norm_type=None, act_type='prelu', mode='NAC', upsample_mode='upconv'):
        super(D_DBPN, self).__init__()

        if upscale_factor == 2:
            stride = 2
            padding = 2
            projection_filter = 6
        elif upscale_factor == 4:
            stride = 4
            padding = 2
            projection_filter = 8
        elif upscale_factor == 8:
            stride = 8
            padding = 2
            projection_filter = 12

        feature_extract_1 = B.ConvBlock(in_channels, 256, kernel_size=3, norm_type=norm_type, act_type=act_type)
        feature_extract_2 = B.ConvBlock(256, num_features, kernel_size=1, norm_type=norm_type, act_type=act_type)

        bp_units = B.DensebackprojBlock(num_features, num_features, projection_filter, bp_stages, stride=stride, valid_padding=False,
                                                padding=padding, norm_type=norm_type, act_type=act_type)

        conv_hr = B.ConvBlock(num_features*bp_stages, out_channels, kernel_size=3, norm_type=None, act_type=None)

        self.network = B.sequential(feature_extract_1, feature_extract_2, bp_units, conv_hr)

    def forward(self, x):
        # TODO: if batch size is 1, should unsqueeze(0), otherwise unexpected stride size error!
        return self.network(x)


class ConvTest(nn.Module):
    def __init__(self,in_channels, out_channels, num_features, num_blocks, upscale_factor=4, norm_type='bn', act_type='relu', mode='NAC', upsample_mode='upconv'):
        super(ConvTest, self).__init__()
        res_blocks = [B.ResBlock(num_features, num_features, num_features, kernel_size=3, norm_type=norm_type, act_type=act_type, mode=mode) for _ in range(2)]
        upsample = B.UpsampleConvBlock(upscale_factor=4, in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1, norm_type=None, act_type=None, mode=mode)
        self.network = B.sequential(res_blocks, upsample)

    def forward(self, x):
        return self.network(x)

