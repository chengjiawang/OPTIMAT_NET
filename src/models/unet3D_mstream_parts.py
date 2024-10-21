# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.CBAM3D import ChannelGate, SpatialGate, SpatialGateFixScale

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch,
                 out_ch,
                 dilation_rate = [1, 1],
                 group=1,
                 is_dense = False,
                 kernel_size = 5):
        super(double_conv, self).__init__()
        self.is_dense = is_dense
        self.padding = ((kernel_size -1)*(np.array(dilation_rate)-1) + kernel_size)//2
        if is_dense:
            self.conv1 = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=self.padding[0], dilation=dilation_rate[0], groups=group),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            )
            self.conv2 = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size=kernel_size, padding=self.padding[1], dilation=dilation_rate[0], groups=group),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True),
            )
        else:

            self.conv = nn.Sequential(
                nn.Conv3d(in_ch, out_ch, kernel_size, padding=self.padding[0], dilation=dilation_rate[0], groups=group),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True),
                nn.Conv3d(out_ch, out_ch, kernel_size, padding=self.padding[1], dilation=dilation_rate[1], groups=group),
                nn.InstanceNorm3d(out_ch, affine=True),
                nn.ReLU(inplace=True)
            )

        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        if torch.cuda.is_available():
            self.conv.cuda()
        if self.is_dense:
            x = self.conv1(x) + x
            x = self.conv2(x) + x
        else:
            x = self.conv(x)
        return x

class single_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, dilation_rate = 1, group=1):
        super(single_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=dilation_rate, dilation=dilation_rate, groups=group),
            nn.InstanceNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )

        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        if torch.cuda.is_available():
            self.conv.cuda()
        x = self.conv(x)
        return x



class multi_input_double_conv(nn.Module):
    ''' multi input and single output '''
    def __init__(self,
                 in_ch,
                 out_ch,
                 split_group = 3,
                 dilation_rate = [1, 1]):
        super(multi_input_double_conv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=split_group)
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.split_size = int(self.out_channels/split_group)
        # check channel
        assert (self.in_channels > 1)

    def forward(self, x):
        x1 = self.conv(x)
        x1max, _ = torch.max(torch.stack(torch.split(x1,
                                                     split_size_or_sections= self.split_size,
                                                     dim=1)), dim=0)
        return x1, x1max

class multi_input_double_conv_rep(nn.Module):
    ''' multi input and single output '''
    def __init__(self,
                 in_ch,
                 out_ch,
                 split_group = 3,
                 dilation_rate = [1, 1]):
        super(multi_input_double_conv_rep, self).__init__()
        self.split_size = int(in_ch / split_group)
        self.conv = double_conv(self.split_size, out_ch, dilation_rate=dilation_rate, group=1)
        self.in_channels = in_ch
        self.out_channels = out_ch

        # check channel
        assert (self.in_channels > 1)

    def forward(self, x):
        xs = list(torch.split(x, split_size_or_sections=self.split_size, dim=1))
        for i, xx in enumerate(xs):
            xs[i] = self.conv(xx)
        x1 = torch.cat(xs, dim=1)
        x1max, _ = torch.max(torch.stack(xs), dim=0)
        return x1, x1max

class multi_input_down_conv_rep(nn.Module):
    ''' multi input and single output '''
    def __init__(self,
                 in_ch,
                 out_ch,
                 split_group = 3,
                 dilation_rate = [1, 1]):
        super(multi_input_down_conv_rep, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=1)
        )
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.split_size = int(in_ch / split_group)
        # check channel
        assert (self.in_channels > 1)

    def forward(self, x):
        xs = list(torch.split(x, split_size_or_sections=self.in_channels, dim=1))
        for i, xx in enumerate(xs):
            xs[i] = self.mpconv(xx)
        x1 = torch.cat(xs, dim=1)
        x1max, _ = torch.max(torch.stack(xs), dim=0)
        return x1, x1max


class multi_input_down_conv(nn.Module):
    ''' multi input and single output '''
    def __init__(self,
                 in_ch,
                 out_ch,
                 split_group = 3,
                 dilation_rate = [1, 1]):
        super(multi_input_down_conv, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            multi_input_double_conv(in_ch, out_ch,
                                    dilation_rate=dilation_rate,
                                    split_group=split_group)
        )
        self.in_channels = in_ch
        self.out_channels = out_ch
        # check channel
        assert (self.in_channels > 1)

    def forward(self, x):
        x1, x1max = self.mpconv(x)
        # print(type(x1))
        # x1max = torch.max(torch.stack(torch.split(x1, 1)), dim=0)
        return x1, x1max


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate = [1, 1], group=1):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=group)
        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        if torch.cuda.is_available() and isinstance(x.data, torch.cuda.FloatTensor):
            self.conv.cuda()
        x = self.conv(x)
        return x

class c1d(nn.Module):
    def __init__(self, in_ch, out_ch, fix_size = 96):
        super(c1d, self).__init__()
        # if not isinstance(fix_sizes, list):
        #     fix_sizes = [fix_sizes]*3
        self.fix_size = fix_size
        self.convs = nn.Sequential(
            nn.InstanceNorm1d(in_ch),
            nn.Softmax(dim=2),
            nn.Conv1d(in_ch, in_ch//2, kernel_size=9, padding=4),
            nn.LeakyReLU(inplace=True),
            nn.Conv1d(in_ch//2, out_ch, kernel_size=5, padding=2),
            # nn.InstanceNorm1d(out_ch)
        )
        self.grid = torch.linspace(-1, 1, self.fix_size)

        # self.out = nn.Softmax(dim=2)

    def forward(self, x):
        if torch.cuda.is_available() and isinstance(x.data, torch.cuda.FloatTensor):
                self.mpconv.to(x.device)

        # sum to x, y, z
        self.grid = self.grid.to(x.device)
        x_ysum = F.softmax(
            self.convs(F.interpolate(torch.mean(x, dim=-3), size=self.fix_size)),
            dim=-1
        )*self.grid
        x_xsum = F.softmax(
            self.convs(F.interpolate(torch.mean(x, dim=-2), size=self.fix_size)),
            dim=-1
        )*self.grid
        x_zsum = F.softmax(
            self.convs(F.interpolate(torch.mean(x, dim=-1), size=self.fix_size)),
            dim=-1
        )*self.grid

        return x_ysum, x_xsum, x_zsum

class c1d_attention(nn.Module):
    def __init__(self, in_ch, out_ch=1, fix_size = 96):
        super(c1d_attention, self).__init__()
        # if not isinstance(fix_sizes, list):
        #     fix_sizes = [fix_sizes]*3
        self.fix_size = fix_size

        self.conv_base = nn.Sequential(
            nn.InstanceNorm3d(in_ch),
            nn.Conv3d(in_ch, 1, kernel_size=9, padding=4, bias=False),
            nn.ReLU(inplace=True)
        )

        # self.at_kconv = nn.Sequential(
        #     nn.Conv3d(2, 1, kernel_size=5, padding=2),
        #     nn.AdaptiveAvgPool3d(self.fix_size)
        # )
        #
        # self.at_qconv = nn.Sequential(
        #     nn.Conv3d(2, 1, kernel_size=5, padding=2),
        #     nn.AdaptiveAvgPool3d(self.fix_size)
        # )
        #
        # self.at_vconv = nn.Sequential(
        #     nn.Conv3d(2, 1, kernel_size=5, padding=2),
        #     nn.AdaptiveAvgPool3d(self.fix_size)
        # )

        self.convs = nn.Sequential(
            # nn.InstanceNorm1d(in_ch),
            # nn.Softmax(dim=2),
            nn.Conv1d(1, in_ch//2, kernel_size=9, padding=4, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_ch//2, out_ch, kernel_size=5, padding=2, bias=False)
            # nn.InstanceNorm1d(out_ch)
        )
        # self.grid = torch.linspace(-1, 1, self.fix_size)

        # self.out = nn.Softmax(dim=2)

    def forward(self, x):
        if torch.cuda.is_available() and isinstance(x.data, torch.cuda.FloatTensor):
            self.conv_base.to(x.device)
            # self.at_kconv.to(x.device)
            # self.at_qconv.to(x.device)
            # self.at_vconv.to(x.device)
            self.convs.to(x.device)

        # sum to x, y, z
        # self.grid = self.grid.to(x.device)
        x = self.conv_base(x)
        # k = self.at_kconv(x)
        # k = k.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
        # q = self.at_qconv(x)
        # q = q.view(x.shape[0], x.shape[1], -1)
        # v = self.at_vconv(x)
        # v = v.view(x.shape[0], x.shape[1], -1).transpose(1, 2)
        #
        # x = torch.bmm(F.softmax(torch.bmm(k, q), dim=-1), v)
        # x = x.transpose(1, 2).view(x.shape[0], -1, x.shape[2], x.shape[3], x.shape[4])
        x_ysum = F.softmax(
            self.convs(F.interpolate(torch.mean(x, dim=(-1, -2)), size=self.fix_size)),
            dim=-1
        )
        x_xsum = F.softmax(
            self.convs(F.interpolate(torch.mean(x, dim=(-1, -3)), size=self.fix_size)),
            dim=-1
        )
        x_zsum = F.softmax(
            self.convs(F.interpolate(torch.mean(x, dim=(-2, -3)), size=self.fix_size)),
            dim=-1
        )

        # x_ysum = self.convs(F.interpolate(torch.mean(x, dim=(-1, -2)), size=self.fix_size))
        # x_xsum = self.convs(F.interpolate(torch.mean(x, dim=(-1, -3)), size=self.fix_size))
        # x_zsum = self.convs(F.interpolate(torch.mean(x, dim=(-2, -3)), size=self.fix_size))


        return x_ysum, x_xsum, x_zsum

class c1d_m(nn.Module):
    def __init__(self, in_ch, in_ch2=1, out_ch=1, fix_size=48):
        super(c1d_m, self).__init__()
        self.fix_size = fix_size
        self.conv1 = c1d_attention(in_ch, out_ch=out_ch, fix_size=fix_size)
        self.convs = nn.Sequential(
            nn.Conv1d(out_ch+in_ch2, out_ch, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.InstanceNorm1d(out_ch),
            nn.Softmax(dim=-1)
        )

        # self.out = nn.Softmax(dim=2)

    def forward(self, x1, x2y, x2x, x2z):
        if torch.cuda.is_available() and isinstance(x1.data, torch.cuda.FloatTensor):
            self.conv1.to(x1.device)
            self.convs.to(x1.device)

        #
        x1y, x1x, x1z = self.conv1(x1)

        xy = self.convs(
            torch.cat([x1y, F.adaptive_avg_pool1d(x2y, output_size=self.fix_size)], dim=1))
        xx = self.convs(
            torch.cat([x1x, F.adaptive_avg_pool1d(x2x, output_size=self.fix_size)], dim=1))
        xz = self.convs(
            torch.cat([x1z, F.adaptive_avg_pool1d(x2z, output_size=self.fix_size)], dim=1))

        return xy, xx, xz
    
class c1d_final(nn.Module):
    def __init__(self, fix_size=16):
        super(c1d_final, self).__init__()
        self.fix_size = fix_size
        # self.mlp = nn.Sequential(
        #     nn.Linear(in_units*out_units, in_units//2, bias=True),
        #     nn.LeakyReLU(inplace=True),
        #     nn.Linear(in_units//2, out_units, bias=False)
        # )
        self.grid = torch.linspace(-1, 1, self.fix_size).view(1, -1)

    def forward(self, xy, xx, xz):
        # out = self.mlp(
        #     torch.cat([
        #         xy.view(xy.shape[0], -1),
        #         xx.view(xx.shape[0], -1),
        #         xz.view(xz.shape[0], -1)
        #     ], dim=-1)
        # )
        self.grid = self.grid.to(xy.device)
        xy = torch.sum(xy.view(xy.shape[0], -1) * self.grid, dim=-1)
        xx = torch.sum(xx.view(xy.shape[0], -1) * self.grid, dim=-1)
        xz = torch.sum(xz.view(xy.shape[0], -1) * self.grid, dim=-1)

        return xy, xx, xz

class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate = [1, 1], group=1, pool=nn.MaxPool3d(2)):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            pool,
            double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=group)
        )
        self.in_channels = in_ch
        self.out_channels = out_ch
    def forward(self, x):
        if torch.cuda.is_available() and isinstance(x.data, torch.cuda.FloatTensor):
                self.mpconv.to(x.device)
        x = self.mpconv(x)
        return x

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class down_CBAM(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate = [1, 1], group=1):
        super(down_CBAM, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=group)
        )

        self.c_block = ChannelGate(
            gate_channels=out_ch,
            reduction_ratio=2
        )

        self.s_block = SpatialGate()

        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        if torch.cuda.is_available() and isinstance(x.data, torch.cuda.FloatTensor):
            self.mpconv.to(x.device)
        x = self.mpconv(x)
        x = self.c_block(x)
        x = self.s_block(x)
        return x

class down_CBAMFixScale(nn.Module):
    def __init__(self, in_ch, out_ch,
                 dilation_rate = [1, 1], group=1,
                 fixSize = 32, pool=nn.MaxPool3d(2)):
        super(down_CBAMFixScale, self).__init__()
        self.mpconv = nn.Sequential(
            pool,
            double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=group)
        )

        self.c_block = ChannelGate(
            gate_channels=out_ch,
            reduction_ratio=2
        )

        self.s_block = SpatialGateFixScale(fixSize=fixSize)
        self.fixSize = fixSize

        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        if torch.cuda.is_available() and isinstance(x.data, torch.cuda.FloatTensor):
            self.mpconv.to(x.device)
        x = self.mpconv(x)
        x = self.c_block(x)
        x, scale = self.s_block.forward_scale(x)
        return x, scale

class down_merge_rep(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate = [1, 1], group=1):
        super(down_merge_rep, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=group)
        )
        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        xs = list(torch.split(x, split_size_or_sections=self.in_channels, dim=1))
        for i, xx in enumerate(xs):
            xs[i] = self.mpconv(xx)
        x1max, _ = torch.max(torch.stack(xs), dim=0)
        return x1max

class up_CBAM(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True, dilation_rate = [1, 1], group=1):
        super(up_CBAM, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        self.c_block = ChannelGate(
            gate_channels=in_ch,
            reduction_ratio=2
        )

        self.s_block = SpatialGate()

        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2,
                                         groups=group)

        self.conv = double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=group)
        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x1, *x2s):
        if torch.cuda.is_available() and isinstance(x1.data, torch.cuda.FloatTensor):
            self.conv.cuda()
            self.up.cuda()
        x1 = self.up(x1)
        for x2 in x2s:
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                            diffY // 2, int(diffY / 2)))
            x1 = torch.cat([x2, x1], dim=1)
        x1 = self.c_block(x1)
        x1 = self.s_block(x1)
        x = self.conv(x1)
        return x

class up(nn.Module):
    def __init__(self, in_ch, out_ch, trilinear=True, dilation_rate = [1, 1], group=1):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if trilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=False)
        else:
            self.up = nn.ConvTranspose3d(in_ch//2, in_ch//2, 2, stride=2,
                                         groups=group)

        self.conv = double_conv(in_ch, out_ch, dilation_rate=dilation_rate, group=group)
        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x1, *x2s):
        if torch.cuda.is_available() and isinstance(x1.data, torch.cuda.FloatTensor):
            self.conv.cuda()
            self.up.cuda()
        x1 = self.up(x1)
        for x2 in x2s:
            diffX = x1.size()[2] - x2.size()[2]
            diffY = x1.size()[3] - x2.size()[3]
            x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                            diffY // 2, int(diffY / 2)))
            x1 = torch.cat([x2, x1], dim=1)
        x = self.conv(x1)
        return x

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.in_channels = in_ch
        self.out_channels = out_ch
    def forward(self, x):
        if torch.cuda.is_available() and isinstance(x.data, torch.cuda.FloatTensor):
            self.conv.cuda()
        x = self.conv(x)
        return x