# sub-parts of the U-Net model

import torch
import torch.nn as nn
import torch.nn.functional as F


class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch, dilation_rate = [1, 1]):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_ch, out_ch, 3, padding=1, dilation=dilation_rate[0]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_ch, out_ch, 3, padding=1, dilation=dilation_rate[1]),
            nn.BatchNorm3d(out_ch),
            nn.ReLU(inplace=True)
        )
        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        x = self.conv(x)
        return x


class inconv(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate = [1, 1]):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch, dilation_rate=dilation_rate)

    def forward(self, x):
        x = self.conv(x)
        return x


class down(nn.Module):
    def __init__(self, in_ch, out_ch, dilation_rate = [1, 1]):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool3d(2),
            double_conv(in_ch, out_ch, dilation_rate=dilation_rate)
        )

        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True, dilation_rate = [1, 1]):
        super(up, self).__init__()

        #  would be a nice idea if the upsampling could be learned too,
        #  but my machine do not have enough memory to handle all those weights
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2, dilation=dilation_rate)

        self.conv = double_conv(in_ch, out_ch, dilation_rate=dilation_rate)
        self.in_channels = in_ch
        self.out_channels = out_ch


    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffX = x1.size()[2] - x2.size()[2]
        diffY = x1.size()[3] - x2.size()[3]
        x2 = F.pad(x2, (diffX // 2, int(diffX / 2),
                        diffY // 2, int(diffY / 2)))
        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x


class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv3d(in_ch, out_ch, 1)
        self.in_channels = in_ch
        self.out_channels = out_ch

    def forward(self, x):
        x = self.conv(x)
        return x