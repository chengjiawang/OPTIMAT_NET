import torch
import math
import torch.nn as nn
import torch.nn.functional as F

class BasicConv(nn.Module):
    def __init__(self, in_planes,
                 out_planes,
                 kernel_size,
                 stride=1,
                 padding=0,
                 dilation=1,
                 groups=1,
                 relu=True,
                 bn=True,
                 bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv3d(in_planes,
                              out_planes,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              groups=groups,
                              bias=bias)
        self.bn = nn.InstanceNorm3d(out_planes, affine=True) if bn else None
        self.relu = nn.LeakyReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=2):
        super(ChannelGate, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )

        self.fusion = nn.Linear(gate_channels*2, gate_channels)

    def forward(self, x):
        ca_x = torch.mean(x.view(x.shape[0], x.shape[1], -1), dim=-1)  # mean pool
        cm_x = torch.max(x.view(x.shape[0], x.shape[1], -1), dim=-1)[0]  # max pool
        ca_x = self.mlp(ca_x)
        cm_x = self.mlp(cm_x)
        c_x = F.sigmoid(self.fusion(torch.cat([ca_x, cm_x], dim=1)))
        c_x = c_x.view(x.shape[0], x.shape[1], 1, 1, 1)
        return x * c_x

class ChannelGate_bk(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=2,
                 pool_types=['avg', 'max']):
        super(ChannelGate_bk, self).__init__()
        self.gate_channels = gate_channels
        self.mlp = nn.Sequential(
            Flatten(),
            nn.Linear(gate_channels, gate_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(gate_channels // reduction_ratio, gate_channels)
            )
        self.pool_types = pool_types
    def forward(self, x):
        channel_att_sum = None
        for pool_type in self.pool_types:
            if pool_type=='avg':
                avg_pool = F.avg_pool3d( x,
                                         (x.size(2), x.size(3), x.size(4)),
                                         stride=(x.size(2), x.size(3), x.size(4)))
                channel_att_raw = self.mlp( avg_pool )
            elif pool_type=='max':
                max_pool = F.max_pool3d( x,
                                         (x.size(2), x.size(3), x.size(4)),
                                         stride=(x.size(2), x.size(3), x.size(4))
                                         )
                channel_att_raw = self.mlp( max_pool )

            elif pool_type=='lse':
                # LSE pool only
                lse_pool = logsumexp_d(x)
                channel_att_raw = self.mlp( lse_pool )

            if channel_att_sum is None:
                channel_att_sum = channel_att_raw
            else:
                channel_att_sum = channel_att_sum + channel_att_raw

        scale = F.sigmoid( channel_att_sum ).unsqueeze(2).unsqueeze(3).expand_as(x)
        return x * scale

def logsumexp_d(tensor):
    tensor_flatten = tensor.view(tensor.size(0), tensor.size(1), -1)
    s, _ = torch.max(tensor_flatten, dim=2, keepdim=True)
    outputs = s + (tensor_flatten - s).exp().sum(dim=2, keepdim=True).log()
    return outputs

class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat(
            (
                torch.max(x,dim=1)[0].unsqueeze(1),
                torch.mean(x,dim=1).unsqueeze(1)
            ),
            dim=1
        )

class SpatialGate(nn.Module):
    def __init__(self):
        super(SpatialGate, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale

    def forward_scale(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out) # broadcasting
        return x * scale, scale

    def get_scale(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return scale


class CBAM(nn.Module):
    def __init__(self, gate_channels, reduction_ratio=16, pool_types=['avg', 'max'], no_spatial=False):
        super(CBAM, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        if not no_spatial:
            self.SpatialGate = SpatialGate()
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

    def forward_scale(self, x):
        x_out = self.ChannelGate(x)
        x_out, scale = self.SpatialGate.foward_scale(x_out)
        return x_out, scale

class SpatialGateFixScale(nn.Module):
    def __init__(self, fixSize = 32):
        super(SpatialGateFixScale, self).__init__()
        kernel_size = 7
        self.fixSize = fixSize
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        return x * scale

    def forward_scale(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        # scale_inter = F.interpolate(scale, size=self.fixSize)
        if self.fixSize is not None:
            scale_inter = F.adaptive_avg_pool3d(scale, output_size=self.fixSize)
            return x * scale, scale_inter
        else:
            return x * scale, scale

    def get_scale(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        scale = F.sigmoid(x_out)  # broadcasting
        # scale_inter = F.interpolate(scale, size=self.fixSize)
        if self.fixSize is not None:
            scale_inter = F.adaptive_avg_pool3d(scale, output_size=self.fixSize)
            return scale_inter
        else:
            return scale

class CBAMScaleFix(nn.Module):
    def __init__(self,
                 gate_channels,
                 reduction_ratio=16,
                 pool_types=['avg', 'max'],
                 no_spatial=False,
                 fixSize = 32):
        super(CBAMScaleFix, self).__init__()
        self.ChannelGate = ChannelGate(gate_channels, reduction_ratio, pool_types)
        self.no_spatial=no_spatial
        self.fixSize = fixSize
        if not no_spatial:
            self.SpatialGate = SpatialGateFixScale(fixSize=fixSize)
    def forward(self, x):
        x_out = self.ChannelGate(x)
        if not self.no_spatial:
            x_out = self.SpatialGate(x_out)
        return x_out

    def forward_scale(self, x):
        x_out = self.ChannelGate(x)
        x_out, scale = self.SpatialGate.foward_scale(x_out)
        return x_out, scale