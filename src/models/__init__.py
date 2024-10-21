from .resnet import resnet8, resnet14, resnet20, resnet32, resnet44, resnet56, resnet110, resnet8x4, resnet32x4
from .resnetv2 import ResNet50
from .wrn import wrn_16_1, wrn_16_2, wrn_40_1, wrn_40_2
from .vgg import vgg19_bn, vgg16_bn, vgg13_bn, vgg11_bn, vgg8_bn
from .mobilenetv2 import mobile_half
from .ShuffleNetv1 import ShuffleV1
from .ShuffleNetv2 import ShuffleV2
from .unets import UNet3D, UNet3D_SpatialAttention,\
    UNet3D_CBAMAttention, UNet3D_AttentionZoom, UNet3D_CBAMAttentionZoom, \
    UNet3D_DSup, UNet3D_dsupwrapper, UNet3D_SA_dsupwrapper


model_dict = {
    'unet3D': UNet3D,
    'unet3D_dsup': UNet3D_dsupwrapper,
    'unet3D_attention': UNet3D_SpatialAttention,
    'unet3D_attention_dsup': UNet3D_SA_dsupwrapper,
    'unet3D_attentionzoom': UNet3D_AttentionZoom,
    'unet3D_CBAMattentionzoom': UNet3D_CBAMAttentionZoom,
    'unet3D_CBAMattention': UNet3D_CBAMAttention,
    'resnet8': resnet8,
    'resnet14': resnet14,
    'resnet20': resnet20,
    'resnet32': resnet32,
    'resnet44': resnet44,
    'resnet56': resnet56,
    'resnet110': resnet110,
    'resnet8x4': resnet8x4,
    'resnet32x4': resnet32x4,
    'ResNet50': ResNet50,
    'wrn_16_1': wrn_16_1,
    'wrn_16_2': wrn_16_2,
    'wrn_40_1': wrn_40_1,
    'wrn_40_2': wrn_40_2,
    'vgg8': vgg8_bn,
    'vgg11': vgg11_bn,
    'vgg13': vgg13_bn,
    'vgg16': vgg16_bn,
    'vgg19': vgg19_bn,
    'MobileNetV2': mobile_half,
    'ShuffleV1': ShuffleV1,
    'ShuffleV2': ShuffleV2,
}
