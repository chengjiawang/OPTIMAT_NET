import aug_transforms.transform3D_torch as transform3D
import aug_transforms.uniform as unitrans
from aug_transforms.transform3D_torch import ToTensor
import math
from aug_transforms.transformation_base import Compose
from torchvision.transforms.transforms import ToTensor as TV_ToTensor

def get_basic_transform3D(normalize=False):
    transform_list = []

    transform_list += [
        ToTensor()
    ]

    if normalize:

        transform_list += [
            unitrans.Normalize_Range(
                min_value=-1,
                max_value=1
            )
        ]

    return Compose(transform_list)

def get_norm_trans3D():
    transform_list = []
    transform_list += [
            unitrans.Normalize_Range(
                min_value=-1,
                max_value=1
            )
        ]

    return Compose(transform_list)

def get_basic_transform3D_nonorm():
    transform_list = []

    transform_list += [
        ToTensor()
    ]

    return Compose(transform_list)

def get_norm_transform3D():
    transform_list = []

    transform_list +=[
        unitrans.Normalize_Range(
            min_value=-1,
            max_value=1
        )
    ]

    return Compose(transform_list)

def get_transform_3D(opt, normalize = False):
    transform_list = []

    if opt.augment_option.lower() in ['affine', 'b', 'thinplate', 'free']:
        transform_list += [
            transform3D.RandomAffine(degrees=10 / 180 * math.pi,
                                     translate=0.3,
                                     scale=1.1,
                                     separate_transform = False,
                                     freq=0.5
                                     )
        ]

    transform_list += [
        ToTensor()
    ]

    if normalize:

        transform_list += [
            unitrans.Normalize_Range(
                min_value=-1,
                max_value=1
            )
        ]

    # if opt.resize_or_crop == 'crop':
    #     transform_list.append(transform3D.RandomCrop3D(opt.fineSize))

    return Compose(transform_list)


def get_transform_3D_test():
    transform_list = []

    transform_list += [
        ToTensor()
    ]

    transform_list += [
        unitrans.Normalize_Range(
            min_value=-1,
            max_value=1
        )
    ]

    # if opt.resize_or_crop == 'crop':
    #     transform_list.append(transform3D.RandomCrop3D(opt.fineSize))

    return Compose(transform_list)

def get_transform_3D_nonorm(opt):
    transform_list = []

    if opt.augment_option.lower() in ['affine', 'b', 'thinplate', 'free']:
        transform_list += [
            transform3D.RandomAffine(degrees=10 / 180 * math.pi,
                                     translate=0.1,
                                     separate_transform = False,
                                     freq=0.9
                                     )
        ]

    transform_list += [
        ToTensor()
    ]
    # if opt.resize_or_crop == 'crop':
    #     transform_list.append(transform3D.RandomCrop3D(opt.fineSize))

    return Compose(transform_list)

def get_basic_transform2D():
    transform_list = []

    transform_list += [
        TV_ToTensor()
    ]

    transform_list += [
        unitrans.Normalize_Range(
            min_value=-1,
            max_value=1
        )
    ]

    return Compose(transform_list)

def get_basic_transform2D_nonorm():
    transform_list = []

    transform_list += [
        tv_wrapper(
            TV_ToTensor()
        )
    ]

    return Compose(transform_list)



def get_transform_2D(opt):
    transform_list = []

    if opt.augment_option == 'affine':
        transform_list.append(
                my_RandomAffine(degrees=20,
                                        translate=(0.2,0.2),
                                        scale=[0.9, 1.1],
                                        shear=1.1,
                                        resample=Image.BICUBIC,
                                        fillcolor=None)
        )

    transform_list.append(
        tv_wrapper(
            TV_ToTensor()
        )
    )

    transform_list += [
        unitrans.Normalize_Range(
            min_value=-1,
            max_value=1
        )
    ]

    return Compose(transform_list)

def get_transform_2D_nonorm(opt):
    transform_list = []

    if opt.augment_option == 'affine':
        transform_list.append(
                my_RandomAffine(degrees=20,
                                        translate=(0.1,0.1),
                                        scale=[0.9, 1.1],
                                        shear=1.1,
                                        resample=Image.BICUBIC,
                                        fillcolor=None)
        )

    transform_list.append(
        tv_wrapper(
            TV_ToTensor()
        )
    )

    return Compose(transform_list)

import numpy as np
from scipy import ndimage as ndi

from PIL import Image

from aug_transforms.transformation_base import \
    transformation_base_class, Compose

from torchvision.transforms import transforms

class tv_wrapper(transformation_base_class):
    """wrap the torchvision transformation classes to allow multiple inputs.

    Args:
        a torchvision transformation object
    """

    def __init__(self, tv_obj):
        super(tv_wrapper, self).__init__()
        self.tv_obj = tv_obj

    def __transforms__(self, *args, **kwargs):
        return self.tv_obj.__call__(*args, **kwargs)

    def get_params(self, *args, **kwargs):
        if hasattr(self.tv_obj, 'get_params'):
            self.tv_obj.get_params(*args, **kwargs)

    def __repr__(self):
        return self.tv_obj.__repr__()

def get_transform_torchvision(opt):
    # osize = [opt.loadSize, opt.loadSize]
    transform_list = []

    # transform_list.append(aug_transforms.Resize(osize))
    if opt.resize_or_crop == 'resize_and_crop':
        # transform_list.append(aug_transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(
            tv_wrapper(
                transforms.RandomCrop(opt.fineSize)
            )
        )

    elif opt.resize_or_crop == 'crop':
        transform_list.append(
            tv_wrapper(
                transforms.RandomCrop(opt.fineSize)
            )
        )

    elif opt.resize_or_crop == 'center_crop':
        transform_list.append(
            tv_wrapper(
                transforms.CenterCrop(opt.fineSize)
            )
        )

    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(
            tv_wrapper(
                transforms.Lambda(
                    lambda img: __scale_width(img, opt.fineSize)
                )
            )
        )

    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(
            tv_wrapper(
               transforms.RandomCrop(opt.fineSize)
            )
        )


    if opt.isTrain and not opt.no_flip:
        transform_list.append(
            tv_wrapper(
                transforms.RandomHorizontalFlip()
            )
        )

    if opt.augment_option == 'affine':
        transform_list.append(
            tv_wrapper(
                transforms.RandomAffine(degrees=20,
                                        translate=(0.1,0.1),
                                        scale=[0.9, 1.1],
                                        shear=1.1,
                                        resample=Image.BICUBIC)
            )
        )

    elif opt.augment_option == 'affine_noscale':
        transform_list.append(
            tv_wrapper(
                transforms.RandomAffine(degrees=90,
                                        translate=(0.3,0.3),
                                        resample=Image.BICUBIC)
            )
        )

    elif opt.augment_option == 'rotation':
        transform_list.append(
            tv_wrapper(
                transforms.RandomRotation(90, Image.BICUBIC)
            )
        )

    elif opt.augment_option == 'none':
        pass

    elif opt.augment_option == 'arbitrary_distort':
        transform_list.append(
            tv_wrapper(
                transforms.RandomAffine(degrees=5,
                                        translate=(0.1, 0.1),
                                        shear=1.1,
                                        resample=Image.BICUBIC)
            )
        )

    else:
        pass

    if not opt.channel_wise_normalization:
        transform_list += [
            tv_wrapper(transforms.ToTensor()),

            tv_wrapper(transforms.Normalize((0.5, 0.5, 0.5),
                                 (0.5, 0.5, 0.5)))
        ]
    else:
        transform_list += [
            tv_wrapper(transforms.ToTensor())
        ]

    # distortion happens after convert to tensor

    # transform_list += [aug_transforms.ToTensor()]
    return Compose(transform_list)

def __scale_width(img, target_width):
    ox, oy, oz = img.shape
    if (ox == target_width):
        return img
    w = target_width
    h = int(target_width * ox / oy)
    z = int(target_width * ox / oz)
    return ndi.zoom(img, np.array([w, h, z])/np.array([ox, oy, oz]))

# torvision random affine is rubbish!
import numbers
import random
import torchvision.transforms.functional as rf

class my_RandomAffine(object):
    """Random affine transformation of the image keeping center invariant

    Args:
        degrees (sequence or float or int): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Set to 0 to desactivate rotations.
        translate (tuple, optional): tuple of maximum absolute fraction for horizontal
            and vertical translations. For example translate=(a, b), then horizontal shift
            is randomly sampled in the range -img_width * a < dx < img_width * a and vertical shift is
            randomly sampled in the range -img_height * b < dy < img_height * b. Will not translate by default.
        scale (tuple, optional): scaling factor interval, e.g (a, b), then scale is
            randomly sampled from the range a <= scale <= b. Will keep original scale by default.
        shear (sequence or float or int, optional): Range of degrees to select from.
            If degrees is a number instead of sequence like (min, max), the range of degrees
            will be (-degrees, +degrees). Will not apply shear by default
        resample ({PIL.Image.NEAREST, PIL.Image.BILINEAR, PIL.Image.BICUBIC}, optional):
            An optional resampling filter.
            See http://pillow.readthedocs.io/en/3.4.x/handbook/concepts.html#filters
            If omitted, or if the image has mode "1" or "P", it is set to PIL.Image.NEAREST.
        fillcolor (int): Optional fill color for the area outside the transform in the output image. (Pillow>=5.0.0)
    """

    def __init__(self, degrees,
                 translate=None,
                 scale=None,
                 shear=None,
                 resample=False,
                 fillcolor=0, freq = 0.99):
        self.freq = freq
        if isinstance(degrees, numbers.Number):
            if degrees < 0:
                raise ValueError("If degrees is a single number, it must be positive.")
            self.degrees = (-degrees, degrees)
        else:
            assert isinstance(degrees, (tuple, list)) and len(degrees) == 2, \
                "degrees should be a list or tuple and it must be of length 2."
            self.degrees = degrees

        if translate is not None:
            assert isinstance(translate, (tuple, list)) and len(translate) == 2, \
                "translate should be a list or tuple and it must be of length 2."
            for t in translate:
                if not (0.0 <= t <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
        self.translate = translate

        if scale is not None:
            assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
                "scale should be a list or tuple and it must be of length 2."
            for s in scale:
                if s <= 0:
                    raise ValueError("scale values should be positive")
        self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                if shear < 0:
                    raise ValueError("If shear is a single number, it must be positive.")
                self.shear = (-shear, shear)
            else:
                assert isinstance(shear, (tuple, list)) and len(shear) == 2, \
                    "shear should be a list or tuple and it must be of length 2."
                self.shear = shear
        else:
            self.shear = shear

        self.resample = resample
        self.fillcolor = fillcolor

    @staticmethod
    def get_params(degrees, translate, scale_ranges, shears, img_size):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = random.uniform(degrees[0], degrees[1])

        if translate is not None:
            max_dx = translate[0] * img_size[0]
            max_dy = translate[1] * img_size[1]
            translations = (np.round(random.uniform(-max_dx, max_dx)),
                            np.round(random.uniform(-max_dy, max_dy)))
        else:
            translations = (0, 0)

        if scale_ranges is not None:
            scale = random.uniform(scale_ranges[0], scale_ranges[1])
        else:
            scale = 1.0

        if shears is not None:
            shear = random.uniform(shears[0], shears[1])
        else:
            shear = 0.0

        return angle, translations, scale, shear

    def __call__(self, *imgs):
        """
            img (PIL Image): Image to be transformed.

        Returns:
            PIL Image: Affine transformed image.
        """
        if self.freq < random.random():
            print('noaffine')
            return imgs[0]


        if len(imgs) == 1:
            img = imgs[0]
            try:
                self.fillcolor = img.min()
            except:
                # if PIL image doesn't have min
                self.fillcolor = None
            ret = self.get_params(self.degrees, self.translate, self.scale, self.shear, img.size)
            return rf.affine(img,
                             *ret,
                             resample=self.resample,
                             fillcolor=self.fillcolor)
        else:
            out_img = []
            ret = self.get_params(self.degrees,
                                  self.translate,
                                  self.scale,
                                  self.shear,
                                  imgs[0].size)
            for img in imgs:
                out_img.append(
                    rf.affine(img,
                              *ret,
                              resample=self.resample,
                              fillcolor=img.min())
                )
            return out_img