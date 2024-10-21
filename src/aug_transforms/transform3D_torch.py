from __future__ import division
import torch
import math
import random

import numpy as np
import numbers
import types

import scipy.ndimage as ndi

from torch.nn import functional as F

from aug_transforms.transformation_base import \
    transformation_base_class

from transforms3d import affines
from transforms3d.euler import euler2mat

__all__ = ["Compose", "ToTensor", "Lambda", "RandomAffine", "CenterCrop"]

class Compose(object):
    """Composes several transforms3D_back together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms3D_back to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs):
        for t in self.transforms:
            imgs = t(*imgs)
        return imgs

def strip_list(in_list):
    while isinstance(in_list[0], list):
        in_list = in_list[0]
    return in_list

class ToTensor(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, *pics):

        """
        Args:
            pic (PIL Image or numpy.ndarray): Image to be converted to tensor.

        Returns:
            Tensor: Converted image.
        """
        pics = strip_list(list(pics))
        if len(pics) == 1:
            return [torch.from_numpy(pics[0].astype('float'))]
        else:
            out_pic = []
            for pic in pics:
                out_pic.append(torch.from_numpy(pic.astype('float')))

            return out_pic

class CenterCrop(transformation_base_class):
    """Crops the given PIL Image at the center.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
    """

    def __init__(self, size):
        super(CenterCrop, self).__init__()
        if isinstance(size, numbers.Number):
            self.size = (int(size), int(size), int(size))
        else:
            self.size = size

    def __transforms__(self, img):
        """
        Args:
            img (PIL Image): Image to be cropped.

        Returns:
            PIL Image: Cropped image.
        """
        diff_dim = (img.shape[-3:] - self.size)//2
        center = img.shape[-3:] // 2
        start = center - diff_dim

        return img[...,
               start[0]:start[0]+self.size[0],
               start[1]:start[1]+self.size[1],
               start[2]:start[2]+self.size[2]
               ]

    def __call__(self, *tensors):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if len(tensors) == 1:
            return self.__transforms__(tensors[0])
        else:
            out_tensor = []
            for tensor in tensors:
                out_tensor.append(self.__transforms__(tensor))
            return out_tensor


class Lambda(transformation_base_class):
    """Apply a user-defined lambda as a transform.

    Args:
        lambd (function): Lambda/function to be used for transform.
    """

    def __init__(self, lambd):
        assert isinstance(lambd, types.LambdaType)
        self.lambd = lambd

    def __transforms__(self, img):
        return self.lambd(img)


class RandomAffine(object):
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

    def __init__(self,
                 degrees=20/180*math.pi,
                 translate=None,
                 scale=None,
                 shear=None,
                 uniform_scale = False,
                 uniform_shear = False,
                 resample=False,
                 separate_transform=True,
                 fillcolor=0,
                 freq = 0.8):
        self.freq = freq
        if degrees is not None:
            if isinstance(degrees, numbers.Number):
                if degrees < 0:
                    raise ValueError("If degrees is a single number, it must be positive.")
                self.degrees = np.array([[-1*degrees, degrees],
                                [-1*degrees, degrees],
                                [-1*degrees, degrees]]).ravel()
            elif isinstance(degrees, (tuple, list)):
                degrees = np.array(list(degrees))
            elif isinstance(degrees, np.ndarray):
                if len(degrees.ravel()) == 2:
                    if degrees[0]>=degrees[1]:
                        raise ValueError("If degrees is tuple or list with 2 elements, \
                        degrees[0] must be smaller than degrees[1].")
                    self.degrees = np.array([[degrees[0], degrees[1]],
                                    [degrees[0], degrees[1]],
                                    [degrees[0], degrees[1]]]).ravel()
                elif len(degrees.ravel()) == 3:
                    if not degrees.all() >= 0:
                        raise ValueError("If degrees is tupe or list of 3 elements, all should be >= 0")
                    self.degrees = np.array([[-degrees[0], degrees[0]],
                                             [-degrees[1], degrees[1]],
                                             [-degrees[2], degrees[2]]]).ravel()
                else:
                    assert len(degrees.ravel()) == 6, \
                        'If degree is tuple or list with more than 3 elements, \
                        it should have 6 elements'
                    self.degrees = degrees.ravel()

        else:
            self.degrees = np.zeros(6)


        if translate is not None:
            if isinstance(translate, numbers.Number):
                assert translate>=0, "if translate is a scaler, it should be >=0"
                self.translate = np.array([[-1*translate, translate],
                                [-1*translate, translate],
                                [-1*translate, translate]]).ravel()
            if isinstance(translate, (tuple, list)):
                translate = np.array(list(translate))
            if isinstance(translate, np.ndarray):
                if not (0.0 <= np.abs(translate).all() <= 1.0):
                    raise ValueError("translation values should be between 0 and 1")
                if len(translate) == 2:
                    if translate[0]>=translate[1]:
                        raise ValueError("If translate is tuple or list with 2 elements, \
                        translate[0] must be smaller than translate[1].")
                    self.translate = np.array([[translate[0], translate[1]],
                                    [translate[0], translate[1]],
                                    [translate[0], translate[1]]]).ravel()
                elif len(translate.ravel()) == 3:
                    if not translate.all() >= 0:
                        raise ValueError("If translate is tupe or list of 3 elements, all should be >= 0")
                    self.translate = np.array([[-translate[0], translate[0]],
                                             [-translate[1], translate[1]],
                                             [-translate[2], translate[2]]]).ravel()
                else:
                    assert len(translate.ravel()) == 6, \
                        'If translate is tuple or list with more than 3 elements, \
                        it should have 6 elements'
                    self.translate = translate.ravel()

        else:
            self.translate = None

        # self.translate = translate

        if scale is not None:
            if isinstance(scale, numbers.Number):
                assert scale>=1, "if scale is a scaler, it should be >=0"
                if uniform_scale:
                    self.scale = np.array([1/scale, scale])
                if not uniform_scale:
                    self.scale = np.array([[1/scale, scale],
                                [1/scale, scale],
                                [1/scale, scale]]).ravel()
            if isinstance(scale, (tuple, list)):
                scale = np.array(list(scale))
            if isinstance(scale, np.ndarray):
                if not scale.all() > 1:
                    raise ValueError("scale values should be bigger than 1")
                if len(scale) == 2:
                    if scale[0]>=scale[1]:
                        raise ValueError("If scale is tuple or list with 2 elements, \
                        scale[0] must be smaller than scale[1].")
                    if uniform_scale:
                        self.scale = np.array([scale[0], scale[1]])
                    if not uniform_scale:
                        self.scale = np.array([[scale[0], scale[1]],
                                        [scale[0], scale[1]],
                                        [scale[0], scale[1]]]).ravel()
                elif len(scale.ravel()) == 3:
                    if not scale.all() >= 1:
                        raise ValueError("If scale is tupe or list of 3 elements, all should be >= 1")
                    if uniform_scale:
                        raise ValueError("scale uniformly then scale can not has more than 2 elements")
                    self.scale = np.array([[1/scale[0], scale[0]],
                                             [1/scale[1], scale[1]],
                                             [1/scale[2], scale[2]]]).ravel()
                else:
                    if uniform_scale:
                        raise ValueError("scale uniformly then scale can not has more than 2 elements")
                    assert len(scale.ravel()) == 6, \
                        'If scale is tuple or list with more than 3 elements, \
                        it should have 6 elements'
                    self.scale = scale.ravel()

        else:
            self.scale = None

        # if scale is not None:
        #     assert isinstance(scale, (tuple, list)) and len(scale) == 2, \
        #         "scale should be a list or tuple and it must be of length 2."
        #     for s in scale:
        #         if s <= 0:
        #             raise ValueError("scale values should be positive")
        # self.scale = scale

        if shear is not None:
            if isinstance(shear, numbers.Number):
                # assert scale>=0, "if scale is a scaler, it should be >=0"
                if uniform_shear:
                    self.shear = np.array([-1*shear, shear])
                if not uniform_shear:
                    self.shear = np.array([[-1*shear, shear],
                                [-1*shear, shear],
                                [-1*shear, shear]]).ravel()
            if isinstance(shear, (tuple, list)):
                shear = np.array(list(shear))
            if isinstance(shear, np.ndarray):
                if len(shear) == 2:
                    if shear[0]>=shear[1]:
                        raise ValueError("If shear is tuple or list with 2 elements, \
                        shear[0] must be smaller than shear[1].")
                    if uniform_shear:
                        self.shear = np.array([shear[0], shear[1]])
                    if not uniform_shear:
                        self.shear = np.array([[shear[0], shear[1]],
                                        [shear[0], shear[1]],
                                        [shear[0], shear[1]]]).ravel()
                elif len(shear.ravel()) == 3:
                    if uniform_shear:
                        raise ValueError("shear uniformly then it can not has more than 2 elements")
                    self.shear = np.array([[-shear[0], shear[0]],
                                             [-shear[1], shear[1]],
                                             [-shear[2], shear[2]]]).ravel()
                else:
                    if uniform_shear:
                        raise ValueError("shear uniformly then shear can not has more than 2 elements")
                    assert len(shear.ravel()) == 6, \
                        'If shear is tuple or list with more than 3 elements, \
                        it should have 6 elements'
                    self.shear = shear.ravel()

        else:
            self.shear = None

        self.resample = resample
        self.fillcolor = fillcolor
        self.uniform_scale = uniform_scale
        self.uniform_shear = uniform_shear
        self.separate_transform = separate_transform

    @staticmethod
    def get_params(degrees,
                   translate,
                   scale_ranges,
                   shears,
                   img_size,
                   uniform_scale,
                   uniform_shear):
        """Get parameters for affine transformation

        Returns:
            sequence: params to be passed to the affine transformation
        """
        angle = np.array([np.random.uniform(degrees[0], degrees[1]),
                 np.random.uniform(degrees[2], degrees[3]),
                 np.random.uniform(degrees[4], degrees[5])])
        if translate is not None:
            min_dx, max_dx, min_dy, max_dy, min_dz, max_dz =  \
                translate[0], translate[1], \
                translate[2], translate[3], \
                translate[4], translate[5]
            # max_dy = translate[1] * img_size[1]
            translations = np.array([np.random.uniform(min_dx, max_dx),
                                     np.random.uniform(min_dy, max_dy),
                                     np.random.uniform(min_dz, max_dz)])
        else:
            translations = np.zeros(3)

        if scale_ranges is not None:
            if uniform_scale:
                temp_scale = np.random.uniform(scale_ranges[0], scale_ranges[1])
                scale = np.array([temp_scale, temp_scale, temp_scale])
            else:
                scale = np.array([
                    np.random.uniform(scale_ranges[0], scale_ranges[1]),
                    np.random.uniform(scale_ranges[2], scale_ranges[3]),
                    np.random.uniform(scale_ranges[4], scale_ranges[5])
                ])
        else:
            scale = np.ones(3)

        if shears is not None:
            if uniform_shear:
                temp_shear = np.random.uniform(shears[0], shears[1])
                shear = np.array([temp_shear, temp_shear, temp_shear])
            else:
                shear = np.array([
                    np.random.uniform(shears[0], shears[1]),
                    np.random.uniform(shears[2], shears[3]),
                    np.random.uniform(shears[4], shears[5]),
                ])
        else:
            shear = np.zeros(3)

        return angle, translations, scale, shear

    def transform_mat(self, img):
        """
            img (np.array): Image to be transformed.

        Returns:
            np.array: Affine transformed image.
        """
        angle, translations, scale, shear = \
            self.get_params(self.degrees,
                              self.translate,
                              self.scale,
                              self.shear,
                              img.shape,
                              uniform_scale=self.uniform_scale,
                              uniform_shear=self.uniform_shear
                              )

        trans_mat = affines.compose(T=translations,
                                    R=euler2mat(*angle),
                                    Z=scale,
                                    S=shear)

        trans_mat_left = np.eye(4)
        trans_mat_left[0, 3] = 2/img.shape[-3]
        trans_mat_left[1, 3] = 2/img.shape[-2]
        trans_mat_left[2, 3] = 2/img.shape[-1]
        trans_mat_right = np.eye(4)
        trans_mat_right[0, 3] = -2/img.shape[-3]
        trans_mat_right[1, 3] = -2/img.shape[-2]
        trans_mat_right[2, 3] = -2/img.shape[-1]
        trans_mat = np.matmul(
            np.matmul(trans_mat_left,
                      trans_mat),
            trans_mat_right
        )

        return trans_mat

    def transform(self, img, mat):
        if isinstance(img, np.ndarray):
            return ndi.affine_transform(img, mat, order=0)
        elif isinstance(img, torch.Tensor):
            mat = mat[:-1, :]
            img_shape = img.shape
            # for 3D transform
            if len(img.shape) != 5:
                img = img.expand(1, 1, img.shape[-3], img.shape[-2], img.shape[-1])
            assert len(img.shape) == 5

            if torch.cuda.is_available():

                img = img.cuda()
                mat = torch.from_numpy(mat).cuda()
            else:
                mat = torch.from_numpy(mat)

            if len(mat.shape) == 2:
                mat = mat.expand(1, mat.shape[-2], mat.shape[-1])
            assert len(mat.shape) == 3

            grid = F.affine_grid(mat, img.shape)
            # print(isinstance(grid, torch.cuda.FloatTensor))
            return F.grid_sample(
                img,
                grid.type(img.type()),
                padding_mode='border',
                mode='nearest'
            ).reshape(img_shape)

    def __call__(self, *imgs):
        imgs = strip_list(list(imgs))
        out_img = []
        if self.freq < random.random():
            if len(imgs) == 1:
                return imgs[0]
            else:
                return imgs
        if not self.separate_transform:
            trans_mat = self.transform_mat(imgs[0])
            for img in imgs:
                out_img.append(self.transform(img, trans_mat))
        else:
            for img in imgs:
                trans_mat = self.transform_mat(img)
                out_img.append(self.transform(img, trans_mat))

        if len(imgs)==1:
            return out_img[0]
        return out_img

    def __repr__(self):
        s = '{name}(degrees={degrees}'
        if self.translate is not None:
            s += ', translate={translate}'
        if self.scale is not None:
            s += ', scale={scale}'
        if self.shear is not None:
            s += ', shear={shear}'
        if self.resample > 0:
            s += ', resample={resample}'
        if self.fillcolor != 0:
            s += ', fillcolor={fillcolor}'
        s += ')'
        d = dict(self.__dict__)
        return s.format(name=self.__class__.__name__, **d)

# distortions


