import numpy as np
import random

import torch
import torch.nn.functional as F
from torch import nn
from torch.autograd import Variable

from PIL import Image
import itertools
from aug_transforms.transformation_base import Compose


from skimage.transform import PiecewiseAffineTransform, warp
from torchvision.transforms.functional import _is_pil_image

from aug_transforms.transformation_base import transformation_base_class

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter

def elastic_transform_3D(image,
                         mask,
                         alpha,
                         sigma,
                         random_state=None):
    """Elastic deformation of images as described in [Simard2003]_ (with modifications).
    .. [Simard2003] Simard, Steinkraus and Platt, "Best Practices for
         Convolutional Neural Networks applied to Visual Document Analysis", in
         Proc. of the International Conference on Document Analysis and
         Recognition, 2003.

     Based on https://gist.github.com/erniejunior/601cdf56d2b424757de5
    """
    if random_state is None:
        random_state = np.random.RandomState(None)

    shape = image.shape

    dx = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dy = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha
    dz = gaussian_filter((random_state.rand(*shape) * 2 - 1), sigma) * alpha

    x, y, z = np.meshgrid(np.arange(shape[1]),
                          np.arange(shape[0]),
                          np.arange(shape[2]))

    indices = np.reshape(y+dy, (-1, 1)), \
              np.reshape(x+dx, (-1, 1)), \
              np.reshape(z+dz, (-1, 1))


    res_image = map_coordinates(image, indices, order=1, mode='nearest').reshape(shape)
    if mask is not None:
        res_mask = map_coordinates(mask, indices, order=1, mode='nearest').reshape(shape)
        return res_image, res_mask
    return res_image

class RandomElasticDeform3D(transformation_base_class):
    def __init__(self,
                 freq = 0.9,
                 random_state = None):
        super(RandomElasticDeform3D, self).__init__()
        self.freq = freq
        if random_state is None:
            self.random_state = np.random.RandomState(None)
        else:
            self.random_state = random_state

    def __get_params__(self,
                       shape,
                       alpha = None,
                       sigma = None):
        import numbers
        if alpha is None:
            self.alpha = np.array(list(shape)) * 10
        elif isinstance(alpha, numbers.Number):
            self.alpha = np.array([alpha, alpha, alpha])
        else:
            self.alpha = alpha

        if sigma is None:
            self.sigma = np.array(list(shape)) * 0.12
        elif isinstance(sigma, numbers.Number):
            self.sigma = np.array([sigma, sigma, sigma])
        else:
            self.sigma = sigma

        dx = gaussian_filter((self.random_state.rand(*shape) * 2 - 1),
                             self.sigma[1]) * self.alpha[1]
        dy = gaussian_filter((self.random_state.rand(*shape) * 2 - 1),
                             self.sigma[0]) * self.alpha[0]
        dz = gaussian_filter((self.random_state.rand(*shape) * 2 - 1),
                             self.sigma[2]) * self.alpha[2]

        x, y, z = np.meshgrid(np.arange(shape[1]),
                              np.arange(shape[0]),
                              np.arange(shape[2]))

        self.shape = shape

        self.indices = np.reshape(y + dy, (-1, 1)), \
                  np.reshape(x + dx, (-1, 1)), \
                  np.reshape(z + dz, (-1, 1))


    def __transforms__(self, tensor):
        print('distorting...')
        deformed_tensor = map_coordinates(tensor,
                                          self.indices,
                                          order=1,
                                          mode='nearest').reshape(self.shape)
        return deformed_tensor

    def __call__(self, *tensors):
        print('tensors: ', len(tensors))
        if self.freq < random.random():
            return tensors[0]

        self.__get_params__(tensors[0].shape)

        if len(tensors) == 1:
            return self.__transforms__(tensors[0])

        else:
            out_tensor = []
            for tensor in tensors:
                out_tensor.append(self.__transforms__(tensor))
            return out_tensor

def _pytorch_identy_grid(shape):
    target_height = shape[0]
    target_width = shape[1]
    target_depth = shape[2]
    target_coordinate = list(itertools.product(range(target_height),
                                               range(target_width),
                                               range(target_depth)))
    target_coordinate = torch.Tensor(target_coordinate)  # HW x 2
    Y, X, Z = target_coordinate.split(1, dim=1)
    Y = Y * 2 / (target_height - 1) - 1
    X = X * 2 / (target_width - 1) - 1
    Z = Z * 2 / (target_depth - 1) - 1
    target_coordinate = torch.cat([Z, X, Y], dim=1)
    return target_coordinate

class RandomElasticDeform3D_pytorch(transformation_base_class):
    def __init__(self,
                 shape,
                 freq = 0.9,
                 alpha=None,
                 sigma=None,
                 is_cuda = True):
        super(RandomElasticDeform3D_pytorch, self).__init__()
        import numbers
        if alpha is None:
            self.alpha = np.array([2, 2, 2])
        elif isinstance(alpha, numbers.Number):
            self.alpha = np.array([alpha, alpha, alpha])
        else:
            self.alpha = alpha

        if sigma is None:
            self.sigma = np.array([0.1, 0.1, 0.1])
        elif isinstance(sigma, numbers.Number):
            self.sigma = np.array([sigma, sigma, sigma])
        else:
            self.sigma = sigma

        self.freq = freq
        self.shape = shape
        self.id_grid = _pytorch_identy_grid(torch.Size(shape)).view(*shape, 3).unsqueeze(0)

        if is_cuda:
            self.id_grid = self.id_grid.cuda()

        self.is_cuda = is_cuda


    def __get_params__(self):

        dx = gaussian_filter(np.random.randn(*self.shape)/ self.shape[1],
                             self.sigma[1])
        dy = gaussian_filter(np.random.randn(*self.shape)/ self.shape[0],
                             self.sigma[0])
        dz = gaussian_filter(np.random.randn(*self.shape)/ self.shape[2],
                             self.sigma[2])

        self.rand_disp = torch.Tensor(np.stack([dz, dx, dy], axis=-1))

        if self.is_cuda:
            self.rand_disp = self.rand_disp.cuda()

        self.rand_disp = self.rand_disp.unsqueeze(0)

    def __transforms__(self, tensor):
        print('distorting...')
        deformed_tensor = F.grid_sample(tensor[-3:].unsqueeze(0).unsqueeze(0),
                                        (self.id_grid+self.rand_disp).type_as(tensor),
                                        mode='nearest',
                                          padding_mode='border').squeeze(0).squeeze(0)
        return deformed_tensor

    def __call__(self, *tensors):
        if self.freq < random.random():
            return tensors[0]

        self.__get_params__()

        if len(tensors) == 1:
            return self.__transforms__(tensors[0])

        else:
            out_tensor = []
            for tensor in tensors:
                out_tensor.append(self.__transforms__(tensor))
            return out_tensor

def ar_distort(image, time_sin):
    rows, cols = image.shape[0], image.shape[1]

    src_cols = np.linspace(0, cols, 20)
    src_rows = np.linspace(0, rows, 20)
    src_rows, src_cols = np.meshgrid(src_rows, src_cols)
    src = np.dstack([src_cols.flat, src_rows.flat])[0]

    # add sinusoidal oscillation to row coordinates
    dst_rows = src[:, 1] - np.sin(np.linspace(0, 3 * np.pi, src.shape[0]))*time_sin
    dst_cols = src[:, 0]

    dst = np.vstack([dst_cols, dst_rows]).T

    tform = PiecewiseAffineTransform()
    tform.estimate(src, dst)

    out_rows = image.shape[0]
    out_cols = cols
    out = warp(image, tform, output_shape=(out_rows, out_cols))
    return out

class ArDistort(transformation_base_class):
    """numpy array or torch tensor an arbitrary sin distortion.

    Args:
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made.
        padding (int or sequence, optional): Optional padding on each border
            of the image. Default is 0, i.e no padding. If a sequence of length
            4 is provided, it is used to pad left, top, right, bottom borders
            respectively.
    """

    def __init__(self, time_of_sin):
        super(ArDistort, self).__init__()
        self.time_sin = time_of_sin

    @staticmethod
    def get_params(img, output_size):
        pass

    def __transforms__(self, img):
        # if PIL image might be slow
        if _is_pil_image(img):
            image = np.asarray(img)
        else:
            image = img
        if _is_pil_image(img):
            return Image.fromarray(ar_distort(image, self.time_sin))
        else:
            return ar_distort(image, self.time_sin)

    def __repr__(self):
        return self.__class__.__name__ + '(time_of_sin={0})'.format(self.time_sin)

class TPSTransform(transformation_base_class):
    def __init__(self,
                 scale_factor,
                 span_range_height = 0.9,
                 span_range_width = 0.9,
                 grid_height = 6,
                 grid_width = 6):
        super(TPSTransform, self).__init__()
        self.scale_factor = scale_factor
        r1 = span_range_height
        r2 = span_range_width
        target_control_points = torch.Tensor(list(itertools.product(
            np.arange(-r1, r1 + 0.00001, 2.0 * r1 / (grid_height - 1)),
            np.arange(-r2, r2 + 0.00001, 2.0 * r2 / (grid_width - 1)),
        )))
        Y, X = target_control_points.split(1, dim=1)
        self.target_control_points = torch.cat([X, Y], dim=1)
        self.tps = TPSGridGen(self.target_control_points)
        self.inter = min(2.0 * r1 / (grid_height - 1),
                         2.0 * r2 / (grid_width - 1))

    @staticmethod
    def get_params(img, output_size):
        pass

    def __transforms__(self, img):
        shifts = torch.rand(
            *self.target_control_points.shape
        ) * self.inter * self.scale_factor

        source_control_points = self.target_control_points + shifts

        source_coordinate = self.tps.forward(
            source_control_points.unsqueeze(0),
            img.unsqueeze(0))

        grid = source_coordinate.view(img.shape[0],
                                      img.shape[1],
                                      img.shape[2],
                                      2)
        transformed_x = F.grid_sample(
            input=img.unsqueeze(0),
            grid=grid,
            padding_mode='border'
        )
        return transformed_x.squeeze(0)


def get_distort(opt):
    transform_list = []
    if opt.distort_factor!=0:
        transform_list.append(ArDistort(opt.distort_factor))
    return Compose(transform_list)

def get_tps_distort(scale_factor):
    transform_list = []
    if scale_factor!=0:
        transform_list.append(TPSTransform(scale_factor))
    return Compose(transform_list)

# phi(x1, x2) = r^2 * log(r), where r = ||x1 - x2||_2
def compute_partial_repr(input_points, control_points):
    N = input_points.size(0)
    M = control_points.size(0)
    if input_points.is_cuda:
        control_points = control_points.cuda()
    else:
        control_points = control_points.cpu()
    pairwise_diff = input_points.view(N, 1, 2) - control_points.view(1, M, 2)

    pairwise_diff_square = pairwise_diff * pairwise_diff
    pairwise_dist = pairwise_diff_square[:, :, 0] + pairwise_diff_square[:, :, 1]
    repr_matrix = 0.5 * pairwise_dist * torch.log(pairwise_dist)

    mask = repr_matrix != repr_matrix
    repr_matrix.masked_fill_(mask, 0)
    return repr_matrix


class TPSGridGen(nn.Module):

    def __init__(self, target_control_points):
        super(TPSGridGen, self).__init__()
        self.target_control_points = target_control_points
        assert target_control_points.ndimension() == 2
        assert target_control_points.size(1) == 2
        N = target_control_points.size(0)
        self.num_points = N
        target_control_points = target_control_points.float()

        # create padded kernel matrix
        forward_kernel = torch.zeros(N + 3, N + 3)
        target_control_partial_repr = compute_partial_repr(target_control_points, target_control_points)
        forward_kernel[:N, :N].copy_(target_control_partial_repr)
        forward_kernel[:N, -3].fill_(1)
        forward_kernel[-3, :N].fill_(1)
        forward_kernel[:N, -2:].copy_(target_control_points)
        forward_kernel[-2:, :N].copy_(target_control_points.transpose(0, 1))
        # compute inverse matrix
        inverse_kernel = torch.inverse(forward_kernel)
        if target_control_points.is_cuda:
            inverse_kernel = inverse_kernel.cuda()


        # register precomputed matrices
        self.register_buffer('inverse_kernel', inverse_kernel)
        self.register_buffer('padding_matrix', torch.zeros(3, 2))


    def forward(self, source_control_points, x):
        assert source_control_points.ndimension() == 3
        assert source_control_points.size(1) == self.num_points
        assert source_control_points.size(2) == 2
        target_height= x.shape[2]
        target_width = x.shape[3]
        # create target cordinate matrix
        HW = target_height * target_width
        target_coordinate = list(itertools.product(range(target_height), range(target_width)))
        target_coordinate = torch.Tensor(target_coordinate) # HW x 2
        Y, X = target_coordinate.split(1, dim = 1)
        Y = Y * 2 / (target_height - 1) - 1
        X = X * 2 / (target_width - 1) - 1
        target_coordinate = torch.cat([X, Y], dim = 1) # convert from (y, x) to (x, y)
        target_coordinate_partial_repr = compute_partial_repr(target_coordinate, self.target_control_points)
        target_coordinate_repr = torch.cat([
            target_coordinate_partial_repr, torch.ones(HW, 1), target_coordinate
        ], dim = 1)
        if x.is_cuda:
            target_coordinate_repr = target_coordinate_repr.cuda()
        self.register_buffer('target_coordinate_repr', target_coordinate_repr)

        batch_size = source_control_points.size(0)
        Y = torch.cat([source_control_points, Variable(self.padding_matrix.expand(batch_size, 3, 2))], 1)
        mapping_matrix = torch.matmul(Variable(self.inverse_kernel), Y)
        source_coordinate = torch.matmul(Variable(self.target_coordinate_repr), mapping_matrix)
        return source_coordinate