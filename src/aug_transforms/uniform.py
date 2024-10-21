import torch
from aug_transforms.transformation_base import transformation_base_class

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
        if len(pics) == 1:
            return torch.from_numpy(pics[0].astype('float'))
        else:
            out_pic = []
            for pic in pics:
                out_pic.append(torch.from_numpy(pic.astype('float')))

            return out_pic


class Normalize_Whitten(transformation_base_class):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self):
        super(Normalize_Whitten, self).__init__()

    def __transforms__(self, tensors):
        if tensors.std == 0:
            epslon = 1e-12
        else:
            epslon = 0.0
        return (tensors - tensors.mean()) \
               / (tensors.std() + epslon)

class Normalize(transformation_base_class):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, mean, std):
        super(Normalize, self).__init__()
#         import numpy as np
        self.mean = float(mean)
        self.std = float(std)

    def __transforms__(self, tensors):
        return (tensors - self.mean) / self.std


class Normalize_Range(transformation_base_class):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: ``(M1,...,Mn)`` and std: ``(S1,..,Sn)`` for ``n`` channels, this transform
    will normalize each channel of the input ``torch.*Tensor`` i.e.
    ``input[channel] = (input[channel] - mean[channel]) / std[channel]``

    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
    """

    def __init__(self, min_value=0, max_value=1):
        super(Normalize_Range, self).__init__()
#         import numpy as np
        self.min_value = min_value
        self.max_value = max_value

    def __transforms__(self, tensors):
        return (tensors - tensors.min()) / \
               (tensors.max() - tensors.min() + (tensors.max() == tensors.min())) * \
               (self.max_value - self.min_value) + self.min_value