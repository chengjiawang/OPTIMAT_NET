class transformation_base_class(object):
    def __init__(self):
        pass
    def __get_params__(self):
        pass

    def __transforms__(self, tensor):
        """
        transformation apply to 1 input
        :return:transformed tensor
        """
        pass

    def __call__(self, *tensors):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.

        Returns:
            Tensor: Normalized Tensor image.
        """
        if len(tensors) == 1:
            return [self.__transforms__(tensors[0])]
        else:
            out_tensor = []
            for tensor in tensors:
                out_tensor.append(self.__transforms__(tensor))
            return out_tensor

def strip_list(in_list):
    while isinstance(in_list[0], list):
        in_list = in_list[0]
    return in_list

class Compose(object):
    """Composes several transforms3D_back together.

    Args:
        transforms (list of ``Transform`` objects): list of transforms3D_back to compose.

    Example:
        >>> aug_transforms.Compose([
        >>>     aug_transforms.CenterCrop(10),
        >>>     aug_transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, *imgs):
        imgs = strip_list(list(imgs))
        for t in self.transforms:
            imgs = t(*imgs)
            if not isinstance(imgs, list):
                imgs = [imgs]
        if len(imgs) == 1:
            return imgs[0]
        return imgs