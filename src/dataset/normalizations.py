class normalize_01(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (Torch or numpy.ndarray): normalize image -1,1.

        Returns:
            Tensor: Converted image.
        """
        return (pic - pic.min())/(pic.max() - pic.min())


    def __repr__(self):
        return self.__class__.__name__ + '()'

class normalize_11(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (Torch or numpy.ndarray): normalize image -1,1.

        Returns:
            Tensor: Converted image.
        """
        return 2*(pic - pic.min())/(pic.max() - pic.min()) - 1


    def __repr__(self):
        return self.__class__.__name__ + '()'

class normalize_white(object):
    """Convert a ``PIL Image`` or ``numpy.ndarray`` to tensor.

    Converts a PIL Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __call__(self, pic):
        """
        Args:
            pic (Torch or numpy.ndarray): normalize image -1,1.

        Returns:
            Tensor: Converted image.
        """
        return (pic - pic.mean())/pic.std()


    def __repr__(self):
        return self.__class__.__name__ + '()'