# full assembly of the sub-parts to form the complete net

from models.CBAM3D import SpatialGateFixScale
from models.unet3D_mstream_parts import *
import numpy as np
import torch
from torch import nn
from dataset.OPTIMAT import make_one_hot
from torch.nn import functional as F

# Defines the GAN loss which uses either LSGAN or the regular GAN.
# When LSGAN is used, it is basically same as MSELoss,
# but it abstracts away the need to create the target label tensor
# that has the same size as the input
class WeightedDiceLoss(nn.Module):
    def __init__(self, homo_scale = 0.0):
        super(WeightedDiceLoss, self).__init__()
        self.homo_scale = homo_scale

    def __call__(self, y_pred, y_true):
        smooth = 1e-12
        if len(y_pred.shape) < len(y_true.shape):
            y_pred = y_pred.unsqueeze(0)
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        temp_greater = class_sums > 0

        class_weights = 1/(smooth*(1-temp_greater.float())+class_sums**2)
        # class_weights = 1/(smooth+class_sums)*temp_greater.float()

        class_weights = class_weights/( torch.sum(class_weights)) * y_true.shape[1]

        temp = y_true_reshaped * y_pred_reshaped

        # intersection = np.sum(temp, [0,1], keepdims=False)
        intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        # homogeneity = temp.std()
        if self.homo_scale >= 0.0:
            homogeneity = homo_func(y_true, y_pred) * self.homo_scale
        else:
            homogeneity = 0.0
        # homogeneity = torch.mean(torch.std(temp, dim=-1))

        dices = torch.sum( (2*intersection*class_weights) /(
                class_sums + torch.sum(torch.sum(y_pred_reshaped,dim =-1), dim=0))*class_weights)

        return -dices - homogeneity
        # return -dices

class DiceLoss_multi(nn.Module):
    def __init__(self, homo_scale = 0.001, num_class = 9):
        super(DiceLoss_multi, self).__init__()
        self.homo_scale = homo_scale
        self.num_class = num_class

    def __call__(self, y_in, y_true):
        smooth = 1e-12

        if len(y_in.shape) < len(y_true.shape):
            y_in = y_in.unsqueeze(0)
        y_pred = y_in[:, :self.num_class, ...]

        # print('y_true:', y_true.shape)
        # print('y_pred:', y_pred.shape)
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        # temp_greater = class_sums > 0

        temp = y_true_reshaped * y_pred_reshaped

        # intersection = np.sum(temp, [0,1], keepdims=False)
        intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        # homogeneity = temp.std()
        # homogeneity = homo_func(y_true, y_pred)
        # homogeneity = torch.mean(torch.std(temp, dim=-1))

        dices = torch.sum( (2*intersection + smooth) /(
                class_sums + torch.sum(torch.sum(y_pred_reshaped,dim =-1), dim=0) + smooth))

        loss = dices

        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        for i in range(1, y_in.shape[1]//self.num_class):
            y_pred = y_in[:, i*self.num_class:(i+1)*self.num_class, ...]
            y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)
            temp = y_true_reshaped * y_pred_reshaped

            # intersection = np.sum(temp, [0,1], keepdims=False)
            intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)
            dices = torch.sum((2 * intersection + smooth) / (
                    class_sums + torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + smooth))

            loss += dices

        return -loss - homogeneity

class WeightedDiceLoss_multi(nn.Module):
    def __init__(self, homo_scale = 0.001, num_class = 9):
        super(WeightedDiceLoss_multi, self).__init__()
        self.homo_scale = homo_scale
        self.num_class = num_class

    def __call__(self, y_in, y_true):
        smooth = 1e-12

        if len(y_in.shape) < len(y_true.shape):
            y_in = y_in.unsqueeze(0)
        y_pred = y_in[:, :self.num_class, ...]

        # print('y_true:', y_true.shape)
        # print('y_pred:', y_pred.shape)
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        temp_greater = class_sums > 0

        # class_weights = 1/(smooth+class_sums)*temp_greater.float()
        class_weights = 1/(smooth*(1-temp_greater.float())+class_sums)

        class_weights = class_weights/( torch.sum(class_weights)) * self.num_class

        temp = y_true_reshaped * y_pred_reshaped

        # intersection = np.sum(temp, [0,1], keepdims=False)
        intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        # homogeneity = temp.std()
        # homogeneity = homo_func(y_true, y_pred)
        # homogeneity = torch.mean(torch.std(temp, dim=-1))

        dices = torch.sum( (2*intersection*class_weights) /(
                class_sums + torch.sum(torch.sum(y_pred_reshaped,dim =-1), dim=0))*class_weights)

        loss = dices

        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        for i in range(1, y_in.shape[1]//self.num_class):
            y_pred = y_in[:, i*self.num_class:(i+1)*self.num_class, ...]
            y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)
            temp = y_true_reshaped * y_pred_reshaped

            # intersection = np.sum(temp, [0,1], keepdims=False)
            intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)
            dices = torch.sum((2 * intersection + smooth) * class_weights / (
                    class_sums + torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + smooth))

            loss += dices

        return -loss - homogeneity
        # return -loss - self.homo_fact*homo_func(y_true, y_pred)

def homo_func(y_true, y_pred, D = 3):
    '''
    homogeneity function
    D: dimension of data
    '''
    if D==3 or D==2:
        # grid = torch.meshgrid(
        #     [
        #         torch.linspace(0, 1, y_true.shape[-3]),
        #         torch.linspace(0, 1, y_true.shape[-2]),
        #         torch.linspace(0, 1, y_true.shape[-1])
        #     ])
        if torch.cuda.is_available():
            grid = torch.meshgrid(
                [
                    torch.linspace(0, 1, y_true.shape[i-D]).cuda() for i in range(D)
                ])
        else:
            grid = torch.meshgrid(
                [
                    torch.linspace(0, 1, y_true.shape[i - D]) for i in range(D)
                ])
    else:
        print('Error: Only 3D and 2D data supported now!')

    # multiply
    temp = torch.cat([
            y_pred.view(y_pred.shape[0]*y_pred.shape[1], -1) * \
            grid[i].unsqueeze(0).expand_as(y_pred).contiguous().view(y_pred.shape[0]*y_pred.shape[1],
                                                                     -1) \
            for i in range(D)
            ], dim=0)

    res = torch.mean(torch.std(temp, dim=1))
    return res

class DynamicWeightedDiceLoss(nn.Module):
    def __init__(self,
                 homo_scale = 0.0,
                 update_step = 128,
                 initial_train_epochs = 128 * 20,
                 num_class = 9):
        super(DynamicWeightedDiceLoss, self).__init__()
        self.cached_weight = None
        self.count=0
        self.homo_scale = homo_scale
        self.count_thres = update_step
        self.initial_steps = initial_train_epochs
        self.ini_count = 0
        self.update = False
        self.num_class = num_class


    def __call__(self, y_pred, y_true):
        smooth = 1e-12
        # print('y_true:', y_true.shape)
        # print('y_pred:', y_pred.shape)
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        temp_greater = class_sums > 0

        if self.ini_count >= self.initial_steps:
            self.update = True

        if self.cached_weight is None:
            class_weights = 1/(smooth+class_sums)*temp_greater.float()

            # class_weights = class_weights/( torch.sum(class_weights))
            class_weights = class_weights/( torch.sum(class_weights)) * y_true.shape[1]
        else:
            class_weights = self.cached_weight

        # class_weights = class_weights.detach()

        temp = y_true_reshaped * y_pred_reshaped

        # intersection = np.sum(temp, [0,1], keepdims=False)
        intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        # homogeneity = torch.mean(torch.std(temp, dim=-1))

        class_dices = (2*intersection + smooth)*class_weights.detach() /(
                class_sums + torch.sum(torch.sum(y_pred_reshaped,dim =-1), dim=0) + smooth)

        if self.count >= self.count_thres:
            # using hard dice as weight
            hard_dices = 2*torch.sum(
                make_one_hot(
                torch.argmax(
                    y_pred_reshaped,
                    dim=1,
                    keepdim=True
                ),
                C=self.num_class
            )*y_true_reshaped,
            dim=(0,-1)) / ( class_sums + torch.sum(
                make_one_hot(
                    torch.argmax(
                        y_pred_reshaped,
                        dim=1,
                        keepdim=True
                    ),
                    C=self.num_class
                ),
                dim=(0,-1)
            ) + smooth )

            self.cached_weight = 1 - hard_dices
            self.cached_weight = self.cached_weight / torch.sum(self.cached_weight) * self.num_class
            self.count = 0

            # using soft dice as weight
            # self.cached_weight = 1 - class_dices
            # self.cached_weight = self.cached_weight / torch.sum(self.cached_weight) * self.num_class
            # self.count = 0

        dices = torch.sum( class_dices )

        self.count += 1
        self.ini_count += 1

        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        # return -dices - 0.001*homogeneity
        return -dices - homogeneity

class DynamicWeightedDiceLoss_multi(nn.Module):
    def __init__(self,
                 homo_scale = 0.001,
                 num_class = 9,
                 update_step = 128,
                 initial_train_epochs = 20 * 128):
        super(DynamicWeightedDiceLoss_multi, self).__init__()
        self.homo_scale = homo_scale
        self.num_class = num_class
        self.cached_weight = None
        self.count=0
        self.count_thres = update_step
        self.initial_steps = initial_train_epochs
        self.ini_count = 0
        self.update = False

    def __call__(self, y_in, y_true):
        smooth = 1e-12

        if len(y_in.shape) < len(y_true.shape):
            y_in = y_in.unsqueeze(0)
        y_pred = y_in[:, :self.num_class, ...]

        # print('y_true:', y_true.shape)
        # print('y_pred:', y_pred.shape)
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        temp_greater = class_sums > 0

        if self.ini_count >= self.initial_steps:
            self.update = True

        if self.cached_weight is None or self.update == False:
            class_weights = 1/(smooth+class_sums)*temp_greater.float()

            # class_weights = class_weights/( torch.sum(class_weights))
            class_weights = class_weights / (torch.sum(class_weights)) * self.num_class
        else:
            class_weights = self.cached_weight

        class_weights = class_weights.detach()
        # print(class_weights)

        # class_weights = 1/(smooth+class_sums)*temp_greater.float()
        #
        # class_weights = class_weights/( torch.sum(class_weights))

        temp = y_true_reshaped * y_pred_reshaped

        # intersection = np.sum(temp, [0,1], keepdims=False)
        intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        # homogeneity = temp.std()
        # homogeneity = homo_func(y_true, y_pred)
        # homogeneity = torch.mean(torch.std(temp, dim=-1))

        class_dices = (2*intersection + smooth)*class_weights /(
                class_sums + torch.sum(torch.sum(y_pred_reshaped,dim =-1), dim=0) + smooth)

        if self.count >= self.count_thres:
            # using hard dice as weight
            hard_dices = 2*torch.sum(
                make_one_hot(
                torch.argmax(
                    y_pred_reshaped,
                    dim=1,
                    keepdim=True
                ),
                C=self.num_class
            )*y_true_reshaped,
            dim=(0,-1)) / ( class_sums + torch.sum(
                make_one_hot(
                    torch.argmax(
                        y_pred_reshaped,
                        dim=1,
                        keepdim=True
                    ),
                    C=self.num_class
                ),
                dim=(0,-1)
            ) + smooth )

            self.cached_weight = 1 - hard_dices
            self.cached_weight = self.cached_weight / torch.sum(self.cached_weight) * self.num_class
            self.count = 0

            # using soft dice as weight
            # self.cached_weight = 1 - class_dices
            # self.cached_weight = self.cached_weight / torch.sum(self.cached_weight) * self.num_class
            # self.count = 0

        loss = torch.sum( class_dices )

        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        for i in range(1, y_in.shape[1]//self.num_class):
            y_pred = y_in[:, i*self.num_class:(i+1)*self.num_class, ...]
            y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)
            temp = y_true_reshaped * y_pred_reshaped

            # intersection = np.sum(temp, [0,1], keepdims=False)
            intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)
            dices = torch.sum((2 * intersection + smooth) * class_weights / (
                    class_sums + torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + smooth))

            loss += dices

        self.count += 1
        self.ini_count += 1

        return -loss - homogeneity
        # return -loss - self.homo_fact*homo_func(y_true, y_pred)

class DynamicWeightedDiceLoss_multi_bk(nn.Module):
    def __init__(self):
        super(DynamicWeightedDiceLoss_multi_bk, self).__init__()
        self.cached_weight = None
        self.count=0
        self.count_thres = 128

    def __call__(self, y_in, y_true):
        smooth = 1e-12
        # print('y_true:', y_true.shape)
        # print('y_pred:', y_pred.shape)
        y_pred = y_in[0]
        y_joint = y_in[1]
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        temp_greater = class_sums > 0

        if self.cached_weight is None:
            class_weights = 1/(smooth+class_sums)*temp_greater.float()

            class_weights = class_weights/( torch.sum(class_weights))
        else:
            class_weights = self.cached_weight

        # class_weights = class_weights.detach()

        temp = y_true_reshaped * y_pred_reshaped

        # intersection = np.sum(temp, [0,1], keepdims=False)
        intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        # homogeneity = torch.mean(torch.std(temp, dim=-1))

        class_dices = (2*intersection + smooth)*class_weights.detach() /(
                class_sums + torch.sum(torch.sum(y_pred_reshaped,dim =-1), dim=0) + smooth)

        if self.count >= self.count_thres:
            self.cached_weight = 1 - class_dices
            self.cached_weight = self.cached_weight / torch.sum(self.cached_weight)
            self.count = 0

        dices = torch.sum( class_dices )

        loss = dices
        for y_pred in y_joint:
            y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)
            temp = y_true_reshaped * y_pred_reshaped

            # intersection = np.sum(temp, [0,1], keepdims=False)
            intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

            # homogeneity = torch.mean(torch.std(temp, dim=-1))
            class_dices = (2 * intersection + smooth) * class_weights.detach() / (
                    class_sums + torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + smooth)

            dices = torch.sum( class_dices )
            loss += dices

        self.count += 1

        # return -dices - 0.001*homogeneity
        return -loss

def dice_index(y_pred, y_true, smooth= 0.0):
    # smooth = 1e-12
    if len(y_pred.shape) < len(y_true.shape):
        y_pred = y_pred.unsqueeze(0)

    y_true_shape = y_true.size()
    y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
    y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

    temp = y_true_reshaped * y_pred_reshaped
    # intersection = np.sum(temp, [0,1], keepdims=False)
    intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

    dices = (2 * intersection + smooth) / (torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0) +
                                torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + smooth)

    return torch.mean(dices)

class WeightedCrossEntropyFromSoftmax_multi(nn.Module):
    def __init__(self,
                 homo_scale=0.001,
                 num_class=9,
                 ignore_index=-100,
                 reduction='sum'):
        super(WeightedCrossEntropyFromSoftmax_multi, self).__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.homo_scale = homo_scale
        self.num_class=num_class

    def __call__(self, y_in, y_true):
        smooth = 1e-12

        if len(y_in.shape) < len(y_true.shape):
            y_in = y_in.unsqueeze(0)
        y_pred = y_in[:, :self.num_class, ...]

        # print('y_true:', y_true.shape)
        # print('y_pred:', y_pred.shape)
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = torch.log(y_pred.view(y_true_shape[0], y_true_shape[1], -1))

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        temp_greater = class_sums > 0

        class_sums = smooth*(1-temp_greater.float())+class_sums
        # class_weights = 1/(smooth+class_sums)*temp_greater.float()
        class_weights = (
                                torch.prod(torch.Tensor(list(y_true_shape))).item() - class_sums
                         )/class_sums


        class_weights = class_weights/( torch.sum(class_weights))
        loss = F.nll_loss(input=y_pred_reshaped,
                          target=torch.argmax(y_true_reshaped, dim=1),
                          weight=class_weights,
                          reduction=self.reduction)

        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        for i in range(1, y_in.shape[1]//self.num_class):
            y_pred = y_in[:, i*self.num_class:(i+1)*self.num_class, ...]
            y_pred_reshaped = torch.log(y_pred.view(y_true_shape[0], y_true_shape[1], -1))
            WCE = F.nll_loss(input=y_pred_reshaped,
                              target=torch.argmax(y_true_reshaped, dim=1),
                              weight=class_weights,
                             reduction=self.reduction)

            loss += WCE

        return loss - homogeneity

class WeightedCrossEntropyFromSoftmax(nn.Module):
    def __init__(self,
                 homo_scale=0.001,
                 ignore_index=-100,
                 reduction='sum'):
        super(WeightedCrossEntropyFromSoftmax, self).__init__()
        # self.weight = weight
        self.ignore_index = ignore_index
        self.reduction = reduction
        self.homo_scale = homo_scale

    def __call__(self, y_pred, y_true):
        smooth = 1e-12
        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = torch.log(y_pred.view(y_true_shape[0], y_true_shape[1], -1))

        class_sums = torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0)

        temp_greater = class_sums > 0

        class_sums = smooth*(1-temp_greater.float())+class_sums
        # class_weights = 1/(smooth+class_sums)*temp_greater.float()
        class_weights = (
                                torch.prod(torch.Tensor(list(y_true_shape))).item() - class_sums
                         )/class_sums
        loss = F.nll_loss(input=y_pred_reshaped,
                          target=torch.argmax(y_true_reshaped, dim=1),
                          weight=class_weights,
                          reduction=self.reduction)

        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        return loss - homogeneity



class DiceIndex(nn.Module):
    def __init__(self):
        super(DiceIndex, self).__init__()

    def __call__(self, y_pred, y_true):
        smooth = 0
        return dice_index(y_pred, y_true, smooth)
        # if len(y_pred.shape) < len(y_true.shape):
        #     y_pred = y_pred.unsqueeze(0)
        #
        # y_true_shape = y_true.size()
        # y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        # y_pred_reshaped = F.softmax(y_pred.view(y_true_shape[0], y_true_shape[1], -1), dim=1)
        #
        # temp = y_true_reshaped * y_pred_reshaped
        # # intersection = np.sum(temp, [0,1], keepdims=False)
        # intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)
        #
        # dices = 2 * intersection / (torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0) +
        #                             torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + smooth)
        #
        # return torch.mean(dices)


class DiceLoss(nn.Module):
    def __init__(self, homo_scale = 0.0, smooth=1e-12):
        super(DiceLoss, self).__init__()
        self.homo_scale = homo_scale
        self.smooth = smooth

    def __call__(self, y_pred, y_true):
        # smooth = 1e-12
        if len(y_pred.shape) < len(y_true.shape):
            y_pred = y_pred.unsqueeze(0)

        y_true_shape = y_true.size()
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        temp = y_true_reshaped * y_pred_reshaped
        # intersection = np.sum(temp, [0,1], keepdims=False)
        intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        dices = 2 * intersection / (torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0) +
                                    torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + self.smooth)


        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        return -torch.sum(dices) - homogeneity

def eval_dice_all(y_true, y_pred):

    y_true_shape = y_true.shape
    # print(y_true_shape)
    # print(y_pred.shape)
    y_true_reshaped = y_true.view(y_true_shape[0], -1)
    y_pred_reshaped = y_pred.view(y_true_shape[0], -1)

    temp = y_true_reshaped * y_pred_reshaped
    # intersection = np.sum(temp, [0,1], keepdims=False)
    intersection = torch.sum(temp, dim=1)
    # print(intersection.shape)

    dices = 2 * intersection / ( torch.sum(y_true_reshaped, dim=1)+
                      torch.sum(y_pred_reshaped, dim=1))

    return dices

def eval_dice_single_class(y_true, y_pred):

    y_true_shape = y_true.size()
    y_true_reshaped = y_true.view(y_true_shape[0], -1)
    y_pred_reshaped = y_pred.view(y_true_shape[0], -1)

    temp = y_true_reshaped * y_pred_reshaped
    # intersection = np.sum(temp, [0,1], keepdims=False)
    intersection = temp.sum()
    dices = (2 * intersection) / ( torch.sum(y_true_reshaped)+
                      torch.sum(y_pred_reshaped, dim=0))
    return dices

class UNet3D_EncoderUnit(nn.Module):
    def __init__(self,
                 n_channels=1,
                 ngf=16,
                 n_block = 4):
        super(UNet3D_EncoderUnit, self).__init__()
        self.modellist = [inconv(n_channels, ngf)]
        for i in range(n_block-1):
            self.modellist += [down((2**i)*ngf, (2**(i+1))*ngf)]
        self.modellist += [down((2**(n_block-1))*ngf, (2**(n_block-1))*ngf)]
        self.modellist = nn.ModuleList(self.modellist)

        self.outlist = [None]*(n_block+1)
        self.n_block = n_block
        # self.inc = inconv(n_channels, ngf)
        # self.down1 = down(ngf, ngf * 2)
        # self.down2 = down(ngf*2, ngf*4)
        # self.down3 = down(ngf*4, ngf*8)
        # self.down4 = down(ngf*8, ngf*8)

    def forward(self, x):
        self.outlist[0] = self.modellist[0](x)
        for i in range(1, self.n_block+1):
            self.outlist[i] = self.modellist[i](self.outlist[i-1])
        return self.outlist

class UNet3D_Decoder_WithDrop(nn.Module):
    def __init__(self,
                 n_classes,
                 ngf=16,
                 n_block=4):
        super(UNet3D_Decoder_WithDrop, self).__init__()
        self.modellist = []
        for i in range(n_block-1):
            self.modellist += [up(2**(n_block-i)*ngf, 2**(n_block-i-2)*ngf)]
        self.modellist += [up(2*ngf, ngf)]
        self.modellist += [nn.Dropout(0.5)]
        self.modellist += [outconv(ngf, n_classes)]
        self.modellist += [nn.Softmax(dim=1)]
        self.modellist = nn.ModuleList(self.modellist)
        self.n_block = n_block
    def forward(self, x_list):
        x = self.modellist[0](x_list[-1], x_list[-2])
        for i in range(1, self.n_block):
            x = self.modellist[i](x, x_list[-1*(i+2)])
        for i in range(self.n_block, len(self.modellist)):
            x = self.modellist[i](x)
        return x

class UNet3D_Decoder(nn.Module):
    def __init__(self,
                 n_classes,
                 ngf=16,
                 n_block=4):
        super(UNet3D_Decoder, self).__init__()
        self.modellist = []
        for i in range(n_block-1):
            self.modellist += [up(2**(n_block-i)*ngf, 2**(n_block-i-2)*ngf)]
        self.modellist += [up(2*ngf, ngf)]
        self.modellist += [outconv(ngf, n_classes)]
        self.modellist += [nn.Softmax(dim=1)]
        self.modellist = nn.ModuleList(self.modellist)
        self.n_block = n_block
    def forward(self, x_list):
        x = self.modellist[0](x_list[-1], x_list[-2])
        for i in range(1, self.n_block):
            x = self.modellist[i](x, x_list[-1*(i+2)])
        for i in range(self.n_block, len(self.modellist)):
            x = self.modellist[i](x)
        return x

class UNet3D_DDecoder(nn.Module):
    def __init__(self,
                 n_classes,
                 ngf=16,
                 n_block=4):
        super(UNet3D_DDecoder, self).__init__()
        self.modellist = []
        for i in range(n_block-1):
            self.modellist += [up(2**(n_block-i)*ngf*2, 2**(n_block-i-2)*ngf*2)]
        self.modellist += [up(2*ngf*2, ngf*2)]
        self.modellist += [outconv(ngf*2, n_classes)]
        self.modellist += [nn.Softmax(dim=1)]
        self.modellist = nn.ModuleList(self.modellist)
        self.n_block = n_block
    def forward(self, x_list):
        x = self.modellist[0](x_list[-1], x_list[-2])
        for i in range(1, self.n_block):
            x = self.modellist[i](x, x_list[-1*(i+2)])
        for i in range(self.n_block, len(self.modellist)):
            x = self.modellist[i](x)
        # print('x_shape:', x.shape)
        return x

class UNet3D_DDecoder_WithDrop(nn.Module):
    def __init__(self,
                 n_classes,
                 ngf=16,
                 n_block=4):
        super(UNet3D_DDecoder_WithDrop, self).__init__()
        self.modellist = []
        for i in range(n_block-1):
            self.modellist += [up(2**(n_block-i)*ngf*2, 2**(n_block-i-2)*ngf*2)]
        self.modellist += [up(2*ngf*2, ngf*2)]
        self.modellist += [nn.Dropout(0.5)]
        self.modellist += [outconv(ngf*2, n_classes)]
        self.modellist += [nn.Softmax(dim=1)]
        self.modellist = nn.ModuleList(self.modellist)
        self.n_block = n_block
    def forward(self, x_list):
        x = self.modellist[0](x_list[-1], x_list[-2])
        for i in range(1, self.n_block):
            x = self.modellist[i](x, x_list[-1*(i+2)])
        for i in range(self.n_block, len(self.modellist)):
            x = self.modellist[i](x)
        # print('x_shape:', x.shape)
        return x

class ListModule(nn.Module):
    def __init__(self, *args):
        super(ListModule, self).__init__()
        idx = 0
        for module in args:
            self.add_module(str(idx), module)
            idx += 1

    def __getitem__(self, idx):
        if idx < 0 or idx >= len(self._modules):
            raise IndexError('index {} is out of range'.format(idx))
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __iter__(self):
        return iter(self._modules.values())

    def __len__(self):
        return len(self._modules)

class UNet3D_MultiEncoder(nn.Module):
    def __init__(self,
                 n_classes,
                 n_modality=3,
                 n_channels=1,
                 ngf=8,
                 normal_train = False,
                 n_block=4,
                 with_dropout = False,
                 ):
        super(UNet3D_MultiEncoder, self).__init__()
        self.encoderlist = [None]*(n_modality)
        for i in range(len(self.encoderlist)):
            self.encoderlist[i] = UNet3D_EncoderUnit(
                 n_channels=n_channels,
                 ngf=ngf,
                 n_block=n_block)
        self.encoderlist = nn.ModuleList(self.encoderlist)
        self.encoderuni = UNet3D_EncoderUnit(
            n_channels=n_channels,
            ngf=ngf,
            n_block=n_block
        )

        if with_dropout:
            self.decoder = UNet3D_DDecoder_WithDrop(n_classes=n_classes,
                                      ngf=ngf,
                                      n_block=n_block)
        else:
            self.decoder = UNet3D_DDecoder(n_classes=n_classes,
                                      ngf=ngf,
                                      n_block=n_block)

        self.n_modality = n_modality
        self.ex_joint = [None]*n_modality
        self.dx_joint = [None]*n_modality
        self.ex1 = None
        self.ex2 = None
        self.cache_ex1 = True
        self.cache_ex2 = True
        self.value_add_loss_ex1 = None
        self.value_add_loss_ex2 = None
        self.TensorType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.normal_train = normal_train

    def normal_train_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]
        # self.ex1 = ex1
        # self.ex2 = ex2

        # self.ex1 = [torch.stack([ex1[i][j] for i in range(len(ex1))]) for j in range(len(ex1[0]))]
        # self.ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]

        # print(ex1[0][0].shape)
        # print(ex2[0][0].shape)
        ex1_max = [
            torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
        ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        # print(len(ex1_max))
        # print(len(ex1_max[0]))
        # print(type(ex1_max[0]))
        # print(ex1_max[0].shape)
        # print(ex2_max[0].shape)
        ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        # print(ex_max[-1].shape)

        dx_max = self.decoder(ex_max)


        # for i in range(self.n_modality):
        #     self.ex_joint[i] = [torch.cat([ex1[i][j], ex2[i][j]], dim=1) for j in range(len(ex1[i]))]
        #     self.dx_joint[i] = self.decoder(self.ex_joint[i])

        return dx_max

    def train_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        if self.cache_ex1:
            temp_ex1 = [
                    torch.mean(torch.var(
                        torch.stack([ex1[i][j] for i in range(len(ex1))]), dim=0
                    )
                    ) for j in range(len(ex1[0]))

            ]
            self.value_add_loss_ex1 = torch.sum(self.TensorType(temp_ex1))

        if self.cache_ex2:
            temp_ex2 = [
                torch.mean(torch.var(
                    torch.stack([ex2[i][j] for i in range(len(ex2))]), dim=0
                )
                ) for j in range(len(ex2[0]))

            ]
            self.value_add_loss_ex2 = torch.sum(self.TensorType(temp_ex2))

        # temp_ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]



        ex1_max = [
            torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
        ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        # print(len(ex1_max))
        # print(len(ex1_max[0]))
        # print(type(ex1_max[0]))
        # print(ex1_max[0].shape)
        # print(ex2_max[0].shape)
        ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex_max)

        for i in range(self.n_modality):
            self.ex_joint[i] = [torch.cat([ex1[i][j], ex2[i][j]], dim=1) for j in range(len(ex1[i]))]
            self.dx_joint[i] = self.decoder(self.ex_joint[i])

        dx_out = torch.cat(self.dx_joint, dim=1)
        dx_out = torch.cat([dx_max, dx_out], dim=1)

        return dx_out

    def test_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        if self.cache_ex1:
            temp_ex1 = [
                    torch.mean(torch.var(
                        torch.stack([ex1[i][j] for i in range(len(ex1))]), dim=0
                    )
                    ) for j in range(len(ex1[0]))

            ]
            self.value_add_loss_ex1 = torch.sum(self.TensorType(temp_ex1))

        if self.cache_ex2:
            temp_ex2 = [
                torch.mean(torch.var(
                    torch.stack([ex2[i][j] for i in range(len(ex2))]), dim=0
                )
                ) for j in range(len(ex2[0]))

            ]
            self.value_add_loss_ex2 = torch.sum(self.TensorType(temp_ex2))
        # self.ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]

        ex1_max = [torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
                   ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex_max)

        return dx_max

    def predict(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        ex1_max = [torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
                   ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex_max)

        return dx_max

    def forward(self, x):
        if self.training and self.normal_train==False:
            return self.train_forward(x)
        elif self.training and self.normal_train==True:
            return self.normal_train_forward(x)
        else:
            return self.test_forward(x)

class UNet3D_MultiEncoder_v1(nn.Module):
    # encoder with only unified convolutions
    def __init__(self,
                 n_classes,
                 n_modality=3,
                 n_channels=1,
                 ngf=32,
                 normal_train = False,
                 n_block=4,
                 with_dropout = False,
                 ):
        super(UNet3D_MultiEncoder_v1, self).__init__()
        # self.encoderlist = [None]*(n_modality)
        # for i in range(len(self.encoderlist)):
        #     self.encoderlist[i] = UNet3D_EncoderUnit(
        #          n_channels=n_channels,
        #          ngf=ngf,
        #          n_block=n_block)
        # self.encoderlist = nn.ModuleList(self.encoderlist)
        self.encoderuni = UNet3D_EncoderUnit(
            n_channels=n_channels,
            ngf=ngf,
            n_block=n_block
        )

        if with_dropout:
            self.decoder = UNet3D_Decoder_WithDrop(n_classes=n_classes,
                                      ngf=ngf,
                                      n_block=n_block)
        else:
            self.decoder = UNet3D_Decoder(n_classes=n_classes,
                                      ngf=ngf,
                                      n_block=n_block)

        self.n_modality = n_modality
        self.ex_joint = [None]*n_modality
        self.dx_joint = [None]*n_modality
        self.ex1 = None
        self.ex2 = None
        self.cache_ex1 = True
        self.cache_ex2 = True
        self.value_add_loss_ex1 = None
        self.value_add_loss_ex2 = None
        self.TensorType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.normal_train = normal_train

    def normal_train_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        # ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]
        # self.ex1 = ex1
        # self.ex2 = ex2

        # self.ex1 = [torch.stack([ex1[i][j] for i in range(len(ex1))]) for j in range(len(ex1[0]))]
        # self.ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]

        # print(ex1[0][0].shape)
        # print(ex2[0][0].shape)

        # ex1_max = [
        #     torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #            for i in range(len(ex1[0]))
        # ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        # print(len(ex1_max))
        # print(len(ex1_max[0]))
        # print(type(ex1_max[0]))
        # print(ex1_max[0].shape)
        # print(ex2_max[0].shape)
        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        # print(ex_max[-1].shape)

        dx_max = self.decoder(ex2_max)


        # for i in range(self.n_modality):
        #     self.ex_joint[i] = [torch.cat([ex1[i][j], ex2[i][j]], dim=1) for j in range(len(ex1[i]))]
        #     self.dx_joint[i] = self.decoder(self.ex_joint[i])

        return dx_max

    def train_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        # ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        # if self.cache_ex1:
        #     temp_ex1 = [
        #             torch.mean(torch.var(
        #                 torch.stack([ex1[i][j] for i in range(len(ex1))]), dim=0
        #             )
        #             ) for j in range(len(ex1[0]))
        #
        #     ]
        #     self.value_add_loss_ex1 = torch.sum(self.TensorType(temp_ex1))

        if self.cache_ex2:
            temp_ex2 = [
                torch.mean(torch.var(
                    torch.stack([ex2[i][j] for i in range(len(ex2))]), dim=0
                )
                ) for j in range(len(ex2[0]))

            ]
            self.value_add_loss_ex2 = torch.sum(self.TensorType(temp_ex2))

        # temp_ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]



        # ex1_max = [
        #     torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #            for i in range(len(ex1[0]))
        # ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        # print(len(ex1_max))
        # print(len(ex1_max[0]))
        # print(type(ex1_max[0]))
        # print(ex1_max[0].shape)
        # print(ex2_max[0].shape)
        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex2_max)

        for i in range(self.n_modality):
            self.ex_joint[i] = [ex2[i][j] for j in range(len(ex2[i]))]
            self.dx_joint[i] = self.decoder(self.ex_joint[i])

        dx_out = torch.cat(self.dx_joint, dim=1)
        # print(dx_out.shape)
        dx_out = torch.cat([dx_max, dx_out], dim=1)
        # print(dx_out.shape)

        return dx_out

    def test_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        # ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        # if self.cache_ex1:
        #     temp_ex1 = [
        #             torch.mean(torch.var(
        #                 torch.stack([ex1[i][j] for i in range(len(ex1))]), dim=0
        #             )
        #             ) for j in range(len(ex1[0]))
        #
        #     ]
        #     self.value_add_loss_ex1 = torch.sum(self.TensorType(temp_ex1))

        if self.cache_ex2:
            temp_ex2 = [
                torch.mean(torch.var(
                    torch.stack([ex2[i][j] for i in range(len(ex2))]), dim=0
                )
                ) for j in range(len(ex2[0]))

            ]
            self.value_add_loss_ex2 = torch.sum(self.TensorType(temp_ex2))
        # self.ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]
        #
        # ex1_max = [torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #            for i in range(len(ex1[0]))
        #            ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex2_max)

        return dx_max

    def predict(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        # ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        # ex1_max = [torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #            for i in range(len(ex1[0]))
        #            ]

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex2_max)

        return dx_max

    def forward(self, x):
        if self.training and self.normal_train==False:
            return self.train_forward(x)
        elif self.training and self.normal_train==True:
            return self.normal_train_forward(x)
        else:
            return self.test_forward(x)

    def vis_encoder_forward(self, x, block = [1]):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        # ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        # if self.cache_ex1:
        #     temp_ex1 = [
        #             torch.mean(torch.var(
        #                 torch.stack([ex1[i][j] for i in range(len(ex1))]), dim=0
        #             )
        #             ) for j in range(len(ex1[0]))
        #
        #     ]
        #     self.value_add_loss_ex1 = torch.sum(self.TensorType(temp_ex1))

        ex2_max = [
            torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
            for i in range(len(ex2[0]))
        ]

        return [ex2_max[i] for i in block]

class UNet3D_MultiEncoder_v2(nn.Module):
    # encoder with only unified convolutions
    def __init__(self,
                 n_classes,
                 n_modality=3,
                 n_channels=1,
                 ngf=32,
                 normal_train = False,
                 n_block=4,
                 with_dropout = False
                 ):
        super(UNet3D_MultiEncoder_v2, self).__init__()
        self.encoderlist = [None]*(n_modality)
        for i in range(len(self.encoderlist)):
            self.encoderlist[i] = UNet3D_EncoderUnit(
                 n_channels=n_channels,
                 ngf=ngf,
                 n_block=n_block)
        self.encoderlist = nn.ModuleList(self.encoderlist)
        # self.encoderuni = UNet3D_EncoderUnit(
        #     n_channels=n_channels,
        #     ngf=ngf,
        #     n_block=n_block
        # )
        if with_dropout:
            self.decoder = UNet3D_Decoder_WithDrop(n_classes=n_classes,
                                      ngf=ngf,
                                      n_block=n_block)
        else:
            self.decoder = UNet3D_Decoder(n_classes=n_classes,
                                      ngf=ngf,
                                      n_block=n_block)

        self.n_modality = n_modality
        self.ex_joint = [None]*n_modality
        self.dx_joint = [None]*n_modality
        self.ex1 = None
        self.ex2 = None
        self.cache_ex1 = True
        self.cache_ex2 = True
        self.value_add_loss_ex1 = None
        self.value_add_loss_ex2 = None
        self.TensorType = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
        self.normal_train = normal_train

    def normal_train_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        # ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]
        # self.ex1 = ex1
        # self.ex2 = ex2

        # self.ex1 = [torch.stack([ex1[i][j] for i in range(len(ex1))]) for j in range(len(ex1[0]))]
        # self.ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]

        # print(ex1[0][0].shape)
        # print(ex2[0][0].shape)

        ex1_max = [
            torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
        ]

        # ex2_max = [
        #     torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #     for i in range(len(ex2[0]))
        # ]

        # print(len(ex1_max))
        # print(len(ex1_max[0]))
        # print(type(ex1_max[0]))
        # print(ex1_max[0].shape)
        # print(ex2_max[0].shape)
        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        # print(ex_max[-1].shape)

        dx_max = self.decoder(ex1_max)


        # for i in range(self.n_modality):
        #     self.ex_joint[i] = [torch.cat([ex1[i][j], ex2[i][j]], dim=1) for j in range(len(ex1[i]))]
        #     self.dx_joint[i] = self.decoder(self.ex_joint[i])

        return dx_max

    def train_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        # ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        if self.cache_ex1:
            temp_ex1 = [
                    torch.mean(torch.var(
                        torch.stack([ex1[i][j] for i in range(len(ex1))]), dim=0
                    )
                    ) for j in range(len(ex1[0]))

            ]
            self.value_add_loss_ex1 = torch.sum(self.TensorType(temp_ex1))

        ex1_max = [
            torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
        ]

        # ex2_max = [
        #     torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #     for i in range(len(ex2[0]))
        # ]

        # print(len(ex1_max))
        # print(len(ex1_max[0]))
        # print(type(ex1_max[0]))
        # print(ex1_max[0].shape)
        # print(ex2_max[0].shape)
        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex1_max)

        for i in range(self.n_modality):
            self.ex_joint[i] = [ex1[i][j] for j in range(len(ex1[i]))]
            self.dx_joint[i] = self.decoder(self.ex_joint[i])

        dx_out = torch.cat(self.dx_joint, dim=1)
        dx_out = torch.cat([dx_max, dx_out], dim=1)

        return dx_out

    def test_forward(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        # ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        if self.cache_ex1:
            temp_ex1 = [
                    torch.mean(torch.var(
                        torch.stack([ex1[i][j] for i in range(len(ex1))]), dim=0
                    )
                    ) for j in range(len(ex1[0]))

            ]
            self.value_add_loss_ex1 = torch.sum(self.TensorType(temp_ex1))

        # if self.cache_ex2:
        #     temp_ex2 = [
        #         torch.mean(torch.var(
        #             torch.stack([ex2[i][j] for i in range(len(ex2))]), dim=0
        #         )
        #         ) for j in range(len(ex2[0]))
        #
        #     ]
        #     self.value_add_loss_ex2 = torch.sum(self.TensorType(temp_ex2))
        # self.ex2 = [torch.stack([ex2[i][j] for i in range(len(ex2))]) for j in range(len(ex2[0]))]
        #
        ex1_max = [torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
                   ]

        # ex2_max = [
        #     torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #     for i in range(len(ex2[0]))
        # ]

        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex1_max)

        return dx_max

    def predict(self, x):
        x = torch.split(x, split_size_or_sections=1, dim=1)
        ex1 = [encoder(x[i]) for i, encoder in enumerate(self.encoderlist)]
        # ex2 = [self.encoderuni(x[i]) for i in range(self.n_modality)]

        ex1_max = [torch.max(torch.stack([ex1[j][i] for j in range(self.n_modality)]), dim=0)[0]
                   for i in range(len(ex1[0]))
                   ]

        # ex2_max = [
        #     torch.max(torch.stack([ex2[j][i] for j in range(self.n_modality)]), dim=0)[0]
        #     for i in range(len(ex2[0]))
        # ]

        # ex_max = [torch.cat(i, dim=1) for i in zip(ex1_max, ex2_max)]
        dx_max = self.decoder(ex1_max)

        return dx_max

    def forward(self, x):
        if self.training and self.normal_train==False:
            return self.train_forward(x)
        elif self.training and self.normal_train==True:
            return self.normal_train_forward(x)
        else:
            return self.test_forward(x)

class UNet3D(nn.Module):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False):
        super(UNet3D, self).__init__()
        self.with_dropout = with_dropout
        self.inc = inconv(n_channels, ngf)
        self.down1 = down(ngf, ngf*2, pool=nn.AvgPool3d(2))
        self.down2 = down(ngf*2, ngf*4, pool=nn.AvgPool3d(2))
        self.down3 = down(ngf*4, ngf*8, pool=nn.AvgPool3d(2))
        self.down4 = down(ngf*8, ngf*8, pool=nn.AvgPool3d(2))
        self.up1 = up(ngf*16, ngf*4)
        self.up2 = up(ngf*8, ngf*2)
        self.up3 = up(ngf*4, ngf)
        self.up4 = up(ngf*2, ngf)
        self.outc = outconv(ngf, num_classes)
        self.soft = nn.Softmax(dim=1)

        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = self.soft(x)
        return x

    def forward_dsup(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x_u1 = self.up1(x5, x4)
        x_u2 = self.up2(x_u1, x3)
        x_u3 = self.up3(x_u2, x2)
        x = self.up4(x_u3, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        # x = self.soft(x)
        return x, x_u3, x_u2, x_u1

    def vis_encoder_forward(self, x, block = 1):
        x1 = self.inc(x)
        if block == 0:
            return x1
        x2 = self.down1(x1)
        if block == 1:
            return x2
        x3 = self.down2(x2)
        if block == 2:
            return x3
        x4 = self.down3(x3)
        if block == 3:
            return x4
        x5 = self.down4(x4)
        if block == 4:
            return x5


class UNet3D_dsupwrapper(nn.Module):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False,
                 output_mode = 0):
        super(UNet3D_dsupwrapper, self).__init__()
        self.unet = UNet3D(
            num_classes=num_classes,
            n_channels=n_channels,
            ngf=ngf,
            with_dropout=with_dropout
        )
        self.core_model_name = 'unet3D'

        self.soft = nn.Softmax(dim=1)

        self.outc_3 = outconv(ngf, num_classes)
        self.up_3 = nn.Upsample(scale_factor=2, mode='trilinear',align_corners=False)
        self.outc_2 = outconv(ngf*2, num_classes)
        self.up_2 = nn.Upsample(scale_factor=4, mode='trilinear',align_corners=False)
        self.outc_1 = outconv(ngf*4, num_classes)
        self.up_1 = nn.Upsample(scale_factor=8, mode='trilinear',align_corners=False)

        self.out_mode = output_mode

        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def forward_normal(self, x):
        x, x_u3, x_u2, x_u1 = self.unet.forward_dsup(x)
        out_3 = self.up_3(self.outc_3(x_u3))
        out_2 = self.up_2(self.outc_2(x_u2))
        out_1 = self.up_1(self.outc_1(x_u1))
        # print(x.shape)
        # print(out_3.shape)
        # print(out_2.shape)
        # print(out_1.shape)
        x = 0.575 * x + 0.25 * out_3 + 0.125 * out_2 + 0.075 * out_1
        x = self.soft(x)
        return x

    def forward_sep(self, x):
        x, x_u3, x_u2, x_u1 = self.unet.forward_dsup(x)
        out_3 = self.soft(self.outc_3(x_u3))
        out_2 = self.soft(self.outc_2(x_u2))
        out_1 = self.soft(self.outc_1(x_u1))
        x = self.soft(x)
        return x, out_3, out_2, out_1

    def forward(self, x):
        if self.out_mode == 0:
            return self.forward_normal(x)
        else:
            return self.forward_sep(x)

class UNet3D_SA_dsupwrapper(UNet3D_dsupwrapper):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout=False, output_mode = 0):
        super(UNet3D_SA_dsupwrapper, self).__init__(
            num_classes=num_classes,
            n_channels=n_channels,
            ngf=ngf,
            with_dropout=with_dropout, output_mode=output_mode
        )
        self.unet = UNet3D_SpatialAttention(
            num_classes=num_classes,
            n_channels=n_channels,
            ngf=ngf,
            with_dropout=with_dropout
        )
        self.core_model_name = 'unet3D_attention'

class UNet3D_DSup(nn.Module):
    """Deeply supervised """
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False,
                 output_mode = 0
                 ):
        super(UNet3D_DSup, self).__init__()
        self.with_dropout = with_dropout
        self.inc = inconv(n_channels, ngf)
        self.down1 = down(ngf, ngf*2, pool=nn.AvgPool3d(2))
        self.down2 = down(ngf*2, ngf*4, pool=nn.AvgPool3d(2))
        self.down3 = down(ngf*4, ngf*8, pool=nn.AvgPool3d(2))
        self.down4 = down(ngf*8, ngf*8, pool=nn.AvgPool3d(2))
        self.up1 = up(ngf*16, ngf*4)
        self.up2 = up(ngf*8, ngf*2)
        self.up3 = up(ngf*4, ngf)
        self.up4 = up(ngf*2, ngf)
        self.outc = outconv(ngf, num_classes)
        self.soft = nn.Softmax(dim=1)
        
        # deeply supervised out conv
        # self.outc_4 = outconv(ngf, num_classes)
        self.outc_3 = outconv(ngf, num_classes)
        self.outc_2 = outconv(ngf*2, num_classes)
        self.outc_1 = outconv(ngf*4, num_classes)

        self.out_mode = output_mode

        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def forward_normal(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        out_1 = self.outc_1(x)
        x = self.up2(x, x3)
        out_2 = self.outc_2(x)
        x = self.up3(x, x2)
        out_3 = self.outc_3(x)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = 0.575*x + 0.25*out_3 + 0.125*out_2 + 0.075*out_1
        x = self.soft(x)

        return x

    def forward_sep(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)

        out_1 = self.soft(self.outc_1(x))

        x = self.up2(x, x3)

        out_2 = self.soft(self.outc_2(x))

        x = self.up3(x, x2)

        out_3 = self.soft(self.outc_3(x))

        x = self.up4(x, x1)

        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        # x = 0.575*x + 0.25*out_3 + 0.125*out_2 + 0.075*out_1
        x = self.soft(x)

        return x, out_3, out_2, out_1

    def forward(self, x):
        if self.out_mode == 0:
            return self.forward_normal(x)
        else:
            return self.forward_sep(x)

    def vis_encoder_forward(self, x, block = 1):
        x1 = self.inc(x)
        if block == 0:
            return x1
        x2 = self.down1(x1)
        if block == 1:
            return x2
        x3 = self.down2(x2)
        if block == 2:
            return x3
        x4 = self.down3(x3)
        if block == 3:
            return x4
        x5 = self.down4(x4)
        if block == 4:
            return x5

class UNet3D_AttentionZoom(nn.Module):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False):
        super(UNet3D_AttentionZoom, self).__init__()
        self.with_dropout = with_dropout
        self.num_classes = num_classes
        self.inc = inconv(n_channels, ngf)
        self.down1 = down(ngf, ngf*2, pool=nn.AvgPool3d(2))
        self.down2 = down(ngf*2, ngf*4, pool=nn.AvgPool3d(2))
        self.down3 = down(ngf*4, ngf*8, pool=nn.AvgPool3d(2))
        self.down4 = down(ngf*8, ngf*8, pool=nn.AvgPool3d(2))
        self.up1 = up(ngf*16, ngf*4)
        self.up2 = up(ngf*8, ngf*2)
        self.up3 = up(ngf*4, ngf)
        self.up4 = up(ngf*2, ngf)
        self.outc = outconv(ngf, num_classes)
        self.soft = nn.Softmax(dim=1)

        # detect brantch 1D conv
        self.inloc = c1d_attention(ngf, fix_size=128)
        self.loc1 = c1d_m(ngf*2, in_ch2=1, fix_size=64)
        self.loc2 = c1d_m(ngf*4, in_ch2=1, fix_size=32)
        self.loc3 = c1d_m(ngf*8, in_ch2=1, fix_size=16)
        self.loc4 = c1d_m(ngf*8, in_ch2=1, fix_size=8)
        self.loc_out = c1d_final(
            # 3*ngf*(128+64+32+16+8),
            fix_size=8
        )


        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def return_coords(self, x, l, ol):
        l = torch.Tensor([l]).to(x.device)
        ol = torch.Tensor([ol]).to(x.device)
        ox = (x+1)/2 * (ol-1)
        x = (x+1)/2 * (l-1)

        return ox, torch.floor(x).int(), torch.ceil(x).int()

    def batch_crop(self, x, xy_l, xy_u, xx_l, xx_u, xz_l, xz_u):
        x = torch.stack(
            [
                xi[..., xy_l[i]:xy_u[i], xx_l[i]:xx_u[i], xz_l[i]:xz_u[i]] for i, xi in enumerate(x)
            ], dim=0
        )
        return x

    def forward_train(self, x):
        x1 = self.inc(x)
        xy, xx, xz = self.inloc(x1)
        print('1: ', xy)
        x2 = self.down1(x1)
        xy, xx, xz = self.loc1(x2, xy, xx, xz)
        print('2: ', xy)
        x3 = self.down2(x2)
        xy, xx, xz = self.loc2(x3, xy, xx, xz)
        print('3: ', xy)
        x4 = self.down3(x3)
        xy, xx, xz = self.loc3(x4, xy, xx, xz)
        print('4: ', xy)
        x5 = self.down4(x4)
        xy, xx, xz = self.loc4(x5, xy, xx, xz)
        print('5: ', xy)
        xy, xx, xz = self.loc_out(xy, xx, xz)
        print('6: ', xy)

        xy, xy_l, xy_u = self.return_coords(xy, x5.shape[-3], x.shape[-3])
        xx, xx_l, xx_u = self.return_coords(xx, x5.shape[-2], x.shape[-2])
        xz, xz_l, xz_u = self.return_coords(xz, x5.shape[-1], x.shape[-1])
        # print(xy, xy_l * 16, (xy_u + 1) * 16)

        # crop the volume
        # x5 = x5[..., xy_l:xy_u + 1, xx_l:xx_u + 1, xz_l:xz_u + 1]
        # x4 = x4[..., xy_l * 2:(xy_u+1) * 2, xx_l * 2:(xx_u+1) * 2, xz_l * 2:(xz_u+1) * 2]
        # x3 = x3[..., xy_l * 4:(xy_u+1) * 4, xx_l * 4:(xx_u+1) * 4, xz_l * 4:(xz_u+1) * 4]
        # x2 = x2[..., xy_l * 8:(xy_u+1) * 8, xx_l * 8:(xx_u+1) * 8, xz_l * 8:(xz_u+1) * 8]
        # x1 = x1[..., xy_l * 16:(xy_u+1) * 16, xx_l * 16:(xx_u+1) * 16, xz_l * 16:(xz_u+1) * 16]

        x5 = self.batch_crop(x5,
                             xy_l,
                             xy_u + 1,
                             xx_l,
                             xx_u + 1,
                             xz_l,
                             xz_u + 1)

        x4 = self.batch_crop(x4,
                             xy_l * 2,
                             (xy_u + 1) * 2,
                             xx_l * 2,
                             (xx_u + 1) * 2,
                             xz_l * 2,
                             (xz_u + 1) * 2)

        x3 = self.batch_crop(x3,
                             xy_l * 4,
                             (xy_u + 1) * 4,
                             xx_l * 4,
                             (xx_u + 1) * 4,
                             xz_l * 4,
                             (xz_u + 1) * 4)

        x2 = self.batch_crop(x2,
                             xy_l * 8,
                             (xy_u + 1) * 8,
                             xx_l * 8,
                             (xx_u + 1) * 8,
                             xz_l * 8,
                             (xz_u + 1) * 8)

        x1 = self.batch_crop(x1,
                             xy_l * 16,
                             (xy_u + 1) * 16,
                             xx_l * 16,
                             (xx_u + 1) * 16,
                             xz_l * 16,
                             (xz_u + 1) * 16)

        # print(x5.shape)
        # print(x4.shape)
        # print(x.shape)

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        if self.with_dropout:
            xo = self.drop(xo)

        xo = self.outc(xo)
        xo = self.soft(xo)

        x_out = torch.min(xo.view(xo.shape[0], -1), dim=-1)[0].view(-1, 1, 1, 1, 1) * \
                torch.ones(
                    x.shape[0],
                    self.num_classes,
                    x.shape[2],
                    x.shape[3],
                    x.shape[4],
                    device=x.device
                )

        x_out[:, -1, :, :, :] -= 1e-14

        for i, x_outi in enumerate(x_out):

            x_out[i,
            ...,
            xy_l[i] * 16:(xy_u[i]+1) * 16,
            xx_l[i] * 16:(xx_u[i]+1) * 16,
            xz_l[i] * 16:(xz_u[i]+1) * 16] = xo[i]

        return x_out, torch.cat([xy.unsqueeze(1), xx.unsqueeze(1), xz.unsqueeze(1)], dim=-1)

    def forward_test(self, x):
        x1 = self.inc(x)
        xy, xx, xz = self.inloc(x1)
        # print('1: ', xy)
        x2 = self.down1(x1)
        xy, xx, xz = self.loc1(x2, xy, xx, xz)
        # print('2: ', xy)
        x3 = self.down2(x2)
        xy, xx, xz = self.loc2(x3, xy, xx, xz)
        # print('3: ', xy)
        x4 = self.down3(x3)
        xy, xx, xz = self.loc3(x4, xy, xx, xz)
        # print('4: ', xy)
        x5 = self.down4(x4)
        xy, xx, xz = self.loc4(x5, xy, xx, xz)
        # print('5: ', xy)
        xy, xx, xz = self.loc_out(xy, xx, xz)
        # print('6: ', xy)

        xy, xy_l, xy_u = self.return_coords(xy, x5.shape[-3], x.shape[-3])
        xx, xx_l, xx_u = self.return_coords(xx, x5.shape[-2], x.shape[-2])
        xz, xz_l, xz_u = self.return_coords(xz, x5.shape[-1], x.shape[-1])
        # print(xy, xy_l * 16, (xy_u + 1) * 16)

        # crop the volume
        # x5 = x5[..., xy_l:xy_u + 1, xx_l:xx_u + 1, xz_l:xz_u + 1]
        # x4 = x4[..., xy_l * 2:(xy_u+1) * 2, xx_l * 2:(xx_u+1) * 2, xz_l * 2:(xz_u+1) * 2]
        # x3 = x3[..., xy_l * 4:(xy_u+1) * 4, xx_l * 4:(xx_u+1) * 4, xz_l * 4:(xz_u+1) * 4]
        # x2 = x2[..., xy_l * 8:(xy_u+1) * 8, xx_l * 8:(xx_u+1) * 8, xz_l * 8:(xz_u+1) * 8]
        # x1 = x1[..., xy_l * 16:(xy_u+1) * 16, xx_l * 16:(xx_u+1) * 16, xz_l * 16:(xz_u+1) * 16]

        x5 = self.batch_crop(x5,
                             xy_l,
                             xy_u + 1,
                             xx_l,
                             xx_u + 1,
                             xz_l,
                             xz_u + 1)

        x4 = self.batch_crop(x4,
                             xy_l * 2,
                             (xy_u + 1) * 2,
                             xx_l * 2,
                             (xx_u + 1) * 2,
                             xz_l * 2,
                             (xz_u + 1) * 2)

        x3 = self.batch_crop(x3,
                             xy_l * 4,
                             (xy_u + 1) * 4,
                             xx_l * 4,
                             (xx_u + 1) * 4,
                             xz_l * 4,
                             (xz_u + 1) * 4)

        x2 = self.batch_crop(x2,
                             xy_l * 8,
                             (xy_u + 1) * 8,
                             xx_l * 8,
                             (xx_u + 1) * 8,
                             xz_l * 8,
                             (xz_u + 1) * 8)

        x1 = self.batch_crop(x1,
                             xy_l * 16,
                             (xy_u + 1) * 16,
                             xx_l * 16,
                             (xx_u + 1) * 16,
                             xz_l * 16,
                             (xz_u + 1) * 16)

        # print(x5.shape)
        # print(x4.shape)
        # print(x.shape)

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        if self.with_dropout:
            xo = self.drop(xo)

        xo = self.outc(xo)
        xo = self.soft(xo)
        x_out = torch.zeros(x.shape[0], self.num_classes, x.shape[2], x.shape[3], x.shape[4], device=x.device)
        for i, x_outi in enumerate(x_out):
            x_out[i,
            ...,
            xy_l[i] * 16:(xy_u[i]+1) * 16,
            xx_l[i] * 16:(xx_u[i]+1) * 16,
            xz_l[i] * 16:(xz_u[i]+1) * 16] = xo[i]

        return x_out

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    def vis_encoder_forward(self, x, block = 1):
        x1 = self.inc(x)
        if block == 0:
            return x1
        x2 = self.down1(x1)
        if block == 1:
            return x2
        x3 = self.down2(x2)
        if block == 2:
            return x3
        x4 = self.down3(x3)
        if block == 3:
            return x4
        x5 = self.down4(x4)
        if block == 4:
            return x5

def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes,
                     out_planes,
                     kernel_size=1,
                     stride=stride,
                     bias=False)

class UNet3D_SpatialAttention_bk(nn.Module):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False):
        super(UNet3D_SpatialAttention_bk, self).__init__()
        self.with_dropout = with_dropout
        self.inc = inconv(n_channels, ngf)
        self.down1 = down(ngf, ngf*2)
        self.down2 = down(ngf*2, ngf*4)
        self.down3 = down(ngf*4, ngf*8)
        self.down4 = down(ngf*8, ngf*8)

        self.avgpool = nn.AdaptiveAvgPool3d(
            (6, 6, 6)
        )
        self.at_key = conv1x1x1(ngf*8, 6*6*6)
        self.at_que = conv1x1x1(ngf*8, 6*6*6)
        self.at_val = conv1x1x1(ngf*8, ngf*8)

        self.up1 = up(ngf*16, ngf*4)
        self.up2 = up(ngf*8, ngf*2)
        self.up3 = up(ngf*4, ngf)
        self.up4 = up(ngf*2, ngf)
        self.outc = outconv(ngf, num_classes)
        self.soft = nn.Softmax(dim=1)

        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        print(x5.shape)
        print(xa.shape)
        print(xa_k.shape)
        xa_k = xa_k.view(xa.shape[0], 8*8*8, -1).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], 8*8*8, -1)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], xa.shape[1], -1).transpose(1, 2)
        scale = torch.sqrt(xa_k.view(xa_k.shape[0], -1).shape[-1])

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/scale, dim=-1), xa_r)
        xa_at = xa_at.transpose(1, 2).view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        # xa_at = self.out_model(xa_at)

        x5 = x5 + xa_at

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = self.soft(x)
        return x

    def vis_encoder_forward(self, x, block = 1):
        x1 = self.inc(x)
        if block == 0:
            return x1
        x2 = self.down1(x1)
        if block == 1:
            return x2
        x3 = self.down2(x2)
        if block == 2:
            return x3
        x4 = self.down3(x3)
        if block == 3:
            return x4
        x5 = self.down4(x4)
        if block == 4:
            return x5

class UNet3D_SpatialAttention(nn.Module):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False):
        super(UNet3D_SpatialAttention, self).__init__()
        self.with_dropout = with_dropout
        self.inc = inconv(n_channels, ngf)
        self.down1 = down(ngf, ngf*2)
        self.down2 = down(ngf*2, ngf*4)
        self.down3 = down(ngf*4, ngf*8)
        self.down4 = down(ngf*8, ngf*8)

        # self.avgpool = nn.AdaptiveAvgPool3d(
        #     (8, 8, 8)
        # )
        self.avgpool = nn.Upsample(size=[8,8,8], mode='trilinear')

        self.at_key = conv1x1x1(ngf*8, ngf*4)
        self.at_que = conv1x1x1(ngf*8, ngf*4)
        self.at_val = conv1x1x1(ngf*8, ngf*4)
        self.at_out = conv1x1x1(ngf*4, ngf*8)

        self.up1 = up(ngf*16, ngf*4)
        self.up2 = up(ngf*8, ngf*2)
        self.up3 = up(ngf*4, ngf)
        self.up4 = up(ngf*2, ngf)
        self.outc = outconv(ngf, num_classes)
        self.soft = nn.Softmax(dim=1)

        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6*6*6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        scale = torch.sqrt(torch.Tensor([6*6*6])).item()

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        x5 = x5 + xa_at

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = self.soft(x)
        return x

    def forward_dsup(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 8*8*8).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 8*8*8)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 8*8*8).transpose(1, 2)
        scale = torch.sqrt(torch.Tensor([8*8*8])).item()

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        x5 = x5 + xa_at

        x_u1 = self.up1(x5, x4)
        x_u2 = self.up2(x_u1, x3)
        x_u3 = self.up3(x_u2, x2)
        x = self.up4(x_u3, x1)

        # x = self.up1(x5, x4)
        # x = self.up2(x, x3)
        # x = self.up3(x, x2)
        # x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        # x = self.soft(x)
        return x, x_u3, x_u2, x_u1

    def vis_encoder_forward(self, x, block = 1):
        x1 = self.inc(x)
        if block == 0:
            return x1
        x2 = self.down1(x1)
        if block == 1:
            return x2
        x3 = self.down2(x2)
        if block == 2:
            return x3
        x4 = self.down3(x3)
        if block == 3:
            return x4
        x5 = self.down4(x4)
        if block == 4:
            return x5

class UNet3D_CBAMAttentionZoom(nn.Module):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False, fixSize = 32):
        super(UNet3D_CBAMAttentionZoom, self).__init__()
        self.with_dropout = with_dropout
        self.num_classes = num_classes
        self.inc = inconv(n_channels, ngf)
        self.down1 = down_CBAMFixScale(ngf, ngf*2, fixSize=fixSize, pool=nn.AvgPool3d(2))
        self.down2 = down_CBAMFixScale(ngf*2, ngf*4, fixSize=fixSize, pool=nn.AvgPool3d(2))
        self.down3 = down_CBAMFixScale(ngf*4, ngf*8, fixSize=fixSize, pool=nn.AvgPool3d(2))
        self.down4 = down_CBAMFixScale(ngf*8, ngf*8, fixSize=fixSize, pool=nn.AvgPool3d(2))

        self.avgpool = nn.AdaptiveAvgPool3d(
            (6, 6, 6)
        )
        self.at_scale = torch.sqrt(torch.Tensor([6 * 6 * 6])).item()

        self.at_key = conv1x1x1(ngf*8, ngf*4)
        self.at_que = conv1x1x1(ngf*8, ngf*4)
        self.at_val = conv1x1x1(ngf*8, ngf*4)
        self.at_out = conv1x1x1(ngf*4, ngf*8)

        self.at_fixscale = SpatialGateFixScale(fixSize=fixSize)
        self.fixSize = fixSize
        if self.fixSize is not None:
            self.fixSize3 = fixSize*fixSize*fixSize
            self.grid = torch.linspace(-1, 1, self.fixSize)
        else:
            self.fixSize3 = None
            self.grid = None
            self.gridy = self.gridx = self.gridz = None
        # self.fat_scale = torch.sqrt(torch.Tensor([self.fixSize3])).item()
        # self.fat_key = conv1x1x1(5, 1)
        # self.fat_que = conv1x1x1(5, 1)
        # self.fat_val = conv1x1x1(5, 1)

        self.fat_fuse = nn.Sequential(
            nn.Conv3d(5, 5, kernel_size=3, bias=False, padding=1),
            nn.LeakyReLU(inplace=True),
            nn.InstanceNorm3d(5),
            nn.Conv3d(5, 1, kernel_size=1, bias=False)
        )



        self.up1 = up_CBAM(ngf*16, ngf*4)
        self.up2 = up_CBAM(ngf*8, ngf*2)
        self.up3 = up_CBAM(ngf*4, ngf)
        self.up4 = up_CBAM(ngf*2, ngf)
        self.outc = outconv(ngf, num_classes)
        self.soft = nn.Softmax(dim=1)


        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def return_coords(self, x, l, ol):
        l = torch.Tensor([l]).to(x.device)
        ol = torch.Tensor([ol]).to(x.device)
        ox = (x+1)/2 * (ol-1)
        x = (x+1)/2 * (l-1)
        x_l = torch.floor(x).int()
        x_u = x_l + 1
        return ox, x_l, x_u

    def batch_crop(self, x, xy_l, xy_u, xx_l, xx_u, xz_l, xz_u):
        x = torch.stack(
            [
                xi[..., xy_l[i]:xy_u[i], xx_l[i]:xx_u[i], xz_l[i]:xz_u[i]] for i, xi in enumerate(x)
            ], dim=0
        )
        return x

    def forward(self, x):
        if self.training:
            return self.forward_train(x)
        else:
            return self.forward_test(x)

    def forward_test(self, x):
        x_out, _ = self.forward_train(x)
        return x_out

    def forward_train(self, x):
        # self.at_scale = self.at_scale.to(x.device)
        # self.fat_scale = self.fat_scale.to(x.device)
        self.grid = self.grid.to(x.device)
        x1 = self.inc(x)
        x2, s2 = self.down1(x1)
        x3, s3 = self.down2(x2)
        x4, s4 = self.down3(x3)
        x5, s5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6*6*6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/self.at_scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        s_at = self.at_fixscale.get_scale(xa_at)
        x5 = x5 + xa_at


        # generate prob map
        s = torch.cat(
            [s2, s3, s4, s5, s_at],
            dim=1
        )

        # s_at_k = self.fat_key(s)
        # s_at_k = s_at_k.view(s_at_k.shape[0], -1, self.fixSize3).transpose(1, 2)
        # s_at_q = self.fat_que(s)
        # s_at_q = s_at_q.view(s_at_q.shape[0], -1, self.fixSize3)
        # s_at_v = self.fat_val(s)
        # s_at_v = s_at_v.view(s_at_v.shape[0], -1, self.fixSize3).transpose(1, 2)
        # # s_at_v = F.softmax(F.instance_norm(s_at_v), dim=1)
        #
        # s_at = torch.bmm(
        #     F.softmax(torch.bmm(s_at_k, s_at_q) / self.fat_scale, dim=-1),
        #     s_at_v
        # )

        s_at = self.fat_fuse(s)
        s_at = F.softmax(s_at.view(s.shape[0], 1, -1), dim=-1)
        s_at = s_at.view(s.shape[0], -1, s.shape[2], s.shape[3], s.shape[4])

        # get estimated coordinates
        xy = torch.sum(
            self.grid.view(1, 1, -1, 1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xx = torch.sum(
            self.grid.view(1, 1, 1, -1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xz = torch.sum(
            self.grid.view(1, 1, 1, 1, -1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xy, xy_l, xy_u = self.return_coords(xy, x5.shape[-3], x.shape[-3])
        xx, xx_l, xx_u = self.return_coords(xx, x5.shape[-2], x.shape[-2])
        xz, xz_l, xz_u = self.return_coords(xz, x5.shape[-1], x.shape[-1])
        # print(xy, xy_l * 16, (xy_u + 1) * 16)

        # crop the volume
        # x5 = x5[..., xy_l:xy_u + 1, xx_l:xx_u + 1, xz_l:xz_u + 1]
        # x4 = x4[..., xy_l * 2:(xy_u+1) * 2, xx_l * 2:(xx_u+1) * 2, xz_l * 2:(xz_u+1) * 2]
        # x3 = x3[..., xy_l * 4:(xy_u+1) * 4, xx_l * 4:(xx_u+1) * 4, xz_l * 4:(xz_u+1) * 4]
        # x2 = x2[..., xy_l * 8:(xy_u+1) * 8, xx_l * 8:(xx_u+1) * 8, xz_l * 8:(xz_u+1) * 8]
        # x1 = x1[..., xy_l * 16:(xy_u+1) * 16, xx_l * 16:(xx_u+1) * 16, xz_l * 16:(xz_u+1) * 16]

        x5 = self.batch_crop(x5,
                             xy_l,
                             xy_u + 1,
                             xx_l,
                             xx_u + 1,
                             xz_l,
                             xz_u + 1)

        x4 = self.batch_crop(x4,
                             xy_l * 2,
                             (xy_u + 1) * 2,
                             xx_l * 2,
                             (xx_u + 1) * 2,
                             xz_l * 2,
                             (xz_u + 1) * 2)

        x3 = self.batch_crop(x3,
                             xy_l * 4,
                             (xy_u + 1) * 4,
                             xx_l * 4,
                             (xx_u + 1) * 4,
                             xz_l * 4,
                             (xz_u + 1) * 4)

        x2 = self.batch_crop(x2,
                             xy_l * 8,
                             (xy_u + 1) * 8,
                             xx_l * 8,
                             (xx_u + 1) * 8,
                             xz_l * 8,
                             (xz_u + 1) * 8)

        x1 = self.batch_crop(x1,
                             xy_l * 16,
                             (xy_u + 1) * 16,
                             xx_l * 16,
                             (xx_u + 1) * 16,
                             xz_l * 16,
                             (xz_u + 1) * 16)

        # print(x5.shape)
        # print(x4.shape)
        # print(x.shape)

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        if self.with_dropout:
            xo = self.drop(xo)

        xo = self.outc(xo)
        xo = self.soft(xo)

        x_out = torch.min(xo.view(xo.shape[0], -1), dim=-1)[0].view(-1, 1, 1, 1, 1) * \
                torch.ones(
                    x.shape[0],
                    self.num_classes,
                    x.shape[2],
                    x.shape[3],
                    x.shape[4],
                    device=x.device
                )

        x_out[:, -1, :, :, :] = 0
        x_out[:, 0,  :, :, :] = 1

        for i, x_outi in enumerate(x_out):
            x_out[i,
            ...,
            xy_l[i] * 16:(xy_u[i] + 1) * 16,
            xx_l[i] * 16:(xx_u[i] + 1) * 16,
            xz_l[i] * 16:(xz_u[i] + 1) * 16] = xo[i]

        return x_out, torch.cat([xy.unsqueeze(1), xx.unsqueeze(1), xz.unsqueeze(1)], dim=-1)

    def forward_train_DSup(self, x):
        assert self.fixSize is None
        assert self.grid is None
        # self.at_scale = self.at_scale.to(x.device)
        # self.fat_scale = self.fat_scale.to(x.device)
        # self.grid = self.grid.to(x.device)
        x1 = self.inc(x)
        x2, s2 = self.down1(x1)
        x3, s3 = self.down2(x2)
        x4, s4 = self.down3(x3)
        x5, s5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6*6*6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/self.at_scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        s_at = self.at_fixscale.get_scale(xa_at)
        x5 = x5 + xa_at

        s_at = F.softmax(s_at.view(s_at.shape[0], 1, -1), dim=-1)
        s_at = s_at.view(s_at.shape[0], -1, x5.shape[2], x5.shape[3], x5.shape[4])

        if self.gridy is None or len(self.gridy) != x5.shape[-3]:
            self.gridy = torch.linspace(-1, 1, x5.shape[-3], device=x.device)
        # get estimated coordinates
        xy = torch.sum(
            self.gridy.view(1, 1, -1, 1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )
        if self.gridx is None or len(self.gridx) != x5.shape[-2]:
            self.gridx = torch.linspace(-1, 1, x5.shape[-2], device=x.device)

        xx = torch.sum(
            self.gridx.view(1, 1, 1, -1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        if self.gridz is None or len(self.gridz) != x5.shape[-3]:
            self.gridz = torch.linspace(-1, 1, x5.shape[-1], device=x.device)

        xz = torch.sum(
            self.gridz.view(1, 1, 1, 1, -1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xy, xy_l, xy_u = self.return_coords(xy, x5.shape[-3], x.shape[-3])
        xx, xx_l, xx_u = self.return_coords(xx, x5.shape[-2], x.shape[-2])
        xz, xz_l, xz_u = self.return_coords(xz, x5.shape[-1], x.shape[-1])
        # print(xy, xy_l * 16, (xy_u + 1) * 16)

        # crop the volume
        # x5 = x5[..., xy_l:xy_u + 1, xx_l:xx_u + 1, xz_l:xz_u + 1]
        # x4 = x4[..., xy_l * 2:(xy_u+1) * 2, xx_l * 2:(xx_u+1) * 2, xz_l * 2:(xz_u+1) * 2]
        # x3 = x3[..., xy_l * 4:(xy_u+1) * 4, xx_l * 4:(xx_u+1) * 4, xz_l * 4:(xz_u+1) * 4]
        # x2 = x2[..., xy_l * 8:(xy_u+1) * 8, xx_l * 8:(xx_u+1) * 8, xz_l * 8:(xz_u+1) * 8]
        # x1 = x1[..., xy_l * 16:(xy_u+1) * 16, xx_l * 16:(xx_u+1) * 16, xz_l * 16:(xz_u+1) * 16]

        x5 = self.batch_crop(x5,
                             xy_l,
                             xy_u + 1,
                             xx_l,
                             xx_u + 1,
                             xz_l,
                             xz_u + 1)

        x4 = self.batch_crop(x4,
                             xy_l * 2,
                             (xy_u + 1) * 2,
                             xx_l * 2,
                             (xx_u + 1) * 2,
                             xz_l * 2,
                             (xz_u + 1) * 2)

        x3 = self.batch_crop(x3,
                             xy_l * 4,
                             (xy_u + 1) * 4,
                             xx_l * 4,
                             (xx_u + 1) * 4,
                             xz_l * 4,
                             (xz_u + 1) * 4)

        x2 = self.batch_crop(x2,
                             xy_l * 8,
                             (xy_u + 1) * 8,
                             xx_l * 8,
                             (xx_u + 1) * 8,
                             xz_l * 8,
                             (xz_u + 1) * 8)

        x1 = self.batch_crop(x1,
                             xy_l * 16,
                             (xy_u + 1) * 16,
                             xx_l * 16,
                             (xx_u + 1) * 16,
                             xz_l * 16,
                             (xz_u + 1) * 16)

        # print(x5.shape)
        # print(x4.shape)
        # print(x.shape)

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        if self.with_dropout:
            xo = self.drop(xo)

        xo = self.outc(xo)
        xo = self.soft(xo)

        x_out = torch.min(xo.view(xo.shape[0], -1), dim=-1)[0].view(-1, 1, 1, 1, 1) * \
                torch.ones(
                    x.shape[0],
                    self.num_classes,
                    x.shape[2],
                    x.shape[3],
                    x.shape[4],
                    device=x.device
                )

        x_out[:, -1, :, :, :] = 0
        x_out[:, 0,  :, :, :] = 1

        for i, x_outi in enumerate(x_out):
            x_out[i,
            ...,
            xy_l[i] * 16:(xy_u[i] + 1) * 16,
            xx_l[i] * 16:(xx_u[i] + 1) * 16,
            xz_l[i] * 16:(xz_u[i] + 1) * 16] = xo[i]

        return x_out, torch.cat([xy.unsqueeze(1), xx.unsqueeze(1), xz.unsqueeze(1)], dim=-1), s_at

    def forward_train_DSup_bk(self, x):
        # self.at_scale = self.at_scale.to(x.device)
        # self.fat_scale = self.fat_scale.to(x.device)
        self.grid = self.grid.to(x.device)
        x1 = self.inc(x)
        x2, s2 = self.down1(x1)
        x3, s3 = self.down2(x2)
        x4, s4 = self.down3(x3)
        x5, s5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6*6*6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/self.at_scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        s_at = self.at_fixscale.get_scale(xa_at)
        x5 = x5 + xa_at


        # generate prob map
        s = torch.cat(
            [s2, s3, s4, s5, s_at],
            dim=1
        )

        s_at = self.fat_fuse(s)
        s_at = F.softmax(s_at.view(s.shape[0], 1, -1), dim=-1)
        s_at = s_at.view(s.shape[0], -1, s.shape[2], s.shape[3], s.shape[4])
        s_at_down = F.adaptive_avg_pool3d(
            s_at,
            output_size=(x5.shape[-3], x5.shape[-2], x5.shape[-1])
        )
        s_at_down = F.softmax(s_at_down.view(s_at_down.shape[0],
                                             s_at_down.shape[1],
                                             -1), dim=-1)
        s_at_down = s_at_down.view(s_at_down.shape[0], s_at_down.shape[1],
                                   x5.shape[-3], x5.shape[-2], x5.shape[-1])

        # get estimated coordinates
        xy = torch.sum(
            self.grid.view(1, 1, -1, 1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xx = torch.sum(
            self.grid.view(1, 1, 1, -1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xz = torch.sum(
            self.grid.view(1, 1, 1, 1, -1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xy, xy_l, xy_u = self.return_coords(xy, x5.shape[-3], x.shape[-3])
        xx, xx_l, xx_u = self.return_coords(xx, x5.shape[-2], x.shape[-2])
        xz, xz_l, xz_u = self.return_coords(xz, x5.shape[-1], x.shape[-1])
        # print(xy, xy_l * 16, (xy_u + 1) * 16)

        # crop the volume
        # x5 = x5[..., xy_l:xy_u + 1, xx_l:xx_u + 1, xz_l:xz_u + 1]
        # x4 = x4[..., xy_l * 2:(xy_u+1) * 2, xx_l * 2:(xx_u+1) * 2, xz_l * 2:(xz_u+1) * 2]
        # x3 = x3[..., xy_l * 4:(xy_u+1) * 4, xx_l * 4:(xx_u+1) * 4, xz_l * 4:(xz_u+1) * 4]
        # x2 = x2[..., xy_l * 8:(xy_u+1) * 8, xx_l * 8:(xx_u+1) * 8, xz_l * 8:(xz_u+1) * 8]
        # x1 = x1[..., xy_l * 16:(xy_u+1) * 16, xx_l * 16:(xx_u+1) * 16, xz_l * 16:(xz_u+1) * 16]

        x5 = self.batch_crop(x5,
                             xy_l,
                             xy_u + 1,
                             xx_l,
                             xx_u + 1,
                             xz_l,
                             xz_u + 1)

        x4 = self.batch_crop(x4,
                             xy_l * 2,
                             (xy_u + 1) * 2,
                             xx_l * 2,
                             (xx_u + 1) * 2,
                             xz_l * 2,
                             (xz_u + 1) * 2)

        x3 = self.batch_crop(x3,
                             xy_l * 4,
                             (xy_u + 1) * 4,
                             xx_l * 4,
                             (xx_u + 1) * 4,
                             xz_l * 4,
                             (xz_u + 1) * 4)

        x2 = self.batch_crop(x2,
                             xy_l * 8,
                             (xy_u + 1) * 8,
                             xx_l * 8,
                             (xx_u + 1) * 8,
                             xz_l * 8,
                             (xz_u + 1) * 8)

        x1 = self.batch_crop(x1,
                             xy_l * 16,
                             (xy_u + 1) * 16,
                             xx_l * 16,
                             (xx_u + 1) * 16,
                             xz_l * 16,
                             (xz_u + 1) * 16)

        # print(x5.shape)
        # print(x4.shape)
        # print(x.shape)

        xo = self.up1(x5, x4)
        xo = self.up2(xo, x3)
        xo = self.up3(xo, x2)
        xo = self.up4(xo, x1)
        if self.with_dropout:
            xo = self.drop(xo)

        xo = self.outc(xo)
        xo = self.soft(xo)

        x_out = torch.min(xo.view(xo.shape[0], -1), dim=-1)[0].view(-1, 1, 1, 1, 1) * \
                torch.ones(
                    x.shape[0],
                    self.num_classes,
                    x.shape[2],
                    x.shape[3],
                    x.shape[4],
                    device=x.device
                )

        x_out[:, -1, :, :, :] = 0
        x_out[:, 0,  :, :, :] = 1

        for i, x_outi in enumerate(x_out):
            x_out[i,
            ...,
            xy_l[i] * 16:(xy_u[i] + 1) * 16,
            xx_l[i] * 16:(xx_u[i] + 1) * 16,
            xz_l[i] * 16:(xz_u[i] + 1) * 16] = xo[i]

        return x_out, torch.cat([xy.unsqueeze(1), xx.unsqueeze(1), xz.unsqueeze(1)], dim=-1), s_at_down

    def vis_encoder_forward(self, x, block = 1):
        x1 = self.inc(x)
        if block == 0:
            return x1
        x2 = self.down1(x1)
        if block == 1:
            return x2
        x3 = self.down2(x2)
        if block == 2:
            return x3
        x4 = self.down3(x3)
        if block == 3:
            return x4
        x5 = self.down4(x4)
        if block == 4:
            return x5

    def forward_pureseg(self, x):
        x1 = self.inc(x)
        x2, _ = self.down1(x1)
        x3, _ = self.down2(x2)
        x4, _ = self.down3(x3)
        x5, _ = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 6 * 6 * 6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6 * 6 * 6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6 * 6 * 6).transpose(1, 2)
        scale = torch.sqrt(torch.Tensor([6 * 6 * 6])).item()

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q) / scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        x5 = x5 + xa_at

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = self.soft(x)
        return x

    def forward_seg(self, x):
        # self.at_scale = self.at_scale.to(x.device)
        # self.fat_scale = self.fat_scale.to(x.device)
        self.grid = self.grid.to(x.device)
        x1 = self.inc(x)
        x2, s2 = self.down1(x1)
        x3, s3 = self.down2(x2)
        x4, s4 = self.down3(x3)
        x5, s5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6*6*6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/self.at_scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        s_at = self.at_fixscale.get_scale(xa_at)
        x5 = x5 + xa_at

        # generate prob map
        s = torch.cat(
            [s2, s3, s4, s5, s_at],
            dim=1
        )

        # s_at_k = self.fat_key(s)
        # s_at_k = s_at_k.view(s_at_k.shape[0], -1, self.fixSize3).transpose(1, 2)
        # s_at_q = self.fat_que(s)
        # s_at_q = s_at_q.view(s_at_q.shape[0], -1, self.fixSize3)
        # s_at_v = self.fat_val(s)
        # s_at_v = s_at_v.view(s_at_v.shape[0], -1, self.fixSize3).transpose(1, 2)
        # # s_at_v = F.softmax(F.instance_norm(s_at_v), dim=1)
        #
        # s_at = torch.bmm(
        #     F.softmax(torch.bmm(s_at_k, s_at_q) / self.fat_scale, dim=-1),
        #     s_at_v
        # )

        s_at = self.fat_fuse(s)
        s_at = F.softmax(s_at.view(s.shape[0], 1, -1), dim=-1)
        s_at = s_at.view(s.shape[0], -1, s.shape[2], s.shape[3], s.shape[4])

        # get estimated coordinates
        xy = torch.sum(
            self.grid.view(1, 1, -1, 1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xx = torch.sum(
            self.grid.view(1, 1, 1, -1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xz = torch.sum(
            self.grid.view(1, 1, 1, 1, -1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xy, xy_l, xy_u = self.return_coords(xy, x5.shape[-3], x.shape[-3])
        xx, xx_l, xx_u = self.return_coords(xx, x5.shape[-2], x.shape[-2])
        xz, xz_l, xz_u = self.return_coords(xz, x5.shape[-1], x.shape[-1])
        # print(xy, xy_l * 16, (xy_u + 1) * 16)

        # crop the volume
        # x5 = x5[..., xy_l:xy_u + 1, xx_l:xx_u + 1, xz_l:xz_u + 1]
        # x4 = x4[..., xy_l * 2:(xy_u+1) * 2, xx_l * 2:(xx_u+1) * 2, xz_l * 2:(xz_u+1) * 2]
        # x3 = x3[..., xy_l * 4:(xy_u+1) * 4, xx_l * 4:(xx_u+1) * 4, xz_l * 4:(xz_u+1) * 4]
        # x2 = x2[..., xy_l * 8:(xy_u+1) * 8, xx_l * 8:(xx_u+1) * 8, xz_l * 8:(xz_u+1) * 8]
        # x1 = x1[..., xy_l * 16:(xy_u+1) * 16, xx_l * 16:(xx_u+1) * 16, xz_l * 16:(xz_u+1) * 16]

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = self.soft(x)

        x_mask = torch.zeros(
            x.shape[0],
            x.shape[2],
            x.shape[3],
            x.shape[4]
        ).to(x.device)

        x_center = torch.cat([xy.unsqueeze(1), xx.unsqueeze(1), xz.unsqueeze(1)], dim=-1)

        for i in range(x.shape[0]):
            x_mask[i,
            xy_l[i] * 16-2:(xy_u[i] + 1) * 16+2,
            xx_l[i] * 16-2:(xx_u[i] + 1) * 16+2,
            xz_l[i] * 16-2:(xz_u[i] + 1) * 16+2] = 1

        return x, x_mask, x_center


class UNet3D_CBAMAttentionZoomDSup(UNet3D_CBAMAttentionZoom):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False, fixSize = 32):
        super(UNet3D_CBAMAttentionZoomDSup, self).__init__(
            num_classes,
            n_channels,
            ngf,
            with_dropout,
            fixSize)

        self.doutc = conv1x1x1(ngf*8, num_classes)
        self.dsoft = nn.Softmax(dim=1)

    def forward_train_DSup(self, x):
        # self.at_scale = self.at_scale.to(x.device)
        # self.fat_scale = self.fat_scale.to(x.device)
        self.grid = self.grid.to(x.device)
        x1 = self.inc(x)
        x2, s2 = self.down1(x1)
        x3, s3 = self.down2(x2)
        x4, s4 = self.down3(x3)
        x5, s5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        xa_k = xa_k.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6*6*6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/self.at_scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        s_at = self.at_fixscale.get_scale(xa_at)
        x5 = x5 + xa_at

        dx5 = self.doutc(x5)
        dx5 = self.dsoft(dx5)

        # generate prob map
        s = torch.cat(
            [s2, s3, s4, s5, s_at],
            dim=1
        )

        s_at = self.fat_fuse(s)
        s_at = F.softmax(s_at.view(s.shape[0], 1, -1), dim=-1)
        s_at = s_at.view(s.shape[0], -1, s.shape[2], s.shape[3], s.shape[4])

        # get estimated coordinates
        xy = torch.sum(
            self.grid.view(1, 1, -1, 1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xx = torch.sum(
            self.grid.view(1, 1, 1, -1, 1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xz = torch.sum(
            self.grid.view(1, 1, 1, 1, -1) * s_at,
            dim=(1, 2, 3, 4)
        )

        xy, xy_l, xy_u = self.return_coords(xy, x5.shape[-3], x.shape[-3])
        xx, xx_l, xx_u = self.return_coords(xx, x5.shape[-2], x.shape[-2])
        xz, xz_l, xz_u = self.return_coords(xz, x5.shape[-1], x.shape[-1])
        # print(xy, xy_l * 16, (xy_u + 1) * 16)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = self.soft(x)

        x_mask = torch.zeros(
            x.shape[0],
            x.shape[2],
            x.shape[3],
            x.shape[4]
        ).to(x.device)

        x_center = torch.cat([xy.unsqueeze(1), xx.unsqueeze(1), xz.unsqueeze(1)], dim=-1)

        for i in range(x.shape[0]):
            x_mask[i,
            xy_l[i] * 16-2:(xy_u[i] + 1) * 16+2,
            xx_l[i] * 16-2:(xx_u[i] + 1) * 16+2,
            xz_l[i] * 16-2:(xz_u[i] + 1) * 16+2] = 1

        return x, x_mask, x_center, dx5

class UNet3D_CBAMAttention(nn.Module):
    def __init__(self,
                 num_classes,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False):
        super(UNet3D_CBAMAttention, self).__init__()
        self.with_dropout = with_dropout
        self.inc = inconv(n_channels, ngf)
        self.down1 = down_CBAM(ngf, ngf*2)
        self.down2 = down_CBAM(ngf*2, ngf*4)
        self.down3 = down_CBAM(ngf*4, ngf*8)
        self.down4 = down_CBAM(ngf*8, ngf*8)

        self.avgpool = nn.AdaptiveAvgPool3d(
            (6, 6, 6)
        )
        self.at_key = conv1x1x1(ngf*8, ngf*4)
        self.at_que = conv1x1x1(ngf*8, ngf*4)
        self.at_val = conv1x1x1(ngf*8, ngf*4)
        self.at_out = conv1x1x1(ngf*4, ngf*8)

        self.up1 = up_CBAM(ngf*16, ngf*4)
        self.up2 = up_CBAM(ngf*8, ngf*2)
        self.up3 = up_CBAM(ngf*4, ngf)
        self.up4 = up_CBAM(ngf*2, ngf)
        self.outc = outconv(ngf, num_classes)
        self.soft = nn.Softmax(dim=1)

        if with_dropout:
            self.drop = nn.Dropout(0.5)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        # attention
        xa = self.avgpool(x5)
        xa_k = self.at_key(xa)
        print(x5.shape)
        print(xa.shape)
        print(xa_k.shape)
        xa_k = xa_k.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        xa_q = self.at_que(xa)
        xa_q = xa_q.view(xa.shape[0], -1, 6*6*6)
        xa_r = self.at_val(xa)
        xa_r = xa_r.view(xa.shape[0], -1, 6*6*6).transpose(1, 2)
        scale = torch.sqrt(torch.Tensor([6*6*6])).item()

        xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/scale, dim=-1), xa_r)
        xa_at = xa_at.view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
        # resample
        xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
        xa_at = self.at_out(xa_at)

        x5 = x5 + xa_at

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.with_dropout:
            x = self.drop(x)

        x = self.outc(x)
        x = self.soft(x)
        return x

    def vis_encoder_forward(self, x, block = 1):
        x1 = self.inc(x)
        if block == 0:
            return x1
        x2 = self.down1(x1)
        if block == 1:
            return x2
        x3 = self.down2(x2)
        if block == 2:
            return x3
        x4 = self.down3(x3)
        if block == 3:
            return x4
        x5 = self.down4(x4)
        if block == 4:
            return x5

# class UNet3D_ZoomAttention(nn.Module):
#     def __init__(self,
#                  num_classes,
#                  n_channels=1,
#                  ngf=32,
#                  with_dropout = False):
#         super(UNet3D_ZoomAttention, self).__init__()
#         self.with_dropout = with_dropout
#         self.inc = inconv(n_channels, ngf)
#         self.down1 = down(ngf, ngf*2)
#         self.down2 = down(ngf*2, ngf*4)
#         self.down3 = down(ngf*4, ngf*8)
#         self.down4 = down(ngf*8, ngf*8)
#
#         self.avgpool = nn.AdaptiveAvgPool3d(
#             (6, 6, 6)
#         )
#         self.at_key = conv1x1x1(ngf*8, 6*6*6)
#         self.at_que = conv1x1x1(ngf*8, 6*6*6)
#         self.at_val = conv1x1x1(ngf*8, ngf*8)
#
#         self.up1 = up(ngf*16, ngf*4)
#         self.up2 = up(ngf*8, ngf*2)
#         self.up3 = up(ngf*4, ngf)
#         self.up4 = up(ngf*2, ngf)
#         self.outc = outconv(ngf, num_classes)
#         self.soft = nn.Softmax(dim=1)
#
#         if with_dropout:
#             self.drop = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x1 = self.inc(x)
#         x2 = self.down1(x1)
#         x3 = self.down2(x2)
#         x4 = self.down3(x3)
#         x5 = self.down4(x4)
#
#         # attention
#         xa = self.avgpool(x5)
#         xa_k = self.at_key(xa)
#         print(x5.shape)
#         print(xa.shape)
#         print(xa_k.shape)
#         xa_k = xa_k.view(xa.shape[0], 6*6*6, -1).transpose(1, 2)
#         xa_q = self.at_que(xa)
#         xa_q = xa_q.view(xa.shape[0], 6*6*6, -1)
#         xa_r = self.at_val(xa)
#         xa_r = xa_r.view(xa.shape[0], xa.shape[1], -1).transpose(1, 2)
#
#         xa_at = torch.bmm(F.softmax(torch.bmm(xa_k, xa_q)/14.7, dim=-1), xa_r)
#         xa_at = xa_at.transpose(1, 2).view(xa.shape[0], -1, xa.shape[2], xa.shape[3], xa.shape[4])
#         # resample
#         xa_at = F.interpolate(xa_at, size=x5.shape[-3:])
#         # xa_at = self.out_model(xa_at)
#
#         x5 = x5 + xa_at
#
#         x = self.up1(x5, x4)
#         x = self.up2(x, x3)
#         x = self.up3(x, x2)
#         x = self.up4(x, x1)
#         if self.with_dropout:
#             x = self.drop(x)
#
#         x = self.outc(x)
#         x = self.soft(x)
#         return x
#
#     def vis_encoder_forward(self, x, block = 1):
#         x1 = self.inc(x)
#         if block == 0:
#             return x1
#         x2 = self.down1(x1)
#         if block == 1:
#             return x2
#         x3 = self.down2(x2)
#         if block == 2:
#             return x3
#         x4 = self.down3(x3)
#         if block == 3:
#             return x4
#         x5 = self.down4(x4)
#         if block == 4:
#             return x5

class UNet3D_multi_inputs(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 ngf=24, input_branches = 3):
        super(UNet3D_multi_inputs, self).__init__()
        self.inc = inconv(n_channels, ngf, group=input_branches)
        self.down1 = down(ngf, ngf*2, group=input_branches)
        self.down2 = down(ngf*2, ngf*4, group=input_branches)
        self.down3 = down(ngf*4, ngf*8, group=input_branches)
        self.down4 = down(ngf*8, ngf*8, group=1)
        self.up1 = up(ngf*16, ngf*4)
        self.up2 = up(ngf*8, ngf*2)
        self.up3 = up(ngf*4, ngf)
        self.up4 = up(ngf*2, ngf)
        self.outc = outconv(ngf, n_classes)
        self.soft = nn.Softmax(dim=1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.soft(x)
        return x

class UNet3D_multi_inputs2(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 ngf=32, input_branches = 3):
        super(UNet3D_multi_inputs2, self).__init__()
        self.inc = multi_input_double_conv(n_channels,
                                           ngf*input_branches,
                                           split_group=input_branches,
                                           dilation_rate=[1, 2])

        self.down1 = multi_input_down_conv(ngf*input_branches,
                                           ngf*2*input_branches,
                                           split_group=input_branches,
                                           dilation_rate=[1, 2])

        self.down2 = multi_input_down_conv(ngf*2*input_branches,
                                           ngf*4*input_branches,
                                           split_group=input_branches,
                                           dilation_rate=[1, 2])

        self.down3 = multi_input_down_conv(ngf*4*input_branches,
                                           ngf*8*input_branches,
                                           split_group=input_branches,
                                           dilation_rate=[1, 2])

        self.down4 = down(ngf*8,
                          ngf*8,
                          group=1,
                          dilation_rate=[1, 2])

        self.up1 = up(ngf*16, ngf*4)

        self.up2 = up(ngf*8, ngf*2)

        self.up3 = up(ngf*4, ngf)

        self.up4 = up(ngf*2, ngf)
        self.drop = nn.Dropout3d(p=0.2)
        self.outc = outconv(ngf, n_classes)
        self.soft = nn.Softmax(dim=1)
        self.input_branches = input_branches
        self.in_channels = n_channels
        self.out_channels = n_classes

    def forward(self, x):
        x1, x1max = self.inc(x)
        # print(type(x1))
        x2, x2max = self.down1(x1)
        x3, x3max = self.down2(x2)
        x4, x4max = self.down3(x3)
        x5 = self.down4(x4max)
        '''
        x5max = torch.argmax(
            torch.stack(
                torch.split(x5, self.input_branches, dim=0)
            )
        )
        '''
        x = self.up1(x5, x4max)
        x = self.up2(x, x3max)
        x = self.up3(x, x2max)
        x = self.up4(x, x1max)
        x = self.drop(x)
        x = self.outc(x)
        x = self.soft(x)
        return x

class UNet3D_Encoder_multiin_rep(nn.Module):
    def __init__(self,
                 n_channels = 3,
                 n_classes=2,
                 ngf=16,
                 input_branches = 3):
        super(UNet3D_Encoder_multiin_rep, self).__init__()
        self.inc = multi_input_double_conv_rep(n_channels,
                                               ngf,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

class UNet3D_multi_inputs_rep(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 ngf=64, input_branches = 3):
        super(UNet3D_multi_inputs_rep, self).__init__()
        self.inc = multi_input_double_conv_rep(n_channels,
                                               ngf,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down1 = multi_input_down_conv_rep(ngf,
                                               ngf*2,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down2 = multi_input_down_conv_rep(ngf*2,
                                               ngf*4,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down3 = multi_input_down_conv_rep(ngf*4,
                                               ngf*8,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down4 = down_merge_rep(ngf*8,
                          ngf*8,
                          dilation_rate=[1, 2])

        self.up1 = up(ngf*16, ngf*4)

        self.up2 = up(ngf*8, ngf*2)

        self.up3 = up(ngf*4, ngf)

        self.up4 = up(ngf*2, ngf)

        self.drop = nn.Dropout3d(p=0.2)
        self.outc = outconv(ngf, n_classes)

        self.soft = nn.Softmax(dim=1)
        self.input_branches = input_branches
        self.in_channels = n_channels
        self.out_channels = n_classes

    def forward(self, x):
        x1, x1max = self.inc(x)
        x2, x2max = self.down1(x1)
        x3, x3max = self.down2(x2)
        x4, x4max = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4max)
        x = self.up2(x, x3max)
        x = self.up3(x, x2max)
        x = self.up4(x, x1max)
        x = self.drop(x)
        x = self.outc(x)
        x = self.soft(x)
        return x

class UNet3D_multi_inputs_rep_mass(nn.Module):
    def __init__(self,
                 n_channels=3,
                 n_classes=2,
                 ngf=32, input_branches = 3):
        super(UNet3D_multi_inputs_rep_mass, self).__init__()
        self.inc = multi_input_double_conv_rep(n_channels,
                                               ngf,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down1 = multi_input_down_conv_rep(ngf,
                                               ngf*2,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down2 = multi_input_down_conv_rep(ngf*2,
                                               ngf*4,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down3 = multi_input_down_conv_rep(ngf*4,
                                               ngf*8,
                                               split_group=input_branches,
                                               dilation_rate=[1, 2])

        self.down4 = down(ngf*8*input_branches,
                          ngf*8*input_branches,
                          group=3,
                          dilation_rate=[1, 2])

        self.up1 = up(ngf*16, ngf*4)

        self.up2 = up(ngf*8, ngf*2)

        self.up3 = up(ngf*4, ngf)

        self.up4 = up(ngf*2, ngf)

        self.drop = nn.Dropout3d(p=0.2)
        self.outc = outconv(ngf, n_classes)

        self.soft = nn.Softmax(dim=1)
        self.input_branches = input_branches
        self.in_channels = n_channels
        self.out_channels = n_classes

    def forward(self, x):
        x1, x1max = self.inc(x)
        x2, x2max = self.down1(x1)
        x3, x3max = self.down2(x2)
        x4, x4max = self.down3(x3)
        x5 = self.down4(x4)

        x5max, _ = torch.max(
            torch.stack(
                torch.split(x5, int(x5.shape[1]/self.input_branches), dim=1)
            ), dim=0
        )

        x = self.up1(x5max, x4max)
        x = self.up2(x, x3max)
        x = self.up3(x, x2max)
        x = self.up4(x, x1max)
        x = self.drop(x)
        x = self.outc(x)
        x = self.soft(x)
        return x

class UNet3D_multi_encoder(nn.Module):
    def __init__(self, n_channels=1, n_classes=1, n_input = 3, n_block=4, ngf=32, max_ngf=512):
        super(UNet3D_multi_encoder, self).__init__()
        # self.inc = inconv(n_channels, ngf)
        self.n_block = n_block

        self.blocks = []
        self.blocks.append([inconv(n_channels, ngf)] * n_input)
        for i in range(n_block):
            ngf_in = min(ngf*(2**i), max_ngf)
            ngf_out = min(ngf*(2**(i+1)), max_ngf)
            self.blocks.append([down(ngf_in, ngf_out)] * n_input)

        ngf_in = self.blocks[-2][0].out_channels*n_input + \
                         self.blocks[-1][0].out_channels*n_input
        ngf_out = self.blocks[-1][0].in_channels*n_input
        self.blocks.append([up(ngf_in, ngf_out)])


        for i in range(1, n_block):
            ngf_in = self.blocks[n_block-i][0].out_channels*n_input + \
                         self.blocks[-1][0].out_channels
            ngf_out = self.blocks[n_block-i][0].in_channels*n_input
            self.blocks.append([up(ngf_in, ngf_out)])

        self.blocks.append(outconv(ngf, n_classes))

        '''
        # first out
        self.ups.append()
        for i in range(1, n_block-1):
            ngf_in = self.down[n_block-1-i][0].out_channels


        self.down1 = down(ngf, ngf*2)
        self.down2 = down(ngf*2, ngf*4)
        self.down3 = down(ngf*4, ngf*8)
        self.down4 = down(ngf*8, ngf*8)

        self.up1 = up(ngf*32, ngf*8)
        self.up2 = up(ngf*16, ngf*2)
        self.up3 = up(ngf*8, ngf)
        self.up4 = up(ngf*2, ngf)
        '''
        self.blocks = nn.ModuleList(self.blocks)
        self.outc = outconv(ngf, n_classes)
        self.soft = nn.Softmax(dim=1)

    def forward(self, input):
        xs = []
        xs.append([self.blocks[0][n_inp](input[n_inp]) for n_inp in range(len(input))]) # input conv

        for n_block in range(1, self.n_block):
            xs.append([self.blocks[n_block][n_inp](xs[n_block-1][n_inp]) for n_inp in range(len(input))])

        temp_mid_xs = [self.blocks[self.n_block][n_inp](xs[self.n_block-1][n_inp]) for n_inp in range(len(input))]

        x = torch.cat(temp_mid_xs, dim=1)

        for n in range(1, self.n_block):
            x = self.blocks[self.n_block+n][0](x, xs[self.n_block-n])

        x = self.outc(x)

        x = self.soft(x)

        return x

    def forward_old(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        x = self.outc(x)
        x = self.soft(x)
        return x

if __name__=='__main__':
    input_m = torch.rand(1,1, 32, 32, 32).cuda()
    model = UNet3D_AttentionZoom(num_classes=2,
                 n_channels=1,
                 ngf=32,
                 with_dropout = False)
    model.train()
    a, b, c, d = model(input_m)
    print(a.shape)
    print(b, c, d)
    model.eval()
    a = model(input_m)
    print(a.shape)
