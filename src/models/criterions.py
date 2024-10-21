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

    if y_true.shape[1] == 1:
        # efficient one hot
        y_true = torch.cat((1 - y_true, y_true), dim=1)

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

class BinaryDiceLoss(nn.Module):
    def __init__(self, homo_scale = 0.0, smooth=1e-12,
                 weighting = False, combining=False):
        super(BinaryDiceLoss, self).__init__()
        self.homo_scale = homo_scale
        self.smooth = smooth
        self.weighting = weighting
        # self.weight = None

    def __call__(self, y_pred, y_true):
        # smooth = 1e-12
        if len(y_pred.shape) < len(y_true.shape):
            y_pred = y_pred.unsqueeze(0)

        if y_true.shape[1] ==1:
            # efficient one hot
            y_true = torch.cat((1 - y_true, y_true), dim=1)

        y_true_shape = y_true.size()
        # y_true_reshaped = y_true[:, 1:, ...].view(y_true_shape[0], y_true_shape[1], -1)
        # y_pred_reshaped = y_pred[:, 1:, ...].view(y_true_shape[0], y_true_shape[1], -1)
        y_true_reshaped = y_true.view(y_true_shape[0], y_true_shape[1], -1)
        y_pred_reshaped = y_pred.view(y_true_shape[0], y_true_shape[1], -1)

        temp = y_true_reshaped * y_pred_reshaped
        # intersection = np.sum(temp, [0,1], keepdims=False)
        if self.weighting:
            weight = torch.sum(y_true_reshaped, dim=-1)
            weight += (weight==0)*1e-14
            weight = 1.0/weight
            total_sum = torch.sum(weight, dim=-1, keepdim=True)
            weight /= total_sum
            temp = torch.sum(temp, dim=-1)
            intersection = torch.sum(weight*temp, dim=0)
            # balance the min

        else:
            intersection = torch.sum(torch.sum(temp, dim=-1), dim=0)

        dices = 2 * intersection / (torch.sum(torch.sum(y_true_reshaped, dim=-1), dim=0) +
                                    torch.sum(torch.sum(y_pred_reshaped, dim=-1), dim=0) + self.smooth)


        if self.homo_scale > 0.0:
            homogeneity = self.homo_scale * homo_func(y_true, y_pred)
        else:
            homogeneity = 0.0

        return -torch.sum(dices) - homogeneity