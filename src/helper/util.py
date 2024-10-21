from __future__ import print_function

import torch
import numpy as np
from collections import OrderedDict
import os

def adjust_learning_rate_new(epoch, optimizer, LUT):
    """
    new learning rate schedule according to RotNet
    """
    lr = next((lr for (max_epoch, lr) in LUT if max_epoch > epoch), LUT[-1][1])
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def adjust_learning_rate(epoch, opt, optimizer):
    """Sets the learning rate to the initial LR decayed by decay rate every steep step"""
    steps = np.sum(epoch > np.asarray(opt.lr_decay_epochs))
    if steps > 0:
        new_lr = opt.learning_rate * (opt.lr_decay_rate ** steps)
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)

        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def load_network(network, state_dict):
    try:
        network.load_state_dict(
        remove_instanceNorm_weights(
            state_dict
        )
    )
    except:
        network.load_state_dict(
                state_dict
        )
    else:
        network.load_state_dict(
            state_dict,
            strict = False
        )
    # network.load_state_dict(torch.load(save_path))
# update learning rate (called once every epoch)

def remove_instanceNorm_weights(state_dict):
    # affect code before 0.4.0
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        if not 'running_mean' in k \
                and not 'running_var' in k \
                and not 'target_coordinate_repr' in k \
                and not 'num_batches_tracked' in k:
            name = k  # remove 'module.' of dataparallel
            new_state_dict[name] = v

    return new_state_dict

if __name__ == '__main__':

    pass
