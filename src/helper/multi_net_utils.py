from __future__ import print_function

import torch
from torch import nn

import numpy as np

class MultiNetTrainableTem(nn.Module):
    """
    composing multiple networks into a single one
    I don't know how to deal with the fucking broking laptop!!!!

    """
    def __init__(self, module_list, ini_T=1.0, trainable=False):
        super(MultiNetTrainableTem, self).__init__()
        # enable parallel train of all networks
        self.num_models = len(module_list)
        self.trainabble = trainable
        self.module_list = module_list
        if trainable:
            self.T = [TT(ini_t=ini_T)]*self.num_models
        else:
            self.T = [ini_T]*self.num_models

    def __len__(self):
        return len(self.module_list)

    def forward(self, x, is_feat=False, preact=False):
        # single input multiple output from multiple networks
        out = [self.module_list[i](x, is_feat=is_feat, preact=preact) for i in range(self.num_models)]
        return out

class TT(nn.Module):
    """trainable temperature"""
    def __init__(self, ini_t = 1.0):
        super(TT, self).__init__()
        # 1/t
        self.TT = nn.Parameter(
            torch.tensor(1/ini_t).float().log(), requires_grad=True
        )

    def forward(self, x):
        x = x*(self.TT.exp())
        return x

class MultiTrainableT(nn.Module):
    """
    Trainable T tables for all students towards multi teachers
    """
    def __init__(self, num_teachders=1, num_students=1):
        super(MultiTrainableT, self).__init__()
        self.num_teachers = num_teachders
        self.num_students = num_students
        self.T_table = nn.Parameter(
            torch.ones(self.num_students, self.num_teachers), requires_grad=True
        )

    def forward(self, p_students, p_teachers):
        """
        teachers and students inputs are logits lists
        :param p_students:
        :param p_teachers:
        :return:
        """


