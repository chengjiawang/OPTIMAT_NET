from __future__ import print_function, division

import sys
import time
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from .util import AverageMeter, accuracy
from models.criterions import DiceIndex

from collections import OrderedDict

def get_teacher_student_feature_tensors(feat_t, logit_t, feat_s, logit_s):
    # extend teacher feature and logits
    feat_t = [feat_t]*len(feat_s)
    feat_t = list(zip(*feat_t))

    for n_feat, feat_n in enumerate(feat_t):
        feat_t[n_feat] = torch.cat(feat_n, dim=0)

    logit_t = [logit_t]*len(logit_s)
    logit_t = torch.cat(logit_t, dim=0)

    # joint student feature and logits
    feat_s = list(zip(*feat_s))
    for n_feat, feat_n in enumerate(feat_s):
        feat_s[n_feat] = torch.cat(feat_n, dim=0)
    logit_s = torch.cat(logit_s, dim=0)

    return feat_t, logit_t, feat_s, logit_s

def train_vanilla(epoch, train_loader, model, criterion, optimizer, opt, accuracy=None):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    end = time.time()
    for idx, (input, target) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()

        # ===================forward=====================
        output = model(input)
        loss = criterion(output, target)
        if accuracy is not None:
            acc = accuracy(output, target)
        else:
            acc = torch.Tensor([0.0])
        losses.update(loss.item(), input.size(0))
        top.update(acc.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        # pass

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc {top.val:.3f} ({top.avg:.3f})'.format(
                   epoch, idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top=top))
            sys.stdout.flush()

    print(' * Acc {top.avg:.3f}'.format(top=top))

    return top.avg, losses.avg

def batch_crop(x, x_loc, fix_size=16, nd = 2):
    xy, xx, xz = x_loc
    if nd ==2:
        xy_l, xy_u = torch.floor(xy).int() - fix_size+1, torch.ceil(xy).int() + fix_size
        xx_l, xx_u = torch.floor(xx).int() - fix_size+1, torch.ceil(xx).int() + fix_size
        # print(xy_u - xy_l)
        # xz_l, xz_u = torch.floor(xz).int() - fix_size, torch.ceil(xz).int() - fix_size
        return x[..., xy_l:xy_u, xx_l:xx_u]
    else:
        xy_l, xy_u = torch.floor(xy).int() - fix_size+1, torch.ceil(xy).int() + fix_size
        xx_l, xx_u = torch.floor(xx).int() - fix_size+1, torch.ceil(xx).int() + fix_size
        xz_l, xz_u = torch.floor(xz).int() - fix_size+1, torch.ceil(xz).int() + fix_size
        return x[..., xy_l:xy_u, xx_l:xx_u, xz_l:xz_u]



def train_dia(epoch, train_loader, model, criterion, optimizer, opt, accuracy=None,
              loc_criterion=nn.SmoothL1Loss(), vis_image=False, deep_sup=False):
    """vanilla training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    loc_losses = AverageMeter()
    seg_losses = AverageMeter()
    pure_seg_losses = AverageMeter()
    if deep_sup:
        rough_cri = nn.MSELoss(reduction='sum')
        rough_seg_losses = AverageMeter()
    top = AverageMeter()

    end = time.time()
    for idx, (input, target, center, input_crop, target_crop) in enumerate(train_loader):
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            center = center.cuda()
            input_crop = input_crop.cuda()
            target_crop = target_crop.cuda()

        # ===================forward=====================
        output, pred_center= model.forward_train(input)
        output_crop = model.forward_pureseg(input_crop.float())

        print('prediction loc: ', pred_center)
        print('center: ', center)
        seg_loss = criterion(output, target)
        pure_seg_loss = criterion(output_crop, target_crop)
        # print('pred_shape: ', pred_center.shape)
        # print('input_shape: ', center.shape)
        loc_loss = loc_criterion(pred_center, center)
        if accuracy is not None:
            acc = accuracy(output, target)
        else:
            acc = torch.Tensor([0.0])

        if deep_sup:
            rough_target = F.avg_pool3d(target, kernel_size=16)
            rough_sum = torch.sum(rough_target, dim=(-1,-2,-3), keepdim=True)
            rough_sum += (rough_sum==0)*1e-14
            rough_target /= rough_sum

            rough_seg_loss = rough_cri(rough.double(), rough_target.double())
            # print(rough_seg_loss.dtype)

            loss = seg_loss + loc_loss + pure_seg_loss + 0.5*rough_seg_loss.double()

            rough_seg_losses.update(rough_seg_loss.item(), input.size(0))
        else:
            loss = seg_loss + 10*loc_loss + pure_seg_loss
            print(loss.dtype)

        losses.update(loss.item(), input.size(0))
        seg_losses.update(seg_loss.item(), input.size(0))
        pure_seg_losses.update(pure_seg_loss.item(), input.size(0))
        loc_losses.update(loc_loss.item(), input.size(0))
        top.update(acc.item(), input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # tensorboard logger
        # pass

        # print info
        if idx % opt.print_freq == 0:
            if deep_sup:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Segmentation Loss {seg_loss.val:.4f} ({seg_loss.avg:.4f})\t'
                      'Pure Seg Loss {pure_seg_loss.val:.4f} ({pure_seg_loss.avg:.4f})\t'
                      'Rough Seg Loss {rough_seg_loss.val:.4f} ({rough_seg_loss.avg:.4f})\t'
                      'detection Loss {loc_loss.val:.4f} ({loc_loss.avg:.4f})\t'
                      'Acc {top.val:.3f} ({top.avg:.3f})'.format(
                    epoch, idx, len(train_loader), batch_time=batch_time,
                    data_time=data_time, loss=losses, seg_loss=seg_losses,
                    pure_seg_loss=pure_seg_losses,
                    rough_seg_loss=rough_seg_losses,
                    loc_loss=loc_losses,
                    top=top))
            else:
                print('Epoch: [{0}][{1}/{2}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Segmentation Loss {seg_loss.val:.4f} ({seg_loss.avg:.4f})\t'
                      'Pure Seg Loss {pure_seg_loss.val:.4f} ({pure_seg_loss.avg:.4f})\t'
                      'detection Loss {loc_loss.val:.4f} ({loc_loss.avg:.4f})\t'
                      'Acc {top.val:.3f} ({top.avg:.3f})'.format(
                       epoch, idx, len(train_loader), batch_time=batch_time,
                       data_time=data_time, loss=losses, seg_loss=seg_losses,
                    pure_seg_loss=pure_seg_losses,
                    loc_loss=loc_losses,
                    top=top))
            sys.stdout.flush()


            if vis_image:
                disp_image_names = [
                    'Whole/input-gtcenter',
                    'Whole/input-predcenter',
                    'Whole/ground-truth-gtcenter',
                    'Whole/ground-truth-predcenter',
                    'Whole/predict-gtcenter',
                    'Whole/predict-predcenter',
                    'Crop/input-gtcenter',
                    'Crop/input-predcenter',
                    'Crop/ground-truth-gtcenter',
                    'Crop/ground-truth-predcenter',
                    'Crop/predict-gtcenter',
                    'Crop/predict-predcenter'
                ]

                output = torch.argmax(output, dim=1).unsqueeze(1)

                center_int = torch.floor(center).int()
                pred_center_int = torch.floor(pred_center).int()
                # print(output.shape)

                disp_images = []
                disp_images += [
                    input[-1, -1, ..., center_int[-1, -1]]
                ]
                disp_images += [
                    input[-1, -1, ..., pred_center_int[-1, -1]]
                ]
                disp_images += [
                    target[-1, -1, ..., center_int[-1, -1]]
                ]
                disp_images += [
                    target[-1, -1, ..., pred_center_int[-1, -1]]
                ]
                disp_images += [
                    output[-1, -1, ..., center_int[-1, -1]]
                ]
                disp_images += [
                    output[-1, -1, ..., pred_center_int[-1, -1]]
                ]

                # crop
                fix_size = 16
                disp_images += [
                    batch_crop(input[-1, -1, ..., center_int[-1,-1]], center[-1], fix_size)
                ]
                disp_images += [
                    batch_crop(input[-1, -1, ..., pred_center_int[-1, -1]], pred_center[-1], fix_size)
                ]
                disp_images += [
                    batch_crop(target[-1, -1, ..., center_int[-1, -1]], center[-1], fix_size)
                ]
                disp_images += [
                    batch_crop(target[-1, -1, ..., pred_center_int[-1, -1]], pred_center[-1], fix_size)
                ]
                disp_images += [
                    batch_crop(output[-1, -1, ..., center_int[-1, -1]], center[-1], fix_size)
                ]
                disp_images += [
                    batch_crop(output[-1, -1, ..., pred_center_int[-1, -1]], pred_center[-1], fix_size)
                ]

    print(' * Acc {top.avg:.3f}'.format(top=top))
    if vis_image:
        return top.avg, losses.avg, seg_losses.avg, loc_losses.avg, disp_images, disp_image_names

    else:

        return top.avg, losses.avg, seg_losses.avg, loc_losses.avg


def train_distill(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill in ['bs', 'bs_t', 'bs_dis', 'bs_dis_t']:
            if opt.bs_feature == 'feature':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
            elif opt.bs_feature == 'logits':
                f_s = logit_s
                f_t = logit_t
            else:
                raise NotImplementedError(opt.bs_feature)

            y = target

            loss_kd = criterion_kd(f_s, f_t, y)

        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    if opt.trainable_T:
        T = 1.0/(criterion_div.T_trans.TT.data.exp())
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} T {T: .3f}'
              .format(top1=top1, top5=top5, T=T))
    else:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def train_distill_multi(epoch, train_loader, module_list, criterion_list, optimizer, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_ss = module_list[0]
    model_tt = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True

        # feat_s, logit_s = model_s(input, is_feat=True, preact=preact)
        with torch.no_grad():
            feat_tt = [None]*len(model_tt)
            logit_tt = [None]*len(model_tt)
            for n_t, model_t in enumerate(model_tt):
                feat_tt[n_t], logit_tt[n_t] = model_t(input, is_feat=True, preact=preact)
                feat_tt[n_t] = [f.detach() for f in feat_tt[n_t]]


        feat_s = []
        logit_s = []
        for n_s, model_s in enumerate(model_ss):
            feat, logit = model_s(input, is_feat=True, preact=preact)
            feat_s.append(feat)
            logit_s.append(logit)

        target = [target] * len(opt.model_ss.split(','))
        target = torch.cat(target, dim=0)

        feat_t, logit_t, feat_s, logit_s = get_teacher_student_feature_tensors(
            feat_t, logit_t, feat_s, logit_s
        )

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill in ['bs', 'bs_t', 'bs_dis', 'bs_dis_t']:
            if opt.bs_feature == 'feature':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
            elif opt.bs_feature == 'logits':
                f_s = logit_s
                f_t = logit_t
            else:
                raise NotImplementedError(opt.bs_feature)

            y = target

            loss_kd = criterion_kd(f_s, f_t, y)

        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    if opt.trainable_T:
        T = 1.0/(criterion_div.T_trans.TT.data.exp())
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} T {T: .3f}'
              .format(top1=top1, top5=top5, T=T))
    else:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, losses.avg

def train_distill_partial(epoch, train_loader,
                          module_list, criterion_list,
                          optimizer, dis_target_classes, opt):
    """One epoch distillation"""
    # set modules as train()
    for module in module_list:
        module.train()
    # set teacher as eval()
    module_list[-1].eval()

    if opt.distill == 'abound':
        module_list[1].eval()
    elif opt.distill == 'factor':
        module_list[2].eval()

    criterion_cls = criterion_list[0]
    criterion_div = criterion_list[1]
    criterion_kd = criterion_list[2]

    model_s = module_list[0]
    model_t = module_list[-1]

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    end = time.time()
    for idx, data in enumerate(train_loader):
        if opt.distill in ['crd']:
            input, target, index, contrast_idx = data
        else:
            input, target, index = data
        data_time.update(time.time() - end)

        input = input.float()
        if torch.cuda.is_available():
            input = input.cuda()
            target = target.cuda()
            index = index.cuda()
            dis_target_classes = dis_target_classes.cuda()
            if opt.distill in ['crd']:
                contrast_idx = contrast_idx.cuda()

        # ===================forward=====================
        preact = False
        if opt.distill in ['abound']:
            preact = True
        feat_s, logit_s = model_s(input, is_feat=True, preact=preact)

        with torch.no_grad():
            feat_t, logit_t = model_t(input, is_feat=True, preact=preact)
            logit_t = torch.nn.functional.softmax(logit_t[..., dis_target_classes])
            feat_t = [f.detach() for f in feat_t]

        # cls + kl div
        loss_cls = criterion_cls(logit_s, target)
        loss_div = criterion_div(logit_s, logit_t)

        # other kd beyond KL divergence
        if opt.distill == 'kd':
            loss_kd = 0
        elif opt.distill == 'hint':
            f_s = module_list[1](feat_s[opt.hint_layer])
            f_t = feat_t[opt.hint_layer]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'crd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t, index, contrast_idx)
        elif opt.distill in ['bs', 'bs_t', 'bs_dis', 'bs_dis_t']:
            if opt.bs_feature == 'feature':
                f_s = feat_s[-1]
                f_t = feat_t[-1]
            elif opt.bs_feature == 'logits':
                f_s = logit_s
                f_t = logit_t
            else:
                raise NotImplementedError(opt.bs_feature)
            y = target
            loss_kd = criterion_kd(f_s, f_t, y)
        elif opt.distill == 'attention':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'nst':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'similarity':
            g_s = [feat_s[-2]]
            g_t = [feat_t[-2]]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'rkd':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'pkt':
            f_s = feat_s[-1]
            f_t = feat_t[-1]
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'kdsvd':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = criterion_kd(g_s, g_t)
            loss_kd = sum(loss_group)
        elif opt.distill == 'correlation':
            f_s = module_list[1](feat_s[-1])
            f_t = module_list[2](feat_t[-1])
            loss_kd = criterion_kd(f_s, f_t)
        elif opt.distill == 'vid':
            g_s = feat_s[1:-1]
            g_t = feat_t[1:-1]
            loss_group = [c(f_s, f_t) for f_s, f_t, c in zip(g_s, g_t, criterion_kd)]
            loss_kd = sum(loss_group)
        elif opt.distill == 'abound':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'fsp':
            # can also add loss to this stage
            loss_kd = 0
        elif opt.distill == 'factor':
            factor_s = module_list[1](feat_s[-2])
            factor_t = module_list[2](feat_t[-2], is_factor=True)
            loss_kd = criterion_kd(factor_s, factor_t)
        else:
            raise NotImplementedError(opt.distill)

        loss = opt.gamma * loss_cls + opt.alpha * loss_div + opt.beta * loss_kd

        acc1, acc5 = accuracy(logit_s, target, topk=(1, 5))
        losses.update(loss.item(), input.size(0))
        top1.update(acc1[0], input.size(0))
        top5.update(acc5[0], input.size(0))

        # ===================backward=====================
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ===================meters=====================
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if idx % opt.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                epoch, idx, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5))
            sys.stdout.flush()

    if opt.trainable_T:
        T = 1.0/(criterion_div.T_trans.TT.data.exp())
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f} T {T: .3f}'
              .format(top1=top1, top5=top5, T=T))
    else:
        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))


    return top1.avg, losses.avg


def validate_partial(val_loader, model, criterion, dis_target_classes, opt):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()
                dis_target_classes = dis_target_classes.cuda()
                dis_target_classes = dis_target_classes.type(target.type())

            # compute output
            output = model(input)
            output = torch.nn.functional.softmax(output[..., dis_target_classes], dim=-1)

            loss = criterion(output, target)

            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

        print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg, top5.avg, losses.avg

def validate(val_loader, model, criterion, opt, accuracy = None):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if accuracy is not None:
                acc = accuracy(output, target)
            else:
                acc = torch.Tensor([0.0])
            losses.update(loss.item(), input.size(0))
            top.update(acc.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top.val:.3f} ({top.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top=top))

        print(' * Acc {top.avg:.3f}'.format(top=top))

    return top.avg, losses.avg

def get_key_point(target_in):
    coords = np.argwhere(target_in[0,0,...].cpu())
    if coords.shape[0] != np.ndim(target_in[0,0,...]):
        coords = coords.transpose(0,1)

    centre_coord = []
    for i in range(np.ndim(target_in[0,0,...])):
        c_, _ = torch.unique(coords[i]).sort()
        centre_coord += [c_[len(c_)//2]]

    return centre_coord



def validate_with_image(val_loader, model, criterion, opt, accuracy = None):
    """validation"""
    batch_time = AverageMeter()
    losses = AverageMeter()
    top = AverageMeter()

    # switch to evaluate mode
    model.eval()

    with torch.no_grad():
        end = time.time()
        for idx, (input, target) in enumerate(val_loader):

            input = input.float()
            if torch.cuda.is_available():
                input = input.cuda()
                target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            if accuracy is not None:
                acc = accuracy(output, target)
            else:
                acc = torch.Tensor([0.0])
            losses.update(loss.item(), input.size(0))
            top.update(acc.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if idx % opt.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Acc {top.val:.3f} ({top.avg:.3f})'.format(
                       idx, len(val_loader), batch_time=batch_time, loss=losses,
                       top=top))

        print(' * Acc {top.avg:.3f}'.format(top=top))

    if not opt.visualize_segmentations:
        return top.avg, losses.avg
    else:
        key_point = get_key_point(target)
        disp_images= []
        disp_image_names = [
            'View1/input',
            'View1/ground-truth',
            'View1/prediction',
            'View2/input',
            'View2/ground-truth',
            'View2/prediction',
            'View3/input',
            'View3/ground-truth',
            'View3/prediction'
        ]

        output = torch.argmax(output, dim=1).unsqueeze(1)

        # disp_images = OrderedDict([])
        #
        # # view 1
        # disp_images.update({disp_image_names[0]: input[0, -1, key_point[0], ...]})
        # disp_images.update({disp_image_names[1]: target[0, -1, key_point[0], ...]})
        # disp_images.update({disp_image_names[2]: output[0, -1, key_point[0], ...]})
        # # view 2
        # disp_images.update({disp_image_names[4]: input[0, -1, :, key_point[1], :]})
        # disp_images.update({disp_image_names[5]: target[0, -1, :, key_point[1], :]})
        # disp_images.update({disp_image_names[6]: output[0, -1, :, key_point[1], :]})
        # # view 3
        # disp_images.update({disp_image_names[7]: input[0, -1, :, :, key_point[2]]})
        # disp_images.update({disp_image_names[8]: target[0, -1, :, :, key_point[2]]})
        # disp_images.update({disp_image_names[9]: output[0, -1, :, :, key_point[2]]})

        # view 1
        disp_images += [input[0, -1, key_point[0], ...]]
        disp_images += [target[0, -1, key_point[0], ...]]
        disp_images += [output[0, -1, key_point[0], ...]]

        # view 2
        disp_images += [input[0, -1, :, key_point[1], :]]
        disp_images += [target[0, -1, :, key_point[1], :]]
        disp_images += [output[0, -1, :, key_point[1], :]]

        disp_images += [input[0, -1, :, :, key_point[2]]]
        disp_images += [target[0, -1, :, :, key_point[2]]]
        disp_images += [output[0, -1, :, :, key_point[2]]]

        return top.avg, losses.avg, disp_images, disp_image_names

