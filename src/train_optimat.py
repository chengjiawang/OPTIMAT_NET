from __future__ import print_function

import os
import argparse
import socket
import time

import tensorboard_logger as tb_logger
import torch
import torch.optim as optim
import torch.nn as nn
import torch.backends.cudnn as cudnn

from models import model_dict

from dataset.cifar100 import get_cifar100_dataloaders

from helper.util import adjust_learning_rate, accuracy, AverageMeter
from helper.loops import train_vanilla as train, validate, validate_with_image

from options.option import parse_option
from dataset.OPTIMAT import get_OPTIMAT_loaders
from dataset import OPTIMAT

from models.criterions import BinaryDiceLoss, DiceIndex

from torch.utils.tensorboard import SummaryWriter
from helper.util import load_network

def main():
    best_acc = 0

    opt = parse_option()
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu

    # dataloader
    if opt.dataset == 'optimat1':
        train_loader, val_loader = \
            get_OPTIMAT_loaders(
                opt=opt,
                data_root=opt.dataroot,
                batch_size=opt.batch_size,
                num_workers=opt.num_workers)
        n_cls = 2

    else:
        raise NotImplementedError(opt.dataset)

    # model
    model = model_dict[opt.model](num_classes=n_cls)

    # optimizer
    # optimizer = optim.SGD(model.parameters(),
    #                       lr=opt.learning_rate,
    #                       momentum=opt.momentum,
    #                       weight_decay=opt.weight_decay)

    optimizer = optim.Adam(
        model.parameters(),
        lr=opt.learning_rate,
        betas=(0.5, 0.999)
    )

    criterion_bce = nn.BCEWithLogitsLoss()
    criterion_dice = BinaryDiceLoss()
    accuracy = DiceIndex()

    if torch.cuda.is_available():
        model = model.cuda()
        criterion_dice = criterion_dice.cuda()
        cudnn.benchmark = True

    if opt.continue_train:
        if opt.which_epoch is not None:
            load_path = os.path.join(
                opt.save_folder,
                'ckpt_epoch_{epoch}.pth'.format(epoch=opt.which_epoch))
        else:
            load_path = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
        state = torch.load(load_path)
        load_network(model, state['model'])

    # tensorboard
    # logger = tb_logger.Logger(logdir=opt.tb_folder, flush_secs=2)
    writer = SummaryWriter(log_dir=opt.tb_folder,
                           flush_secs = 2)
    # image_writer = SummaryWriter(
    #     log_dir=os.path.join(
    #         opt.tb_folder,
    #         'images'
    #     ), flush_secs=2
    # )


    # routine
    if opt.dynamic_batch_dim:
        import numpy as np
        dim1s = np.arange(96, 176 + 1, step=16)
        dim2s = np.arange(96, 224 + 1, step=16)

    for epoch in range(1, opt.epochs + 1):

        adjust_learning_rate(epoch, opt, optimizer)
        print("==> training...")

        time1 = time.time()
        if opt.dynamic_batch_dim and epoch >1:
            # imagesizes

            dim1 = np.random.choice(dim1s)
            dim2 = np.random.choice(dim2s)

            train_loader.set_image_size([dim1, dim2])
            train_loader.set_batch_size(
                int((176*224) // (dim1*dim2) * opt.batch_size)
            )

            print('image_size: (', dim1, dim2, ')')
            print('batch_size: ', train_loader.batch_size)

        train_acc, train_loss = train(epoch,
                                      train_loader,
                                      model,
                                      criterion_dice,
                                      optimizer,
                                      opt,
                                      accuracy=accuracy)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        # logger.log_value('train_acc', train_acc, epoch)
        # logger.log_value('train_loss', train_loss, epoch)

        writer.add_scalar('Train/train_acc', train_acc, epoch)
        writer.add_scalar('Train/train_loss', train_loss, epoch)


        if not opt.visualize_segmentations:
            test_acc, test_loss = validate(val_loader,
                                       model,
                                       criterion_dice,
                                       opt,
                                       accuracy=accuracy)
            # logger.log_value('test_acc', test_acc, epoch)
            # logger.log_value('test_loss', test_loss, epoch)
            writer.add_scalar('Test/test_acc', test_acc, epoch)
            writer.add_scalar('Test/test_loss', test_loss, epoch)
        else:
            test_acc, test_loss, disp_images, disp_image_names = validate_with_image(
                val_loader,
                model,
                criterion_dice,
                opt,
                accuracy=accuracy
            )
            # logger.log_value('test_acc', test_acc, epoch)
            # logger.log_value('test_loss', test_loss, epoch)
            writer.add_scalar('Test/test_acc', test_acc, epoch)
            writer.add_scalar('Test/train_loss', test_loss, epoch)

            for n, name in enumerate(disp_image_names):
                print(name)
                writer.add_image(
                    tag=name,
                    img_tensor=disp_images[n].unsqueeze(0).expand(3, -1, -1),
                    global_step=0
                )

        # save the best model
        if test_acc > best_acc:
            best_acc = test_acc
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'best_acc': best_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, '{}_best.pth'.format(opt.model))
            print('saving the best model!')
            torch.save(state, save_file)

        # regular saving
        if epoch % opt.save_freq == 0:
            print('==> Saving...')
            state = {
                'epoch': epoch,
                'model': model.state_dict(),
                'accuracy': test_acc,
                'optimizer': optimizer.state_dict(),
            }
            save_file = os.path.join(opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            torch.save(state, save_file)

    # This best accuracy is only for printing purpose.
    # The results reported in the paper/README is from the last epoch.
    print('best accuracy:', best_acc)

    # save model
    state = {
        'opt': opt,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }
    save_file = os.path.join(opt.save_folder, '{}_last.pth'.format(opt.model))
    torch.save(state, save_file)
    writer.close()


if __name__ == '__main__':
    main()
