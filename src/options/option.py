import socket
import argparse

import os


def parse_option():
    hostname = socket.gethostname()

    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument(
        '--gpu',
        type=str,
        default='1',
        help='gpu num'
    )

    parser.add_argument(
        '--dataroot',
        type=str,
        default='./OPTIMATData/OPTIMAT1',
        help='dataset path'
    )

    parser.add_argument(
        '--augment_option',
        type=str,
        default='affine',
        choices=['affine', 'free', 'thinplate', 'none']
    )

    parser.add_argument(
        '--num_cache_volumes',
        type=int,
        default=2,
        help='number of cached volumes'
    )

    parser.add_argument(
        '--visualize_segmentations',
        action='store_true',
        help='whether or not to display the predicted segmentation'
    )

    parser.add_argument(
        '--num_cache_steps',
        type=int,
        default=None,
        help='number of cached steps'
    )

    parser.add_argument(
        '--resolution',
        type=float,
        default=1.0,
        help='ratio of the original resolution'
    )

    parser.add_argument(
        '--no_recache',
        type=int,
        default=0,
        help='stop recaching new volumes for debugging purpose only'
    )

    parser.add_argument(
        '--no_flip',
        type=bool,
        default=False,
        help='whether to flip in augmentation'
    )
    parser.add_argument('--continue_train', action='store_true', help='whether to continue train or net')
    parser.add_argument('--which_epoch', type=int, default=None, help='number of epoch to load')
    parser.add_argument('--dynamic_batch_dim', type=bool, default=True, help='use dynamically batch_size and image size')
    parser.add_argument('--print_freq', type=int, default=1, help='print frequency')
    parser.add_argument('--tb_freq', type=int, default=10, help='tb frequency')
    parser.add_argument('--save_freq', type=int, default=2, help='save frequency')
    parser.add_argument('--batch_size', type=int, default=4, help='batch_size')
    parser.add_argument('--num_workers', type=int, default=1, help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=2000, help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

    # dataset
    parser.add_argument('--model', type=str, default='unet3D',
                        choices=['unet3D', 'unet3D_attention_dsup', 'unet3D_dsup', 'unet3D_attention', 'sdnet', 'attention_unet', 'unet3D_CBAMattentionzoom'])

    parser.add_argument('--dataset',
                        type=str,
                        default='Epi',
                        choices=['Spine', 'Dia', 'Epi', 'Head', 'Neck', 'TotalHip', 'TotalHipO'],
                        help='dataset')

    parser.add_argument('-t', '--trial',
                        type=int, default=0,
                        help='the experiment id')

    opt = parser.parse_args()

    # set the path according to the environment

    opt.model_path = './save/models'
    opt.tb_path = './save/tensorboard'
    opt.tb_path = os.path.join(
        opt.tb_path, opt.dataset
    )

    if not os.path.isdir(opt.tb_path):
        os.makedirs(opt.tb_path)


    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = '{}_{}_lr_{}_decay_{}_trial_{}'.format(opt.model, opt.dataset, opt.learning_rate,
                                                            opt.weight_decay, opt.trial)

    opt.tb_folder = os.path.join(opt.tb_path, opt.model_name)
    if not os.path.isdir(opt.tb_folder):
        os.makedirs(opt.tb_folder)

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)

    # opt.dataroot = os.path.join(
    #     opt.dataroot, opt.dataset
    # )
    # opt.save_folder = os.path.join(
    #     opt.save_folder, opt.dataset
    # )

    return opt