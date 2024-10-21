import os.path
import random
import torch
import torch.nn.functional as F

import nibabel as nib
import glob

import numpy as np
import scipy.ndimage as ndi
from PIL import Image
from nibabel.processing import resample_to_output

from data.base_dataset import BaseDataset

from aug_transforms.distortion_2D import RandomElasticDeform2D
from aug_transforms.get_transforms import get_transform_2D, get_basic_transform2D
from data.image_readers import nifti_reader_3D


def _make_dataset3D(dir):
    print(dir)
    assert os.path.isdir(dir)
    images = glob.glob(os.path.join(dir, '*'))
    return images

def Setsize(img, scale_factor, order=3):
    return ndi.zoom(img, zoom=scale_factor, order=order)

def crop_1D(dat, axis):
    # argwhere will give you the coordinates of every non-zero point
    true_points = np.argwhere(dat)
    # take the smallest points and use them as the top left of your crop
    top_left = true_points.min(axis=0)
    # take the largest points and use them as the bottom right of your crop
    bottom_right = true_points.max(axis=0)

    datc = np.take(dat,
                   np.arange(start=top_left[axis], stop=bottom_right[axis]+1),
                   axis=axis)
    return datc

def crop_3D(dat):
    datc = crop_1D(dat, axis=0)
    datc = crop_1D(datc, axis=1)
    datc = crop_1D(datc, axis=2)
    return datc

def crop_center3D(img,cropx,cropy):
    y,x, _ = img.shape
    startx = x//2-(cropx//2)
    starty = y//2-(cropy//2)
    return img[starty:starty+cropy,startx:startx+cropx, ...]

def normalize3D2D(dat):
    assert(isinstance(dat, np.ndarray))
    max_vec = np.max(dat, axis=(0,1), keepdims=True)
    min_vec = np.min(dat, axis=(0,1), keepdims=True)
    return (dat-min_vec)/(max_vec - min_vec + 1e-12)

def _pad_to_shape_3D2D(vol_in, target_size):
    axial_paddings = target_size - np.array(vol_in.shape[:2])
    axial_paddings = np.max(
        np.stack([axial_paddings, np.zeros(2)]),
        axis=0
    )
    axial_paddings = axial_paddings.astype('int32')

    return np.pad(vol_in, [(axial_paddings[0] // 2,
                                         axial_paddings[0] - axial_paddings[0] // 2),
                                         (axial_paddings[1] // 2,
                                         axial_paddings[1] - axial_paddings[1] // 2),
                                         (0, 0)],
                               mode='edge')

def _center_crop_to_shape3D2D(vol_in, target_shape):
    if not isinstance(target_shape, list):
        target_shape = np.array([target_shape, target_shape])
    half_dist = target_shape//2
    lens = np.array(vol_in.shape[:2])
    centres = lens//2
    starts = centres - half_dist
    ends = starts + target_shape
    return vol_in[
        max(0, starts[0]):min(lens[0], ends[0]),
           max(0, starts[1]):min(lens[1], ends[1]),
           ...
    ]

def _pad_to_shape_3D(vol_in, target_size):
    axial_paddings = target_size - np.array(vol_in.shape[:2])
    axial_paddings = np.max(
        np.stack([axial_paddings, np.zeros(2)]),
        axis=0
    )
    axial_paddings = axial_paddings.astype('int32')

    return np.pad(vol_in, [(axial_paddings[0] // 2,
                            axial_paddings[0] - axial_paddings[0] // 2),
                            (axial_paddings[1] // 2,
                            axial_paddings[1] - axial_paddings[1] // 2),
                           (axial_paddings[2] // 2,
                            axial_paddings[2] - axial_paddings[2] // 2)],
                               mode='edge')

def _pad_to_shape_for_downsample3D(vol_in, downsample_times = 2):
    # vol_in = vol_in
    timerate = 2**downsample_times
    minval = vol_in.min()
    oshapes = np.array(list(vol_in.shape[-3:]))
    pads = (timerate - (oshapes%timerate))%timerate
    if isinstance(vol_in, np.ndarray):
        if pads.sum()>0:
            vol_in = np.pad(vol_in,
                          ((0, int(pads[0])), (0, int(pads[1])), (0, int(pads[2]))),
                          mode='edge'
                          )
        return vol_in

    if pads.sum()>0:
        # print(vol_in.shape)
        return F.pad(vol_in.unsqueeze(0),
                     (0, int(pads[2]), 0, int(pads[1]), 0, int(pads[0])),
                     "replicate").squeeze(0)
    else:
        return vol_in

class MRBrainS3D2D(BaseDataset):

    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        self.dir_AB = opt.dataroot
        self.batch_size = opt.batchSize

        self.AB_paths = _make_dataset3D(self.dir_AB)

        # print(self.AB_paths)
        # print(self.AB_paths)
        self.num_pairs = len(self.AB_paths)  # just for paired
        self.A_folder = os.path.join(self.dir_AB, 'trainingData')
        self.B_folder = os.path.join(self.dir_AB, 'trainingData')
        self.A_filename = os.path.join('pre', 'reg_IR.nii.gz')
        self.B_filename = os.path.join('pre', 'FLAIR.nii.gz')
        self.mask_name = 'segm.nii.gz'

        self.A_paths = _make_dataset3D(self.A_folder)
        # self.B_paths = _make_dataset3D(self.B_folder)

        self.A_paths.sort()
        self.B_paths = self.A_paths

        self.A_num = len(self.A_paths)
        self.B_num = len(self.B_paths)

        # be careful with the numpy array and tensor operations
        self.transform_A = get_basic_transform2D()
        self.transform_B = get_transform_2D(self.opt)
        if self.opt.registration_mode.lower() in {'free', 'b', 'thinplate'}:
            self.distort = RandomElasticDeform2D()
        # self.transformB = get_transform(self.opt)

        # self.transform = transforms.Compose(transform_list)

        self.num_cache_volumes = opt.num_cache_volumes
        self.num_cache_steps = opt.num_cache_steps//self.batch_size * self.batch_size
        self.cached_volumeA = np.zeros((opt.fineSize,opt.fineSize,1))
        self.cached_volumeB = np.zeros((opt.fineSize,opt.fineSize,1))
        self.cache_counter = 0
        self.slice_num = None

    def re_cache(self):
        self.selected_A = np.random.choice(a=np.arange(start=0,
                                                        stop=self.A_num,
                                                        step=self.num_cache_volumes),
                                            size=1,
                                            replace=True)[0]

        # selecrted B
        self.selected_B = self.selected_A

        # cache volumes
        self.cached_volumeA = None
        self.cached_volumeB = None

        # for i in range(len(self.selected_As)):
        self.A_path = os.path.join(
            self.A_paths[self.selected_A], self.A_filename
        )
        self.B_path = os.path.join(
            self.B_paths[self.selected_B], self.B_filename
        )
        self.mask_path = os.path.join(
            self.A_paths[self.selected_A], self.mask_name
        )

        print(self.A_path)

        temp_volumeA_data, temp_volumeA_header = nifti_reader_3D(
            self.A_path
        )

        temp_volumeB_data, temp_volumeB_header = nifti_reader_3D(
            self.B_path
        )

        temp_mask_data, temp_mask_header = nifti_reader_3D(
            self.mask_path
        )


        # # 1. calculate dimensions of uniform
        # temp_volumeA_data = temp_volumeA.get_data()
        # temp_volumeB_data = temp_volumeB.get_data()

        # crop z and normalize
        temp_volumeA_data = normalize3D2D(temp_volumeA_data)
        temp_volumeB_data = normalize3D2D(temp_volumeB_data)
        # temp_mask_data = temp_mask_data > 0
        temp_volumeA_data[temp_mask_data<=0] = temp_volumeA_data.min()
        temp_volumeB_data[temp_mask_data<=0] = temp_volumeB_data.min()
        # zoom to res
        scale_factorA = 1.0/self.opt.resolution
        scale_factorB = 1.0/self.opt.resolution

        temp_volumeA_data = ndi.zoom(temp_volumeA_data,
                                     zoom=[scale_factorA, scale_factorA, scale_factorA], order=3, mode='nearest')
        temp_volumeB_data = ndi.zoom(temp_volumeB_data,
                                     zoom=[scale_factorB, scale_factorB, scale_factorB], order=3, mode='nearest')

        temp_volumeA_data = _pad_to_shape_3D2D(
            temp_volumeA_data, self.opt.fineSize
        )

        temp_volumeB_data = _pad_to_shape_3D2D(
            temp_volumeB_data, self.opt.fineSize
        )

        # crop to uniform shape
        temp_volumeA_data = _center_crop_to_shape3D2D(
            temp_volumeA_data, self.opt.fineSize
        )
        temp_volumeB_data = _center_crop_to_shape3D2D(
            temp_volumeB_data, self.opt.fineSize
        )

        # temp_volumeA_data = _pad_to_shape_for_downsample3D(
        #     temp_volumeA_data
        # )
        #
        # temp_volumeB_data = _pad_to_shape_for_downsample3D(
        #     temp_volumeB_data
        # )

        if self.cached_volumeA is not None:
            self.cached_volumeA = np.concatenate((self.cached_volumeA, temp_volumeA_data),
                                                 axis=-1)
            self.cached_volumeB = np.concatenate((self.cached_volumeB, temp_volumeB_data),
                                                 axis=-1)
        else:
            self.cached_volumeA = temp_volumeA_data
            self.cached_volumeB = temp_volumeB_data

        self.slice_num = min(
            temp_volumeA_data.shape[-1],
            temp_volumeB_data.shape[-1]
        )


        self.cache_counter += 1
        print(self.cache_counter)

    def __getitem__(self, index):
        # AB_path has "A" and "B" folder
        # then A is saved in 'A' B in 'B'
        # if self.cache_counter%self.num_cache_steps==0:

        # cache volumes
        # cache volumes
        if int(self.cache_counter) > int(self.num_cache_steps) and \
                int(self.cache_counter) % self.batch_size == 0 and \
                self.opt.no_recache == 0:
            # select cached volumes
            self.re_cache()
            self.cache_counter = 0
        elif int(self.cache_counter) == 0:
            self.re_cache()
        self.cache_counter += 1

        index = index%self.slice_num
        # indexA_max = self.cached_volumeA.shape[-1]
        # indexB_max = self.cached_volumeB.shape[-1]

        # change later
        # indexA = indexB = np.random.randint(low=1, high=min(indexA_max, indexB_max))

        indexA = indexB = index
        # print(index)

        A = self.cached_volumeA[:, :, indexA]
        B = self.cached_volumeB[:, :, indexB]

        while A.std() == 0 or B.std() == 0:
            indexA_max = self.cached_volumeA.shape[-1]
            indexB_max = self.cached_volumeB.shape[-1]
            indexA = indexB = np.random.randint(
                low=1,
                high=min(indexA_max, indexB_max)
            )
            A = self.cached_volumeA[:, :, indexA]
            B = self.cached_volumeB[:, :, indexB]


        if np.any(np.isnan(A)) or np.any(np.isnan(B)):
            print(A.shape)
            print(B.shape)
            print(index)
            print('nan in cashed volume')
            A[np.isnan(A)] = -1
            B[np.isnan(B)] = -1

        if self.opt.registration_mode.lower() in {'free', 'b', 'thinplate'}:
            B = self.distort(B)

            if np.any(np.isnan(B)):
                print(B.shape)
                print(index)
                print('nan after distort')
                B[np.isnan(B)] = -1

        A = Image.fromarray(A)
        B = Image.fromarray(B)

        A = self.transform_A(A)
        B = self.transform_B(B)

        if torch.sum(torch.isnan(A)) or torch.sum(torch.isnan(B)):
            print(A.shape)
            print(B.shape)
            print(index)
            print('nan after transform')
            A[torch.isnan(A)] = -1
            B[torch.isnan(B)] = -1


        # A = _pad_to_shape_for_downsample3D(
        #     A
        # )
        #
        # B = _pad_to_shape_for_downsample3D(
        #     B
        # )
        #

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(A.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            A = A.index_select(2, idx)
            B = B.index_select(2, idx)
            if torch.sum(torch.isnan(A)) or torch.sum(torch.isnan(B)):
                print(A.shape)
                print(B.shape)
                print(index)
                print('nan after random flip')
                A[torch.isnan(A)] = -1
                B[torch.isnan(B)] = -1

        return {'A': A, 'B': B,
                'A_paths': self.A_path, 'B_paths':self.B_path}

    def __len__(self):
        return len(self.A_paths)*48

    def name(self):
        return 'MRBrainS3D2D'
