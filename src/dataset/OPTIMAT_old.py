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

from torch.utils.data import DataLoader
# from .base_dataset import BaseDataset
from aug_transforms.get_transforms import get_transform_3D, get_basic_transform3D, get_norm_transform3D
from aug_transforms.distortion_3D import RandomElasticDeform3D, RandomElasticDeform3D_pytorch
from dataset.image_readers import nifti_reader_3D
from dataset.normalizationsF import (normalize_01, normalize_white)


def _make_dataset3D(dir, key_word='*'):
    assert os.path.isdir(dir)
    images = glob.glob(
        os.path.join(
            dir, '**', '*'+key_word
        )
        , recursive=True
    )
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
                               mode='minimum')

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
                               mode='minimum')

def _pad_to_shape_for_downsample3D(vol_in, downsample_times = 3):
    # vol_in = vol_in
    timerate = 2**downsample_times
    minval = vol_in.min()
    oshapes = np.array(list(vol_in.shape[-3:]))
    pads = (timerate - (oshapes%timerate))%timerate
    # print(pads)
    if isinstance(vol_in, np.ndarray):
        if pads.sum()>0:
            vol_in = np.pad(vol_in,
                          ((0, int(pads[0])), (0, int(pads[1])), (0, int(pads[2]))),
                          mode='minimum'
                          )
        return vol_in

    if pads.sum()>0:
        # print(vol_in.shape)
        return F.pad(vol_in.unsqueeze(0),
                     (0, int(pads[2]), 0, int(pads[1]), 0, int(pads[0])),
                     "replicate").squeeze(0)
    else:
        return vol_in

def make_one_hot(labels, C=2):
    '''
    Converts an integer label torch.autograd.Variable to a one-hot Variable.

    Parameters
    ----------
    labels : torch.autograd.Variable of torch.cuda.LongTensor
        N x 1 x H x W, where N is batch size.
        Each value is an integer representing correct classification.
    C : integer.
        number of classes in labels.

    Returns
    -------
    target : torch.autograd.Variable of torch.cuda.FloatTensor
        N x C x H x W, where C is class number. One-hot encoded.
    '''

    shape = list(labels.shape)
    shape[1] = C
    if torch.cuda.is_available():
        one_hot = torch.cuda.FloatTensor(*shape).zero_()
        target = one_hot.scatter_(1, labels.type('torch.cuda.LongTensor'), 1.0)
    else:
        one_hot = torch.FloatTensor(*shape).zero_()
        target = one_hot.scatter_(1, labels.type('torch.LongTensor'), 1.0)

    return target

def select_centres2D(label_vol, vol_occur, n_class=None, skip_ratio=4):
    if n_class is None:
        n_class = len(np.unique(label_vol))
    vx, vy = np.meshgrid(
        np.arange(start=np.random.choice(skip_ratio),
                  stop=label_vol.shape[0], step=skip_ratio),
        np.arange(start=np.random.choice(skip_ratio),
                  stop=label_vol.shape[1], step=skip_ratio)
        )
    class_occurs = np.random.choice(
        np.arange(start=1, stop=n_class),
        size=vol_occur
    )

    centres = []

    for cl in class_occurs:
        inds = np.arange(len(label_vol[vx, vy].ravel()))
        inds_t = label_vol[vx, vy].ravel()==cl
        inds = inds*inds_t
        inds = inds[inds!=0]
        sel_cind = np.random.choice(
            inds
        )
        sel_c = np.array([i.ravel()[sel_cind] for i in [vx, vy]])
        centres.append(sel_c)
    return centres, class_occurs

def select_centres(label_vol, vol_occur, n_class=None, skip_ratio=4):
    if n_class is None:
        n_class = len(np.unique(label_vol))
    vx, vy, vz = np.meshgrid(
        np.arange(start=np.random.choice(skip_ratio),
                  stop=label_vol.shape[0], step=skip_ratio),
        np.arange(start=np.random.choice(skip_ratio),
                  stop=label_vol.shape[1], step=skip_ratio),
        np.arange(start=np.random.choice(skip_ratio),
                  stop=label_vol.shape[2], step=skip_ratio)
        )
    class_occurs = np.random.choice(
        np.arange(start=1, stop=n_class),
        size=vol_occur
    )

    centres = []

    for cl in class_occurs:
        inds = np.arange(len(label_vol[vx, vy, vz].ravel()))
        inds_t = label_vol[vx, vy, vz].ravel()==cl
        inds = inds*inds_t
        # gc.collect()
        inds = inds[inds!=0]
        sel_cind = np.random.choice(
            inds
        )
        # gc.collect()
        sel_c = np.array([i.ravel()[sel_cind] for i in [vx, vy, vz]])
        # gc.collect()
        centres.append(sel_c)

    return centres, class_occurs


def extract_subvolumn(vol, centre_coord, subvolumn_size, limit_within_vol=True):

    vol_minx, vol_maxx, target_minx, target_maxx = \
        extract_sub_range(vol.shape[0],
                               centre_coord[0],
                               subvolumn_size[0],
                               limit_within_vol=limit_within_vol)
    vol_miny, vol_maxy, target_miny, target_maxy = \
        extract_sub_range(vol.shape[1],
                               centre_coord[1],
                               subvolumn_size[1],
                               limit_within_vol=limit_within_vol)
    vol_minz, vol_maxz, target_minz, target_maxz = \
        extract_sub_range(vol.shape[2],
                               centre_coord[2],
                               subvolumn_size[2],
                               limit_within_vol=limit_within_vol)


    sub_vol = np.ones(subvolumn_size)*np.min(vol)
    sub_vol[target_minx:target_maxx, target_miny:target_maxy, target_minz:target_maxz] = \
        vol[vol_minx:vol_maxx, vol_miny:vol_maxy, vol_minz:vol_maxz]
    # print('vol')

    '''
    plt.figure(1)
    plt.imshow(vol[:, :, centre_coord[2]], cmap='gray')
    plt.plot(centre_coord[1], centre_coord[0], 'ro', color=(0.9, 0.9, 1.0), alpha=0.8)
    plt.figure(2)
    plt.imshow(sub_vol[:, :, 31], cmap='gray')
    plt.show(block=True)
    '''

    return sub_vol

def extract_sub_range(vol_dim, subvol_dim_centre, subvol_dim_size,
                      limit_within_vol = True):
    assert(vol_dim >= subvol_dim_size)
    half_dim_size = subvol_dim_size//2.0
    ind_min = subvol_dim_centre - half_dim_size

    if subvol_dim_size%2 == 0:
        ind_max = subvol_dim_centre + half_dim_size - 1
    else:
        ind_max = subvol_dim_centre + half_dim_size

    if limit_within_vol:
        target_min_coord = 0
        target_max_coord = subvol_dim_size
        if ind_min<0:
            vol_min_coord = 0
            vol_max_coord = subvol_dim_size
        elif ind_max+1 > vol_dim:
            vol_max_coord = vol_dim
            vol_min_coord = vol_dim - subvol_dim_size
        else:
            vol_min_coord = ind_min
            vol_max_coord = ind_max + 1

    else:
        if ind_min <0:
            # print('situlation1')
            target_min_coord = -ind_min
            target_max_coord = subvol_dim_size
            vol_min_coord = 0
            vol_max_coord = subvol_dim_size+ind_min
        elif ind_max+1 > vol_dim:
            # print('situation2')
            target_min_coord = 0
            target_max_coord = vol_dim-ind_max-1 #subvol_dim_size - (ind_max+1-vol_dim)
            vol_min_coord =  -subvol_dim_size - vol_dim+ind_max+1  #vol_dim - ind_max -1
            vol_max_coord = vol_dim
        else:
            # print('situation3')
            target_min_coord = 0
            target_max_coord = subvol_dim_size
            vol_min_coord = ind_min
            vol_max_coord = ind_max+1


    return int(vol_min_coord), \
           int(vol_max_coord), \
           int(target_min_coord), \
           int(target_max_coord)

class OPTIMAT(torch.utils.data.Dataset):
    def __init__(self,
                 opt,
                 train=True,
                 make_one_hot = False
                 ):
        super(OPTIMAT, self).__init__()
        self.train = train
        self.make_one_hot = make_one_hot
        self.initialize(opt=opt)

    def extract_patient_name(self, in_string, final_word):
        return in_string[:-len(final_word)]


    def initialize(self, opt):
        self.opt = opt
        self.path = opt.dataroot
        self.data_key_word = '*Spine.nii'
        self.label_key_word = '*Spine_obj.nii'
        self.__image_size = None
        self.uniform_size = None

        self.train_path = os.path.join(
            self.path, 'train'
        )

        self.test_path = os.path.join(
            self.path, 'test'
        )

        self.train_data_files = _make_dataset3D(
            dir=self.train_path,
            key_word=self.data_key_word
        )
        self.train_data_files.sort()

        self.train_label_files = _make_dataset3D(
            dir=self.train_path,
            key_word=self.label_key_word
        )
        self.train_label_files.sort()

        self.train_arb_p = 0.5

        self.test_data_files = _make_dataset3D(
            self.test_path,
            key_word=self.data_key_word
        )
        self.test_data_files.sort()

        self.test_label_files = _make_dataset3D(
            self.test_path,
            key_word=self.label_key_word
        )
        self.test_label_files.sort()

        self.train_num = len(self.train_data_files)
        self.test_num = len(self.test_data_files)
        self.train_label_num = len(self.train_label_files)
        self.test_label_num = len(self.test_label_files)

        if self.train_num != self.train_label_num or \
            self.test_num != self.test_label_num:
            raise NotImplementedError('Only full supervision now!')

        # be careful with the numpy array and tensor operations
        if self.opt.augment_option.lower() == 'none':
            self.transform = get_basic_transform3D(normalize=False)
        else:
            self.transform = get_transform_3D(self.opt, normalize=False)
        self.test_transform = get_basic_transform3D(normalize=False)
        self.normalize = get_norm_transform3D()


        if self.opt.augment_option.lower() in {'free', 'b', 'thinplate'}:
            self.distort = RandomElasticDeform3D()

        # self.transformB = get_transform(self.opt)

        # self.transform = transforms.Compose(transform_list)

        self.num_cache_volumes = opt.num_cache_volumes
        if opt.num_cache_steps is None:
            self.num_cache_steps = self.__len__()
        else:
            self.num_cache_steps = opt.num_cache_steps
        self.cache_counter = 0

    def re_cache(self):
        # check the selected patients
        self.selected_patient = np.random.choice(a=np.arange(start=0,
                                                        stop=self.train_num),
                                            size=self.num_cache_volumes,
                                            replace=False)

        # selecrted B
        self.selected_train_data = [
            self.train_data_files[i] for i in self.selected_patient
        ]

        self.selected_train_label = [
            self.train_label_files[i] for i in self.selected_patient
        ]

        self.patient_names = [self.extract_patient_name(
            self.train_data_files[i], self.data_key_word
        ) for i in self.selected_patient]

        print('recached: ')
        print(self.patient_names)

        # cache single volume
        image_sizes = []
        # cache volumes
        self.cached_volumes = []
        self.cached_labels = []
        for n_patient,patient in enumerate(self.patient_names):

            train_data = self.selected_train_data[n_patient]
            train_label = self.selected_train_label[n_patient]

            print('caching ', train_data, '...')

            train_volume_data, train_volume_header = nifti_reader_3D(
                train_data
            )

            train_volume_label, train_label_header = nifti_reader_3D(
                train_label
            )

            # # 1. calculate dimensions of uniform
            # temp_volumeA_data = temp_volumeA.get_data()
            # temp_volumeB_data = temp_volumeB.get_data()

            # crop z and normalize
            # temp_volumeA_data = normalize3D2D(temp_volumeA_data)
            # temp_volumeB_data = normalize3D2D(temp_volumeB_data)

            # zoom to res
            if hasattr(self.opt, 'resolution') and self.opt.resolution !=1:
                scale_factorA = 1.0/self.opt.resolution
                scale_factorB = 1.0/self.opt.resolution

                train_volume_data = ndi.zoom(train_volume_data,
                                             zoom=[scale_factorA,
                                                   scale_factorA,
                                                   scale_factorA],
                                             order=3, mode='constant')

                train_volume_label = ndi.zoom(train_volume_label,
                                             zoom=[scale_factorB,
                                                   scale_factorB,
                                                   scale_factorB],
                                              order=3, mode='constant')

            # temp_volumeA_data = _pad_to_shape_3D2D(
            #     temp_volumeA_data, self.opt.fineSize
            # )
            #
            # temp_volumeB_data = _pad_to_shape_3D2D(
            #     temp_volumeB_data, self.opt.fineSize
            # )

            # normalization
            train_volume_data = normalize_01(
                train_volume_data
            )

            train_volume_data = _pad_to_shape_for_downsample3D(
                train_volume_data
            )

            train_volume_label = _pad_to_shape_for_downsample3D(
                train_volume_label
            )

            train_volume_data = normalize_white(train_volume_data)

            self.cached_volumes += [train_volume_data]
            self.cached_labels += [train_volume_label]

            # get a proper subvolume size
            image_sizes += [
                np.array(train_volume_data.shape)
            ]

        self.uniform_size = np.array(image_sizes).min(axis=0)
        # if self.image_size is None:
        #     self.image_size = self.uniform_size
        # self.cached_volumes = self.distort(self.cached_volume)

        self.cache_counter += 1


        print(self.cache_counter)

    def __getitem__(self, index):
        if self.train:
            return self.get_item_train(index)
        else:
            return self.get_item_test(index)

    @property
    def image_size(self):
        return self.__image_size
    @image_size.setter
    def image_size(self, value):
        self.__image_size = value
    @image_size.deleter
    def image_size(self):
        self.__image_size = None

    def get_item_train(self, index):
        # AB_path has "A" and "B" folder
        # then A is saved in 'A' B in 'B'
        # if self.cache_counter%self.num_cache_steps==0:

        # cache volumes
        # cache volumes

        if int(self.cache_counter) % int(self.num_cache_steps) == 0 and self.opt.no_recache == 0:
            # select cached volumes
            self.re_cache()
        elif int(self.cache_counter) == 0:
            self.re_cache()
        self.cache_counter += 1

        volume_index = np.random.choice(
            self.num_cache_volumes,
            # size=self.opt.batch_size,
            # replace=True
        )
        print('numv: ', len(self.cached_volumes))
        print('numv: ', self.num_cache_volumes)
        print('numl:', volume_index)

        cached_volume = self.cached_volumes[volume_index]
        cached_label = self.cached_labels[volume_index]
        # path size for training
        # forground should be apears according to probability p: self.train_arb_p
        # then for the rest 1-p it should be randomly picked
        # for p, use the arbitrary centres

        # select centres
        if self.__image_size is None:
            sub_vol_data = cached_volume[
                           :self.uniform_size[0],
                           :self.uniform_size[1],
                           :self.uniform_size[2]
                           ]

            sub_vol_label = cached_label[
                            :self.uniform_size[0],
                            :self.uniform_size[1],
                            :self.uniform_size[2]
                            ]
        else:
            if random.random() < self.train_arb_p:
                sub_vol_centre = self.select_centres2D(
                    cached_label
                )

                sub_vol_data = self.extract_subvolumn(
                    cached_volume, sub_vol_centre, self.__image_size[:2]
                )

                sub_vol_label = self.extract_subvolumn(
                    cached_label, sub_vol_centre, self.__image_size[:2]
                )

            else:
                cached_size = np.array(cached_volume.shape)
                valid_range = cached_size[:2] - np.array(self.__image_size)
                x_start = np.random.randint(valid_range[0])
                y_start = np.random.randint(valid_range[1])

                sub_vol_data = cached_volume[
                    x_start:x_start+self.image_size[0],
                    y_start:y_start+self.image_size[1],
                    ...
                ]
                sub_vol_label = cached_label[
                    x_start:x_start + self.image_size[0],
                    y_start:y_start + self.image_size[1],
                    ...
                ]

        temp = self.transform([sub_vol_data, sub_vol_label])
        sub_vol_data = temp[0]
        sub_vol_data = self.normalize(sub_vol_data)
        sub_vol_label = temp[1]

        sub_vol_data = sub_vol_data.unsqueeze(0)
        sub_vol_label = sub_vol_label.unsqueeze(0)

        if self.make_one_hot:
            sub_vol_label = make_one_hot(
                sub_vol_label,
                C=torch.max(sub_vol_label)+1
            )

        if (not self.opt.no_flip) and random.random() < 0.5:
            idx = [i for i in range(sub_vol_data.size(2) - 1, -1, -1)]
            idx = torch.LongTensor(idx)
            sub_vol_data = sub_vol_data.index_select(2, idx)
            sub_vol_label = sub_vol_label.index_select(2, idx)

        return sub_vol_data, sub_vol_label

    def get_item_test(self, index):
        # AB_path has "A" and "B" folder
        # then A is saved in 'A' B in 'B'
        # if self.cache_counter%self.num_cache_steps==0:

        test_data_file = self.test_data_files[index]
        test_label_file = self.test_label_files[index]

        test_volume_data, test_data_header = nifti_reader_3D(
            test_data_file
        )

        test_volume_label, test_label_header = nifti_reader_3D(
            test_label_file
        )

        # zoom to res
        if hasattr(self.opt, 'resolution') and self.opt.resolution != 1:
            scale_factorA = 1.0 / self.opt.resolution
            scale_factorB = 1.0 / self.opt.resolution

            test_volume_data = ndi.zoom(test_volume_data,
                                         zoom=[scale_factorA,
                                               scale_factorA,
                                               scale_factorA],
                                         order=3, mode='constant')

            test_volume_label = ndi.zoom(test_volume_label,
                                          zoom=[scale_factorB,
                                                scale_factorB,
                                                scale_factorB],
                                          order=3, mode='constant')

        # temp_volumeA_data = _pad_to_shape_3D2D(
        #     temp_volumeA_data, self.opt.fineSize
        # )
        #
        # temp_volumeB_data = _pad_to_shape_3D2D(
        #     temp_volumeB_data, self.opt.fineSize
        # )

        # normalization
        test_volume_data = normalize_01(
            test_volume_data
        )

        test_volume_data = _pad_to_shape_for_downsample3D(
            test_volume_data
        )

        test_volume_label = _pad_to_shape_for_downsample3D(
            test_volume_label
        )

        test_volume_data = normalize_white(test_volume_data)

        temp = self.test_transform([test_volume_data, test_volume_label])
        test_volume_data = temp[0].unsqueeze(0)
        test_volume_data = self.normalize(test_volume_data)
        test_volume_label = temp[1].unsqueeze(0)



        if self.make_one_hot:
            test_volume_label = make_one_hot(
                test_volume_label,
                C=torch.max(test_volume_label)+1
            )

        return test_volume_data, test_volume_label

    def __len__(self):
        if self.train:
            return len(self.train_data_files) * 100
        else:
            return len(self.test_data_files)

    def name(self):
        return 'FlexibleSizeDatasetOPTIMAT'

    def select_centres2D(self, label_vol, n_class=None):
        # background is 0
        if n_class is None:
            n_class = len(np.unique(label_vol))

        if n_class > 2:
            target_class = np.random.choice(
                np.arange(start=1, stop = n_class)
            )
        else:
            target_class = 1

        label_vol1 = np.sum((label_vol==target_class).astype(np.float), axis=-1)
        indices = np.argwhere(label_vol1>0)
        centers_ind = np.random.choice(len(indices))

        return indices[centers_ind]

    def extract_subvolumn(self, vol, centre_coord, subvolumn_size, limit_within_vol=True):

        vol_minx, vol_maxx, target_minx, target_maxx = \
            extract_sub_range(vol.shape[0],
                              centre_coord[0],
                              subvolumn_size[0],
                              limit_within_vol=limit_within_vol)
        vol_miny, vol_maxy, target_miny, target_maxy = \
            extract_sub_range(vol.shape[1],
                              centre_coord[1],
                              subvolumn_size[1],
                              limit_within_vol=limit_within_vol)

        return vol[...,vol_minx:vol_maxx, vol_miny:vol_maxy, :]

class OPTIMAT_loader(DataLoader):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None
                 ):
        super(OPTIMAT_loader, self).__init__(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=sampler,
            batch_sampler = batch_sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            timeout=timeout,
            worker_init_fn=worker_init_fn
        )
        self.input_shuffle = shuffle
        self.input_batch_sampler = batch_sampler
        self.input_sampler = sampler
        self.input_pin_memory = pin_memory
        self.input_num_workers = num_workers
        self.input_drop_last =drop_last
        self.input_timeout = timeout
        self.input_worker_init_fn = worker_init_fn

    # @property
    # def batch_size(self):
    #     return self.batch_size
    #
    # @batch_size.setter
    # def batch_size(self, value):
    #     self.batch_size = value

    def set_batch_size(self, batch_size):
        self._DataLoader__initialized = False
        self.__init__(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=self.input_shuffle,
            sampler=self.input_sampler,
            batch_sampler=self.input_batch_sampler,
            num_workers=self.input_num_workers,
            pin_memory=self.input_pin_memory,
            drop_last=self.input_drop_last,
            timeout=self.input_timeout,
            worker_init_fn=self.input_worker_init_fn
        )
        # self.__initialized = False
        # self.batch_size = batch_size
        self._DataLoader__initialized = True
        # self.__initialized = True


    def set_image_size(self, image_size):
        self.dataset.image_size = image_size


if __name__=='__main__':
    # unit test
    print('ficl')

    import socket
    import argparse
    import numpy as np

    def parse_option():

        hostname = socket.gethostname()

        parser = argparse.ArgumentParser('argument for training')

        parser.add_argument(
            '--dataroot',
            type=str,
            default='/home/cwang/OPTIMAT/OPTIMAT_Deeplearning_Segmentation/OPTIMATData/OPTIMAT1',
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

        parser.add_argument('--print_freq', type=int, default=100, help='print frequency')
        parser.add_argument('--tb_freq', type=int, default=500, help='tb frequency')
        parser.add_argument('--save_freq', type=int, default=40, help='save frequency')
        parser.add_argument('--batch_size', type=int, default=64, help='batch_size')
        parser.add_argument('--num_workers', type=int, default=8, help='num of workers to use')
        parser.add_argument('--epochs', type=int, default=240, help='number of training epochs')

        # optimization
        parser.add_argument('--learning_rate', type=float, default=0.05, help='learning rate')
        parser.add_argument('--lr_decay_epochs', type=str, default='150,180,210',
                            help='where to decay lr, can be a list')
        parser.add_argument('--lr_decay_rate', type=float, default=0.1, help='decay rate for learning rate')
        parser.add_argument('--weight_decay', type=float, default=5e-4, help='weight decay')
        parser.add_argument('--momentum', type=float, default=0.9, help='momentum')

        # dataset
        parser.add_argument('--model', type=str, default='unet',
                            choices=['unet', 'sdnet', 'attention_unet', ])

        parser.add_argument('--dataset',
                            type=str,
                            default='optimat1',
                            choices=['optimat1'],
                            help='dataset')

        parser.add_argument('-t', '--trial',
                            type=int, default=0,
                            help='the experiment id')

        opt = parser.parse_args()

        # set the path according to the environment

        opt.model_path = './save/models'
        opt.tb_path = './save/tensorboard'

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

        return opt


    opt = parse_option()

    dataset = OPTIMAT(
        opt=opt
    )

    # print(len(dataset))
    # print(dataset.train_path)
    # print(dataset.train_data_files)

    train_loader = OPTIMAT_loader(dataset=dataset,
                              batch_size=2,
                              shuffle=True,
                              num_workers=2)

    a,b = train_loader.dataset[0]
    print(a.shape)

    # train_loader.set_batch_size(5)
    # train_loader.batch_size = 3
    for i, (a, b) in enumerate(train_loader):
        print('batch size: ', train_loader.batch_size)

        print(i)
        print(a.shape)
        print(b.shape)
        a.cuda()
        b.cuda()
        train_loader.set_batch_size(
            np.random.randint(1,5)
        )


    # train_loader.dataset.train = False
    #
    # a,b = train_loader.dataset[0]
    # print(b.shape)

