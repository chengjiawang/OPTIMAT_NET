import numpy as np
import os, sys, gc
import matplotlib.pyplot as plt
import nibabel as nib
import nrrd
'''
sampling_pack_path = "/mnt/home/ubuntu/Unet-ants/code/sampling/"
models_pack_path = "/mnt/home/ubuntu/Unet-ants/code/models"

sys.path.append(sampling_pack_path)
sys.path.append(models_pack_path)
'''
from sampling import transforms as tx
import keras.backend as K
import tensorflow as tf

allLabel = [0., 205., 420., 500., 550., 600., 820., 850.]


# define generator
class Loader3D2D(object):
    def __init__(self, dataset, half_z=4, target_shape=np.array([512, 512]), batch_size=1,
                 allLabel=allLabel):
        self.dataset = dataset
        self.allLabel = allLabel
        self.batch_size = batch_size
        self.half_z = half_z
        self.target_shape = target_shape
        self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)

    def gen(self):
        while True:  # generate forever
            n_image = np.random.randint(low=0, high=self.dataset.num_inputs)

            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            true_labels = self.trans_val_target.transform(true_labels)

            true_label_z = np.sum(true_labels, axis=(0, 1)) > 0

            indeces = range(int(true_label_z.shape[0]))

            minz = np.min(true_label_z * indeces)

            maxz = np.max(true_label_z * indeces)

            indecesz = np.random.randint(low=minz, high=maxz, size=self.batch_size)

            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            ini_seg = self.trans_val_target.transform(
                self.dataset.iniseg_loader(
                    os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                )
            )
            if true_labels.shape[0] != self.target_shape[0] \
                    or true_labels.shape[1] != self.target_shape[1]:
                true_labels = self.trans_resample.transform(true_labels)
                input_im = self.trans_resample.transform(input_im)
                ini_seg = self.trans_resample.transform(ini_seg)

            batchesX = list()
            batchesY = list()
            for n_select in range(self.batch_size):
                sample_input = self._process_batch_im(input_im, indecesz[n_select])
                sample_iniseg = self._process_batch_im(ini_seg, indecesz[n_select])
                batchesX.append(np.concatenate((sample_input, sample_iniseg), axis=2))
                sample_target = self._process_batch_target(true_labels, indecesz[n_select])
                sample_target = self.trans_onehot.transform(sample_target)
                batchesY.append(sample_target)
            # gc.collect()
            # print(np.array(batchesX).shape)
            # print(np.array(batchesY).shape)
            yield np.expand_dims(np.array(batchesX), axis=-1), np.array(batchesY)

    def _process_batch_im(self,
                          vol,
                          ind):
        # print(ind)
        indz_min = ind - self.half_z
        indz_max = ind + self.half_z
        # print(indz_min, indz_max)

        select_volume = np.zeros((vol.shape[0],
                                  vol.shape[1],
                                  2 * self.half_z + 1), dtype='float32')

        if indz_min < 0:
            select_volume[:, :, -indz_min:] = vol[:, :,
                                              :2 * self.half_z + 1 + indz_min]
        elif indz_max + 1 > vol.shape[2]:
            select_volume[:, :, :indz_max + 1 - vol.shape[2]] = \
                vol[:, :, vol.shape[2] - indz_max - 1:]
        else:
            select_volume = vol[:, :, indz_min:indz_max + 1]

        return select_volume

    def _process_batch_target(self,
                              vol,
                              ind):
        select_volume = vol[:, :, ind]
        return select_volume

# define generator
class LoaderRes3D(object):
    def __init__(self, dataset,
                 subvolumn_size = (64, 64, 64),
                 allLabel=None,
                 batch_size = 10,
                 volumn_num = 3,
                 is_refine = False,
                 is_lazy=False,
                 half_z=4,
                 target_shape=np.array([512, 512])):

        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))


        self.half_z = half_z
        self.target_shape = target_shape


        self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_refine = is_refine
        self.is_lazy = is_lazy

        self.counter = -1
        # self.counter = np.random.randint(self.num_inputs)
    # def gen(self):


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        #K.clear_session()
        # tf.reset_default_graph()
        if self.is_lazy:
            self.counter+=1
            if self.is_refine:
                return self.gen_subvolumns_refine_lazyaug()
            return self.gen_subvolumns_lazyaug()
        else:
            if self.is_refine:
                return self.gen_subvolumns_refine_aug()
            return self.gen_subvolumns_aug()

    next = __next__
    '''
    def gen(self):
        while True:  # generate forever
            n_image = np.random.randint(low=0, high=self.dataset.num_inputs)

            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            true_labels = self.trans_val_target.transform(true_labels)

            true_label_z = np.sum(true_labels, axis=(0, 1)) > 0

            indeces = range(int(true_label_z.shape[0]))

            minz = np.min(true_label_z * indeces)

            maxz = np.max(true_label_z * indeces)

            indecesz = np.random.randint(low=minz, high=maxz, size=self.batch_size)

            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            ini_seg = self.trans_val_target.transform(
                self.dataset.iniseg_loader(
                    os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                )
            )
            if true_labels.shape[0] != self.target_shape[0] \
                    or true_labels.shape[1] != self.target_shape[1]:
                true_labels = self.trans_resample.transform(true_labels)
                input_im = self.trans_resample.transform(input_im)
                ini_seg = self.trans_resample.transform(ini_seg)

            batchesX = list()
            batchesY = list()
            for n_select in range(self.batch_size):
                sample_input = self._process_batch_im(input_im, indecesz[n_select])
                sample_iniseg = self._process_batch_im(ini_seg, indecesz[n_select])
                batchesX.append(np.concatenate((sample_input, sample_iniseg), axis=2))
                sample_target = self._process_batch_target(true_labels, indecesz[n_select])
                sample_target = self.trans_onehot.transform(sample_target)
                batchesY.append(sample_target)
            gc.collect()
            # print(np.array(batchesX).shape)
            # print(np.array(batchesY).shape)
            yield np.expand_dims(np.array(batchesX), axis=-1), np.array(batchesY)
    '''

    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4):
        if n_class is None:
            n_class = len(np.unique(label_vol))
        vx, vy, vz = np.meshgrid(
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[2], step=skip_ratio)
            )
        class_occurs = np.random.choice(np.arange(start=1, stop=n_class), size=vol_occur)
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


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):

        vol_minx, vol_maxx, target_minx, target_maxx = \
            self.extract_sub_range(vol.shape[0],
                                   centre_coord[0],
                                   self.subvolumn_size[0],
                                   limit_within_vol=limit_within_vol)
        vol_miny, vol_maxy, target_miny, target_maxy = \
            self.extract_sub_range(vol.shape[1],
                                   centre_coord[1],
                                   self.subvolumn_size[1],
                                   limit_within_vol=limit_within_vol)
        vol_minz, vol_maxz, target_minz, target_maxz = \
            self.extract_sub_range(vol.shape[2],
                                   centre_coord[2],
                                   self.subvolumn_size[2],
                                   limit_within_vol=limit_within_vol)


        sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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


    def gen_subvolumn_target(self):
        while True:
            # gc.collect()
            # select volumes and occurance
            batch_X = []
            volume_indeces, volume_occurs = self.select_indices()
            for n_volind in range(len(volume_indeces)):
                # gc.collect()
                n_image = volume_indeces[n_volind]
                true_labels = self.dataset.target_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )

                # transform label values to ind
                true_labels = self.trans_val_target.transform(true_labels)

                # calculate labels sum
                # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

                # select subvolume centres
                centre_coords, centre_Ys = self.select_centres(true_labels,
                                                    volume_occurs,
                                                    n_class=int(len(self.allLabel)))

                batch_Y = self.trans_onehot.transform(centre_Ys)

                # load image
                input_im = self.trans_val_input.transform(
                    self.dataset.input_loader(
                        os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                    )
                )

                for centre_coord in centre_coords:
                    batch_im= self.extract_subvolumn(
                        input_im, centre_coord
                    )
                    batch_X.append(batch_im)
            batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)

            yield batch_X, batch_Y


    def gen_subvolumns(self):
        while True:
            # select volumes and occurance
            batch_X = []
            batch_Y = []
            volume_indeces, volume_occurs = self.select_indices()
            # print(volume_indeces)
            for n_volind in range(len(volume_indeces)):
                n_image = volume_indeces[n_volind]
                true_labels = self.dataset.target_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )

                # transform label values to ind
                true_labels = self.trans_val_target.transform(true_labels)

                # calculate labels sum
                # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

                # select subvolume centres
                centre_coords, centre_Ys = self.select_centres(true_labels,
                                                    volume_occurs[n_volind],
                                                    n_class=int(len(self.allLabel)))


                # load image
                input_im = self.trans_val_input.transform(
                    self.dataset.input_loader(
                        os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                    )
                )

                # check size
                if self.subvolumn_size[-1] > input_im.shape[-1]:
                    input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

                # print(true_labels.shape)
                # print(input_im.shape)
                for centre_coord in centre_coords:
                    batch_im = self.extract_subvolumn(
                        input_im, centre_coord
                    )
                    batch_X.append(batch_im)
                    batch_target = self.extract_subvolumn(
                        true_labels, centre_coord
                    )
                    batch_target = self.trans_onehot.transform(batch_target)
                    batch_Y.append(batch_target)
            batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)
            batch_Y = np.array(batch_Y, copy=False)
            # print('xbatchshape:', np.array(batch_X).shape)
            # print('ybatchshape:', np.array(batch_Y).shape)
            yield batch_X, batch_Y

    def gen_subvolumns_aug(self):
        #while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels = aug_tx.transform(input_im, true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )
                batch_X.append(batch_im)
                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                batch_target = self.trans_onehot.transform(batch_target)
                batch_Y.append(batch_target)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)

    def gen_subvolumns_lazyaug(self):

        #while True:
        # self.counter+=1
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        # volume_indeces, volume_occurs = self.select_indices()
        volume_indeces = [self.counter % self.num_inputs]
        volume_occurs = [self.batch_size]
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels = aug_tx.transform(input_im, true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )
                batch_X.append(batch_im)
                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                batch_target = self.trans_onehot.transform(batch_target)
                batch_Y.append(batch_target)

        # print(gc.garbage)
        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)

    def gen_subvolumns_refine_aug(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # iniseg
            ini_segs = self.dataset.iniseg_loader(
                os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
            )

            ini_segs = self.trans_val_target.transform(ini_segs)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )
            # print('sampling...')
            #print(ini_segs.shape)
            #print(true_labels.shape)
            # print(input_im.shape)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)
                ini_segs = self.trans_cropZ.transform(ini_segs)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels, ini_segs = aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                batch_target = self.trans_onehot.transform(batch_target)
                batch_inisegs = self.extract_subvolumn(
                    ini_segs, centre_coord
                )
                mix_shape = list(batch_im.shape)
                mix_shape[-1] *=2
                batch_mix = np.zeros(tuple(mix_shape))
                batch_mix[..., 0::2] = batch_im
                batch_mix[..., 1::2] = batch_inisegs

                batch_X.append(batch_mix)

                batch_Y.append(batch_target)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)

    def gen_subvolumns_refine_lazyaug(self):

        # while True:
        # self.counter+=1
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        # volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        volume_indeces = [self.counter%self.num_inputs]
        volume_occurs = [self.batch_size]

        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # iniseg
            ini_segs = self.dataset.iniseg_loader(
                os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
            )

            ini_segs = self.trans_val_target.transform(ini_segs)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )
            # print('sampling...')
            #print(ini_segs.shape)
            #print(true_labels.shape)
            # print(input_im.shape)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)
                ini_segs = self.trans_cropZ.transform(ini_segs)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels, ini_segs = aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                batch_target = self.trans_onehot.transform(batch_target)
                batch_inisegs = self.extract_subvolumn(
                    ini_segs, centre_coord
                )
                mix_shape = list(batch_im.shape)
                mix_shape[-1] *=2
                batch_mix = np.zeros(tuple(mix_shape))
                batch_mix[..., 0::2] = batch_im
                batch_mix[..., 1::2] = batch_inisegs

                batch_X.append(batch_mix)

                batch_Y.append(batch_target)
        batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)
        batch_Y = np.array(batch_Y, copy=False)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return batch_X,batch_Y


    def _process_batch_im(self,
                          vol,
                          ind):
        # print(ind)
        indz_min = ind - self.half_z
        indz_max = ind + self.half_z
        # print(indz_min, indz_max)

        select_volume = np.zeros((vol.shape[0],
                                  vol.shape[1],
                                  2 * self.half_z + 1), dtype='float32')

        if indz_min < 0:
            select_volume[:, :, -indz_min:] = vol[:, :,
                                              :2 * self.half_z + 1 + indz_min]
        elif indz_max + 1 > vol.shape[2]:
            select_volume[:, :, :indz_max + 1 - vol.shape[2]] = \
                vol[:, :, vol.shape[2] - indz_max - 1:]
        else:
            select_volume = vol[:, :, indz_min:indz_max + 1]

        return select_volume

    def _process_batch_target(self,
                              vol,
                              ind):
        select_volume = vol[:, :, ind]
        return select_volume



# define generator
class LoaderRes3DOneClass(object):
    def __init__(self, dataset,
                 subvolumn_size = (64, 64, 64),
                 allLabel=None,
                 target_class_index = 1,
                 batch_size = 10,
                 volumn_num = 3, is_refine = False, is_lazy = False,
                 half_z=4, target_shape=np.array([512, 512])):
        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.target_class_ind = target_class_index
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))


        self.half_z = half_z
        self.target_shape = target_shape


        self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_refine = is_refine
        self.target_class_prop = 0.5
        self.num_classes = int(len(self.allLabel))
        self.select_prop = np.ones(self.num_classes)* \
                           (1-self.target_class_prop)/(self.num_classes-1)
        self.select_prop[self.target_class_ind] = self.target_class_prop
        self.is_lazy = is_lazy

        self.counter = -1

        # todo: set lazy length
        self.lazy_length = 10

    # def gen(self):


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.is_lazy:
            self.counter+=1
            if self.is_refine:
                return self.gen_subvolumns_refine_lazyaug()
            return self.gen_subvolumns_lazyaug()
        else:
            if self.is_refine:
                return self.gen_subvolumns_refine_aug()
            return self.gen_subvolumns_aug()

    def __call__(self, *args, **kwargs):
        pass

    next = __next__
    '''
    def gen(self):
        while True:  # generate forever
            n_image = np.random.randint(low=0, high=self.dataset.num_inputs)

            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            true_labels = self.trans_val_target.transform(true_labels)

            true_label_z = np.sum(true_labels, axis=(0, 1)) > 0

            indeces = range(int(true_label_z.shape[0]))

            minz = np.min(true_label_z * indeces)

            maxz = np.max(true_label_z * indeces)

            indecesz = np.random.randint(low=minz, high=maxz, size=self.batch_size)

            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            ini_seg = self.trans_val_target.transform(
                self.dataset.iniseg_loader(
                    os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                )
            )
            if true_labels.shape[0] != self.target_shape[0] \
                    or true_labels.shape[1] != self.target_shape[1]:
                true_labels = self.trans_resample.transform(true_labels)
                input_im = self.trans_resample.transform(input_im)
                ini_seg = self.trans_resample.transform(ini_seg)

            batchesX = list()
            batchesY = list()
            for n_select in range(self.batch_size):
                sample_input = self._process_batch_im(input_im, indecesz[n_select])
                sample_iniseg = self._process_batch_im(ini_seg, indecesz[n_select])
                batchesX.append(np.concatenate((sample_input, sample_iniseg), axis=2))
                sample_target = self._process_batch_target(true_labels, indecesz[n_select])
                sample_target = self.trans_onehot.transform(sample_target)
                batchesY.append(sample_target)
            gc.collect()
            # print(np.array(batchesX).shape)
            # print(np.array(batchesY).shape)
            yield np.expand_dims(np.array(batchesX), axis=-1), np.array(batchesY)
    '''

    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4):
        if n_class is None:
            n_class = len(np.unique(label_vol))
        vx, vy, vz = np.meshgrid(np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
                                 np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
                                 np.arange(start=np.random.choice(skip_ratio), stop= label_vol.shape[2], step=skip_ratio)
                                 )
        class_occurs = np.random.choice(
                                self.allLabel,
                                size=vol_occur,
                                p=self.select_prop)

        centres = []
        for cl in class_occurs:
            inds = np.arange(len(label_vol[vx, vy, vz].ravel()))
            inds_t = label_vol[vx, vy, vz].ravel()==cl
            inds = inds*inds_t
            gc.collect()
            inds = inds[inds!=0]
            sel_cind = np.random.choice(
                inds
            )
            # gc.collect()
            sel_c = np.array([i.ravel()[sel_cind] for i in [vx, vy, vz]])
            # gc.collect()
            centres.append(sel_c)
        return centres, class_occurs


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):

        vol_minx, vol_maxx, target_minx, target_maxx = \
            self.extract_sub_range(vol.shape[0],
                                   centre_coord[0],
                                   self.subvolumn_size[0],
                                   limit_within_vol=limit_within_vol)
        vol_miny, vol_maxy, target_miny, target_maxy = \
            self.extract_sub_range(vol.shape[1],
                                   centre_coord[1],
                                   self.subvolumn_size[1],
                                   limit_within_vol=limit_within_vol)
        vol_minz, vol_maxz, target_minz, target_maxz = \
            self.extract_sub_range(vol.shape[2],
                                   centre_coord[2],
                                   self.subvolumn_size[2],
                                   limit_within_vol=limit_within_vol)


        sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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


    def gen_subvolumn_target(self):
        while True:
            # gc.collect()
            # select volumes and occurance
            batch_X = []
            volume_indeces, volume_occurs = self.select_indices()
            for n_volind in range(len(volume_indeces)):
                # gc.collect()
                n_image = volume_indeces[n_volind]
                true_labels = self.dataset.target_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )

                # transform label values to ind
                true_labels = self.trans_val_target.transform(true_labels)

                # calculate labels sum
                # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

                # select subvolume centres
                centre_coords, centre_Ys = self.select_centres(true_labels,
                                                    volume_occurs,
                                                    n_class=int(len(self.allLabel)))

                batch_Y = self.trans_onehot.transform(centre_Ys)

                # load image
                input_im = self.trans_val_input.transform(
                    self.dataset.input_loader(
                        os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                    )
                )

                for centre_coord in centre_coords:
                    batch_im= self.extract_subvolumn(
                        input_im, centre_coord
                    )
                    batch_X.append(batch_im)

            yield np.expand_dims(np.array(batch_X, copy=False), axis=-1), batch_Y


    def gen_subvolumns(self):
        while True:
            # select volumes and occurance
            batch_X = []
            batch_Y = []
            volume_indeces, volume_occurs = self.select_indices()
            # print(volume_indeces)
            for n_volind in range(len(volume_indeces)):
                n_image = volume_indeces[n_volind]
                true_labels = self.dataset.target_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )

                # transform label values to ind
                # true_labels = self.trans_val_target.transform(true_labels)

                # calculate labels sum
                # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

                # select subvolume centres
                centre_coords, centre_Ys = self.select_centres(true_labels,
                                                    volume_occurs[n_volind],
                                                    n_class=int(len(self.allLabel)))


                # load image
                input_im = self.trans_val_input.transform(
                    self.dataset.input_loader(
                        os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                    )
                )

                # check size
                if self.subvolumn_size[-1] > input_im.shape[-1]:
                    input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

                # print(true_labels.shape)
                # print(input_im.shape)
                for centre_coord in centre_coords:
                    batch_im = self.extract_subvolumn(
                        input_im, centre_coord
                    )
                    batch_X.append(batch_im)
                    batch_target = self.extract_subvolumn(
                        true_labels == self.allLabel[self.target_class_ind], centre_coord
                    )
                    # batch_target = self.trans_onehot.transform(batch_target)
                    batch_Y.append(batch_target)
            # print(gc.garbage)
            # print('xbatchshape:', np.array(batch_X).shape)
            # print('ybatchshape:', np.array(batch_Y).shape)
            return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.expand_dims(np.array(batch_Y, copy=False), axis=-1)

    def gen_subvolumns_aug(self):
        #while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            # true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels = aug_tx.transform(input_im, true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )
                batch_X.append(batch_im)
                batch_target = self.extract_subvolumn(
                    true_labels == self.allLabel[self.target_class_ind], centre_coord
                )
                # batch_target = self.trans_onehot.transform(batch_target)
                batch_Y.append(batch_target.astype('int'))
        batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)
        batch_Y = np.expand_dims(np.array(batch_Y, copy=False), axis=-1)
        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return batch_X, batch_Y

    def gen_subvolumns_lazyaug(self):

        #while True:
        # self.counter+=1
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        # volume_indeces, volume_occurs = self.select_indices()
        volume_indeces = [self.counter % self.num_inputs]
        volume_occurs = [self.batch_size]
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            # true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels = aug_tx.transform(input_im, true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )
                batch_X.append(batch_im)
                batch_target = self.extract_subvolumn(
                    true_labels == self.allLabel[self.target_class_ind], centre_coord
                )
                # batch_target = self.trans_onehot.transform(batch_target)
                batch_Y.append(batch_target.astype('int'))

        # print(gc.garbage)
        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.expand_dims(np.array(batch_Y, copy=False), axis=-1)

    def gen_subvolumns_refine_aug(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            # true_labels = self.trans_val_target.transform(true_labels)

            # iniseg
            ini_segs = self.dataset.iniseg_loader(
                os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
            )

            # ini_segs = self.trans_val_target.transform(ini_segs)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )
            # print('sampling...')
            #print(ini_segs.shape)
            #print(true_labels.shape)
            # print(input_im.shape)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)
                ini_segs = self.trans_cropZ.transform(ini_segs)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels, ini_segs = aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn(
                    true_labels==self.allLabel[self.target_class_ind], centre_coord
                )
                # batch_target = self.trans_onehot.transform(batch_target)
                batch_inisegs = self.extract_subvolumn(
                    ini_segs, centre_coord
                )
                mix_shape = list(batch_im.shape)
                mix_shape[-1] *=2
                batch_mix = np.zeros(tuple(mix_shape))
                batch_mix[..., 0::2] = batch_im
                batch_mix[..., 1::2] = batch_inisegs.astype('float32')

                batch_X.append(batch_mix)

                batch_Y.append(batch_target)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.expand_dims(np.array(batch_Y, copy=False), axis=-1)

    def gen_subvolumns_refine_lazyaug(self):

        # while True:
        # self.counter+=1
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        # volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        volume_indeces = [self.counter%self.num_inputs]
        volume_occurs = [self.batch_size]

        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            # true_labels = self.trans_val_target.transform(true_labels)
            # true_labels = true_labels==self.allLabel[self.target_class_ind]

            # iniseg
            ini_segs = self.dataset.iniseg_loader(
                os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
            )

            # ini_segs = self.trans_val_target.transform(ini_segs)
            # ini_segs = ini_segs==self.allLabel[self.target_class_ind]

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )
            # print('sampling...')
            #print(ini_segs.shape)
            #print(true_labels.shape)
            # print(input_im.shape)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)
                ini_segs = self.trans_cropZ.transform(ini_segs)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels, ini_segs = aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn(
                    true_labels==self.allLabel[self.target_class_ind], centre_coord
                )
                # batch_target = self.trans_onehot.transform(batch_target)
                batch_inisegs = self.extract_subvolumn(
                    ini_segs.astype('float32'), centre_coord
                )
                mix_shape = list(batch_im.shape)
                mix_shape[-1] *=2
                batch_mix = np.zeros(tuple(mix_shape))
                batch_mix[..., 0::2] = batch_im
                batch_mix[..., 1::2] = batch_inisegs

                batch_X.append(batch_mix)

                batch_Y.append(batch_target)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.expand_dims(np.array(batch_Y, copy=False), axis=-1)


    def _process_batch_im(self,
                          vol,
                          ind):
        # print(ind)
        indz_min = ind - self.half_z
        indz_max = ind + self.half_z
        # print(indz_min, indz_max)

        select_volume = np.zeros((vol.shape[0],
                                  vol.shape[1],
                                  2 * self.half_z + 1), dtype='float32')

        if indz_min < 0:
            select_volume[:, :, -indz_min:] = vol[:, :,
                                              :2 * self.half_z + 1 + indz_min]
        elif indz_max + 1 > vol.shape[2]:
            select_volume[:, :, :indz_max + 1 - vol.shape[2]] = \
                vol[:, :, vol.shape[2] - indz_max - 1:]
        else:
            select_volume = vol[:, :, indz_min:indz_max + 1]

        return select_volume


class LoaderRes3DResample(object):
    def __init__(self, dataset,
                 subvolumn_size = (64, 64, 64),
                 allLabel=None,
                 batch_size = 10,
                 volumn_num = 3,
                 is_refine = False,
                 is_lazy=False,
                 half_z=4,
                 target_voxel_size = np.array([1,1,1])):

        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))


        self.half_z = half_z
        self.target_voxel_size = target_voxel_size


        self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        # self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_refine = is_refine
        self.is_lazy = is_lazy

        self.counter = -1
        self.header_loader = nib.load

        self.exclude_background = 0
        # self.counter = np.random.randint(self.num_inputs)
    # def gen(self):


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        #K.clear_session()
        # tf.reset_default_graph()
        if self.is_lazy:
            self.counter+=1
        return self.gen()

    next = __next__
    '''
    def gen(self):
        while True:  # generate forever
            n_image = np.random.randint(low=0, high=self.dataset.num_inputs)

            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            true_labels = self.trans_val_target.transform(true_labels)

            true_label_z = np.sum(true_labels, axis=(0, 1)) > 0

            indeces = range(int(true_label_z.shape[0]))

            minz = np.min(true_label_z * indeces)

            maxz = np.max(true_label_z * indeces)

            indecesz = np.random.randint(low=minz, high=maxz, size=self.batch_size)

            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            ini_seg = self.trans_val_target.transform(
                self.dataset.iniseg_loader(
                    os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                )
            )
            if true_labels.shape[0] != self.target_shape[0] \
                    or true_labels.shape[1] != self.target_shape[1]:
                true_labels = self.trans_resample.transform(true_labels)
                input_im = self.trans_resample.transform(input_im)
                ini_seg = self.trans_resample.transform(ini_seg)

            batchesX = list()
            batchesY = list()
            for n_select in range(self.batch_size):
                sample_input = self._process_batch_im(input_im, indecesz[n_select])
                sample_iniseg = self._process_batch_im(ini_seg, indecesz[n_select])
                batchesX.append(np.concatenate((sample_input, sample_iniseg), axis=2))
                sample_target = self._process_batch_target(true_labels, indecesz[n_select])
                sample_target = self.trans_onehot.transform(sample_target)
                batchesY.append(sample_target)
            gc.collect()
            # print(np.array(batchesX).shape)
            # print(np.array(batchesY).shape)
            yield np.expand_dims(np.array(batchesX), axis=-1), np.array(batchesY)
    '''

    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4):
        if n_class is None:
            n_class = len(np.unique(label_vol))
        vx, vy, vz = np.meshgrid(
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[2], step=skip_ratio)
            )

        class_occurs = np.random.choice(np.arange(start=self.exclude_background,
                                                  stop=n_class),
                                        size=vol_occur)
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


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):

        if np.any(vol.shape<self.subvolumn_size):
            vol_mins = np.zeros(int(len(vol.shape))).astype('int')
            target_mins = np.zeros(int(len(vol.shape))).astype('int')
            vol_maxs = np.zeros(int(len(vol.shape))).astype('int')
            target_maxs = np.zeros(int(len(vol.shape))).astype('int')
            for i in range(len(vol.shape)):
                if vol.shape[i] < self.subvolumn_size[i]:
                    vol_mins[i] = 0
                    target_mins[i] = 0
                    vol_maxs[i] = vol.shape[i]
                    target_maxs[i] = vol.shape[i]
                else:
                    vol_mins[i], vol_maxs[i], target_mins[i], target_maxs[i] = \
                        self.extract_sub_range(vol.shape[i],
                                               centre_coord[i],
                                               self.subvolumn_size[i],
                                               limit_within_vol = limit_within_vol
                        )
            sub_vol = np.ones(self.subvolumn_size) * np.min(vol)

            sub_vol[target_mins[0]:target_maxs[0],
                target_mins[1]:target_maxs[1],
                target_mins[2]:target_maxs[2]] = \
                vol[vol_mins[0]:vol_maxs[0], vol_mins[1]:vol_maxs[1], vol_mins[2]:vol_maxs[2]]
        else:
            vol_minx, vol_maxx, target_minx, target_maxx = \
                self.extract_sub_range(vol.shape[0],
                                       centre_coord[0],
                                       self.subvolumn_size[0],
                                       limit_within_vol=limit_within_vol)
            vol_miny, vol_maxy, target_miny, target_maxy = \
                self.extract_sub_range(vol.shape[1],
                                       centre_coord[1],
                                       self.subvolumn_size[1],
                                       limit_within_vol=limit_within_vol)
            vol_minz, vol_maxz, target_minz, target_maxz = \
                self.extract_sub_range(vol.shape[2],
                                       centre_coord[2],
                                       self.subvolumn_size[2],
                                       limit_within_vol=limit_within_vol)


            sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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

    def gen(self):

        # while True:
        # self.counter+=1
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        if self.is_lazy:
            volume_indeces = [self.counter % self.num_inputs]
            volume_occurs = [self.batch_size]
        else:
            volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)


        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )


            # load image
            # print(self.dataset.inputs[n_image][0])
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                ).astype('float')
            )

            im_info = self.header_loader(
                os.path.join(self.dataset.base_path,
                             self.dataset.inputs[n_image][0]))
            im_vox_size = np.array(im_info.header.get_zooms())
            if not np.any(im_vox_size==self.target_voxel_size):
                resample_trans = tx.ResampleVoxel(target_voxel_size=self.target_voxel_size,
                                                  original_voxel_size=im_vox_size)
                input_im, true_labels = resample_trans.transform(input_im, true_labels)

            # print(input_im.shape)
            # print(true_labels.shape)
            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                # print(input_im.shape, true_labels.shape)
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-50, 50),  # rotate btwn -15 & 15 degrees
                                     translation_range=(-0.2, 0.2),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.5, 2.0),  # between 15% zoom-in and 15% zoom-o
                                     turn_off_frequency=5, # how often to just turn off random affine transform (units=#samples)
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0,
                                     flip_probability=0.4)
            input_im, true_labels = aug_tx.transform(input_im, true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            true_labels = self.trans_val_target.transform(true_labels)

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))

            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                batch_target = self.trans_onehot.transform(batch_target)

                batch_X.append(batch_im)

                batch_Y.append(batch_target)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)


# define generator
def header_reader(path):
    if path.endswith('.nii.gz'):
        head = nib.load(path)
    elif path.endswith('.nrrd'):
        head = nrrd.read(path)[1]
    else:
        raise ValueError('File Format is not supported')
    return head

class LoaderData3D(object):
    def __init__(self, dataset,
                 subvolumn_size = (64, 64, 64),
                 allLabel=None,
                 batch_size = 10,
                 volumn_num = 3,
                 is_refine = False,
                 is_lazy=False,
                 uni_vox_resample = False,
                 augment_transform=True,
                 onehot_trans = True,
                 output_shape = None):

        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]
        else:
            self.subvolumn_size = None

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))




        self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.to_onehot = onehot_trans
        if self.to_onehot:
            self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        # self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_refine = is_refine
        self.is_lazy = is_lazy

        self.counter = -1
        # self.counter = np.random.randint(self.num_inputs)
        self.augment_transform = augment_transform
        self.uni_vox = uni_vox_resample
        # head loader
        self.head_loader = header_reader

        if output_shape is None:
            self.output_shape = self.subvolumn_size
        else:
            self.output_shape = output_shape


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.is_lazy:
            self.counter+=1
        return self.gen()
        #K.clear_session()
        # tf.reset_default_graph()


    next = __next__

    def gen(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        if self.is_lazy:
            # volume_indeces, volume_occurs = self.select_indices()
            volume_indeces = [self.counter % self.num_inputs]
            volume_occurs = [self.batch_size]
        else:
            volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            im_shape = np.array(list(input_im.shape))
            if self.is_refine:
            # iniseg
                ini_segs = self.dataset.iniseg_loader(
                    os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                )

                ini_segs = self.trans_val_target.transform(ini_segs)
            # print('sampling...')
            # print(ini_segs.shape)
            # print(true_labels.shape)
            # print(input_im.shape)

            if self.uni_vox:
                input_im_info = self.head_loader(os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0]))
                input_im_vox_size = np.array(input_im_info.header.get_zooms())
                resample_trans = tx.ResampleVoxel(target_voxel_size=np.array([1,1,1]),
                                                  original_voxel_size=input_im_vox_size).transform

                input_im, true_labels = resample_trans(input_im, true_labels)
                if self.is_refine:
                    ini_segs = resample_trans(ini_segs)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)
                if self.is_refine:
                    ini_segs = self.trans_cropZ.transform(ini_segs)

            # resample

            if self.augment_transform:
                # augmentation
                aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                         translation_range=(0.1, 0.1),
                                         # translate btwn -10% and 10% horiz, -10% and 10% vert
                                         shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                         zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                         turn_off_frequency=5,
                                         fill_value='min',
                                         target_fill_mode='constant',
                                         target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
                if self.is_refine:

                    input_im, true_labels, ini_segs = aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)

                else:
                    input_im, true_labels = aug_tx.transform(input_im,
                                                                             true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                           volume_occurs[n_volind],
                                                           n_class=int(len(self.allLabel)))

            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn_with_shape(
                    true_labels, centre_coord, output_shape=self.output_shape
                )
                # calculate the z dim
                if self.to_onehot:
                    batch_target = self.trans_onehot.transform(batch_target)
                # print(batch_target.shape)

                if self.is_refine:
                    batch_inisegs = self.extract_subvolumn(
                        ini_segs, centre_coord
                    )
                    mix_shape = list(batch_im.shape)
                    mix_shape.append(2)
                    batch_mix = np.zeros(mix_shape)
                    batch_mix[..., 0] = batch_im
                    batch_mix[..., 1] = batch_inisegs
                    # print(batch_mix.shape)

                    batch_X.append(batch_mix)
                else:
                    batch_X.append(batch_im)


                batch_Y.append(batch_target)


        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        if not self.is_refine:
            return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)
        return np.array(batch_X, copy=False), np.array(batch_Y, copy=False)
    '''
        while True:  # generate forever
            n_image = np.random.randint(low=0, high=self.dataset.num_inputs)

            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            true_labels = self.trans_val_target.transform(true_labels)

            true_label_z = np.sum(true_labels, axis=(0, 1)) > 0

            indeces = range(int(true_label_z.shape[0]))

            minz = np.min(true_label_z * indeces)

            maxz = np.max(true_label_z * indeces)

            indecesz = np.random.randint(low=minz, high=maxz, size=self.batch_size)

            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            ini_seg = self.trans_val_target.transform(
                self.dataset.iniseg_loader(
                    os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                )
            )
            if true_labels.shape[0] != self.target_shape[0] \
                    or true_labels.shape[1] != self.target_shape[1]:
                true_labels = self.trans_resample.transform(true_labels)
                input_im = self.trans_resample.transform(input_im)
                ini_seg = self.trans_resample.transform(ini_seg)

            batchesX = list()
            batchesY = list()
            for n_select in range(self.batch_size):
                sample_input = self._process_batch_im(input_im, indecesz[n_select])
                sample_iniseg = self._process_batch_im(ini_seg, indecesz[n_select])
                batchesX.append(np.concatenate((sample_input, sample_iniseg), axis=2))
                sample_target = self._process_batch_target(true_labels, indecesz[n_select])
                sample_target = self.trans_onehot.transform(sample_target)
                batchesY.append(sample_target)
            gc.collect()
            # print(np.array(batchesX).shape)
            # print(np.array(batchesY).shape)
            yield np.expand_dims(np.array(batchesX), axis=-1), np.array(batchesY)
    '''

    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4):
        if n_class is None:
            n_class = len(np.unique(label_vol))
        vx, vy, vz = np.meshgrid(
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[2], step=skip_ratio)
            )
        class_occurs = np.random.choice(np.arange(start=1, stop=n_class), size=vol_occur)
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


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):
        if np.any(vol.shape<self.subvolumn_size):
            vol_mins = target_mins = vol_maxs = target_maxs = np.zeros(int(len(vol.shape)))
            for i in range(len(vol.shape)):
                if vol.shape[i] < self.subvolumn_size[i]:
                    vol_mins[i] = target_mins[i] = 0
                    vol_maxs[i] = target_maxs[i] = vol.shape[i]
                else:
                    vol_mins[i], vol_maxs[i], target_mins[i], target_maxs[i] = \
                        self.extract_sub_range(vol.shape[i],
                                               centre_coord[i],
                                               self.subvolumn_size[i],
                                               limit_within_vol = limit_within_vol
                        )
            sub_vol = np.ones(self.subvolumn_size) * np.min(vol)
            sub_vol[target_mins[0]:target_maxs[0],
                target_mins[1]:target_maxs[1],
                target_mins[2]:target_maxs[2]] = \
                vol[vol_mins[0]:vol_maxs[0], vol_mins[1]:vol_maxs[1], vol_mins[2]:vol_maxs[2]]


        else:
            vol_minx, vol_maxx, target_minx, target_maxx = \
                self.extract_sub_range(vol.shape[0],
                                       centre_coord[0],
                                       self.subvolumn_size[0],
                                       limit_within_vol=limit_within_vol)
            vol_miny, vol_maxy, target_miny, target_maxy = \
                self.extract_sub_range(vol.shape[1],
                                       centre_coord[1],
                                       self.subvolumn_size[1],
                                       limit_within_vol=limit_within_vol)
            vol_minz, vol_maxz, target_minz, target_maxz = \
                self.extract_sub_range(vol.shape[2],
                                       centre_coord[2],
                                       self.subvolumn_size[2],
                                       limit_within_vol=limit_within_vol)


            sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_subvolumn_with_shape(self, vol, centre_coord, output_shape, limit_within_vol=True):
        # find the corresponding target volumn first
        sub_vol = self.extract_subvolumn(vol, centre_coord, limit_within_vol)
        assert(np.all(sub_vol.shape >= output_shape))
        half_out_shape = np.array(output_shape)//2
        # print(half_out_shape)
        centre_out = [sub_vol.shape[i]//2 if sub_vol.shape[i]%2==1 else sub_vol.shape[i]//2-1
                      for i in range(len(sub_vol.shape))]
        # 3D case
        vol_starts = [max(0, centre_out[i] - half_out_shape[i]) for i in range(len(half_out_shape))]
        vol_ends = [centre_out[i] + half_out_shape[i]+1 for i in range(len(half_out_shape))]
        sub_volnew = np.zeros(output_shape)
        sub_volnew[:, :, ...] = sub_vol[vol_starts[0]:vol_ends[0],
                  vol_starts[1]:vol_ends[1],
                  vol_starts[2]:vol_ends[2]]
        # print(sub_volnew.shape)
        if sub_volnew.ndim <3:
            for i in range(len(output_shape)):
                if half_out_shape[i] == 0:
                    sub_vol = np.expand_dims(sub_vol, i)
            #print(sub_vol.shape)
        return np.expand_dims(sub_volnew, axis=-1)

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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


    def gen_subvolumn_target(self):
        while True:
            # gc.collect()
            # select volumes and occurance
            batch_X = []
            volume_indeces, volume_occurs = self.select_indices()
            for n_volind in range(len(volume_indeces)):
                # gc.collect()
                n_image = volume_indeces[n_volind]
                true_labels = self.dataset.target_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )

                # transform label values to ind
                true_labels = self.trans_val_target.transform(true_labels)

                # calculate labels sum
                # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

                # select subvolume centres
                centre_coords, centre_Ys = self.select_centres(true_labels,
                                                    volume_occurs,
                                                    n_class=int(len(self.allLabel)))

                if self.to_onehot:
                    batch_Y = self.trans_onehot.transform(centre_Ys)

                # load image
                input_im = self.trans_val_input.transform(
                    self.dataset.input_loader(
                        os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                    )
                )

                for centre_coord in centre_coords:
                    batch_im= self.extract_subvolumn(
                        input_im, centre_coord
                    )
                    batch_X.append(batch_im)
            batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)

            yield batch_X, batch_Y


    def gen_subvolumns(self):
        while True:
            # select volumes and occurance
            batch_X = []
            batch_Y = []
            volume_indeces, volume_occurs = self.select_indices()
            # print(volume_indeces)
            for n_volind in range(len(volume_indeces)):
                n_image = volume_indeces[n_volind]
                true_labels = self.dataset.target_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )

                # transform label values to ind
                true_labels = self.trans_val_target.transform(true_labels)

                # calculate labels sum
                # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

                # select subvolume centres
                centre_coords, centre_Ys = self.select_centres(true_labels,
                                                    volume_occurs[n_volind],
                                                    n_class=int(len(self.allLabel)))


                # load image
                input_im = self.trans_val_input.transform(
                    self.dataset.input_loader(
                        os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                    )
                )

                # check size
                if self.subvolumn_size[-1] > input_im.shape[-1]:
                    input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

                # print(true_labels.shape)
                # print(input_im.shape)
                for centre_coord in centre_coords:
                    batch_im = self.extract_subvolumn(
                        input_im, centre_coord
                    )
                    batch_X.append(batch_im)
                    batch_target = self.extract_subvolumn(
                        true_labels, centre_coord
                    )
                    if self.to_onehot:
                        batch_target = self.trans_onehot.transform(batch_target)
                    batch_Y.append(batch_target)
            batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)
            batch_Y = np.array(batch_Y, copy=False)
            # print('xbatchshape:', np.array(batch_X).shape)
            # print('ybatchshape:', np.array(batch_Y).shape)
            yield batch_X, batch_Y

    def gen_subvolumns_aug(self):
        #while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels = aug_tx.transform(input_im, true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )
                batch_X.append(batch_im)
                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                if self.to_onehot:
                    batch_target = self.trans_onehot.transform(batch_target)
                batch_Y.append(batch_target)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)

    def gen_subvolumns_lazyaug(self):

        #while True:
        # self.counter+=1
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        # volume_indeces, volume_occurs = self.select_indices()
        volume_indeces = [self.counter % self.num_inputs]
        volume_occurs = [self.batch_size]
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels = aug_tx.transform(input_im, true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )
                batch_X.append(batch_im)
                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                if self.to_onehot:
                    batch_target = self.trans_onehot.transform(batch_target)
                batch_Y.append(batch_target)

        # print(gc.garbage)
        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)

    def gen_subvolumns_refine_aug(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # iniseg
            ini_segs = self.dataset.iniseg_loader(
                os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
            )

            ini_segs = self.trans_val_target.transform(ini_segs)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )
            # print('sampling...')
            #print(ini_segs.shape)
            #print(true_labels.shape)
            # print(input_im.shape)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)
                ini_segs = self.trans_cropZ.transform(ini_segs)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels, ini_segs = aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                if self.to_onehot:
                    batch_target = self.trans_onehot.transform(batch_target)
                batch_inisegs = self.extract_subvolumn(
                    ini_segs, centre_coord
                )
                mix_shape = list(batch_im.shape)
                mix_shape[-1] *=2
                batch_mix = np.zeros(tuple(mix_shape))
                batch_mix[..., 0::2] = batch_im
                batch_mix[..., 1::2] = batch_inisegs

                batch_X.append(batch_mix)

                batch_Y.append(batch_target)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return np.expand_dims(np.array(batch_X, copy=False), axis=-1), np.array(batch_Y, copy=False)

    def gen_subvolumns_refine_lazyaug(self):

        # while True:
        # self.counter+=1
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        # volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        volume_indeces = [self.counter%self.num_inputs]
        volume_occurs = [self.batch_size]

        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # iniseg
            ini_segs = self.dataset.iniseg_loader(
                os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
            )

            ini_segs = self.trans_val_target.transform(ini_segs)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )
            # print('sampling...')
            #print(ini_segs.shape)
            #print(true_labels.shape)
            # print(input_im.shape)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)
                ini_segs = self.trans_cropZ.transform(ini_segs)

            # augmentation
            aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                     translation_range=(0.1, 0.1),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)
            input_im, true_labels, ini_segs = aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                volume_occurs[n_volind],
                                                n_class=int(len(self.allLabel)))


            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target = self.extract_subvolumn(
                    true_labels, centre_coord
                )
                if self.to_onehot:
                    batch_target = self.trans_onehot.transform(batch_target)
                batch_inisegs = self.extract_subvolumn(
                    ini_segs, centre_coord
                )
                mix_shape = list(batch_im.shape)
                mix_shape[-1] *=2
                batch_mix = np.zeros(tuple(mix_shape))
                batch_mix[..., 0::2] = batch_im
                batch_mix[..., 1::2] = batch_inisegs

                batch_X.append(batch_mix)

                batch_Y.append(batch_target)
        batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)
        batch_Y = np.array(batch_Y, copy=False)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return batch_X,batch_Y

    '''
    def _process_batch_im(self,
                          vol,
                          ind):
        # print(ind)
        indz_min = ind - self.half_z
        indz_max = ind + self.half_z
        # print(indz_min, indz_max)

        select_volume = np.zeros((vol.shape[0],
                                  vol.shape[1],
                                  2 * self.half_z + 1), dtype='float32')

        if indz_min < 0:
            select_volume[:, :, -indz_min:] = vol[:, :,
                                              :2 * self.half_z + 1 + indz_min]
        elif indz_max + 1 > vol.shape[2]:
            select_volume[:, :, :indz_max + 1 - vol.shape[2]] = \
                vol[:, :, vol.shape[2] - indz_max - 1:]
        else:
            select_volume = vol[:, :, indz_min:indz_max + 1]

        return select_volume

    def _process_batch_target(self,
                              vol,
                              ind):
        select_volume = vol[:, :, ind]
        return select_volume
        '''

class LoaderData3DFixShapeNrrd(object):
    def __init__(self, dataset,
                 subvolumn_size = (128, 128, 128),
                 allLabel=np.arange(start=0, stop=4, step=1),
                 batch_size = 10,
                 volumn_num = 3,
                 is_refine = False,
                 is_lazy=False,
                 uni_vox_resample = False,
                 augment_transform=True,
                 onehot_trans = True,
                 output_shape = None):

        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]
        else:
            self.subvolumn_size = None

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))



        if allLabel is not None:
            self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.input_type_cast = tx.TypeCast('float32')
        self.to_onehot = onehot_trans
        self.shape_trans = tx.FixShape(self.subvolumn_size)
        if self.to_onehot:
            self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        # self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_refine = is_refine
        self.is_lazy = is_lazy

        self.counter = -1
        # self.counter = np.random.randint(self.num_inputs)

        self.uni_vox = uni_vox_resample
        # head loader
        self.head_loader = header_reader

        self.augment_transform = augment_transform
        if output_shape is None:
            self.output_shape = self.subvolumn_size
            self.augment_transform = True
        else:
            self.output_shape = output_shape

        if self.augment_transform:
            self.aug_tx = tx.RandomAffine(rotation_range=(-45, 45),  # rotate btwn -15 & 15 degrees
                                     translation_range=(-0.2, 0.2),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.is_lazy:
            self.counter+=1
        return self.gen()
        #K.clear_session()
        # tf.reset_default_graph()


    next = __next__

    def gen(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        if self.is_lazy:
            # volume_indeces, volume_occurs = self.select_indices()
            volume_indeces = [self.counter % self.num_inputs]
            volume_occurs = [self.batch_size]
        else:
            volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )
            # true labels has shift
            label_header = self.head_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )
            label_shift = np.array(label_header['keyvaluepairs']['Segmentation_ReferenceImageExtentOffset'].split(' ')).astype('int')


            # true_labels = np.rollaxis(true_labels, 0, -1)

            true_labels = np.append(np.expand_dims(np.zeros(true_labels.shape[1:]), axis=0),
                                     true_labels, axis=0)

            true_labels = np.argmax(true_labels, axis=0)


            # load image
            input_im = self.trans_val_input.transform(
                self.input_type_cast.transform(
                    self.dataset.input_loader(
                        os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                    )
                )
            )
            temp_label = np.zeros(input_im.shape)
            temp_label[label_shift[0]:label_shift[0]+true_labels.shape[0],
            label_shift[1]:label_shift[1]+true_labels.shape[1],
            label_shift[2]:label_shift[2]+true_labels.shape[2], ...] = true_labels
            true_labels = temp_label
            # print(true_labels.shape)

            # extract overlaped area

            # im_shape = np.array(list(input_im.shape))
            if self.is_refine:
            # iniseg
                ini_segs = self.dataset.iniseg_loader(
                    os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                )

                if self.allLabel is not None:
                    ini_segs = self.trans_val_target.transform(ini_segs)
                else:
                    # ini_segs = np.rollaxis(ini_segs, 0, -1)
                    ini_segs = np.append(np.expand_dims(np.zeros(ini_segs.shape[1:]), axis=0),
                                         ini_segs)
                    ini_segs = np.argmax(ini_segs, axis=0)
                    if self.to_onehot:
                        ini_segs = self.trans_onehot.transform(ini_segs)


            # print('sampling...')
            # print(ini_segs.shape)
            # print(true_labels.shape)
            # print(input_im.shape)

            if self.uni_vox:
                # header loader presently only work with nifti images
                input_im_info = self.head_loader(os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0]))
                input_im_vox_size = np.array(input_im_info.header.get_zooms())
                resample_trans = tx.ResampleVoxel(target_voxel_size=np.array([1,1,1]),
                                                  original_voxel_size=input_im_vox_size).transform

                input_im, true_labels = resample_trans(input_im, true_labels)
                if self.is_refine:
                    ini_segs = resample_trans(ini_segs)

            # resample
            # print(true_labels.shape)
            if self.augment_transform:
                if self.is_refine:

                    input_im, true_labels, ini_segs = self.aug_tx.transform3objs(input_im,
                                                                    true_labels,
                                                                    ini_segs)
                else:
                    input_im, true_labels = self.aug_tx.transform(input_im,
                                                                             true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]
            input_im = self.shape_trans.transform(input_im)
            true_labels = self.shape_trans.transform(true_labels)

            if self.to_onehot:
                true_labels = self.trans_onehot.transform(true_labels)

            # print(true_labels.shape)
            # debugging code
            '''
            import matplotlib.pyplot as plt
            plt.figure(0)
            plt.imshow(input_im[:, :, 100])
            plt.figure(1)
            plt.imshow(true_labels[:, :, 100, 1])
            plt.figure(2)
            plt.imshow(true_labels[:, :, 100, 2])
            plt.figure(3)
            plt.imshow(true_labels[:, :, 100, 3])
            plt.figure(4)
            plt.imshow(true_labels[:, :, 100, 0])
            plt.show(block=True)
            '''

            batch_X.append(input_im)
            batch_Y.append(true_labels)


        batch_X = np.expand_dims(np.array(batch_X, copy=False), axis=-1)
        batch_Y = np.array(batch_Y, copy=False)

        # print(batch_X.shape)
        # print(batch_Y.shape)
        # debuging code

        '''
        import matplotlib.pyplot as plt
        plt.figure(0)
        plt.imshow(batch_X[0, :, :, 100, 0])
        plt.figure(1)
        plt.imshow(batch_Y[0, :, :, 100, 1])
        plt.figure(2)
        plt.imshow(batch_Y[0, :, :, 100, 2])
        plt.figure(3)
        plt.imshow(batch_Y[0, :, :, 100, 3])
        plt.figure(4)
        plt.imshow(batch_Y[0, :, :, 100, 0])
        plt.show(block=True)
        '''

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return batch_X, batch_Y

    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4):
        if n_class is None:
            n_class = len(np.unique(label_vol))
        vx, vy, vz = np.meshgrid(
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[2], step=skip_ratio)
            )
        class_occurs = np.random.choice(np.arange(start=1, stop=n_class), size=vol_occur)
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


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):
        if np.any(vol.shape<self.subvolumn_size):
            vol_mins = target_mins = vol_maxs = target_maxs = np.zeros(int(len(vol.shape)))
            for i in range(len(vol.shape)):
                if vol.shape[i] < self.subvolumn_size[i]:
                    vol_mins[i] = target_mins[i] = 0
                    vol_maxs[i] = target_maxs[i] = vol.shape[i]
                else:
                    vol_mins[i], vol_maxs[i], target_mins[i], target_maxs[i] = \
                        self.extract_sub_range(vol.shape[i],
                                               centre_coord[i],
                                               self.subvolumn_size[i],
                                               limit_within_vol = limit_within_vol
                        )
            sub_vol = np.ones(self.subvolumn_size) * np.min(vol)
            sub_vol[target_mins[0]:target_maxs[0],
                target_mins[1]:target_maxs[1],
                target_mins[2]:target_maxs[2]] = \
                vol[vol_mins[0]:vol_maxs[0], vol_mins[1]:vol_maxs[1], vol_mins[2]:vol_maxs[2]]


        else:
            vol_minx, vol_maxx, target_minx, target_maxx = \
                self.extract_sub_range(vol.shape[0],
                                       centre_coord[0],
                                       self.subvolumn_size[0],
                                       limit_within_vol=limit_within_vol)
            vol_miny, vol_maxy, target_miny, target_maxy = \
                self.extract_sub_range(vol.shape[1],
                                       centre_coord[1],
                                       self.subvolumn_size[1],
                                       limit_within_vol=limit_within_vol)
            vol_minz, vol_maxz, target_minz, target_maxz = \
                self.extract_sub_range(vol.shape[2],
                                       centre_coord[2],
                                       self.subvolumn_size[2],
                                       limit_within_vol=limit_within_vol)


            sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_subvolumn_with_shape(self, vol, centre_coord, output_shape, limit_within_vol=True):
        # find the corresponding target volumn first
        sub_vol = self.extract_subvolumn(vol, centre_coord, limit_within_vol)
        assert(np.all(sub_vol.shape >= output_shape))
        half_out_shape = np.array(output_shape)//2
        # print(half_out_shape)
        centre_out = [sub_vol.shape[i]//2 if sub_vol.shape[i]%2==1 else sub_vol.shape[i]//2-1
                      for i in range(len(sub_vol.shape))]
        # 3D case
        vol_starts = [max(0, centre_out[i] - half_out_shape[i]) for i in range(len(half_out_shape))]
        vol_ends = [centre_out[i] + half_out_shape[i]+1 for i in range(len(half_out_shape))]
        sub_volnew = np.zeros(output_shape)
        sub_volnew[:, :, ...] = sub_vol[vol_starts[0]:vol_ends[0],
                  vol_starts[1]:vol_ends[1],
                  vol_starts[2]:vol_ends[2]]
        # print(sub_volnew.shape)
        if sub_volnew.ndim <3:
            for i in range(len(output_shape)):
                if half_out_shape[i] == 0:
                    sub_vol = np.expand_dims(sub_vol, i)
            #print(sub_vol.shape)
        return np.expand_dims(sub_volnew, axis=-1)

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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


class LoaderData3DFixShapeNrrdCrop(object):
    def __init__(self, dataset,
                 subvolumn_size = (128, 128, 128),
                 allLabel=np.arange(start=0, stop=4, step=1),
                 batch_size = 10,
                 volumn_num = 3,
                 is_refine = False,
                 is_lazy=False,
                 uni_vox_resample = False,
                 augment_transform=True,
                 onehot_trans = True,
                 output_shape = None):

        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]
        else:
            self.subvolumn_size = None

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))



        if allLabel is not None:
            self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.input_type_cast = tx.TypeCast('float32')
        self.to_onehot = onehot_trans
        self.shape_trans = tx.FixShape(self.subvolumn_size)
        if self.to_onehot:
            self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        # self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_refine = is_refine
        self.is_lazy = is_lazy

        self.counter = -1
        # self.counter = np.random.randint(self.num_inputs)

        self.uni_vox = uni_vox_resample
        # head loader
        self.head_loader = header_reader

        #
        self.cache_volume_indeces = np.array([-1])
        self.reload_vol = True
        self.input_im = None
        self.true_labels = None


        self.augment_transform = augment_transform
        if output_shape is None:
            self.output_shape = self.subvolumn_size
            self.augment_transform = True
        else:
            self.output_shape = output_shape

        if self.augment_transform:
            self.aug_tx = tx.RandomAffine(rotation_range=(-45, 45),  # rotate btwn -15 & 15 degrees
                                     translation_range=(-0.2, 0.2),
                                     # translate btwn -10% and 10% horiz, -10% and 10% vert
                                     shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                     zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                     turn_off_frequency=5,
                                     fill_value='min',
                                     target_fill_mode='constant',
                                     target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.is_lazy:
            self.counter+=1
        return self.gen()
        #K.clear_session()
        # tf.reset_default_graph()


    next = __next__

    def gen(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        if self.is_lazy:
            # volume_indeces, volume_occurs = self.select_indices()
            volume_indeces = np.array([self.counter //20 % self.num_inputs])
            volume_occurs = np.array([self.batch_size])
        else:
            volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)

        if np.all(volume_indeces == self.cache_volume_indeces):
            self.reload_vol = False
        else:
            self.reload_vol = True

        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            if not self.reload_vol:
                true_labels = self.true_labels
                input_im = self.input_im
            else:
                # true_labels
                true_labels = self.dataset.target_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )
                # true labels has shift
                label_header = self.head_loader(
                    os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
                )
                label_shift = np.array(label_header['keyvaluepairs']['Segmentation_ReferenceImageExtentOffset'].split(' ')).astype('int')


                # true_labels = np.rollaxis(true_labels, 0, -1)

                true_labels = np.append(np.expand_dims(np.zeros(true_labels.shape[1:]), axis=0),
                                         true_labels, axis=0)

                true_labels = np.argmax(true_labels, axis=0)
                # print('all_labels: ', np.sum(true_labels))


                # load image
                input_im = self.trans_val_input.transform(
                    self.input_type_cast.transform(
                        self.dataset.input_loader(
                            os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                        )
                    )
                )
                temp_label = np.zeros(input_im.shape)
                temp_label[label_shift[0]:label_shift[0]+true_labels.shape[0],
                label_shift[1]:label_shift[1]+true_labels.shape[1],
                label_shift[2]:label_shift[2]+true_labels.shape[2], ...] = true_labels
                true_labels = temp_label

                self.true_labels = true_labels
                self.input_im = input_im
                ''' # arbitrary crop
                temp_image = input_im[label_shift[0]:label_shift[0]+true_labels.shape[0],
                label_shift[1]:label_shift[1]+true_labels.shape[1],
                label_shift[2]:label_shift[2]+true_labels.shape[2], ...]
                input_im = temp_image
                '''
                # print(true_labels.shape)

                # extract overlaped area

                # im_shape = np.array(list(input_im.shape))
                if self.is_refine:
                # iniseg
                    ini_segs = self.dataset.iniseg_loader(
                        os.path.join(self.dataset.iniseg_path, self.dataset.inisegs[n_image][0])
                    )

                    if self.allLabel is not None:
                        ini_segs = self.trans_val_target.transform(ini_segs)
                    else:
                        # ini_segs = np.rollaxis(ini_segs, 0, -1)
                        ini_segs = np.append(np.expand_dims(np.zeros(ini_segs.shape[1:]), axis=0),
                                             ini_segs)
                        ini_segs = np.argmax(ini_segs, axis=0)
                        if self.to_onehot:
                            ini_segs = self.trans_onehot.transform(ini_segs)


                # print('sampling...')
                # print(ini_segs.shape)
                # print(true_labels.shape)
                # print(input_im.shape)

                if self.uni_vox:
                    # header loader presently only work with nifti images
                    input_im_info = self.head_loader(os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0]))
                    input_im_vox_size = np.array(input_im_info.header.get_zooms())
                    resample_trans = tx.ResampleVoxel(target_voxel_size=np.array([1,1,1]),
                                                      original_voxel_size=input_im_vox_size).transform

                    input_im, true_labels = resample_trans(input_im, true_labels)
                    if self.is_refine:
                        ini_segs = resample_trans(ini_segs)

                # resample
                # print(true_labels.shape)
                if self.augment_transform:
                    if self.is_refine:

                        input_im, true_labels, ini_segs = self.aug_tx.transform3objs(input_im,
                                                                        true_labels,
                                                                        ini_segs)
                    else:
                        input_im, true_labels = self.aug_tx.transform(input_im,
                                                                                 true_labels)

                # calculate labels sum
                # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]
                # input_im = self.shape_trans.transform(input_im)
                # true_labels = self.shape_trans.transform(true_labels)

                # if self.to_onehot:
                #     true_labels = self.trans_onehot.transform(true_labels)

                # print(true_labels.shape)
                # debugging code
                '''
                import matplotlib.pyplot as plt
                plt.figure(0)
                plt.imshow(input_im[:, :, 100])
                plt.figure(1)
                plt.imshow(true_labels[:, :, 100, 1])
                plt.figure(2)
                plt.imshow(true_labels[:, :, 100, 2])
                plt.figure(3)
                plt.imshow(true_labels[:, :, 100, 3])
                plt.figure(4)
                plt.imshow(true_labels[:, :, 100, 0])
                plt.show(block=True)
                '''
            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                           volume_occurs[n_volind])

            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn_with_shape(
                    input_im, centre_coord, output_shape=self.subvolumn_size
                )
                batch_X.append(batch_im)
                batch_target = self.extract_subvolumn_with_shape(
                    true_labels, centre_coord, output_shape=self.subvolumn_size
                )
                if self.to_onehot:
                    batch_target = self.trans_onehot.transform(batch_target)
                batch_Y.append(batch_target)

            # batch_X.append(input_im)
            # batch_Y.append(true_labels)


        batch_X = np.array(batch_X, copy=False)
        batch_Y = np.array(batch_Y, copy=False)

        # old volume indecs
        self.cache_volume_indeces = volume_indeces

        # print(batch_X.shape)
        # print(batch_Y.shape)
        # debuging code

        '''
        import matplotlib.pyplot as plt
        plt.figure(0)
        plt.imshow(batch_X[0, :, :, 100, 0])
        plt.figure(1)
        plt.imshow(batch_Y[0, :, :, 100, 1])
        plt.figure(2)
        plt.imshow(batch_Y[0, :, :, 100, 2])
        plt.figure(3)
        plt.imshow(batch_Y[0, :, :, 100, 3])
        plt.figure(4)
        plt.imshow(batch_Y[0, :, :, 100, 0])
        plt.show(block=True)
        '''

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        return batch_X, batch_Y

    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4,
                       include_background=False):

        if n_class is None:
            n_class = len(np.unique(label_vol))

        vx, vy, vz = np.meshgrid(
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[2], step=skip_ratio)
            )
        if include_background:
            class_occurs = np.random.choice(np.arange(start=0, stop=n_class), size=vol_occur)
        else:
            class_occurs = np.random.choice(np.arange(start=1, stop=n_class), size=vol_occur)
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


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):
        if np.any(vol.shape<self.subvolumn_size):
            vol_mins = target_mins = vol_maxs = target_maxs = np.zeros(int(len(vol.shape)))
            for i in range(len(vol.shape)):
                if vol.shape[i] < self.subvolumn_size[i]:
                    vol_mins[i] = target_mins[i] = 0
                    vol_maxs[i] = target_maxs[i] = vol.shape[i]
                else:
                    vol_mins[i], vol_maxs[i], target_mins[i], target_maxs[i] = \
                        self.extract_sub_range(vol.shape[i],
                                               centre_coord[i],
                                               self.subvolumn_size[i],
                                               limit_within_vol = limit_within_vol
                        )
            sub_vol = np.ones(self.subvolumn_size) * np.min(vol)
            sub_vol[target_mins[0]:target_maxs[0],
                target_mins[1]:target_maxs[1],
                target_mins[2]:target_maxs[2]] = \
                vol[vol_mins[0]:vol_maxs[0], vol_mins[1]:vol_maxs[1], vol_mins[2]:vol_maxs[2]]


        else:
            vol_minx, vol_maxx, target_minx, target_maxx = \
                self.extract_sub_range(vol.shape[0],
                                       centre_coord[0],
                                       self.subvolumn_size[0],
                                       limit_within_vol=limit_within_vol)
            vol_miny, vol_maxy, target_miny, target_maxy = \
                self.extract_sub_range(vol.shape[1],
                                       centre_coord[1],
                                       self.subvolumn_size[1],
                                       limit_within_vol=limit_within_vol)
            vol_minz, vol_maxz, target_minz, target_maxz = \
                self.extract_sub_range(vol.shape[2],
                                       centre_coord[2],
                                       self.subvolumn_size[2],
                                       limit_within_vol=limit_within_vol)


            sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_subvolumn_with_shape(self, vol, centre_coord, output_shape, limit_within_vol=True):
        # find the corresponding target volumn first
        sub_vol = self.extract_subvolumn(vol, centre_coord, limit_within_vol)
        assert(np.all(sub_vol.shape >= output_shape))
        half_out_shape = np.array(output_shape)//2
        # print(half_out_shape)
        centre_out = [sub_vol.shape[i]//2 if sub_vol.shape[i]%2==1 else sub_vol.shape[i]//2-1
                      for i in range(len(sub_vol.shape))]
        # 3D case
        vol_starts = [max(0, centre_out[i] - half_out_shape[i]) for i in range(len(half_out_shape))]
        vol_ends = [centre_out[i] + half_out_shape[i]+1 for i in range(len(half_out_shape))]
        sub_volnew = np.zeros(output_shape)
        sub_volnew[:, :, ...] = sub_vol[vol_starts[0]:vol_ends[0],
                  vol_starts[1]:vol_ends[1],
                  vol_starts[2]:vol_ends[2]]
        # print(sub_volnew.shape)
        if sub_volnew.ndim <3:
            for i in range(len(output_shape)):
                if half_out_shape[i] == 0:
                    sub_vol = np.expand_dims(sub_vol, i)
            #print(sub_vol.shape)
        return np.expand_dims(sub_volnew, axis=-1)

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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

class LoaderData3D_e2e(object):
    def __init__(self, dataset,
                 subvolumn_size = (64, 64, 64),
                 allLabel=None,
                 batch_size = 10,
                 volumn_num = 3,
                 is_lazy=False,
                 uni_vox_resample = False,
                 augment_transform=True,
                 onehot_trans = True,
                 output_shape = None):

        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]
        else:
            self.subvolumn_size = None

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))




        self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.to_onehot = onehot_trans
        if self.to_onehot:
            self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        # self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_lazy = is_lazy

        self.counter = -1
        # self.counter = np.random.randint(self.num_inputs)
        self.augment_transform = augment_transform
        self.uni_vox = uni_vox_resample
        # head loader
        self.head_loader = header_reader

        if output_shape is None:
            self.output_shape = self.subvolumn_size
        else:
            self.output_shape = output_shape


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.is_lazy:
            self.counter+=1
        return self.gen()
        #K.clear_session()
        # tf.reset_default_graph()


    next = __next__

    def gen(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        batch_Y_1 = []
        if self.is_lazy:
            # volume_indeces, volume_occurs = self.select_indices()
            volume_indeces = [self.counter % self.num_inputs]
            volume_occurs = [self.batch_size]
        else:
            volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            im_shape = np.array(list(input_im.shape))
            # print('sampling...')
            # print(ini_segs.shape)
            # print(true_labels.shape)
            # print(input_im.shape)

            if self.uni_vox:
                input_im_info = self.head_loader(os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0]))
                input_im_vox_size = np.array(input_im_info.header.get_zooms())
                resample_trans = tx.ResampleVoxel(target_voxel_size=np.array([1,1,1]),
                                                  original_voxel_size=input_im_vox_size).transform

                input_im, true_labels = resample_trans(input_im, true_labels)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)


            # resample

            if self.augment_transform:
                # augmentation
                aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                         translation_range=(0.1, 0.1),
                                         # translate btwn -10% and 10% horiz, -10% and 10% vert
                                         shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                         zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                         turn_off_frequency=5,
                                         fill_value='min',
                                         target_fill_mode='constant',
                                         target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)

                input_im, true_labels = aug_tx.transform(input_im,
                                                                             true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                           volume_occurs[n_volind],
                                                           n_class=int(len(self.allLabel)))

            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target_1 = self.extract_subvolumn_with_shape(
                    true_labels, centre_coord, output_shape=self.subvolumn_size
                )

                batch_target = self.extract_subvolumn_with_shape(
                    true_labels, centre_coord, output_shape=self.output_shape
                )
                # calculate the z dim
                if self.to_onehot:
                    batch_target_1 = self.trans_onehot.transform(batch_target_1)
                    batch_target = self.trans_onehot.transform(batch_target)
                # print(batch_target.shape)


                batch_X.append(batch_im)


                batch_Y.append(batch_target)
                batch_Y_1.append(batch_target_1)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        # print('ybatch1shape:', np.array(batch_Y_1).shape)

        return np.expand_dims(np.array(batch_X, copy=False), axis=-1),\
               {'modeli_out':np.array(batch_Y_1, copy=False),
                'models_out':np.array(batch_Y, copy=False)}


    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4):
        if n_class is None:
            n_class = len(np.unique(label_vol))
        vx, vy, vz = np.meshgrid(
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[2], step=skip_ratio)
            )
        class_occurs = np.random.choice(np.arange(start=1, stop=n_class), size=vol_occur)
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


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):
        if np.any(vol.shape<self.subvolumn_size):
            vol_mins = target_mins = vol_maxs = target_maxs = np.zeros(int(len(vol.shape)))
            for i in range(len(vol.shape)):
                if vol.shape[i] < self.subvolumn_size[i]:
                    vol_mins[i] = target_mins[i] = 0
                    vol_maxs[i] = target_maxs[i] = vol.shape[i]
                else:
                    vol_mins[i], vol_maxs[i], target_mins[i], target_maxs[i] = \
                        self.extract_sub_range(vol.shape[i],
                                               centre_coord[i],
                                               self.subvolumn_size[i],
                                               limit_within_vol = limit_within_vol
                        )
            sub_vol = np.ones(self.subvolumn_size) * np.min(vol)
            sub_vol[target_mins[0]:target_maxs[0],
                target_mins[1]:target_maxs[1],
                target_mins[2]:target_maxs[2]] = \
                vol[vol_mins[0]:vol_maxs[0], vol_mins[1]:vol_maxs[1], vol_mins[2]:vol_maxs[2]]


        else:
            vol_minx, vol_maxx, target_minx, target_maxx = \
                self.extract_sub_range(vol.shape[0],
                                       centre_coord[0],
                                       self.subvolumn_size[0],
                                       limit_within_vol=limit_within_vol)
            vol_miny, vol_maxy, target_miny, target_maxy = \
                self.extract_sub_range(vol.shape[1],
                                       centre_coord[1],
                                       self.subvolumn_size[1],
                                       limit_within_vol=limit_within_vol)
            vol_minz, vol_maxz, target_minz, target_maxz = \
                self.extract_sub_range(vol.shape[2],
                                       centre_coord[2],
                                       self.subvolumn_size[2],
                                       limit_within_vol=limit_within_vol)


            sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_subvolumn_with_shape(self, vol, centre_coord, output_shape, limit_within_vol=True):
        # find the corresponding target volumn first
        sub_vol = self.extract_subvolumn(vol, centre_coord, limit_within_vol)
        assert(np.all(sub_vol.shape >= output_shape))
        half_out_shape = np.array(output_shape)//2
        # print(half_out_shape)
        centre_out = [sub_vol.shape[i]//2 if sub_vol.shape[i]%2==1 else sub_vol.shape[i]//2-1
                      for i in range(len(sub_vol.shape))]
        # 3D case
        vol_starts = [max(0, centre_out[i] - half_out_shape[i]) for i in range(len(half_out_shape))]
        vol_ends = [centre_out[i] + half_out_shape[i]+1 for i in range(len(half_out_shape))]
        sub_volnew = np.zeros(output_shape)
        sub_volnew[:, :, ...] = sub_vol[vol_starts[0]:vol_ends[0],
                  vol_starts[1]:vol_ends[1],
                  vol_starts[2]:vol_ends[2]]
        # print(sub_volnew.shape)
        if sub_volnew.ndim <3:
            for i in range(len(output_shape)):
                if half_out_shape[i] == 0:
                    sub_vol = np.expand_dims(sub_vol, i)
            #print(sub_vol.shape)
        return np.expand_dims(sub_volnew, axis=-1)

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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


class LoaderData3D_e2e_tfbug(object):
    def __init__(self, dataset,
                 subvolumn_size = (64, 64, 64),
                 allLabel=None,
                 batch_size = 10,
                 volumn_num = 3,
                 is_lazy=False,
                 uni_vox_resample = False,
                 augment_transform=True,
                 onehot_trans = True,
                 output_shape = None):

        self.dataset = dataset

        if isinstance(subvolumn_size, (tuple, list)):
            self.subvolumn_size = subvolumn_size
        elif isinstance(subvolumn_size, (float, int)):
            self.subvolumn_size = [subvolumn_size for i in range(3)]
        else:
            self.subvolumn_size = None

        self.volumn_num = volumn_num

        self.allLabel = allLabel
        self.batch_size = batch_size

        self.num_inputs = int(len(self.dataset.inputs))




        self.trans_val_target = tx.ReplaceValWithInd(allLabel=allLabel)
        self.trans_val_input = tx.MinMaxScaler((-1, 1))
        self.to_onehot = onehot_trans
        if self.to_onehot:
            self.trans_onehot = tx.OneHotOriginalNoReshape(num_classes=int(len(self.allLabel)))
        # self.trans_resample = tx.CVDownSampleXY(target_shape=target_shape)
        self.trans_cropZ = tx.CropZLast(dimz=self.subvolumn_size[-1])

        self.is_lazy = is_lazy

        self.counter = -1
        # self.counter = np.random.randint(self.num_inputs)
        self.augment_transform = augment_transform
        self.uni_vox = uni_vox_resample
        # head loader
        self.head_loader = header_reader

        if output_shape is None:
            self.output_shape = self.subvolumn_size
        else:
            self.output_shape = output_shape


    def __len__(self):
        return (32 + self.batch_size - 1) // self.batch_size

    def __next__(self):
        if self.is_lazy:
            self.counter+=1
        return self.gen()
        #K.clear_session()
        # tf.reset_default_graph()


    next = __next__

    def gen(self):
        # while True:
        # select volumes and occurance
        batch_X = []
        batch_Y = []
        batch_Y_1 = []
        if self.is_lazy:
            # volume_indeces, volume_occurs = self.select_indices()
            volume_indeces = [self.counter % self.num_inputs]
            volume_occurs = [self.batch_size]
        else:
            volume_indeces, volume_occurs = self.select_indices()
        # print(volume_indeces)
        for n_volind in range(len(volume_indeces)):
            n_image = volume_indeces[n_volind]

            # true_labels
            true_labels = self.dataset.target_loader(
                os.path.join(self.dataset.base_path, self.dataset.targets[n_image][0])
            )

            # transform label values to ind
            true_labels = self.trans_val_target.transform(true_labels)

            # load image
            input_im = self.trans_val_input.transform(
                self.dataset.input_loader(
                    os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0])
                )
            )

            im_shape = np.array(list(input_im.shape))
            # print('sampling...')
            # print(ini_segs.shape)
            # print(true_labels.shape)
            # print(input_im.shape)

            if self.uni_vox:
                input_im_info = self.head_loader(os.path.join(self.dataset.base_path, self.dataset.inputs[n_image][0]))
                input_im_vox_size = np.array(input_im_info.header.get_zooms())
                resample_trans = tx.ResampleVoxel(target_voxel_size=np.array([1,1,1]),
                                                  original_voxel_size=input_im_vox_size).transform

                input_im, true_labels = resample_trans(input_im, true_labels)

            # check size
            if self.subvolumn_size[-1] > input_im.shape[-1]:
                input_im, true_labels = self.trans_cropZ.transform(input_im, true_labels)


            # resample

            if self.augment_transform:
                # augmentation
                aug_tx = tx.RandomAffine(rotation_range=(-15, 15),  # rotate btwn -15 & 15 degrees
                                         translation_range=(0.1, 0.1),
                                         # translate btwn -10% and 10% horiz, -10% and 10% vert
                                         shear_range=(-10, 10),  # shear btwn -10 and 10 degrees
                                         zoom_range=(0.85, 1.15),  # between 15% zoom-in and 15% zoom-out
                                         turn_off_frequency=5,
                                         fill_value='min',
                                         target_fill_mode='constant',
                                         target_fill_value=0)  # how often to just turn off random affine transform (units=#samples)

                input_im, true_labels = aug_tx.transform(input_im,
                                                                             true_labels)

            # calculate labels sum
            # label_sums = [np.sum(true_labels==i) for i in range(len(self.allLabel))]

            # select subvolume centres
            centre_coords, centre_Ys = self.select_centres(true_labels,
                                                           volume_occurs[n_volind],
                                                           n_class=int(len(self.allLabel)))

            # print(true_labels.shape)
            # print(input_im.shape)
            for centre_coord in centre_coords:
                batch_im = self.extract_subvolumn(
                    input_im, centre_coord
                )

                batch_target_1 = self.extract_subvolumn_with_shape(
                    true_labels, centre_coord, output_shape=self.subvolumn_size
                )

                batch_target = self.extract_subvolumn_with_shape(
                    true_labels, centre_coord, output_shape=self.output_shape
                )
                # calculate the z dim
                if self.to_onehot:
                    batch_target_1 = self.trans_onehot.transform(batch_target_1)
                    batch_target = self.trans_onehot.transform(batch_target)
                # print(batch_target.shape)


                batch_X.append(batch_im)


                batch_Y.append(batch_target)
                batch_Y_1.append(batch_target_1)

        # print('xbatchshape:', np.array(batch_X).shape)
        # print('ybatchshape:', np.array(batch_Y).shape)
        # print('ybatch1shape:', np.array(batch_Y_1).shape)

        return np.expand_dims(np.array(batch_X, copy=False), axis=-1),\
               {'modeli_out':np.array(batch_Y_1, copy=False),
                'models_out':np.array(batch_Y, copy=False)}


    def select_indices(self):
        volume_indeces = np.random.choice(self.num_inputs, self.volumn_num, replace=False)
        temp_choices = np.random.choice(volume_indeces, self.batch_size, replace=True)
        volume_occurs = np.array([np.sum(temp_choices==volume_indeces[i])
                                  for i in range(len(volume_indeces))])
        return volume_indeces, volume_occurs

    def select_centres(self, label_vol, vol_occur, n_class=None, skip_ratio=4):
        if n_class is None:
            n_class = len(np.unique(label_vol))
        vx, vy, vz = np.meshgrid(
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[0], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[1], step=skip_ratio),
            np.arange(start=np.random.choice(skip_ratio), stop=label_vol.shape[2], step=skip_ratio)
            )
        class_occurs = np.random.choice(np.arange(start=1, stop=n_class), size=vol_occur)
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


    def extract_subvolumn(self, vol, centre_coord, limit_within_vol=True):
        if np.any(vol.shape<self.subvolumn_size):
            vol_mins = target_mins = vol_maxs = target_maxs = np.zeros(int(len(vol.shape)))
            for i in range(len(vol.shape)):
                if vol.shape[i] < self.subvolumn_size[i]:
                    vol_mins[i] = target_mins[i] = 0
                    vol_maxs[i] = target_maxs[i] = vol.shape[i]
                else:
                    vol_mins[i], vol_maxs[i], target_mins[i], target_maxs[i] = \
                        self.extract_sub_range(vol.shape[i],
                                               centre_coord[i],
                                               self.subvolumn_size[i],
                                               limit_within_vol = limit_within_vol
                        )
            sub_vol = np.ones(self.subvolumn_size) * np.min(vol)
            sub_vol[target_mins[0]:target_maxs[0],
                target_mins[1]:target_maxs[1],
                target_mins[2]:target_maxs[2]] = \
                vol[vol_mins[0]:vol_maxs[0], vol_mins[1]:vol_maxs[1], vol_mins[2]:vol_maxs[2]]


        else:
            vol_minx, vol_maxx, target_minx, target_maxx = \
                self.extract_sub_range(vol.shape[0],
                                       centre_coord[0],
                                       self.subvolumn_size[0],
                                       limit_within_vol=limit_within_vol)
            vol_miny, vol_maxy, target_miny, target_maxy = \
                self.extract_sub_range(vol.shape[1],
                                       centre_coord[1],
                                       self.subvolumn_size[1],
                                       limit_within_vol=limit_within_vol)
            vol_minz, vol_maxz, target_minz, target_maxz = \
                self.extract_sub_range(vol.shape[2],
                                       centre_coord[2],
                                       self.subvolumn_size[2],
                                       limit_within_vol=limit_within_vol)


            sub_vol = np.ones(self.subvolumn_size)*np.min(vol)
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

    def extract_subvolumn_with_shape(self, vol, centre_coord, output_shape, limit_within_vol=True):
        # find the corresponding target volumn first
        sub_vol = self.extract_subvolumn(vol, centre_coord, limit_within_vol)
        assert(np.all(sub_vol.shape >= output_shape))
        half_out_shape = np.array(output_shape)//2
        # print(half_out_shape)
        centre_out = [sub_vol.shape[i]//2 if sub_vol.shape[i]%2==1 else sub_vol.shape[i]//2-1
                      for i in range(len(sub_vol.shape))]
        # 3D case
        vol_starts = [max(0, centre_out[i] - half_out_shape[i]) for i in range(len(half_out_shape))]
        vol_ends = [centre_out[i] + half_out_shape[i]+1 for i in range(len(half_out_shape))]
        sub_volnew = np.zeros(output_shape)
        sub_volnew[:, :, ...] = sub_vol[vol_starts[0]:vol_ends[0],
                  vol_starts[1]:vol_ends[1],
                  vol_starts[2]:vol_ends[2]]
        # print(sub_volnew.shape)
        if sub_volnew.ndim <3:
            for i in range(len(output_shape)):
                if half_out_shape[i] == 0:
                    sub_vol = np.expand_dims(sub_vol, i)
            #print(sub_vol.shape)
        return np.expand_dims(sub_volnew, axis=-1)

    def extract_sub_range(self, vol_dim, subvol_dim_centre, subvol_dim_size,
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