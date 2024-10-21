import os
import numpy as np

# for dicom
import pydicom

# for nifti
import nibabel as nib
from nibabel.processing import resample_to_output

# for nrrd
try:
    import nrrd
except:
    pass

# for other image files
from PIL import Image

# dicom readers
def dicomReadSlice_reader3D(dicomPath, ext = '.dcm', sort_order='ascend'):
    '''
    need test!!!
    :param dicomPath:
    :return: arrayDicom, dicomHeaders
    '''
    imList = []
    lstFilesDCM = []

    for dirName, subdirList, fileList in os.walk(dicomPath):
        for filename in fileList:
            if ext in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))

    sliceLocations = np.zeros(len(lstFilesDCM))

    # Get ref file
    RefDs = pydicom.read_file(lstFilesDCM[0])

    # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
    ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))

    # Load spacing values (in mm)
    # ConstPixelSpacing = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))

    # x = np.arange(0.0, (ConstPixelDims[0] + 1) * ConstPixelSpacing[0], ConstPixelSpacing[0])
    # y = np.arange(0.0, (ConstPixelDims[1] + 1) * ConstPixelSpacing[1], ConstPixelSpacing[1])
    # z = np.arange(0.0, (ConstPixelDims[2] + 1) * ConstPixelSpacing[2], ConstPixelSpacing[2])

    # The array is sized based on 'ConstPixelDims'
    arrayDicom = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)
    dicomHeaders = list()

    # loop through all the DICOM files
    for i, filenameDCM in enumerate(lstFilesDCM):
        # read the file
        ds = pydicom.read_file(filenameDCM)
        # store the raw image data
        arrayDicom[:, :, i] = ds.pixel_array

        # find slice location
        IOP = np.array(ds.ImageOrientationPatient).astype('float32')
        IPP = np.array(ds.ImagePositionPatient).astype('float32')
        sliceLocations[i] = np.dot( np.cross(IOP[:3], IOP[3:]),
                                    IPP)
        dicomHeaders.append(ds)

    slice_inds = np.argsort(sliceLocations)
    if sort_order=='descend':
        slice_inds = slice_inds[::-1]
    elif sort_order=='ascend':
        pass

    arrayDicom = arrayDicom[:, :, slice_inds]
    dicomHeaders = np.array(dicomHeaders)[slice_inds]

    return arrayDicom, dicomHeaders

# convert dicom to nifti
def dicom3D_to_nifti(in_path, out_path):
    # import nibabel as nib
    arrayDicom, dicomHeaders = dicomReadSlice_reader3D(in_path)
    im = nib.Nifti1Image(arrayDicom, np.eye(4))
    im.to_filename(out_path)

# read nifti file
def nifti_reader_3D(path, to_output = False):
    temp_data = nib.load(path)
    if temp_data.get_data().ndim == 4:
        temp_data = nib.four_to_three(temp_data)[0]
    if to_output:
        temp_data = resample_to_output(temp_data, mode='nearest')
    return temp_data.get_data(), temp_data.header

def nifti_reader_3D_affine(path, to_output = False):
    temp_data = nib.load(path)
    if temp_data.get_data().ndim == 4:
        temp_data = nib.four_to_three(temp_data)[0]
    if to_output:
        temp_data = resample_to_output(temp_data, mode='nearest')
    return temp_data.get_data(), temp_data.header, temp_data.affine

def nifti_reader_4D(path):
    temp = nib.load(path)
    return temp.get_data(), temp.header

# read nrrd file
def nrrd_reader_3D(path):
    temp = nrrd.read(path)
    return temp[0], temp[1]

# read 2D dicom file
def dicom_reader_2D(path):
    ds = pydicom.read_file(path)
    return ds.pixel_array, ds

# read other images
def img_reader_2D(path, n_channel=1):
    img = Image.open(path).convert('RGB')
    if n_channel==1:
        img = img[0, ...] * 0.299 + img[1, ...] * 0.587 + img[2, ...] * 0.114
    return img, None

# simple test
if __name__=="__main__":
    # fill the dicom path to it
    dicom_path = '/home/cwang/styleEx/pairedData/pair2/11216/20170307_125954.383000/8_CaScSeq__3.0__B35f__70%/'
    out_path = './test.nii.gz'

    dicom3D_to_nifti(dicom_path, out_path)
    print('done')