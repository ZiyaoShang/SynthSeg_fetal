import nibabel as nib
import glob
import os
import numpy as np

def print_file_info(file_path = '/home/zshang/SP/data/ZURICH/mri/sub-010_rec-mial_T2w.nii.gz', seg=False):
    # Load the NIfTI file
    img = nib.load(file_path)

    # same as aff mtx
    # qform = img.header.get_qform()
    # print("QForm value:")
    # print(qform)

    print("Image shape:", img.shape)

    qform_code = img.header["qform_code"]
    print("QForm Code:", qform_code)

    print("Affine matrix:")
    print(img.affine)

    n_dims = 3
    resolution = img.header['pixdim'][1:n_dims + 1]
    print("Resolution:", resolution)

def preprocess_fetal():

    # Specify the path to the folder and the file extension pattern
    folder_path = '/home/zshang/SP/data/ZURICH/experiments/epoch1'
    file_extension = '*.nii.gz'  # you can specify a particular extension, e.g., '*.txt' for text files

    # List all files in the folder with the specified extension
    files = glob.glob(os.path.join(folder_path, file_extension))

    # Print the list of files
    for file in files:
        img = nib.load(file)
        print(img.get_fdata().shape)
