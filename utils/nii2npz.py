#!/usr/bin/python3
import numpy as np
import medpy.io as medio


def nii2npz():
    data_nii_pth = 'example/mr_train_1001_image.nii.gz'
    label_nii_pth = 'example/mr_train_1001_label.nii.gz'
    npz_pth = 'example/mr_train_1001.npz'
    data_arr, _ = medio.load(data_nii_pth)
    label_arr, _ = medio.load(label_nii_pth)
    np.savez(npz_pth, data_arr, label_arr)


if __name__=="__main__":
    nii2npz()