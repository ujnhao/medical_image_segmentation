#!/usr/bin/python3
"""
   image preprocess for MRI
"""
import os
import numpy as np
import SimpleITK as itk
import cv2
from skimage import transform
import matplotlib.pyplot as plt


# show image with plt and image data(array)
def show_image(*image_data_list):
    max_row = int(len(image_data_list)/2 + 0.5)
    plt.figure()
    for i in range(0, len(image_data_list)):
        plt.subplot(max_row, 2, i+1)
        plt.imshow(image_data_list[i], cmap="gray")
    plt.show()


# 读取训练集的图片
def read_image_name(fid):
    image_list = os.listdir(fid)  # 读取nii文件夹
    image_name_list = []
    for name_str in image_list:
        str_list = str.split(name_str, ".", -1)
        name = str_list[0]
        name = name.rstrip("_image")
        name = name.rstrip("_label")
        if name not in image_name_list:
            image_name_list.append(name)
    return image_name_list


# Re_sample images to 1mm spacing with SimpleITK
def re_sample_img(itk_img, out_spacing=[1.0, 1.0, 1.0], is_label=False):
    original_spacing = itk_img.GetSpacing()
    original_size = itk_img.GetSize()
    out_size = [
        int(np.round(original_size[0] * (original_spacing[0] / out_spacing[0]))),
        int(np.round(original_size[1] * (original_spacing[1] / out_spacing[1]))),
        int(np.round(original_size[2] * (original_spacing[2] / out_spacing[2])))]
    re_sample = itk.ResampleImageFilter()
    re_sample.SetOutputSpacing(out_spacing)
    re_sample.SetSize(out_size)
    re_sample.SetOutputDirection(itk_img.GetDirection())
    re_sample.SetOutputOrigin(itk_img.GetOrigin())
    re_sample.SetTransform(itk.Transform())
    re_sample.SetDefaultPixelValue(itk_img.GetPixelIDValue())
    if is_label:
        re_sample.SetInterpolator(itk.sitkNearestNeighbor)
    else:
        # re_sample.SetInterpolator(itk.sitkBSpline)
        re_sample.SetInterpolator(itk.sitkLinear)
    return re_sample.Execute(itk_img)


# 将源文件和标签文件裁剪为(256,256,256)大小
def image_crop(val_image, val_label):
    val_image = val_image.transpose(2, 1, 0)  # 原始数组为三维数组(x,y,z),则原始axis排列为(0,1,2),则transpose()的默认参数为(2,1,0)，
    # 得到转置后的数组视图，不影响原数组的内容以及大小，这里实际上是x轴和z轴进行了交换
    val_image = val_image[::-1, ::-1, :]  # 进行了一个左右翻转

    val_label = val_label.transpose(2, 1, 0)  # 原始数组为三维数组(x,y,z),则原始axis排列为(0,1,2),则transpose()的默认参数为(2,1,0)，
    val_label = val_label[::-1, ::-1, :]  # 此时的label为uint8类型的，需要转换为bool类型再进行操作
    bool_label = val_label.astype(np.bool)  # 将label转换成bool类型
    axis_list = np.where(bool_label)  # 输出满足条件（即非0）元素的坐标，这里的坐标以元组的形式给出，原数组有三维，所以tuple中有三个数组
    # print axis_list#输出元组,元组中有三个数组，数组用arry数组存储

    center_x = (axis_list[0].max() + axis_list[0].min()) / 2  # 获得x轴的中间值
    center_y = (axis_list[1].max() + axis_list[1].min()) / 2  # 获得y轴的中间值
    center_z = (axis_list[2].max() + axis_list[2].min()) / 2  # 获得z轴的中间值
    center_point = [np.array(center_x, np.int32), np.array(center_y, np.int32),
                    np.array(center_z, np.int32)]  # 用列表来存储三个arry数组，arry数组中有两个参数第一个为x轴的坐标，第二个参数为数据类型dtype
    # print("center_point: {}".format(center_point))
    # 将标签和原图像裁剪为(256, 256, 256)大小
    label_block = val_label
    image_block = val_image
    if center_point[0] >= 128:
        label_block = label_block[center_point[0] - 128:center_point[0] + 128, :, :]
        image_block = image_block[center_point[0] - 128:center_point[0] + 128, :, :]
    else:
        label_block = label_block[128 - center_point[0]:384 - center_point[0], :, :]
        image_block = image_block[128 - center_point[0]:384 - center_point[0], :, :]

    if center_point[1] >= 128:
        label_block = label_block[:, center_point[1] - 128:center_point[1] + 128, :]
        image_block = image_block[:, center_point[1] - 128:center_point[1] + 128, :]
    else:
        label_block = label_block[:, 128 - center_point[1]:384 - center_point[1], :]
        image_block = image_block[:, 128 - center_point[1]:384 - center_point[1], :]

    if center_point[2] >= 128:
        label_block = label_block[:, :, center_point[2] - 128:center_point[2] + 128]
        image_block = image_block[:, :, center_point[2] - 128:center_point[2] + 128]
    else:
        label_block = label_block[:, :, 128 - center_point[2]:384 - center_point[2]]
        image_block = image_block[:, :, 128 - center_point[2]:384 - center_point[2]]
    # print("label_block: {}".format(label_block.shape))
    # print("image_block: {}".format(image_block.shape))
    # block_sum = np.sum(label_block.astype(np.bool))  # 裁剪后标签中的label中bool值为真的像素个数
    # print("block_sum: ", block_sum)
    # all_sum = np.sum(bool_label)  # 裁剪前标签中的label中bool值为真的像素个数
    # print("all_sum: ", all_sum)
    return image_block, label_block


# 归一化
def normalize(img):
    img = img.astype(np.float32)
    img = (img - img.min()) / (img.max() - img.min())
    return img


# image_resize (256, 256, 256)
def image_resize(img_data):
    (x, y, z) = img_data.shape
    x = max(x, 256)
    y = max(y, 256)
    z = max(z, 256)
    img_data = transform.resize(img_data, (x, y, z))
    return img_data


def mri_image_preprocess():
    file_path = "ct_train2"
    npz_file_path = "npz_" + file_path
    # file_path = "ct_train1"
    # read all nii.gz
    image_name_list = read_image_name(file_path)
    print(image_name_list)
    image_list = []
    for image_name in image_name_list:
        image_path = file_path + "/" + image_name + "_image.nii.gz"
        label_path = file_path + "/" + image_name + "_label.nii.gz"
        npz_path = npz_file_path + "/" + image_name + ".npz"

        # 读取图像
        itk_image = itk.ReadImage(image_path)
        itk_label = itk.ReadImage(label_path)
        # 重采样
        itk_image = re_sample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False)
        itk_label = re_sample_img(itk_label, out_spacing=[1.0, 1.0, 1.0], is_label=True)
        val_image, val_label = itk.GetArrayFromImage(itk_image), itk.GetArrayFromImage(itk_label)
        # 归一化
        val_image = normalize(val_image)
        val_label = normalize(val_label)
        # 缩放
        (x, y, z) = val_image.shape
        if min(x, y, z) < 256:
            val_image = image_resize(val_image)
            val_label = image_resize(val_label)
        # 剪切256 x 256 x 256
        val_image, val_label = image_crop(val_image, val_label)
        # image_list.append(val_image[:, :, 129])
        # image_list.append(val_label[:, :, 129])
        # show_image(val_image[:, :, 129], val_label[:, :, 129])
        np.savez(npz_path, val_image[:, :, 127:130], val_image[:, :, 127:130])
        # cv2.imshow(image_name+"_image", val_image[:, :, 127:130])
        # cv2.imshow(image_name+"_label", val_label[:, :, 127:130])
        # cv2.waitKey(1000)


if __name__ == "__main__":
    mri_image_preprocess()





