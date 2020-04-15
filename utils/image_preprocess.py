#!/usr/bin/python3
"""
   image preprocess for CT or MR
"""
import os
import nibabel as nib
import matplotlib.pyplot as plt
import cv2
import numpy as np
import SimpleITK as itk
import tensorflow as tf


# read all image path from file
def read_image_path(file_path):
    image_path_list = os.listdir(file_path)  # 读取nii文件夹
    return image_path_list


# show image with plt and image data(array)
def show_image(*image_data_list):
    max_row = int(len(image_data_list)/2 + 0.5)
    plt.figure()
    for i in range(0, len(image_data_list)):
        plt.subplot(max_row, 2, i+1)
        plt.imshow(image_data_list[i], cmap="gray")
    plt.show()


def image_info(itk_img):
    print("itk_image_array.shape: {}".format(itk.GetArrayFromImage(itk_img).shape))
    # 原点Origin，大小Size，间距Spacing和方向Direction
    print("itk_image.origin: {}".format(itk_img.GetOrigin()))
    print("itk_image.size: {}".format(itk_img.GetSize()))
    print("itk_image.spacing: {}".format(itk_img.GetSpacing()))
    print("itk_image.direction: {}".format(itk_img.GetDirection()))
    # 纬度信息
    print("itk_image.dimension: {}".format(itk_img.GetDimension()))
    print("itk_image.width(矢状面): {}".format(itk_img.GetWidth()))
    print("itk_image.height(冠状面): {}".format(itk_img.GetHeight()))
    print("itk_image.depth(横断面): {}".format(itk_img.GetDepth()))
    # 体素类型
    print("itk_image.pixel_id_value: {}".format(itk_img.GetPixelIDValue()))
    print("itk_image.pixel_id_type_as_string: {}".format(itk_img.GetPixelIDTypeAsString()))
    print("itk_image.number_of_components_per_pixel: {}".format(itk_img.GetNumberOfComponentsPerPixel()))


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


# 将源文件和标签文件裁剪为(256,256,256)大小后保存
def image_crop(val_image, label_image):
    # val_image = val_image.transpose(2, 1, 0)  # 原始数组为三维数组(x,y,z),则原始axis排列为(0,1,2),则transpose()的默认参数为(2,1,0)，
    # 得到转置后的数组视图，不影响原数组的内容以及大小，这里实际上是x轴和z轴进行了交换
    val_image = val_image[::-1, ::-1, :]  # 进行了一个左右翻转
    label_image = label_image[::-1, ::-1, :]  # 此时的label为uint8类型的，需要转换为bool类型再进行操作
    bool_label = label_image.astype(np.bool)  # 将label转换成bool类型
    axis_list = np.where(bool_label)  # 输出满足条件（即非0）元素的坐标，这里的坐标以元组的形式给出，原数组有三维，所以tuple中有三个数组
    # print axis_list#输出元组,元组中有三个数组，数组用arry数组存储

    center_x = (axis_list[0].max() + axis_list[0].min()) / 2  # 获得x轴的中间值
    center_y = (axis_list[1].max() + axis_list[1].min()) / 2  # 获得y轴的中间值
    center_z = (axis_list[2].max() + axis_list[2].min()) / 2  # 获得z轴的中间值
    center_point = [np.array(center_x, np.int32), np.array(center_y, np.int32),
                    np.array(center_z, np.int32)]  # 用列表来存储三个arry数组，arry数组中有两个参数第一个为x轴的坐标，第二个参数为数据类型dtype
    print("center_point: {}".format(center_point))
    # 将标签和原图像裁剪为(256, 256, 256)大小
    label_block = label_image
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
    print("label_block: {}".format(label_block.shape))
    print("image_block: {}".format(image_block.shape))
    block_sum = np.sum(label_block.astype(np.bool))  # 裁剪后标签中的label中bool值为真的像素个数
    print("block_sum: ", block_sum)
    all_sum = np.sum(bool_label)  # 裁剪前标签中的label中bool值为真的像素个数
    print("all_sum: ", all_sum)
    show_image(image_block[128, :, :], image_block[:, 128, :], image_block[:, :, 128],
                label_block[128, :, :], label_block[:, 128, :], label_block[:, :, 128])


if __name__ == "__main__":
    image_path = "example/ct_train_1001_image.nii.gz"
    label_path = "example/ct_train_1001_label.nii.gz"
    itk_image = itk.ReadImage(image_path)
    itk_label = itk.ReadImage(label_path)
    itk_image_array = itk.GetArrayFromImage(itk_image)
    itk_label_array = itk.GetArrayFromImage(itk_label)
    show_image(itk_image_array[80, :, :], itk_image_array[:, 128, :], itk_image_array[:, :, 128],
               itk_label_array[80, :, :], itk_label_array[:, 128, :], itk_label_array[:, :, 128])
    image_info(itk_image)
    # show_image(itk_image_array[80, :, :], itk_image_array[:, 256, :], itk_image_array[:, :, 256])

    new_itk_image = re_sample_img(itk_image, out_spacing=[1.0, 1.0, 1.0], is_label=False)
    new_itk_label = re_sample_img(itk_label, out_spacing=[1.0, 1.0, 1.0], is_label=True)

    new_itk_image_array = itk.GetArrayFromImage(new_itk_image)
    new_itk_label_array = itk.GetArrayFromImage(new_itk_label)
    show_image(new_itk_image_array[80, :, :], new_itk_image_array[:, 128, :], new_itk_image_array[:, :, 128],
               new_itk_label_array[80, :, :], new_itk_label_array[:, 128, :], new_itk_label_array[:, :, 128])
    image_info(new_itk_image)
    # image_crop(new_itk_image_array, new_itk_label_array)
    # image_info(new_itk_image)
    # show_image(itk_image_array[80, :, :], itk_image_array[:, 256, :], itk_image_array[:, :, 256],
    #            new_itk_image_array[80, :, :], new_itk_image_array[:, 256, :], new_itk_image_array[:, :, 256])
    # itk_label = re_sample_img(itk_label, out_spacing=[1.0, 1.0, 1.0], is_label=True)

    # img_data = itk.GetArrayFromImage(itk_image)
    # lab_data = itk.GetArrayFromImage(itk_label)

    # file_name = "mr_train_nii_gz"
    # image_paths = read_image_path(file_name)
    # nib_img, img_data = read_image_data(file_name + "/" + image_paths[0])

    # new_img_data = tf.image.per_image_standardization(img_data)
    # show_image(img_data[128, :, :], lab_data[128, :, :])
    # for i in range(img_data.shape[-1]):
    #     show_image(normal_image(img_data[:, :, i]), lab_data[:, :, i])


