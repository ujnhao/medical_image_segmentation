#!/usr/bin/python3
import numpy as np
import os
import tensorflow as tf
import cv2


# 读取训练集的图片
def read_image_name(fid):
    image_list = os.listdir(fid)  # 读取npz文件夹
    image_name_list = []
    for name_str in image_list:
        str_list = str.split(name_str, ".", -1)
        name = str_list[0]
        if name not in image_name_list:
            image_name_list.append(name)
    return image_name_list


# npz to tfs
def npz2tfrecords(image_path, tfrecord_path):
    file, _ = os.path.split(tfrecord_path)
    if not os.path.exists(file):
        os.makedirs(file)
    npz_data = np.load(image_path)
    data_vol_val = npz_data['arr_0']
    label_vol_val = npz_data['arr_1']
    # print(data_vol_val.shape)
    # cv2.imshow("data", data_vol_val)
    # cv2.imshow("label", label_vol_val)
    # cv2.waitKey(0)
    # return

    dsize_dim0_val = data_vol_val.shape[0]
    dsize_dim1_val = data_vol_val.shape[1]
    dsize_dim2_val = data_vol_val.shape[2]
    lsize_dim0_val = label_vol_val.shape[0]
    lsize_dim1_val = label_vol_val.shape[1]
    lsize_dim2_val = label_vol_val.shape[2]

    writer = tf.python_io.TFRecordWriter(tfrecord_path)

    feature = {'data_vol': tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(data_vol_val.tostring())])),
        'dsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim0_val])),
        'dsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim1_val])),
        'dsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[dsize_dim2_val])),
        'lsize_dim0': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim0_val])),
        'lsize_dim1': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim1_val])),
        'lsize_dim2': tf.train.Feature(int64_list=tf.train.Int64List(value=[lsize_dim2_val])),
        'label_vol': tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(label_vol_val.tostring())])), }

    example = tf.train.Example(features=tf.train.Features(feature=feature))
    writer.write(example.SerializeToString())
    writer.close()


def do_npz2tfs_handle():
    img_file = "ct_train"
    npz_img_file = "npz_" + img_file
    tfs_img_file = "tfs_" + img_file
    npz_img_name_list = read_image_name(npz_img_file)
    for npz_img_name in npz_img_name_list:
        npz_img_path = npz_img_file + "/" + npz_img_name + ".npz"
        tfs_img_path = tfs_img_file + "/" + npz_img_name + ".tfrecords"
        npz2tfrecords(npz_img_path, tfs_img_path)


if __name__ == "__main__":
    do_npz2tfs_handle()

