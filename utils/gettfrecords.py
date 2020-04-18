#!/usr/bin/python3
import numpy as np
import os
import tensorflow as tf
import cv2

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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


def read_decode_samples(image_list, shuffle=False):
    decomp_feature = {
        # image size, dimensions of 3 consecutive slices
        'dsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'dsize_dim2': tf.FixedLenFeature([], tf.int64), # 3
        # label size, dimension of the middle slice
        'lsize_dim0': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim1': tf.FixedLenFeature([], tf.int64), # 256
        'lsize_dim2': tf.FixedLenFeature([], tf.int64), # 1
        # image slices of size [256, 256, 3]
        'data_vol': tf.FixedLenFeature([], tf.string),
        # label slice of size [256, 256, 1]
        'label_vol': tf.FixedLenFeature([], tf.string)}

    raw_size = [256, 256, 3]
    volume_size = [256, 256, 3]
    label_size = [256, 256, 1]

    data_queue = tf.train.string_input_producer(image_list, shuffle=shuffle)
    reader = tf.TFRecordReader()
    fid, serialized_example = reader.read(data_queue)
    parser = tf.parse_single_example(serialized_example, features=decomp_feature)

    data_vol = tf.decode_raw(parser['data_vol'], tf.float32)
    data_vol = tf.reshape(data_vol, raw_size)
    data_vol = tf.slice(data_vol, [0, 0, 0], volume_size)

    label_vol = tf.decode_raw(parser['label_vol'], tf.float32)
    label_vol = tf.reshape(label_vol, raw_size)
    label_vol = tf.slice(label_vol, [0, 0, 1], label_size)
    data_vol, label_vol = tf.train.shuffle_batch([data_vol, label_vol], 1, 500, 100)
    return data_vol, label_vol


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
    # filepth = "example/mr_train_1001_slice_48.tfrecords"
    # data_vol, label_vol = read_decode_samples([filepth], True)
    # print(data_vol)
    #
    # sess = tf.Session()
    # init = tf.global_variables_initializer()
    # sess.run(init)
    #
    # # **5.启动队列进行数据读取
    # # 下面的 coord 是个线程协调器，把启动队列的时候加上线程协调器。
    # # 这样，在数据读取完毕以后，调用协调器把线程全部都关了。
    # coord = tf.train.Coordinator()
    # threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    # y_outputs = list()
    # _X_batch, _y_batch = sess.run([data_vol, label_vol])
    # img_data = _X_batch[0]
    # lab_data = _y_batch[0]
    # cv2.imshow("img_data", img_data)
    # cv2.imshow("lab_data", lab_data)
    # cv2.waitKey(0)
    #
    # # **6.最后记得把队列关掉
    # coord.request_stop()
    # coord.join(threads)
    #
