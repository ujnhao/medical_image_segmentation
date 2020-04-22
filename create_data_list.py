import os
import numpy as np


def make_datalist(data_fd, data_list):
    filename_all = os.listdir(data_fd)
    filename_all = [data_fd+'/'+filename+'\n' for filename in filename_all if filename.endswith('.tfrecords')]

    np.random.shuffle(filename_all)
    np.random.shuffle(filename_all)
    with open(data_list, 'w') as fp:
        fp.writelines(filename_all)


if __name__ == '__main__':
    data_fd = "list/mr_val_tfs"
    data_list = 'list/mr_val_list'
    make_datalist(data_fd, data_list)
    data_fd = "list/mr_train_tfs"
    data_list = 'list/mr_train_list'
    make_datalist(data_fd, data_list)
    data_fd = "list/ct_val_tfs"
    data_list = 'list/ct_val_list'
    make_datalist(data_fd, data_list)
    data_fd = "list/ct_train_tfs"
    data_list = 'list/ct_train_list'
    make_datalist(data_fd, data_list)
