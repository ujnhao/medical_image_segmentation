"""
    This is the basic training script for the baseline MRI or CT Model
    It is used to train the source segmenter
"""
import os
import sys
import logging
import datetime
import argparse

import tensorflow as tf
from tensorflow.python import debug as tf_debug
import source_segmenter as drn
import numpy as np
from lib import _read_lists
import json
import platform


with open('./config_param.json') as config_file:
    config = json.load(config_file)

logging.basicConfig(filename="general_log", level=logging.DEBUG)
currtime = datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def main():
    output_path = "./tmp_exps/mr_baseline"
    restore = False  # set True if resume training from stored model
    restored_path = output_path
    lr_update_flag = False  # Set True if want to use a new learning rate for fine-tuning
    num_cls = 5
    batch_size = 10
    training_iters = 10
    checkpoint_space = 1500
    image_summeris = True

    cost_kwargs = {
        "cross_flag": True,  # use cross entropy loss
        "miu_cross": 1.0,
        "dice_flag": True,  # use dice loss
        "miu_dice": 1.0,
        "regularizer": 1e-4
    }

    opt_kwargs = {
        "learning_rate": 1e-3
    }

    try:
        os.makedirs(output_path)
    except Exception as err:
        print("folder exist!", err)

    net = drn.Full_DRN(channels=3, batch_size=batch_size, n_class=num_cls, image_summeris=image_summeris,
                       cost_kwargs=cost_kwargs)
    print("Network has been built!")

    train_list = _read_lists(config["train_fid"])
    val_list = _read_lists(config["val_fid"])
    print("train_list_len:%d" % len(train_list))
    print("val_list_len:%d" % len(val_list))

    trainer = drn.Trainer(net, train_list=train_list, val_list=val_list, num_cls=num_cls,
                          batch_size=batch_size, opt_kwargs=opt_kwargs, checkpoint_space=checkpoint_space,
                          optimizer=config["optimizer"], lr_update_flag=lr_update_flag)

    # start tensorboard before getting started
    if platform.system() is "Linux":
        command1 = "tensorboard --logdir=" + output_path + " --port=6999 " + " &"
        os.system(command1)

    print("Now start training...")
    if restore is True:
        trainer.train(output_path=output_path, training_iters=training_iters, epochs=config["epochs"], restore=True,
                      restored_path=restored_path)
    else:
        trainer.train(output_path=output_path, training_iters=training_iters, epochs=config["epochs"])


if __name__ == "__main__":
    main()
