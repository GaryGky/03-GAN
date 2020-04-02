# encoding: utf-8

import os


class Config:
    '''where to write all the logging information during training(includes saved models)'''
    log_dir = './train_log'

    '''where to write model snapshots to'''
    log_model_dir = os.path.join(log_dir, 'models')

    exp_name = os.path.basename(log_dir)

    nr_instances = 50  # 选择50张图像产生对抗样本
    minibatch_size = 1
    nr_channel = 3
    image_shape = (32, 32,3)
    nr_class = 10 # 标签类别数量
    nr_epoch = 500  ### you may need to increase nr_epoch to 4000 or more for targeted adversarial attacks

    weight_decay = 1e-10

    show_interval = 10

    '''mean and standard deviation for normalizing the image input '''
    mean = 120.707
    std = 64.15

config = Config()
