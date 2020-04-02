# encoding: utf-8

import tensorflow.compat.v1 as tf
import numpy as np

from common import config

mean, std = config.mean, config.std


class Attack():

    def __init__(self, model, batchsize):
        self.batchsize = batchsize
        self.model = model  # pretrained vgg model used as classifier

    '''Build computation graph for generating adversarial examples'''

    def generate_graph(self, pre_noise, x, groundTruth, target=None):
        noise = 10 * tf.tanh(pre_noise)
        # print(x)
        x_noise = x + noise  ## add perturbation and get adversarial examples: 加入噪声获得对抗样本
        x_clip = tf.clip_by_value(x_noise, 0, 255)  # 放缩到0-255
        x_round = x_clip + tf.stop_gradient(x_clip // 1 - x_clip)  # 在对抗样本处进行反传截断
        x_norm = (x_round - mean) / (std + 1e-7)  # 进行标准化
        logits = self.model.build(x_norm)  # 流过vgg16
        preds = tf.nn.softmax(logits)  # softmax激活
        if target != None:
            print("<<< target mode >>>")
            target_one_hot = tf.one_hot(target, config.nr_class)
        else:
            target_one_hot = tf.one_hot(groundTruth, config.nr_class)

        "计算损失和acc：使用交叉熵"
        alpha = 0.0001
        "loss尽量小 === 最大化熵 === 欺骗成功"
        # loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=logits)) * (-1)
        "target攻击"
        loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(labels=target_one_hot, logits=logits))
        # l2_loss = tf.reduce_mean((x_round - x) ** 2)
        # loss += l2_loss * alpha
        "在untarget 攻击中，acc=0表示攻击成功；"
        "在target 攻击中，acc=1表示攻击成功"
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                                              tf.cast(tf.argmax(target_one_hot, 1), dtype=tf.int32)), tf.float32))
        return acc, loss, x_round

    '''Build a graph for evaluating the classification result of adversarial examples'''

    def evaluate(self, x, gt):
        x = (x - mean) / (std + 1e-7)
        logits = self.model.build(x)  # 使用vgg计算一次x的标签
        preds = tf.nn.softmax(logits)  # preds 得到标签
        gt_one_hot = tf.one_hot(gt, config.nr_class)  # gt: 样本实际的标签
        acc = tf.reduce_mean(tf.cast(tf.equal(tf.cast(tf.argmax(preds, 1), dtype=tf.int32),
                                              tf.cast(tf.argmax(gt_one_hot, 1), dtype=tf.int32)), tf.float32))
        return acc, tf.argmax(preds, 1)
