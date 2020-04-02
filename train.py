# encoding: utf-8

import numpy as np
import tensorflow.compat.v1 as tf
import cv2
from attack import Attack
from common import config
from custom_vgg16_bn import Vgg16
from dataset import Dataset
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

tf.disable_eager_execution()
configTf = tf.ConfigProto()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
configTf.gpu_options.allow_growth = True


def get_dataset_batch(ds_name):  # 得到一个数据加载的迭代器
    dataset = Dataset(ds_name)
    ds_gnr = dataset.load().instance_generator
    ds = tf.data.Dataset.from_generator(ds_gnr, output_types=(tf.float32, tf.int32), )
    ds = ds.repeat(config.nr_epoch)  # 500
    ds = ds.batch(config.minibatch_size)  # 1
    ds_iter = ds.make_one_shot_iterator()
    sample_gnr = ds_iter.get_next()
    return sample_gnr, dataset


def output_img(x, session=None, out_path=None):
    if session is not None:
        img = np.clip(session.run(x), 0, 1) * 255
    else:
        img = x
    img = img.astype('uint8')
    img = np.reshape(img, config.image_shape)
    cv2.imwrite(out_path, img)


def main():
    ## load dataset
    global adv_examples
    train_batch_gnr, train_set = get_dataset_batch(ds_name='train')

    data = tf.placeholder(tf.float32, shape=(config.minibatch_size,) + config.image_shape, name='data')
    label = tf.placeholder(tf.int32, shape=(None,), name='label')  # placeholder for targetted label
    groundTruth = tf.placeholder(tf.int32, shape=(None,), name='groundTruth')  # 真实的样本标签

    pre_noise = tf.Variable(tf.zeros([1, 32, 32, 3], dtype=tf.float32))
    vgg16 = Vgg16()
    attack = Attack(vgg16, config.minibatch_size)  # 初始化一个攻击模型对象
    acc, loss, adv = attack.generate_graph(pre_noise, data, groundTruth, label)
    acc_gt, preds = attack.evaluate(data, groundTruth)

    placeholders = {
        'data': data,
        'label': label,
        'groundTruth': groundTruth,

    }

    lr = 0.01
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(loss, [pre_noise])  # 优化目标是noise
    train = opt.apply_gradients(grads)
    ## create a session
    tf.set_random_seed(12345)  # ensure consistent results
    succ = 0
    noise_l2 = 0
    target = np.array([7])  # horse
    with tf.Session(config=configTf) as sess:
        # print(train_set.minibatches)
        sess.run(tf.global_variables_initializer())  # init all variables
        for idx in range(train_set.minibatches):
            global_cnt = 0
            images, labels = sess.run(train_batch_gnr)
            for epoch in range(1, config.nr_epoch + 1):
                global_cnt += 1
                feed_dict = {
                    placeholders['data']: images,
                    placeholders['label']: target,
                    placeholders['groundTruth']: labels,
                }
                "target attack should convert all the pics to horse"
                _, accuracy, loss_batch, adv_examples = sess.run([train, acc, loss, adv],
                                                                 feed_dict=feed_dict)

                if global_cnt % config.show_interval == 0:  # 10
                    # train_writer.add_summary(summary, global_cnt)
                    print(
                        "e:{}/{}, {}".format(idx, train_set.minibatches, epoch),
                        'loss: {:.3f}'.format(loss_batch),
                        'accuracy: {:3f}'.format(accuracy),
                    )
                    # outPath = './out/' + '{}_{}.jpg'.format(idx, global_cnt)
                    # print(outPath)
                    # output_img(pre_noise, sess, outPath)

            print('Training for batch {} is done'.format(idx))
            output_img(x=adv_examples, out_path='./out/adv_examples/{}.png'.format(idx))  # 输出对抗样本
            accuracy_gt, predAdv = sess.run([acc_gt, preds],
                                            feed_dict={placeholders['data']: adv_examples,
                                                       placeholders['label']: target})  # 计算对抗样本精确率
            succ = (idx * succ + 1 - accuracy_gt) / (idx + 1)
            noise_l2 = (idx * (noise_l2) + ((adv_examples - images)) ** 2) / (idx + 1)
            print("====> label for adv is :", predAdv) # 希望是 9 = truck
            print('Success rate of this attack is {}'.format(succ))
            print('Noise norm of this attack is {}'.format(np.mean(noise_l2)))


if __name__ == "__main__":
    print(tf.__version__)
    main()
