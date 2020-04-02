# encoding: utf-8
import pickle

import cv2
import numpy as np

from common import config


class Dataset():
    dataset_path = './dataset/cifar-10-batches-py/'

    def __init__(self, dataset_name):
        self.ds_name = dataset_name  # 加载数据集的名称 : train/test
        self.minibatch_size = config.minibatch_size  # batch-size
        self.instances = config.nr_instances  # 标签类别的数量
        if self.ds_name == 'train':
            self.files = [self.dataset_path + 'data_batch_{}'.format(i + 1) for i in range(5)]  # 总共有5个batch
        else:
            self.files = [self.dataset_path + 'test_batch']  # 有一个测试用的batch

    def load(self):
        datas_list, labels_list = [], []
        for file in self.files:  # data_batch_* 都是以pickle的形式保存的
            with open(file, 'rb') as f:
                samples = pickle.load(f, encoding='bytes')
                datas_list.extend(samples[b'data'])
                labels_list.extend(samples[b'labels'])
        self.samples_mat = {'X': datas_list, 'Y': labels_list}  # 得到所有的数据和标签
        file = open('dataset/cifar-10-batches-py/batches.meta', 'rb')
        meta = pickle.load(file)
        self.label_names = meta['label_names']
        return self

    @property
    def total_instances(self):
        return self.instances

    @property
    def minibatches(self):
        return self.instances // config.minibatch_size

    def instance_generator(self):  # 跟数据集的存储方式有关
        for i in range(self.instances):
            img_r = self.samples_mat['X'][i][:1024].reshape(config.image_shape[0], config.image_shape[1], 1)
            img_g = self.samples_mat['X'][i][1024:2048].reshape(config.image_shape[0], config.image_shape[1], 1)
            img_b = self.samples_mat['X'][i][2048:].reshape(config.image_shape[0], config.image_shape[1], 1)
            img = np.concatenate((img_r, img_g, img_b), axis=2)
            label = self.samples_mat['Y'][i]
            # img = affineTrans(img)  # 加入仿射变换
            # img = noise(img, 0.9)  # 加入椒盐噪声
            # img = bluring(img)  # 模糊化
            yield img.astype(np.float32), np.array(label, dtype=np.int32)


def affineTrans(img):
    pts1 = np.float32([[10, 10], [20, 5], [5, 20]])
    pts2 = np.float32([[10, 8], [18, 5], [5, 20]])
    M = cv2.getAffineTransform(pts1, pts2)
    return cv2.warpAffine(img, M, (32, 32))


def bluring(img):
    return cv2.blur(img, (5, 5))


def noise(img, SNR=0.7):
    img_ = img.transpose(2, 1, 0)
    c, h, w = img_.shape
    mask = np.random.choice((0, 1, 2), size=(1, h, w), p=[SNR, (1 - SNR) / 2., (1 - SNR) / 2.])
    mask = np.repeat(mask, c, axis=0)  # 按channel 复制到 与img具有相同的shape
    img_[mask == 1] = 255  # pepper
    img_[mask == 2] = 0  # white
    return img_.transpose(2, 1, 0)


if __name__ == "__main__":
    ds = Dataset('train')
    ds = ds.load()
    gen = ds.instance_generator()

    imggrid = []
    label_names = ds.label_names
    while True:
        for i in range(25):
            img, label = next(gen)
            # img = affineTrans(img)
            # img = cv2.blur(img, (5, 5))
            img = noise(img, 0.5)
            print(img.shape)
            img = cv2.resize(img, (96, 96))
            cv2.putText(img, label_names[label], (0, config.image_shape[0]), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (255, 255, 255), 2)
            imggrid.append(img)
            # print(img.shape, label)

        imggrid = np.array(imggrid).reshape((5, 5, img.shape[0], img.shape[1], img.shape[2]))
        imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5 * img.shape[0], 5 * img.shape[1], 3))

        cv2.imshow('', imggrid.astype('uint8'))
        cv2.waitKey(0)
        cv2.imwrite('./Notes/img/noise_0.5.png', imggrid)
        print("done")
        break
