import pickle

import cv2
import numpy as np
import tensorflow as tf

from custom_vgg16_bn import Vgg16

vgg16 = Vgg16()
imgPath = 'out/target/out/adv_examples/'

file = open('dataset/cifar-10-batches-py/batches.meta', 'rb')
meta = pickle.load(file)
label_names = meta['label_names']
print(label_names[9])

imggrid = []

cnt = 0
acc = np.zeros([10])
for index in range(100):
    inp = cv2.imread(imgPath + '{}.png'.format(index))
    # inp = cv2.cvtColor(inp, cv2.COLOR_BGR2RGB)
    tmp = inp
    inp = np.reshape(inp, (1, 32, 32, 3))
    inp = tf.convert_to_tensor(inp)
    inp = tf.cast(inp, tf.float32)
    logits = vgg16.build(inp)
    preds = tf.nn.softmax(logits)
    label = np.int(tf.cast(tf.argmax(preds, 1), dtype=tf.int32))  # 取得标签
    inp = np.array(inp)
    inp = tmp
    inp = cv2.resize(inp, (96, 96))  # 放大图片
    acc[label] += 1
    print(cnt, "===>", label_names[label])
    cnt += 1

    # if (cnt <= 25):
    #     cv2.putText(inp, label_names[label], (0, 32), cv2.FONT_HERSHEY_SIMPLEX,
    #                 1, (255, 255, 255), 2)
    #     imggrid.append(inp)
    #     # print(img.shape, label)
    # if cnt == 25:
    #     print(acc)
    #     imggrid = np.array(imggrid).reshape(5, 5, inp.shape[0], inp.shape[1], inp.shape[2])
    #     imggrid = imggrid.transpose((0, 2, 1, 3, 4)).reshape((5 * inp.shape[0], 5 * inp.shape[1], 3))
    #     cv2.imshow('', imggrid.astype('uint8'))
    #     cv2.waitKey(0)
    #     # cv2.imwrite('./Notes/img/res/targetRes_new.png', imggrid)
    #     print("done")

import matplotlib.pyplot as plt

print(acc)
ticks = label_names
plt.figure(figsize=(10, 5))
plt.bar(range(10), acc, tick_label=ticks, color='y')
plt.savefig("./bar1.jpg")
plt.show()
