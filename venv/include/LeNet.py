# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-5-23 下午10:54
# File     :LeNet.py
# Location:/Home/PycharmProjects/..
from keras import layers
from keras.models import Model
import numpy as np


class ImageLoader(Loader):
    def get_picture(self, content, index):
        # file head is 16 byte
        # 28 * 28 is a picture
        start = index * 28 * 28 + 16
        picture = []
        for i in range(28):
            # add a element
            picture.append([])
            for j in range(28):
                byte1 = content[start + i * 28 + j]
                picture[i].append(byte1)
        return picture


def lenet_5(in_shape=(32, 32, 1), n_classes=10, opt='sgd'):
    in_layer = layers.Input(in_shape)
    conv1 = layers.Conv2D(filters=20, kernel_size=5, padding='same', activation='relu')(in_layer)
    pool1 = layers.MaxPool2D()(conv1)
    conv2 = layers.Conv2D(filters=50, kernel_size=5, padding='same', activation='relu')(pool1)
    pool2 = layers.MaxPool2D()(conv2)
    flatten = layers.Flatten()(pool2)
    dense1 = layers.Dense(500, activation='relu')(flatten)
    preds = layers.Dense(n_classes, activation='softmax')(dense1)
    model = Model(in_layer, preds)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    return model


if __name__ == '__main__':
    # model = lenet_5()
    # image = ImageLoader('train-images.idx3-ubyte', 5)
    image = ImageLoader('t10k-images.idx3-ubyte')
    binfile = open(filename, 'rb')
    print(binfile.read(10))
    # print(model.summary())
