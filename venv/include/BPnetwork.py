# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-5-28 下午12:39
# File     :BPnetwork.py
# Location:/Home/PycharmProjects/..
import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from sklearn import datasets
from numpy import random
from sklearn.model_selection import train_test_split


def Split(data):
    num = 10
    x = np.floor(10*random.rand(num, 2))
    y = np.floor(10*random.rand(num, 1))
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.4, random_state=0)
    return x_train, x_test, y_train, y_test


def BP():
    model = Sequential()
    model.add(Dense(12, input_dim=2, init='uniform'))
    model.add(Activation('relu'))
    model.add(Dense(1, input_dim=12))
    model.compile(loss='mean_squared_error', optimizer='sgd')
    return model


if __name__ == '__main__':

    x_train, x_test, y_train, y_test = Split(datasets.load_iris())
    model = BP()
    model.fit(x_train, y_train, epochs=20, verbose=1, validation_data=(x_test, y_test))
    y_pre = model.predict(x_test)
    print(y_pre)
    print(y_test)
    print((y_pre != y_test).sum())
    print('Accuracy:', (y_pre == y_test).sum() / len(y_pre))
    # print('Test Loss:', score[0])
    # print('Test accuracy:', score[1])
