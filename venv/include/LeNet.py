# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-5-23 下午10:54
# File     :LeNet.py
# Location:/Home/PycharmProjects/..
import keras
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Dense, Flatten
from keras.models import Sequential


def LeNet_5():
    model = Sequential()
    # First Layer
    # input: 32 * 32
    # kernel: 5 * 5 with 6
    # feature map(output): 28 * 28 with 6
    model.add(Conv2D(6, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    # Second Layer
    # input: 28 * 28 with 6
    # pool size: 2 * 2
    # feature map(output): 14 * 14 with 6
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Third Layer
    # input: 14 * 14 with 6
    # kernel: 5 * 5 with 16
    # feature map(output): 10 * 10 with 16
    # with the combination from 6 to 16
    model.add(Conv2D(16, kernel_size=(5, 5), activation='relu'))
    # Forth Layer
    # input: 10 * 10 with 16
    # pool size: 2 * 2
    # feature map(output): 5 * 5 with 16
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Trans the matrix to vector
    model.add(Flatten())
    # Fifth Layer
    # input: 5 * 5 with 16
    # kernel: 5 * 5 with 120
    # feature map(output): 1 * 1 with 120
    model.add(Dense(120, activation='relu'))
    # Sixth Layer
    # input: 1 * 1 with 120
    # calculate: multiply with parameter and add a bias
    # feature map(output): 1 * 1
    model.add(Dense(84, activation='relu'))
    # Seventh Layer
    # input: 1 * 1 with 120
    # calculate: RBF(SUM((xj-wij)^2) tend to 0)
    # feature map(output): 1 * 1
    model.add(Dense(10, activation='softmax'))
    model.compile(loss=keras.metrics.categorical_crossentropy, optimizer=keras.optimizers.Adam(), metrics=['accuracy'])
    return model


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
    x_train = x_train / 255
    x_test = x_test / 255
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)
    model = LeNet_5()
    model.fit(x_train, y_train, batch_size=128, epochs=20, verbose=1, validation_data=(x_test, y_test))
    score = model.evaluate(x_test, y_test)
    print('Test Loss:', score[0])
    print('Test accuracy:', score[1])

    # model.predict(x_test[0])
