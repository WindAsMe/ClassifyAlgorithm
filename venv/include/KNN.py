# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-5-22 下午8:50
# File     :KNN.py
# Location:/Home/PycharmProjects/..
import numpy as np


def knn(trainData, trainLabel, testData, K):
    # The predict label
    predictLabel = []
    # For every test data
    for i in range(0, len(testData)):
        # To calculate the distance for test data with every sample
        dis = []
        for j in range(0, len(trainData)):
            a = 0
            for length in range(0, len(trainData[j])):
                a += abs(pow(trainData[j][length] - testData[i][length], 2))
            dis.append(a)
        d = dict()
        for j in range(0, len(dis)):
            if dis[j] <= K * K:
                if d.get(trainLabel[j]) is None:
                    d[trainLabel[j]] = 1
                else:
                    d[trainLabel[j]] += 1
        time = 0
        for key in d.keys():
            if d[key] > time:
                time = d[key]
                predictLabel.append(key)
    return predictLabel


if __name__ == '__main__':
    train_data = np.array([[2, 1], [2, 3], [1, 2], [1, 3], [0, 2], [2, 0], [2, 4], [4, 2]])
    train_label = [0, 0, 0, 0, 1, 1, 1, 1]
    test_data = np.array([[2, 2], [0, 0]])
    print(knn(train_data, train_label, test_data, 5))
