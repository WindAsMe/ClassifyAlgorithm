# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-5-23 下午9:56
# File     :knn.py
# Location:/Home/PycharmProjects/..
from sklearn import neighbors
from sklearn import datasets


if __name__ == '__main__':
    # knn
    knn_classify = neighbors.KNeighborsClassifier()
    iris = datasets.load_iris()
    knn_classify.fit(iris.data, iris.target)
    predict_label = knn_classify.predict([[0.1, 0.2, 0.3, 0.4]])
    print(predict_label)
