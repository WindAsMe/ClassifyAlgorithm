# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-5-23 下午10:51
# File     :svm.py
# Location:/Home/PycharmProjects/..
from sklearn import svm
from sklearn import datasets


if __name__ == '__main__':
    # svm
    svm_classify = svm.SVC(C=0.8, kernel='rbf', gamma=20)
    iris = datasets.load_iris()
    svm_classify.fit(iris.data, iris.target)
    predict_label = svm_classify.predict([[0.1, 0.2, 0.3, 0.4]])
    print(predict_label)
