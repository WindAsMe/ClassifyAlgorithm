# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-6-20 上午11:58
# File     :tree.py
# Location:/Home/PycharmProjects/..
import sklearn
from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split


if __name__ == '__main__':
    iris = datasets.load_iris()
    dtree = DecisionTreeClassifier()
    dtree.fit(iris.data, iris.target)
    y_pre = dtree.predict(iris.data)
    print("Decision Tree\n sample: %d  wrong: %d" % (iris.data.shape[0], (iris.target != y_pre).sum()))
