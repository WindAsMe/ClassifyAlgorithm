# !/usr/bin/python3
# -- coding: UTF-8 --
# Author   :WindAsMe
# Date     :19-5-27 下午9:55
# File     :bayes.py
# Location:/Home/PycharmProjects/..
from sklearn import datasets
# Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
# Multinomial Naive Bayes
from sklearn.naive_bayes import MultinomialNB
# Bernoulli Naive Bayes
from sklearn.naive_bayes import BernoulliNB

if __name__ == '__main__':
    iris = datasets.load_iris()
    gnb = GaussianNB()
    gnb.fit(iris.data, iris.target)
    y_pre = gnb.predict(iris.data)
    print("Gaussian Naive Bayes\n sample: %d  wrong: %d" % (iris.data.shape[0], (iris.target != y_pre).sum()))

    mnb = MultinomialNB()
    mnb.fit(iris.data, iris.target)
    y_pre = mnb.predict(iris.data)
    print("Multinomial Naive Bayes\n sample: %d  wrong: %d" % (iris.data.shape[0], (iris.target != y_pre).sum()))

    bnb = BernoulliNB()
    bnb.fit(iris.data, iris.target)
    y_pre = bnb.predict(iris.data)
    print("Bernoulli Naive Bayes\n sample: %d  wrong: %d" % (iris.data.shape[0], (iris.target != y_pre).sum()))
