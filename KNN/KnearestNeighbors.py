# k nearest neighbors
# @rviniciussouza
#
# Equipe: Vinicius Rodrigues, Marcus Magalhães, Daniel Veloso Braga

import numpy
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics

iris = load_iris()

X = iris.data[:,:2]
y = iris.target

xt = numpy.concatenate([X[:40,:], X[51:90,:], X[101:140,:]])
yt = numpy.concatenate([y[:40], y[51:90], y[101:140]])

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(xt, yt)

xv = numpy.concatenate([X[:40,:], X[51:90,:], X[101:140,:]])
yv = numpy.concatenate([y[:40], y[51:90], y[101:140]])

pred = knn.predict(xv)

print(metrics.accuracy_score(pred, yv))