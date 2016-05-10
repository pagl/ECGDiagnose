#-*- coding: utf-8 -*-

import scipy.io
from sklearn import svm
import numpy as np
import matplotlib.pyplot as plt

def svm_learning(X, Y):
  clf = svm.SVC(C=65536, gamma=2.44e-044)
  clf.fit(X, Y)
  print(clf.predict([ 0.44067216,  0.45812484,  0.47282945,  0.51718433,  0.53712795,
        0.5439901 ,  0.57148691,  0.58477731,  0.53844574,  0.4848824 ,
        0.42371767,  0.39437274,  0.36930259,  0.36319576,  0.36292256,
        0.32747079,  0.25828682,  0.3636618 ,  0.5969749 ,  1.22962655,
        1.73652569,  1.32664482,  0.57743304,  0.36848299,  0.28599255,
        0.31475894,  0.38366971,  0.41998929,  0.42087317,  0.40774348,
        0.41542523,  0.41012193]))


def main():
  features = scipy.io.loadmat("features.mat")['features']
  diseases = scipy.io.loadmat("diseases.mat")['diseases'][0]

  svm_learning(features, diseases)


if __name__ == '__main__':
  main()
