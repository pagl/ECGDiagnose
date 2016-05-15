#-*- coding: utf-8 -*-

import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import RandomizedPCA


################## CLASSIFIERS DEFINITION ##################

names = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Decision Tree",
         "Random Forest", 
         "AdaBoost", 
         "Naive Bayes"]

classifiers = [KNeighborsClassifier(20), 
               SVC(kernel="linear", C=0.025),
               SVC(gamma=2.44e-044, C=65536), 
               DecisionTreeClassifier(max_depth=5),
               RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
               AdaBoostClassifier(),
               GaussianNB()]  

############################################################


def visualize(clf, name, ax, X_train, y_train, X_test, y_test):
  pca = RandomizedPCA(n_components=2)
  pca.fit(X_train)
  X_train = pca.transform(X_train)
  pca.fit(X_test)
  X_test = pca.transform(X_test)

  h = .02
  x_min, x_max = X_test[:, 0].min() - 1, X_test[:, 0].max() + 1
  y_min, y_max = X_test[:, 1].min() - 1, X_test[:, 1].max() + 1
  xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
  
  clf.fit(X_train, y_train)
  Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
  Z = Z.reshape(xx.shape)
  
  ax.contourf(xx, yy, Z, cmap = plt.cm.Paired, alpha=0.9)
  ax.scatter(X_test[:, 0], X_test[:, 1], c = y_test, cmap=plt.cm.Paired)
  ax.set_title(name)
  ax.set_xlim(xx.min(), xx.max())
  ax.set_ylim(yy.min(), yy.max())
  ax.set_xticks(())
  ax.set_yticks(())
 
  plt.tight_layout()
  plt.show()


def accuracy(clf, name, X_train, y_train, X_test, y_test):  
  clf.fit(X_train, y_train)
  predictions = clf.predict(X_test)
  accuracy = accuracy_score(y_test, predictions)
  print("%s: %f" % (name, accuracy))


def test(X, y):
  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .15)
  figure, ax = plt.subplots(2, 4)
  for name, clf, i in zip(names, classifiers, range(len(classifiers))):
    #visualize(clf, name, ax[i / 4, i % 4], X_train, y_train, X_test, y_test)
    accuracy(clf, name, X_train, y_train, X_test, y_test)


def main():
  features = scipy.io.loadmat("features.mat")['features']
  diseases = scipy.io.loadmat("diseases.mat")['diseases'][0]
  test(features, diseases)


if __name__ == '__main__':
  main()
