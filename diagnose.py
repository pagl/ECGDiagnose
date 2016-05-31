#-*- coding: utf-8 -*-

import csv
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import BernoulliRBM
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedKFold
from sklearn.pipeline import Pipeline, FeatureUnion;
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest
from sklearn.cross_validation import train_test_split
from sklearn import metrics


##############################################################
################### CLASSIFIERS DEFINITION ###################
##############################################################

names = ["Nearest Neighbors", 
         "Decision Tree",
         "Random Forest", 
         "AdaBoost", 
         "Naive Bayes",
         "Linear SVM", 
         "RBF SVM",
         "One Vs Rest"]
        
classifiers = [KNeighborsClassifier(n_neighbors=10), 
               DecisionTreeClassifier(max_depth=25),
               RandomForestClassifier(max_depth=25, n_estimators=20, max_features=3),
               AdaBoostClassifier(),
               GaussianNB(),
               SVC(kernel="linear"),
               SVC(gamma=2.44e-044, C=65536),
               OneVsRestClassifier(SVC(kernel="linear"))]  


##############################################################
########################    METRICS   ########################   
##############################################################

def accuracy (y_test, pred):  
  return metrics.accuracy_score(y_test, pred)

def auc (y_test, pred):
  return metrics.roc_auc_score(y_test, pred)

def average_precision (y_test, pred):
  return metrics.average_precision_score(y_test, pred)

def f1_score (y_test, pred):
  return metrics.f1_score(y_test, pred, average='macro')

def precision (y_test, pred):
  return metrics.precision_score(y_test, pred, average='macro')

def mean_squared_error (y_test, pred):
  return metrics.mean_squared_error(y_test, pred)

def recall_score (y_test, pred):
  return metrics.recall_score(y_test, pred, average='macro')

def median_absolute_error (y_test, pred):
  return metrics.median_absolute_error(y_test, pred)

def classification_report (y_test, y_pred,class_names):
  return metrics.classification_report(y_test, y_pred,target_names = class_names,digits = 4)
  
##############################################################


def test(X, y, class_names,folds=10,):
  file = open('diagnose.csv', 'w+')
  file_report = open('report.csv', 'w+')

  file.write('%d/%d,%s,%s,%s,%s,%s\n' % (len(X), len(X[0]), "Accuracy", "F1", "Precision", "Mean Square Error", "Recall"))
  X = np.array(X)
  y = np.array(y)

  # Pętla po badanych przez nas klasyfikatorach
  for name, clf in zip(names, classifiers):
    accuracyV = 0
    aucV = 0
    apV = 0
    f1V = 0
    precisionV = 0
    mean_squared_errorV = 0
    recall_scoreV = 0

    # Sprawiedliwy podział zbioru danych na 10 podzbiorów 
    skf = StratifiedKFold(y, shuffle = True, n_folds=folds)

    # Wybor zbioru testowego do podsumowania
    X_F_train, X_F_test, y_F_train, y_F_test = train_test_split(X, y, test_size=0.1)

    for train_index, test_index in skf:
      X_train, X_test = X[train_index], X[test_index]
      y_train, y_test = y[train_index], y[test_index]

      # Umieszczanie cech wewnątrz klasyfikatora
      clf.fit(X_train, y_train)

      # Predykcja choroby
      pred = clf.predict(X_test)

      # Ewaluacja metryk
      accuracyV += accuracy(y_test, pred)
      f1V += f1_score(y_test, pred)
      precisionV += precision(y_test, pred)
      mean_squared_errorV += mean_squared_error(y_test, pred)
      recall_scoreV += recall_score(y_test, pred)

    accuracyV /= folds
    f1V /= folds
    precisionV /= folds
    mean_squared_errorV /= folds
    recall_scoreV /= folds

    # Umieszczanie cech wewnątrz klasyfikatora
    clf.fit(X_F_train, y_F_train)
    
    # Predykcja choroby
    pred = clf.predict(X_F_test)

    print("\n\n%s:" % name)
    print ("Accuracy: %f" % accuracyV)
    print ("F1 Score: %f" % f1V)
    print ("Precision: %f" % precisionV)
    print ("Mean Squared Error: %f" % mean_squared_errorV)
    print ("Recall: %f" % recall_scoreV)   
    print(classification_report(y_F_test, pred,class_names))

    file.write('%s,%s,%s,%s,%s,%s\n' % (name, str(accuracyV), str(f1V), str(precisionV), str(mean_squared_errorV), str(recall_scoreV)))
    file_report.write('%s\n%s\n\n' % (name, classification_report(y_F_test, pred,class_names)))
  file.close()
  file_report.close()


def feature_selection_test(features, diseases,class_names):
  
  # Dekompozycja cech sygnalu do 2 wymiarow
  #pca = PCA(n_components=2)

  # Wbudowana selekcja cech
  #selection = SelectKBest(k=1)

  #combined_features = FeatureUnion([("pca", pca), ("univ_select", selection)])
  #X_features = combined_features.fit(features, diseases).transform(features)

  test(features, diseases,class_names)


def remove_index(features, diseases, it):
  features_temp = []
  diseases_temp = []
  for index,x in enumerate(features):
    
    if(diseases[index] !=it+1):
      features_temp.append(x)
      diseases_temp.append(diseases[index])
  return features_temp,diseases_temp


def remove_useless_cases(features,diseases,diseases_name, number_threshold, remove = True):
  count_array = [0] * len(diseases_name)
  if(remove):
    for index, x in enumerate(diseases):
      count_array[x-1]+=1
    temp = 0
    for index,i in enumerate(count_array):
      if(i < number_threshold or diseases_name[index-temp].find("n/a") != -1):
        diseases_name = np.delete(diseases_name,index-temp,0)
        temp += 1
        features,diseases = remove_index(features,diseases,index)
  return features,diseases,diseases_name


def main():
  features = scipy.io.loadmat("features.mat")['features']
  diseases = scipy.io.loadmat("diseases.mat")['diseases'][0]
  diseases_name = scipy.io.loadmat("diseases_name.mat")['target_names']

  features, diseases, diseases_name = remove_useless_cases(features, diseases, diseases_name, 50, False)
  feature_selection_test(features, diseases, diseases_name)
  


if __name__ == '__main__':
  main()
