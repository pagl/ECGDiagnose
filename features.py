#-*- coding: utf-8 -*-

import scipy.io
import scipy.signal
import numpy
import matplotlib.pyplot
import pywt
import glob, os
import preprocessing
import numpy as np
import random


def display_plot(data):
  fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)
  fig.subplots_adjust(hspace = 0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
  if (len(np.shape(data)) == 2):
    for y in data:    
      ax1.plot(range(0,len(y)), y)
      ax2.specgram(y, Fs = 1000)
  else:
    ax1.plot(range(0,len(data)), data)
    ax2.specgram(data, Fs = 1000)
 

def flatten(matrix):
  i = 0
  flat = []
  for row in matrix:
    for el in row:
      flat.insert(i, el)
      i += 1
  return flat


def get_disease(file):
  disease_header = "Reason for admission:"
  with open(file) as f:
    lines = f.readlines()
  
  for line in lines:
    if (line.find(disease_header) != -1):
      disease = line[line.find(disease_header) + len(disease_header) + 1 : line.find("\\")]
      break
  return disease


def mean(features):
  mean = []
  for j in range(len(features[0])):
    sum = 0
    for i in range(len(features)):
      sum = sum + features[i][j]
    mean.append(sum / len(features))
  return mean

def get_features():
  features = []
  diseases = []
  os.chdir("patients")
  for dir in glob.glob("patient*"):
    print("Processing:\t" + str(dir))
    os.chdir(dir)
    for file in glob.glob("*.mat"):
      matrix = scipy.io.loadmat(file)
 
      # Pobranie cech sygnału
      wavelet = preprocessing.getFeature(matrix['val'][0], '0')

      # Wartość średnia z otrzymanego wavelet'u
      mean_val = mean(wavelet)

      # Sprawdzenie czy powiodła się ekstrakcja cech
      if (len(mean_val) != 32): continue
      
      # Dopisanie uśrednionych cech do macierzy cech
      features.append(mean_val)
   
      # Dopisanie choroby do listy
      diseases.append(get_disease(file[:-4] + '.hea'))
      
      # Wyświetlenie otrzymanych danych na wykresach   
      #display_plot(mean_val)
      #display_plot(features)

    os.chdir("..")
  return (features, diseases)


def fuzzy_values(diseases):
  fuzzy_diseases = []
  next_value = 1
  dict = {}
  for disease in diseases:
    if (not disease in dict):
      dict[disease] = next_value
      next_value += 1
    fuzzy_diseases.append(dict[disease])
  return fuzzy_diseases


def main():
  features, diseases = get_features()
  diseases = fuzzy_values(diseases)
  os.chdir("..")
  scipy.io.savemat('features.mat', {'features':features})
  scipy.io.savemat('diseases.mat', {'diseases':diseases})
  matplotlib.pyplot.show()


########################################################################
###################  JUST TO VISUALIZE OUR FEATURES  ################### 
########################################################################
def features_presentation(number_of_plots):
  os.chdir("patients")
  plot_number = 0
  dir_list = glob.glob("patient*")
  
  for i in range(number_of_plots):
    os.chdir(random.choice(dir_list))

    files_list = glob.glob("*.mat")
    file = random.choice(files_list)
    print("Processing:\t" + str(file))
   
    matrix = scipy.io.loadmat(file)
    plot_number = plot_number + 1

    # Pobranie cech sygnału
    wavelet = preprocessing.getFeature(matrix['val'][0], '0')

    # Wartość średnia z otrzymanego wavelet'u
    mean_val = mean(wavelet)
      
    # Wyświetlenie otrzymanych danych na wykresach   
    fig = matplotlib.pyplot.figure()
    fig.subplots_adjust(hspace = 0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
      
      
    ax1 = matplotlib.pyplot.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = matplotlib.pyplot.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = matplotlib.pyplot.subplot2grid((3, 2), (2, 0))
    ax4 = matplotlib.pyplot.subplot2grid((3, 2), (2, 1))
    ax1.set_title(str(get_disease(file[:-4] + '.hea')))
    ax1.set_xlim([0, len(mean_val) - 1])
    ax2.set_xlim([0, len(mean_val) - 1])
      
    for y in wavelet:      
      ax1.plot(range(0,len(y)), y)
      ax3.specgram(y, Fs = 1000, cmap=matplotlib.pyplot.cm.gist_rainbow)
    ax2.plot(range(0, len(mean_val)), mean_val)
    ax4.specgram(mean_val, Fs = 1000, cmap=matplotlib.pyplot.cm.gist_rainbow)

    os.chdir("..")
  matplotlib.pyplot.show()
########################################################################


if __name__ == '__main__':
  #main()
  features_presentation(6)
