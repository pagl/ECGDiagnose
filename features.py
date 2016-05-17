#-*- coding: utf-8 -*-

import math
import scipy.io
import scipy.signal
import numpy
import matplotlib.pyplot
import pywt
import glob, os
import preprocessing
import numpy as np
import random
import sklearn.preprocessing


'''   Spłaszczenie macierzy do jednego wymiaru
      @param matrix Macierz wejściowa
      @return Macierz reprezentowana w jednym wymiarze
'''
def flatten(matrix):
  i = 0
  flat = []
  for row in matrix:
    for el in row:
      flat.insert(i, el)
      i += 1
  return flat


'''   Przygotowuje liste chorób dla danego pliku
      @param file Nazwa pliku .hea
      @return Choroba odpowiadająca danemu plikowi
'''
def get_disease(file):
  disease_header = "Reason for admission:"
  with open(file) as f:
    lines = f.readlines()
  
  for line in lines:
    if (line.find(disease_header) != -1):
      disease = line[line.find(disease_header) + len(disease_header) + 1 : line.find("\\")]
      break
  return disease


'''   Wartość średnia
      @param signal Zbiór sygnałów otrzymanych z transformacji falkowej
      @return Lista reprezentująca wartość średnią dla każdego punktu
'''
def mean(signal):
  mean = []
  for j in range(len(signal[0])):
    sum = 0
    for i in range(len(signal)):
      sum += signal[i][j]
    mean.append(sum / len(signal))
  return mean



'''   Wariancja
      @param signal Zbiór sygnałów otrzymanych z transformacji falkowej
      @param mean Wyliczona wartość średnia dla każdego punktu
      @return Lista reprezentująca wariancje dla każdego punktu
'''
def variance(signal, mean):
  variance = []
  for j in range(len(signal[0])):
    sum = 0
    for i in range(len(signal)):
      sum += pow(signal[i][j] - mean[j], 2)
    variance.append(sum / len(signal))
  return variance



'''   Odchylenie standardowe 
      @param variance Wariancja z wcześniej otrzymanego sygnału
      @return Lista reprezentująca odchylenie standardowe dla każdego punktu
'''
def stddev(variance):
  stddev = []
  for i in range(len(variance)):
    stddev.append(math.sqrt(float(variance[i])))
  return stddev



'''   Ekstrakcja cech sygnału EKG
      1) Wyrównanie sygnału 
      2) Normalizacja sygnału
      3) Wykrycie uderzeń serca (Pan, Tompkin's AlgorithM)
      4) Transformacja falkowa
      5) Wartość średnia transformacji falkowej
      6) Wariancja transformacji falkowej
      7) Odchylenie standardowe transformacji falkowej
'''
def get_features():
  features = []
  diseases = []
  os.chdir("patients")
  for dir in glob.glob("patient*"):
    print("Processing:\t" + str(dir))
    os.chdir(dir)
    for file in glob.glob("*.mat"):
      matrix = scipy.io.loadmat(file)['val']
      local_features = []
      correct = True

      for signal in matrix:
        # Wyrównanie sygnału
        mvavg = moving_average(signal, 5001)
 
        # Pobranie cech sygnału
        wavelet = preprocessing.getFeature(mvavg, '0')

        # Wartość średnia z otrzymanego wavelet'u
        meanV = mean(wavelet)
        if (len(meanV) != 32): 
          correct = False
          break

        # Wariancja z otrzymanego wavelet'u
        varianceV = variance(wavelet, meanV)

        # Odchylenie standardowe z otrzymanego wavelet'u
        stddevV = stddev(varianceV)

        # Transformata Fouriera przeprowadzona po średniej wartości transformaty falkowej
        fftV = fourier_transform(meanV)

        # Dopisanie wyliczonych cech z wavelet'u
        local_features.append(meanV)
        #local_features.append(varianceV)    # Redundancja cech? 
        local_features.append(stddevV)
        local_features.append(fftV)
      
      # Sprawdzenie czy powiodła się ekstrakcja cech
      if (correct == True):
        # Spłaszczenie i dopisane lokalnych cech do macierzy cech
        features.append(np.asarray(local_features).reshape(-1))

        # Dopisanie choroby do listy
        diseases.append(get_disease(file[:-4] + '.hea'))

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


def moving_average(signal, window=101):
  weights = np.repeat(1.0, window) / window
  smas = np.convolve(signal, weights, 'valid')
  signal = signal[(window - 1) / 2 : len(signal) - ((window - 1) / 2)]
  new_signal = []
  for index, x in enumerate(signal):
    new_signal.append(signal[index] - smas[index])
  return new_signal


def fourier_transform(signal):
  signalFFT = np.abs(np.fft.fft(signal))**2
  return signalFFT
  

def main():
  features, diseases = get_features()
  diseases = fuzzy_values(diseases)
  os.chdir("..")
  scipy.io.savemat('features.mat', {'features':features})
  scipy.io.savemat('diseases.mat', {'diseases':diseases})
  matplotlib.pyplot.show()


########################################################################
#######################  FEATURES VISUALIZATION  ####################### 
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
   
    matrix = scipy.io.loadmat(file)['val']
    print (np.shape(matrix))
    plot_number = plot_number + 1
    signal = matrix[0]
    signal = moving_average(signal, 5001)
    #signal = sklearn.preprocessing.normalize(signal)

    #matplotlib.pyplot.plot(range(0, len(signal)), signal)
    #matplotlib.pyplot.figure()
    #matplotlib.pyplot.plot(range(0, len(mean_avg)), mean_avg)
    #matplotlib.pyplot.show()
    
    # Pobranie cech sygnału
    wavelet = preprocessing.getFeature(signal, '0')
    wavelet = sklearn.preprocessing.normalize(wavelet)

    #print(np.shape(wavelet))
    #wavelet = sklearn.preprocessing.normalize(wavelet)
    #print(np.shape(wavelet))
    # Wartość średnia z otrzymanego wavelet'u
    meanV = mean(wavelet)
    varianceV = variance(wavelet, meanV)
    stddevV = stddev(varianceV)
    fftV = fourier_transform(meanV)
    

    features = []
    features.append(meanV)
    #features.append(varianceV)
    features.append(stddevV)
    features.append(fftV)
    features = np.asarray(features).reshape(-1)
    print(np.shape(meanV))
    print(np.shape(features))
      
    # Wyświetlenie otrzymanych danych na wykresach   
    fig = matplotlib.pyplot.figure()
    fig.subplots_adjust(hspace = 0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
      
      
    ax1 = matplotlib.pyplot.subplot2grid((3, 2), (0, 0), colspan=2)
    ax2 = matplotlib.pyplot.subplot2grid((3, 2), (1, 0), colspan=2)
    ax3 = matplotlib.pyplot.subplot2grid((3, 2), (2, 0))
    ax4 = matplotlib.pyplot.subplot2grid((3, 2), (2, 1))
    ax1.set_title(str(get_disease(file[:-4] + '.hea')))
    ax1.set_xlim([0, len(features) - 1])
    ax2.set_xlim([0, len(features) - 1])
      
    for y in wavelet:      
      ax1.plot(range(0,len(y)), y)
      ax3.specgram(y, Fs = 32, cmap=matplotlib.pyplot.cm.gist_rainbow)
    ax2.plot(range(0, len(features)), features)
    ax4.specgram(features, Fs = 32, cmap=matplotlib.pyplot.cm.gist_rainbow)

    os.chdir("..")
  matplotlib.pyplot.show()
########################################################################


if __name__ == '__main__':
  main()
  #features_presentation(3)
