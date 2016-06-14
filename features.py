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
import qrsdetect as qrs
from collections import Counter


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
  licznik = 0

  for dir in glob.glob("patient*"):
    print("Processing:\t" + str(dir))

    licznik = licznik + 1
    os.chdir(dir)

    for file in glob.glob("*.mat"):
      matrix = scipy.io.loadmat(file)['val']
      local_features = []
      correct = True
      info = {'name': "x", 'age': 50, 'sex': 'm', 'samplingrate' : 1000}

      # Sprawdzenie czy każdy pomiar ma 15 sygnałów
      if(len(matrix) != 15):
        print("Not enough data")
      else:
        matrix_avg = []
        signalsNo = 12

        # Wyrównanie sygnału     
        for signal in matrix[0:signalsNo]:
          matrix_avg.append(moving_average(signal, 5001))

        # Wykrycie odcinków QRS
        ecg = qrs.Ecg(matrix_avg[0], info) 
        ecg.qrsDetect(0)
        QRS_peaks = ecg.QRSpeaks

        # Pobranie cech sygnału
        try:
          wavelet = preprocessing.getFeature(matrix_avg[0:signalsNo], QRS_peaks)
          wavelet = wavelet[0:len(wavelet) - 2]

          # Wartość średnia z otrzymanego wavelet'u
          meanV = mean(wavelet)
          if (len(meanV) != 32 * signalsNo): 
            correct = False
            break

          # Wariancja z otrzymanego wavelet'u
          varianceV = variance(wavelet, meanV)

          # Odchylenie standardowe z otrzymanego wavelet'u
          stddevV = stddev(varianceV)

          # Dopisanie wyliczonych cech z wavelet'u
          local_features.append(meanV)
          local_features.append(stddevV)
      
          # Sprawdzenie czy powiodła się ekstrakcja cech
          if (correct == True):

            # Transformata Fouriera 
            for signal in matrix_avg:
              fftV = get_freq_amp(fourier_transform(signal))
              fftFreq, fftAmp = [], []
              for val in fftV:
                fftFreq.append(val[0])
                fftAmp.append(val[1])
              local_features.append(get_fourier_features(fftFreq, fftAmp))

            # Spłaszczenie i dopisane lokalnych cech do macierzy cech
            features.append(np.asarray(local_features).reshape(-1))
            diseases.append(get_disease(file[:-4] + '.hea'))

          counter = 0
          for y in wavelet:
            if (len(y) == (32 + 4) * signalsNo):   # 32 WAVELET + 4 AR MODEL
              counter += 1
              # Dopisanie cech do listy
              features.append(y)
            
              # Dopisanie klasy do listy
              diseases.append(get_disease(file[:-4] + '.hea'))

          print("Features:", str(np.shape(features)))
          print("Diseases:", str(np.shape(diseases)))

        except ValueError:
          print("#############################################")
          print("ValueError: maxlag < nobs; [SKIP]")
          print("#############################################")

    os.chdir("..")
  return (features, diseases)


def fuzzy_values(diseases):
  fuzzy_diseases = []
  final_diseases = []
  next_value = 1
  dict = {}
  for disease in diseases:
    if (not disease in dict):
      dict[disease] = next_value
      next_value += 1
      final_diseases.append(disease)
    fuzzy_diseases.append(dict[disease])
  return fuzzy_diseases,final_diseases


def moving_average(signal, window=101):
  weights = np.repeat(1.0, window) / window
  smas = np.convolve(signal, weights, 'valid')
  signal = signal[(window - 1) / 2 : len(signal) - ((window - 1) / 2)]
  new_signal = []
  for index, x in enumerate(signal):
    new_signal.append(signal[index] - smas[index])
  return new_signal


def fourier_transform(signal):
  signalFFT = np.abs(scipy.fftpack.fft(signal)) ** 2
  return signalFFT


def get_freq_amp(fftV):
  vals = []

  N = len(fftV)
  K = 1000

  fftV[0] = 0
  fftV = sklearn.preprocessing.normalize(fftV).reshape(-1)

  for i, val in enumerate(fftV):
    freq = i * K / N
    vals.append([freq, val])
  return vals


def get_fourier_features(fftFreq, fftAmp):
  features = []

  # Maksymalna częstotliwość
  max_freq = 50

  #Krok do pomiaru średniej częstotliwości
  step = 0.5
  idx = 0

  for freq in np.linspace(0, max_freq, (1 / step) * max_freq):
    sum = 0
    count = 0

    while (fftFreq[idx] <= freq):
      sum += fftAmp[idx]
      idx += 1
      count += 1

    features.append(sum / count)
  return features


def main():
  features, diseases = get_features()
  print(Counter(diseases))
  diseases,disease_name = fuzzy_values(diseases)
  print(disease_name)
  os.chdir("..")
  scipy.io.savemat('features_fft_12.mat', {'features':features})
  scipy.io.savemat('diseases_fft_12.mat', {'diseases':diseases})
  scipy.io.savemat('diseases_name_fft_12.mat',{'target_names':disease_name})
  matplotlib.pyplot.show()


#########################################################################
########################  FEATURE VISUALIZATION  ######################## 
#########################################################################

def features_presentation(number_of_plots):
  os.chdir("patients")
  dir_list = glob.glob("patient*")
  
  for i in range(number_of_plots):
    os.chdir(random.choice(dir_list))

    files_list = glob.glob("*.mat")
    file = random.choice(files_list)
   
    # Wczytanie 12 sygnałów
    matrix = scipy.io.loadmat(file)['val']

    # Wybór pierwszego sygnału (i)
    signal = matrix[0]

    # Wyliczenie średniej kroczącej
    mv_avg = moving_average(signal, 5001)

    # Pobranie cech transformaty falkowej i ich normalizacja
    wavelet = preprocessing.getFeature(signal, '0')
    wavelet = sklearn.preprocessing.normalize(wavelet)

    # Wartość średnia po wszystkich uderzeniach
    meanV = mean(wavelet)

    # Wariancja po wszystkich uderzeniach
    varianceV = variance(wavelet, meanV)

    # Odchylenie standardowe wszystkich uderzeń
    stddevV = stddev(varianceV)

    # Transformata Fourier'a na pełnym sygnale
    fftV = fourier_transform(signal)

    # Wyświetlenie otrzymanych danych na wykresach   
    fig = matplotlib.pyplot.figure(facecolor=(1,1,1))
    fig.subplots_adjust(hspace = 0.3, wspace=0.29, left = 0.03, right = 0.99, top = 0.96, bottom = 0.03)  
      
    ax1 = matplotlib.pyplot.subplot2grid((3, 4), (0, 0), colspan=2)
    ax2 = matplotlib.pyplot.subplot2grid((3, 4), (1, 0), colspan=2)
    ax3 = matplotlib.pyplot.subplot2grid((3, 4), (0, 2), colspan=2)
    ax4 = matplotlib.pyplot.subplot2grid((3, 4), (1, 2), colspan=2)
    ax5 = matplotlib.pyplot.subplot2grid((3, 4), (2, 2), colspan=2)
    ax6 = matplotlib.pyplot.subplot2grid((3, 4), (2, 0))
    ax7 = matplotlib.pyplot.subplot2grid((3, 4), (2, 1))
   
    ax1.set_axis_bgcolor((0.95, 0.95, 0.95))
    ax2.set_axis_bgcolor((0.95, 0.95, 0.95))
    ax3.set_axis_bgcolor((0.95, 0.95, 0.95))
    ax4.set_axis_bgcolor((0.95, 0.95, 0.95))
    ax5.set_axis_bgcolor((0.95, 0.95, 0.95))
    ax6.set_axis_bgcolor((0.95, 0.95, 0.95))
    ax7.set_axis_bgcolor((0.95, 0.95, 0.95))

    ax1.set_title(str(get_disease(file[:-4] + '.hea')))
    ax2.set_title("Moving average")
    ax3.set_title("Wavelet transform")
    ax4.set_title("Average")
    ax5.set_title("Standard Deviation")
    ax6.set_title("Fourier Transform")
    ax7.set_title("Spectrogram")

    ax1.set_xlim([0, len(signal) - 1])
    ax2.set_xlim([0, len(mv_avg) - 1])
    ax3.set_xlim([0, len(wavelet[0]) - 1])
    ax4.set_xlim([0, len(meanV) - 1])
    ax5.set_xlim([0, len(stddevV) - 1])
    ax6.set_xlim([0, len(fftV) - 1])

    ax1.plot(range(0, len(signal)), signal)
    ax2.plot(range(0, len(mv_avg)), mv_avg)  
    ax4.plot(range(0, len(meanV)), meanV)
    ax5.plot(range(0, len(stddevV)), stddevV)
    ax6.plot(range(0, len(fftV)), fftV)
    ax7.specgram(fftV, Fs = 1000, cmap=matplotlib.pyplot.cm.gist_rainbow)

    ax4.scatter(range(0, len(meanV)), meanV, marker='o')
    ax5.scatter(range(0, len(stddevV)), stddevV, marker='o')
      
    for y in wavelet:      
      ax3.plot(range(0,len(y)), y)
      ax3.scatter(range(0, len(y)), y, marker='o')

    os.chdir("..")
  matplotlib.pyplot.show()


if __name__ == '__main__':
  #main()   # Odkomentować kiedy chcemy wydobyć odpowiednie cechy
  features_presentation(1)
