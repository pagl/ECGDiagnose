#-*- coding: utf-8 -*-

import math
import scipy.io
import scipy.signal
import numpy
import matplotlib.pyplot
import pywt
import glob, os, subprocess
import preprocessing
import numpy as np
import random
import sklearn.preprocessing
import qrsdetect as qrs



#------ GLOBAL VARIABLES ------# 

FEATURES_PRESENTATION = False  # Present graphically features [Set to False, to extract features!]
NUMBER_OF_EXAMPLES    = 2      # Number of examples to present graphically


GET_WAVELET = True             # Extract wavelet transform coefficients
GET_MEAN    = True             # Extract mean average from wavelet coefficients
GET_STDDEV  = True             # Extract standard deviation from wavelet coefficients
GET_FFT     = True             # Extract FFT spectrum
GET_AR      = True             # Extract auto regressive model from signal

NUMBER_OF_LEADS = 1            # [1 - 12] Each patient have 12 different leads

HEARTBEATS_SEPARATELY = False  # Represent each heartbeat as different training example 

#------------------------------# 



'''   Finds name of a disease in header file
      @param file Header file name
      @return Disease name
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



'''   Counts mean average from wavelet transform coefficient values
      @param signal Data received from wavelet transform
      @return List containing mean average from wavelet transform coefficient values
'''
def mean(signal):
  mean = []
  for j in range(len(signal[0])):
    sum = 0
    for i in range(len(signal)):
      sum += signal[i][j]
    mean.append(sum / len(signal))
  return mean



'''   Counts variance from wavelet transform coefficient values
      @param signal Data received from wavelet transform
      @param mean Mean average from wavelet transform coefficient values
      @return List containing variance from wavelet transform coefficient values
'''
def variance(signal, mean):
  variance = []
  for j in range(len(signal[0])):
    sum = 0
    for i in range(len(signal)):
      sum += pow(signal[i][j] - mean[j], 2)
    variance.append(sum / len(signal))
  return variance



'''   Counts standard deviation from wavelet transform coefficient values
      @param variance Variance value from wavelet transform coefficient values 
      @return List containing standard deviation of wavelet transform coefficient values
'''
def stddev(variance):
  stddev = []
  for i in range(len(variance)):
    stddev.append(math.sqrt(float(variance[i])))
  return stddev



'''   Change diseases to numeric values
      @param diseases Array containing diseases
      @return List of diseases as numeric values, and their string representation
'''
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
  return fuzzy_diseases, final_diseases



'''   Does moving average on given signal, and substract current value from 
      average value, to align signal in one point.
      @param signal Original signal
      @param window Value how wide the window should be
      @return Aligned signal
'''
def moving_average(signal, window=101):
  weights = np.repeat(1.0, window) / window
  smas = np.convolve(signal, weights, 'valid')
  signal = signal[(window - 1) / 2 : len(signal) - ((window - 1) / 2)]
  new_signal = []
  for index, x in enumerate(signal):
    new_signal.append(signal[index] - smas[index])
  return new_signal



'''   Power FFT spectrum
      @param signal Original signal
      @return Power FFT spectrum
'''
def fourier_transform(signal):
  signalFFT = np.abs(scipy.fftpack.fft(signal)) ** 2
  return signalFFT



'''   Counts amplitude for each frequency from given fft spectrum
      @param fftV FFT spectrum
      @return List of frequencies and corresponding amplitudes
'''
def get_freq_amp(fftV):
  vals = []

  N = len(fftV)
  K = 1000

  fftV[0] = 0
  fftV = sklearn.preprocessing.normalize(fftV.reshape(-1, 1)).flatten()

  for i, val in enumerate(fftV):
    freq = i * K / N
    vals.append([freq, val])
  return vals



'''   Extract set of features from given amplitude and frequency list
      @param fftFreq List of frequencies
      @param fftAmp List of amplitudes
      @return List of features from fft spectrum
'''
def get_fourier_features(fftFreq, fftAmp):
  features = []

  # Maximum frequency
  max_freq = 50

  # Frequency step [Number of features = max_freq / step]
  step = 0.2
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



'''   Extract features from ECG signal
       - Align signal
       - Detect QRS
       - Detect heartbeats (Pan, Tompkin's Algorithm)
       - Wavelet transform + AR Model
       - Mean average
       - Standard deviation
       - FFT spectrum
'''
def get_features():
  features = []
  diseases = []
  os.chdir("patients")
  processed = 1
  total = int(os.popen("find . -name *.mat | wc -l").read())

  for dir in glob.glob("patient*"):
    os.chdir(dir)
    for file in glob.glob("*.mat"):
      print("Processing: %s   [%d / %d]" % (str(file), processed, total))
      matrix = scipy.io.loadmat(file)['val']
      local_features = []
      correct = True
      info = {'name': "data", 'age': 1, 'sex': 'm', 'samplingrate' : 1000}

      if(len(matrix) != 15):
        print("Patient's data is not complete")
      else:
        matrix_avg = []

        # Align the signal   
        for signal in matrix[0:NUMBER_OF_LEADS]:
          matrix_avg.append(moving_average(signal, 5001))

        try:
          wavelet = []	

          # Get wavelet transform coefficients
          if (GET_MEAN or GET_STDDEV or GET_AR or GET_WAVELETS):
          
            # Detect QRS
            ecg = qrs.Ecg(matrix_avg[0], info) 
            ecg.qrsDetect(0)
            QRS_peaks = ecg.QRSpeaks

            if (GET_AR):
              wavelet = preprocessing.getFeatureWithArWithQrs(matrix_avg[0:NUMBER_OF_LEADS], QRS_peaks)
            else:
              wavelet = preprocessing.getFeatureWithQrs(matrix_avg[0:NUMBER_OF_LEADS], QRS_peaks)
            wavelet = wavelet[0:len(wavelet) - 2]

          # Get mean average from wavelet transform coefficients
          if (GET_MEAN):
            meanV = mean(wavelet)
            if (len(meanV) == 32 * NUMBER_OF_LEADS): 
              local_features.append(meanV)

          # Get standard deviation from wavelet transform coefficients
          if (GET_STDDEV):
            meanV = mean(wavelet)
            if (len(meanV) == 32 * NUMBER_OF_LEADS):
              varianceV = variance(wavelet, meanV)
              stddevV = stddev(varianceV)
              local_features.append(stddevV)
          
          if (GET_FFT):
            # Get Fast Fourier Transform spectrum
            for signal in matrix_avg:
              fftV = get_freq_amp(fourier_transform(signal))
              fftFreq, fftAmp = [], []
              for val in fftV:
                fftFreq.append(val[0])
                fftAmp.append(val[1])
              local_features.append(get_fourier_features(fftFreq, fftAmp))

          if (HEARTBEATS_SEPARATELY):
            for y in wavelet:
              if (GET_AR):
                if (len(y) == (32 + 4) * NUMBER_OF_LEADS):  
                  features.append(y)
                  diseases.append(get_disease(file[:-4] + '.hea'))
              elif (GET_WAVELET):
                if (len(y) == 32 * NUMBER_OF_LEADS):
                  features.append(y)
                  diseases.append(get_disease(file[:-4] + '.hea'))
          else:
            features.append(np.asarray(local_features).flatten())
            diseases.append(get_disease(file[:-4] + '.hea'))
     
          processed += 1
        except ValueError:
          processed += 1
          print("#############################################")
          print("ValueError: maxlag < nobs; [SKIP]")
          print("#############################################")

    os.chdir("..")
  return (features, diseases)



# ------ FEATURES PRESENTATION ------ #

def features_presentation (number_of_plots):
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
    wavelet = preprocessing.getFeature(mv_avg, '0')
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


def main():
  if (FEATURES_PRESENTATION):
    features_presentation(NUMBER_OF_EXAMPLES)
  else:
    features, diseases = get_features()
    diseases,disease_name = fuzzy_values(diseases)
    os.chdir("..")
    scipy.io.savemat('features.mat', {'features':features})
    scipy.io.savemat('diseases.mat', {'diseases':diseases})
    scipy.io.savemat('diseases_name.mat',{'target_names':disease_name})
    matplotlib.pyplot.show()


if __name__ == '__main__':
  main()
