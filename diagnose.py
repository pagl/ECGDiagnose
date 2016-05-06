import scipy.io
import scipy.signal
import numpy
import matplotlib.pyplot
import pywt
import cv2
import glob, os

def display_plot(y):
  fig, (ax1, ax2) = matplotlib.pyplot.subplots(2, 1)
  fig.subplots_adjust(hspace = 0.1, left = 0.05, right = 0.95, top = 0.95, bottom = 0.05)
  x = [i for i in range(len(y))]
  ax1.plot(x, y)
  ax2.specgram(y, Fs = 1000)
 

def flatten(matrix):
  i = 0
  flat = []
  for row in matrix:
    for el in row:
      flat.insert(i, el)
      i += 1
  return flat


def wavelet_transform(matrix):
  wavelet = []
  for row in matrix:
    cA, cD = pywt.dwt(row, 'db1')
    wavelet.append(cA)
    wavelet.append(cD)
  return flatten(wavelet)
    

def get_disease(file):
  disease_header = "Reason for admission:"
  with open(file) as f:
    lines = f.readlines()
  
  for line in lines:
    if (line.find(disease_header) != -1):
      disease = line[line.find(disease_header) + len(disease_header) + 1 : line.find("\\") - 1]
      break
  return disease


def get_features():
  features = []
  diseases = []
  os.chdir("patients")

  for file in glob.glob("*.mat"):
    matrix = scipy.io.loadmat(file)
    wavelet = wavelet_transform(matrix['val']) 
    features.append(wavelet)
    diseases.append(get_disease(file[:-4] + '.hea'))
    display_plot(wavelet)
  return (features, diseases)


def write_diseases(diseases):
  next_value = 1
  dict = {}
  f = open('diseases', 'w')
  for disease in diseases:
    if (not dict.has_key(disease)):
      dict[disease] = next_value
      next_value += 1
    f.write(str(dict[disease]))
    f.write('\n')
  f.close()


def main():
  features, diseases = get_features()
  os.chdir("..")
  scipy.io.savemat('features.mat', {'features':features})
  write_diseases(diseases)
  matplotlib.pyplot.show()


if __name__ == '__main__':
  main()
