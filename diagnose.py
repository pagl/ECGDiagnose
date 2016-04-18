import scipy.io
import numpy
import matplotlib.pyplot
import pywt
import cv2
import glob, os

def display_plot(y):
  matplotlib.pyplot.figure()
  x = [i for i in range(len(y))]
  matplotlib.pyplot.plot(x, y)
 

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


def main():
  features, diseases = get_features()
  print(diseases)
  os.chdir("..")
  scipy.io.savemat('features.mat', {'features':features})
  matplotlib.pyplot.show()


if __name__ == '__main__':
  main()
