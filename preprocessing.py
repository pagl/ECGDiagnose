
import pywt
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import scipy.io as spio
import scipy.signal as spsi
import qrsdetect as qrs
import statsmodels.tsa.ar_model as ar_model

WT_times = 5
mV = 1.0
def preprocess(data): ## data - pojedynczy wykres z spio.loadmat; jak jest wiecej wykresow to podac data[i] do funkcji
	array = []
	x=[]
	y=[]
	for index, cell in enumerate(data):
		x.append(index)
		y.append(float(cell)/mV)
	(wtaxA,wtaxD) = makeWT(y,WT_times)
	#print(min(wtaxD))
	qrsArray = find_QRS(wtaxD,len(wtaxD)) ## dzialalo na wtaxD ale sie zepsulo
	qrsArray = [x * (2**WT_times) for x in qrsArray]
	#array = extract_arrays(wtaxD,qrsArray,WT_times)
	return qrsArray,wtaxA
def makeWT(y,times): ## robi pare razy WT na approx.
	(wtaxA, wtaxD) = pywt.dwt((y),'db1')
	if(times > 1):
		for i in range(times-1):
			(wtaxA,wtaxD) = pywt.dwt((wtaxA),'db1')
	ymin = min(wtaxD)
	for index,x in enumerate(wtaxD):
		wtaxD[index-1]=wtaxD[index-1]-ymin
	ymin = min(wtaxA)
	for index,x in enumerate(wtaxA):
		wtaxA[index-1]=wtaxA[index-1] - ymin
	
	return wtaxA,wtaxD


def armodel(y,cutlist):
	array = []
	result = []
	offset = 20
	for index,x in enumerate(cutlist[1:-1]):
		x = int(x)
		x2 = x + y[x-offset:x+offset].index(max(y[x-offset:x+offset]))
		x2 = x2 - offset
		i = x2 - 200
		i1 = x2 + 100
		array.append(y[i:i1].copy())
	
	for x in array:
		ar = ar_model.AR(x)
		arfit = ar.fit(maxlag=3,method='cmle',disp = 0)
		result.append(arfit)
	return result
		

def extract_arrays(y,cutlist,times): 	
	arfit = armodel(y,cutlist)
	y,wtaxD=makeWT(y,times)	
	cutlist = [int(x / (2**(times))) for x in cutlist]
	
	arrays = []
	offset = 5 
	for index,x in enumerate(cutlist[1:-1]):
		x2 = x+ y[x-offset:x+offset].tolist().index(max(y[x-offset:x+offset]))
		x2=x2-offset
		i = x2 -20*(2**(3-times))
		i1 = x2 + 12*(2**(3-times))
		arrays.append(y[i:i1].copy())
	return arrays


def extract_arrays_with_arfit(y, cutlist, times):
	arfit = armodel(y,cutlist)
	y,wtaxD=makeWT(y,times)	
	cutlist = [int(x / (2**(times))) for x in cutlist]
	
	arrays = []
	arrays_with_arfit = []
	offset = 5 
	for index,x in enumerate(cutlist[1:-1]):
		x2 = x+ y[x-offset:x+offset].tolist().index(max(y[x-offset:x+offset]))
		x2=x2-offset
		i = x2 -20*(2**(3-times))
		i1 = x2 + 12*(2**(3-times))
		arrays.append(y[i:i1].copy())
		
		arrays_with_arfit.append(np.concatenate([arrays[index],arfit[index].params]))
	return arrays_with_arfit
	return arrays


def find_QRS(y,n):
         ymax = max(y)
         divide = WT_times
         if(WT_times < 1):
              divide = 0
         test_time = int(100/(2**divide))
         #print(test_time)
         iter = list(range(0,n+1,test_time)) ## co ile sprawdzany jest potencjalne R
         ylen = n
         ###
         #plt.plot(range(0,len(y)),y)
         maxlist=[]
         xlist=[]
         v2=[]
         final_list = []
         y2 = []
         for i in y:
              y2.append(i)
         for i in range(0,len(iter)-1):
              maxlist.append(max(y2[iter[i]:iter[i+1]]))
              v = y2[iter[i]:iter[i+1]].index(max(y2[iter[i]:iter[i+1]]))
              v2.append( int(v) + int(iter[i]))
         vmax = max(y)
         for licz, i in enumerate(maxlist):
              if(vmax*0.7 < i):
                      final_list.append(v2[licz])
         for index in range(0,len(final_list)-1):
              if(final_list[index+1]-final_list[index] < float((test_time/2))):
                      i=1
         #print(final_list)
         last = final_list[len(final_list)-1]
         final_list[:] = [x for index,x in enumerate(final_list[:-1],start = 1) if not determine(final_list[index-1],final_list[index],test_time)]
         if(last-final_list[len(final_list)-1] > test_time):
              final_list.append(last)
        # print(final_list,last)
         return final_list
         
         
def determine(x1,x2,time):
	#print(x2-x1,time, x1, x2)
	if((x2-x1)<(time)):
		return True
	else:
		return False

def moving_average(signal, window=101):
  weights = np.repeat(1.0, window) / window
  smas = np.convolve(signal, weights, 'valid')
  signal = signal[(window - 1) / 2 : len(signal) - ((window - 1) / 2)]
  new_signal = []
  for index, x in enumerate(signal):
    new_signal.append(signal[index] - smas[index])
  return new_signal


def getFeatureWithQrs(data, qrs_peaks): 
	mV = 2000.0
	multiple_arrays = []
	for i in qrs_peaks:
		multiple_arrays.append([])
	for index, signal in enumerate(data):
		data2 = []
		for i in signal:
			data2.append(i / mV)
		arrays = extract_arrays(data2, qrs_peaks, 3)
		for index,array in enumerate(arrays):
			multiple_arrays[index].extend(array)
	return multiple_arrays	
	return arrays

def getFeatureWithArWithQrs (data, qrs_peaks): 
	mV = 2000.0
	multiple_arrays = []
	for i in qrs_peaks:
		multiple_arrays.append([])
	for index, signal in enumerate(data):
		data2 = []
		for i in signal:
			data2.append(i / mV)
		arrays = extract_arrays_with_arfit(data2, qrs_peaks, 3)
		for index,array in enumerate(arrays):
			multiple_arrays[index].extend(array)
	return multiple_arrays	
	return arrays


def getFeature(data,info):
	info = {'name': "Roman", 'age': 50, 'sex': 'm', 'samplingrate' : 1000}
	data2 = []	
	mV = 2000.0 
	for i in data:
		data2.append(i / mV)
	ecg = qrs.Ecg(data2,info) 
	ecg.qrsDetect(0)
	arrays = extract_arrays(data2, ecg.QRSpeaks, 3) 
	return arrays
