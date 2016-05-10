
import pywt
import matplotlib.pyplot as plt
import matplotlib.lines as lines
import numpy as np
import scipy.io as spio
import scipy.signal as spsi
import qrsdetect as qrs
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
def extract_arrays(y,cutlist,times): ## roz
	y,wtaxD=makeWT(y,times)
	cutlist = [int(x / (2**(times))) for x in cutlist]
	arrays = []
	offset = 5 # ile sprawdzac w poblizu //powinno zalezec od WT_times ale kij z tym
	for index,x in enumerate(cutlist[1:-1]):
		x2 = x+ y[x-offset:x+offset].tolist().index(max(y[x-offset:x+offset]))
		x2=x2-offset
		i = x2 -20*(2**(3-times))
		i1 = x2 + 12*(2**(3-times))
		arrays.append(y[i:i1].copy())
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

def getFeature(data,info): ## da
	'''
	info: A dictionary object holding info
                    - 'name' = Patients name
                    - 'age' = Age in years
                    - 'sex' = 'm', 'f' or 'u' for unknown
                    - 'samplingrate' = Hz
	'''
	info = {'name': "Roman", 'age': 50, 'sex': 'm', 'samplingrate' : 1000} ## pote
	data2 = []	
	mV = 2000.0 ## tuta
	for i in data: ##zmi
		data2.append(i / mV)
	ecg = qrs.Ecg(data2,info) ##inicjalizacja ecg
	ecg.qrsDetect(0) ## 0 - numer wykresu, niby jak jest wiecej wykresow to mozna innego uzyc, ale blad wyskakuje
	arrays = extract_arrays(data2,ecg.QRSpeaks,3) ## wyciecie pojedynczych uderzen dzieki QRSpeaks
	return arrays

def main():
	data = spio.loadmat('s0015lrem.mat')
#	data = spio.loadmat('100m.mat')
	
	data = data['val']
	data = data[0]
	info = 0
	fig = plt.figure('ECG chart')
	ax1 = plt.subplot(1, 1 ,1)
	arrays = getFeature(data,info)
	for i in arrays:
		ax1.plot(range(0,len(i)),i)
	'''
	data2 = []
	for i in data:
		data2.append(i/2000.0)
	'''
	'''
	infodict`: A dictionary object holding info
                    - 'name' = Patients name
                    - 'age' = Age in years
                    - 'sex' = 'm', 'f' or 'u' for unknown
                    - 'samplingrate' = Hz
	'''
	'''
	info = {'name': "roman", 'age': 50, 'sex': 'm', 'samplingrate' : 360}
	ecg = qrs.Ecg(data2,info)
	#ecg.qrsDetect(0)
	#print(data, len(data))
	fig = plt.figure('ECG chart')
	(qrs_coords,wtaxD) = preprocess(data2.copy())
	ax1 = plt.subplot(1, 1 ,1)
	ecg.qrsDetect(0)
	qrs_coords = ecg.QRSpeaks
	#ax1.plot(range(0,len(data)),data)
	#print(qrs_coords,len(qrs_coords))
	
	data2 = extract_arrays(data2,qrs_coords,3)
	for i in data2:
		ax1.plot(range(0,len(i)),i)
	
	## rysowanie linii
	line = []
	#print(qrs_coords)
	print("liczba wykrytych udzerzen: ")
	print(len(qrs_coords))
	for i in qrs_coords:
		line.append((i,2500))
		line.append((i,-1))
	(xs, ys) = zip(*line)
	for j in range(0,len(xs)-1,2):
		ax1.add_line(lines.Line2D(xs[j:j+2], ys[j:j+2], linewidth=1, color = 'red'))
	'''
	plt.show()

if __name__ == '__main__':
    main()
