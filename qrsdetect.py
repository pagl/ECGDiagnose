#!/usr/bin/env python

# QRS detection in ECGs using a modified Pan Tomkins method.
# Modifications based on OSEA documentation - Patrick Hamilton

from __future__ import division
import scipy
import scipy.signal
import scipy.io as spio
import pylab
import datetime

import sys
sys.path.append('/data/dropbox/programming')
#import ecgtools.basic_tools

class QrsDetectionError(Exception):
    """Raise error related to qrs detection"""
    def __init__(self, value):
        self.parameter = value

    def __str__(self):
        return repr(self.parameter)

class Ecg():
    """The ECG data
    """
    def __init__(self, ecgdata, infodict):
        """
        Arguments:
        - 'ecgdata' : The ecgdata as an array - points x leads
                      or a vector
        - `infodict`: A dictionary object holding info
                    - 'name' = Patients name
                    - 'age' = Age in years
                    - 'sex' = 'm', 'f' or 'u' for unknown
                    - 'samplingrate' = Hz
                    
        """
        self.infodict = infodict
        self._readinfo()
        self.data = scipy.array(ecgdata)

        # convert vector to column array
        if len(self.data.shape) == 1:
            self.data = scipy.array([self.data]).transpose()

        self.points, self.leads = self.data.shape
        if len(self.data.shape) > 1 and self.leads > self.points:
            raise QrsDetectionError("Data has more columns than rows")

    def _warning(self, msg):
        """Handle warning messages"""
        # TODO: verbosity to determine how the warning will be treated.
        #print(msg)

    def _readinfo(self):
         """Read the info and fill missing values with defaults
         """
         # defaults 
         self.name =  ''
         self.age =  0
         self.sex = 'u'
         self.samplingrate = 360 # TODO!!!

         for variable, key in [
             (self.name, 'name'),
             (self.age, 'age'),
             (self.sex, 'sex'),
             (self.samplingrate, 'samplingrate')]:
             try:
                 variable = self.infodict[key]
             except KeyError:
                 self._warning("Info does not contain %s, replacing with %s"
                                      %(key, variable))

    def qrsDetect(self, qrslead=0):
         """Detect QRS onsets using modified PT algorithm
         """
         # If ecg is a vector, it will be used for qrs detection.
         # If it is a matrix, use qrslead (default 0)
         if len(self.data.shape) == 1:
             self.raw_ecg = self.data
         else:
             self.raw_ecg = self.data[:,qrslead]
         
         self.filtered_ecg = self.bpfilter(self.raw_ecg)
         self.diff_ecg  = scipy.diff(self.filtered_ecg)
         self.sq_ecg = abs(self.diff_ecg)
         self.int_ecg = self.mw_integrate(self.sq_ecg)
         
         # Construct buffers with last 8 values 
         self._initializeBuffers(self.int_ecg)
         
         peaks = self.peakDetect(self.int_ecg)
         self.checkPeaks(peaks, self.int_ecg)

         # compensate for delay during integration
         self.QRSpeaks = self.QRSpeaks  - 40 * (self.samplingrate / 1000)
         
         #print ("length of qrs peaks and ecg", len(self.QRSpeaks), len(self.raw_ecg))
         #print(self.QRSpeaks)
         return self.QRSpeaks

    def qrs_detect_multiple_leads(self, leads=[]):
        """Use multiple leads for qrs detection.
        Leads to use may be given as list of lead indices.
        Default is to use all leads"""
        # leads not specified, switch to all leads
        if leads == []:
            leads = range(self.leads)

        # qrs detection for each lead
        qrspeaks = []
        for lead in leads:
            qrspeaks.append(self.qrsDetect(lead))

        # DEBUG
        #print ("length of qrs in different channels")
        #print ([len(x) for x in qrspeaks])

        # zero pad detections to match lengths
        maxlength = max([len(qrspeak_lead) for qrspeak_lead in
                         qrspeaks])
        for lead in range(len(qrspeaks)):
            qrspeaks[lead] = self._zeropad(qrspeaks[lead], maxlength)

        #DEBUG
        #print ("max length ", maxlength)
        #print ([len(x) for x in qrspeaks])
        
        qrspeaks_array = scipy.array(qrspeaks).transpose()
        self.QRSpeaks = self.multilead_peak_match(qrspeaks_array)
        return self.QRSpeaks

    def _zeropad(self, shortvec, l):
        """Pad the vector shortvec with terminal zeros to length l"""
        return scipy.hstack((shortvec, scipy.zeros((l - len(shortvec)), dtype='int')))
        
    def write_ann(self, annfile):
        """Write an annotation file for the QRS onsets in a format
        that is usable with wrann"""
        fi = open(annfile, 'w')
        for qrspeak in self.QRSpeaks:
            fi.write('%s %s %s %s %s %s\n' %(self._sample_to_time(qrspeak), qrspeak, 'N', 0, 0, 0))
        fi.close()

    def _sample_to_time(self, sample):
        """convert from sample number to a string representing
        time in a format required for the annotation file.
        This is in the form (hh):mm:ss.sss"""
        time_ms = int(sample*1000 / self.samplingrate)
        hr, min, sec, ms = time_ms//3600000 % 24, time_ms//60000 % 60, \
                           time_ms//1000 % 60, time_ms % 1000
        timeobj = datetime.time(hr, min, sec, ms*1000) # last val is microsecs
        return timeobj.isoformat()[:-3] # back to ms
         
    def visualize_qrs_detection(self, savefilename = False):
        """Plot the ecg at various steps of processing for qrs detection.
        Will not plot more than 10 seconds of data.
        If filename is input, image will be saved"""
        ecglength = len(self.raw_ecg)
        ten_seconds = 200 * self.samplingrate
        
        if ecglength >= ten_seconds:
            segmentend = ten_seconds
        elif ecglength < ten_seconds:
            segmentend = ecglength

        segmentQRSpeaks = [peak for peak in self.QRSpeaks if peak < segmentend]

        pylab.figure()
        pylab.subplot(611)
        pylab.plot(self.raw_ecg[:segmentend])
        pylab.ylabel('Raw ECG', rotation='horizontal')
        pylab.subplot(612)
        pylab.plot(self.filtered_ecg[:segmentend])
        pylab.ylabel('Filtered ECG',rotation='horizontal')
        pylab.subplot(613)
        pylab.plot(self.diff_ecg[:segmentend])
        pylab.ylabel('Differential',rotation='horizontal')
        pylab.subplot(614)
        pylab.plot(self.sq_ecg[:segmentend])
        pylab.ylabel('Squared differential',rotation='horizontal')
        pylab.subplot(615)
        pylab.plot(self.int_ecg[:segmentend])
        pylab.ylabel('Integrated', rotation='horizontal')
        pylab.subplot(616)
        pylab.hold(True)
        pylab.plot(self.raw_ecg[:segmentend])
        pylab.plot(segmentQRSpeaks, self.raw_ecg[segmentQRSpeaks], 'xr')
        pylab.hold(False)
        pylab.ylabel('QRS peaks', rotation='horizontal')

        if savefilename:
            pylab.savefig(savefilename)
        else:
            pylab.show()
        
    def _initializeBuffers(self, ecg):
        """Initialize the 8 beats buffers using values
        from the first 8 one second intervals        
        """
        srate = self.samplingrate
        # signal peaks are peaks in the 8 segments
        self.signal_peak_buffer = [max(ecg[start*srate:start*srate+srate])
                                                  for start in range(8)]
        self.noise_peak_buffer = [0] * 8
        self.rr_buffer = [1] * 8
        self._updateThreshold()
        
    def _updateThreshold(self):
        """Calculate threshold based on amplitudes of last
        8 signal and noise peaks"""
        noise = scipy.mean(self.noise_peak_buffer)
        signal = scipy.mean(self.signal_peak_buffer)
        self.threshold = noise + 0.3125 * (signal - noise)
        self.threshold = self.threshold*2

    def peakDetect(self, ecg):
        """Determine all points that form a local maximum
        """
        # list all local maxima
        all_peaks = [i for i in range(1,len(ecg)-1)
                     if ecg[i-1] < ecg[i] > ecg[i+1]]
        peak_amplitudes = [ecg[peak] for peak in all_peaks]

        final_peaks = []

        # restrict to peaks that are larger than anything else 200 ms
        # on either side
        minimumRR = self.samplingrate * 0.2

        # start with first peak
        peak_candidate_index = all_peaks[0]
        peak_candidate_amplitude = peak_amplitudes[0]

        # test successively against other peaks
        for peak_index, peak_amplitude in zip(all_peaks, peak_amplitudes):
            # if new peak is less than minimumRR away and is larger,
            # it becomes candidate
            if peak_index - peak_candidate_index <= minimumRR and\
                                  peak_amplitude > peak_candidate_amplitude:
                peak_candidate_index = peak_index
                peak_candidate_amplitude = peak_amplitude

            # if new peak is more than 200 ms away, candidate is added to
            # final peak and new peak becomes candidate
            elif peak_index - peak_candidate_index > minimumRR:
                final_peaks.append(peak_candidate_index)
                peak_candidate_index = peak_index
                peak_candidate_amplitude = peak_amplitude

            else:
                pass

        return final_peaks


    def checkPeaks(self, peaks, ecg):
        """Check the given peaks one by one according to
        thresholds that are constantly updated"""
        amplitudes = [ecg[peak] for peak in peaks]
        self.QRSpeaks = []
        
        for index, peak in enumerate(peaks):
            amplitude = amplitudes[index]
            # accept if larger than threshold and slope in raw signal
            # is +-30% of previous slopes
            #print(index,amplitude,self.threshold)
            if amplitude > self.threshold:
                self.acceptasQRS(peak, amplitude)

            # acccept as qrs if higher than half threshold,
            # but is 360 ms after last qrs and next peak
            # is more than 1.5 rr intervals away
            # just abandon it if there is no peak before
            # or after
            elif amplitude > self.threshold/2 and \
                     len(self.QRSpeaks) > 0 and \
                     len(peaks) > index+1:
                meanrr = scipy.mean(self.rr_buffer)
                lastQRSms = (peak - self.QRSpeaks[-1]) * (
                                               1000 / self.samplingrate)
                lastQRS_to_next_peak = peaks[index+1] - self.QRSpeaks[-1]

                if lastQRSms > 360 and lastQRS_to_next_peak > 1.5 * meanrr:
                    self.acceptasQRS(peak, amplitude)

                else:
                    self.acceptasNoise(peak, amplitude)

            # if not either of these it is noise
            else:
                self.acceptasNoise(peak, amplitude)

        self.QRSpeaks = scipy.array(self.QRSpeaks)
        return


    def acceptasQRS(self, peak, amplitude):
        self.QRSpeaks.append(peak)

        self.signal_peak_buffer.pop(0)
        self.signal_peak_buffer.append(amplitude)

        if len(self.QRSpeaks) > 1:
            self.rr_buffer.pop(0)
            self.rr_buffer.append(self.QRSpeaks[-1] - self.QRSpeaks[-2])
        self._updateThreshold()

    def acceptasNoise(self, peak, amplitude):
        self.noise_peak_buffer.pop(0)
        self.noise_peak_buffer.append(amplitude)
            
    def mw_integrate(self, ecg):
        """
        Integrate the ECG signal over a defined
        time period. 
        """
        # window of 80 ms - better than using a wider window
        window_length = int(80 * (self.samplingrate / 1000))

        int_ecg = scipy.zeros_like(ecg)
        cs = ecg.cumsum()
        
        int_ecg[window_length:] = (cs[window_length:] -
                                   cs[:-window_length]) / window_length
        int_ecg[:window_length] = cs[:window_length] / scipy.arange(
                                                   1, window_length + 1)
        
        return int_ecg

    def bpfilter(self, ecg):
         """Bandpass filter the ECG with a bandpass setting of
         5 to 15 Hz"""
         # relatively basic implementation for now
         # TODO:
         # Explore - different stopbands, filtfilt, padding
         Nyq = self.samplingrate / 2

         wn = [5/ Nyq, 15 / Nyq]
         b,a = scipy.signal.butter(2, wn, btype = 'bandpass')
         return scipy.signal.filtfilt(b,a,ecg)
       #  return ecgtools.basic_tools.filtfilt(b,a,ecg)


    def multilead_peak_match(self, peaks):
        """Reconcile QRS detections from multiple leads.
        peaks is a matrix of peak_times x leads.
        If the number of rows is different,
        pad shorter series with zeros at end"""
        ms90 = 90 * self.samplingrate / 1000
        Npeaks, Nleads = peaks.shape
        current_peak = 0
        final_peaks = []

        while current_peak < len(peaks):
            all_values = peaks[current_peak, :]
            outer = all_values.max()
            outerlead = all_values.argmax()
            inner = all_values.min()
            innerlead = all_values.argmin()

            #
            near_inner = sum(all_values < inner + ms90)
            near_outer = sum(all_values > outer - ms90)

            #all are within 90 ms
            if near_inner == near_outer == Nleads:
                final_peaks.append(int(scipy.median(all_values)))
                current_peak += 1

            # max is wrong
            elif near_inner > near_outer:
                peaks[current_peak+1:Npeaks, outerlead] = peaks[current_peak:Npeaks-1, outerlead]
                peaks[current_peak, outerlead] = scipy.median(all_values)
                # do not change current peak now

            # min is wrong
            elif near_inner <= near_outer:
                peaks[current_peak:Npeaks-1, innerlead] = peaks[current_peak+1:Npeaks, innerlead]
                peaks[-1, innerlead] = 0

        return final_peaks


def main():
    """The main program
    """
 
    import pylab
    #import ecgtools.io
 

  #  data, info = ecgtools.io.readPrucka('/data/dropbox/programming/ecgtools/samples/prucka_test.txt')
    data = spio.loadmat('116m.mat')
    data = data['val']
    data = data[0]
    data2 = []
    for i in data:
        data2.append(i/2000.0)
    '''
	infodict`: A dictionary object holding info
                    - 'name' = Patients name
                    - 'age' = Age in years
                    - 'sex' = 'm', 'f' or 'u' for unknown
                    - 'samplingrate' = Hz
	'''
    info = {'name': "roman", 'age': 50, 'sex': 'm', 'samplingrate' : 1000}
    ecg = Ecg(data2, info)

   # ecg.qrs_detect_multiple_leads([7,8])
    ecg.qrsDetect(0)

    # plot the detection
    ecg.visualize_qrs_detection()

   # ecg.write_ann('/data/tmp/testann.txt')
      


def test():
    """Test multilead peak match"""
    testpeaks = scipy.array([[1,501,1000,1490,2000, 0],
                        [0,320, 530, 1100, 1530, 2010],
                        [10, 990, 1400, 1990, 0, 0]])
    testpeaks =  testpeaks.transpose()
    newpeaks = multilead_peak_match(testpeaks)
    #print(newpeaks)

    
      
if __name__ == "__main__":
    main()
