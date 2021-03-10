from serial import Serial
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pandas as pd
import seaborn as sns
import scipy.signal as signal
#import scipy.io
import biosppy
#from drawnow import *

start_time = time.time()
arduino_array = []
ekg_t =[]

arduino_data = Serial('com3', 9600)
fs = 100


def create_plot():
    plt.cla()
    raw_signal = np.array(arduino_array)
    signal_len = len(raw_signal)

    fs = 100
    t = np.array(ekg_t)/1000

    #detrend
    detrend_signal = scipy.signal.detrend(raw_signal, axis=-1, type='linear', bp=0)
    # Signal Proccesing
    cutoff = 30 # frekuensi cutoff
    nyq = 0.5 * fs
    wn = cutoff/(nyq)
    b, a = signal.butter(4,0.5, btype = 'low', analog=False)
    filtered_signal = signal.lfilter(b,a,detrend_signal)
    smoothed_signal = signal.cspline1d(filtered_signal,lamb=0.3)


    #rpeak
    t = np.arange(0,len(raw_signal))/100
    multiplier = 3
    tres = max(smoothed_signal) * multiplier

    sm_rpeak = biosppy.signals.ecg.ssf_segmenter(signal=smoothed_signal, sampling_rate=fs, threshold=tres, before=0.03, after=0.01)
    sm_rpeak = sm_rpeak['rpeaks']/fs
    r_index3 = []
    for k in sm_rpeak:
        r_index3.append(np.where(t == k))
        
    if r_index3 == []:
        plt.plot(t, smoothed_signal)
    else:
        r_index3 = np.array(r_index3)
        plt.plot(t, smoothed_signal)
        plt.scatter(t[r_index3],smoothed_signal[r_index3],c='r')
        plt.xlabel('Time(s)')


    #jumlah R
    poin_rpeak = sm_rpeak
    RR_interval = np.diff(poin_rpeak)
    #  Heart rate (beat per minute) 60s / RR_interval
    heart_rate = np.mean(60/(RR_interval))
    #  Standar deviasi dari RR interval yang sudah difilter
    SDNN = np.std(RR_interval)
    #  Persentase dari selisih RR interval yang lebih dari 50ms (0.05s).
    NN50 = np.sum(np.abs(np.diff(RR_interval))>.05)
    pNN50 = (NN50/len(RR_interval)) * 100
    RMSSD = (np.mean(np.diff(RR_interval) ** 2))**0.5

    plt.annotate(f'Heart Rate: {heart_rate:.2f}bpm', xy=(0,0.95 ), xycoords='axes fraction')
    plt.annotate(f'SDNN: {SDNN:.2f}ms', xy=(0, 0.90 ), xycoords='axes fraction')
    plt.annotate(f'pNN50: {pNN50:.2f}%', xy=(0,0.85 ), xycoords='axes fraction')
    plt.annotate(f'RMSSD: {RMSSD:.2f}ms', xy=(0, 0.80 ), xycoords='axes fraction')
    plt.ylim(-250,500)
    plt.pause(0.1)
    
plt.figure()
monitoring = True
while monitoring:
    while (arduino_data.inWaiting() == 0):
        pass
    
    arduino_bytes = arduino_data.readline()
    arduino_string = arduino_bytes.decode('latin-1')
    temp = arduino_string.split('\r\n')
    temp_2 = temp[0].split('t')
    
    
    try:
        current_time = time.time()
        elapsed_time = current_time - start_time
        arduino_float = float(temp_2[0])
        time_float = float(temp_2[1])
        
        arduino_array.append(arduino_float)
        ekg_t.append(time_float)
        
        print(time_float,'ms')

        
        if len(ekg_t)% 300 == 0:
            create_plot()

        for i in ekg_t:
            if i > 60000:
                monitoring = False
    
    except ValueError:
        pass


print("Finished iterating in: " + str(int(elapsed_time))  + " seconds")


raw_signal = arduino_array
raw_signal = np.array(raw_signal)
signal_len = len(raw_signal)

#plot raw signal
plt.figure()
fs = 100
t = np.arange(0,len(raw_signal))/fs
plt.plot(t,raw_signal)
plt.title('Raw Signal')
plt.xlabel('Time(s)')
plt.ylabel('Amplitude(mV)')

#plot detrended signal vs raw signal
plt.figure()
detrend_signal = scipy.signal.detrend(raw_signal, axis=-1, type='linear', bp=0)
plt.plot(t,raw_signal)
plt.plot(t,detrend_signal)
plt.title('Raw Signal vs Detrended Signal')

# Signal Proccesing
cutoff = 30 # frekuensi cutoff
nyq = 0.5 * fs
wn = cutoff/(nyq)
b, a = signal.butter(4,wn, btype = 'low', analog=False)
filtered_signal = signal.lfilter(b,a,detrend_signal)
smoothed_signal = signal.cspline1d(filtered_signal,lamb=0.3)

fig, ax = plt.subplots(2, sharex = True)
ax[0].plot(t,detrend_signal)
ax[0].plot(t,filtered_signal)
ax[0].set_title('Raw Signal vs Filtered Signal')

ax[1].plot(t,smoothed_signal)
ax[1].set_title('Smoothed_Signal')

plt.tight_layout()

# respon frekuensi
#plt.figure()
#w, h = signal.freqs(a, b)
#plt.semilogx(w, 20 * np.log10(abs(h)))
#plt.title('Response Frequency')
#plt.xlabel('Frequency')
#plt.ylabel('Amplitude response [dB]')
#plt.grid()


#rpeak detection
multiplier = 3
tres = max(smoothed_signal) * multiplier
fig, ax = plt.subplots(3, sharex = True, sharey = True)
raw_rpeak = biosppy.signals.ecg.ssf_segmenter(signal=detrend_signal, sampling_rate=fs, threshold=tres, before=0.03, after=0.01)
raw_rpeak = raw_rpeak['rpeaks']/fs
r_index = []
for i in raw_rpeak:
    r_index.append(np.where(t == i))
    
if r_index == []:
    ax[0].plot(t,detrend_signal)
    ax[0].set_title('Raw Signal, NO R peak Detected')
else:
    r_index = np.array(r_index)
    ax[0].plot(t,detrend_signal)
    ax[0].scatter(t[r_index],detrend_signal[r_index],c='r')
    ax[0].set_title('Raw Signal R Peak')


fil_rpeak = biosppy.signals.ecg.ssf_segmenter(signal=filtered_signal, sampling_rate=fs, threshold=tres, before=0.03, after=0.01)
fil_rpeak = fil_rpeak['rpeaks']/fs
r_index2 = []
for j in fil_rpeak:
    r_index2.append(np.where(t == j))
if r_index2 == []:
    ax[1].plot(t,filtered_signal)
    ax[1].set_title('Filtered Signal, NO R peak Detected')
else:
    r_index2 = np.array(r_index2)
    ax[1].plot(t,filtered_signal)
    ax[1].scatter(t[r_index2],filtered_signal[r_index2],c='r')
    ax[1].set_title('Filtered Signal R Peak')


sm_rpeak = biosppy.signals.ecg.ssf_segmenter(signal=smoothed_signal, sampling_rate=fs, threshold=tres, before=0.03, after=0.01)
sm_rpeak = sm_rpeak['rpeaks']/fs
r_index3 = []
for k in sm_rpeak:
    r_index3.append(np.where(t == k))
if r_index3 == []:
    ax[2].plot(t, smoothed_signal)
    ax[2].set_title('Smoothed Signal, NO R peak Detected')
else:
    r_index3 = np.array(r_index3)
    ax[2].plot(t, smoothed_signal)
    ax[2].scatter(t[r_index3],smoothed_signal[r_index3],c='r')
    ax[2].set_title('Smoothed Signal R Peak')

plt.tight_layout()
#jumlah R
poin_rpeak = sm_rpeak
poin_rpeak
RR_interval = np.diff(poin_rpeak)
RR_interval

#  Heart rate (beat per minute) 60s / RR_interval
heart_rate = np.mean(60/np.mean(RR_interval))
print('Heart Rate:',heart_rate)
#  Standar deviasi dari RR interval yang sudah difilter
SDNN = np.std(RR_interval)
print('SDNN:',SDNN)
#  Persentase dari selisih RR interval yang lebih dari 50ms (0.05s).
NN50 = np.sum(np.abs(np.diff(RR_interval))>.05)
pNN50 = (NN50/len(RR_interval)) * 100
print('PNN50:',pNN50)
RMSSD = (np.mean(np.diff(RR_interval) ** 2))**0.5
print('RMSSD:',RMSSD)

import nn_classifier #Panggil file kodingan yang sudah dilatih dengan jst
pred, pred_prob = nn_classifier.input_fitur(pNN50,RMSSD)
if pred == [0]:
    print(f'Normal Sinus Rythm Detected with score: {pred_prob:.2f}')
else:
    print(f'Arrythmia Detected with score {pred_prob:.2f} ')

plt.figure()
time_test = np.arange(0,len(RR_interval))
plt.step(time_test,RR_interval)
plt.title('RRInterval Plot')
plt.xlabel('Time')
plt.ylabel('RR Interval')
plt.ylim(0,2)
plt.annotate(f'Heart Rate: {heart_rate:.2f}bpm', xy=(0,0.95 ), xycoords='axes fraction')
plt.annotate(f'SDNN: {SDNN:.2f}ms', xy=(0, 0.90 ), xycoords='axes fraction')
plt.annotate(f'pNN50: {pNN50:.2f}%', xy=(0,0.85 ), xycoords='axes fraction')
plt.annotate(f'RMSSD: {RMSSD:.2f}ms', xy=(0, 0.80 ), xycoords='axes fraction')
if pred == [0]:
    plt.annotate(f'Normal Sinus Rythm Detected with score: {pred_prob:.2f}', xy=(0, 0.75 ), xycoords='axes fraction')
else:
    plt.annotate(f'Arrythmia Detected with score {pred_prob:.2f} ',xy=(0, 0.75 ), xycoords='axes fraction')


plt.show()
import xlsxwriter
name = input('Masukkan Nama:')
workbook = xlsxwriter.Workbook(f'{name}.xlsx')
worksheet = workbook.add_worksheet()

a_hr = [heart_rate]   
a_sdnn = [SDNN]
a_pnn50 = [pNN50]
a_rmssd = [RMSSD]
a_pred = [pred]

array = [arduino_array,
         ekg_t,
         a_hr,
         a_sdnn,
         a_pnn50,
         a_rmssd,
         a_pred]

row = 0

for col, data in enumerate(array):
    worksheet.write_column(row, col, data)

workbook.close()
