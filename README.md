e# real-time_ecg_classification

Hardware:
1. Arduino UNO
2. AD8232 ECG sensor and some electrode

Software:
1. Arduino IDE
2. Python IDLE
3. ECG Dataset from physionet.org (feature already extracted at Dataset.csv)

Using python library "Serial", you can communicate arduino serial monitor value to python.
Low pass filter, and cubic spline interpolation is used to eliminate unecessary noise, so you can detect r peak easier.
Classification is done using artificial neural network, using MLPClassifier from sci-kit learn.
Monitoring done in 60 seconds, and every 3 seconds ECG curve will updated while also presenting BPM,SDNN,RMSSD,PNN50 Value in real time.
After 60 seconds, system will automatically determine one of the two classes, which Normal Sinus Rythm or Arrythmia.
