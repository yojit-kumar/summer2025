import pyedflib as edf
import numpy as np
import pandas as pd
import ETC

file1 = "/home/user/eeg-motor-movementimagery-dataset-1.0.0/files/S002/S002R02.edf"
f = edf.EdfReader(file1)

n = f.signals_in_file
signal_labels = f.getSignalLabels()

print(f"number of channels = {n}")
print(f"labels = {signal_labels}")

sigbufs = np.zeros((n, f.getNSamples()[0]))
for i in np.arange(n):
    sigbufs[i,:] = f.readSignal(i)

print(f.getNSamples())

signal = f.readSignal(1)[:9600]
print(signal)
print(signal[9600:])
print(len(signal))

seq = ETC.partition(signal, n_bins=2)
res = ETC.compute_1D(seq, verbose=False).get('NETC1D')
print(res)
