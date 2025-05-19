import numpy as np
import matplotlib.pyplot as plt
import ETC
from ETC_self import *

def generate(length):
    rng = np.random.default_rng()
    list = rng.random(length)

    return list

length = np.arange(500,10000,500)
trials = 10
bins = 50

myETC_mean = []
ETCpy_mean = []

for n in length:
    myETC_vals = []
    ETCpy_vals = []

    for _ in range(trials):
        seq = generate(n)
        myETC_vals.append(etc(seq, num_bins=bins, normalized=False))
        seq = ETC.partition(seq, n_bins=bins)
        ETCpy_vals.append(ETC.compute_1D(seq).get('ETC1D'))

    myETC_mean.append(np.mean(myETC_vals))
    ETCpy_mean.append(np.mean(ETCpy_vals))

deviation = (np.array(myETC_mean) - np.array(ETCpy_mean))

plt.figure(figsize=(10,10))
plt.scatter(length, ETCpy_mean, marker='o', label='ETCpy', alpha=0.5, color='blue')
plt.scatter(length, myETC_mean, marker='x', label='ETC_self', alpha=0.5, color='red')

plt.xlabel('length of sequence')
plt.ylabel('ETC value')
plt.title('Comparision of the values between diff ETC implementation')

plt.legend()
plt.grid()
plt.show()

plt.figure(figsize=(10,10))
plt.plot(length, deviation, marker='o', label='deviation', alpha=0.7, color='black')

plt.xlabel('length of sequence')
plt.ylabel('deviation (myETC - ETCpy)')
plt.title('Comparision of the values between diff ETC implementation')

plt.legend()
plt.grid()
plt.show()




