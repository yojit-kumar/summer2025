import numpy as np
import matplotlib.pyplot as plt

from ordpy import permutation_entropy 
from ETC import partition, compute_1D

# Transition probabilities
P10 = 0.8  # P(1|0)
P01 = 0.1  # P(0|1)

# Simulation parameters
num_simulations = 50  # Number of runs to average over
max_length = 500     # Maximum length of time series
window_size = 20      # Moving window block size


def method1_complexity(series, D=3, t=1):
    value = permutation_entropy(series, dx=D, taux=t, normalized=True)
    return value

def method2_complexity(series, bins=4):
    series = partition(series, n_bins=bins)
    value = compute_1D(series).get('NETC1D')
    return value

# Function to simulate Markov chain
def simulate_markov_chain(length, P10, P01, initial_state=0):
    states = np.zeros(length, dtype=int)
    states[0] = initial_state
    for i in range(1, length):
        if states[i-1] == 0:
            states[i] = np.random.choice([0,1], p=[1-P10, P10])
        else:
            states[i] = np.random.choice([1,0], p=[1-P01, P01])
    return states

# Arrays to store complexity values
lengths = np.arange(20, max_length + 1)
method1_results = np.zeros_like(lengths, dtype=float)
method2_results = np.zeros_like(lengths, dtype=float)

# Main simulation loop
for L in lengths:
    method1_vals = []
    method2_vals = []
    for _ in range(num_simulations):
        ts = simulate_markov_chain(L, P10, P01)
        method1_vals.append(method1_complexity(ts))
        method2_vals.append(method2_complexity(ts))
    method1_results[np.where(lengths==L)[0][0]] = np.mean(method1_vals)
    method2_results[np.where(lengths==L)[0][0]] = np.mean(method2_vals)

    print(f"{(np.where(lengths == L)[0][0] +1)*100/len(lengths) :.2f}")

# Moving average over window blocks
def moving_average(values, window_size):
    return np.convolve(values, np.ones(window_size)/window_size, mode='valid')

smoothed_m1 = moving_average(method1_results, window_size)
smoothed_m2 = moving_average(method2_results, window_size)
smoothed_lengths = lengths[window_size-1:]  # Adjusted x-axis

# Plotting
plt.figure(figsize=(10,6))
plt.plot(smoothed_lengths, smoothed_m1, label="Permutation Entropy", color="blue")
plt.plot(smoothed_lengths, smoothed_m2, label="ETC", color="red")
plt.xlabel("Time Series Length (L)")
plt.ylabel("Complexity Measure")
plt.title("Convergence of Complexity Measures on a Two-State Markov Process")
plt.legend()
plt.grid(True)
plt.show()
