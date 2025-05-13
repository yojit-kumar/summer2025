import numpy as np
import matplotlib.pyplot as plt

def tent_maps(x, p):
    return np.where(x < p, x/p, (1-x) / (1-p))

def simulate(p, eps, n):
    X = np.zeros(n)
    Y = np.zeros(n)
    X[0], Y[0] = np.random.rand(), np.random.rand()
    for i in range(1, n):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * tent_maps(X[i-1], p)
    return X, Y

def cc(x, y):
    """Calculate Pearson correlation coefficient"""
    x_m = np.mean(x)
    y_m = np.mean(y)
    num = np.sum((x - x_m) * (y - y_m))
    den = np.sqrt(np.sum((x - x_m)**2)) * np.sqrt(np.sum((y - y_m)**2))
    return num/den if den != 0 else 0.0

def mutual_information(x, y, bins=10):
    """Calculate mutual information between two time series"""
    joint_dist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_dist = np.histogram(x, bins=x_edges)[0]
    y_dist = np.histogram(y, bins=y_edges)[0]
    
    pxy = joint_dist / joint_dist.sum()
    px = x_dist / x_dist.sum()
    py = y_dist / y_dist.sum()
    
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi

def pair_frequencies(S):
    """Calculate pair frequencies in a sequence"""
    L = len(S)
    freq_of_pairs = {}  # dictionary to hold frequencies corresponding to pairs
    index_of_pairs = {}  # dictionary to hold the first occurrence of the pair, to be used in tie-breaking

    for i in range(L-1):
        pair = S[i:i+2]
        if pair not in freq_of_pairs:
            freq_of_pairs[pair] = 1
            index_of_pairs[pair] = i
        else:
            freq_of_pairs[pair] += 1
    return freq_of_pairs, index_of_pairs

def tie_break(pair, index_of_pairs):
    """Determine tie-breaks between similar pair candidates for substitution"""
    a, b = map(int, list(pair))
    if a == 0 and b == 0:  # have to use an if clause as both 1 & 0 are of scale 1
        scale = a+b+2
    elif a == 0 or b == 0:
        scale = a+b+1
    else:
        scale = a+b  # scale is roughly the length of the pattern
    smallest_digit = min(a, b)
    first_occurence = index_of_pairs[pair]

    return [scale, smallest_digit, first_occurence]

def selection(freq, index):
    """Select the pair to be substituted using successive tie-break steps"""
    max_freq = max(freq.values())
    candidates = [k for k, v in freq.items() if v == max_freq]  # selecting all keys with maximum values
    
    if len(candidates) > 1:
        min_scale = min(tie_break(pair, index)[0] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index)[0] == min_scale]  # selecting the keys with least scales
        
    if len(candidates) > 1:
        min_digit = min(tie_break(pair, index)[1] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index)[1] == min_digit]  # selecting the keys with the smallest individual digit
            
    if len(candidates) != 1:
        min_first_occurence = min(tie_break(pair, index)[2] for pair in candidates)
        candidates = [pair for pair in candidates if tie_break(pair, index)[2] == min_first_occurence]  # selecting the pair which occur first
    
    most_repeated = candidates[0]
    return most_repeated

def etc(sequence):
    """Calculate Entropy Temporal Complexity steps for a binary sequence"""
    # Make a copy to avoid changing the original sequence
    S = sequence.copy()
    
    # Check if the sequence is already binary (0,1)
    unique_values = np.unique(S)
    if len(unique_values) > 2:
        raise ValueError("Input sequence must be binary (only two unique values)")
    
    # Convert to string representation with 0 and 1
    if len(unique_values) == 2:
        # Map the smaller value to 0 and larger to 1
        binary_seq = "".join(['0' if x == min(unique_values) else '1' for x in S])
    else:
        # If only one value, convert all to 0
        binary_seq = "".join(['0' for _ in S])
    
    t = 0  # step counter
    
    # NSRPS algorithm
    while len(binary_seq) > 1:
        freq, index = pair_frequencies(binary_seq)
        if not freq:  # If no pairs left
            break
        most_repeated = selection(freq, index)
        new_symbol = str(t+2)  # Start new symbols from 2
        binary_seq = binary_seq.replace(most_repeated, new_symbol)
        t += 1  # counting iterations
    
    return t

def binarize_timeseries(ts, bins=16):
    """Convert a time series to a binary sequence using equal-width bins"""
    # Create bins between min and max values
    bin_edges = np.linspace(np.min(ts), np.max(ts), bins+1)
    
    # Digitize the time series (bin indices are 1-based)
    binned = np.digitize(ts, bin_edges)
    
    # Convert to binary by taking the parity (odd/even)
    binary = binned % 2
    
    return binary

def metc_nsrps(x, y, bins=16):
    """Calculate METC using NSRPS algorithm"""
    # Binarize the time series
    x_bin = binarize_timeseries(x, bins)
    y_bin = binarize_timeseries(y, bins)
    
    # Concatenate for joint complexity
    xy_bin = np.concatenate((x_bin, y_bin))
    
    # Calculate ETC for each
    etc_x = etc(x_bin)
    etc_y = etc(y_bin)
    etc_xy = etc(xy_bin)
    
    # METC formula
    metc_value = etc_x + etc_y - etc_xy
    
    return metc_value

# Simulation parameters
p = 0.4999
epsilons = np.linspace(0, 1, 21)  # Reduced number of points for faster calculation
trials = 10  # Reduced number of trials for faster calculation
n = 100

# Initialize arrays for metrics
ccs = np.zeros_like(epsilons)
mis = np.zeros_like(epsilons)
metcs = np.zeros_like(epsilons)

# Run simulations for different coupling strengths
for idx, eps in enumerate(epsilons):
    cc_vals = []
    mi_vals = []
    metc_vals = []
    print(f"Processing epsilon = {eps:.2f} ({idx+1}/{len(epsilons)})")
    
    for trial in range(trials):
        X, Y = simulate(p, eps, n)
        cc_vals.append(cc(X, Y))
        mi_vals.append(mutual_information(X, Y, bins=10))
        metc_vals.append(metc_nsrps(X, Y, bins=16))
        
    ccs[idx] = np.mean(cc_vals)
    mis[idx] = np.mean(mi_vals)
    metcs[idx] = np.mean(metc_vals)

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(epsilons, ccs, marker='o', linestyle='-', label='Correlation Coef.')
plt.plot(epsilons, mis / np.max(mis), marker='x', linestyle='-', label='Mutual Info (normalized)')
plt.plot(epsilons, metcs / np.max(metcs), marker='s', linestyle='-', label='METC (normalized)')
plt.xlabel('Coupling strength (ε)')
plt.ylabel('Measure (normalized)')
plt.title('Comparison of dependency measures vs coupling strength')
plt.grid(True)
plt.legend()
plt.tight_layout()
plt.show()

# First-return subplots for different epsilon values
e_values = [0.0, 0.2, 0.4, 0.5, 0.8, 1.0]
cols = 3
rows = int(np.ceil(len(e_values) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 8), sharex=True, sharey=True)
axes = axes.flatten()

for idx, eps in enumerate(e_values):
    ax = axes[idx]
    X, Y = simulate(p, eps, n)
    ax.scatter(X[:-1], X[1:], marker='o', s=30, alpha=0.7, label='Master (X)', color='blue')
    ax.scatter(Y[:-1], Y[1:], marker='x', s=30, alpha=0.7, label='Slave (Y)', color='red')
    ax.set_title(f'ε = {eps}')
    if idx % cols == 0:
        ax.set_ylabel('Value at n+1')
    if idx >= cols * (rows - 1):
        ax.set_xlabel('Value at n')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize='small')

# Hide any unused subplots
for j in range(idx + 1, rows * cols):
    fig.delaxes(axes[j])

fig.suptitle('First-Return Maps for Various Coupling Strengths (ε)', y=1.02)
plt.tight_layout()
plt.show()
