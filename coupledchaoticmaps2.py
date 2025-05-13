import numpy as np
import matplotlib.pyplot as plt
import zlib

# 1D tent map definition
def tent_map(x, p):
    return np.where(x < p, x / p, (1 - x) / (1 - p))

# Simulation of master-slave coupled maps
def simulate(p, eps, n, discard):
    X = np.zeros(n + discard)
    Y = np.zeros(n + discard)
    X[0], Y[0] = np.random.rand(), np.random.rand()
    for i in range(1, n + discard):
        X[i] = tent_map(X[i-1], p)
        Y[i] = (1 - eps) * tent_map(Y[i-1], p) + eps * tent_map(X[i-1], p)
    return X[discard:], Y[discard:]

# Pearson correlation
#def correlation(x, y):
#    return np.corrcoef(x, y)[0, 1]
def correlation(x, y):
    x_mean = np.mean(x)
    y_mean = np.mean(y)
    numerator = np.sum((x - x_mean) * (y - y_mean))
    denominator = np.sqrt(np.sum((x - x_mean)**2)) * np.sqrt(np.sum((y - y_mean)**2))
    return numerator / denominator if denominator != 0 else 0.0


# Histogram-based mutual information
def mutual_information(x, y, bins=10):
    joint, x_edges, y_edges = np.histogram2d(x, y, bins=bins, density=True)
    px = np.histogram(x, bins=x_edges, density=True)[0]
    py = np.histogram(y, bins=y_edges, density=True)[0]
    pxy = joint / joint.sum()
    px = px / px.sum()
    py = py / py.sum()
    mi = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > 0:
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi

# Simple METC via zlib compression lengths
def metc(x, y, bins=16):
    xq = np.digitize(x, np.linspace(0, 1, bins))
    yq = np.digitize(y, np.linspace(0, 1, bins))
    bx = bytes(xq)
    by = bytes(yq)
    bxy = bx + by
    Cx = len(zlib.compress(bx))
    Cy = len(zlib.compress(by))
    Cxy = len(zlib.compress(bxy))
    return Cx + Cy - Cxy

# Parameters
p = 0.4999
epsilons = np.linspace(0, 1, 51)
trials = 50
n = 100
discard = 50

# Allocate arrays
ccs = np.zeros_like(epsilons)
mis = np.zeros_like(epsilons)
metcs = np.zeros_like(epsilons)

# Compute CC, MI, METC vs epsilon
for idx, eps in enumerate(epsilons):
    cc_vals, mi_vals, metc_vals = [], [], []
    for _ in range(trials):
        X, Y = simulate(p, eps, n, discard)
        cc_vals.append(correlation(X, Y))
        mi_vals.append(mutual_information(X, Y, bins=10))
        metc_vals.append(metc(X, Y, bins=16))
    ccs[idx] = np.mean(cc_vals)
    mis[idx] = np.mean(mi_vals)
    metcs[idx] = np.mean(metc_vals)

# Plot CC, MI, METC vs epsilon
plt.figure(figsize=(8, 8))
plt.plot(epsilons, ccs, marker='o', label='Correlation Coef.')
#plt.plot(epsilons, mis, marker='x', label='Mutual Info')
#plt.plot(epsilons, metcs, marker='s', label='METC')
plt.xlabel('Coupling ε')
plt.ylabel('Measure')
#plt.title('Mean CC, MI, METC vs ε')
plt.title('Mean CC vs. coupling')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# First-return subplots for different epsilon values
# Plot X[n] vs X[n+1] and Y[n] vs Y[n+1] in each subplot
e_values = [0.0, 0.2, 0.4, 0.41, 0.45, 0.5, 0.8, 1.0]
cols = 4
rows = int(np.ceil(len(e_values) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for idx, eps in enumerate(e_values):
    ax = axes[idx]
    X, Y = simulate(p, eps, n, discard)
    ax.scatter(X[:-1], X[1:], marker='o', alpha=0.7, label='Xₙ vs Xₙ₊₁')
    ax.scatter(Y[:-1], Y[1:], marker='x', alpha=0.7, label='Yₙ vs Yₙ₊₁')
    ax.set_title(f'ε = {eps}')
    if idx % cols == 0:
        ax.set_ylabel('Value at n+1')
    if idx >= cols * (rows - 1):
        ax.set_xlabel('Value at n')
    ax.legend(fontsize='small')

# Hide any unused subplots
for j in range(idx + 1, rows * cols):
    fig.delaxes(axes[j])

fig.suptitle('First-Return Maps for Various ε', y=1.02)
plt.tight_layout()
plt.show()
