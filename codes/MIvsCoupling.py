import numpy as np
import matplotlib.pyplot as plt

def tent_maps(x,p):
    return np.where(x < p, x/p, (1-x) / (1-p))

def simulate(p, eps, n):
    X = np.zeros(n)
    Y = np.zeros(n)
    X[0], Y[0] = np.random.rand(), np.random.rand()
    for i in range(1, n):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * tent_maps(X[i-1],p)
    return X, Y

def mutual_information(x, y, bins=10):
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
                mi += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return mi

p = 0.4999
epsilons = np.linspace(0, 1, 51)
trials = 50
n = 100

mis = np.zeros_like(epsilons)

for idx, eps in enumerate(epsilons):
    mi_vals = []
    for _ in range(trials):
        X, Y = simulate(p, eps, n)
        mi_vals.append(mutual_information(X, Y, bins=10))
    mis[idx] = np.mean(mi_vals)

plt.figure(figsize=(8, 8))
plt.plot(epsilons, mis, marker='x', label='Mutual Info')
plt.xlabel('Coupling')
plt.ylabel('Measure')
plt.title('Mean MI vs. Coupling')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
