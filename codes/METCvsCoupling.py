import numpy as np
import matplotlib.pyplot as plt
from ETC import etc

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

def metc(x, y, bins=2):
    Cx = etc(x, num_bins=bins)
    Cy = etc(y, num_bins=bins)
    Cxy = etc(x+y, num_bins=bins)

    return Cx + Cy - Cxy

p = 0.4999
epsilons = np.linspace(0, 1, 51)
trials = 50
n = 100

metcs = np.zeros_like(epsilons)

for idx, eps in enumerate(epsilons):
    cc_vals = []
    mi_vals = []
    metc_vals = []
    for _ in range(trials):
        X, Y = simulate(p, eps, n)
        metc_vals.append(metc(X, Y, 2))
    metcs[idx] = np.mean(metc_vals)

plt.figure(figsize=(8, 8))
plt.plot(epsilons, metcs, marker='s', label='METC')
plt.xlabel('Coupling')
plt.ylabel('METC')
plt.title('Mean METC vs Coupling')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()
