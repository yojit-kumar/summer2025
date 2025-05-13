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

def cc(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    num = np.sum( (x - x_m)*(y - y_m))
    den = np.sqrt(np.sum((x - x_m)**2)) * np.sqrt(np.sum((y-y_m)**2))

    return num/den if den != 0 else 0.0

p = 0.4999
epsilons  = np.linspace(0,1,11)
trials = 50
n = 100

ccs = np.zeros_like(epsilons)

for idx, eps in enumerate(epsilons):
    cc_vals = []
    for _ in range(trials):
        X, Y = simulate(p, eps, n)
        cc_vals.append(cc(X,Y))
    ccs[idx] = np.mean(cc_vals)

plt.figure(figsize=(5,5))
plt.plot(epsilons, ccs, marker = 'o', label='correlation coef')
plt.xlabel('coupilng')
plt.ylabel('CC')
plt.grid()
plt.legend()
plt.show()

