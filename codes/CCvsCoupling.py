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
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps *X[i-1]
    return X, Y

def cc(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    num = np.sum( (x - x_m)*(y - y_m))
    den = np.sqrt(np.sum((x - x_m)**2)) * np.sqrt(np.sum((y-y_m)**2))

    return num/den if den != 0 else 0.0

#def cc(x,y):
#    num = np.cov(x,y)
#    den = np.std(x) * np.std(y)
#    return num/den

p = 0.4999
epsilons  = np.linspace(0,1,21)
trials = 50
n = 100

ccs = np.zeros_like(epsilons)

for idx, eps in enumerate(epsilons):
    cc_vals = []
    for _ in range(trials):
        X, Y = simulate(p, eps, n)
        cc_vals.append(cc(X,Y))
    ccs[idx] = np.mean(cc_vals)

plt.figure(figsize=(8,8))
plt.plot(epsilons, ccs, marker = 'o', label='Correlation Coef')
plt.xlabel('Coupling')
plt.ylabel('CC')
plt.title('Mean CC vs. Coupling')
plt.grid()
plt.legend()
plt.show()

