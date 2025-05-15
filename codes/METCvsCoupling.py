import numpy as np
import matplotlib.pyplot as plt
#import ETC
from ETC_self import etc

def tent_maps(x,p):
    return np.where(x < p, x/p, (1-x) / (1-p))

def simulate(p, eps, n):
    X = np.zeros(n)
    Y = np.zeros(n)
    X[0], Y[0] = np.random.rand(), np.random.rand()
    for i in range(1, n):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * X[i-1]
    return X, Y

def metc_self(x, y, bins=2, normalized=True, verbose=False):
    Cx = etc(x, num_bins=bins, normalized=normalized, verbose=verbose)
    Cy = etc(y, num_bins=bins, normalized=normalized, verbose=verbose)
    Cxy = etc(np.concatenate((x,y),axis=None), num_bins=bins, normalized=normalized, verbose=verbose)

    return Cx + Cy - Cxy

#def metc(x, y, bins=2):
#    x = ETC.partition(x, n_bins=bins)
#    y = ETC.partition(y, n_bins=bins)

#    ETCx = ETC.compute_1D(x, verbose=False).get('NETC1D')
#    ETCy = ETC.compute_1D(y,verbose=False).get('NETC1D')
#    ETCxy = ETC.compute_2D(x,y,verbose=False).get('NETC2D')

#    return ETCx + ETCy - ETCxy

if __name__ == '__main__':
    p = 0.4999
    epsilons = np.linspace(0,1,21)
    trials = 50 
    n = 100

    metcs = np.zeros_like(epsilons)

    for idx, eps in enumerate(epsilons):
        metc_vals = []
      
        for _ in range(trials):
            X, Y = simulate(p, eps, n)
            metc_vals.append(metc_self(X, Y, normalized=True))
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
