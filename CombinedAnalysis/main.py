import numpy as np
import matplotlib.pyplot as plt
from ETC_self import *
#import ETC



def tent_maps(x,p):
    return np.where(x < p, x/p, (1-x)/(1-p))



def simulate(p, eps, n, delay):
    X = np.zeros(n + 100)
    Y = np.zeros(n + 100)

    rng = np.random.default_rng(7)

    X[0], Y[0] = rng.random(), rng.random()

    for i in range(1, n + 100):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * X[i-delay]

    return X[100:], Y[100:]



def correlation_coefficient(X, Y):
    X_m = np.mean(X)
    Y_m = np.mean(Y)

    numerator = np.sum((X - X_m) * (Y- Y_m))
    denominator = np.sqrt(np.sum((X - X_m)**2)) * np.sqrt(np.sum((Y - Y_m)**2))

    return numerator/denominator if denominator != 0 else 0.0



def mutual_information(X, Y, bins=10):
    joint_dist, X_edges, Y_edges = np.histogram2d(X, Y, bins=bins)
    X_dist = np.histogram(X, bins=X_edges)[0]
    Y_dist = np.histogram(Y, bins = Y_edges)[0]

    Pxy = joint_dist / joint_dist.sum()
    Px = X_dist / X_dist.sum()
    Py = Y_dist / Y_dist.sum()

    mi = 0.0
    for i in range(len(Px)):
        for j in range(len(Py)):
            if Pxy[i,j] > 0:
                mi += Pxy[i,j] * np.log2(Pxy[i,j] / (Px[i] * Py[j]))

    return mi



def mutual_etc(X, Y, bins=2):
    ETCx = etc(X, num_bins=bins, normalized=True, verbose=False)
    ETCy = etc(Y, num_bins=bins, normalized=True, verbose=False)
    ETCxy = etc(np.concatenate((X,Y),axis=None), num_bins=bins, normalized=True, verbose=False)

    return ETCx + ETCy - ETCxy



if __name__ == '__main__':
    p = 0.4999
    epsilons= np.linspace(0,1,21)
    trials = 50
    n = 1000
    bins = 2
    delay = [0,1,2,5,10]

    cols = 5
    rows = int(np.ceil(len(delay)/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(15,4), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=0.05)
    axes = axes.flatten()

    for idx1, d in enumerate(delay):
        ax = axes[idx1]

        ccs = np.zeros_like(epsilons)
        mis = np.zeros_like(epsilons)
        metcs = np.zeros_like(epsilons)

        for idx2, eps in enumerate(epsilons):
            cc_vals = []
            mi_vals = []
            metc_vals = []

            for _ in range(trials):
                X, Y = simulate(p, eps, n, delay=d)
                
                cc_vals.append(correlation_coefficient(X, Y))
                mi_vals.append(mutual_information(X, Y, bins=bins))
                metc_vals.append(mutual_etc(X, Y, bins=bins))

            ccs[idx2] = np.mean(cc_vals)
            mis[idx2] = np.mean(mi_vals)
            metcs[idx2] = np.mean(metc_vals)

        ax.plot(epsilons, ccs, alpha=0.7, color='Red', label='Correlation Coefficient')
        ax.plot(epsilons, mis, alpha=0.7, color='Green', label='Mutual Information')
        ax.plot(epsilons, metcs, alpha=0.7, color='Blue', label='Mutual Effort-To-Compress Ratio')
        ax.set_title(f'delay = {d}')

        if idx1 % cols == 0 :
            ax.set_ylabel('Mean Measure')
        if idx1 >= cols*(rows-1):
            ax.set_xlabel('Coupling')

    fig.suptitle('Mean CC, MI, METC vs. Coupling for different delay')
    plt.legend()
    plt.tight_layout()
    plt.show()



