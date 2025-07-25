import numpy as np
import matplotlib.pyplot as plt
import ETC

from ETC_self import *
from simulation import simulate

def metc(x, y, bins=2):
    x = ETC.partition(x, n_bins=bins)
    y = ETC.partition(y, n_bins=bins)

    ETCx = ETC.compute_1D(x, verbose=False).get('NETC1D')
    ETCy = ETC.compute_1D(y, verbose=False).get('NETC1D')
    ETCxy = ETC.compute_2D(x, y, verbose=False).get('NETC2D')

    return ETCx + ETCy - ETCxy

def metc_self(x, y, bins=2, normalized=True, verbose=False):
    Cx = etc(x, num_bins=bins, normalized=normalized, verbose=verbose)
    Cy = etc(y, num_bins=bins, normalized=normalized, verbose=verbose)
    Cxy = etc(np.concatenate((x,y),axis=None), num_bins=bins, normalized=normalized, verbose=verbose)

    return Cx + Cy - Cxy

if __name__ == "__main__":
    p = 0.4999
    epsilons  = np.linspace(0,1,51)
    trials = 50
    n = 100
    bins=2
    
    delay = [0,1,2,3]

    cols=4
    rows = int(np.ceil(len(delay)/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12,4), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx1, d in enumerate(delay):
        ax = axes[idx1]
        
        metcs = np.zeros_like(epsilons)
        metc_selfs = np.zeros_like(epsilons)
        for idx2, eps in enumerate(epsilons):
            metc_vals = []
            metc_self_vals = []
            for _ in range(trials):
                X, Y = simulate(p, eps, n, delay=d)
                metc_self_vals.append(metc_self(X,Y,bins=bins))
                metc_vals.append(metc(X,Y,bins=bins))
            metcs[idx2] = np.mean(metc_vals)
            metc_selfs[idx2] = np.mean(metc_self_vals)

        ax.plot(epsilons, metcs, alpha=0.7, color='red', label='ETCpy')
        ax.plot(epsilons, metc_selfs, alpha=0.7, color='blue', label='ETC_self')
        ax.set_title(f'delay = {d}')
        
        if idx1 % cols == 0:
            ax.set_ylabel('Mutual ETC')
        if idx1 >= cols*(rows-1):
            ax.set_xlabel('coupling')

    fig.suptitle('Mutual ETC v/s Coupling for various delay levels')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'plots/metc_analysis/METC_vs_coupling_n{n}_b{bins}')
    plt.show()

