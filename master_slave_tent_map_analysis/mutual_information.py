import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate

def mi(x, y, bins=16):
    joint_dist, x_edges, y_edges = np.histogram2d(x, y, bins=bins)
    x_dist = np.histogram(x, bins=x_edges)[0]
    y_dist = np.histogram(y, bins=y_edges)[0]
    
    pxy = joint_dist / np.sum(joint_dist)
    px = x_dist / np.sum(x_dist)
    py = y_dist / np.sum(y_dist)

    value = 0.0
    for i in range(len(px)):
        for j in range(len(py)):
            if pxy[i, j] > 0:
                value += pxy[i, j] * np.log2(pxy[i, j] / (px[i] * py[j]))
    return value

if __name__ == "__main__":
    p = 0.4999
    epsilons  = np.linspace(0,1,51)
    trials = 50
    n = 100
    bins = 8
    
    delay = [0,1,2,5]

    cols=4
    rows = int(np.ceil(len(delay)/cols))

    fig, axes = plt.subplots(rows, cols, figsize=(12,4), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx1, d in enumerate(delay):
        ax = axes[idx1]
        
        mis = np.zeros_like(epsilons)
        for idx2, eps in enumerate(epsilons):
            mi_vals = []
            for _ in range(trials):
                X, Y = simulate(p, eps, n, delay=d)
                mi_vals.append(mi(X,Y))
            mis[idx2] = np.mean(mi_vals)

        ax.plot(epsilons, mis, alpha=0.7, color='red')
        ax.set_title(f'delay = {d}')
        
        if idx1 % cols == 0:
            ax.set_ylabel('Mutual Information')
        if idx1 >= cols*(rows-1):
            ax.set_xlabel('coupling')

    fig.suptitle('Mutual Information v/s Coupling for various delay levels')
    plt.tight_layout()
    plt.savefig(f'plots/mi_analysis/mi_vs_coupling_n{n}_b{bins}')
    plt.show()

