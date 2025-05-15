import numpy as np
import matplotlib.pyplot as plt

def tent_maps(x,p):
    return np.where(x < p, x/p, (1-x) / (1-p))

def simulate(p, eps, n, delay=1):
    X = np.zeros(n)
    Y = np.zeros(n)
    X[0], Y[0] = np.random.rand(), np.random.rand()
    for i in range(1, n):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * X[i-delay]
    return X, Y

def mutual_information(x, y, bins=25):
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

if __name__ == "__main__":
    p = 0.4999
    epsilons  = np.linspace(0,1,51)
    trials = 50
    n = 1000
    
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
                mi_vals.append(mutual_information(X,Y))
            mis[idx2] = np.mean(mi_vals)

        ax.plot(epsilons, mis, alpha=0.7, color='red')
        ax.set_title(f'delay = {d}')
        
        if idx1 % cols == 0:
            ax.set_ylabel('Mutual Information')
        if idx1 >= cols*(rows-1):
            ax.set_xlabel('coupling')

    fig.suptitle('Mutual Information v/s Coupling for various delay levels')
    plt.tight_layout()
    plt.show()

