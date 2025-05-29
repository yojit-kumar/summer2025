import numpy as np
import matplotlib.pyplot as plt

def tent_maps(x, p):
    return np.where( x < p, x/p, (1-x)/(1-p) )

def simulate(p, eps, n, delay=1):
    X = np.zeros(n + 100)
    Y = np.zeros(n + 100)

    rng = np.random.default_rng(7)

    X[0], Y[0] = rng.random(), rng.random()

    for i in range(1, n + 100):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * X[i-delay]
    
    return X[100:], Y[100:]

if __name__=="__main__":
    p = 0.4999
    n = 100


    epsilons = [0.0, 0.2, 0.4, 0.41, 0.45, 0.5, 0.8, 1.0]

    cols = 4
    rows = int(np.ceil(len(epsilons) / cols))
    
    fig, axes = plt.subplots(rows, cols, figsize=(12, 6), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, eps in enumerate(epsilons):
        ax = axes[idx]
        X, Y = simulate(p, eps, n)
        ax.scatter(X[:-1], X[1:], marker='o', alpha=0.7, label='Xn vs Xn+1')
        ax.scatter(Y[:-1], Y[1:], marker='x', alpha=0.7, label='Yn vs Yn+1')
        ax.set_title(f'epsilon = {eps}')
        if idx % cols == 0:
            ax.set_ylabel('Value at n+1')
        if idx >= cols * (rows - 1):
            ax.set_xlabel('Value at n')
        ax.legend(fontsize='small')


    fig.suptitle('First-Return Maps for Various Couplings')
    plt.tight_layout()
    plt.savefig('plots/combined_curve_for_master_slave_tent_maps.png')

    plt.show()
    
