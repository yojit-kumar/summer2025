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

def cc(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    num = np.sum( (x - x_m)*(y - y_m))
    den = np.sqrt(np.sum((x - x_m)**2)) * np.sqrt(np.sum((y-y_m)**2))

    return num/den if den != 0 else 0.0

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
        
        ccs = np.zeros_like(epsilons)
        for idx2, eps in enumerate(epsilons):
            cc_vals = []
            for _ in range(trials):
                X, Y = simulate(p, eps, n, delay=d)
                cc_vals.append(cc(X,Y))
            ccs[idx2] = np.mean(cc_vals)

        ax.plot(epsilons, ccs, alpha=0.7, color='blue')
        ax.set_title(f'delay = {d}')
        
        if idx1 % cols == 0:
            ax.set_ylabel('Correlation Coeffcient')
        if idx1 >= cols*(rows-1):
            ax.set_xlabel('coupling')

    fig.suptitle('Correlation Coefficient v/s Coupling for various delay levels')
    plt.tight_layout()
    plt.show()
