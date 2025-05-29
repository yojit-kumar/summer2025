import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate


def mean_sq_error(x, y):
    x = np.array(x)
    y = np.array(y)

    return np.mean( (y-x)**2 )


def curve(p, x_value, n, trials=50):
    mse_values = []

    for _ in range(trials):
        X, Y = simulate(p, x_value, n)
        mse = mean_sq_error(X,Y)
        mse_values.append(mse)

    y_value = np.mean(mse_values)

    return y_value



if __name__=="__main__":
    p_array=[0.4999,0.4,0.2,0.1]
    n = 100
    epsilons = np.linspace(0,1,51)
    trials = 50
    
    cols = 4
    rows = int( np.ceil(len(p_array)/cols) )

    fig, axes = plt.subplots(rows, cols, figsize=(12,6), sharex=True, sharey=True)
    axes = axes.flatten()

    for idx, p in enumerate(p_array):
        ax = axes[idx]

        x_axis = epsilons
        y_axis = [curve(p, x_value, n, trials=trials) for x_value in x_axis]

        ax.plot(x_axis, y_axis, alpha=0.7)

        ax.set_title(f'p = {p}')

        if idx % cols == 0:
            ax.set_ylabel('Mean Square Error of Y wrt X')
            ax.set_xlabel('Coupling')

    plt.tight_layout()
    plt.savefig(f'plots/MSE_vs_Coupling_over_{trials}trials')
    plt.show()



