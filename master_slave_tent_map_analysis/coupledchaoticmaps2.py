import numpy as np
import matplotlib.pyplot as plt

from simulation import simulate


e_values = [0.0, 0.2, 0.4, 0.41, 0.45, 0.5, 0.8, 1.0]
cols = 4
rows = int(np.ceil(len(e_values) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for idx, eps in enumerate(e_values):
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

# Hide any unused subplots
for j in range(idx + 1, rows * cols):
    fig.delaxes(axes[j])

fig.suptitle('First-Return Maps for Various Couplings', y=1.02)
plt.tight_layout()
plt.show()
