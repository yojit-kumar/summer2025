import numpy as np
import matplotlib.pyplot as plt
import zlib

def tent_maps(x,p):
    return np.where(x < p, x/p, (1-x) / (1-p))

def simulate(p, eps, n):
    X = np.zeros(n)
    Y = np.zeros(n)
    X[0], Y[0] = np.random.rand(), np.random.rand()
    for i in range(1, n):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * tent_maps(X[i-1],p)
    return X, Y

def cc(x,y):
    x_m = np.mean(x)
    y_m = np.mean(y)
    num = np.sum( (x - x_m)*(y - y_m))
    den = np.sqrt(np.sum((x - x_m)**2)) * np.sqrt(np.sum((y-y_m)**2))

    return num/den if den != 0 else 0.0


def mutual_information(x, y, bins=10):
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
                mi += pxy[i, j] * np.log(pxy[i, j] / (px[i] * py[j]))
    return mi

# Simple METC via zlib compression lengths
def metc(x, y, bins=16):
    xq = np.digitize(x, np.linspace(0, 1, bins))
    yq = np.digitize(y, np.linspace(0, 1, bins))
    bx = bytes(xq)
    by = bytes(yq)
    bxy = bx + by
    Cx = len(zlib.compress(bx))
    Cy = len(zlib.compress(by))
    Cxy = len(zlib.compress(bxy))
    return Cx + Cy - Cxy

p = 0.4999
epsilons = np.linspace(0, 1, 51)
trials = 50
n = 100

ccs = np.zeros_like(epsilons)
mis = np.zeros_like(epsilons)
metcs = np.zeros_like(epsilons)

for idx, eps in enumerate(epsilons):
    cc_vals = []
    mi_vals = []
    metc_vals = []
    for _ in range(trials):
        X, Y = simulate(p, eps, n)
        cc_vals.append(cc(X,Y))
        mi_vals.append(mutual_information(X, Y, bins=10))
        metc_vals.append(metc(X, Y, bins=16))
    ccs[idx] = np.mean(cc_vals)
    mis[idx] = np.mean(mi_vals)
    metcs[idx] = np.mean(metc_vals)

plt.figure(figsize=(8, 8))
plt.plot(epsilons, ccs, marker='o', label='Correlation Coef.')
plt.plot(epsilons, mis, marker='x', label='Mutual Info')
#plt.plot(epsilons, metcs, marker='s', label='METC')
plt.xlabel('Coupling ε')
plt.ylabel('Measure')
plt.title('Mean CC, MI, METC vs ε')
plt.title('Mean CC vs. coupling')
plt.grid()
plt.legend()
plt.tight_layout()
plt.show()

# First-return subplots for different epsilon values
# Plot X[n] vs X[n+1] and Y[n] vs Y[n+1] in each subplot
e_values = [0.0, 0.2, 0.4, 0.41, 0.45, 0.5, 0.8, 1.0]
cols = 4
rows = int(np.ceil(len(e_values) / cols))
fig, axes = plt.subplots(rows, cols, figsize=(12, 6), sharex=True, sharey=True)
axes = axes.flatten()

for idx, eps in enumerate(e_values):
    ax = axes[idx]
    X, Y = simulate(p, eps, n)
    ax.scatter(X[:-1], X[1:], marker='o', alpha=0.7, label='Xₙ vs Xₙ₊₁')
    ax.scatter(Y[:-1], Y[1:], marker='x', alpha=0.7, label='Yₙ vs Yₙ₊₁')
    ax.set_title(f'ε = {eps}')
    if idx % cols == 0:
        ax.set_ylabel('Value at n+1')
    if idx >= cols * (rows - 1):
        ax.set_xlabel('Value at n')
    ax.legend(fontsize='small')

# Hide any unused subplots
for j in range(idx + 1, rows * cols):
    fig.delaxes(axes[j])

fig.suptitle('First-Return Maps for Various ε', y=1.02)
plt.tight_layout()
plt.show()
