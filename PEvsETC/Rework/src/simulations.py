import numpy as np

from numba import njit
from complexity import logistic_map



@njit
def simulate(a, L, transient, noise=0.0):
    np.random.seed(42)
    x0 = np.random.random()

    n = L + transient
    series = np.zeros(n)
    series[0] = x0
    
    for i in range(1,n):
        series[i] = logistic_map(series[i-1],a)
    
    if noise > 0.0:
        for i in range(1,n):
            delta = np.random.normal(0.0, noise)
            series[i] += delta
        series = np.clip(series, 0.0, 1.0)

    return series[transient:]


if __name__=="__main__":
    print("simulation generating code")
