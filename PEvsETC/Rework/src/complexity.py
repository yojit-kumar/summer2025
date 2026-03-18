import numpy as np
from numba import njit
from ordpy import permutation_entropy
from ETC import partition, compute_1D


@njit
def logistic_map(x, a):
    return a * x * (1 - x)



def lyapunov_exponent(a, L, transient):
    rng = np.random.default_rng(42)
    x0 = rng.random()

    x = x0
    value = 0.0

    for _ in range(transient):
        x = logistic_map(x, a)

    for _ in range(L):
        x = logistic_map(x, a)
        arg = np.abs(a * (1 - 2*x))

        if arg > 0:
            value += np.log(arg)
        else:
            value += -np.inf

    return value/L


def pe_method(series, D, t):
    value = permutation_entropy(series, dx=D, taux=t, normalized=True)
    return value


def etc_method(series, bins):
    series = partition(series, n_bins=bins)
    value = compute_1D(series).get('NET1CD')
    return value


if __name__=="__main__":
    print("python script contains functions to measure complexity")
