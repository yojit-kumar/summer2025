import numpy as np

def tent_maps(x, p):
    return np.where( x < p, x/p, (1-x)/(1-p) )

def simulate(p, eps, n, delay=1):
    X = np.zeros(n)
    Y = np.zeros(n)

    rng = np.random.default_rng(7)

    X[0], Y[0] = rng.random(), rng.random()

    for i in range(1, n + 100):
        X[i] = tent_maps(X[i-1], p)
        Y[i] = (1 - eps) * tent_maps(Y[i-1], p) + eps * X[i-delay]
    
    return X[100:], Y[100:]

if __name__=="__main__":
    p = 0.4999
    eps = 0.2
    n = 100

    X, Y = simulate(p, eps, n)
    
    print(f'p (skew paramter) = {p}, eps (coupling) = {eps}, n (sequence length) = {n}')

    print(f'X time series: {X}')
    print(f'Y time series: {Y}')
