import numpy as np
import matplotlib.pyplot as plt

from ordpy import permutation_entropy
from ETC import partition, compute_1D



def logistic_map(x, a):
    return a * x * (1 - x)


def simulate(a, L, transient):
    rng = np.random.default_rng(42)
    x0 = rng.random()

    n = L + transient
    series = np.zeros(n)
    series[0] = x0
 
    for i in range(1,n):
        series[i] = logistic_map(series[i-1],a)

    return series[transient:]


def lyapunov_exponent(a, L, transient):
    rng = np.random.default_rng(42)
    x0 = rng.random()
    
    x= x0
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


def method1(series, D, t):
    value = permutation_entropy(series, dx=D, taux=t, normalized=True)
    return value


def method2(series, bins):
    series = partition(series, n_bins=bins)
    value = compute_1D(series).get('NETC1D')
    return value
    

def parameter_sweep(a_values, L, transient, D, t, bins, verbose=True):
    lyapunov = []
    permutation = []
    etc = []
    
    for i, a in enumerate(a_values):
        if verbose and i%20 == 0:
            print(f"{int(100*i/len(a_values))}%\n", end="")

        series = simulate(a, L, transient)

        lyapunov.append(lyapunov_exponent(a, L, transient))
        permutation.append(method1(series, D, t))
        etc.append(method2(series, bins=bins))


    lyapunov = np.array(lyapunov)
    permutation = np.array(permutation)
    etc = np.array(etc)

    return lyapunov, permutation, etc


def plotting(a_values, lyapunov, permutation, etc):
    lyapunov = (lyapunov-np.min(lyapunov))/np.max(lyapunov)
    permutation = (permutation-np.min(permutation))/np.max(permutation)
    etc = (etc-np.min(etc))/np.max(etc)

    plt.figure(figsize=(15,12))
    plt.plot(a_values, lyapunov, 'r-', label='Lyapunov Exponent')
    plt.plot(a_values, permutation, 'g-', label='Permutation Entropy')
    plt.plot(a_values, etc, 'b-', label='Effort-To-Compress Ratio')

    plt.xlabel('Parameter a')
    plt.ylabel('Complexity Measures')

    plt.title('Complexity Measures vs Logistics Map Parameter')
    plt.grid()
    plt.legend()

    plt.show()

def correlation(a_values, transient, D, t, bins):
    test_case = np.arange(20,210,10)
    #test_case = np.append(test_case, [500,1000])
    #test_case = np.arange(100, 1100, 100)

    cases = len(test_case)
    case = 1
    print("Progress :\n")

    l_p_cc = []
    l_e_cc = []
    for L in test_case:
        l, p, e = parameter_sweep(a_values, L, transient, D, t, bins, verbose=False)
        l_p_cc.append(np.corrcoef(l,p)[0,1])
        l_e_cc.append(np.corrcoef(l,e)[0,1])
        print(f"Case {case}/{cases}\n")
        case +=1
 

    plt.figure(figsize=(15,12))
    plt.plot(test_case, l_p_cc, 'g-', label='Permutation Entropy', alpha=0.8)
    plt.plot(test_case, l_e_cc, 'b-', label='ETC', alpha=0.8)

    plt.xlabel('length of time series')
    plt.ylabel('Correlation Coeffecient with Lyapunov Exponent')

    plt.title('Correlation Coefficient of different methods with Lyapunov Exponent')
    plt.grid()
    plt.legend()

    plt.show()


if __name__=="__main__":
    L = 1000
    transient = 100
    D = 6 
    t = 2 
    bins = 5
    
    a_values = np.linspace(3.5, 4, 500)
    l, p, e = parameter_sweep(a_values, L, transient, D, t, bins)
    plotting(a_values, l, p, e)
    correlation(a_values, transient, D, t, bins)
