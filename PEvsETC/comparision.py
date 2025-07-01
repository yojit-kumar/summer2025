import numpy as np
import matplotlib.pyplot as plt

from ordpy import permutation_entropy
from ETC import partition, compute_1D



def logistic_map(x, a):
    return a * x * (1 - x)


def simulate(a, L, transient, noise=0.0):
    rng = np.random.default_rng(42)
    x0 = rng.random()

    n = L + transient
    series = np.zeros(n)
    series[0] = x0
 
    for i in range(1,n):
        series[i] = logistic_map(series[i-1],a)
    
    if noise > 0:
        for i in range(1,n):
            noise = rng.normal(0, noise)
            series[i] += noise
            series[i] = np.clip(series[i], 0, 1)

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
    

def parameter_sweep(a_values, L, transient, D, t, bins, noise=0.0, verbose=True):
    lyapunov = []
    permutation = []
    etc = []
    
    for i, a in enumerate(a_values):
        if verbose and i%20 == 0:
            print(f"{int(100*i/len(a_values))}%\n", end="")

        series = simulate(a, L, transient, noise)

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


def plotting_with_noise(a_values, results_dict):
    """Plot comparison of methods with different Gaussian noise levels"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Lyapunov exponent (clean, reference)
    lyap_clean = results_dict[0.0][0]  # Clean Lyapunov
    axes[0, 0].plot(a_values, lyap_clean, 'k-', linewidth=2, label='Lyapunov (clean)')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Parameter a')
    axes[0, 0].set_ylabel('Lyapunov Exponent')
    axes[0, 0].set_title('Lyapunov Exponent (Reference)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Permutation Entropy with different noise levels
    colors = ['green', 'orange', 'red', 'purple']
    for i, (noise_level, (lyap, perm, etc)) in enumerate(results_dict.items()):
        if i < len(colors):
            label = f'Noise σ={noise_level}' if noise_level > 0 else 'Clean'
            axes[0, 1].plot(a_values, perm, color=colors[i], linewidth=1.5, 
                          label=label, alpha=0.8)
    
    axes[0, 1].set_xlabel('Parameter a')
    axes[0, 1].set_ylabel('Permutation Entropy')
    axes[0, 1].set_title('Permutation Entropy vs Gaussian Noise Level')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: ETC with different noise levels
    for i, (noise_level, (lyap, perm, etc)) in enumerate(results_dict.items()):
        if i < len(colors):
            label = f'Noise σ={noise_level}' if noise_level > 0 else 'Clean'
            axes[1, 0].plot(a_values, etc, color=colors[i], linewidth=1.5, 
                          label=label, alpha=0.8)
    
    axes[1, 0].set_xlabel('Parameter a')
    axes[1, 0].set_ylabel('ETC Ratio')
    axes[1, 0].set_title('Effort-To-Compress vs Gaussian Noise Level')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Correlation with Lyapunov vs noise level
    noise_levels = list(results_dict.keys())
    perm_correlations = []
    etc_correlations = []
    
    for noise_level, (lyap, perm, etc) in results_dict.items():
        perm_corr = np.corrcoef(lyap, perm)[0, 1]
        etc_corr = np.corrcoef(lyap, etc)[0, 1]
        perm_correlations.append(perm_corr)
        etc_correlations.append(etc_corr)
    
    axes[1, 1].plot(noise_levels, perm_correlations, 'go-', linewidth=2, 
                   markersize=6, label='Permutation Entropy')
    axes[1, 1].plot(noise_levels, etc_correlations, 'bo-', linewidth=2, 
                   markersize=6, label='ETC Ratio')
    axes[1, 1].set_xlabel('Gaussian Noise Level (σ)')
    axes[1, 1].set_ylabel('Correlation with Lyapunov')
    axes[1, 1].set_title('Robustness to Gaussian Noise')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    axes[1, 1].set_ylim([0, 1])
    
    plt.tight_layout()
    plt.show()


def gaussian_noise_analysis(a_values, L, transient, D, t, bins, noise_levels):
    """
    Analyze how Gaussian noise affects the complexity measures
    """
    print(f"\nGaussian Noise Robustness Analysis")
    print("=" * 40)
    
    results = {}
    
    for noise_level in noise_levels:
        print(f"Processing noise level σ = {noise_level:.3f}...")
        lyap, perm, etc = parameter_sweep(a_values, L, transient, D, t, bins, 
                                        noise_level, verbose=False)
        results[noise_level] = (lyap, perm, etc)
    
    # Plot results
    plotting_with_noise(a_values, results)
    
    # Print correlation analysis
    print("\nCorrelation with Lyapunov Exponent:")
    print("-" * 40)
    print(f"{'Noise Level':<12} {'Perm. Entropy':<15} {'ETC Ratio':<12}")
    print("-" * 40)
    
    for noise_level, (lyap, perm, etc) in results.items():
        perm_corr = np.corrcoef(lyap, perm)[0, 1]
        etc_corr = np.corrcoef(lyap, etc)[0, 1]
        print(f"{noise_level:<12.3f} {perm_corr:<15.4f} {etc_corr:<12.4f}")
    
    return results


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

    noise_levels = [0.005, 0.01, 0.02, 0.05, 0.1]
    results = gaussian_noise_analysis(a_values, L, transient, D, t, bins, noise_levels)
