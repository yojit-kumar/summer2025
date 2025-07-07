import numpy as np
import matplotlib.pyplot as plt

from ordpy import permutation_entropy
from ETC import partition, compute_1D
from scipy.integrate import solve_ivp


def rossler(t, state, a=0.1, b=0.1, c=4):
    x, y, z = state
    dxdt = -y - z
    dydt = x + a*y
    dzdt = b + z*(x - c)
    return [dxdt, dydt, dzdt]


def simulate(c, L, transient, noise=0.0, dt=0.01):
    rng = np.random.default_rng(42)
    state = rng.random(3)

    n = L + transient
    series = np.zeros(n)
    series[0] = state[0]
 
    for i in range(1, n):
        dxdt, dydt, dzdt = rossler(None, state, c=c)
        state[0] += dxdt*dt
        state[1] += dydt*dt
        state[2] += dzdt*dt

        series[i] = state[0]
    
    if noise > 0:
        for i in range(1, n):
            delta = rng.normal(0, noise)
            series[i] += delta 

    return series[transient:]


def rossler_jacobian(state, a=0.1, b=0.1, c=4.0):
    x, y, z = state
    return np.array([
        [ 0,     -1,     -1],
        [ 1,      a,      0],
        [ z,      0,  x - c]
    ])


def lyapunov_exponent(a=0.1, b=0.1, c=4.0, t_max=500, dt=0.01, delta0=1e-8):
    """
    Compute the largest Lyapunov exponent for the Rössler system
    using variational equations.
    """
    t_span = [0, t_max]
    t_eval = np.arange(0, t_max, dt)

    # Initial state and perturbation
    rng = np.random.default_rng(42)
    state = rng.random(3)
    delta = rng.normal(0, 1, 3)
    delta /= np.linalg.norm(delta)
    delta *= delta0

    sum_log_divergence = 0.0
    steps = 0

    for t_idx in range(len(t_eval) - 1):
        # Integrate system
        sol = solve_ivp(lambda t, y: rossler(t, y, a, b, c), 
                        [t_eval[t_idx], t_eval[t_idx+1]], 
                        state, 
                        method="RK45", t_eval=[t_eval[t_idx+1]])
        state = sol.y[:, -1]

        # Linearize and evolve perturbation
        J = rossler_jacobian(state, a, b, c)
        delta = delta + dt * J @ delta

        # Compute divergence
        norm_delta = np.linalg.norm(delta)
        if norm_delta == 0:
            norm_delta = 1e-16

        # Accumulate log divergence
        sum_log_divergence += np.log(norm_delta / delta0)
        steps += 1

        # Renormalize perturbation
        delta = (delta / norm_delta) * delta0

    # Average divergence rate
    le = sum_log_divergence / (steps * dt)
    return le


def method1(series, D, t):
    value = permutation_entropy(series, dx=D, taux=t, normalized=True)
    return value


def method2(series, bins):
    series = partition(series, n_bins=bins)
    value = compute_1D(series).get('NETC1D')
    return value
    

def parameter_sweep(c_values, L, transient, D, t, bins, noise=0.0, verbose=True):
    lyapunov = []
    permutation = []
    etc = []
    
    for i, c in enumerate(c_values):
        if verbose and i % 20 == 0:
            print(f"{int(100*i/len(c_values))}%")

        series = simulate(c, L, transient, noise)

        lyapunov.append(lyapunov_exponent(c=c, t_max=L*0.01, dt=0.01))
        permutation.append(method1(series, D, t))
        etc.append(method2(series, bins=bins))

    if verbose:
        print("100%")

    lyapunov = np.array(lyapunov)
    permutation = np.array(permutation)
    etc = np.array(etc)

    return lyapunov, permutation, etc


def plotting(c_values, lyapunov, permutation, etc):
    plt.figure(figsize=(15, 12))
    plt.plot(c_values, lyapunov, 'r-', label='Lyapunov Exponent')
    plt.plot(c_values, permutation, 'g-', label='Permutation Entropy')
    plt.plot(c_values, etc, 'b-', label='Effort-To-Compress Ratio')

    plt.xlabel('Parameter c')
    plt.ylabel('Complexity Measures')

    plt.title('Complexity Measures vs Rossler Parameter')
    plt.grid()
    plt.legend()

    plt.show()


def correlation(c_values, transient, D, t, bins):
    test_case = np.arange(1000, 10000, 1000)
    cases = len(test_case)
    case = 1
    print("Progress:")

    l_p_cc = []
    l_e_cc = []
    for L in test_case:
        l, p, e = parameter_sweep(c_values, L, transient, D, t, bins, verbose=False)
        l_p_cc.append(np.corrcoef(l, p)[0, 1])
        print(f"permutation: {np.corrcoef(l, p)[0, 1]}")
        l_e_cc.append(np.corrcoef(l, e)[0, 1])
        print(f"etc: {np.corrcoef(l, e)[0, 1]}")
        print(f"Case {case}/{cases}")
        case += 1
 
    plt.figure(figsize=(15, 12))
    plt.plot(test_case, l_p_cc, 'g-', label='Permutation Entropy', alpha=0.8)
    plt.plot(test_case, l_e_cc, 'b-', label='ETC', alpha=0.8)

    plt.xlabel('Length of time series')
    plt.ylabel('Correlation Coefficient with Lyapunov Exponent')

    plt.title('Correlation Coefficient of different methods with Lyapunov Exponent')
    plt.grid()
    plt.legend()
    
    plt.show()


def plotting_with_noise(c_values, results_dict):
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Lyapunov exponent (clean, reference)
    lyap_clean = results_dict[0.0][0]
    axes[0, 0].plot(c_values, lyap_clean, 'k-', linewidth=2, label='Lyapunov (clean)')
    axes[0, 0].axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    axes[0, 0].set_xlabel('Parameter c')
    axes[0, 0].set_ylabel('Lyapunov Exponent')
    axes[0, 0].set_title('Lyapunov Exponent (Reference)')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Plot 2: Permutation Entropy with different noise levels
    for i, (noise, (lyap, perm, etc)) in enumerate(results_dict.items()):
        if i < len(results_dict):
            label = f'Noise σ={noise}' if noise > 0 else 'Clean'
            axes[0, 1].plot(c_values, perm, linewidth=1.5, label=label, alpha=0.8)
    
    axes[0, 1].set_xlabel('Parameter c')
    axes[0, 1].set_ylabel('Permutation Entropy')
    axes[0, 1].set_title('Permutation Entropy vs Gaussian Noise Level')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Plot 3: ETC with different noise levels
    for i, (noise, (lyap, perm, etc)) in enumerate(results_dict.items()):
        if i < len(results_dict):
            label = f'Noise σ={noise}' if noise > 0 else 'Clean'
            axes[1, 0].plot(c_values, etc, linewidth=1.5, label=label, alpha=0.8)
    
    axes[1, 0].set_xlabel('Parameter c')
    axes[1, 0].set_ylabel('ETC Ratio')
    axes[1, 0].set_title('Effort-To-Compress vs Gaussian Noise Level')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Plot 4: Correlation with Lyapunov vs noise level
    noise_levels = list(results_dict.keys())
    perm_correlations = []
    etc_correlations = []
    
    for noise, (lyap, perm, etc) in results_dict.items():
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


def gaussian_noise_analysis(c_values, L, transient, D, t, bins, noise_levels):
    results = {}
    
    for noise in noise_levels:
        print(f"Processing noise level σ = {noise:.3f}...")
        lyap, perm, etc = parameter_sweep(c_values, L, transient, D, t, bins, 
                                        noise=noise, verbose=False)
        results[noise] = (lyap, perm, etc)
    
    # Plot results
    plotting_with_noise(c_values, results)
    
    for noise, (lyap, perm, etc) in results.items():
        perm_corr = np.corrcoef(lyap, perm)[0, 1]
        etc_corr = np.corrcoef(lyap, etc)[0, 1]
        print(f"{noise:<12.3f} {perm_corr:<15.4f} {etc_corr:<12.4f}")
    
    return results


if __name__ == "__main__":
    L = 1000 
    transient = 100
    D = 8 
    t = 1 
    bins = 4 
    
    c_values = np.linspace(4, 18, 500)

    l, p, e = parameter_sweep(c_values, L, transient, D, t, bins)
    plotting(c_values, l, p, e)
    correlation(c_values, transient, D, t, bins)

    #noise_levels = [0.0, 0.005, 0.01, 0.02, 0.05, 0.1]
    #results = gaussian_noise_analysis(c_values, L, transient, D, t, bins, noise_levels)
