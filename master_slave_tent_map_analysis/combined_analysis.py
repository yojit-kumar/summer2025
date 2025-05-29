import numpy as np
import matplotlib.pyplot as plt
import ETC

from simulation import simulate
from correlation_coeffcient import cc
from mutual_information import mi
from mutual_ETC import metc
from mutual_ETC import metc_self

from timeit import default_timer as timer


if __name__ == '__main__':
    for p in [0.4999, 0.4, 0.2, 0.1]:
        for bins in [2, 5, 8, 16]:
            for n in [100, 1000, 10000]:

                #p = 0.4999
                epsilons= np.linspace(0,1,21)
                trials = 50
                #n = 1000
                #bins = 2
                delay = [0,1,2,3]

                start = timer()

                cols = 4 
                rows = int(np.ceil(len(delay)/cols))

                fig, axes = plt.subplots(rows, cols, figsize=(12,4), sharex=True, sharey=True)
                fig.subplots_adjust(wspace=0.05)
                axes = axes.flatten()

                for idx1, d in enumerate(delay):
                    ax = axes[idx1]

                    ccs = np.zeros_like(epsilons)
                    mis = np.zeros_like(epsilons)
                    metcs = np.zeros_like(epsilons)

                    for idx2, eps in enumerate(epsilons):
                        cc_vals = []
                        mi_vals = []
                        metc_vals = []

                        for _ in range(trials):
                            X, Y = simulate(p, eps, n, delay=d)
                            
                            cc_vals.append(cc(X, Y))
                            mi_vals.append(mi(X, Y, bins=bins))
                            metc_vals.append(metc(X, Y, bins=bins))

                        ccs[idx2] = np.mean(cc_vals)
                        mis[idx2] = np.mean(mi_vals)
                        metcs[idx2] = np.mean(metc_vals)

                        print(f"d={d} | eps={eps} | time={timer()-start:.3f}")

                    ax.plot(epsilons, ccs, alpha=0.7, color='Red', label='Correlation Coefficient')
                    ax.plot(epsilons, mis, alpha=0.7, color='Green', label='Mutual Information')
                    ax.plot(epsilons, metcs, alpha=0.7, color='Blue', label='Mutual Effort-To-Compress Ratio')
                    ax.set_title(f'delay = {d}')

                    if idx1 % cols == 0 :
                        ax.set_ylabel('Mean Measure')
                    if idx1 >= cols*(rows-1):
                        ax.set_xlabel('Coupling')

                fig.suptitle('Mean CC, MI, METC vs. Coupling for different delay')
                plt.legend()
                plt.tight_layout()
                plt.savefig(f'plots/combined_analysis/p{p}_n{n}_b{bins}.png')
              #  plt.show()
