#!/usr/bin/env python3
"""Opt-in oracle: scipy.optimize.dual_annealing on the shared functions.

An INDEPENDENT, peer-reviewed annealer (generalized simulated annealing)
run on the same functions/budgets/seed-count as the in-tree matrix, for
context only -- different algorithm family, never a CI gate.

Usage:  compare_scipy.py [n_seeds] > scipy.tsv
Needs:  scipy (pip install scipy)
"""

import sys

import numpy as np
from scipy import optimize

FUNCS = {
    "sphere6": (6, -1.0, 1.0, lambda x: float(np.sum(x * x) / 6.0), 1e-3),
    "rosenbrock6": (6, -2.0, 2.0,
                    lambda x: float(sum(100.0 * (x[i + 1] - x[i] ** 2) ** 2
                                        + (1 - x[i]) ** 2
                                        for i in range(5)) / (101.0 * 5)),
                    1e-2),
    "rastrigin6": (6, -5.12, 5.12,
                   lambda x: float(np.sum(x * x - 10 * np.cos(2 * np.pi * x)
                                          + 10) / 120.0), 5e-3),
    "ackley10": (10, -32.768, 32.768,
                 lambda x: float(-20 * np.exp(-0.2 * np.sqrt(np.mean(x * x)))
                                 - np.exp(np.mean(np.cos(2 * np.pi * x)))
                                 + 20 + np.e), 5e-2),
}


def main():
    n_seeds = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print("config\tfunction\tseeds\tsuccess\tmedian_best\twall_ms")
    import time
    for name, (dim, lo, hi, f, tol) in FUNCS.items():
        succ, bests = 0, []
        t0 = time.time()
        for s in range(n_seeds):
            r = optimize.dual_annealing(
                f, bounds=[(lo, hi)] * dim, maxfun=300 * dim,
                seed=1000 + s, no_local_search=True)
            bests.append(r.fun)
            if r.fun < tol:
                succ += 1
        bests.sort()
        print(f"scipy-dual-annealing\t{name}\t{n_seeds}\t{succ}\t"
              f"{bests[len(bests) // 2]:g}\t{(time.time() - t0) * 1e3:.0f}")


if __name__ == "__main__":
    main()
