#!/usr/bin/env python3
"""Compare two annealing benchmark TSVs (A = baseline, B = candidate).

Prints every cell whose success count moved, with Wilson 95% intervals,
and flags cells whose intervals do not overlap (a resolved difference).

Usage:  summarize.py baseline.tsv candidate.tsv
"""

import math
import sys


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(path):
    rows = {}
    with open(path) as fh:
        header = fh.readline()
        assert header.startswith("config"), f"{path}: not a benchmark TSV"
        for line in fh:
            cfg, fn, n, succ, med, wall = line.rstrip("\n").split("\t")
            rows[(cfg, fn)] = (int(n), int(succ), float(med), float(wall))
    return rows


def main():
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    a, b = load(sys.argv[1]), load(sys.argv[2])
    moved = resolved = 0
    for key in sorted(set(a) | set(b)):
        if key not in a or key not in b:
            print(f"{key[0]:20s} {key[1]:20s} present in only one file")
            continue
        na, sa, ma, wa = a[key]
        nb, sb, mb, wb = b[key]
        if sa == sb:
            continue
        moved += 1
        lo_a, hi_a = wilson(sa, na)
        lo_b, hi_b = wilson(sb, nb)
        sep = lo_b > hi_a or hi_b < lo_a
        resolved += sep
        print(f"{key[0]:20s} {key[1]:20s} {sa}/{na} [{lo_a:.2f},{hi_a:.2f}]"
              f" -> {sb}/{nb} [{lo_b:.2f},{hi_b:.2f}]"
              f"  {'RESOLVED' if sep else 'CIs overlap'}"
              f"  median {ma:g} -> {mb:g}  wall {wa:.0f} -> {wb:.0f} ms")
    print(f"\n{moved} cell(s) moved, {resolved} resolved beyond CI overlap")


if __name__ == "__main__":
    main()
