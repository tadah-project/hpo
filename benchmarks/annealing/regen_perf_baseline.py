#!/usr/bin/env python3
"""Regenerate tests/data/anneal_perf_baseline.tsv and review the drift.

Reruns the performance matrix with TADAH_PERF_RECORD=1 (which rewrites the
baseline in place), then prints, for every cell whose success count moved,
the old and new counts with Wilson 95% intervals. The review criterion for
accepting a new baseline is NON-INFERIORITY of the cells your change did
not intend to move -- never a blind refresh.

Usage:  regen_perf_baseline.py <build_dir>   (e.g. .../build_asa_dev)
"""

import math
import os
import pathlib
import shutil
import subprocess
import sys

HERE = pathlib.Path(__file__).resolve()
BASELINE = HERE.parents[2] / "tests" / "data" / "anneal_perf_baseline.tsv"
N = 50


def wilson(k: int, n: int = N, z: float = 1.96):
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return ((c - h) / d, (c + h) / d)


def load(path):
    rows = {}
    for line in path.read_text().splitlines():
        cfg, fn, succ, med = line.split("\t")
        rows[(cfg, fn)] = (int(succ), float(med))
    return rows


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    test_bin = pathlib.Path(sys.argv[1]) / "tests" / "test_hpo"
    if not test_bin.exists():
        raise SystemExit(f"no test binary at {test_bin}")
    old = load(BASELINE) if BASELINE.exists() else {}
    backup = BASELINE.with_suffix(".tsv.bak")
    if BASELINE.exists():
        shutil.copy(BASELINE, backup)
    env = dict(os.environ, TADAH_PERF_RECORD="1")
    subprocess.run([str(test_bin), "[perf_matrix]"], env=env, check=True)
    new = load(BASELINE)

    moved = 0
    for key in sorted(new):
        ns, nm = new[key]
        os_, om = old.get(key, (None, None))
        if os_ is None or os_ != ns:
            moved += 1
            lo_n, hi_n = wilson(ns)
            tag = "NEW-CELL" if os_ is None else (
                "better" if ns > os_ else "WORSE")
            oldtxt = "-" if os_ is None else f"{os_}/{N}"
            if os_ is not None:
                lo_o, hi_o = wilson(os_)
                sep = "non-overlapping!" if (lo_n > hi_o or hi_n < lo_o) \
                    else "CIs overlap"
            else:
                sep = ""
            print(f"{key[0]:24s} {key[1]:20s} {oldtxt:>6s} -> {ns}/{N}  "
                  f"[{lo_n:.2f},{hi_n:.2f}]  {tag:8s} {sep}  "
                  f"median {om} -> {nm}")
    if moved == 0:
        print("baseline unchanged (bit-identical matrix)")
    else:
        print(f"\n{moved} cell(s) moved; previous baseline kept at "
              f"{backup}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
