#!/usr/bin/env python3
"""Pretty-print a strategy_matrix summary CSV as a markdown table.

Usage:
    summarise.py <summary.csv> [-o <out.md>]
"""
import argparse
import csv
import math
import sys


def fmt(v):
    try:
        f = float(v)
        if math.isnan(f):
            return "NaN"
        if abs(f) < 1e-3 or abs(f) > 1e4:
            return f"{f:.3e}"
        return f"{f:.4f}"
    except (TypeError, ValueError):
        return str(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="summary.csv produced by strategy_matrix.sh")
    ap.add_argument("-o", "--out", help="output markdown path (default: stdout)")
    args = ap.parse_args()

    with open(args.csv) as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        print("(empty)", file=sys.stderr)
        return 1

    rows.sort(key=lambda r: float(r["best_loss"]) if r["best_loss"] not in ("NaN", "") else float("inf"))

    out = []
    out.append("| rank | name | wall (s) | best loss | evals | exit |")
    out.append("|---:|---|---:|---:|---:|---:|")
    for i, r in enumerate(rows, 1):
        out.append(f"| {i} | {r['name']} | {r['wall_s']} | {fmt(r['best_loss'])} | {r['evals']} | {r['exit']} |")

    text = "\n".join(out) + "\n"
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
    else:
        sys.stdout.write(text)


if __name__ == "__main__":
    sys.exit(main() or 0)
