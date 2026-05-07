#!/usr/bin/env python3
"""
Extract metrics from an hpo_run.log for BASINS measurement comparisons.

Usage:
    parse_hpo_log.py <hpo_run.log>

Emits a self-describing block to stdout with: total wall time, eval count,
NEW BEST events (eval, loss), BASINS round/stage transitions with timestamps,
counts of BAD_POT / StagnationStop / simplex-collapse / nlopt-failure /
gap-skip warnings.
"""
import re
import sys
from pathlib import Path


_TS = re.compile(r"^\[(\d{2}):(\d{2}):(\d{2})\]")


def _parse_ts(line):
    m = _TS.match(line)
    if not m:
        return None
    h, mi, s = (int(x) for x in m.groups())
    return h * 3600 + mi * 60 + s


def _wall_seconds(first, last):
    if first is None or last is None:
        return None
    delta = last - first
    if delta < 0:  # midnight wrap
        delta += 86400
    return delta


def main(path):
    text = Path(path).read_text(errors="replace").splitlines()

    started_iso = None
    finished_iso = None
    first_ts = None
    last_ts = None

    new_best = []  # (eval, loss)
    rounds = []   # (timestamp_str, round, stage_or_step_text)
    cnt = {
        "bad_pot": 0,
        "stagnation_stop": 0,
        "simplex_collapse": 0,
        "nlopt_failure": 0,
        "gap_skip": 0,
        "basin_skipped_hopeless": 0,
        "basin_no_improvement": 0,
    }
    final_eval = None
    final_best = None

    re_started = re.compile(r"Started\s+:\s*(\S+\s\S+)")
    re_finished = re.compile(r"Finished\s+:\s*(\S+\s\S+)")
    re_newbest = re.compile(r"NEW BEST\s+eval=\s*(\d+)\s+loss=([0-9.eE+\-]+)")
    re_round = re.compile(r"BASINS (round=\d+ \w+(?:\s+step=\d+)?[^|]*)")
    re_summary_eval = re.compile(r"---\s+(\d+)\s+evals done\s+.*best=([0-9.eE+\-]+)")

    for line in text:
        ts = _parse_ts(line)
        if ts is not None:
            if first_ts is None:
                first_ts = ts
            last_ts = ts
        if started_iso is None:
            m = re_started.search(line)
            if m:
                started_iso = m.group(1)
        m = re_finished.search(line)
        if m:
            finished_iso = m.group(1)
        m = re_newbest.search(line)
        if m:
            new_best.append((int(m.group(1)), float(m.group(2))))
        m = re_round.search(line)
        if m:
            ts_match = _TS.match(line)
            ts_str = "[" + ":".join(ts_match.groups()) + "]" if ts_match else ""
            rounds.append((ts_str, m.group(1).strip()))
        m = re_summary_eval.search(line)
        if m:
            final_eval = int(m.group(1))
            final_best = float(m.group(2))
        if "BAD_POT" in line or "skipping LAMMPS" in line:
            cnt["bad_pot"] += 1
        if "StagnationStop" in line:
            cnt["stagnation_stop"] += 1
        if "simplex" in line and "too small" in line:
            cnt["simplex_collapse"] += 1
        if "nlopt failure" in line:
            cnt["nlopt_failure"] += 1
        if "gap-skip" in line:
            cnt["gap_skip"] += 1
        if "skipped at tol" in line and "hopeless" in line:
            cnt["basin_skipped_hopeless"] += 1
        if "(no improvement)" in line:
            cnt["basin_no_improvement"] += 1

    wall = _wall_seconds(first_ts, last_ts)

    print(f"# {path}")
    print(f"started        : {started_iso}")
    print(f"finished       : {finished_iso}")
    print(f"wall_seconds   : {wall}")
    print(f"final_eval     : {final_eval}")
    print(f"final_best     : {final_best}")
    print(f"new_best_count : {len(new_best)}")
    print(f"counts         : "
          f"bad_pot={cnt['bad_pot']} stagnation_stop={cnt['stagnation_stop']} "
          f"simplex_collapse={cnt['simplex_collapse']} "
          f"nlopt_failure={cnt['nlopt_failure']} gap_skip={cnt['gap_skip']} "
          f"basin_hopeless_skipped={cnt['basin_skipped_hopeless']} "
          f"basin_no_improvement={cnt['basin_no_improvement']}")
    print()
    print("# new_best trajectory (eval, loss)")
    for ev, ls in new_best:
        print(f"{ev}\t{ls:.6e}")
    print()
    print("# round/stage transitions (timestamp, event)")
    for ts, evt in rounds:
        print(f"{ts}\t{evt}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(f"usage: {sys.argv[0]} <hpo_run.log>")
    main(sys.argv[1])
