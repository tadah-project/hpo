#!/usr/bin/env python3
"""Compare an example's actual run output against its reference_fingerprint.json.

Reads the same files generate_fingerprint.py reads, then for each tracked
quantity checks |actual - reference| <= max(abs_tol, rel_tol * |reference|).

The failure counters of hpo_run.log's DIAGNOSTICS ("NaN/Inf clampings",
"Evaluations that threw") must not exceed the fingerprint's
nan_clampings_max and threw_max (0 when absent); a journal without either
line fails, since its guard cannot be applied.

An optional "journal" list names lines hpo_run.log must contain, for what
the losses and hyperparameters cannot show (e.g. example 25's graded
failures): each entry is {"what": <text>, "regex": <Python regex, matched
per line>} and may carry "skip_if_env": {<VAR>: <value>}, which leaves the
check out -- with a NOTE line -- when the environment sets VAR to that
value (e.g. TADAH_PC_ISOLATION=THREAD, which overrides the mode a check
is about).

Exit codes:
  0 -- all checks passed (within tolerance)
  1 -- at least one check failed
  2 -- example dir lacks a fingerprint (skipped)
  3 -- run output missing (example was not executed)
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


def parse_tabular_last_row(path: Path):
    header = None
    last = None
    for raw in path.read_text().splitlines():
        s = raw.strip()
        if not s:
            continue
        if s.startswith("#"):
            if header is None:
                header = s.lstrip("#").split()
            continue
        last = s.split()
    if header is None or last is None:
        raise RuntimeError(f"{path}: could not parse")
    return header, last


# The DIAGNOSTICS failure counters: (parse_log key, hpo_run.log label,
# the fingerprint field that holds the most a run may have).
COUNTERS = (("nan_clampings", "NaN/Inf clampings", "nan_clampings_max"),
            ("threw", "Evaluations that threw", "threw_max"))


def parse_log(path: Path):
    txt = path.read_text()
    out = {}
    m = re.search(r"Total evaluations\s*:\s*([0-9]+)", txt)
    out["total_evaluations"] = int(m.group(1)) if m else None
    m = re.search(r"Optimiser status\s*:\s*(.+)", txt)
    out["optimiser_status"] = m.group(1).strip() if m else None
    for key, label, _most in COUNTERS:
        m = re.search(re.escape(label) + r"\s*:\s*([0-9]+)", txt)
        out[key] = int(m.group(1)) if m else None
    return out


def check_journal(fp: dict, log_text: str):
    """Failures and notes of the fingerprint's "journal" checks."""
    failures, notes = [], []
    for chk in fp.get("journal", []):
        forced = {var: val for var, val in (chk.get("skip_if_env") or {}).items()
                  if os.environ.get(var) == val}
        if forced:
            notes.append(f"journal check not applied ({chk['what']}): "
                         + ", ".join(f"{var}={val}" for var, val in forced.items()))
            continue
        if not re.search(chk["regex"], log_text, re.MULTILINE):
            failures.append(f"journal: no hpo_run.log line matches "
                            f"{chk['regex']!r} ({chk['what']})")
    return failures, notes


def within_tol(actual: float, ref: float, abs_tol: float, rel_tol: float) -> bool:
    return abs(actual - ref) <= max(abs_tol, rel_tol * abs(ref))


def compare(example_dir: Path):
    fp_path = example_dir / "reference_fingerprint.json"
    if not fp_path.exists():
        return 2, [f"no fingerprint at {fp_path}"], []

    fp = json.loads(fp_path.read_text())

    log_path = example_dir / "hpo_run.log"
    loss_path = example_dir / "best_loss.tadah"
    params_path = example_dir / "best_params.tadah"
    if not (log_path.exists() and loss_path.exists() and params_path.exists()):
        return 3, [f"run output missing in {example_dir} (run example first)"], []

    log = parse_log(log_path)
    loss_hdr, loss_row = parse_tabular_last_row(loss_path)
    params_hdr, params_row = parse_tabular_last_row(params_path)

    failures = []

    # best_loss -- pull TOTAL_LOSS/N for full precision
    if "TOTAL_LOSS/N" in loss_hdr:
        idx = loss_hdr.index("TOTAL_LOSS/N")
        try:
            actual_loss = float(loss_row[idx])
            ref = fp["best_loss"]
            if not within_tol(actual_loss, ref["value"],
                              ref["abs_tol"], ref["rel_tol"]):
                failures.append(
                    f"best_loss: actual={actual_loss:.6e} "
                    f"ref={ref['value']:.6e} "
                    f"(abs_tol={ref['abs_tol']:.1e}, rel_tol={ref['rel_tol']:.1e})"
                )
        except (ValueError, IndexError) as e:
            failures.append(f"best_loss parse error: {e}")

    # components -- a reference column that the run no longer writes is a
    # failure, not a skip: a renamed or dropped column used to pass silently.
    for col, ref in fp.get("components", {}).items():
        if col not in loss_hdr:
            failures.append(f"component {col}: column missing from best_loss.tadah "
                            f"(regenerate the fingerprint if the rename is intended)")
        else:
            try:
                actual = float(loss_row[loss_hdr.index(col)])
                if not within_tol(actual, ref["value"],
                                  ref["abs_tol"], ref["rel_tol"]):
                    failures.append(
                        f"component {col}: actual={actual:.6e} "
                        f"ref={ref['value']:.6e}"
                    )
            except (ValueError, IndexError) as e:
                failures.append(f"component {col} parse error: {e}")

    # hyperparameters -- duplicate column names carry a 1-based [k]
    # suffix (must mirror generate_fingerprint.py's disambiguation).
    disamb, seen = [], {}
    for col in params_hdr:
        seen[col] = seen.get(col, 0) + 1
        disamb.append(col if params_hdr.count(col) == 1
                      else f"{col}[{seen[col]}]")
    for col, ref in fp.get("hyperparameters", {}).items():
        if col not in disamb:
            failures.append(f"hyperparam {col}: column missing from best_params.tadah "
                            f"(regenerate the fingerprint if the rename is intended)")
        else:
            try:
                actual = float(params_row[disamb.index(col)])
                if not within_tol(actual, ref["value"],
                                  ref["abs_tol"], ref["rel_tol"]):
                    failures.append(
                        f"hyperparam {col}: actual={actual:.6e} "
                        f"ref={ref['value']:.6e}"
                    )
            except (ValueError, IndexError) as e:
                failures.append(f"hyperparam {col} parse error: {e}")

    # evals
    eval_ref = fp.get("evaluations")
    if eval_ref and log["total_evaluations"] is not None:
        diff = abs(log["total_evaluations"] - eval_ref["value"])
        if diff > eval_ref["tol"]:
            failures.append(
                f"evaluations: actual={log['total_evaluations']} "
                f"ref={eval_ref['value']} (tol={eval_ref['tol']})"
            )

    # status
    status_accepted = fp.get("status_accepted") or []
    if status_accepted and log["optimiser_status"] not in status_accepted:
        failures.append(
            f"status: actual={log['optimiser_status']!r} "
            f"not in accepted list ({len(status_accepted)} variants)"
        )

    # failure counters: NaN/Inf clampings, evaluations that threw. A
    # missing line fails: a renamed or dropped counter used to leave its
    # guard out silently.
    for key, label, most_field in COUNTERS:
        most = fp.get(most_field, 0)
        if log[key] is None:
            failures.append(f"{key}: no hpo_run.log line '{label} : N' "
                            f"(DIAGNOSTICS), so max={most} cannot be checked")
        elif log[key] > most:
            failures.append(f"{key}: actual={log[key]} > max={most}")

    # journal lines
    journal_failures, notes = check_journal(fp, log_path.read_text())
    failures += journal_failures

    return (0 if not failures else 1), failures, notes


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("example_dir", type=Path)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    code, msgs, notes = compare(args.example_dir)
    name = args.example_dir.name
    if code == 0:
        if not args.quiet:
            print(f"  PASS  {name}")
    elif code == 1:
        print(f"  FAIL  {name}")
        for m in msgs:
            print(f"        - {m}")
    elif code == 2:
        print(f"  SKIP  {name}: {msgs[0]}")
    elif code == 3:
        print(f"  MISSING {name}: {msgs[0]}")
    for n in notes:
        print(f"  NOTE  {name}: {n}")
    return code


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
