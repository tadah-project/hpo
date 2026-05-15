#!/usr/bin/env python3
"""Generate reference_fingerprint.json for a single HPO example.

Reads:
  <example>/best_loss.tadah   -- last row gives final per-component & total/N loss
  <example>/best_params.tadah -- last row gives final hyperparameter vector
  <example>/hpo_run.log       -- Best total loss, Total evaluations, Optimiser status

Tolerance policy (overridable per example via tolerance_overrides.json):
  best_loss:    abs 1e-6,  rel 1e-3
  components:   abs 1e-6,  rel 5e-3
  hyperparams:  abs 1e-3,  rel 1e-3
  evals:        exact (tol 0) by default

The TADAH HPO bundle is bit-for-bit deterministic when SEED is set
(verified for examples 03, 13, 15). The defaults absorb cross-machine
BLAS/LAPACK reorder and double-precision printf rounding while still
detecting a material regression in the optimiser path. Override per
example when a wider tolerance is justified (stochastic-without-seed,
adaptive eval-count, etc.).
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

DEFAULT_TOL = {
    "best_loss":  {"abs_tol": 1e-6, "rel_tol": 1e-3},
    "component":  {"abs_tol": 1e-6, "rel_tol": 5e-3},
    "hyperparam": {"abs_tol": 1e-3, "rel_tol": 1e-3},
    "evals_tol":  0,
}


def parse_tabular_last_row(path: Path):
    """Parse files like best_loss.tadah / best_params.tadah.

    First non-comment, non-blank line is taken as a header (after stripping
    leading '#'). Subsequent rows are values. Returns (header, last_values).
    """
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
        raise RuntimeError(f"{path}: could not parse header/values")
    return header, last


def parse_log(path: Path):
    txt = path.read_text()
    out = {}

    m = re.search(r"Best total loss\s*:\s*([0-9.eE+-]+)", txt)
    out["best_total_loss"] = float(m.group(1)) if m else None

    m = re.search(r"Total evaluations\s*:\s*([0-9]+)", txt)
    out["total_evaluations"] = int(m.group(1)) if m else None

    m = re.search(r"Optimiser status\s*:\s*(.+)", txt)
    out["optimiser_status"] = m.group(1).strip() if m else None

    m = re.search(r"Best ERMSE\s*:\s*([0-9.eE+-]+)", txt)
    out["best_ermse_meV_per_atom"] = float(m.group(1)) if m else None

    m = re.search(r"Total wall time\s*:\s*(.+)", txt)
    out["total_walltime_str"] = m.group(1).strip() if m else None

    m = re.search(r"NaN/Inf clampings\s*:\s*([0-9]+)", txt)
    out["nan_clampings"] = int(m.group(1)) if m else None

    return out


def build_fingerprint(example_dir: Path, overrides: dict) -> dict:
    name = example_dir.name

    loss_hdr, loss_row = parse_tabular_last_row(example_dir / "best_loss.tadah")
    params_hdr, params_row = parse_tabular_last_row(example_dir / "best_params.tadah")
    log = parse_log(example_dir / "hpo_run.log")

    # The hpo_run.log "Best total loss" field is truncated to ~5 sig-figs.
    # Prefer the full-precision TOTAL_LOSS/N column from best_loss.tadah.
    if "TOTAL_LOSS/N" in loss_hdr:
        idx = loss_hdr.index("TOTAL_LOSS/N")
        try:
            log["best_total_loss"] = float(loss_row[idx])
        except (ValueError, IndexError):
            pass

    tol_loss = {**DEFAULT_TOL["best_loss"], **(overrides.get("best_loss_tol") or {})}
    tol_comp = {**DEFAULT_TOL["component"], **(overrides.get("component_tol") or {})}
    tol_hp   = {**DEFAULT_TOL["hyperparam"], **(overrides.get("hyperparam_tol") or {})}
    evals_tol = overrides.get("evals_tol", DEFAULT_TOL["evals_tol"])

    # best_loss columns: skip leading "ITERATION", trailing "TOTAL_LOSS/N"
    components = {}
    for i, col in enumerate(loss_hdr):
        if col in ("ITERATION", "TOTAL_LOSS/N"):
            continue
        try:
            components[col] = {
                "value": float(loss_row[i]),
                "abs_tol": tol_comp["abs_tol"],
                "rel_tol": tol_comp["rel_tol"],
            }
        except (ValueError, IndexError):
            continue

    # best_params columns: skip leading ITERATION
    hyperparams = {}
    for i, col in enumerate(params_hdr):
        if col == "ITERATION":
            continue
        try:
            hyperparams[col] = {
                "value": float(params_row[i]),
                "abs_tol": tol_hp["abs_tol"],
                "rel_tol": tol_hp["rel_tol"],
            }
        except (ValueError, IndexError):
            continue

    return {
        "schema_version": 1,
        "example": name,
        "description": overrides.get("description", ""),
        "best_loss": {
            "value": log["best_total_loss"],
            "abs_tol": tol_loss["abs_tol"],
            "rel_tol": tol_loss["rel_tol"],
        },
        "components": components,
        "hyperparameters": hyperparams,
        "evaluations": {
            "value": log["total_evaluations"],
            "tol": evals_tol,
        },
        "best_ermse_meV_per_atom": log.get("best_ermse_meV_per_atom"),
        "status_accepted": overrides.get(
            "status_accepted", [log["optimiser_status"]]
        ),
        "nan_clampings_max": overrides.get("nan_clampings_max", 0),
        "walltime_hint": log.get("total_walltime_str"),
        "fast": overrides.get("fast", True),
        "tolerance_rationale": overrides.get(
            "tolerance_rationale",
            "Bundle is bit-deterministic when SEED is set; tolerances absorb "
            "cross-machine BLAS/LAPACK reorder and FP printf rounding while "
            "still flagging a material regression."
        ),
    }


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("example_dir", type=Path)
    p.add_argument("--overrides", type=Path, default=None,
                   help="JSON file with per-example overrides")
    p.add_argument("--out", type=Path, default=None,
                   help="Write fingerprint here (default: <dir>/reference_fingerprint.json)")
    args = p.parse_args(argv)

    overrides = {}
    if args.overrides and args.overrides.exists():
        all_over = json.loads(args.overrides.read_text())
        overrides = all_over.get(args.example_dir.name, {})

    fp = build_fingerprint(args.example_dir, overrides)
    out_path = args.out or (args.example_dir / "reference_fingerprint.json")
    out_path.write_text(json.dumps(fp, indent=2) + "\n")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
