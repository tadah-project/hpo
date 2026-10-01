#!/usr/bin/env python3
"""Generate reference_fingerprint.json for a single HPO example.

Reads:
  <example>/best_loss.tadah   -- last row gives final per-component & total/N loss
  <example>/best_params.tadah -- last row gives final hyperparameter vector
  <example>/hpo_run.log       -- Best total loss, Total evaluations, Optimiser status

The optional "journal" checks (lines hpo_run.log must contain; see
compare_fingerprint.py) come from the example's entry in
tolerance_overrides.json, so a regeneration keeps them.

Tolerance policy (overridable per example via tolerance_overrides.json):
  best_loss:    abs 1e-6,  rel 1e-3
  components:   abs 1e-6,  rel 5e-3
  hyperparams:  abs 1e-3,  rel 1e-3
  evals:        exact (tol 0) by default

Examples that set SEED repeat run to run on one machine. 13_restart does
not: RESTART RANDOM reseeds from std::random_device, so it pins only
values that do not depend on the restarts. The defaults absorb
cross-machine BLAS/LAPACK reorder and double-precision printf rounding
while still detecting a material regression in the optimiser path -- but
a few examples react to last-ulp changes of the LAMMPS read-back (05's
evaluation count, which evaluation is best along 16's flat FIXWEIGHT(14)),
and their overrides in tolerance_overrides.json say so. Override per
example when a wider tolerance is justified (stochastic-without-seed,
adaptive eval-count, rounding-decided stops or ties, etc.).
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

DEFAULT_RATIONALE = (
    "Bundle is bit-deterministic when SEED is set; tolerances absorb "
    "cross-machine BLAS/LAPACK reorder and FP printf rounding while "
    "still flagging a material regression."
)


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
        # SLOT is the replica/slot bookkeeping column (D19), not a loss term.
        if col in ("ITERATION", "SLOT", "TOTAL_LOSS/N"):
            continue
        try:
            components[col] = {
                "value": float(loss_row[i]),
                "abs_tol": tol_comp["abs_tol"],
                "rel_tol": tol_comp["rel_tol"],
            }
        except (ValueError, IndexError):
            continue

    # best_params columns: skip leading ITERATION. Repeated column names
    # (e.g. OPTIM FIXWEIGHT (2-4) -> three FIXWEIGHT columns) get a
    # 1-based [k] suffix; MUST mirror compare_fingerprint.py, which used
    # to silently compare the FIRST duplicate against the LAST one
    # stored here (dict collision) -- a flat-direction change in any
    # other duplicate column then looked like a regression.
    hyperparams = {}
    seen = {}
    for i, col in enumerate(params_hdr):
        if col in ("ITERATION", "SLOT"):
            continue
        seen[col] = seen.get(col, 0) + 1
        key = col if params_hdr.count(col) == 1 else f"{col}[{seen[col]}]"
        try:
            hyperparams[key] = {
                "value": float(params_row[i]),
                "abs_tol": tol_hp["abs_tol"],
                "rel_tol": tol_hp["rel_tol"],
            }
        except (ValueError, IndexError):
            continue

    fp = {
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
            "tolerance_rationale", DEFAULT_RATIONALE),
    }
    if overrides.get("journal"):
        fp["journal"] = overrides["journal"]
    return fp


def override_mismatches(fp: dict, overrides: dict) -> list:
    """The fields build_fingerprint() takes from an example's overrides
    entry that `fp` does not hold as the entry gives them: an overrides
    entry changed without regenerating the fingerprint, or a fingerprint
    edited by hand. Fields a regeneration keeps from the old fingerprint
    when the entry has none (description, journal) are compared only when
    the entry sets them."""
    out = []

    def cmp(field, have, want):
        if have != want:
            out.append(f"{field}: fingerprint {have!r}, overrides {want!r}")

    tol_loss = {**DEFAULT_TOL["best_loss"], **(overrides.get("best_loss_tol") or {})}
    tol_comp = {**DEFAULT_TOL["component"], **(overrides.get("component_tol") or {})}
    tol_hp   = {**DEFAULT_TOL["hyperparam"], **(overrides.get("hyperparam_tol") or {})}
    bl = fp.get("best_loss") or {}
    cmp("best_loss tolerances", {k: bl.get(k) for k in tol_loss}, tol_loss)
    for name, c in (fp.get("components") or {}).items():
        cmp(f"components.{name} tolerances", {k: c.get(k) for k in tol_comp}, tol_comp)
    for name, h in (fp.get("hyperparameters") or {}).items():
        cmp(f"hyperparameters.{name} tolerances", {k: h.get(k) for k in tol_hp}, tol_hp)
    cmp("evaluations.tol", (fp.get("evaluations") or {}).get("tol"),
        overrides.get("evals_tol", DEFAULT_TOL["evals_tol"]))
    for field in ("description", "status_accepted", "journal"):
        if field in overrides:
            cmp(field, fp.get(field), overrides[field])
    cmp("nan_clampings_max", fp.get("nan_clampings_max"),
        overrides.get("nan_clampings_max", 0))
    cmp("fast", fp.get("fast"), overrides.get("fast", True))
    cmp("tolerance_rationale", fp.get("tolerance_rationale"),
        overrides.get("tolerance_rationale", DEFAULT_RATIONALE))
    return out


def check_all(examples_dir: Path, overrides_path: Path) -> int:
    """Checks every <examples_dir>/*/reference_fingerprint.json against
    the overrides file (and that every overrides entry has a fingerprint);
    prints each mismatch and returns the number of examples that have
    one."""
    all_over = json.loads(overrides_path.read_text())
    names = sorted(k for k in all_over if not k.startswith("_"))
    fps = sorted(examples_dir.glob("*/reference_fingerprint.json"))
    bad = 0
    for name in names:
        if not (examples_dir / name / "reference_fingerprint.json").exists():
            print(f"{name}: has an overrides entry but no reference_fingerprint.json")
            bad += 1
    for path in fps:
        name = path.parent.name
        fp = json.loads(path.read_text())
        problems = override_mismatches(fp, all_over.get(name, {}))
        if fp.get("example") != name:
            problems.insert(0, f"example: {fp.get('example')!r}, directory {name!r}")
        for problem in problems:
            print(f"{name}: {problem}")
        bad += bool(problems)
    print(f"{len(fps)} fingerprints checked against {overrides_path.name}: "
          f"{bad} with mismatches")
    return bad


def main(argv):
    p = argparse.ArgumentParser()
    p.add_argument("example_dir", type=Path,
                   help="the example to fingerprint; with --check, the "
                        "directory that holds the examples")
    p.add_argument("--overrides", type=Path, default=None,
                   help="JSON file with per-example overrides")
    p.add_argument("--out", type=Path, default=None,
                   help="Write fingerprint here (default: <dir>/reference_fingerprint.json)")
    p.add_argument("--check", action="store_true",
                   help="write nothing; exit 1 if any fingerprint's fields "
                        "that come from --overrides differ from it")
    args = p.parse_args(argv)

    if args.check:
        if args.overrides is None:
            p.error("--check needs --overrides")
        return 1 if check_all(args.example_dir, args.overrides) else 0

    overrides = {}
    if args.overrides and args.overrides.exists():
        all_over = json.loads(args.overrides.read_text())
        overrides = all_over.get(args.example_dir.name, {})

    fp = build_fingerprint(args.example_dir, overrides)
    out_path = args.out or (args.example_dir / "reference_fingerprint.json")
    # Hand-written fields of an existing fingerprint survive a regeneration:
    # the description (when the overrides file has none), the free-text note
    # on the evaluation count, and every top-level field this script does
    # not produce (e.g. a "status_note"), each kept after the field it
    # followed. Everything measured is refreshed.
    if out_path.exists():
        try:
            prev = json.loads(out_path.read_text())
        except (ValueError, OSError):
            prev = {}
        if not fp.get("description") and prev.get("description"):
            fp["description"] = prev["description"]
        note = (prev.get("evaluations") or {}).get("note")
        if note and "note" not in fp["evaluations"]:
            fp["evaluations"]["note"] = note
            old_evals = (prev.get("evaluations") or {}).get("value")
            if old_evals != fp["evaluations"]["value"]:
                print(f"note: evaluations changed {old_evals} -> "
                      f"{fp['evaluations']['value']}; check the carried-over "
                      f"evaluations.note", file=sys.stderr)
        fp = keep_hand_written(fp, prev)
    out_path.write_text(json.dumps(fp, indent=2) + "\n")
    print(f"wrote {out_path}")


def keep_hand_written(fp: dict, prev: dict) -> dict:
    """fp plus every top-level key of prev that fp lacks, each inserted
    after the key it followed in prev (at the front if it was first)."""
    out = dict(fp)
    order = list(prev)
    for i, key in enumerate(order):
        if key in out:
            continue
        anchor = next((k for k in reversed(order[:i]) if k in out), None)
        keys = list(out)
        pos = keys.index(anchor) + 1 if anchor is not None else 0
        items = list(out.items())
        items.insert(pos, (key, prev[key]))
        out = dict(items)
    return out


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
