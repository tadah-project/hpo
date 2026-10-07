#!/usr/bin/env python3
"""Unit tests of compare_fingerprint.py's failure-count guards: the
DIAGNOSTICS counters of hpo_run.log, "NaN/Inf clampings" and "Evaluations
that threw", checked against the fingerprint's nan_clampings_max and
threw_max.

Pins: a run with evaluations that threw fails its fingerprint unless
threw_max allows them ("threw: actual=2 > max=0"; a fingerprint without
threw_max allows none), as a run with NaN/Inf clampings fails unless
nan_clampings_max allows them; a journal without either counter line fails,
since its guard cannot be applied; a run with neither passes.
Before the fix: only the NaN/Inf counter was read, so a run with 2
evaluations that threw passed (an evaluation that threw counted as a
NaN/Inf clamping until HPOLog counted it apart), and a journal without a
counter line passed with its guard left out. The clean run and the NaN/Inf
cases pass as well (pins).

Each case writes a synthetic example run (reference_fingerprint.json,
best_loss.tadah, best_params.tadah, hpo_run.log) in a temporary directory.
The CTest hpo_example_fingerprint_compare (HPO/tests/CMakeLists.txt) runs it.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import compare_fingerprint as cf  # noqa: E402

STATUS = "TADAH:RANDOM: MAXEVAL reached"


def journal(nan: int | None, threw: int | None, total: int = 10) -> str:
    """The DIAGNOSTICS and RESULT lines of an hpo_run.log as HPOLog writes
    them; a counter given as None is left out."""
    lines = ["-- DIAGNOSTICS " + "-" * 57,
             f"  {'Total evaluations':<24}: {total}"]
    if nan is not None:
        lines.append(f"  {'NaN/Inf clampings':<24}: {nan}  ({100.0 * nan / total:.1f}%)")
    if threw is not None:
        lines.append(f"  {'Evaluations that threw':<24}: {threw}  "
                     f"({100.0 * threw / total:.1f}%)")
    lines += ["-- RESULT " + "-" * 62,
              f"  {'Optimiser status':<24}: {STATUS}"]
    return "\n".join(lines) + "\n"


def fingerprint(**maxima) -> dict:
    fp = {
        "schema_version": 1,
        "example": "99_example",
        "best_loss": {"value": 0.5, "abs_tol": 1e-6, "rel_tol": 1e-3},
        "components": {},
        "hyperparameters": {},
        "evaluations": {"value": 10, "tol": 0},
        "status_accepted": [STATUS],
    }
    fp.update(maxima)
    return fp


class FailureCountGuards(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "99_example"
        self.dir.mkdir()
        (self.dir / "best_loss.tadah").write_text(
            "# ITERATION TOTAL_LOSS/N\n1 0.5\n")
        (self.dir / "best_params.tadah").write_text("# ITERATION RCUT2B\n1 5.0\n")

    def tearDown(self):
        self._tmp.cleanup()

    def compare(self, fp: dict, log: str):
        (self.dir / "reference_fingerprint.json").write_text(json.dumps(fp))
        (self.dir / "hpo_run.log").write_text(log)
        code, failures, _notes = cf.compare(self.dir)
        return code, failures

    def test_clean_run_passes(self):
        code, failures = self.compare(
            fingerprint(nan_clampings_max=0, threw_max=0), journal(0, 0))
        self.assertEqual((code, failures), (0, []))

    def test_evaluations_that_threw_fail(self):
        for fp in (fingerprint(nan_clampings_max=0, threw_max=0),
                   fingerprint(nan_clampings_max=0)):
            with self.subTest(fp=fp):
                code, failures = self.compare(fp, journal(0, 2))
                self.assertEqual(code, 1)
                self.assertIn("threw: actual=2 > max=0", failures)

    def test_evaluations_that_threw_within_threw_max_pass(self):
        code, failures = self.compare(
            fingerprint(nan_clampings_max=0, threw_max=2), journal(0, 2))
        self.assertEqual((code, failures), (0, []))

    def test_nan_clampings_fail(self):
        code, failures = self.compare(
            fingerprint(nan_clampings_max=0, threw_max=0), journal(2, 0))
        self.assertEqual(code, 1)
        self.assertIn("nan_clampings: actual=2 > max=0", failures)

    def test_a_missing_counter_line_fails(self):
        for log, label in ((journal(0, None), "Evaluations that threw"),
                           (journal(None, 0), "NaN/Inf clampings")):
            with self.subTest(label=label):
                code, failures = self.compare(
                    fingerprint(nan_clampings_max=0, threw_max=0), log)
                self.assertEqual(code, 1)
                self.assertTrue(any(label in f and "no hpo_run.log line" in f
                                    for f in failures), failures)


if __name__ == "__main__":
    unittest.main()
