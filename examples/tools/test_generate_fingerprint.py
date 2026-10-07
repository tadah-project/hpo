#!/usr/bin/env python3
"""Unit tests of generate_fingerprint.py: the fields a fingerprint takes from
its tolerance_overrides.json entry, as build_fingerprint() writes them,
carry_over() keeps them on a regeneration and override_mismatches() checks
them (the CTest hpo_example_fingerprint_overrides_sync).

Each case builds a fingerprint from a synthetic example run (best_loss.tadah,
best_params.tadah, hpo_run.log in a temporary directory).
"""
from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import generate_fingerprint as gf  # noqa: E402


def write_run(d: Path, status: str = "COMPLETED") -> None:
    (d / "best_loss.tadah").write_text(
        "# ITERATION L(ERMSE) TOTAL_LOSS/N\n1 0.5 0.5\n")
    (d / "best_params.tadah").write_text("# ITERATION RCUT2B\n1 5.0\n")
    (d / "hpo_run.log").write_text(
        "Best total loss : 0.5\n"
        "Total evaluations : 10\n"
        f"Optimiser status : {status}\n")


class FingerprintOverrides(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self._tmp.name) / "99_example"
        self.dir.mkdir()
        write_run(self.dir)

    def tearDown(self):
        self._tmp.cleanup()

    def regenerate(self, overrides: dict, prev: dict | None = None) -> dict:
        fp = gf.build_fingerprint(self.dir, overrides)
        return gf.carry_over(fp, prev) if prev is not None else fp

    # A regeneration passes the check, whatever the entry holds.

    def test_regenerated_fingerprint_passes(self):
        for overrides in ({},
                          {"hyperparam_tol": {"rel_tol": 0.01, "note": "x"}},
                          {"journal": []},
                          {"description": ""},
                          {"status_accepted": ["COMPLETED", "MAXEVAL_REACHED"]},
                          {"journal": [{"regex": "^x", "why": "y"}]},
                          {"threw_max": 2}):
            with self.subTest(overrides=overrides):
                prev = {"description": "hand-written", "status_note": "kept",
                        "journal": [{"regex": "^old", "why": "old"}]}
                fp = self.regenerate(overrides, prev)
                self.assertEqual(gf.override_mismatches(fp, overrides), [])

    # A fingerprint left behind by an overrides edit fails the check.

    def test_tolerance_drift_is_detected(self):
        fp = self.regenerate({})
        problems = gf.override_mismatches(fp, {"component_tol": {"rel_tol": 0.02}})
        self.assertTrue(any("components" in p for p in problems), problems)

    def test_removed_status_accepted_is_detected(self):
        fp = self.regenerate({"status_accepted": ["COMPLETED", "MAXEVAL_REACHED"]})
        problems = gf.override_mismatches(fp, {})
        self.assertTrue(any("status_accepted" in p for p in problems), problems)

    def test_removed_journal_is_detected_and_dropped(self):
        journal = [{"regex": "^x", "why": "y"}]
        fp = self.regenerate({"journal": journal})
        problems = gf.override_mismatches(fp, {})
        self.assertTrue(any("journal" in p for p in problems), problems)
        # Regenerating with the entry's journal removed drops it.
        again = self.regenerate({}, prev=fp)
        self.assertNotIn("journal", again)

    def test_threw_max_comes_from_the_overrides(self):
        # Pins: a fingerprint records threw_max, the evaluations that threw
        # compare_fingerprint.py allows, from the overrides entry (0 by
        # default, as nan_clampings_max), and the check reports a
        # fingerprint whose threw_max differs from its entry. Before the
        # fix: no fingerprint had a threw_max (KeyError), and nothing
        # compared one.
        self.assertEqual(self.regenerate({})["threw_max"], 0)
        self.assertEqual(self.regenerate({"threw_max": 3})["threw_max"], 3)
        problems = gf.override_mismatches(self.regenerate({}), {"threw_max": 1})
        self.assertTrue(any("threw_max" in p for p in problems), problems)

    def test_hand_written_fields_are_kept(self):
        prev = {"description": "hand-written", "status_note": "kept"}
        fp = self.regenerate({}, prev)
        self.assertEqual(fp["description"], "hand-written")
        self.assertEqual(fp["status_note"], "kept")


if __name__ == "__main__":
    unittest.main()
