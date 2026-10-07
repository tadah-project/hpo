#!/usr/bin/env python3
"""The example scripts print a run's status with journal_status
(common/journal_status.sh): the whole "Optimiser status" value of
hpo_run.log, which contains colons ("TADAH:RANDOM: MAXEVAL reached",
"ASA: MAXEVAL reached", "STOPPED -- stagnation: no new best in ...").

Pins: journal_status prints each of those statuses whole, and no example
script extracts the status itself (they all call journal_status).
Before the fix: the scripts (common/run.sh, run_all.sh, and the run.sh of
16, 17 and 27) each cut the value at its first colon (awk -F': *'),
printing "TADAH", "ASA" or "STOPPED -- stagnation"; journal_status was
that extraction.

The CTest hpo_example_journal_status (HPO/tests/CMakeLists.txt) runs it.
"""
from __future__ import annotations

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

EXAMPLES = Path(__file__).resolve().parent.parent
HELPER = EXAMPLES / "common" / "journal_status.sh"

STATUSES = [
    "TADAH:RANDOM: MAXEVAL reached",
    "ASA: MAXEVAL reached",
    "DLIB:BFGS: FTOL_ABS reached",
    "STOPPED -- stagnation: no new best in 3 evals after 2 restart(s)",
    "MAXEVAL_REACHED -- maximum evaluations exhausted",
    "FAILED -- best-potential persistence: cannot fsync: No space left",
]


def journal(status: str) -> str:
    """The end of an hpo_run.log, its RESULT section as HPOLog writes it."""
    return ("-- RESULT " + "-" * 62 + "\n"
            f"  {'Optimiser status':<24}: {status}\n"
            f"  {'Best total loss':<24}: 1.0000e-01  (eval 7)\n")


def journal_status(log: Path) -> str:
    out = subprocess.run(
        ["bash", "-c", 'source "$1" && journal_status "$2"', "bash",
         str(HELPER), str(log)],
        check=True, capture_output=True, text=True)
    return out.stdout.rstrip("\n")


class JournalStatus(unittest.TestCase):
    def test_prints_the_whole_status(self):
        with tempfile.TemporaryDirectory() as d:
            log = Path(d) / "hpo_run.log"
            for status in STATUSES:
                with self.subTest(status=status):
                    log.write_text(journal(status))
                    self.assertEqual(journal_status(log), status)

    def test_every_script_uses_the_helper(self):
        """No example script reads the status line itself."""
        offenders = []
        for script in sorted(EXAMPLES.rglob("*.sh")):
            if script == HELPER:
                continue
            if "Optimiser status" in script.read_text():
                offenders.append(str(script.relative_to(EXAMPLES)))
        self.assertEqual(offenders, [],
                         "scripts that extract the status themselves")


if __name__ == "__main__":
    sys.exit(unittest.main())
