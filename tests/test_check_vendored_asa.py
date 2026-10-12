#!/usr/bin/env python3
"""Unit tests of check_vendored_asa.py (CTest hpo_vendored_asa_gate_tool).

The gate must fail when an upstream file of the vendored ASA loses its CRLF
line endings, whole or in part, and its git leg must fail when nothing keeps
git from converting them and pass when a '-text' attribute does. Fake HPO
trees in a temporary directory: external/asa/ with the six upstream files,
each a few CRLF lines. The git cases need the git that $GIT names (the CTest
entry sets it to CMake's GIT_EXECUTABLE) and are skipped without it; the
entry then reports Skipped.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "check_vendored_asa.py"
UPSTREAM = ["asa.c", "asa.h", "asa_usr_asa.h",
            "ASA-LICENSE", "ASA-NOTES", "ASA-README.txt"]


def git_executable():
    git = os.environ.get("GIT") or shutil.which("git")
    if not git:
        return None
    try:
        subprocess.run([git, "--version"], capture_output=True, check=True)
    except (OSError, subprocess.CalledProcessError):
        return None
    return git


GIT = git_executable()


class VendoredAsaGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.hpo = Path(self._tmp.name) / "hpo"
        self.asa = self.hpo / "external" / "asa"
        self.asa.mkdir(parents=True)
        for name in UPSTREAM:
            (self.asa / name).write_bytes(
                f"/* {name} */\r\nint x;\r\n\r\nint y;\r\n".encode())
        # A TADAH file next to them, LF as the rest of HPO: not checked.
        (self.asa / "asa_build.c").write_bytes(b'#include "asa.c"\n')

    def tearDown(self):
        self._tmp.cleanup()

    def run_gate(self, *extra):
        return subprocess.run(
            [sys.executable, str(GATE), str(self.hpo), *extra],
            capture_output=True, text=True)

    def test_crlf_tree_passes(self):
        r = self.run_gate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_lf_file_fails(self):
        p = self.asa / "asa.c"
        p.write_bytes(p.read_bytes().replace(b"\r\n", b"\n"))
        r = self.run_gate()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("asa.c: 4 line(s) end in LF alone, the first line 1",
                      r.stdout)

    def test_one_lf_line_fails(self):
        p = self.asa / "asa.h"
        p.write_bytes(b"/* asa.h */\r\nint x;\n\r\n")
        r = self.run_gate()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("asa.h: 1 line(s) end in LF alone, the first line 2",
                      r.stdout)

    def test_missing_final_crlf_fails(self):
        p = self.asa / "ASA-NOTES"
        p.write_bytes(b"notes\r\nlast")
        r = self.run_gate()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("ASA-NOTES: the last line has no CRLF", r.stdout)

    def test_lone_cr_fails(self):
        p = self.asa / "ASA-LICENSE"
        p.write_bytes(b"one\rtwo\r\n")
        r = self.run_gate()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("ASA-LICENSE: 1 line(s) hold a CR that ends no line",
                      r.stdout)

    def test_missing_file_fails(self):
        (self.asa / "asa_usr_asa.h").unlink()
        r = self.run_gate()
        self.assertEqual(r.returncode, 1, r.stdout)
        self.assertIn("asa_usr_asa.h: missing", r.stdout)

    def test_no_git_is_a_skip(self):
        r = self.run_gate("--git=")
        self.assertEqual(r.returncode, 77, r.stdout)

    @unittest.skipUnless(GIT, "no git: set GIT to a git executable")
    def test_git_leg_fails_without_attributes(self):
        r = self.run_gate(f"--git={GIT}")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        for setting in ("core.autocrlf=input", "core.autocrlf=true",
                        "core.attributesFile with '* text eol=lf'"):
            self.assertIn("external/asa/asa.c: `git add --renormalize` under "
                          + setting + " changes its bytes", r.stdout)
        self.assertIn("attributes read: none", r.stdout)

    @unittest.skipUnless(GIT, "no git: set GIT to a git executable")
    def test_git_leg_passes_with_text_unset(self):
        (self.hpo / ".gitattributes").write_text(
            "".join(f"external/asa/{n} -text\n" for n in UPSTREAM))
        r = self.run_gate(f"--git={GIT}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    @unittest.skipUnless(GIT, "no git: set GIT to a git executable")
    def test_git_leg_reads_a_nested_gitattributes(self):
        (self.asa / ".gitattributes").write_text(
            "".join(f"{n} -text\n" for n in UPSTREAM))
        r = self.run_gate(f"--git={GIT}")
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    @unittest.skipUnless(GIT, "no git: set GIT to a git executable")
    def test_git_leg_fails_for_a_file_left_out(self):
        (self.hpo / ".gitattributes").write_text(
            "".join(f"external/asa/{n} -text\n" for n in UPSTREAM
                    if n != "ASA-README.txt"))
        r = self.run_gate(f"--git={GIT}")
        self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
        self.assertIn("external/asa/ASA-README.txt:", r.stdout)
        self.assertNotIn("external/asa/asa.c:", r.stdout)


if __name__ == "__main__":
    unittest.main()
