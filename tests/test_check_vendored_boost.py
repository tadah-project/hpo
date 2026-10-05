#!/usr/bin/env python3
"""Unit tests of check_vendored_boost.py (CTest hpo_vendored_boost_gate_tool).

The gate must follow the includes of the Boost headers HPO uses, as the
compiler resolves them, and fail when one is missing from the vendored tree
or comes from elsewhere (a system Boost). Fake trees in a temporary directory:
  src/use.cpp            #include <boost/a.hpp>
  vendored/boost/a.hpp   #include <boost/b.hpp>
  vendored/boost/b.hpp   (only in the complete tree)
  system/boost/b.hpp     (a 'system' Boost, on CPATH)
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
GATE = HERE / "check_vendored_boost.py"
CXX = os.environ.get("CXX", "c++")


class VendoredBoostGate(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        (root / "hpo" / "src").mkdir(parents=True)
        (root / "hpo" / "src" / "use.cpp").write_text("#include <boost/a.hpp>\n")
        (root / "vendored" / "boost").mkdir(parents=True)
        (root / "vendored" / "boost" / "a.hpp").write_text(
            "#pragma once\n#include <boost/b.hpp>\n")
        (root / "system" / "boost").mkdir(parents=True)
        (root / "system" / "boost" / "b.hpp").write_text("#pragma once\n")
        self.root = root

    def tearDown(self):
        self._tmp.cleanup()

    def run_gate(self, cpath=None):
        env = dict(os.environ)
        env.pop("CPATH", None)
        if cpath:
            env["CPATH"] = str(cpath)
        return subprocess.run(
            [sys.executable, str(GATE), str(self.root / "hpo"),
             str(self.root / "vendored"), "--cxx", CXX],
            capture_output=True, text=True, env=env)

    def test_complete_tree_passes(self):
        (self.root / "vendored" / "boost" / "b.hpp").write_text("#pragma once\n")
        r = self.run_gate()
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    def test_missing_transitive_header_fails(self):
        r = self.run_gate()
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("boost/a.hpp", r.stdout)

    def test_transitive_header_from_a_system_boost_fails(self):
        r = self.run_gate(cpath=self.root / "system")
        self.assertNotEqual(r.returncode, 0, r.stdout)
        self.assertIn("b.hpp", r.stdout)


if __name__ == "__main__":
    unittest.main()
