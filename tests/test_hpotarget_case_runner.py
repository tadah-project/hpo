#!/usr/bin/env python3
"""Tests of hpotarget_case_runner (tests/tools/hpotarget_case_runner.cpp),
which HPOTARGET_CASE_RUNNER names (CTest sets it): the parser selector
--parser old|new, the mode "describe", which reports what the parse set
(tadah::hpo::testing::describe_parse), and the cases that read a target
file and a training configuration in place, in a copy of a directory tree
(tree, cwd, target_file, config_file, config_drop, config_add, copies).

Before the fix: the runner takes no option and no mode "describe", and
every test fails."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_hpotarget_generated as gen  # noqa: E402

CONFIG = """ATOM H
WATOM 1.0
DBFILE dummy.db
INIT2B true
INITMB false
TYPE2B D2_BP 2 2 H H
CGRID2B 1.0 2.0
SGRID2B 0.5 1.0
RCTYPE2B Cut_Cos
RCUT2B 5.0
VERBOSE 0
MODEL M_BLR BF_Linear
OALGO 4
LAMBDA 1e-3
"""
TARGET = """OPTIM CGRID2B (1) 0.5 1.5
PC_ERMSE 0 1
OPTIMIZER
STRATEGY
LIB NLOPT
ALGO LN_SBPLX
MAXEVAL 10
ENDSTRATEGY
ENDOPTIMIZER
"""


def runner() -> pathlib.Path:
    r = os.environ.get("HPOTARGET_CASE_RUNNER", "")
    if not r:
        raise unittest.SkipTest("HPOTARGET_CASE_RUNNER is not set")
    return pathlib.Path(r)


def toml_value(v) -> str:
    if isinstance(v, str):
        return gen.toml_string(v)
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{gen.toml_string(k)} = {toml_value(x)}"
                                for k, x in v.items()) + " }"
    raise TypeError(type(v))


class Runner(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def run_cases(self, cases: list[dict], *options: str
                  ) -> tuple[int, str, list[dict] | None]:
        text = []
        for c in cases:
            text.append("[[case]]")
            text += [f"{k} = {toml_value(v)}" for k, v in c.items()]
            text.append("")
        cf, rf = self.dir / "cases.toml", self.dir / "results.json"
        cf.write_text("\n".join(text))
        if rf.exists():
            rf.unlink()
        p = subprocess.run([str(runner()), *options, str(cf), str(rf)],
                           capture_output=True, text=True)
        res = json.loads(rf.read_text()) if rf.exists() else None
        return p.returncode, p.stderr, res

    def test_parser_old_is_the_default(self):
        cases = [{"id": "a", "mode": "parse", "file": TARGET,
                  "config": CONFIG},
                 {"id": "b", "mode": "parse", "file": "FOO 1\n" + TARGET,
                  "config": CONFIG}]
        rc0, err0, res0 = self.run_cases(cases)
        rc1, err1, res1 = self.run_cases(cases, "--parser", "old")
        self.assertEqual((rc0, rc1), (0, 0), err0 + err1)
        self.assertEqual(res0, res1)
        self.assertEqual(res1[0]["parse_error"], "")
        self.assertIn("unknown key 'FOO'", res1[1]["parse_error"])

    def test_parser_new_is_refused_until_it_exists(self):
        rc, err, res = self.run_cases(
            [{"id": "a", "mode": "parse", "file": TARGET, "config": CONFIG}],
            "--parser", "new")
        self.assertEqual(rc, 2)
        self.assertIsNone(res)
        self.assertIn("--parser new: this build has no registry-driven "
                      "parser of the target file", err)

    def test_parser_takes_old_or_new(self):
        case = [{"id": "a", "mode": "parse", "file": TARGET,
                 "config": CONFIG}]
        rc, err, res = self.run_cases(case, "--parser", "newest")
        self.assertEqual(rc, 2)
        self.assertIsNone(res)
        self.assertIn("--parser takes old or new, not 'newest'", err)
        self.assertIn("usage: hpotarget_case_runner [--parser old|new] "
                      "CASES RESULTS", err)
        rc, err, _ = self.run_cases(case, "--frobnicate")
        self.assertEqual(rc, 2)
        self.assertIn("unknown option '--frobnicate'", err)
        p = subprocess.run([str(runner()), "--parser"], capture_output=True,
                           text=True)
        self.assertEqual(p.returncode, 2)
        self.assertIn("--parser takes old or new", p.stderr)

    def test_describe_reports_the_parse(self):
        rc, err, res = self.run_cases(
            [{"id": "ok", "mode": "describe", "file": TARGET,
              "config": CONFIG},
             {"id": "bad", "mode": "describe", "file": "FOO 1\n" + TARGET,
              "config": CONFIG},
             {"id": "parse", "mode": "parse", "file": TARGET,
              "config": CONFIG}], "--parser", "old")
        self.assertEqual(rc, 0, err)
        ok, bad, parse = res
        self.assertEqual(ok["parse_error"], "")
        self.assertTrue(ok["dump"].startswith('target_file = "target.hpo"\n'),
                        ok["dump"][:200])
        self.assertIn('\nspec.algo = "LN_SBPLX"\n', ok["dump"])
        self.assertIn("unknown key 'FOO'", bad["parse_error"])
        self.assertEqual(bad["dump"], "")
        self.assertEqual(parse["dump"], "")

    def test_files_read_in_place(self):
        tree = self.dir / "tree"
        (tree / "work").mkdir(parents=True)
        (tree / "work" / "original.hpo").write_text(TARGET)
        (tree / "common").mkdir()
        (tree / "common" / "train.tadah").write_text(
            CONFIG.replace("RCUT2B 5.0\n", "RCUT2B 5.0\nNUMERIC 12\n"))
        case = {"id": "a", "mode": "describe", "tree": str(tree),
                "cwd": "work", "target_file": "t.hpo",
                "config_file": "../common/train.tadah",
                "config_drop": ["NUMERIC"],
                "config_add": {"NUMERIC": ["10"]},
                "copies": [{"from": "original.hpo", "to": "t.hpo"}]}
        rc, err, res = self.run_cases([case])
        self.assertEqual(rc, 0, err)
        r = res[0]
        self.assertEqual(r["parse_error"], "")
        d = r["dump"]
        self.assertTrue(d.startswith('target_file = "t.hpo"\n'), d[:200])
        self.assertIn('\ncontext[NUMERIC] = ["10"]\n', d)
        # The tree is copied: the source is not written.
        self.assertFalse((tree / "work" / "t.hpo").exists())

    def test_inconsistent_cases_are_refused(self):
        rc, err, _ = self.run_cases(
            [{"id": "a", "mode": "parse", "file": TARGET,
              "target_file": "t.hpo", "config": CONFIG}])
        self.assertEqual(rc, 2)
        self.assertIn("file and target_file are both given", err)
        rc, err, _ = self.run_cases(
            [{"id": "a", "mode": "parse", "file": TARGET,
              "config": CONFIG, "config_file": "c.tadah"}])
        self.assertEqual(rc, 2)
        self.assertIn("config and config_file are both given", err)
        rc, err, _ = self.run_cases(
            [{"id": "a", "mode": "check", "file": TARGET,
              "config_file": "c.tadah"}])
        self.assertEqual(rc, 2)
        self.assertIn("mode check takes the target file and the training "
                      "configuration as text", err)
        rc, err, _ = self.run_cases(
            [{"id": "a", "mode": "describe", "file": TARGET,
              "config": CONFIG, "cwd": "nowhere"}])
        self.assertEqual(rc, 2)
        self.assertIn("the directory nowhere of the case does not exist", err)


if __name__ == "__main__":
    unittest.main()
