#!/usr/bin/env python3
"""Unit tests of tests/hpotarget_corpus.py, the corpus of target files:
which strings of a test count as a target file, how the strings of a
Python and a C++ test are read without running them, how each source
pairs a target file with its training configuration (on synthetic trees),
and that the corpus of the real trees holds every target file of HPO's
tests/data, every example of the registry, the target files of the tests
of Tadah.MLIP and of its documentation. The real Tadah.MLIP tree is the
one TADAH_UMBRELLA_SOURCE_DIR names (CTest sets it in a build of
Tadah.MLIP); without it those cases are skipped.

Before the fix: hpotarget_corpus.py does not exist and the import fails."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import textwrap
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HPO / "blueprint"))

import hpotarget_corpus as corpus  # noqa: E402

HR, REG = corpus.load_registry(HPO / "blueprint")

TARGET = """# a comment
OPTIM CGRID2B (1) 0.5 1.5
PC_ERMSE 0 1
OPTIMIZER
  STRATEGY
    LIB NLOPT
    ALGO LN_SBPLX
    MAXEVAL 10
  ENDSTRATEGY
ENDOPTIMIZER
"""


def umbrella(test: unittest.TestCase) -> pathlib.Path:
    d = os.environ.get("TADAH_UMBRELLA_SOURCE_DIR", "")
    if not d:
        test.skipTest("TADAH_UMBRELLA_SOURCE_DIR is not set (a build "
                      "without Tadah.MLIP)")
    return pathlib.Path(d)


class LooksLikeTarget(unittest.TestCase):
    def check(self, text: str) -> bool:
        return corpus.looks_like_target(HR, REG, text)

    def test_a_target_file(self):
        self.assertTrue(self.check(TARGET))
        self.assertTrue(self.check("LOSS L2\nPC_ERMSE 0 1\n"))
        # INIT's STRATEGY is METHOD: it opens no block and takes values.
        self.assertTrue(self.check(
            "OPTIMIZER\nINIT\nSTRATEGY RANDOM 2\nENDINIT\nLIB NLOPT\n"
            "ALGO LN_SBPLX\nENDOPTIMIZER\n"))
        # A removed keyword is a line of a target file (refused).
        self.assertTrue(self.check("PARALLEL 2\nLOSS L2\n"))

    def test_not_a_target_file(self):
        # One line.
        self.assertFalse(self.check("LOSS L2\n"))
        # A key the registry does not know.
        self.assertFalse(self.check("LOSS L2\nFOO 1\n"))
        # A refit file: its OPTIMIZER block stands inside a stage block.
        self.assertFalse(self.check(
            "REFINE\n  OPTIMIZER\n    LIB CERES\n  ENDOPTIMIZER\nENDREFINE\n"))
        # Messages that start with a key.
        self.assertFalse(self.check(
            "OPTIMIZER block given more than once\nLine: OPTIMIZER\n"))
        self.assertFalse(self.check(
            "OPTIMIZER block must appear at the top level\nLOSS L2\n"))
        # A training configuration.
        self.assertFalse(self.check("ATOM Ta\nRCUT2B 5.0\n"))


class PythonStrings(unittest.TestCase):
    def strings(self, source: str):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "m.py"
            p.write_text(textwrap.dedent(source))
            return corpus.python_strings(p)

    def test_folding(self):
        whole, templates = self.strings('''
            OBJECTIVE = """LOSS L2
            QFLUSH 1
            """
            TEMPLATE = """OPTIMIZER
            LIB NLOPT
            ALGO LN_SBPLX
            MAXEVAL {maxeval}
            ENDOPTIMIZER
            """
            PLAIN = "LOSS L2\\nPC_ERMSE 0 1\\n"
            CASES = {"a": {"hpo": "OPTIMIZER\\n" + OBJECTIVE + "PC_ERMSE 0 1\\n"}}

            def f(work, n):
                work.write_text(TEMPLATE.format(maxeval=12))
                work.write_text(TEMPLATE.format(maxeval=n))
                work.write_text(PLAIN)
        ''')
        texts = [t for _, t in whole]
        self.assertIn("OPTIMIZER\nLOSS L2\nQFLUSH 1\nPC_ERMSE 0 1\n", texts)
        self.assertIn("OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 12\n"
                      "ENDOPTIMIZER\n", texts)
        self.assertIn("LOSS L2\nPC_ERMSE 0 1\n", texts)
        # The parts of other strings are not files of their own.
        self.assertNotIn("LOSS L2\nQFLUSH 1\n", texts)
        self.assertFalse(any("{maxeval}" in t for t in texts))
        # The template filled with a variable is named.
        self.assertEqual([line for line, _ in templates], [16])


class CppStrings(unittest.TestCase):
    def strings(self, source: str):
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "t.cpp"
            p.write_text(source)
            return corpus.cpp_strings(p)

    def test_runs(self):
        src = r'''
#include "catch2/catch.hpp"
int n = 1'000'000;
char q = '"';
void f(std::ostream& os, int k) {
    os << "OPTIMIZER\n"    // line 1
          "LIB NLOPT\n"    /* a "comment" */
          "ENDOPTIMIZER\n";
    os << "LOSS L2\n" << k << "PC_ERMSE 0 1\n";
    g("a" + std::string("b"), R"x(LOSS L2
PC_ERMSE 0 1
)x");
    h("tab\there", "\x41\101");
}
'''
        runs = self.strings(src)
        texts = [t for _, t in runs]
        self.assertIn("OPTIMIZER\nLIB NLOPT\nENDOPTIMIZER\n", texts)
        # A run followed by `<<` is a part.
        self.assertNotIn("LOSS L2\n", texts)
        # The last run of a stream expression is read.
        self.assertIn("PC_ERMSE 0 1\n", texts)
        self.assertIn("LOSS L2\nPC_ERMSE 0 1\n", texts)
        self.assertIn("tab\there", texts)
        self.assertIn("AA", texts)
        # Line numbers.
        self.assertIn((6, "OPTIMIZER\nLIB NLOPT\nENDOPTIMIZER\n"), runs)


class SyntheticTrees(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, rel: str, text: str = "") -> pathlib.Path:
        p = self.root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)
        return p

    def test_hpo_data(self):
        hpo = self.root / "HPO"
        self.write("HPO/tests/data/ta/target_files/a.hpo", TARGET)
        self.write("HPO/tests/data/ta/target_files/b.hpo", TARGET)
        self.write("HPO/tests/data/ta/target_files/b.tadah", "ATOM Ta\n")
        self.write("HPO/tests/data/ta/target_files/fixweight_no_dbfile.hpo",
                   TARGET)
        self.write("HPO/tests/data/other/c.hpo", TARGET)
        c = corpus.Corpus()
        corpus.read_hpo_data(c, hpo)
        by_id = {i.id: i.case for i in c.inputs}
        data = str(hpo / "tests" / "data" / "ta")
        self.assertEqual(by_id["hpo-data:ta/target_files/a.hpo"],
                         {"tree": data, "cwd": "target_files",
                          "target_file": "a.hpo",
                          "config_file": "../common/config.tadah"})
        self.assertEqual(
            by_id["hpo-data:ta/target_files/b.hpo"]["config_file"], "b.tadah")
        nodb = by_id["hpo-data:ta/target_files/fixweight_no_dbfile.hpo"]
        self.assertEqual(nodb["config_drop"], ["DBFILE"])
        self.assertEqual(nodb["config_add"]["FIXWEIGHT"], ["0"] * 14)
        self.assertEqual(len(c.problems), 1)
        self.assertIn("other/c.hpo has no training configuration",
                      c.problems[0])

    def test_docs(self):
        tm = self.root / "Tadah.MLIP"
        self.write("Tadah.MLIP/tests/doc_commands/doc_commands.py",
                   (corpus_doc_commands()).read_text())
        files = "Tadah.MLIP/docs/source/nfp/files"
        self.write(f"{files}/manifest.txt",
                   "# a comment\n"
                   "unit2 | nfp/unit2.rst | tutorial/unit2 | - | 60 | -\n")
        self.write("Tadah.MLIP/docs/source/nfp/unit2.rst", textwrap.dedent("""\
            Unit 2
            ======

            .. code-block:: bash

               tadah hpo -c ../unit1/config.train --validation v.val \\
                 --hpotarget search.hpo
               tadah hpo --check --config=config.train --hpotarget=other.hpo
            """))
        self.write(f"{files}/tutorial/unit2/search.hpo", TARGET)
        self.write(f"{files}/tutorial/unit2/other.hpo", TARGET)
        self.write(f"{files}/tutorial/unit3/alone.hpo", TARGET)
        self.write(f"{files}/tutorial/unit3/config.train", "ATOM Ta\n")
        self.write(f"{files}/tutorial/unit4/lost.hpo", TARGET)
        self.write("Tadah.MLIP/docs/source/nfp_examples/first.hpo", TARGET)
        c = corpus.Corpus()
        corpus.read_docs(c, REG, tm)
        by_id = {i.id: i for i in c.inputs}
        tree = str(self.root / files)
        s = by_id["docs:nfp/files/tutorial/unit2/search.hpo"]
        self.assertEqual(s.case, {"tree": tree, "cwd": "tutorial/unit2",
                                  "target_file": "search.hpo",
                                  "config_file": "../unit1/config.train"})
        self.assertEqual(s.origin, "nfp/unit2.rst, line 6")
        self.assertEqual(
            by_id["docs:nfp/files/tutorial/unit2/other.hpo"]
            .case["config_file"], "config.train")
        self.assertEqual(by_id["docs:nfp/files/tutorial/unit3/alone.hpo"].case,
                         {"tree": tree, "cwd": "tutorial/unit3",
                          "target_file": "alone.hpo",
                          "config_file": "config.train"})
        first = by_id["docs:nfp_examples/first.hpo"].case
        self.assertEqual(first["file"], TARGET)
        self.assertEqual(first["config"], REG.contexts["default"].config)
        self.assertEqual(len(c.problems), 1)
        self.assertIn("nfp/files/tutorial/unit4/lost.hpo", c.problems[0])

    def test_cases_text(self):
        c = corpus.Corpus()
        c.add("registry", "X:0", "here", {"file": 'A "q"\n', "config": "C",
                                          "env": {"K": "v"},
                                          "config_add": {"F": ["0", "1"]}})
        c.add("registry", "X:0", "there", {"file": "B"})
        import tomllib
        doc = tomllib.loads(corpus.cases_text(c))
        self.assertEqual([x["id"] for x in doc["case"]],
                         ["registry:X:0", "registry:X:0#2"])
        self.assertEqual(doc["case"][0]["mode"], "describe")
        self.assertEqual(doc["case"][0]["file"], 'A "q"\n')
        self.assertEqual(doc["case"][0]["env"], {"K": "v"})
        self.assertEqual(doc["case"][0]["config_add"], {"F": ["0", "1"]})


def corpus_doc_commands() -> pathlib.Path:
    """doc_commands.py of the real Tadah.MLIP tree, or a copy of its
    interface for a build without it."""
    d = os.environ.get("TADAH_UMBRELLA_SOURCE_DIR", "")
    if d:
        return pathlib.Path(d) / "tests" / "doc_commands" / "doc_commands.py"
    p = pathlib.Path(tempfile.mkdtemp()) / "doc_commands.py"
    p.write_text(textwrap.dedent('''\
        import shlex
        from collections import namedtuple
        Command = namedtuple("Command", "line argv")
        def bash_commands(path):
            out, pending, start = [], "", None
            lines = open(path).read().splitlines()
            inside = False
            for n, line in enumerate(lines, 1):
                if line.strip() == ".. code-block:: bash":
                    inside = True
                    continue
                if not inside or not line.strip():
                    continue
                if start is None:
                    start = n
                if line.rstrip().endswith("\\\\"):
                    pending += line.strip()[:-1] + " "
                    continue
                out.append(Command(start, shlex.split(pending + line)))
                pending, start = "", None
            return out
        '''))
    return p


class RealTrees(unittest.TestCase):
    def test_hpo_and_registry(self):
        c = corpus.collect(HPO, HPO / "blueprint", None)
        self.assertEqual(c.problems, [])
        self.assertEqual(c.read, ["hpo-data", "registry"])
        ids = {i.id for i in c.inputs}
        data = HPO / "tests" / "data"
        for f in data.rglob("*.hpo"):
            self.assertIn(f"hpo-data:{f.relative_to(data).as_posix()}", ids)
        examples = [x for x in REG.cases() if x.kind == "example"]
        self.assertEqual(sum(1 for i in c.inputs if i.source == "registry"),
                         len(examples))
        self.assertEqual(len(ids), len(c.inputs))

    def test_umbrella(self):
        tm = umbrella(self)
        c = corpus.collect(HPO, HPO / "blueprint", tm)
        self.assertEqual(c.problems, [])
        self.assertEqual(c.read, list(corpus.SOURCES))
        origins = [i.origin for i in c.inputs if i.source == "mlip-tests"]
        for test in ("tests/test_tadah_cli_exit_status.cpp",
                     "tests/cli_failed_evaluations/"
                     "check_hpo_failed_evaluations.py",
                     "tests/cli_balance/check_fit_balance.py",
                     "tests/cli_signal/check_hpo_signal.py"):
            self.assertTrue(any(o.startswith(f"Tadah.MLIP/{test},")
                                for o in origins), test)
        ids = {i.id for i in c.inputs}
        source = tm / "docs" / "source"
        for f in (source / "nfp_examples").glob("*.hpo"):
            self.assertIn(f"docs:nfp_examples/{f.name}", ids)
        files = source / "nfp" / "files"
        for f in files.rglob("*.hpo"):
            self.assertIn(f"docs:nfp/files/{f.relative_to(files).as_posix()}",
                          ids)


if __name__ == "__main__":
    unittest.main()
