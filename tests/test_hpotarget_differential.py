#!/usr/bin/env python3
"""Unit tests of tests/hpotarget_differential.py, the cases on which
test_hpotarget_differential.cpp compares the two parsers of the target
file: on the real registry and trees, every example and invalid entry of
the registry is a case, in the mode its check says (an entry checked by
`tadah hpo --check` through the startup checks), composed in its wrapper
with its context, environment and the registry's fixture files; every case
of the generated checks is a case (defaults read as parse); every input of
the corpus is a case; the ids are unique and name their source; the cases
file reads back as written. The Tadah.MLIP tree is the one
TADAH_UMBRELLA_SOURCE_DIR names (CTest sets it in a build of Tadah.MLIP);
without it the corpus is read without the sources of Tadah.MLIP.

Before the fix: hpotarget_differential.py does not exist and the import
fails."""

from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import tomllib
import unittest

HERE = pathlib.Path(__file__).resolve().parent
HPO = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HPO / "blueprint"))

import check_hpotarget_generated as generated  # noqa: E402
import hpotarget_corpus as corpus  # noqa: E402
import hpotarget_differential as differential  # noqa: E402

REGISTRY = HPO / "blueprint"


def umbrella() -> pathlib.Path | None:
    d = os.environ.get("TADAH_UMBRELLA_SOURCE_DIR", "")
    return pathlib.Path(d) if d else None


class Cases(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.hr, cls.reg = corpus.load_registry(REGISTRY)
        cls.cases, cls.problems = differential.all_cases(HPO, REGISTRY,
                                                         umbrella())
        cls.by_id = dict(cls.cases)

    def of(self, source: str) -> dict[str, dict]:
        return {cid: c for cid, c in self.cases
                if cid.split(":", 1)[0] == source}

    def test_no_problem_and_every_source(self):
        self.assertEqual(self.problems, [])
        for s in differential.SOURCES:
            self.assertTrue(self.of(s), s)
        self.assertEqual(len(self.by_id), len(self.cases))

    def test_every_registry_case(self):
        examples = [c for c in self.reg.cases() if c.kind == "example"]
        invalid = [c for c in self.reg.cases() if c.kind == "invalid"]
        self.assertEqual(len(self.of("registry-example")), len(examples))
        self.assertEqual(len(self.of("registry-invalid")), len(invalid))
        for c in examples + invalid:
            got = self.by_id[f"registry-{c.kind}:{c.owner}:{c.index}"]
            comp = self.hr.compose(self.reg, c.wrap, c.text, c.context)
            ctx = self.reg.contexts[comp.context]
            self.assertEqual(got["file"], comp.file)
            self.assertEqual(got["config"], ctx.config)
            self.assertEqual(got["has_train_db"], ctx.has_train_db)
            self.assertEqual(got["has_validation"], ctx.has_validation)
            self.assertEqual(got.get("env", {}), dict(c.env))
            self.assertEqual(got["fixtures"], corpus.fixtures(self.reg))
            self.assertEqual(got["mode"],
                             "parse" if c.check == "parse" else "startup")
        self.assertTrue(any(c.check == "check" for c in invalid))

    def test_every_generated_case(self):
        g = generated.Generator(self.hr, self.reg)
        for name in generated.CHECK_NAMES:
            getattr(g, f"gen_{name}")()
        got = self.of("generated")
        self.assertEqual(len(got), len(g.cases))
        for c in g.cases:
            case = got[f"generated:{c.case['id']}"]
            self.assertEqual(case["file"], c.case["file"])
            self.assertEqual(case["mode"], "parse"
                             if c.case["mode"] == "defaults"
                             else c.case["mode"])

    def test_every_corpus_input(self):
        c = corpus.collect(HPO, REGISTRY, umbrella())
        got = self.of("corpus")
        self.assertEqual(len(got), len(c.inputs))
        for i in c.inputs:
            self.assertEqual(got[f"corpus:{i.id}"],
                             {"mode": "parse", **i.case})

    def test_the_cases_file_reads_back(self):
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "sub" / "cases.toml"
            argv = ["cases", str(out)]
            if umbrella():
                argv.append(f"--tadah-mlip={umbrella()}")
            self.assertEqual(differential.main(argv), 0)
            data = tomllib.loads(out.read_text(encoding="utf-8"))
        cases = data["case"]
        self.assertEqual(len(cases), len(self.cases))
        for t, (cid, case) in zip(cases, self.cases):
            self.assertEqual(t["id"], cid)
            self.assertIn(t["mode"], ("parse", "startup"))
            for k, v in case.items():
                self.assertEqual(t[k], v, f"{cid}: {k}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
