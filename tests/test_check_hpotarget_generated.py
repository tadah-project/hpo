#!/usr/bin/env python3
"""Unit tests of tests/check_hpotarget_generated.py on the synthetic
registry of test_hpotarget_registry.py: the TOML it writes for the runner,
the frame it reads an error from, the cases each check generates, and what
each check makes of a result the parser could give (a refusal where the
registry says there is one, an acceptance where it says there is none),
without running the parser."""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import tempfile
import textwrap
import tomllib
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "blueprint"))

import check_hpotarget_generated as gen  # noqa: E402
import hpotarget_registry as hr  # noqa: E402
import test_hpotarget_registry as synthetic  # noqa: E402


def result(parse="", startup="", values=None):
    return {"parse_error": parse, "startup_error": startup,
            "check_error": "", "values": values or {}}


class Generated(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        d = pathlib.Path(self._tmp.name)
        files = {"hpotarget.toml": synthetic.MAIN,
                 "part_a.toml": synthetic.PART_A,
                 "part_b.toml": synthetic.PART_B,
                 "optimizers_gen.toml": synthetic.OPTIMIZERS,
                 "optimizers_doc.toml": synthetic.OPTIMIZER_DOCS}
        for name, text in files.items():
            (d / name).write_text(textwrap.dedent(text))
        self.reg = hr.load(d, templates=False)
        self.g = gen.Generator(hr, self.reg)

    def tearDown(self):
        self._tmp.cleanup()

    def cases(self, check):
        return {c.label: c for c in self.g.cases if c.check == check}

    def test_toml_of_the_cases(self):
        self.assertEqual(gen.toml_string('a "b"\n\\c\x01'),
                         '"a \\"b\\"\\n\\\\c\\u0001"')
        with tempfile.TemporaryDirectory() as d:
            p = pathlib.Path(d) / "cases.toml"
            gen.write_cases(p, [{"id": "x", "mode": "parse", "file": "A\n",
                                 "config": "ATOM H\n", "has_train_db": True,
                                 "env": {"K": "v"},
                                 "fixtures": [{"path": "f", "text": "t"}]}])
            data = tomllib.loads(p.read_text())
        self.assertEqual(data["case"][0]["file"], "A\n")
        self.assertEqual(data["case"][0]["env"], {"K": "v"})
        self.assertEqual(data["case"][0]["fixtures"][0]["path"], "f")

    def test_frame_what(self):
        err = "Error on line 4: unknown key 'X' in this block\nLine: X 1"
        self.assertEqual(self.g.frame_what(err, 4),
                         "unknown key 'X' in this block")
        self.assertIsNone(self.g.frame_what(err, 3))

    def test_blocks(self):
        # FAILSCORE belongs to ROOT: refused in OPTIMIZER with the block's
        # unknown-key message; SEED is refused in INNER, which refuses it.
        self.g.gen_blocks()
        c = self.cases("blocks")
        self.assertEqual(set(c), {f"{k} in {p}" for k in
                                  ("FAILSCORE", "LIB", "SEED")
                                  for p in ("ROOT", "OPTIMIZER", "INNER")})
        in_opt = c["FAILSCORE in OPTIMIZER"]
        refused = result("Error on line 3: unknown key 'FAILSCORE' in this "
                         "block\nLine: FAILSCORE 10")
        self.assertIsNone(in_opt.judge(refused))
        self.assertIn("is not refused as out of place",
                      in_opt.judge(result()))
        self.assertIsNone(c["FAILSCORE in ROOT"].judge(result()))
        self.assertIn("is refused in ROOT, which the registry lists",
                      c["FAILSCORE in ROOT"].judge(result(
                          'Error on line 1: unknown key "FAILSCORE"\n'
                          'Line: FAILSCORE 10')))
        seed = result("Error on line 4: SEED is not accepted inside an "
                      "INNER block: the enclosing run draws it; set SEED on "
                      "the enclosing OPTIMIZER block\nLine: SEED 7")
        self.assertIsNone(c["SEED in INNER"].judge(seed))

    def test_repeat(self):
        self.g.gen_repeat()
        c = self.cases("repeat")
        # last_wins: the doubled line is read.
        self.assertIsNone(c["ROOT.FAILSCORE twice"].judge(result()))
        self.assertIn("is refused", c["ROOT.FAILSCORE twice"].judge(
            result("Error on line 2: no\nLine: FAILSCORE 10")))
        self.assertIn("FAILSCORE 10\nFAILSCORE 10",
                      c["ROOT.FAILSCORE twice"].case["file"])

    def test_ranges(self):
        # SEED: an integer from 0 to UINT64_MAX.
        self.g.gen_ranges()
        c = self.cases("ranges")
        self.assertEqual({k for k in c if k.startswith("OPTIMIZER.SEED")},
                         {"OPTIMIZER.SEED 0", "OPTIMIZER.SEED -1",
                          "OPTIMIZER.SEED 18446744073709551615",
                          "OPTIMIZER.SEED 18446744073709551616"})
        below = c["OPTIMIZER.SEED -1"]
        self.assertIsNone(below.judge(result(
            "Error on line 3: SEED must be an integer >= 0, got '-1'\n"
            "Line: SEED -1")))
        self.assertIn("is not refused at its line", below.judge(result()))
        self.assertIn("a message the registry does not give",
                      below.judge(result("Error on line 3: SEED is bad\n"
                                         "Line: SEED -1")))
        self.assertIsNone(c["OPTIMIZER.SEED 0"].judge(result()))
        # FAILSCORE, positive: 0 refused, 1 accepted.
        self.assertIn("ROOT.FAILSCORE 0", c)
        self.assertIn("ROOT.FAILSCORE 1", c)

    def test_the_largest_value_runs_through_the_startup_checks(self):
        # Pins: the max of a range runs as its example does (startup, the
        # default), so a value the parser takes and the run cannot use
        # fails; the other values are read only.
        # Before the fix: every range case was read only, and K, K_TOTAL
        # and TOPK of 18446744073709551615 passed, which stop tadah hpo
        # with "vector::reserve".
        self.g.gen_ranges()
        c = self.cases("ranges")
        top = c["OPTIMIZER.SEED 18446744073709551615"]
        self.assertEqual(top.case["mode"], "startup")
        for label in ("OPTIMIZER.SEED 0", "OPTIMIZER.SEED -1",
                      "OPTIMIZER.SEED 18446744073709551616"):
            self.assertEqual(c[label].case["mode"], "parse")
        self.assertIn("a value in the range of OPTIMIZER.SEED is refused: "
                      "vector::reserve",
                      top.judge(result(startup="vector::reserve")))
        self.assertIsNone(top.judge(result()))

    def test_ranges_of_every_example(self):
        # Pins: each example of an entry gives its own cases, so the value
        # is judged where the file uses it (K_TOTAL replaces the K of a
        # single METHOD line; after two lines it caps what they draw).
        key = next(k for k in self.reg.keys if k.id == "OPTIMIZER.SEED")
        key.examples.append(dataclasses.replace(
            key.examples[0], index=1, wrap="inner_a", doc="In INNER."))
        self.g.gen_ranges()
        c = self.cases("ranges")
        self.assertIn("OPTIMIZER.SEED 0 (example 1)", c)
        self.assertIn("INNER\nSEED 0\nENDINNER",
                      c["OPTIMIZER.SEED 0 (example 1)"].case["file"])

    def test_ranges_of_init_methods(self):
        # Pins: the K of an INIT method is a range too, its line METHOD
        # <name> <K>, refused with [reading]'s message and the argument's
        # what.
        # Before the fix: no case judged an INIT method's argument.
        case = hr.Case(owner="METHOD.RANDOM", part="part_a.toml",
                       kind="example", index=0, wrap="optimizer",
                       text="METHOD RANDOM 3", check="startup")
        self.reg.init_methods.append(hr.InitMethod(
            id="METHOD.RANDOM", part="part_a.toml", name="RANDOM",
            doc="d", examples=[case], args=[hr.Arg(
                name="K", kind="integer", min=0, max=1000000,
                what="INIT METHOD RANDOM K")]))
        self.g.gen_ranges()
        c = self.cases("ranges")
        self.assertEqual({k for k in c if k.startswith("METHOD.")},
                         {"METHOD.RANDOM 0", "METHOD.RANDOM -1",
                          "METHOD.RANDOM 1000000", "METHOD.RANDOM 1000001"})
        self.assertIn("METHOD RANDOM 1000001",
                      c["METHOD.RANDOM 1000001"].case["file"])
        self.assertEqual(c["METHOD.RANDOM 1000000"].case["mode"], "startup")
        self.assertIsNone(c["METHOD.RANDOM 1000001"].judge(result(
            "Error on line 3: INIT METHOD RANDOM K must be an integer from "
            "0 to 1000000, got '1000001'\nLine: METHOD RANDOM 1000001")))
        self.assertIn("is not refused at its line",
                      c["METHOD.RANDOM 1000001"].judge(result()))

    def test_ranges_of_param_keys(self):
        # Pins: a PARAM value runs through the startup checks, which read
        # it, at every value: in the range it starts, past it it is
        # refused, at parse or at startup and at any line, with a message
        # of its entry or of [reading].
        # Before the fix: no case judged a PARAM key's argument.
        self.g.gen_ranges()
        c = self.cases("ranges")
        low = c["PARAM.TADAH:GRID.GRID_POINTS 1"]
        self.assertEqual(low.case["mode"], "startup")
        self.assertEqual(c["PARAM.TADAH:GRID.GRID_POINTS 2"].case["mode"],
                         "startup")
        self.assertIn("PARAM GRID_POINTS 1", low.case["file"])
        self.assertIsNone(low.judge(result(startup=(
            "Error on line 4: GRID_POINTS must be at least 2\n"
            "Line: ENDOPTIMIZER"))))
        self.assertIn("is not refused: tadah hpo starts",
                      low.judge(result()))
        self.assertIn("a message the registry does not give",
                      low.judge(result(startup="Error on line 4: no\n"
                                               "Line: ENDOPTIMIZER")))
        self.assertIn("a value in the range of PARAM.TADAH:GRID.GRID_POINTS "
                      "is refused",
                      c["PARAM.TADAH:GRID.GRID_POINTS 2"].judge(result(
                          startup="no")))

    def test_a_group_passes_when_one_case_does(self):
        # Pins: an in-range PARAM value passes when it starts in the file
        # of one example of its entry (ASA's User_Initial_Parameters 0
        # starts only where the search has no starting point), and fails
        # with the problem of the first case when it starts in none.
        self.g.gen_ranges()
        c = self.cases("ranges")["PARAM.TADAH:GRID.GRID_POINTS 2"]
        self.assertEqual(c.group, "PARAM.TADAH:GRID.GRID_POINTS 2")
        other = dataclasses.replace(c, case=dict(c.case, id="other"))
        self.g.cases = [c, other]
        refused = result(startup="no")
        problems, counts = gen.judge_all(
            self.g, {c.case["id"]: refused, "other": result()})
        self.assertEqual((problems, counts), ([], {"ranges": 2}))
        problems, _ = gen.judge_all(
            self.g, {c.case["id"]: refused, "other": refused})
        self.assertEqual(len(problems), 1)
        self.assertIn("GRID_POINTS 2: a value in the range", problems[0])

    def test_frame_any(self):
        err = "Error on line 12: it is wrong\nLine: ENDOPTIMIZER"
        self.assertEqual(self.g.frame_any(err), "it is wrong")
        self.assertIsNone(self.g.frame_any("vector::reserve"))

    def test_words(self):
        self.g.gen_words()
        c = self.cases("words")
        self.assertEqual(set(c), {"OPTIMIZER.LIB NLOPT",
                                  "OPTIMIZER.LIB TADAH"})
        self.assertIsNone(c["OPTIMIZER.LIB NLOPT"].judge(result()))
        self.assertIn("the word NLOPT is refused",
                      c["OPTIMIZER.LIB NLOPT"].judge(result(
                          "Error on line 3: LIB is wrong\nLine: LIB NLOPT")))

    def test_params_need_wrappers_that_name_their_optimizer(self):
        self.g.gen_params()
        self.assertEqual(self.g.problems,
                         ["params: no wrapper names its optimizer: the PARAM "
                          "keys of each optimiser cannot be checked"])

    def test_defaults(self):
        self.g.default_results = [{"ROOT.FAILSCORE": ["1000000"],
                                   "LOSS.HUBER.delta": ["50"]}]
        self.assertEqual(self.g.judge_defaults(), [])
        self.g.default_results = [{"ROOT.FAILSCORE": ["10"],
                                   "LOSS.HUBER.delta": ["50"],
                                   "ROOT.QFLUSH": ["10"]}]
        problems = self.g.judge_defaults()
        self.assertIn("defaults: ROOT.FAILSCORE: the registry gives '1e6', "
                      "the code 10", problems)
        self.assertIn("defaults: the runner reports ROOT.QFLUSH = 10, which "
                      "has no default in the registry", problems)
        self.g.default_results = [{"LOSS.HUBER.delta": ["50"]}]
        self.assertIn("defaults: ROOT.FAILSCORE: the literal default '1e6' "
                      "is reported by no value of the runner "
                      "(hpotarget_case_runner, defaults_of)",
                      self.g.judge_defaults())

    def test_text_defaults(self):
        # Each number of a text default is a value the runner reports,
        # unless unchecked says why none holds it; each number the runner
        # reports is in the text.
        key = self.reg.keys[0]                   # ROOT.FAILSCORE
        delta = {"LOSS.HUBER.delta": ["50"]}
        key.default = hr.Default(text="1e6, or 10 under a grid")
        self.g.default_results = [{"ROOT.FAILSCORE": ["1000000"], **delta}]
        self.assertEqual(self.g.judge_defaults(),
                         ["defaults: ROOT.FAILSCORE: the text default says "
                          "10, the code gives 1000000"])
        key.default.unchecked = "The grid applies it; no setting holds it."
        self.assertEqual(self.g.judge_defaults(), [])
        self.g.default_results = [{"ROOT.FAILSCORE": ["5"], **delta}]
        self.assertEqual(self.g.judge_defaults(),
                         ["defaults: ROOT.FAILSCORE: the code gives 5, which "
                          "the text default does not say ('1e6, or 10 under "
                          "a grid')"])
        key.default.unchecked = None
        self.g.default_results = [delta]
        self.assertIn("the text default holds numbers",
                      self.g.judge_defaults()[0])

    def test_params_take_the_name_of_every_entry(self):
        # A PARAM name of two entries is listed on the optimisers of both.
        reg = self.reg
        w = reg.wrappers["optimizer"]
        reg.wrappers["opt_grid"] = dataclasses.replace(
            w, name="opt_grid", optimizer="TADAH:GRID")
        reg.wrappers["opt_nlopt"] = dataclasses.replace(
            w, name="opt_nlopt", optimizer="NLOPT:LN_SBPLX")
        reg.param_policies.append(hr.ParamPolicy(
            id="POLICY.NLOPT", part="part_b.toml", optimizers=["NLOPT"],
            doc="d", msg="PARAM {key} is refused"))
        self.g.gen_params()
        self.assertEqual(set(self.cases("params")),
                         {"PARAM.TADAH:GRID.GRID_POINTS on TADAH:GRID",
                          "PARAM.TADAH:GRID.GRID_POINTS on NLOPT:LN_SBPLX"})
        reg.params.append(dataclasses.replace(
            reg.params[0], id="PARAM.NLOPT.GRID_POINTS",
            optimizers=["NLOPT"]))
        self.g.cases.clear()
        self.g.gen_params()
        self.assertEqual(set(self.cases("params")),
                         {"PARAM.TADAH:GRID.GRID_POINTS on TADAH:GRID",
                          "PARAM.NLOPT.GRID_POINTS on NLOPT:LN_SBPLX"})

    def test_option_words_in_any_case(self):
        # An example may write an option word in lower case; a flag keeps
        # its case.
        def option(name, text):
            case = hr.Case(owner="OPTIMIZER." + name, part="x",
                           kind="example", index=0, wrap="optimizer",
                           text=text, check="parse")
            return hr.LineOption(id="OPTIMIZER." + name, option=name,
                                 part="x", lines=["SEED"], synopsis=name,
                                 doc="d", examples=[case])
        got = self.g.option_in_example(option("WIDE", "SEED 7 wide"))
        self.assertEqual(got[2][got[3]], "wide")
        self.assertIsNone(self.g.option_in_example(
            option("--wide", "SEED 7 --WIDE")))


if __name__ == "__main__":
    unittest.main(verbosity=2)
