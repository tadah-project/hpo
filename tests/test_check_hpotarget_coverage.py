#!/usr/bin/env python3
"""Unit tests of tests/check_hpotarget_coverage.py on synthetic C++ sources
and a synthetic registry: comment stripping, brace matching, the three
kinds of extraction (comparisons, tables, calls), the refusal of a
comparison no rule classifies, of a rule that finds nothing or whose
function is not found once, the comparison of the names found with the
registry's in both directions, and the command without a registry."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import pathlib
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_hpotarget_coverage as cov  # noqa: E402

SOURCE = r'''
// key == "COMMENTED" in a line comment
/* key == "ALSO_COMMENTED"
   across lines */
static const char* kNames[] = {"A", "B"};
struct Row { const char* name; int n; };
static const Row kRows[] = {
    {"FIRST", 1},
    {"SECOND", 2},   // {"THIRD", 3}
};
bool parse(const std::string& key, const std::string& v)
{
    const std::string url = "http://not/a/comment";
    if (key == "ALPHA") return true;
    if (key != "BETA") { if (v == "ON") return false; }
    if (sp.lib == "NLOPT") return false;
    if (tokens[i] == "-var") return false;
    lookup(sp, "gamma");
    return false;
}
bool other(const std::string& x)
{
    return x == "DELTA";
}
'''


@dataclasses.dataclass
class FakeKey:
    scope: str
    group: str = ""


class FakeRegistry:
    def __init__(self, vocab, keys=()):
        self.vocab = vocab
        self.keys = list(keys)

    def vocabulary(self, name):
        return set(self.vocab.get(name, ()))


class Extraction(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "a.cpp").write_text(SOURCE)

    def tearDown(self):
        self._tmp.cleanup()

    def run_rules(self, rules):
        with mock.patch.object(cov, "EXTRACTION", rules):
            return cov.extract(self.root)

    def test_strip_comments_keeps_strings_and_lines(self):
        out = cov.strip_comments(SOURCE)
        self.assertNotIn("COMMENTED", out)
        self.assertNotIn("THIRD", out)
        self.assertIn('"http://not/a/comment"', out)
        self.assertEqual(out.count("\n"), SOURCE.count("\n"))
        self.assertEqual(cov.strip_comments('s = "/* x */"; // y'),
                         's = "/* x */";     ')

    def test_braced_skips_braces_in_literals(self):
        src = 'f() { s = "}"; c = \'{\'; { } } tail'
        a, b = cov.braced(src, 0)
        self.assertEqual(src[a:b], 'f() { s = "}"; c = \'{\'; { } }'[4:])

    def test_comparisons_tables_and_calls(self):
        rule = cov.Rule(
            "parse", "src/a.cpp", r"^bool parse\(",
            operands={"key": ("key:ROOT",), "v": ("word",)},
            ignore={"sp.lib": "an optimiser name"},
            ignore_literals={"-var": "a LAMMPS option"},
            calls=[(r'lookup\(\s*\w+\s*,\s*"([^"]+)"\)', ("param",))])
        tables = cov.Rule("tables", "src/a.cpp", compare=False,
                          tables=[(r"kNames\s*\[\]", "all", ("block",)),
                                  (r"kRows\s*\[\]", "first", ("removed",))])
        found, problems = self.run_rules([rule, tables])
        self.assertEqual(problems, [])
        got = {(f.name, f.vocabularies) for f in found}
        self.assertEqual(got, {("ALPHA", ("key:ROOT",)),
                               ("BETA", ("key:ROOT",)),
                               ("ON", ("word",)), ("gamma", ("param",)),
                               ("A", ("block",)), ("B", ("block",)),
                               ("FIRST", ("removed",)),
                               ("SECOND", ("removed",))})
        alpha = next(f for f in found if f.name == "ALPHA")
        self.assertEqual(alpha.where, "src/a.cpp:14 (parse)")

    def test_unclassified_comparison(self):
        rule = cov.Rule("parse", "src/a.cpp", r"^bool parse\(",
                        operands={"key": ("key:ROOT",)})
        _, problems = self.run_rules([rule])
        self.assertEqual(len(problems), 3)
        self.assertIn('src/a.cpp:15: unclassified comparison in parse: '
                      'v == "ON"', problems[0])
        self.assertIn('sp.lib == "NLOPT"', problems[1])
        self.assertIn('tokens[i] == "-var"', problems[2])

    def test_literal_on_the_left(self):
        (self.root / "src" / "b.cpp").write_text(
            'bool f(std::string k) {\n  return "X" == k;\n}\n')
        rule = cov.Rule("f", "src/b.cpp", r"^bool f\(", operands={})
        _, problems = self.run_rules([rule])
        self.assertTrue(any('the literal "X" is on the left' in p
                            for p in problems), problems)

    def test_rule_that_finds_nothing_or_no_function(self):
        none = cov.Rule("other", "src/a.cpp", r"^bool other\(",
                        ignore={"x": "not a name"})
        guard = cov.Rule("guard", "src/a.cpp", r"^bool other\(",
                         ignore={"x": "not a name"}, finds_names=False)
        missing = cov.Rule("gone", "src/a.cpp", r"^bool gone\(")
        twice = cov.Rule("twice", "src/a.cpp", r"^bool \w+\(")
        table = cov.Rule("table", "src/a.cpp", compare=False,
                         tables=[(r"kMissing", "all", ("block",))])
        _, problems = self.run_rules([none, guard, missing, twice, table])
        text = "\n".join(problems)
        self.assertIn("other: the rule finds no name", text)
        self.assertNotIn("guard:", text)
        self.assertIn("gone: the definition /^bool gone\\(/ is found 0 "
                      "times", text)
        self.assertIn("twice: the definition /^bool \\w+\\(/ is found 2 "
                      "times", text)
        self.assertIn("table: no table /kMissing/", text)


class Compare(unittest.TestCase):
    VOCAB = ("block", "word")

    def found(self, *items):
        return [cov.Found(n, v, f"src/a.cpp:{i} (rule)")
                for i, (n, v) in enumerate(items, 1)]

    def compare(self, found, reg):
        with mock.patch.object(cov, "VOCABULARIES", self.VOCAB):
            return cov.compare(found, reg)

    def test_equal_names(self):
        reg = FakeRegistry({"block": {"A"}, "word": {"ON"}})
        found = self.found(("A", ("block",)), ("ON", ("word", "block")))
        self.assertEqual(self.compare(found, reg), [])

    def test_missing_from_the_registry_once_per_name(self):
        reg = FakeRegistry({"block": {"A"}, "word": {"ON"}})
        found = self.found(("A", ("block",)), ("ON", ("word",)),
                           ("B", ("block",)), ("B", ("block",)))
        self.assertEqual(self.compare(found, reg), [
            "missing from the registry: 'B' (block), src/a.cpp:3 (rule) "
            "and 1 more place(s)"])

    def test_not_in_the_parser(self):
        reg = FakeRegistry({"block": {"A", "Z"}, "word": {"ON"}})
        found = self.found(("A", ("block",)), ("ON", ("word",)))
        problems = self.compare(found, reg)
        self.assertEqual(len(problems), 1)
        self.assertIn("not in the parser: 'Z' (block)", problems[0])

    def test_vocabulary_no_rule_reads(self):
        reg = FakeRegistry({"block": {"A"}})
        problems = self.compare(self.found(("A", ("block",))), reg)
        self.assertEqual(problems,
                         ["no extraction rule reads the vocabulary word"])

    def test_key_scope_no_rule_reads(self):
        reg = FakeRegistry({"block": {"A"}, "word": {"ON"}},
                           keys=[FakeKey("EXPLORE"), FakeKey("OPTIMIZER",
                                                             "OPTIMIZER")])
        found = self.found(("A", ("block",)), ("ON", ("word",)))
        self.assertEqual(self.compare(found, reg), [
            "not in the parser: the registry has keys of scope EXPLORE, a "
            "scope no extraction rule reads"])


class Command(unittest.TestCase):
    def test_no_registry_fails_with_a_message(self):
        with tempfile.TemporaryDirectory() as d:
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cov.main([str(HERE.parent), "--registry", d])
            self.assertEqual(code, 1)
            self.assertIn("hpotarget_registry.py not found", out.getvalue())


class RealSources(unittest.TestCase):
    """Every rule of EXTRACTION finds its function, table or call in the
    HPO sources, and no scanned function holds an unclassified
    comparison."""

    def test_rules_match_the_sources(self):
        _, problems = cov.extract(HERE.parent)
        self.assertEqual(problems, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
