#!/usr/bin/env python3
"""Unit tests of tests/check_hpotarget_coverage.py on synthetic C++ sources
and a synthetic registry: comment stripping, brace matching, the three
kinds of extraction (comparisons, tables, calls) and a vocabulary a
function reads from the generated tables of the registry, the refusal of a
comparison no rule classifies, of a rule that finds nothing or whose
function is not found once, the comparison of the names found with the
registry's in both directions, the command without a registry, and a
rule on a file of MLIP (mlip:<path>, read from the --mlip directory); on a
copy of the HPO and MLIP sources, a top-level key the registry-driven
parser compares that the registry does not know; and that no rule reads
the hand-written parser the registry-driven one replaced.

The rules on MLIP's files read the MLIP source directory that
TADAH_MLIP_SOURCE_DIR names (CTest sets it); a test that needs it fails
when it is not set."""

from __future__ import annotations

import contextlib
import dataclasses
import io
import os
import pathlib
import shutil
import sys
import tempfile
import textwrap
import unittest
from unittest import mock

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))


def mlip_source_dir(test: unittest.TestCase) -> pathlib.Path:
    """MLIP's source directory, TADAH_MLIP_SOURCE_DIR (the rules on mlip:
    files read it); fails @p test when it is not set."""
    d = os.environ.get("TADAH_MLIP_SOURCE_DIR", "")
    if not d:
        test.fail("TADAH_MLIP_SOURCE_DIR is not set: it names MLIP's source "
                  "directory, which the rules on mlip: files read (CTest "
                  "sets it)")
    return pathlib.Path(d)

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

    def test_scoped_tables(self):
        # The lines of a PC_* option and the TYPE of a STRATEGY key come
        # from the second fields of their tables.
        self.assertEqual(cov.pc_line_option_of('{"--gate", false, true}'),
                         ("line_option:PC_FAILPROBE",))
        self.assertEqual(cov.pc_line_option_of('{"--script", true, true}'),
                         ("line_option:PC_LAMMPS",
                          "line_option:PC_FAILPROBE"))
        self.assertEqual(cov.strategy_type_key_of(
            '{"TOPK", StrategyType::PRESTAGE, "why"}'),
            ("strategy_type_key:PRESTAGE",))
        (self.root / "src" / "t.cpp").write_text(
            'static const Opt k[] = {\n  {"--a", true, false},\n'
            '  {"--b", false, true},\n};\n')
        rule = cov.Rule("t", "src/t.cpp", compare=False,
                        tables=[(r"\bk\s*\[\]", "first",
                                 cov.pc_line_option_of)])
        found, problems = self.run_rules([rule])
        self.assertEqual(problems, [])
        self.assertEqual({(f.name, f.vocabularies) for f in found},
                         {("--a", ("line_option:PC_LAMMPS",)),
                          ("--b", ("line_option:PC_FAILPROBE",))})

    def test_comparisons_the_rules_cannot_read(self):
        # A key compared with a named constant, a constructed string or
        # compare() is reported, unless the rule gives the reason.
        (self.root / "src" / "u.cpp").write_text(
            'bool f(const std::string& key) {\n'
            '  if (key == kNewKey) return true;\n'
            '  if (key == std::string("NEW")) return true;\n'
            '  if (key.compare("NEWER") == 0) return true;\n'
            '  return key == "OLD";\n}\n')
        rule = cov.Rule("u", "src/u.cpp", r"^bool f\(",
                        operands={"key": ("key:ROOT",)})
        _, problems = self.run_rules([rule])
        text = "\n".join(problems)
        self.assertIn("src/u.cpp:2: comparison in u that this check cannot "
                      "read: 'key == kNewKey'", text)
        self.assertIn("'key == std::string('", text)
        self.assertIn("'.compare('", text)
        ok = cov.Rule("u", "src/u.cpp", r"^bool f\(",
                      operands={"key": ("key:ROOT",)},
                      unread_ok={"key == kNewKey": "r1",
                                 "key == std::string(": "r2",
                                 ".compare(": "r3"})
        _, problems = self.run_rules([ok])
        self.assertEqual(problems, [])

    def test_name_shaped_literals_no_rule_reads(self):
        # A set of names the rule does not read: each name is handed on
        # for compare(), unless the rule says why it is no name.
        (self.root / "src" / "v.cpp").write_text(
            'bool f(const std::string& key) {\n'
            '  static const std::set<std::string> s = {"QFLUSH", "--opt",'
            ' "lower", "A B"};\n'
            '  if (s.count(key)) return true;\n'
            '  return key == "OLD";\n}\n')
        unread = []
        rule = cov.Rule("v", "src/v.cpp", r"^bool f\(",
                        operands={"key": ("key:ROOT",)},
                        not_names={"--opt": "not a name"})
        with mock.patch.object(cov, "EXTRACTION", [rule]):
            cov.extract(self.root, unread)
        self.assertEqual([lit for lit, _ in unread], ["QFLUSH"])

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


class MlipFiles(unittest.TestCase):
    """A rule whose file is written mlip:<path> reads <path> under the
    --mlip directory (MLIP's OPTIM reader), and is a problem without it."""

    # Pins: with HPO's src/a.cpp holding SOURCE and MLIP's src/b.cpp a
    # function `bool optim(...)` comparing k with "ZZ_OPTIM", a rule on
    # "mlip:src/b.cpp" finds 'ZZ_OPTIM' (optim_target) at
    # "mlip:src/b.cpp:<line>" when extract() is given the MLIP directory,
    # and without it reports "mlip:src/b.cpp: optim: the rule reads a
    # source file of MLIP, and no MLIP source directory is given (--mlip)";
    # main() takes --mlip.
    # Before the fix: fails, extract() takes no MLIP directory (TypeError)
    # and main() refuses --mlip.
    def test_a_rule_reads_the_mlip_directory(self):
        with tempfile.TemporaryDirectory() as d:
            hpo, mlip = pathlib.Path(d) / "hpo", pathlib.Path(d) / "mlip"
            (hpo / "src").mkdir(parents=True)
            (mlip / "src").mkdir(parents=True)
            (hpo / "src" / "a.cpp").write_text(SOURCE)
            (mlip / "src" / "b.cpp").write_text(
                "bool optim(const std::string& k)\n{\n"
                "    return k == \"ZZ_OPTIM\";\n}\n")
            rule = cov.Rule("optim", "mlip:src/b.cpp", r"^bool optim\(",
                            operands={"k": ("optim_target",)})
            with mock.patch.object(cov, "EXTRACTION", [rule]):
                found, problems = cov.extract(hpo, mlip=mlip)
                self.assertEqual(problems, [])
                self.assertEqual(
                    [(f.name, f.vocabularies, f.where) for f in found],
                    [("ZZ_OPTIM", ("optim_target",),
                      "mlip:src/b.cpp:3 (optim)")])
                found, problems = cov.extract(hpo)
                self.assertEqual(found, [])
                self.assertEqual(problems, [
                    "mlip:src/b.cpp: optim: the rule reads a source file "
                    "of MLIP, and no MLIP source directory is given "
                    "(--mlip)"])
                out = io.StringIO()
                with contextlib.redirect_stdout(out):
                    code = cov.main([str(hpo), "--mlip", str(mlip),
                                     "--registry",
                                     str(HERE.parent / "blueprint")])
                self.assertEqual(code, 1)
                self.assertIn("missing from the registry: 'ZZ_OPTIM' "
                              "(optim_target), mlip:src/b.cpp:3 (optim)",
                              out.getvalue())


class GeneratedTables(unittest.TestCase):
    """A function that reads a vocabulary from the generated tables of
    the registry (hpotarget_registry_gen.h) instead of naming it."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "g.cpp").write_text(textwrap.dedent('''\
            bool group_key(const std::string& key)
            {
                for (const R::Key& k : R::kKeys)
                    if (!k.group.empty() && k.name == key) return true;
                return false;
            }
            bool other(const std::string& key) { return key == "X"; }
            '''))

    def tearDown(self):
        self._tmp.cleanup()

    # Pins: a rule's `generated` names a use of a generated table (a
    # regular expression) and the vocabularies it reads; where the
    # function holds that use, every name of those vocabularies of the
    # registry counts as one the parser recognises (compare(), on the
    # names extract() reports as generated), and no name is missing; the
    # use is reported with its place. Before the fix: fails, Rule takes no
    # `generated`.
    def test_a_vocabulary_read_from_the_generated_tables(self):
        rule = cov.Rule("the group keys", "src/g.cpp",
                        r"^bool group_key\(",
                        generated=[(r"R::kKeys\b", ("group_key",))],
                        ignore={"k.name": "a name of the generated table"})
        generated: list = []
        with mock.patch.object(cov, "EXTRACTION", [rule]):
            found, problems = cov.extract(self.root, generated=generated)
        self.assertEqual(problems, [])
        self.assertEqual(found, [])
        self.assertEqual(generated, [(("group_key",),
                                      "src/g.cpp:3 (the group keys)")])
        reg = FakeRegistry({"group_key": {"LIB", "ALGO"}, "block": {"A"}})
        with mock.patch.object(cov, "VOCABULARIES", ("group_key", "block")):
            problems = cov.compare(
                [cov.Found("A", ("block",), "src/g.cpp:7 (other)")], reg,
                generated=generated)
        self.assertEqual(problems, [])

    # Pins: a rule whose function holds no use of its generated table is
    # refused ("the rule finds no use of the generated table"), as a rule
    # that finds no name is. Before the fix: fails, Rule takes no
    # `generated`.
    def test_a_generated_table_the_function_does_not_read(self):
        rule = cov.Rule("other", "src/g.cpp", r"^bool other\(",
                        operands={"key": ("block",)},
                        generated=[(r"R::kBlocks\b", ("block",))])
        with mock.patch.object(cov, "EXTRACTION", [rule]):
            _, problems = cov.extract(self.root, generated=[])
        self.assertEqual(problems, [
            "src/g.cpp: other: the rule finds no use of the generated "
            "table /R::kBlocks\\b/: the extraction rule needs updating"])


@dataclasses.dataclass
class FakeArg:
    words: list


@dataclasses.dataclass
class FakeEntry:
    id: str
    args: list


class HandlerWords(unittest.TestCase):
    """The handlers of the registry-driven parser read the word of a word
    argument with read_word(), which takes the words from the generated
    tables: a handler that calls it recognises the words of the arguments
    of its entry of the registry (the key, line option or form whose C++
    name, gen_hpotarget.cxx_name(), is the handler's)."""

    HANDLERS = textwrap.dedent('''\
        void Handlers::ROOT_ISO(KeyRead& r)
        {
            r.p().t.iso =
                read_word(arg0(r.key), single_value(r.iss, r.word), r.word);
        }
        void Handlers::ROOT_N(KeyRead& r)
        {
            r.p().t.n = 1;
        }
        void Handlers::GATE_SEED(OptionRead& r)
        {
            r.seed = read_word(arg0(r.option), r.value, r.key) == "WARN";
        }
        void Handlers::ROOT_GONE(KeyRead& r)
        {
            read_word(arg0(r.key), "x", r.word);
        }
        ''')

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = pathlib.Path(self._tmp.name)
        (self.root / "src").mkdir()
        (self.root / "src" / "h.cpp").write_text(self.HANDLERS)

    def tearDown(self):
        self._tmp.cleanup()

    # Pins: a rule's handler_words finds each definition of a handler
    # (`void Handlers::<name>(`) whose body calls read_word(), with the
    # place of the call; compare() counts the words of the arguments of the handler's
    # entry as recognised (the entry of the key ROOT.ISO for ROOT_ISO, of
    # the line option GATE.SEED for GATE_SEED), a handler that calls no
    # read_word() recognises none ('OTHER' of ROOT.N is not in the parser),
    # and a handler whose name is no entry's is a problem.
    # Before the fix: fails, Rule takes no handler_words.
    def test_the_words_a_handler_reads(self):
        rule = cov.Rule("the handlers", "src/h.cpp", handler_words=True,
                        operands={"read_word(arg0(r.option), r.value, "
                                  "r.key)": ("word",)})
        reads: list = []
        with mock.patch.object(cov, "EXTRACTION", [rule]):
            found, problems = cov.extract(self.root, handler_reads=reads)
        self.assertEqual(problems, [])
        self.assertEqual(reads, [("ROOT_ISO", "src/h.cpp:4 (the handlers)"),
                                 ("GATE_SEED",
                                  "src/h.cpp:12 (the handlers)"),
                                 ("ROOT_GONE",
                                  "src/h.cpp:16 (the handlers)")])
        reg = FakeRegistry({"word": {"PROCESS", "THREAD", "WARN", "ERROR",
                                     "OTHER"}})
        reg.entries = lambda: [
            FakeEntry("ROOT.ISO", [FakeArg(["PROCESS", "THREAD"])]),
            FakeEntry("ROOT.N", [FakeArg(["OTHER"])]),
            FakeEntry("GATE.SEED", [FakeArg(["ERROR", "WARN"])])]
        with mock.patch.object(cov, "VOCABULARIES", ("word",)):
            problems = cov.compare(found, reg, handler_reads=reads)
        self.assertEqual(problems, [
            "the handler Handlers::ROOT_GONE at src/h.cpp:16 (the handlers) "
            "reads a word, and no entry of the registry has its name",
            "not in the parser: 'OTHER' (word) is declared in the registry, "
            "and the parser's sources do not recognise it"])


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
        reg = FakeRegistry({"block": {"A"}, "word": {"ON"}})
        problems = self.compare(self.found(("A", ("block",))), reg)
        self.assertIn("no extraction rule reads the vocabulary word",
                      problems)

    def test_empty_vocabulary_needs_no_rule(self):
        # A vocabulary the registry leaves empty (the keys of TYPE PLAIN)
        # needs no rule that reads it.
        reg = FakeRegistry({"block": {"A"}})
        self.assertEqual(self.compare(self.found(("A", ("block",))), reg),
                         [])

    def test_unread_literal_unknown_to_the_registry(self):
        reg = FakeRegistry({"block": {"A"}, "word": {"ON"},
                            "registry_name": {"A", "ON"}})
        found = self.found(("A", ("block",)), ("ON", ("word",)))
        with mock.patch.object(cov, "VOCABULARIES", self.VOCAB):
            problems = cov.compare(found, reg, [("NEWKEY", "src/a.cpp:3"),
                                                ("ON", "src/a.cpp:4")])
        self.assertEqual(len(problems), 1)
        self.assertIn("unclassified literal: 'NEWKEY' at src/a.cpp:3",
                      problems[0])

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

    # Pins (MLIP's files): the rules on mlip: files, the OPTIM lines,
    # find their functions and tables in the MLIP sources of
    # TADAH_MLIP_SOURCE_DIR, and the HPO sources hold no OPTIM rule's
    # function. Before the fix: fails, extract() takes no MLIP directory.
    def test_rules_match_the_sources(self):
        _, problems = cov.extract(HERE.parent, mlip=mlip_source_dir(self))
        self.assertEqual(problems, [])

    # Pins: the target file has one parser, the registry-driven one
    # (src/hpotarget_parser.cpp, its handlers and the readers they call):
    # no rule reads the hand-written parser of src/hpo_targets.cpp
    # (ParsedTarget::read and the readers only it called); the one rule on
    # that file reads the constructor of the targets (the environment
    # override TADAH_OPTIM_SCALE and its words).
    # Before the fix: fails, six rules read ParsedTarget::read and its
    # readers.
    def test_no_rule_reads_the_hand_written_parser(self):
        on = [r.label for r in cov.EXTRACTION
              if r.file == "src/hpo_targets.cpp"]
        self.assertEqual(on, ["the OPTIM_SCALE override"])


class TopLevelKeys(unittest.TestCase):
    """A top-level key the registry-driven parser compares by name
    (src/hpotarget_parser.cpp, operand `name`) is read from the sources
    and compared with the keys of the top level of the registry."""

    # Pins: on a copy of the files the rules read, with a comparison
    # `name == "ZZTOPKEY"` added to src/hpotarget_parser.cpp (a top-level
    # key only the code holds), the check against the registry exits 1 and
    # reports 'ZZTOPKEY' as missing from the registry, as a key of the top
    # level (key:ROOT), at src/hpotarget_parser.cpp.
    # Before the fix: the test read is_top_level_key() of the hand-written
    # parser (src/hpo_targets.cpp).
    def test_a_name_only_the_code_holds_is_reported(self):
        with tempfile.TemporaryDirectory() as d:
            root, mlip = pathlib.Path(d) / "hpo", pathlib.Path(d) / "mlip"
            mlip_src = mlip_source_dir(self)
            for f in sorted({r.file for r in cov.EXTRACTION}):
                if f.startswith(cov.MLIP_PREFIX):
                    f = f[len(cov.MLIP_PREFIX):]
                    (mlip / f).parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(mlip_src / f, mlip / f)
                    continue
                (root / f).parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(HERE.parent / f, root / f)
            src = root / "src" / "hpotarget_parser.cpp"
            text = src.read_text()
            src.write_text(text + "\nbool zz_top_level(const std::string& "
                           "name)\n{\n    return name == \"ZZTOPKEY\";\n}\n")
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                code = cov.main([str(root), "--mlip", str(mlip),
                                 "--registry",
                                 str(HERE.parent / "blueprint")])
        self.assertEqual(code, 1, out.getvalue())
        self.assertIn("missing from the registry: 'ZZTOPKEY' (key:ROOT), "
                      "src/hpotarget_parser.cpp:", out.getvalue())


if __name__ == "__main__":
    unittest.main(verbosity=2)
