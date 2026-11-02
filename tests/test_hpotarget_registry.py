#!/usr/bin/env python3
"""Unit tests of blueprint/hpotarget_registry.py, the loader of the
target-file registry, on synthetic registries written to a temporary
directory: what it returns, and each of its checks (unknown tables and
fields, types, required fields, duplicate ids, references in a full and a
partial load, placeholders, wrappers, examples, the impersonal voice of
documentation text) and the --check command."""

from __future__ import annotations

import contextlib
import io
import json
import pathlib
import sys
import tempfile
import textwrap
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "blueprint"))

import hpotarget_registry as hr  # noqa: E402

MAIN = r'''
[meta]
version = 1
api = 1
parts = ["part_a.toml", "part_b.toml"]
anneal_keys = "anneal_keys.toml"
optimizers = "optimizers_gen.toml"
optimizer_docs = "optimizers_doc.toml"

[abbreviations]
MLSL = "multi-level single-linkage"

[lexing]
doc = "Lines are read one by one."
comment = "#"
continuation = "\\"
key_case = "exact"
error_frame = "line {line} of {file}: {what}\nLine: {text}"
unfinished_msg = "the last line continues and the file has no next line"
missing_file_msg = "the target file {file} cannot be opened"
bool_true = ["true"]
bool_false = ["false"]
timespec_words = ["off"]

[reading]
doc = "Numbers are decimal."
integer_msg = "{what} must be an integer >= {min}, got '{tok}'"
integer_range_msg = "{what} must be an integer from {min} to {max}, got '{tok}'"
note_msg = " ({note})"
finite_msg = "{what} must be a finite number, got '{tok}'{underflow}"
positive_msg = "{what} must be a positive {noun}, got '{tok}'{underflow}"
underflow_note = " (too small)"
requires_value_msg = "{key} requires a value"
single_value_msg = "{key} takes a single value (unexpected '{tok}')"
bool_msg = "{what}: expected boolean, got '{tok}'"
timespec_msg = "{what} must be a time, got '{tok}'"
script_timeout_msg = "{what} must be positive, got '{tok}'"
script_timeout_large_msg = "{what} is too large, got '{tok}'"

[structure]
doc = "Blocks nest."
check_order = ["block_lines", "removed", "inner_refusal", "group_keys",
               "init_restart_keys", "noise_keys", "log_hp", "strategy_keys",
               "block_unknown", "root_keys"]
open_order = ["alias", "parents", "requires", "repeat", "key_alone", "open"]
close_order = ["match", "key_alone", "close_steps", "close"]
line_steps = { ROOT = ["repeat", "value", "options"], group = ["value", "repeat", "leftover"], block = ["value", "repeat"] }
eof_order = ["open_block", "k_total_caps", "script_timeout_unread",
             "loss_unread", "metric_target_loss", "density_key_arming",
             "archive_directories", "optimizer_required",
             "observable_with_scripts", "hotvar_injected", "hotvar_unique",
             "observable_numerical", "observable_analytical",
             "script_gradients"]
end_without_open_msg = "{end} without matching {name}"
end_mismatch_msg = "{end} does not close {open} of line {open_line}"
end_takes_no_value_msg = "{end} closes {name} (unexpected '{tok}')"
open_takes_no_value_msg = "{name} opens a block (unexpected '{tok}')"
unclosed_msg = "{name} block not closed with {end}"
unknown_root_msg = "unknown key \"{key}\""
root_key_in_block_msg = "{key} is a top-level key inside {name} of line {open_line} ({end})"
removed_msg = "'{key}' was removed from the grammar: {hint}"

[group.OPTIMIZER]
doc = "The optimiser keys."
blocks = ["OPTIMIZER", "INNER"]
leftover_msg = "{key} takes no further value here (unexpected '{tok}')"

[[block]]
name = "OPTIMIZER"
end = "ENDOPTIMIZER"
parents = ["ROOT"]
repeat = "error"
repeat_msg = "OPTIMIZER again (first on line {first_line})"
misplaced_msg = "OPTIMIZER inside {parent} of line {parent_line}"
unknown_msg = "unknown key '{key}' in this block"
doc = "The search."

[[block.invalid]]
wrap = "root"
text = "OPTIMIZER"
error = "OPTIMIZER again (first on line 2)"

[[block]]
name = "INNER"
end = "ENDINNER"
parents = ["OPTIMIZER"]
repeat = "merge"
misplaced_msg = "INNER outside OPTIMIZER"
unknown_msg = "unknown key '{key}' inside INNER block (only {keys} are valid)"
refuse_msg = "{key} is not accepted inside an INNER block: {why}; set {key} on the enclosing {enclosing} block"
refuse = [{ key = "SEED", why = "the enclosing run draws it" }]
aliases = { STRATEGY = "OPTIMIZER.LIB" }
doc = "The local search."

[wrapper.root]
doc = "The slot first."
block = "ROOT"
text = """
{line}
OPTIMIZER
LIB NLOPT
ENDOPTIMIZER
"""

[wrapper.optimizer]
doc = "The slot inside OPTIMIZER."
block = "OPTIMIZER"
text = """
OPTIMIZER
LIB NLOPT
{line}
ENDOPTIMIZER
"""

[wrapper.inner]
doc = "The slot inside INNER."
block = "INNER"
text = """
OPTIMIZER
LIB NLOPT
INNER
{line}
ENDINNER
ENDOPTIMIZER
"""

[context.default]
doc = "A model."
config = "ATOM H\n"

[[key]]
id = "ROOT.FAILSCORE"
name = "FAILSCORE"
blocks = ["ROOT"]
synopsis = "FAILSCORE <score>"
repeat = "last_wins"
doc = "The score of a failed evaluation."
args = [{ name = "score", kind = "positive", noun = "finite number", what = "{key}" }]
default = { literal = "1e6" }

[[key.examples]]
wrap = "root"
text = "FAILSCORE 10"

[[key.invalid]]
wrap = "root"
text = "FAILSCORE 0"
error = "FAILSCORE must be a positive finite number, got '0'"
'''

PART_A = r'''
[[key]]
id = "OPTIMIZER.LIB"
name = "LIB"
group = "OPTIMIZER"
synopsis = "LIB <library>"
repeat = "last_wins"
doc = "The library of the optimiser."
args = [{ name = "library", kind = "word", words = ["NLOPT", "TADAH"] }]

[[key.examples]]
wrap = "optimizer"
text = "LIB TADAH"

[[key.examples]]
wrap = "optimizer"
text = "LIB NLOPT"
doc = "The other library."

[[key.invalid]]
wrap = "optimizer"
text = "LIB NLOPT TADAH"
error = "LIB takes no further value here (unexpected 'TADAH')"

[[key]]
id = "OPTIMIZER.SEED"
name = "SEED"
group = "OPTIMIZER"
synopsis = "SEED <n>"
repeat = "last_wins"
doc = "The seed of the optimiser."
args = [{ name = "n", kind = "integer", min = 0, max = "UINT64_MAX" }]

[[key.examples]]
wrap = "optimizer"
text = "SEED 7"

[[key.invalid]]
wrap = "inner_a"
text = "SEED 1"
error = "SEED is not accepted inside an INNER block: the enclosing run draws it; set SEED on the enclosing OPTIMIZER block"

[wrapper.inner_a]
doc = "The slot inside INNER."
block = "INNER"
context = "ctx_a"
text = """
OPTIMIZER
INNER
{line}
ENDINNER
ENDOPTIMIZER
"""

[context.ctx_a]
doc = "Another model."
config = "ATOM Ta\n"
has_validation = false
'''

PART_B = r'''
[[param]]
id = "PARAM.TADAH:GRID.GRID_POINTS"
name = "GRID_POINTS"
optimizers = ["TADAH:GRID"]
doc = "Points per dimension."
args = [{ name = "n", kind = "integer", min = 2 }]
checks = [{ msg = "GRID_POINTS must be at least 2", doc = "A value below 2." }]

[[param.examples]]
wrap = "optimizer"
text = "PARAM GRID_POINTS 5"

[[param.invalid]]
wrap = "optimizer"
at = "ENDOPTIMIZER"
text = "PARAM GRID_POINTS 1"
error = "GRID_POINTS must be at least 2"

[[loss_function]]
id = "LOSS.HUBER"
name = "HUBER"
doc = "The Huber loss."
params = [{ name = "delta", default = 50.0, gt = 0.0 }]

[[rule]]
id = "RULE.FAILSCORE_with_GRID"
title = "A rule between two entries"
doc = "A rule between two entries."
check = "startup"
keys = ["ROOT.FAILSCORE", "PARAM.TADAH:GRID.GRID_POINTS"]
'''

OPTIMIZER_DOCS = '''
[library.TADAH]
doc = "The optimisers of Tadah! itself."

[optimizer."TADAH:GRID"]
doc = "A search of the points of a regular grid."
'''

OPTIMIZERS = '''
[meta]
schema = 1

[[optimizer]]
lib = "TADAH"
algo = "GRID"
status = "registered"

[[optimizer]]
lib = "TADAH"
algo = "NOPE"
status = "refused"
'''


class Registry(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.files = {"hpotarget.toml": MAIN, "part_a.toml": PART_A,
                      "part_b.toml": PART_B,
                      "optimizers_gen.toml": OPTIMIZERS,
                      "optimizers_doc.toml": OPTIMIZER_DOCS}

    def tearDown(self):
        self._tmp.cleanup()

    def write(self):
        for name, text in self.files.items():
            (self.dir / name).write_text(textwrap.dedent(text))

    def load(self, **kw):
        self.write()
        kw.setdefault("templates", False)
        return hr.load(self.dir, **kw)

    def errors(self, **kw) -> str:
        self.write()
        kw.setdefault("templates", False)
        with self.assertRaises(hr.RegistryError) as cm:
            hr.load(self.dir, **kw)
        return "\n".join(cm.exception.errors)

    def edit(self, name, old, new):
        self.assertIn(old, self.files[name])
        self.files[name] = self.files[name].replace(old, new, 1)

    # ---- what it returns -------------------------------------------------
    def test_loads_every_file_into_dataclasses(self):
        reg = self.load()
        self.assertTrue(reg.full)
        self.assertEqual(reg.files, ["hpotarget.toml", "part_a.toml",
                                     "part_b.toml"])
        self.assertEqual([b.name for b in reg.blocks], ["OPTIMIZER", "INNER"])
        self.assertEqual({k.id for k in reg.keys},
                         {"ROOT.FAILSCORE", "OPTIMIZER.LIB", "OPTIMIZER.SEED"})
        fs = next(k for k in reg.keys if k.name == "FAILSCORE")
        self.assertIsInstance(fs, hr.Key)
        self.assertEqual(fs.scope, "ROOT")
        self.assertEqual(fs.default, hr.Default(literal="1e6"))
        self.assertEqual(fs.args[0].kind, "positive")
        self.assertEqual(reg.block("INNER").refuse,
                         [hr.Refusal("SEED", "the enclosing run draws it")])
        self.assertEqual(reg.params[0].id, "PARAM.TADAH:GRID.GRID_POINTS")
        self.assertEqual(reg.loss_functions[0].params[0].default, 50.0)
        self.assertEqual(reg.rules[0].keys,
                         ["ROOT.FAILSCORE", "PARAM.TADAH:GRID.GRID_POINTS"])
        self.assertFalse(reg.contexts["ctx_a"].has_validation)
        self.assertEqual(reg.notes, [])

    def test_cases_take_their_defaults_and_their_file(self):
        reg = self.load()
        cases = reg.cases()
        self.assertEqual(len(cases), 10)
        ex = next(c for c in cases if c.kind == "example")
        self.assertEqual((ex.owner, ex.part, ex.check, ex.at),
                         ("ROOT.FAILSCORE", "hpotarget.toml", "startup", 1))
        inv = [c for c in cases if c.kind == "invalid"]
        self.assertTrue(all(c.check == "parse" for c in inv))
        self.assertEqual([c.owner for c in reg.cases("part_a.toml")],
                         ["OPTIMIZER.LIB", "OPTIMIZER.LIB", "OPTIMIZER.LIB",
                          "OPTIMIZER.SEED", "OPTIMIZER.SEED"])

    def test_vocabularies(self):
        reg = self.load()
        self.assertEqual(reg.vocabulary("block"), {"OPTIMIZER", "INNER"})
        self.assertEqual(reg.vocabulary("block_end"),
                         {"ENDOPTIMIZER", "ENDINNER"})
        self.assertEqual(reg.vocabulary("key:ROOT"), {"FAILSCORE"})
        self.assertEqual(reg.vocabulary("key:OPTIMIZER"), set())
        self.assertEqual(reg.vocabulary("group_key"), {"LIB", "SEED"})
        self.assertEqual(reg.vocabulary("inner_refused"), {"SEED"})
        self.assertEqual(reg.vocabulary("alias"), {"STRATEGY"})
        self.assertEqual(reg.vocabulary("param"), {"GRID_POINTS"})
        self.assertEqual(reg.vocabulary("loss"), {"HUBER"})
        self.assertTrue({"NLOPT", "TADAH", "true", "false", "off"}
                        <= reg.vocabulary("word"))
        self.assertEqual([k.id for k in reg.keys_in_block("INNER")],
                         ["OPTIMIZER.LIB", "OPTIMIZER.SEED"])
        with self.assertRaises(KeyError):
            reg.vocabulary("nothing")

    # ---- strictness --------------------------------------------------------
    def test_unknown_field(self):
        self.edit("part_b.toml", 'name = "HUBER"', 'name = "HUBER"\ncolour = 1')
        self.assertIn("unknown field 'colour'", self.errors())

    def test_unknown_table(self):
        self.files["part_b.toml"] += "\n[[keys]]\nid = 'X'\n"
        self.assertIn("[keys]: unknown table", self.errors())

    def test_main_only_table_in_a_part(self):
        self.files["part_b.toml"] += "\n[group.MORE]\nblocks = []\n"
        self.assertIn("[group]: belongs in hpotarget.toml", self.errors())

    def test_missing_required_field(self):
        self.edit("part_a.toml", 'synopsis = "LIB <library>"\n', "")
        self.assertIn("required field 'synopsis' is missing", self.errors())

    def test_wrong_type(self):
        self.edit("part_a.toml", 'repeat = "last_wins"', "repeat = 3")
        self.assertIn("field 'repeat' is not str", self.errors())

    def test_unknown_argument_kind_and_misplaced_limits(self):
        self.edit("part_b.toml", 'kind = "integer", min = 2',
                  'kind = "count", min = 2')
        e = self.errors()
        self.assertIn("kind 'count' is not one of", e)
        self.assertIn("min and max belong to a number", e)

    def test_word_argument_lists_its_words(self):
        self.edit("part_a.toml", ', words = ["NLOPT", "TADAH"]', "")
        self.assertIn("an argument of kind word lists its words",
                      self.errors())

    def test_an_example_writes_one_of_the_words(self):
        # Guard: the word an example writes at the place of a word
        # argument is one of its words, compared in any case; a word left
        # out of `words` while an example shows it fails (a mutation of
        # the registry's BUMPGATE, words = ["CANDIDATE"] with an example of
        # BUMPGATE STRICT, passed every check of the registry).
        self.edit("part_a.toml", 'words = ["NLOPT", "TADAH"]',
                  'words = ["TADAH"]')
        self.assertIn("OPTIMIZER.LIB: part_a.toml OPTIMIZER.LIB example 1 "
                      "writes NLOPT where the argument <library> takes one "
                      "of its words (TADAH)", self.errors())
        self.edit("part_a.toml", 'words = ["TADAH"]',
                  'words = ["NLOPT", "TADAH"]')
        self.edit("part_a.toml", 'text = "LIB TADAH"', 'text = "LIB tadah"')
        self.assertEqual(self.load().keys[1].examples[0].text, "LIB tadah")

    def test_each_word_is_shown_by_an_example(self):
        # Guard: each word of a word argument is shown by an example, so a
        # word left out of `words` is caught through it; the lines of a
        # wrapper are not an example's (the root wrapper has LIB NLOPT
        # after its slot), and a partial load does not see every example.
        self.edit("part_a.toml", '[[key.examples]]\nwrap = "optimizer"\n'
                                 'text = "LIB NLOPT"\n'
                                 'doc = "The other library."\n', "")
        self.assertIn("OPTIMIZER.LIB: no example shows the word NLOPT of "
                      "the argument <library>: add an example that writes "
                      "it", self.errors())
        partial = self.load(parts=["part_a.toml"], resolve=False)
        self.assertFalse(partial.full)
        self.assertIn("OPTIMIZER.LIB", {k.id for k in partial.keys})

    def test_duplicate_id_across_files(self):
        self.edit("part_b.toml", 'id = "LOSS.HUBER"\nname = "HUBER"',
                  'id = "LOSS.HUBER"\nname = "HUBER"')
        self.files["part_a.toml"] += textwrap.dedent('''
            [[loss_function]]
            id = "LOSS.HUBER"
            name = "HUBER"
            doc = "Again."
            ''')
        self.assertIn("id 'LOSS.HUBER' is also the id of an entry",
                      self.errors())

    def test_same_key_name_valid_in_a_common_block(self):
        self.files["part_b.toml"] += textwrap.dedent('''
            [[key]]
            id = "INNER.SEED"
            name = "SEED"
            blocks = ["INNER"]
            synopsis = "SEED <n>"
            repeat = "last_wins"
            doc = "Another seed."
            ''')
        self.assertIn("SEED is also valid in INNER as OPTIMIZER.SEED",
                      self.errors())

    def test_every_entry_kind_has_an_invalid_line(self):
        self.edit("part_b.toml", '''
[[param.invalid]]
wrap = "optimizer"
at = "ENDOPTIMIZER"
text = "PARAM GRID_POINTS 1"
error = "GRID_POINTS must be at least 2"
''', "")
        self.assertIn("PARAM.TADAH:GRID.GRID_POINTS: an entry has an invalid "
                      "line", self.errors())

    def test_line_option_repeat_msg_goes_with_error(self):
        option = textwrap.dedent('''
            [[line_option]]
            id = "SEED.WIDE"
            option = "WIDE"
            lines = ["SEED"]
            synopsis = "WIDE"
            repeat = "error"
            doc = "A wider draw."
            ''')
        self.files["part_b.toml"] += option
        self.assertIn("SEED.WIDE: repeat_msg goes with repeat error, and "
                      "only with it", self.errors())
        self.files["part_b.toml"] = self.files["part_b.toml"].replace(
            'repeat = "error"\ndoc = "A wider', 'repeat = "last_wins"\n'
            'repeat_msg = "WIDE twice"\ndoc = "A wider')
        self.assertIn("SEED.WIDE: repeat_msg goes with repeat error, and "
                      "only with it", self.errors())

    def test_key_id_is_its_scope_and_name(self):
        self.edit("hpotarget.toml", 'id = "ROOT.FAILSCORE"',
                  'id = "TOP.FAILSCORE"')
        self.assertIn("id is <scope>.FAILSCORE", self.errors())

    def test_key_has_a_group_or_blocks(self):
        self.edit("part_a.toml", 'group = "OPTIMIZER"\nsynopsis = "LIB',
                  'group = "OPTIMIZER"\nblocks = ["INNER"]\nsynopsis = "LIB')
        self.assertIn("a key has a group or blocks", self.errors())

    def test_unknown_block_parent_and_group(self):
        self.edit("hpotarget.toml", 'parents = ["OPTIMIZER"]',
                  'parents = ["STRATEGY"]')
        self.edit("part_a.toml", 'group = "OPTIMIZER"', 'group = "OPT"')
        e = self.errors()
        self.assertIn("parent 'STRATEGY' is not ROOT or a block", e)
        self.assertIn("group 'OPT' is not a group", e)

    def test_block_end_and_repeat_message(self):
        self.edit("hpotarget.toml", 'end = "ENDINNER"', 'end = "END_INNER"')
        self.edit("hpotarget.toml", 'repeat = "merge"',
                  'repeat = "merge"\nrepeat_msg = "again"')
        e = self.errors()
        self.assertIn("end is ENDINNER", e)
        self.assertIn("repeat_msg goes with repeat error", e)

    def test_check_order_is_the_parsers(self):
        self.edit("hpotarget.toml", '"removed", "inner_refusal"',
                  '"inner_refusal", "removed"')
        self.assertIn("check_order is not", self.errors())

    # ---- what the parser's engine reads ------------------------------------
    # Pins: [structure] gives the order of the steps of the parse:
    # open_order (a line that opens a block), close_order (an END line),
    # eof_order (the end of the file), each an order of the steps the
    # loader names (each once, none other), and line_steps, the steps of a
    # key's line at the top level, in a group and in a block (names of
    # LINE_STEPS, each once, value among them, leftover for a group only).
    # Before the fix: the four fields are unknown fields of [structure].
    def test_structure_orders_of_the_parse(self):
        reg = self.load()
        st = reg.structure.fields
        self.assertEqual(st["open_order"], hr.OPEN_STEPS)
        self.assertEqual(st["close_order"], hr.CLOSE_STEPS)
        self.assertEqual(st["eof_order"], hr.EOF_STEPS)
        self.assertEqual(st["line_steps"]["group"],
                         ["value", "repeat", "leftover"])
        self.edit("hpotarget.toml", '"requires", "repeat", "key_alone"',
                  '"key_alone", "requires", "repeat"')
        self.assertEqual(self.load().structure.fields["open_order"],
                         ["alias", "parents", "key_alone", "requires",
                          "repeat", "open"])
        self.edit("hpotarget.toml", '"key_alone", "requires", "repeat"',
                  '"requires", "requires", "repeat"')
        self.assertIn("open_order is not an order of alias, parents, "
                      "requires, repeat, key_alone, open (each once)",
                      self.errors())
        self.edit("hpotarget.toml", '"requires", "requires", "repeat"',
                  '"requires", "repeat", "key_alone"')
        self.edit("hpotarget.toml", '"close_steps", "close"]', '"close"]')
        self.assertIn("close_order is not an order of", self.errors())
        self.edit("hpotarget.toml", '"close"]', '"close_steps", "close"]')
        self.edit("hpotarget.toml", '"open_block", ', '"open_block", "eof", ')
        self.assertIn("eof_order is not an order of", self.errors())
        self.edit("hpotarget.toml", '"open_block", "eof", ', '"open_block", ')
        self.edit("hpotarget.toml", 'block = ["value", "repeat"] }',
                  'block = ["repeat", "leftover"], INIT = ["value"] }')
        e = self.errors()
        self.assertIn("line_steps has the places ROOT, group, block, each "
                      "once", e)
        self.assertIn("line_steps.block: the steps hold value", e)
        self.assertIn("line_steps.block: leftover is a step of a key of a "
                      "group", e)
        self.edit("hpotarget.toml",
                  'block = ["repeat", "leftover"], INIT = ["value"] }',
                  'block = ["value", "value", "parse"] }')
        e = self.errors()
        self.assertIn("line_steps.block: 'parse' is not one of repeat, "
                      "value, options, leftover", e)
        self.assertIn("line_steps.block: a step is given twice", e)
        self.edit("hpotarget.toml", 'block = ["value", "value", "parse"] }',
                  'block = ["value", "repeat"] }')
        self.edit("hpotarget.toml", 'open_order = ["alias", "parents", '
                  '"requires", "repeat", "key_alone", "open"]\n', "")
        self.assertIn("required field 'open_order' is missing",
                      self.errors())

    # Pins: a key's own `steps` replace those of its place (a metric line
    # reads its values before its repeat), with the rules of line_steps.
    # Before the fix: steps is an unknown field of a key.
    def test_key_steps(self):
        self.edit("hpotarget.toml", 'repeat = "last_wins"\ndoc = "The score',
                  'repeat = "last_wins"\nsteps = ["value", "repeat", '
                  '"options"]\ndoc = "The score')
        fs = next(k for k in self.load().keys if k.name == "FAILSCORE")
        self.assertEqual(fs.steps, ["value", "repeat", "options"])
        self.assertEqual(next(k for k in self.load().keys
                              if k.name == "LIB").steps, [])
        self.edit("hpotarget.toml", 'steps = ["value", "repeat", "options"]',
                  'steps = ["repeat", "leftover"]')
        e = self.errors()
        self.assertIn("ROOT.FAILSCORE: steps: the steps hold value", e)
        self.assertIn("ROOT.FAILSCORE: steps: leftover is a step of a key of "
                      "a group", e)
        self.edit("hpotarget.toml", 'steps = ["repeat", "leftover"]', "")
        self.edit("part_a.toml", 'repeat = "last_wins"\ndoc = "The library',
                  'repeat = "last_wins"\nsteps = ["leftover", "value"]\n'
                  'doc = "The library')
        lib = next(k for k in self.load().keys if k.name == "LIB")
        self.assertEqual(lib.steps, ["leftover", "value"])

    # Pins: a block's close_steps name the checks its END line runs, in
    # order, each a step the loader names and each in one block only.
    # Before the fix: close_steps is an unknown field of a block.
    def test_block_close_steps(self):
        self.edit("hpotarget.toml", 'unknown_msg = "unknown key \'{key}\' in '
                  'this block"', 'unknown_msg = "unknown key \'{key}\' in '
                  'this block"\nclose_steps = ["optimizer_driver"]')
        self.assertEqual(self.load().block("OPTIMIZER").close_steps,
                         ["optimizer_driver"])
        self.assertEqual(self.load().block("INNER").close_steps, [])
        self.edit("hpotarget.toml", 'refuse_msg = "{key} is not accepted',
                  'close_steps = ["optimizer_driver", "inner_audit"]\n'
                  'refuse_msg = "{key} is not accepted')
        e = self.errors()
        self.assertIn("close_steps: 'inner_audit' is not one of "
                      "optimizer_driver, strategy_type_keys, "
                      "multistart_n_starts, restart_restarts, "
                      "explore_driver, inner_driver", e)
        self.assertIn("BLOCK.INNER: close_steps: 'optimizer_driver' is also "
                      "a step of BLOCK.OPTIMIZER (a step runs at one END "
                      "line)", e)

    # Pins: `at` of a close check of a block names the line its refusal
    # names (end, the default; key_line; last_end_line); a check of
    # another kind takes none.
    # Before the fix: at is an unknown field of a check.
    def test_close_check_at(self):
        self.edit("hpotarget.toml", 'doc = "The local search."',
                  'doc = "The local search."\nclose_checks = [\n'
                  '  { msg = "INNER needs LIB", doc = "At ENDINNER." },\n'
                  '  { msg = "INNER needs ALGO", doc = "At ENDINNER.", '
                  'at = "last_end_line" },\n]\n\n'
                  '[[block.invalid]]\nwrap = "inner"\ntext = "LIB NLOPT"\n'
                  'error = "INNER needs LIB"\n\n'
                  '[[block.invalid]]\nwrap = "inner"\ntext = "LIB TADAH"\n'
                  'error = "INNER needs ALGO"\n')
        checks = self.load().block("INNER").close_checks
        self.assertEqual([c.at for c in checks], ["end", "last_end_line"])
        self.edit("hpotarget.toml", 'at = "last_end_line"', 'at = "start"')
        self.assertIn("at 'start' is not one of end, key_line, "
                      "last_end_line", self.errors())
        self.edit("hpotarget.toml", 'at = "start"', 'at = "end"')
        self.edit("part_b.toml", 'doc = "A value below 2." }',
                  'doc = "A value below 2.", at = "key_line" }')
        self.assertIn("at belongs to a close check of a block",
                      self.errors())

    # Pins: `case` says how a word is matched: `exact` (the default) or
    # `any` case, on a word argument, a line option and a form with a word;
    # the boolean and loss readers match in any case by their kind.
    # Before the fix: case is an unknown field.
    def test_word_case(self):
        self.edit("part_a.toml", 'words = ["NLOPT", "TADAH"] }',
                  'words = ["NLOPT", "TADAH"], case = "any" }')
        reg = self.load()
        lib = next(k for k in reg.keys if k.name == "LIB")
        self.assertEqual(lib.args[0].case, "any")
        seed = next(k for k in reg.keys if k.name == "SEED")
        self.assertEqual(seed.args[0].case, "exact")
        self.edit("part_a.toml", 'case = "any" }', 'case = "upper" }')
        self.assertIn("case 'upper' is not one of exact, any", self.errors())
        self.edit("part_a.toml", ', case = "upper" }', ' }')
        self.edit("part_a.toml", 'max = "UINT64_MAX" }',
                  'max = "UINT64_MAX", case = "any" }')
        self.assertIn("case belongs to an argument of kind word",
                      self.errors())
        self.edit("part_a.toml", 'max = "UINT64_MAX", case = "any" }',
                  'max = "UINT64_MAX" }')
        self.files["part_b.toml"] += textwrap.dedent('''
            [[line_option]]
            id = "SEED.WIDE"
            option = "WIDE"
            lines = ["SEED"]
            synopsis = "WIDE"
            case = "any"
            refused = "Not read."
            doc = "A wider draw."

            [[line_option.invalid]]
            wrap = "optimizer"
            text = "SEED 1 WIDE"
            error = "SEED takes no further value here (unexpected 'WIDE')"

            [[form]]
            id = "SEED.word"
            line = "FAILSCORE"
            synopsis = "FAILSCORE <n>"
            case = "any"
            refused = "Not read."
            doc = "A form."

            [[form.invalid]]
            wrap = "root"
            text = "FAILSCORE 0"
            error = "FAILSCORE must be a positive finite number, got '0'"
            ''')
        e = self.errors()
        self.assertIn("id is FAILSCORE.<form>", e)
        self.assertIn("case belongs to a form with a word", e)
        self.edit("part_b.toml", 'id = "SEED.word"', 'id = "FAILSCORE.word"')
        self.edit("part_b.toml", 'synopsis = "FAILSCORE <n>"',
                  'word = "BIG"\nsynopsis = "FAILSCORE <n>"')
        reg = self.load(resolve=False)
        self.assertEqual(reg.line_options[0].case, "any")
        self.assertEqual(reg.forms[0].case, "any")

    # Pins: repeat error_until_read, of a key that a later line reads (the
    # second line is refused while no line between reads the first), takes
    # its repeat_msg; a line option has no such rule.
    # Before the fix: error_until_read is no repeat rule.
    def test_repeat_error_until_read(self):
        self.edit("hpotarget.toml", 'repeat = "last_wins"\ndoc = "The score',
                  'repeat = "error_until_read"\ndoc = "The score')
        self.assertIn("ROOT.FAILSCORE: repeat_msg goes with repeat error or "
                      "error_until_read, and only with them", self.errors())
        self.edit("hpotarget.toml", 'repeat = "error_until_read"\n',
                  'repeat = "error_until_read"\nrepeat_msg = "FAILSCORE '
                  'again (first on line {first_line})"\n')
        self.files["hpotarget.toml"] += textwrap.dedent('''
            [[key.invalid]]
            wrap = "root"
            text = "FAILSCORE 1\\nFAILSCORE 2"
            error = "FAILSCORE again (first on line 1)"
            ''')
        fs = next(k for k in self.load().keys if k.name == "FAILSCORE")
        self.assertEqual(fs.repeat, "error_until_read")
        self.files["part_b.toml"] += textwrap.dedent('''
            [[line_option]]
            id = "SEED.WIDE"
            option = "WIDE"
            lines = ["SEED"]
            synopsis = "WIDE"
            repeat = "error_until_read"
            refused = "Not read."
            doc = "A wider draw."
            ''')
        self.assertIn("SEED.WIDE: repeat 'error_until_read' is not one of "
                      "add, error, last_wins, warn_last_wins", self.errors())

    # Pins: `effect` of a key is the {effect} of the close check of
    # STRATEGY: a key that a strategy type lists has one, no other key.
    # Before the fix: effect is an unknown field of a key.
    def test_strategy_type_key_effect(self):
        self.files["part_b.toml"] += textwrap.dedent('''
            [[strategy_type]]
            id = "STRATEGY.TYPE.WIDE"
            name = "WIDE"
            keys = ["OPTIMIZER.SEED"]
            doc = "A wide search."
            ''')
        self.assertIn("OPTIMIZER.SEED: effect goes with a key a strategy "
                      "type lists, and only with it", self.errors())
        self.edit("part_a.toml", 'doc = "The seed of the optimiser."',
                  'doc = "The seed of the optimiser."\neffect = "only TYPE '
                  'WIDE draws"')
        seed = next(k for k in self.load().keys if k.name == "SEED")
        self.assertEqual(seed.effect, "only TYPE WIDE draws")
        self.edit("part_a.toml", 'effect = "only TYPE WIDE draws"',
                  'effect = "only {type} draws"')
        self.assertIn("effect: holds the placeholder {type}; the text goes "
                      "into a message as it is", self.errors())
        self.edit("part_a.toml", '\neffect = "only {type} draws"', "")
        self.edit("part_a.toml", 'doc = "The library of the optimiser."',
                  'doc = "The library of the optimiser."\neffect = "x"')
        self.assertIn("OPTIMIZER.LIB: effect goes with a key a strategy "
                      "type lists, and only with it", self.errors())

    # Pins: `what` of a rule gives the {what} of its message for a line of
    # each entry it names (what a key set in two blocks configures).
    # Before the fix: what is an unknown field of a rule.
    def test_rule_what(self):
        self.edit("part_b.toml", 'keys = ["ROOT.FAILSCORE", '
                  '"PARAM.TADAH:GRID.GRID_POINTS"]',
                  'keys = ["ROOT.FAILSCORE", "PARAM.TADAH:GRID.GRID_POINTS"]\n'
                  'msg = "{key} sets the {what} twice"\n'
                  'what = { "ROOT.FAILSCORE" = "score" }\n'
                  'untested = [{ msg = "{key} sets the {what} twice", why = '
                  '"A test." }]')
        rule = self.load().rules[0]
        self.assertEqual(rule.what, {"ROOT.FAILSCORE": "score"})
        self.edit("part_b.toml", 'what = { "ROOT.FAILSCORE" = "score" }',
                  'what = { "OPTIMIZER.LIB" = "{lib}" }')
        e = self.errors()
        self.assertIn("what: 'OPTIMIZER.LIB' is not one of its keys", e)
        self.assertIn("what.OPTIMIZER.LIB: holds the placeholder {lib}", e)
        self.edit("part_b.toml", 'what = { "OPTIMIZER.LIB" = "{lib}" }',
                  'what = { "ROOT.FAILSCORE" = "score" }')
        self.edit("part_b.toml", 'msg = "{key} sets the {what} twice"\n',
                  'msg = "{key} is set twice"\n')
        self.edit("part_b.toml", 'untested = [{ msg = "{key} sets the {what} '
                  'twice"', 'untested = [{ msg = "{key} is set twice"')
        self.assertIn("what fills the {what} of the message, which has none",
                      self.errors())

    # Pins: `explain` of a PARAM key is its line of the PARAM table of
    # `tadah explain`, a text without placeholders.
    # Before the fix: explain is an unknown field of a PARAM key.
    def test_param_explain(self):
        self.edit("part_b.toml", 'doc = "Points per dimension."',
                  'doc = "Points per dimension."\nexplain = "points per '
                  'dimension (>= 2)"')
        self.assertEqual(self.load().params[0].explain,
                         "points per dimension (>= 2)")
        self.assertEqual(self.load().keys[0].effect, "")
        self.edit("part_b.toml", 'explain = "points per dimension (>= 2)"',
                  'explain = "points per {name}"')
        self.assertIn("explain: holds the placeholder {name}", self.errors())

    def test_rule_id_and_check(self):
        self.edit("part_b.toml", 'id = "RULE.FAILSCORE_with_GRID"',
                  'id = "FAILSCORE_with_GRID"')
        self.edit("part_b.toml", 'check = "startup"', 'check = "run"')
        e = self.errors()
        self.assertIn("id is RULE.<name>", e)
        self.assertIn("check 'run' is not parse or startup", e)

    def test_rule_names_entries_of_the_registry(self):
        self.edit("part_b.toml", '"PARAM.TADAH:GRID.GRID_POINTS"]',
                  '"PARAM.TADAH:GRID.GRID_SIZE"]')
        self.assertIn("RULE.FAILSCORE_with_GRID: entry "
                      "'PARAM.TADAH:GRID.GRID_SIZE' is not in the registry",
                      self.errors())

    def test_optimiser_of_a_param_key(self):
        self.edit("part_b.toml", 'optimizers = ["TADAH:GRID"]',
                  'optimizers = ["TADAH:NOPE", "DLIB"]')
        e = self.errors()
        self.assertIn("optimiser 'TADAH:NOPE' is not '*', a LIB or a "
                      "registered LIB:ALGO", e)
        self.assertIn("optimiser 'DLIB' is not", e)

    # ---- references --------------------------------------------------------
    def test_reference_into_a_part_not_loaded_is_a_note(self):
        reg = self.load(parts=["part_b.toml"])
        self.assertFalse(reg.full)
        self.assertEqual(reg.files, ["hpotarget.toml", "part_b.toml"])
        self.assertEqual(len(reg.notes), 2)
        self.assertIn("alias key 'OPTIMIZER.LIB' not checked", reg.notes[0])
        self.assertIn("refused key of the group 'SEED' not checked",
                      reg.notes[1])

    def test_reference_to_nothing_is_an_error_in_a_full_load(self):
        self.edit("hpotarget.toml", 'aliases = { STRATEGY = "OPTIMIZER.LIB" }',
                  'aliases = { STRATEGY = "OPTIMIZER.LIBRARY" }')
        self.assertIn("alias key 'OPTIMIZER.LIBRARY' is not in the registry",
                      self.errors())

    def test_resolve_false_makes_it_a_note(self):
        self.edit("hpotarget.toml", 'aliases = { STRATEGY = "OPTIMIZER.LIB" }',
                  'aliases = { STRATEGY = "OPTIMIZER.LIBRARY" }')
        reg = self.load(resolve=False)
        self.assertEqual(len(reg.notes), 1)

    def test_part_not_in_meta_parts(self):
        self.assertIn("is not a part file of meta.parts",
                      self.errors(parts=["part_c.toml"]))

    def test_main_file_alone(self):
        reg = self.load(parts=[])
        self.assertEqual(reg.files, ["hpotarget.toml"])
        self.assertEqual(len(reg.keys), 1)

    # ---- placeholders, wrappers and examples ---------------------------------
    def test_placeholder_of_a_fixed_message(self):
        self.edit("hpotarget.toml", "OPTIMIZER inside {parent}",
                  "OPTIMIZER inside {tok}")
        self.assertIn("misplaced_msg: placeholder {tok} is not one of "
                      "parent, parent_line", self.errors())

    def test_placeholder_of_a_key_message(self):
        self.edit("hpotarget.toml", 'what = "{key}"', 'what = "{kye}"')
        self.assertIn("what: placeholder {kye} is not one of", self.errors())

    def test_placeholder_in_an_example_or_an_error(self):
        self.edit("hpotarget.toml", "got '0'\"", "got '{tok}'\"")
        self.assertIn("error: holds the placeholder {tok}", self.errors())

    def test_lammps_variable_is_not_a_placeholder(self):
        # ${name} is a LAMMPS variable reference: in a message, an example
        # or an error it is written as it is; {name} alone is a placeholder.
        self.edit("hpotarget.toml", "got '0'\"",
                  "got '0' (timer timeout ${maxtime})\"")
        self.edit("hpotarget.toml", "takes no further value here",
                  "(timer timeout ${maxtime}) takes no further value here")
        self.edit("part_a.toml", "LIB takes no further value here",
                  "LIB (timer timeout ${maxtime}) takes no further value "
                  "here")
        reg = self.load()
        self.assertIn("${maxtime}", reg.keys[0].invalid[0].error)
        self.assertIn("${maxtime}", reg.groups["OPTIMIZER"].leftover_msg)
        self.edit("hpotarget.toml", "(timer timeout ${maxtime}) takes",
                  "(timer timeout {maxtime}) takes")
        self.assertIn("leftover_msg: placeholder {maxtime} is not one of "
                      "key, tok", self.errors())

    def test_key_checks(self):
        self.edit("hpotarget.toml", 'default = { literal = "1e6" }',
                  'default = { literal = "1e6" }\n'
                  'checks = [{ msg = "{key} needs a score", '
                  'doc = "No value follows the key." }]')
        fs = next(k for k in self.load().keys if k.name == "FAILSCORE")
        self.assertEqual(fs.checks, [hr.Check("{key} needs a score",
                                              "No value follows the key.")])
        self.edit("hpotarget.toml", '"{key} needs a score"',
                  '"{kye} needs a score"')
        self.assertIn("ROOT.FAILSCORE.checks[0]: msg: placeholder {kye}",
                      self.errors())

    def test_param_policy_quirk(self):
        self.files["part_b.toml"] += textwrap.dedent('''
            [[param_policy]]
            id = "PARAM.TADAH:GRID"
            optimizers = ["TADAH:GRID"]
            doc = "GRID_POINTS is its only PARAM key."
            quirk = "A word value is accepted and has no effect."
            ''')
        reg = self.load()
        self.assertEqual(reg.param_policies[0].quirk,
                         "A word value is accepted and has no effect.")
        self.edit("part_b.toml", "A word value is accepted",
                  "Your word value is accepted")
        self.assertIn("PARAM.TADAH:GRID: quirk: 'Your' addresses the reader",
                      self.errors())

    def test_wrapper_has_one_slot_line(self):
        self.edit("part_a.toml", "INNER\n{line}\n", "INNER {line}\n")
        self.assertIn("a wrapper has one slot", self.errors())
        self.edit("part_a.toml", "INNER {line}\n", "INNER\n{line}\n{line}\n")
        self.assertIn("a wrapper has one slot", self.errors())

    def test_wrapper_context_must_exist_where_it_is_read(self):
        self.edit("part_a.toml", 'context = "ctx_a"', 'context = "ctx_b"')
        self.assertIn("context 'ctx_b' is not a [context.*] table",
                      self.errors())

    def test_case_names_a_wrapper(self):
        self.edit("hpotarget.toml", 'wrap = "root"\ntext = "FAILSCORE 10"',
                  'wrap = "top"\ntext = "FAILSCORE 10"')
        self.assertIn("wrap 'top' is not a wrapper", self.errors())

    def test_main_file_cannot_use_a_wrapper_of_a_part(self):
        self.edit("hpotarget.toml", 'wrap = "root"\ntext = "FAILSCORE 10"',
                  'wrap = "inner_a"\ntext = "FAILSCORE 10"')
        self.assertIn("wrapper 'inner_a' is in part_a.toml", self.errors())

    def test_invalid_case_fields(self):
        self.edit("hpotarget.toml", 'text = "FAILSCORE 0"',
                  'text = "FAILSCORE 0"\nat = 0\ncheck = "run"')
        e = self.errors()
        self.assertIn("at counts logical lines from 1", e)
        self.assertIn("check 'run' is not parse, startup or check", e)
        self.edit("hpotarget.toml", 'at = 0\ncheck = "run"', 'at = "2x"')
        self.assertIn("at '2x' is not a line number, a key or none",
                      self.errors())

    def test_invalid_case_needs_its_error(self):
        self.edit("hpotarget.toml",
                  "error = \"FAILSCORE must be a positive finite number, "
                  "got '0'\"\n", "")
        self.assertIn("required field 'error' is missing", self.errors())

    def test_documentation_is_impersonal(self):
        self.edit("part_b.toml", 'doc = "Points per dimension."',
                  'doc = "Set your points per dimension."')
        self.assertIn("'your' addresses the reader", self.errors())

    # ---- the text of the reference pages -------------------------------------
    def test_rule_has_a_title(self):
        # Pins: a rule has a title, the phrase the reference pages show as
        # its heading and as the text of a link to it: no full stop at its
        # end, at most 80 characters.
        # Before the fix: title is an unknown field of a rule.
        reg = self.load()
        self.assertEqual(reg.rules[0].title, "A rule between two entries")
        self.edit("part_b.toml", 'title = "A rule between two entries"',
                  'title = "A rule between two entries."')
        self.assertIn("RULE.FAILSCORE_with_GRID: title: a title is a phrase "
                      "without a full stop at its end", self.errors())
        self.edit("part_b.toml", 'title = "A rule between two entries."',
                  'title = "' + "x" * 81 + '"')
        self.assertIn("RULE.FAILSCORE_with_GRID: title: longer than 80 "
                      "characters", self.errors())
        self.edit("part_b.toml", 'title = "' + "x" * 81 + '"\n', "")
        self.assertIn("required field 'title' is missing", self.errors())

    def test_documentation_text_holds_no_placeholder(self):
        # Pins: {name} marks a value in a message only. Documentation text
        # (doc, long_doc, quirk, condition, title, a rule's msg_doc, the doc
        # of an argument or a case, a default's text) says in words what
        # the message holds; ${name}, a LAMMPS variable, is text. A check's
        # doc, which says what its message holds, may name them.
        # Before the fix: a placeholder in documentation text passed.
        self.edit("hpotarget.toml", 'default = { literal = "1e6" }',
                  'default = { literal = "1e6" }\n'
                  'checks = [{ msg = "{key} needs {what}", '
                  'doc = "{what} names the value." }]')
        self.edit("part_b.toml", 'doc = "A rule between two entries."',
                  'doc = "{list} names the entries."\n'
                  'msg_doc = "The {keys} of the message."')
        self.edit("part_b.toml", 'doc = "Points per dimension."',
                  'doc = "Points per dimension, {n} of them."')
        self.edit("part_a.toml", 'text = "SEED 1"',
                  'text = "SEED 1"\ndoc = "Refused with {why}."')
        self.edit("hpotarget.toml", 'doc = "The search."',
                  'doc = "The search; ${maxtime} is text."')
        e = self.errors()
        self.assertIn("RULE.FAILSCORE_with_GRID: doc: holds the placeholder "
                      "{list}; documentation text says in words what a "
                      "message holds", e)
        self.assertIn("PARAM.TADAH:GRID.GRID_POINTS: doc: holds the "
                      "placeholder {n}", e)
        self.assertIn("invalid[0]: doc: holds the placeholder {why}", e)
        self.assertIn("RULE.FAILSCORE_with_GRID: msg_doc: holds the "
                      "placeholder {keys}", e)
        self.assertNotIn("maxtime", e)
        self.assertNotIn("checks[0]: doc", e)

    def test_exclusive_minimum_and_integral_number(self):
        # Pins: min_open = true leaves min out of the range (a value above
        # min), integral = true takes whole numbers only; min_open needs a
        # min, integral an argument of kind number.
        # Before the fix: both are unknown fields of an argument.
        self.edit("part_b.toml", 'kind = "integer", min = 2 }',
                  'kind = "number", min = 2, min_open = true, '
                  'integral = true }')
        a = self.load().params[0].args[0]
        self.assertEqual((a.min, a.min_open, a.integral), (2, True, True))
        self.edit("part_b.toml", 'min = 2, min_open', 'min_open')
        self.assertIn("min_open needs a min", self.errors())
        self.edit("part_b.toml", 'kind = "number", min_open = true',
                  'kind = "integer", min = 2')
        self.assertIn("integral belongs to an argument of kind number",
                      self.errors())

    def test_an_example_after_the_first_has_a_doc(self):
        # Pins: the reference pages show an example's doc as the caption of
        # its code; the first example of an entry may go without one, each
        # later one says what it shows.
        # Before the fix: a second example without a doc passed.
        self.files["hpotarget.toml"] += textwrap.dedent('''
            [[key.examples]]
            wrap = "root"
            text = "FAILSCORE 20"
            ''')
        self.assertIn("ROOT.FAILSCORE example 1: an example after the first "
                      "of an entry has a doc", self.errors())
        self.files["hpotarget.toml"] += 'doc = "A larger score."\n'
        self.assertEqual(self.load().keys[0].examples[1].doc,
                         "A larger score.")

    def test_repeat_doc(self):
        # Pins: repeat_doc, when a key has it, is the repeat rule as the
        # reference shows it, for a key whose forms repeat differently.
        # Before the fix: repeat_doc is an unknown field of a key.
        self.edit("part_a.toml", 'repeat = "last_wins"\ndoc = "The library',
                  'repeat = "last_wins"\nrepeat_doc = "A later line '
                  'replaces the library."\ndoc = "The library')
        lib = next(k for k in self.load().keys if k.name == "LIB")
        self.assertEqual(lib.repeat_doc, "A later line replaces the library.")

    def test_lammps_script_section(self):
        # Pins: [lammps_script] says what a LAMMPS script of a PC_LAMMPS
        # line provides; its fixture is the script the tests run. It is
        # the entry LAMMPS_SCRIPT.
        # Before the fix: lammps_script is an unknown table.
        self.files["part_b.toml"] += textwrap.dedent('''
            [lammps_script]
            doc = "A script takes the candidate potential."
            fixture = "script_b"

            [fixture.script_b]
            path = "b.in"
            doc = "A script."
            text = "pair_style tadah\\npair_coeff * * EXTERN H\\n"
            ''')
        reg = self.load()
        self.assertEqual(reg.lammps_script.id, "LAMMPS_SCRIPT")
        self.assertEqual(reg.lammps_script.fields["fixture"], "script_b")
        self.assertIs(reg.by_id()["LAMMPS_SCRIPT"], reg.lammps_script)
        self.edit("part_b.toml", 'fixture = "script_b"',
                  'fixture = "script_c"')
        self.assertIn("[lammps_script]: fixture 'script_c' is not a "
                      "[fixture.*] table", self.errors())

    def test_optimizer_docs(self):
        # Pins: the file meta.optimizer_docs names describes every library
        # and every registered LIB:ALGO of the optimiser list; a name that
        # is no library or registered optimiser is an error, and its text
        # is impersonal.
        # Before the fix: optimizer_docs is an unknown field of [meta].
        reg = self.load()
        self.assertEqual(reg.optimizer_docs.optimizers,
                         {"TADAH:GRID": "A search of the points of a "
                                        "regular grid."})
        self.assertEqual(reg.optimizer_docs.libraries,
                         {"TADAH": "The optimisers of Tadah! itself."})
        self.files["optimizers_doc.toml"] = OPTIMIZER_DOCS.replace(
            '[optimizer."TADAH:GRID"]', '[optimizer."TADAH:NOPE"]').replace(
            "[library.TADAH]", "[library.NONE]").replace(
            "of Tadah! itself", "you run")
        e = self.errors()
        self.assertIn("optimizers_doc.toml: TADAH:GRID, a registered "
                      "optimiser, has no doc", e)
        self.assertIn("optimizers_doc.toml [optimizer.TADAH:NOPE]: is not a "
                      "registered optimiser of optimizers_gen.toml", e)
        self.assertIn("optimizers_doc.toml: the library TADAH has no doc", e)
        self.assertIn("optimizers_doc.toml [library.NONE]: is not a library "
                      "of optimizers_gen.toml", e)
        self.assertIn("'you' addresses the reader", e)

    def test_abbreviations(self):
        # Pins: [abbreviations] spells out the abbreviations of the
        # registry's text, for the reference pages to write out at their
        # first use on a page.
        # Before the fix: abbreviations is an unknown table.
        self.assertEqual(self.load().abbreviations,
                         {"MLSL": "multi-level single-linkage"})
        self.edit("hpotarget.toml", 'MLSL = "multi',
                  '"ML SL" = "multi')
        self.assertIn("abbreviation 'ML SL' is not one word", self.errors())

    # ---- the command ---------------------------------------------------------
    # ---- what the generated checks and the conformance test rely on ----
    def test_meta_version(self):
        self.edit("hpotarget.toml", "version = 1", "version = 7")
        self.assertIn("version 7 is not the layout this loader reads, 1",
                      self.errors())

    def test_wrapper_names_its_block(self):
        self.edit("hpotarget.toml", 'block = "OPTIMIZER"\n', "")
        self.assertIn("[wrapper.optimizer]: required field 'block' is "
                      "missing", self.errors())
        self.edit("part_a.toml", 'block = "INNER"', 'block = "OPTIMIZER"')
        self.assertIn("[wrapper.inner_a]: block is 'OPTIMIZER', but the slot "
                      "is in INNER", self.errors())

    def test_wrapper_names_a_registered_optimizer(self):
        self.edit("hpotarget.toml", 'block = "ROOT"\n',
                  'block = "ROOT"\noptimizer = "TADAH:NOPE"\n')
        self.assertIn("optimizer 'TADAH:NOPE' is not a registered LIB:ALGO",
                      self.errors())

    def test_each_place_has_its_wrapper(self):
        self.edit("hpotarget.toml", "[wrapper.inner]", "[wrapper.inner2]")
        self.assertIn("[wrapper.inner] is the wrapper of the block INNER",
                      self.errors())

    def test_key_needs_an_example_and_an_invalid_line(self):
        self.edit("part_a.toml", '[[key.examples]]\nwrap = "optimizer"\n'
                                 'text = "SEED 7"\n', "")
        self.assertIn("OPTIMIZER.SEED: a key that is not refused has an "
                      "example", self.errors())
        self.files["part_a.toml"] += textwrap.dedent('''
            [[key]]
            id = "OPTIMIZER.POPULATION"
            name = "POPULATION"
            group = "OPTIMIZER"
            synopsis = "POPULATION <n>"
            repeat = "last_wins"
            refused = "No optimiser here keeps a population."
            doc = "The population."
            ''')
        e = self.errors()
        self.assertIn("OPTIMIZER.POPULATION: a key has an invalid line", e)
        self.assertNotIn("OPTIMIZER.POPULATION: a key that is not refused",
                         e)

    def test_example_puts_the_key_in_one_of_its_blocks(self):
        # INNER refuses SEED: an example of SEED inside INNER is out of
        # place, and so is FAILSCORE inside OPTIMIZER.
        self.edit("part_a.toml", 'wrap = "optimizer"\ntext = "SEED 7"',
                  'wrap = "inner"\ntext = "SEED 7"')
        self.edit("hpotarget.toml", 'wrap = "root"\ntext = "FAILSCORE 10"',
                  'wrap = "optimizer"\ntext = "FAILSCORE 10"')
        e = self.errors()
        self.assertIn("example 0 puts SEED in INNER, which is not one of its "
                      "blocks (OPTIMIZER)", e)
        self.assertIn("example 0 puts FAILSCORE in OPTIMIZER, which is not "
                      "one of its blocks (ROOT)", e)
        self.edit("hpotarget.toml", 'wrap = "optimizer"\ntext = "FAILSCORE '
                                    '10"', 'wrap = "root"\ntext = "QFLUSH 1"')
        self.assertIn("example 0 has no FAILSCORE line", self.errors())

    def test_arguments_are_in_the_synopsis(self):
        self.edit("part_a.toml", 'synopsis = "SEED <n>"',
                  'synopsis = "SEED <count>"')
        self.assertIn("the argument <n> is not in the synopsis 'SEED "
                      "<count>'", self.errors())

    def test_doc_has_at_most_two_sentences(self):
        self.edit("part_b.toml", 'doc = "Points per dimension."',
                  'doc = "Points per dimension. Two. Three."')
        self.assertIn("PARAM.TADAH:GRID.GRID_POINTS: doc: 3 sentences",
                      self.errors())

    def test_long_doc_of_blocks_and_sections(self):
        self.edit("hpotarget.toml", 'doc = "The search."',
                  'doc = "The search."\nlong_doc = "The detail."')
        self.edit("hpotarget.toml", 'doc = "Blocks nest."',
                  'doc = "Blocks nest."\nlong_doc = "The detail."')
        reg = self.load()
        self.assertEqual(reg.block("OPTIMIZER").long_doc, "The detail.")
        self.assertEqual(reg.structure.long_doc, "The detail.")

    def test_error_is_a_message_of_the_registry(self):
        self.edit("hpotarget.toml",
                  "FAILSCORE must be a positive finite number, got '0'",
                  "FAILSCORE is wrong")
        self.assertIn("ROOT.FAILSCORE invalid 0: the error is no message of "
                      "the entry", self.errors())
        # msg_of names the entry whose message it is.
        self.edit("hpotarget.toml", 'error = "FAILSCORE is wrong"',
                  'error = "FAILSCORE is wrong"\nmsg_of = "RULE.nothing"')
        self.assertIn("msg_of 'RULE.nothing' is not an entry", self.errors())
        self.files["part_b.toml"] = self.files["part_b.toml"].replace(
            'check = "startup"\nkeys',
            'check = "startup"\nmsg = "{key} is wrong"\nkeys')
        self.edit("hpotarget.toml", 'msg_of = "RULE.nothing"',
                  'msg_of = "RULE.FAILSCORE_with_GRID"')
        self.load()

    def test_every_message_has_a_case_or_a_reason(self):
        self.give_every_message_a_reason()
        self.load(templates=True)
        self.edit("hpotarget.toml", "{ msg = \"the last line continues and the "
                                    "file has no next line\", why "
                                    "= \"Not exercised here.\" }, ", "")
        self.assertIn("LEXING: unfinished_msg: no invalid case shows the "
                      "message 'the last line continues and the file has no "
                      "next line'",
                      self.errors(templates=True))

    def test_untested_names_a_message_of_the_entry(self):
        self.give_every_message_a_reason()
        self.edit("hpotarget.toml",
                  '"the target file {file} cannot be opened"',
                  '"the target file is missing"')
        self.assertIn("untested: 'the target file is missing' is no message "
                      "of the "
                      "entry", self.errors(templates=True))

    def test_check_mode_shows_a_refusal_of_the_run(self):
        # A refusal of the run names the line of the file it is about, as
        # PC_WORKER_TIMEOUT's under PC_ISOLATION THREAD does, or none (at =
        # "none"), as an environment variable's does: both load. (The
        # loader refused a case of the run that named a line.)
        self.edit("hpotarget.toml", 'text = "FAILSCORE 0"',
                  'text = "FAILSCORE 0"\ncheck = "check"')
        self.load()
        self.edit("hpotarget.toml", 'check = "check"',
                  'check = "check"\nat = "none"')
        self.load()
        self.edit("hpotarget.toml", 'wrap = "root"\ntext = "FAILSCORE 10"',
                  'wrap = "root"\ncheck = "check"\ntext = "FAILSCORE 10"')
        self.assertIn("check 'check' is not parse or startup", self.errors())

    def test_unchecked_goes_with_a_text_default(self):
        self.edit("hpotarget.toml", 'default = { literal = "1e6" }',
                  'default = { literal = "1e6", unchecked = "why" }')
        self.assertIn("default.unchecked goes with a text default",
                      self.errors())
        self.edit("hpotarget.toml",
                  'default = { literal = "1e6", unchecked = "why" }',
                  'default = { text = "1e6", unchecked = "No setting holds '
                  'it." }')
        reg = self.load()
        self.assertEqual(reg.keys[0].default.unchecked,
                         "No setting holds it.")

    def test_helpers_compose_and_place_a_line(self):
        reg = self.load()
        case = reg.keys[1].examples[0]          # LIB TADAH in OPTIMIZER
        comp, ln, block = hr.key_line(reg, case, "LIB")
        self.assertEqual((comp.slot, ln.first, ln.text, block),
                         (3, 3, "LIB TADAH", "OPTIMIZER"))
        self.assertEqual(hr.block_wrappers(reg),
                         {"ROOT": "root", "OPTIMIZER": "optimizer",
                          "INNER": "inner"})
        lines = hr.logical_lines("A 1 \\\n  2 # c\n\nB\n")
        self.assertEqual([(x.first, x.last, x.text, x.key) for x in lines],
                         [(1, 2, "A 1   2", "A"), (4, 4, "B", "B")])
        rx = hr.template_regex("{key} is set twice (first on line "
                               "{first_line})", {"key": "QFLUSH"})
        self.assertTrue(rx.match("QFLUSH is set twice (first on line 3)"))
        self.assertFalse(rx.match("DIGITS is set twice (first on line 3)"))
        self.assertEqual(hr.sentences("One. Two, e.g. this. Three"), 2)

    def give_every_message_a_reason(self):
        """untested for every message no invalid case of the synthetic
        registry shows, so that a full --check passes."""
        def reasons(*msgs):
            items = ", ".join("{ msg = %s, why = \"Not exercised here.\" }"
                              % json.dumps(m) for m in msgs)
            return f"untested = [{items}]\n"
        self.edit("hpotarget.toml", '[lexing]\n', '[lexing]\n' + reasons(
            "the last line continues and the file has no next line",
            "the target file {file} cannot be opened"))
        self.edit("hpotarget.toml", '[reading]\n', '[reading]\n' + reasons(
            "{what} must be an integer >= {min}, got '{tok}'",
            "{what} must be an integer from {min} to {max}, got '{tok}'",
            " ({note})",
            "{what} must be a finite number, got '{tok}'{underflow}",
            "{key} requires a value",
            "{key} takes a single value (unexpected '{tok}')",
            "{what}: expected boolean, got '{tok}'",
            "{what} must be a time, got '{tok}'",
            "{what} must be positive, got '{tok}'",
            "{what} is too large, got '{tok}'"))
        self.edit("hpotarget.toml", '[structure]\n', '[structure]\n' +
                  reasons("{end} without matching {name}",
                          "{end} does not close {open} of line {open_line}",
                          "{end} closes {name} (unexpected '{tok}')",
                          "{name} opens a block (unexpected '{tok}')",
                          "{name} block not closed with {end}",
                          'unknown key "{key}"',
                          "{key} is a top-level key inside {name} of line "
                          "{open_line} ({end})",
                          "'{key}' was removed from the grammar: {hint}"))
        self.edit("hpotarget.toml", 'name = "OPTIMIZER"\n',
                  'name = "OPTIMIZER"\n' + reasons(
                      "OPTIMIZER inside {parent} of line {parent_line}",
                      "unknown key '{key}' in this block"))
        self.edit("hpotarget.toml", 'name = "INNER"\n',
                  'name = "INNER"\n' + reasons(
                      "INNER outside OPTIMIZER",
                      "unknown key '{key}' inside INNER block (only {keys} "
                      "are valid)"))

    def test_check_command(self):
        self.give_every_message_a_reason()
        self.write()
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hr.main(["--check", "--dir", str(self.dir)]), 0)
        self.assertIn("OK: hpotarget.toml, part_a.toml, part_b.toml", out.getvalue())
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            self.assertEqual(hr.main(["--check", "--dir", str(self.dir),
                                      "--part", "part_b.toml"]), 0)
        self.assertIn("note: BLOCK.INNER: alias key", out.getvalue())
        self.edit("part_b.toml", 'name = "HUBER"', 'name = "HUBER"\nx = 1')
        self.write()
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(hr.main(["--check", "--dir", str(self.dir)]), 1)
        self.assertIn("error: part_b.toml [[loss_function]] LOSS.HUBER: "
                      "unknown field 'x'", err.getvalue())

    def test_toml_error_names_the_file(self):
        self.files["part_b.toml"] += "\n[[param\n"
        self.assertIn("part_b.toml: not TOML", self.errors())

    def test_a_table_of_an_entry_sits_inside_it(self):
        # TOML gives a [[param.invalid]] table to the last [[param]] above
        # it, also after a [[loss_function]] and a [[rule]]: such a case
        # sits where a reader of the entry does not see it, and a [[param]]
        # added between would take it over. The loader refuses it, naming
        # the line, the entry and the table it follows.
        self.files["part_b.toml"] += (
            '\n[[param.invalid]]\nwrap = "optimizer"\nat = "ENDOPTIMIZER"\n'
            'text = "PARAM GRID_POINTS 0"\n'
            'error = "GRID_POINTS must be at least 2"\n')
        self.write()
        lines = (self.dir / "part_b.toml").read_text().splitlines()
        at = lines.index("[[param.invalid]]", lines.index("[[rule]]")) + 1
        rule = lines.index("[[rule]]") + 1
        param = lines.index("[[param]]") + 1
        self.assertIn(f"part_b.toml: line {at}: [[param.invalid]] belongs to "
                      f"the [[param]] entry of line {param}, but follows "
                      f"[[rule]] (line {rule}); the tables of an entry follow "
                      f"it, before the next entry", self.errors())
        # Inside its entry, the same table is accepted.
        self.files["part_b.toml"] = PART_B.replace(
            '[[param.examples]]',
            '[[param.invalid]]\nwrap = "optimizer"\nat = "ENDOPTIMIZER"\n'
            'text = "PARAM GRID_POINTS 0"\n'
            'error = "GRID_POINTS must be at least 2"\n\n[[param.examples]]', 1)
        self.load()


if __name__ == "__main__":
    unittest.main(verbosity=2)
