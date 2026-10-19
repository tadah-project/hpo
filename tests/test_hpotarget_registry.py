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

[lexing]
doc = "Lines are read one by one."
comment = "#"
continuation = "\\"
key_case = "exact"
error_frame = "Error on line {line}: {what}\nLine: {text}"
unfinished_msg = "unfinished line {line}"
missing_file_msg = "HPO file {file} not found"
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
end_without_open_msg = "{end} without matching {name}"
end_mismatch_msg = "{end} does not close {open} of line {open_line}"
end_takes_no_value_msg = "{end} closes {name} (unexpected '{tok}')"
open_takes_no_value_msg = "{name} opens a block (unexpected '{tok}')"
unclosed_msg = "{name} block not closed with {end}"
unknown_root_msg = "unknown key \"{key}\""
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
text = """
{line}
OPTIMIZER
LIB NLOPT
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

[[key]]
id = "OPTIMIZER.SEED"
name = "SEED"
group = "OPTIMIZER"
synopsis = "SEED <n>"
repeat = "last_wins"
doc = "The seed of the optimiser."
args = [{ name = "n", kind = "integer", min = 0, max = "UINT64_MAX" }]

[[key.invalid]]
wrap = "inner_a"
text = "SEED 1"
error = "SEED is not accepted"

[wrapper.inner_a]
doc = "The slot inside INNER."
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

[[loss_function]]
id = "LOSS.HUBER"
name = "HUBER"
doc = "The Huber loss."
params = [{ name = "delta", default = 50.0, gt = 0.0 }]

[[rule]]
id = "RULE.FAILSCORE_with_GRID"
doc = "A rule between two entries."
check = "startup"
keys = ["ROOT.FAILSCORE", "PARAM.TADAH:GRID.GRID_POINTS"]
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
                      "optimizers_gen.toml": OPTIMIZERS}

    def tearDown(self):
        self._tmp.cleanup()

    def write(self):
        for name, text in self.files.items():
            (self.dir / name).write_text(textwrap.dedent(text))

    def load(self, **kw):
        self.write()
        return hr.load(self.dir, **kw)

    def errors(self, **kw) -> str:
        self.write()
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
        self.assertEqual(len(cases), 4)
        ex = next(c for c in cases if c.kind == "example")
        self.assertEqual((ex.owner, ex.part, ex.check, ex.at),
                         ("ROOT.FAILSCORE", "hpotarget.toml", "startup", 1))
        inv = [c for c in cases if c.kind == "invalid"]
        self.assertTrue(all(c.check == "parse" for c in inv))
        self.assertEqual([c.owner for c in reg.cases("part_a.toml")],
                         ["OPTIMIZER.SEED"])

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
        self.assertIn("check 'run' is not parse or startup", e)
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

    # ---- the command ---------------------------------------------------------
    def test_check_command(self):
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


if __name__ == "__main__":
    unittest.main(verbosity=2)
