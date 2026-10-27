#!/usr/bin/env python3
"""Unit tests of blueprint/gen_hpotarget.py, the generator of the C++
tables of the target-file registry, on the synthetic registry of
test_hpotarget_registry.py: the C++ name of an id and the refusal of an id
without one or of two ids of one name, the escaping of a message, the
placeholders of the messages the loader fixes (taken from the loader, not
written into the generator), the tables of the two headers and what they
leave out, that two runs write the same text, the headers compiled and
read back by a C++ compiler, and --header and --check."""

from __future__ import annotations

import contextlib
import io
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import textwrap
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / "blueprint"))

import gen_hpotarget as gen  # noqa: E402
import hpotarget_registry as hr  # noqa: E402
import test_hpotarget_registry as synthetic  # noqa: E402


class Generator(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.dir = pathlib.Path(self._tmp.name)
        self.files = {"hpotarget.toml": synthetic.MAIN,
                      "part_a.toml": synthetic.PART_A,
                      "part_b.toml": synthetic.PART_B,
                      "optimizers_gen.toml": synthetic.OPTIMIZERS,
                      "optimizers_doc.toml": synthetic.OPTIMIZER_DOCS}

    def tearDown(self):
        self._tmp.cleanup()

    def write(self):
        for name, text in self.files.items():
            (self.dir / name).write_text(textwrap.dedent(text))

    def edit(self, name, old, new):
        self.assertIn(old, self.files[name])
        self.files[name] = self.files[name].replace(old, new, 1)

    def headers(self):
        self.write()
        return gen.headers(hr.load(self.dir, templates=False))

    def has(self, part, text):
        """`part` is in `text`, both with their white space runs as one
        blank (the headers wrap a long row)."""
        flat = re.sub(r"\s+", " ", text)
        want = re.sub(r"\s+", " ", part)
        if want not in flat:
            self.fail(f"not in the header: {part!r}")

    def lacks(self, part, text):
        if part in text:
            self.fail(f"in the header: {part!r}")

    def full_registry(self):
        """The synthetic registry as the loader's full check takes it
        (a reason for each message no invalid line shows), which
        gen_hpotarget.py's command line loads."""
        synthetic.Registry.give_every_message_a_reason(self)

    def main(self, *argv):
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = gen.main(list(argv))
        return code, out.getvalue(), err.getvalue()

    # Pins: an id's C++ name has '.' and ':' as '_' and no '-'; the keys,
    # the blocks and the PARAM keys of the registry are enumerators of
    # those names, in registry order.
    # Before the fix: fails, gen_hpotarget.py does not exist (the import
    # of this file fails).
    def test_names(self):
        self.assertEqual(gen.cxx_name("ROOT.FAILSCORE"), "ROOT_FAILSCORE")
        self.assertEqual(gen.cxx_name("PC_LAMMPS.--hotvar"),
                         "PC_LAMMPS_hotvar")
        self.assertEqual(gen.cxx_name("PARAM.ASA:ASA.Delta_X"),
                         "PARAM_ASA_ASA_Delta_X")
        self.assertEqual(
            gen.cxx_name("PC_OBSERVABLE.cohesive_energy.--max-wall-time"),
            "PC_OBSERVABLE_cohesive_energy_maxwalltime")
        h = self.headers()[gen.REGISTRY_HEADER]
        self.has("enum class KeyId : unsigned short {\n"
                 "  ROOT_FAILSCORE,\n  OPTIMIZER_LIB,\n  OPTIMIZER_SEED,\n"
                 "};", h)
        self.has("enum class BlockId : unsigned char {\n"
                 "  ROOT,\n  OPTIMIZER,\n  INNER,\n};", h)
        p = self.headers()[gen.PARAM_HEADER]
        self.has("enum class ParamId : unsigned short {\n"
                 "  PARAM_TADAH_GRID_GRID_POINTS,\n};", p)

    # Pins: an id that gives no identifier ('(', a leading digit, '__')
    # stops the generator naming it, and so do two ids of one C++ name,
    # named both, in a list and in a registry (two rules whose ids differ
    # by a '-').
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_names_refused(self):
        for ident in ("A.(B)", "1A.B", "A._B", "A.B__C"):
            with self.assertRaises(gen.GenerationError) as cm:
                gen.cxx_name(ident)
            self.assertIn(repr(ident), str(cm.exception))
        with self.assertRaises(gen.GenerationError) as cm:
            gen.cxx_names(["X.A-B", "X.AB"], "the test")
        self.assertIn("'X.A-B' and 'X.AB' both give the C++ name X_AB",
                      str(cm.exception))
        rule = self.files["part_b.toml"].split("[[rule]]")[1]
        self.files["part_b.toml"] += "\n[[rule]]" + rule.replace(
            'id = "RULE.FAILSCORE_with_GRID"',
            'id = "RULE.FAILSCORE_with-_GRID"')
        with self.assertRaises(gen.GenerationError) as cm:
            self.headers()
        message = ("'RULE.FAILSCORE_with_GRID' and "
                   "'RULE.FAILSCORE_with-_GRID' both give the C++ name "
                   "RULE_FAILSCORE_with_GRID")
        self.assertIn(message, str(cm.exception))
        self.full_registry()
        self.write()
        code, _, err = self.main("--check", "--registry", str(self.dir),
                                 "-o", str(self.dir))
        self.assertEqual(code, 1)
        self.assertIn(message, err)

    # Pins: a message is a C++ literal of its text: a backslash, a double
    # quote, a newline and a tab escaped, another control character in
    # three octal digits (so a following digit does not join it), UTF-8 as
    # it is; the registry's continuation (a backslash) and frame (a
    # newline) read so in the header.
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_escaping(self):
        self.assertEqual(gen.cstr('say "a\\b"\n\t\x017—'),
                         '"say \\"a\\\\b\\"\\n\\t\\0017—"')
        h = self.headers()[gen.REGISTRY_HEADER]
        self.has('inline constexpr std::string_view continuation = '
                 '"\\\\";', h)
        self.has('inline constexpr std::string_view error_frame = '
                 '"Error on line {line}: {what}\\nLine: {text}";', h)
        self.has('"unknown key \\"{key}\\""', h)

    # Pins: the placeholders of each message whose set the loader fixes
    # (FIXED_PLACEHOLDERS) are in the header, sorted, as the loader has
    # them: a frame the loader gives {file} has "file" among them, with no
    # change to the generator; the readers' and the entries' sets too.
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_placeholders_come_from_the_loader(self):
        h = self.headers()[gen.REGISTRY_HEADER]
        self.has("inline constexpr std::string_view "
                 "placeholders_lexing_error_frame[] = {\n"
                 '    "line",\n    "text",\n    "what",\n};', h)
        self.has("lexing_unfinished_msg = Span<std::string_view>"
                 "{nullptr, 0};", h)
        self.has("placeholders_reading[] = {", h)
        self.has("placeholders_entry[] = {", h)
        frame = ("lexing", "error_frame")
        saved = hr.FIXED_PLACEHOLDERS[frame]
        hr.FIXED_PLACEHOLDERS[frame] = saved | {"file"}
        try:
            self.edit("hpotarget.toml",
                      'error_frame = "Error on line {line}: {what}',
                      'error_frame = "Error on line {line} of {file}: {what}')
            h = self.headers()[gen.REGISTRY_HEADER]
        finally:
            hr.FIXED_PLACEHOLDERS[frame] = saved
        self.has("placeholders_lexing_error_frame[] = {\n"
                 '    "file",\n    "line",\n    "text",\n    "what",\n};',
                 h)
        self.has('"Error on line {line} of {file}: {what}\\nLine: '
                 '{text}"', h)

    # Pins: the tables hold the registry's data: the blocks with their
    # parents, aliases, refusals and the keys each takes (INNER: LIB, not
    # the SEED it refuses), the keys with their places, arguments, limits
    # (UINT64_MAX exactly), words and literal default, the top-level keys,
    # the loss functions, the rules, check_order, the PARAM keys and the
    # PARAM keys of each optimiser; documentation (doc, synopsis, the
    # sentence of a default), the examples and the invalid lines stay out.
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_tables(self):
        texts = self.headers()
        h, p = texts[gen.REGISTRY_HEADER], texts[gen.PARAM_HEADER]
        self.has("Alias{\"STRATEGY\", KeyId::OPTIMIZER_LIB}", h)
        self.has("Refusal{\"SEED\", \"the enclosing run draws it\"}", h)
        self.has("inline constexpr KeyId BLOCK_INNER_keys[] = {\n"
                 "    KeyId::OPTIMIZER_LIB,\n};", h)
        self.has("Block{BlockId::INNER, \"INNER\", \"ENDINNER\",\n"
                 "      block_bit(BlockId::OPTIMIZER), BlockRepeat::Merge,",
                 h)
        self.has("Limit{true, \"UINT64_MAX\", 1.8446744073709552e+19, "
                 "true,\n", h)
        self.has("18446744073709551615ull}", h)
        self.has("inline constexpr std::string_view "
                 "OPTIMIZER_LIB_arg0_words[] = {\n"
                 '    "NLOPT",\n    "TADAH",\n};', h)
        self.has("ArgKind::Positive", h)
        self.has('"1e6"', h)
        self.has("inline constexpr std::string_view top_level_keys[] = "
                 '{\n    "FAILSCORE",\n};', h)
        self.has("LossParam{\"delta\", 50.0, true, 0.0}", h)
        self.has("Rule{\"RULE.FAILSCORE_with_GRID\", RuleCheck::Startup, "
                 "\"\"}", h)
        self.has("CheckStage::BlockLines,\n    CheckStage::Removed,", h)
        self.has("inline constexpr Span<LineOption> kLineOptions"
                 "{nullptr, 0};", h)
        self.has("OptimiserParams{\"TADAH:GRID\", "
                 "Span<ParamId>{detail::optimiser0_params, 1}}", p)
        self.has('"GRID_POINTS must be at least 2"', p)
        self.has("#include <tadah/hpo/hpotarget_registry_gen.h>", p)
        for doc in ("The score of a failed evaluation.", "FAILSCORE <score>",
                    "FAILSCORE 10", "FAILSCORE 0", "A rule between two",
                    "Points per dimension.", "A value below 2."):
            self.lacks(doc, h + p)

    # Pins: two runs write the same text, which names neither the
    # registry's directory nor the time.
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_deterministic(self):
        first = self.headers()
        second = self.headers()
        self.assertEqual(first, second)
        for text in first.values():
            self.lacks(str(self.dir), text)
            self.lacks(self._tmp.name, text)

    # Pins: a registry the loader refuses, and a loader of another API,
    # give no header: exit 1 with the reason.
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_refused_registry(self):
        self.edit("hpotarget.toml", 'key_case = "exact"',
                  'key_case = "exact"\nunknown_field = 1')
        self.write()
        code, _, err = self.main("--header", "--registry", str(self.dir),
                                 "-o", str(self.dir / "out"))
        self.assertEqual(code, 1)
        self.assertIn("unknown_field", err)
        self.assertFalse((self.dir / "out").exists())
        self.edit("hpotarget.toml", 'key_case = "exact"\nunknown_field = 1',
                  'key_case = "exact"')
        saved = gen.LOADER_API
        gen.LOADER_API = saved + 1
        try:
            with self.assertRaises(gen.GenerationError) as cm:
                self.headers()
        finally:
            gen.LOADER_API = saved
        self.assertIn(f"the generator reads API {saved + 1}",
                      str(cm.exception))

    # Pins: --header writes both headers and --check passes on them; a
    # header that differs, or one that is missing, fails --check naming it
    # and the build target that regenerates it; --header again reports the
    # headers up to date.
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_header_and_check(self):
        self.full_registry()
        self.write()
        out = self.dir / "include"
        args = ["--registry", str(self.dir), "-o", str(out)]
        code, text, _ = self.main("--header", *args)
        self.assertEqual(code, 0)
        self.assertEqual(sorted(p.name for p in out.iterdir()),
                         sorted([gen.REGISTRY_HEADER, gen.PARAM_HEADER]))
        self.assertEqual(self.main("--check", *args)[0], 0)
        self.assertIn("is up to date", self.main("--header", *args)[1])
        header = out / gen.PARAM_HEADER
        header.write_text(header.read_text() + "// edited\n")
        code, _, err = self.main("--check", *args)
        self.assertEqual(code, 1)
        self.assertIn(f"{header} differs from what the registry gives", err)
        self.assertIn("--target hpo_regenerate_registries", err)
        (out / gen.REGISTRY_HEADER).unlink()
        code, _, err = self.main("--check", *args)
        self.assertEqual(code, 1)
        self.assertIn(f"{out / gen.REGISTRY_HEADER} does not exist", err)

    # Pins: the headers of the synthetic registry, whose tables of line
    # options and forms are empty, compile as C++17 without a warning
    # (-Wall -Wextra -pedantic -Werror), twice included; read back, the
    # continuation, the frame, an escaped message and a UTF-8 one are the
    # registry's text, the INNER block takes LIB, and fill() puts values
    # into a template (${name} kept). Needs a C++ compiler: $CXX, which
    # the CTest entry sets, else c++ or g++ on the PATH.
    # Before the fix: fails, gen_hpotarget.py does not exist.
    def test_headers_compile(self):
        cxx = os.environ.get("CXX") or shutil.which("c++") or \
            shutil.which("g++")
        if not cxx:
            self.skipTest("no C++ compiler: set CXX")
        self.edit("hpotarget.toml", 'doc = "The search."',
                  'doc = "The search."\nunreachable_msgs = '
                  '["a \\"quoted\\" \\\\ text — ${maxtime}"]')
        texts = self.headers()
        inc = self.dir / "inc" / "tadah" / "hpo"
        inc.mkdir(parents=True)
        for name, text in texts.items():
            (inc / name).write_text(text, encoding="utf-8")
        src = self.dir / "read_back.cpp"
        src.write_text(textwrap.dedent("""\
            #include <tadah/hpo/hpo_param_registry_gen.h>
            #include <tadah/hpo/hpo_param_registry_gen.h>
            #include <tadah/hpo/hpotarget_registry_gen.h>
            #include <iostream>
            namespace R = tadah::hpo::hpotarget_registry;
            int main()
            {
                std::cout << R::lexing::continuation << '|'
                          << R::lexing::error_frame << '|'
                          << R::kBlocks[0].unreachable_msgs[0] << '|'
                          << R::kKeys[static_cast<int>(
                                 R::kBlocks[1].keys[0])].name << '|'
                          << R::fill(R::kBlocks[0].unreachable_msgs[0], {})
                          << '|'
                          << R::fill("{a}-{b}", {{"a", "1"}, {"b", "2"}})
                          << '|' << R::kLineOptions.size
                          << '|'
                          << tadah::hpo::param_registry::kParams[0].name;
            }
            """), encoding="utf-8")
        exe = self.dir / "read_back"
        build = subprocess.run(
            [cxx, "-std=c++17", "-Wall", "-Wextra", "-pedantic", "-Werror",
             "-I", str(self.dir / "inc"), str(src), "-o", str(exe)],
            capture_output=True, text=True)
        self.assertEqual(build.returncode, 0, build.stderr)
        run = subprocess.run([str(exe)], capture_output=True)
        self.assertEqual(run.returncode, 0)
        quoted = 'a "quoted" \\ text — ${maxtime}'
        self.assertEqual(
            run.stdout.decode("utf-8"),
            "|".join(["\\", "Error on line {line}: {what}\nLine: {text}",
                      quoted, "LIB", quoted, "1-2", "0", "GRID_POINTS"]))


if __name__ == "__main__":
    unittest.main()
