#!/usr/bin/env python3
"""Generated checks of the target-file registry against the parser.

    check_hpotarget_generated.py --runner RUNNER [--registry DIR]
                                 [--only CHECK] [--keep DIR]

The conformance test (test_hpotarget_registry_conformance.cpp) runs the
examples and invalid lines written in the registry. This check derives
more cases from the structured fields of the registry, runs them with
hpotarget_case_runner (tests/tools/hpotarget_case_runner.cpp), which
reads each file with the parser of `tadah hpo`, and compares what the
parser did with what the registry says. Exits 1 and lists each
difference, 0 when there is none. --only runs one check; --keep keeps the
case and result files in DIR.

The checks, and the change of the registry each one catches
-----------------------------------------------------------
blocks    The line of each key, taken from its first example, is put into
          the wrapper of each of the nine places a line can be (ROOT and
          the eight blocks: hpotarget_registry.block_wrappers). Where the
          registry lists the place for a key of that name (its blocks, or
          the blocks of its group less those that refuse it), the parser
          must not refuse the line as out of place; elsewhere it must,
          with the block's refusal message, the key's own misplaced_msg,
          the top-level unknown-key message at the top level, and in a
          block the structure's root_key_in_block_msg for a key the
          registry lists at the top level only, the block's unknown-key
          message for any other key (MAXTIME, which the blocks of an
          optimiser take as well, included). Catches a block added to or
          left out of `blocks`, a key in the wrong group, a lost entry of
          a name the parser reads in more places (ROOT.MAXTIME), and a
          key the parser names as a top-level key in a block that the
          registry does not list at the top level only, or the reverse.
repeat    A key whose repeat is error, last_wins or warn_last_wins: its
          line, from its first example, is given again right after it.
          error: the second line is refused with repeat_msg ({first_line}
          the line of the first, {block} its block, {key} the key); the
          other two: the file is read. The same for a line option, given
          twice on its line (an option word is found in any case, as
          the parser reads it). Catches a wrong repeat rule or
          repeat_msg.
          A rule `add` is not generated: a line that adds may still be
          refused twice with the same value (two OPTIM lines of one
          index), which the entry's own cases show.
ranges    A key, INIT method, PARAM key or line option with one argument
          of kind integer, number or positive and a min or max: the line
          of each of its examples (of the first, for a line option) with
          the value min, max (accepted, unless min_open) and the values
          past them (refused at that line, with the argument's msg, or
          [reading]'s integer_msg, integer_range_msg or positive_msg,
          filled). The max runs as its example does, through the startup
          checks unless the example says parse, so a value the parser
          takes and the run cannot use (a count of starting points drawn
          when the file is read) fails; the other values are read only.
          A PARAM value, which the startup checks read, runs through them
          at every value: in the range it starts in the file of one of the
          entry's examples at least (ASA's User_Initial_Parameters 0 only
          in a search without a starting point), and past it it is
          refused in each, at parse or at startup, with a message of its
          entry or of [reading]. Catches a changed min, max, min_open or
          kind, and a max the run refuses.
words     A word argument at a fixed place of its line: each word of
          `words`, put into the line of the first example, is accepted;
          each of `refused_words` is refused with its message. Catches a
          word added to `words` that the parser refuses, and a refused
          word it accepts. A word left out of `words` is caught by
          hpotarget_registry.py --check, which requires each word to be
          shown by an example and the word each example writes at the
          argument's place to be one of `words`.
params    A [[param]] whose optimisers are named (not *): its first
          example runs in each wrapper whose `optimizer` is one of them
          (accepted when tadah hpo starts), and in each wrapper of an
          optimiser that refuses unknown PARAM keys (its [[param_policy]]
          has a msg) but is not named (refused, naming the key). A PARAM
          name of several entries (tol_gradinf of DLIB:BFGS and of CERES)
          is listed on the optimisers of all of them. Catches an
          optimiser added to or left out of `optimizers`.
lines     A line option of a top-level line (PC_LAMMPS, PC_FAILPROBE, the
          metric lines): the option, with its values from its first
          example, is appended to the first example of every other such
          line; where the option's `lines` holds that line it is read
          there, elsewhere the line is refused naming the option as the
          example writes it. Catches a line added to or left out of
          `lines`.
defaults  hpotarget_case_runner reads three files (a file of the root
          wrapper; a file with RHOGATE and RHOBARGATE lines; a file with a
          PC_LAMMPS --failvar line and a PC_FAILPROBE line) and reports
          the value the code gives each entry with a default (its table
          `defaults_of`). Every literal default must equal the reported
          value (numbers compared as numbers). Every number written in a
          text default must be one of the reported values, unless the
          default says why no setting holds it (`unchecked`), and every
          number the runner reports must be written in the text. A
          literal default the runner does not report, a text default
          with numbers and neither a reported value nor `unchecked`, or
          a reported value with no default, fails. Catches a changed
          default.

What it cannot see
------------------
- the place of a key that has no example (the loader requires one for
  every key that is not refused), and a block of a place the nine
  wrappers do not cover (a block inside an INIT block of EXPLORE);
- a repeat rule `add`; a key or option whose line the generated file
  makes invalid for another reason before the parser reaches it (each
  check reports a case the parser refuses for another reason at another
  line as a failure, so such a case is visible, not silent);
- the range of an argument that is not the only value of its line or
  option, of a [[form]] argument and of a loss parameter; words of an
  argument after an optional or repeated one; a word the parser reads
  that is left out of `words` together with every example that shows it;
- a value the startup checks refuse in some files only: the max of a
  range runs in the files of its examples, and its min is read only;
- PARAM keys read by every optimiser (`optimizers = ["*"]`), and the
  PARAM keys of the annealers (anneal_keys.toml has its own check);
- the options of a form (OPTIM AUTO, PC_OBSERVABLE cohesive_energy) on
  another line: they are words of their form, which the parser reads
  nowhere else;
- defaults the code applies where a value is used (value_or): THREADS and
  GRAD_STRATEGY are reported as the runner's table repeats them; a text
  default without a number is not compared, and the numbers of a text
  default with `unchecked` that the runner does not report are taken as
  written.
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from typing import Callable

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_REGISTRY = HERE.parent / "blueprint"
NUMBER = re.compile(r"(?<![A-Za-z_\d.])[-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
                    r"(?![A-Za-z_\d])")
CHECK_NAMES = ("blocks", "repeat", "ranges", "words", "params", "lines",
               "defaults")


# ------------------------------------------------------------ TOML out ----

def toml_string(s: str) -> str:
    out = ['"']
    for ch in s:
        o = ord(ch)
        if ch == '"':
            out.append('\\"')
        elif ch == "\\":
            out.append("\\\\")
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif o < 0x20 or o == 0x7f:
            out.append(f"\\u{o:04x}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def write_cases(path: pathlib.Path, cases: list[dict]) -> None:
    lines = []
    for c in cases:
        lines.append("[[case]]")
        for k in ("id", "mode", "file", "config", "validation"):
            if k in c:
                lines.append(f"{k} = {toml_string(c[k])}")
        for k in ("has_train_db", "has_validation"):
            if k in c:
                lines.append(f"{k} = {'true' if c[k] else 'false'}")
        if c.get("env"):
            items = ", ".join(f"{k} = {toml_string(v)}"
                              for k, v in sorted(c["env"].items()))
            lines.append(f"env = {{ {items} }}")
        if c.get("fixtures"):
            items = ", ".join(
                f"{{ path = {toml_string(f['path'])}, "
                f"text = {toml_string(f['text'])} }}" for f in c["fixtures"])
            lines.append(f"fixtures = [{items}]")
        lines.append("")
    path.write_text("\n".join(lines))


# ---------------------------------------------------------- the checks ----

@dataclass
class Gen:
    """A generated case: the runner's case and what to make of its
    result (a failure text, or None). Cases of one `group` pass when one
    of them passes."""
    check: str
    label: str
    case: dict
    judge: Callable[[dict], str | None]
    group: str | None = None


@dataclass
class Generator:
    hr: object                      # the hpotarget_registry module
    reg: object
    cases: list[Gen] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)

    # ---- composing files ----------------------------------------------
    def fixtures(self) -> list[dict]:
        return [{"path": f.path, "text": f.text}
                for f in self.reg.fixtures.values()]

    def context(self, name: str) -> dict:
        c = self.reg.contexts[name]
        return {"config": c.config, "has_train_db": c.has_train_db,
                "has_validation": c.has_validation}

    def add(self, check: str, label: str, file: str, context: str,
            mode: str, judge: Callable[[dict], str | None],
            env: dict | None = None, group: str | None = None) -> None:
        case = {"id": f"{check}:{len(self.cases)}:{label}", "mode": mode,
                "file": file, "fixtures": self.fixtures(),
                **self.context(context)}
        if env:
            case["env"] = env
        self.cases.append(Gen(check, label, case, judge, group))

    def frame_what(self, error: str, line: int) -> str | None:
        """The {what} of `error` when it is framed at `line`, else None."""
        frame = self.reg.lexing.fields["error_frame"]
        rx = self.hr.template_regex(frame, {"line": str(line)})
        if not rx.match(error):
            return None
        pre = frame.split("{what}")[0].replace("{line}", str(line))
        post = frame.split("{what}")[1]
        rest = error[len(pre):]
        cut = post.split("{text}")[0]
        i = rest.rfind(cut)
        return rest[:i] if i >= 0 else rest

    @staticmethod
    def replace_line(file: str, ln, new_text: str) -> str:
        lines = file.split("\n")
        lines[ln.first - 1:ln.last] = [new_text]
        return "\n".join(lines)

    @staticmethod
    def insert_after(file: str, ln, new_text: str) -> str:
        lines = file.split("\n")
        lines[ln.last:ln.last] = [new_text]
        return "\n".join(lines)

    # ---- places -------------------------------------------------------
    def place_blocks(self, k) -> set[str]:
        reg = self.reg
        if k.group in reg.groups:
            return {b for b in reg.groups[k.group].blocks
                    if k.name not in {r.key for r in reg.block(b).refuse}}
        return set(k.blocks)

    def placement_patterns(self, place: str, name: str,
                           places: set[str]) -> list[re.Pattern]:
        """The refusals of the key `name` in `place`, which the registry
        does not list for it; `places` are those it lists."""
        hr, reg = self.hr, self.reg
        out = []
        if place == "ROOT":
            out.append(hr.template_regex(
                reg.structure.fields["unknown_root_msg"], {"key": name}))
        else:
            b = reg.block(place)
            if places == {"ROOT"}:
                # A key of the top level only is named as a top-level key,
                # with the block that is still open, and not as unknown.
                out.append(hr.template_regex(
                    reg.structure.fields["root_key_in_block_msg"],
                    {"key": name, "name": place, "end": b.end}))
            else:
                out.append(hr.template_regex(b.unknown_msg, {"key": name}))
            for r in b.refuse:
                if r.key == name:
                    out.append(hr.template_regex(
                        b.refuse_msg, {"key": name, "why": r.why}))
        for k in reg.keys:
            if k.name == name and k.misplaced_msg:
                out.append(hr.template_regex(k.misplaced_msg, {"key": name}))
        return out

    def gen_blocks(self) -> None:
        reg, hr = self.reg, self.hr
        places: dict[str, set[str]] = {}
        for k in reg.keys:
            places.setdefault(k.name, set()).update(self.place_blocks(k))
        done = set()
        for k in reg.keys:
            if k.name in done or not k.examples:
                continue
            found = hr.key_line(reg, k.examples[0], k.name)
            if found is None:
                self.problems.append(f"blocks: {k.id}: its first example has "
                                     f"no {k.name} line")
                continue
            done.add(k.name)
            probe = found[1].text
            for place, wname in hr.block_wrappers(reg).items():
                comp = hr.compose(reg, wname, probe)
                listed = place in places[k.name]
                pats = self.placement_patterns(place, k.name,
                                               places[k.name])
                line = comp.slot

                def judge(res, line=line, listed=listed, pats=pats,
                          place=place, name=k.name):
                    err = res["parse_error"]
                    what = self.frame_what(err, line) if err else None
                    refused = what is not None and any(
                        p.match(what) for p in pats)
                    if listed and refused:
                        return (f"{name} is refused in {place}, which the "
                                f"registry lists: {err}")
                    if not listed and not refused:
                        got = err or "the file is read"
                        return (f"{name} in {place}, which the registry "
                                f"does not list, is not refused as out of "
                                f"place: {got}")
                    return None
                self.add("blocks", f"{k.name} in {place}", comp.file,
                         comp.context, "parse", judge)

    # ---- repeat -------------------------------------------------------
    def gen_repeat(self) -> None:
        reg, hr = self.reg, self.hr
        for k in reg.keys:
            if not k.examples or k.repeat == "add":
                continue
            found = hr.key_line(reg, k.examples[0], k.name)
            if found is None:
                continue
            comp, ln, block = found
            file = self.insert_after(comp.file, ln, ln.text)
            new_line = ln.last + 1
            mode = "parse"
            if k.repeat == "error":
                fill = {"first_line": str(ln.last), "block": block,
                        "key": k.name}
                rx = hr.template_regex(k.repeat_msg, fill)

                def judge(res, rx=rx, line=new_line, kid=k.id):
                    err = res["parse_error"]
                    what = self.frame_what(err, line) if err else None
                    if what is None or not rx.match(what):
                        return (f"{kid} given twice is not refused with its "
                                f"repeat_msg at line {line}: "
                                f"{err or 'the file is read'}")
                    return None
            else:
                def judge(res, kid=k.id, rule=k.repeat):
                    if res["parse_error"]:
                        return (f"{kid} given twice ({rule}) is refused: "
                                f"{res['parse_error']}")
                    return None
            self.add("repeat", f"{k.id} twice", file, comp.context, mode,
                     judge)
        for o in reg.line_options:
            if not o.examples or o.repeat == "add" or o.refused:
                continue
            got = self.option_in_example(o)
            if got is None:
                continue
            comp, ln, toks, j, n = got
            group = toks[j:j + 1 + n]
            new = toks[:j + 1 + n] + group + toks[j + 1 + n:]
            file = self.replace_line(comp.file, ln, " ".join(new))
            line = ln.first
            if o.repeat == "error":
                rx = hr.template_regex(o.repeat_msg)

                def judge(res, rx=rx, line=line, oid=o.id):
                    err = res["parse_error"]
                    what = self.frame_what(err, line) if err else None
                    if what is None or not rx.match(what):
                        return (f"{oid} given twice on its line is not "
                                f"refused with its repeat_msg: "
                                f"{err or 'the file is read'}")
                    return None
            else:
                def judge(res, oid=o.id, rule=o.repeat):
                    if res["parse_error"]:
                        return (f"{oid} given twice on its line ({rule}) is "
                                f"refused: {res['parse_error']}")
                    return None
            self.add("repeat", f"{o.id} twice", file, comp.context, "parse",
                     judge)

    def option_in_example(self, o):
        """The first example's line holding the option o: the composed
        file, the logical line, its tokens, the index of the option and the
        number of its values."""
        hr, reg = self.hr, self.reg
        c = o.examples[0]
        comp = hr.compose(reg, c.wrap, c.text, c.context)
        for ln in hr.logical_lines(comp.file):
            if ln.first < comp.slot:
                continue
            toks = ln.text.split()
            # Option words are read in any case (an example may show
            # `asinh`); a flag (--name) is one case only.
            up = [t if t.startswith("--") else t.upper() for t in toks]
            if o.option not in up[1:]:
                continue
            j = up.index(o.option, 1)
            if o.option.startswith("--"):
                n = 0
                while j + 1 + n < len(toks) and \
                        not toks[j + 1 + n].startswith("--"):
                    n += 1
            else:
                n = len(o.args)
            return comp, ln, toks, j, n
        return None

    # ---- ranges -------------------------------------------------------
    @staticmethod
    def limit(v):
        hr = sys.modules["hpotarget_registry"]
        return hr.INT_LIMITS.get(v, v) if isinstance(v, str) else v

    @staticmethod
    def fmt(v) -> str:
        if isinstance(v, float) and v.is_integer() and abs(v) < 1e15:
            return str(int(v))
        return repr(v) if isinstance(v, float) else str(v)

    def range_values(self, a):
        """(value, accepted) pairs at and past the limits of `a`."""
        lo, hi = self.limit(a.min), self.limit(a.max)
        out = []
        if a.kind == "positive":
            out += [("0", False), ("1", True)]
            return out
        whole = a.kind == "integer" or a.integral
        step = 1 if whole else 1
        if lo is not None:
            if a.min_open:
                out.append((self.fmt(lo), False))
            else:
                out.append((self.fmt(lo), True))
                out.append((self.fmt(lo - step), False))
        if hi is not None:
            if (self.fmt(hi), True) not in out:     # a range of one value
                out.append((self.fmt(hi), True))
            out.append((self.fmt(hi + step), False))
        return out

    def range_message(self, a, what_key: str, value: str, entry=None):
        """The messages a value past the range of `a` may get: the
        argument's own, another message of its entry, or the [reading]
        message of its kind, filled."""
        hr, rd = self.hr, self.reg.reading.fields
        own = [hr.template_regex(t, {"tok": value})
               for _, t, _ in hr.entry_templates(entry)] if entry else []
        if a.msg:
            return [hr.template_regex(a.msg, {"tok": value,
                                              "key": what_key})] + own
        what = (a.what or "{key}").replace("{key}", what_key)
        fill = {"what": what, "tok": value}
        if a.min is not None:
            fill["min"] = self.fmt(self.limit(a.min))
        if a.max is not None:
            fill["max"] = self.fmt(self.limit(a.max))
        if a.kind == "positive":
            if a.noun:
                fill["noun"] = a.noun
            return [hr.template_regex(rd["positive_msg"], fill)] + own
        if a.kind == "integer":
            # A maximum that is the reader's own (the largest value of the
            # type) is not named: integer_msg; another one is.
            return [hr.template_regex(rd["integer_range_msg"], fill),
                    hr.template_regex(rd["integer_msg"], fill)] + own
        return [hr.template_regex(t, fill) for k, t in rd.items()
                if isinstance(t, str) and k.endswith("_msg")] + own

    @staticmethod
    def ranged(a, required: bool = True) -> bool:
        """An argument the ranges check takes: a number with a limit, or a
        positive number; with `required`, one the line must give once."""
        if required and (a.optional or a.repeat):
            return False
        return a.kind in ("integer", "number", "positive") and \
            (a.min is not None or a.max is not None or a.kind == "positive")

    def distinct(self, cases):
        """The examples of an entry whose files differ."""
        seen, out = set(), []
        for c in cases:
            key = (c.wrap, c.text, c.context)
            if key not in seen:
                seen.add(key)
                out.append(c)
        return out

    @staticmethod
    def label(eid: str, value: str, i: int) -> str:
        return f"{eid} {value}" + (f" (example {i})" if i else "")

    def gen_ranges(self) -> None:
        hr, reg = self.hr, self.reg
        for k in reg.keys:
            if len(k.args) != 1 or not self.ranged(k.args[0]):
                continue
            a = k.args[0]
            for i, c in enumerate(self.distinct(k.examples)):
                found = hr.key_line(reg, c, k.name)
                if found is None:
                    continue
                comp, ln, _ = found
                for value, ok in self.range_values(a):
                    file = self.replace_line(comp.file, ln, f"{k.name} {value}")
                    self.add_range(self.label(k.id, value, i), file,
                                   comp.context, ln.first, ok,
                                   self.range_message(a, k.name, value, k),
                                   k.id, self.mode_of(c, a, value, ok))
        # INIT methods: METHOD <name> <value>.
        for m in reg.init_methods:
            if len(m.args) != 1 or not self.ranged(m.args[0]):
                continue
            a = m.args[0]
            for i, c in enumerate(self.distinct(m.examples)):
                got = self.line_of(c, lambda t, m=m: len(t) == 3 and
                                   t[1] == m.name)
                if got is None:
                    continue
                comp, ln, toks = got
                for value, ok in self.range_values(a):
                    file = self.replace_line(
                        comp.file, ln, " ".join(toks[:2] + [value]))
                    self.add_range(self.label(m.id, value, i), file,
                                   comp.context, ln.first, ok,
                                   self.range_message(a, toks[0], value, m),
                                   m.id, self.mode_of(c, a, value, ok))
        # PARAM keys: PARAM <name> <value>, whose values the startup checks
        # read, so every case is judged when tadah hpo starts.
        for prm in reg.params:
            if len(prm.args) != 1 or not self.ranged(prm.args[0]):
                continue
            a = prm.args[0]
            for i, c in enumerate(self.distinct(prm.examples)):
                got = self.line_of(c, lambda t, prm=prm: len(t) == 3 and
                                   t[1].split("[")[0] == prm.name)
                if got is None:
                    continue
                comp, ln, toks = got
                rxs = [hr.template_regex(t) for _, t, _ in
                       hr.entry_templates(prm)] + \
                    [hr.template_regex(t) for k, t in
                     reg.reading.fields.items()
                     if isinstance(t, str) and k.endswith("_msg")]
                for value, ok in self.range_values(a):
                    file = self.replace_line(
                        comp.file, ln, " ".join(toks[:2] + [value]))
                    self.add_param_range(self.label(prm.id, value, i), file,
                                         comp.context, ok, rxs, prm.id,
                                         f"{prm.id} {value}")
        for o in reg.line_options:
            if not o.examples or len(o.args) != 1 or o.refused or \
                    not self.ranged(o.args[0], required=False):
                continue
            a = o.args[0]
            got = self.option_in_example(o)
            if got is None:
                continue
            comp, ln, toks, j, n = got
            if n != 1:
                continue
            for value, ok in self.range_values(a):
                new = toks[:j + 1] + [value] + toks[j + 2:]
                file = self.replace_line(comp.file, ln, " ".join(new))
                self.add_range(f"{o.id} {value}", file, comp.context,
                               ln.first, ok,
                               self.range_message(a, toks[0], value, o),
                               o.id, self.mode_of(o.examples[0], a, value,
                                                  ok))

    def line_of(self, c, match):
        """The first line of the case `c`, from its slot on, whose tokens
        `match` takes: the composed file, the line and its tokens."""
        hr = self.hr
        comp = hr.compose(self.reg, c.wrap, c.text, c.context)
        for ln in hr.logical_lines(comp.file):
            toks = ln.text.split()
            if ln.first >= comp.slot and match(toks):
                return comp, ln, toks
        return None

    def mode_of(self, c, a, value: str, ok: bool) -> str:
        """How a range case is run: the largest value of a range at the
        check of its example (startup unless the example says parse), so
        that a value the parser takes and the run cannot use fails; the
        others are read only, as the value is judged at its line."""
        hi = self.limit(a.max)
        if ok and hi is not None and value == self.fmt(hi):
            return c.check
        return "parse"

    def add_range(self, label, file, context, line, ok, rxs, eid,
                  mode="parse"):
        def judge(res):
            err = res["parse_error"] or res["startup_error"]
            what = self.frame_what(err, line) if err else None
            if ok:
                if err:
                    return f"{label}: a value in the range of {eid} is " \
                           f"refused: {err}"
                return None
            if what is None:
                return (f"{label}: a value past the range of {eid} is not "
                        f"refused at its line: {err or 'the file is read'}")
            if not any(r.match(what) for r in rxs):
                return (f"{label}: refused with a message the registry "
                        f"does not give for the argument: {what}")
            return None
        self.add("ranges", label, file, context, mode, judge)

    def frame_any(self, error: str) -> str | None:
        """The {what} of `error` when it is framed at any line, else
        None."""
        frame = self.reg.lexing.fields["error_frame"]
        pre = frame.split("{line}")[0]
        m = re.match(re.escape(pre) + r"(\d+)", error)
        return self.frame_what(error, int(m.group(1))) if m else None

    def add_param_range(self, label, file, context, ok, rxs, eid, group):
        """A PARAM value at or past a limit: the startup checks read it, so
        a value in the range starts in the file of one example of the
        entry at least (ASA's User_Initial_Parameters 0 only where the
        search has no starting point, 1 only where it has one), and one
        past it is refused in each, at parse or at startup, with a message
        of the entry or of [reading]."""
        def judge(res):
            err = res["parse_error"] or res["startup_error"]
            if ok:
                if err:
                    return f"{label}: a value in the range of {eid} is " \
                           f"refused: {err}"
                return None
            if not err:
                return (f"{label}: a value past the range of {eid} is not "
                        f"refused: tadah hpo starts")
            what = self.frame_any(err)
            if not any(r.match(what if what is not None else err)
                       for r in rxs):
                return (f"{label}: refused with a message the registry "
                        f"does not give for the argument: {err}")
            return None
        self.add("ranges", label, file, context, "startup", judge,
                 group=group if ok else None)

    # ---- words --------------------------------------------------------
    def gen_words(self) -> None:
        hr, reg = self.hr, self.reg
        for k in reg.keys:
            if not k.examples:
                continue
            for pos, a in hr.word_positions(k.args):
                found = hr.key_line(reg, k.examples[0], k.name)
                if found is None:
                    continue
                comp, ln, _ = found
                toks = ln.text.split()
                if len(toks) <= 1 + pos:
                    continue
                self.add_words(k.id, a, comp, ln, toks, 1 + pos)
        for o in reg.line_options:
            if not o.examples or o.refused:
                continue
            for pos, a in hr.word_positions(o.args):
                got = self.option_in_example(o)
                if got is None:
                    continue
                comp, ln, toks, j, n = got
                if pos >= n:
                    continue
                self.add_words(o.id, a, comp, ln, toks, j + 1 + pos)

    def add_words(self, eid, a, comp, ln, toks, at):
        hr = self.hr
        for w in a.words:
            new = toks[:at] + [w] + toks[at + 1:]
            file = self.replace_line(comp.file, ln, " ".join(new))

            def judge(res, w=w, line=ln.first):
                err = res["parse_error"]
                if err and self.frame_what(err, line) is not None:
                    return f"{eid}: the word {w} is refused: {err}"
                return None
            self.add("words", f"{eid} {w}", file, comp.context, "parse",
                     judge)
        for w, msg in a.refused_words.items():
            new = toks[:at] + [w] + toks[at + 1:]
            file = self.replace_line(comp.file, ln, " ".join(new))
            rx = hr.template_regex(msg, {"tok": w, "key": toks[0]})

            def judge(res, w=w, rx=rx, line=ln.first):
                err = res["parse_error"]
                what = self.frame_what(err, line) if err else None
                if what is None or not rx.match(what):
                    return (f"{eid}: the refused word {w} is not refused "
                            f"with its message: {err or 'the file is read'}")
                return None
            self.add("words", f"{eid} {w} refused", file, comp.context,
                     "parse", judge)

    # ---- params -------------------------------------------------------
    def refusing_policy(self, optimizer: str) -> bool:
        lib = optimizer.split(":")[0]
        for p in self.reg.param_policies:
            if optimizer in p.optimizers or lib in p.optimizers:
                return bool(p.msg)
        return False

    def gen_params(self) -> None:
        hr, reg = self.hr, self.reg
        wrappers = [w for w in reg.wrappers.values() if w.optimizer]
        if not wrappers:
            self.problems.append("params: no wrapper names its optimizer: "
                                 "the PARAM keys of each optimiser cannot "
                                 "be checked")
            return
        # A PARAM name can belong to several entries (one per optimiser
        # family): a wrapper's optimiser takes the name when any of them
        # lists it.
        lists_of: dict[str, set[str]] = {}
        for p in reg.params:
            lists_of.setdefault(p.name, set()).update(p.optimizers)
        for p in reg.params:
            if not p.examples or "*" in p.optimizers:
                continue
            text = p.examples[0].text
            covered = set()
            for w in wrappers:
                lib = w.optimizer.split(":")[0]
                if w.optimizer in p.optimizers or lib in p.optimizers:
                    covered.add(w.optimizer)
                every = lists_of[p.name]
                listed = (w.optimizer in every or lib in every
                          or "*" in every)
                if not listed and not self.refusing_policy(w.optimizer):
                    continue
                if listed and w.optimizer not in covered:
                    continue
                comp = hr.compose(reg, w.name, text)

                def judge(res, listed=listed, w=w, name=p.name, pid=p.id):
                    err = res["parse_error"] or res["startup_error"]
                    if listed and err:
                        return (f"{pid} is refused on {w.optimizer}, which "
                                f"it lists: {err}")
                    if not listed and name not in err:
                        return (f"{pid} on {w.optimizer}, which it does not "
                                f"list, is not refused: "
                                f"{err or 'tadah hpo starts'}")
                    return None
                self.add("params", f"{p.id} on {w.optimizer}", comp.file,
                         comp.context, "startup", judge)
            for name in p.optimizers:
                if not any(o == name or o.split(":")[0] == name
                           for o in covered):
                    self.problems.append(f"params: {p.id}: no wrapper runs "
                                         f"{name}")

    # ---- lines --------------------------------------------------------
    def gen_lines(self) -> None:
        hr, reg = self.hr, self.reg
        root_lines = {}
        for k in reg.keys:
            if "ROOT" in k.blocks and k.examples:
                root_lines[k.name] = k
        option_lines = sorted({ln for o in reg.line_options for ln in o.lines
                               if ln in root_lines})
        lines_of: dict[str, set[str]] = {}
        own_of: dict[str, list] = {}
        for o in reg.line_options:
            lines_of.setdefault(o.option, set()).update(o.lines)
            own_of.setdefault(o.option, []).extend(
                hr.template_regex(t) for _, t, _ in hr.entry_templates(o))
        for o in reg.line_options:
            if o.refused or not o.examples or \
                    not set(o.lines) & set(option_lines):
                continue
            got = self.option_in_example(o)
            if got is None:
                continue
            _, _, toks, j, n = got
            group = toks[j:j + 1 + n]
            for line_key in option_lines:
                if line_key == toks[0]:
                    continue
                base = root_lines[line_key]
                found = hr.key_line(reg, base.examples[0], line_key)
                if found is None:
                    continue
                comp, ln, _ = found
                file = self.replace_line(comp.file, ln,
                                         ln.text + " " + " ".join(group))
                listed = line_key in lines_of[o.option]
                own = own_of[o.option]

                # The refusal names the option as the line wrote it.
                def judge(res, listed=listed, line=ln.first,
                          lk=line_key, oid=o.id, own=own, tok=group[0]):
                    err = res["parse_error"]
                    what = self.frame_what(err, line) if err else None
                    if listed and what is not None and \
                            not any(r.match(what) for r in own):
                        return (f"{oid} on {lk}, which its lines hold, is "
                                f"refused: {err}")
                    if not listed and (what is None or tok not in what):
                        return (f"{oid} on {lk}, which its lines do not "
                                f"hold, is not refused naming it: "
                                f"{err or 'the file is read'}")
                    return None
                self.add("lines", f"{o.id} on {line_key}", file,
                         comp.context, "parse", judge)

    # ---- defaults -----------------------------------------------------
    def gen_defaults(self) -> None:
        hr, reg = self.hr, self.reg
        files = [hr.compose(reg, "root", "OPTIM CGRID2B (2) AUTO\n"
                                         "OPTIM SGRID2B (1) AUTO")]
        if "main_ream" in reg.wrappers:
            files.append(hr.compose(reg, "main_ream",
                                    "RHOGATE 0\nRHOBARGATE bcc 3.3 0.9 1.1"))
        if any(f.path == "lines_pc.in" for f in reg.fixtures.values()):
            files.append(hr.compose(
                reg, "root",
                "PC_LAMMPS --script lines_pc.in --varloss E 1 1 "
                "--failvar V\n"
                "PC_FAILPROBE --script lines_pc.in --outvar S"))
            files[-1] = hr.Composed(files[-1].file.replace(
                "PC_ERMSE 0 1\n", ""), files[-1].slot, files[-1].context)
        self.default_results: list[dict] = []
        for i, comp in enumerate(files):
            def judge(res, i=i):
                if res["parse_error"]:
                    return f"defaults file {i} is refused: {res['parse_error']}"
                self.default_results.append(res["values"])
                return None
            self.add("defaults", f"file {i}", comp.file, comp.context,
                     "defaults", judge)

    def judge_defaults(self) -> list[str]:
        reg = self.reg
        values: dict[str, list[str]] = {}
        for v in self.default_results:
            for k, vs in v.items():
                values.setdefault(k, [])
                values[k] += [x for x in vs if x not in values[k]]
        out = []
        literal: dict[str, str] = {}
        for e in reg.entries():
            d = getattr(e, "default", None)
            if d is None:
                continue
            if d.literal is not None:
                literal[e.id] = d.literal
            elif d.text and NUMBER.search(d.text):
                # Each number of the text is a value the runner reports,
                # unless `unchecked` says why no setting holds it; each
                # number the runner reports is in the text.
                said = [m.group(0) for m in NUMBER.finditer(d.text)]
                got = values.get(e.id, [])
                if not got and not d.unchecked:
                    out.append(f"defaults: {e.id}: the text default holds "
                               f"numbers ({d.text!r}) that no value of the "
                               "runner checks: report it in "
                               "hpotarget_case_runner, or say why no "
                               "setting holds it (unchecked)")
                    continue
                for n in said:
                    if got and not d.unchecked and \
                            not any(self.same(n, v) for v in got):
                        out.append(f"defaults: {e.id}: the text default "
                                   f"says {n}, the code gives "
                                   f"{', '.join(got)}")
                for v in got:
                    if NUMBER.fullmatch(v) and \
                            not any(self.same(n, v) for n in said):
                        out.append(f"defaults: {e.id}: the code gives {v}, "
                                   f"which the text default does not say "
                                   f"({d.text!r})")
        for lf in reg.loss_functions:
            for p in lf.params:
                literal[f"{lf.id}.{p.name}"] = str(p.default)
        for eid, lit in sorted(literal.items()):
            if eid not in values:
                out.append(f"defaults: {eid}: the literal default {lit!r} "
                           "is reported by no value of the runner "
                           "(hpotarget_case_runner, defaults_of)")
                continue
            if not any(self.same(lit, v) for v in values[eid]):
                out.append(f"defaults: {eid}: the registry gives {lit!r}, "
                           f"the code {', '.join(values[eid])}")
        text_ids = {e.id for e in reg.entries()
                    if getattr(e, "default", None) is not None
                    and e.default.text}
        for eid in sorted(values):
            if eid not in literal and eid not in text_ids:
                out.append(f"defaults: the runner reports {eid} = "
                           f"{', '.join(values[eid])}, which has no "
                           "default in the registry")
        return out

    @staticmethod
    def same(a: str, b: str) -> bool:
        try:
            x, y = float(a), float(b)
            return x == y or math.isclose(x, y, rel_tol=1e-12)
        except ValueError:
            return a.split() == b.split()


def judge_all(g: Generator, by_id: dict[str, dict]
              ) -> tuple[list[str], dict[str, int]]:
    """The problems of the generator and of each case's result, and the
    number of cases of each check. A group passes when one of its cases
    does, and fails with the problem of its first case otherwise."""
    problems = list(g.problems)
    counts: dict[str, int] = {}
    groups: dict[str, list[str | None]] = {}
    for c in g.cases:
        counts[c.check] = counts.get(c.check, 0) + 1
        res = by_id.get(c.case["id"])
        why = f"{c.label}: no result" if res is None else c.judge(res)
        why = f"{c.check}: {why}" if why else None
        if c.group is not None:
            groups.setdefault(c.group, []).append(why)
        elif why:
            problems.append(why)
    for whys in groups.values():
        if all(whys):
            problems.append(whys[0])
    return problems, counts


def run(runner: pathlib.Path, registry: pathlib.Path, only: str | None,
        keep: pathlib.Path | None) -> int:
    sys.path.insert(0, str(registry))
    import hpotarget_registry as hr  # noqa: E402
    try:
        reg = hr.load(registry)
    except hr.RegistryError as e:
        for line in e.errors:
            print(f"registry error: {line}")
        print(f"check_hpotarget_generated.py: the registry does not load "
              f"({len(e.errors)} error(s))")
        return 1
    g = Generator(hr, reg)
    for name in CHECK_NAMES:
        if only in (None, name):
            getattr(g, f"gen_{name}")()
    with tempfile.TemporaryDirectory() as tmp:
        d = keep or pathlib.Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        cases_file, results_file = d / "cases.toml", d / "results.json"
        write_cases(cases_file, [c.case for c in g.cases])
        proc = subprocess.run([str(runner), str(cases_file),
                               str(results_file)], capture_output=True,
                              text=True)
        if proc.returncode != 0:
            print(proc.stdout + proc.stderr)
            print(f"check_hpotarget_generated.py: the runner failed "
                  f"({proc.returncode})")
            return 1
        results = json.loads(results_file.read_text())
    problems, counts = judge_all(g, {r["id"]: r for r in results})
    if only in (None, "defaults"):
        problems += g.judge_defaults()
    for p in problems:
        print(p)
    summary = ", ".join(f"{counts.get(n, 0)} {n}" for n in CHECK_NAMES
                        if only in (None, n))
    if problems:
        print(f"check_hpotarget_generated.py: {len(problems)} problem(s) in "
              f"{len(g.cases)} cases ({summary})")
        return 1
    print(f"check_hpotarget_generated.py: OK: {len(g.cases)} cases "
          f"({summary})")
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--runner", type=pathlib.Path, required=True)
    ap.add_argument("--registry", type=pathlib.Path,
                    default=DEFAULT_REGISTRY)
    ap.add_argument("--only", choices=CHECK_NAMES)
    ap.add_argument("--keep", type=pathlib.Path)
    a = ap.parse_args(argv)
    return run(a.runner, a.registry, a.only, a.keep)


if __name__ == "__main__":
    sys.exit(main())
