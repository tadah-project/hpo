#!/usr/bin/env python3
"""The target-file registry: load and check hpotarget.toml and its parts.

The registry describes the target file of `tadah hpo` (--hpotarget): how a
line is read, the blocks and where each may open, every key with its
values, default, range and refusals, the options of the PC_* lines, the
OPTIM grammar and the PARAM keys of the optimisers. hpotarget.toml holds
the frame and lists the part files in meta.parts; blueprint/README.md
("The target-file registry") describes every table and field.

load() reads the files with tomllib (Python 3.11 or later, standard
library only), checks them strictly and returns a Registry of dataclasses:

  - every table and field is one the schema below knows, with the type it
    declares, and every required field is there;
  - every table of an entry ([[key.invalid]], ...) follows its entry, before
    a table of another kind (TOML gives it to the last entry above it,
    wherever it sits);
  - every id is unique, across all the files;
  - every reference resolves: a block's parents, a key's group or blocks,
    a block alias, an INNER refusal, a strategy type's keys, the lines of
    an option, a wrapper or a training configuration an example names,
    the optimisers of a PARAM key;
  - every {name} placeholder of a message is one that message may hold,
    and no example, error or wrapper text holds one by mistake;
  - every wrapper is a file with exactly one slot line, {line};
  - no documentation text addresses the reader or holds a placeholder,
    every example after the first of an entry has a doc (the caption the
    reference pages show), and a rule has a title;
  - the optimiser descriptions (meta.optimizer_docs) describe every library
    and every registered optimiser of the optimiser list, and nothing else.

With parts= naming a subset of meta.parts (the command's --part), the
files not loaded can hold what a reference names: such references are
listed as notes instead of errors, and the full check resolves them.

Command line:
  hpotarget_registry.py --check [--part FILE] [--dir DIR]
exits 0 and prints a summary when the files pass, else 1 and every error.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys
import tomllib
from dataclasses import dataclass, field
from typing import Any, Iterator

API = 1
MAIN_FILE = "hpotarget.toml"
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent

# {name} is a placeholder; ${name}, a LAMMPS variable reference, is text.
PLACEHOLDER = re.compile(r"(?<!\$)\{([A-Za-z_][A-Za-z0-9_]*)\}")
ID_RE = re.compile(r"^[A-Za-z0-9_.:\-\[\]()]+$")
WORD_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")
# Words that address the reader (DOCS-style impersonal voice).
READER_WORDS = re.compile(r"\b(you|your|yours|yourself|we|our|us|let's)\b",
                          re.IGNORECASE)

# The placeholders a message written for a key, an option or a rule may
# hold: the names the parser's own messages use for what they insert.
PLACEHOLDERS = {
    "algo", "block", "count", "default", "effect", "enclosing", "end",
    "env", "file", "first_line", "flag", "form", "high", "hint", "home",
    "index", "indices", "key", "keys", "kind", "lib", "line", "list", "low",
    "max", "min", "missing", "name", "note", "noun", "open", "open_line",
    "option", "parent", "parent_line", "phase", "script", "seed", "text",
    "tok", "type", "underflow", "value", "var", "what", "why", "width",
}
# Fields whose placeholders are fixed by the parser's frame.
FIXED_PLACEHOLDERS = {
    ("lexing", "error_frame"): {"line", "what", "text"},
    ("lexing", "unfinished_msg"): set(),
    ("lexing", "missing_file_msg"): {"file"},
    ("structure", "end_without_open_msg"): {"end", "name"},
    ("structure", "end_mismatch_msg"): {"end", "open", "open_line"},
    ("structure", "end_takes_no_value_msg"): {"end", "name", "tok"},
    ("structure", "open_takes_no_value_msg"): {"name", "tok"},
    ("structure", "unclosed_msg"): {"name", "end"},
    ("structure", "unknown_root_msg"): {"key"},
    ("structure", "root_key_in_block_msg"): {"key", "name", "open_line", "end"},
    ("structure", "removed_msg"): {"key", "hint"},
    ("group", "leftover_msg"): {"key", "tok"},
    ("block", "repeat_msg"): {"first_line"},
    ("block", "misplaced_msg"): {"parent", "parent_line"},
    ("block", "unknown_msg"): {"key", "keys", "open_line"},
    ("block", "refuse_msg"): {"key", "why", "enclosing"},
}
READING_PLACEHOLDERS = {"what", "min", "max", "tok", "note", "noun",
                        "underflow", "key"}

CHECK_STAGES = ["block_lines", "removed", "inner_refusal", "group_keys",
                "init_restart_keys", "noise_keys", "log_hp", "strategy_keys",
                "block_unknown", "root_keys"]
BLOCK_REPEAT = {"error", "error_per_parent", "merge"}
KEY_REPEAT = {"last_wins", "warn_last_wins", "error", "add"}
ARG_KINDS = {"integer", "number", "positive", "word", "name", "bool",
             "timespec", "indices", "tokens", "loss"}
INT_LIMITS = {"INT_MAX": 2**31 - 1, "UINT_MAX": 2**32 - 1,
              "UINT64_MAX": 2**64 - 1}
CHECKS = {"parse", "startup"}
# An invalid case may also run as `tadah hpo --check` does (hpo_run with
# CHECK true), for a refusal the run makes before it reads its data, such
# as that of an environment override.
INVALID_CHECKS = CHECKS | {"check"}
REGISTRY_VERSION = 1
DOC_SENTENCES = 2


class RegistryError(Exception):
    """The registry files do not pass the checks; `errors` lists why."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("\n".join(errors))


# ---------------------------------------------------------------- data ----

@dataclass
class Case:
    """One `examples` or `invalid` entry: a target-file text in a wrapper."""
    owner: str            # id of the entry it belongs to
    part: str             # file it is written in
    kind: str             # "example" or "invalid"
    index: int            # position in its list, from 0
    wrap: str             # wrapper name, or "none"
    text: str
    check: str            # "parse" or "startup"
    context: str | None = None
    env: dict[str, str] = field(default_factory=dict)
    doc: str = ""
    error: str | None = None
    at: int | str = 1
    msg_of: str = ""      # id of the entry whose message the error is


@dataclass
class Check:
    """A refusal a block or an entry makes, with when it applies."""
    msg: str
    doc: str


@dataclass
class Untested:
    """A message of an entry that no invalid case shows, and why."""
    msg: str
    why: str


@dataclass
class Arg:
    name: str
    kind: str
    optional: bool = False
    repeat: bool = False
    min: int | float | str | None = None
    max: int | float | str | None = None
    min_open: bool = False      # min itself is refused: a value above min
    integral: bool = False      # a number with no fractional part
    words: list[str] = field(default_factory=list)
    refused_words: dict[str, str] = field(default_factory=dict)
    what: str = ""
    msg: str = ""
    noun: str = ""
    default: Any = None
    doc: str = ""


@dataclass
class Default:
    literal: str | None = None
    text: str | None = None
    unchecked: str | None = None   # why no setting holds a text's number


@dataclass
class Refusal:
    key: str
    why: str


@dataclass
class LossParam:
    name: str
    default: float
    gt: float | None = None
    doc: str = ""


@dataclass
class Section:
    """[lexing], [reading], [structure] or [optim]: named fields."""
    id: str
    part: str
    fields: dict[str, Any]
    doc: str = ""
    long_doc: str = ""
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Meta:
    version: int
    api: int
    parts: list[str]
    anneal_keys: str
    optimizers: str
    optimizer_docs: str


@dataclass
class Group:
    id: str
    name: str
    part: str
    blocks: list[str]
    leftover_msg: str
    doc: str
    long_doc: str = ""
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Block:
    id: str
    name: str
    part: str
    end: str
    parents: list[str]
    repeat: str
    misplaced_msg: str
    unknown_msg: str
    doc: str
    repeat_msg: str = ""
    unknown_first: bool = False
    aliases: dict[str, str] = field(default_factory=dict)
    refuse: list[Refusal] = field(default_factory=list)
    refuse_msg: str = ""
    unreachable_msgs: list[str] = field(default_factory=list)
    requires: list[Check] = field(default_factory=list)
    close_checks: list[Check] = field(default_factory=list)
    quirk: str = ""
    long_doc: str = ""
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Key:
    id: str
    name: str
    part: str
    synopsis: str
    repeat: str
    doc: str
    group: str = ""
    blocks: list[str] = field(default_factory=list)
    args: list[Arg] = field(default_factory=list)
    repeat_msg: str = ""
    repeat_doc: str = ""
    misplaced_msg: str = ""
    default: Default | None = None
    long_doc: str = ""
    alias_of: str = ""
    refused: str = ""
    quirk: str = ""
    env: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)

    @property
    def scope(self) -> str:
        return self.id[: -len(self.name) - 1]


@dataclass
class LineOption:
    id: str
    option: str
    part: str
    lines: list[str]
    synopsis: str
    doc: str
    args: list[Arg] = field(default_factory=list)
    repeat: str = "last_wins"
    repeat_msg: str = ""
    default: Default | None = None
    long_doc: str = ""
    refused: str = ""
    quirk: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Form:
    id: str
    line: str
    part: str
    synopsis: str
    doc: str
    word: str = ""
    args: list[Arg] = field(default_factory=list)
    long_doc: str = ""
    refused: str = ""
    quirk: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Removed:
    id: str
    name: str
    part: str
    hint: str
    doc: str = ""
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Rule:
    id: str
    part: str
    title: str
    doc: str
    check: str
    msg: str = ""
    msg_doc: str = ""     # what the message holds, said in words
    keys: list[str] = field(default_factory=list)
    quirk: str = ""
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Env:
    id: str
    name: str
    part: str
    key: str
    doc: str
    values: list[str] = field(default_factory=list)
    msg: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class LossFunction:
    id: str
    name: str
    part: str
    doc: str
    params: list[LossParam] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class StrategyType:
    id: str
    name: str
    part: str
    doc: str
    keys: list[str] = field(default_factory=list)
    quirk: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class InitMethod:
    id: str
    name: str
    part: str
    doc: str
    args: list[Arg] = field(default_factory=list)
    refused: str = ""
    quirk: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class OptimTarget:
    id: str
    key: str
    part: str
    status: str
    doc: str
    indices: list[int] = field(default_factory=list)
    msg: str = ""
    condition: str = ""
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Param:
    id: str
    name: str
    part: str
    optimizers: list[str]
    doc: str
    args: list[Arg] = field(default_factory=list)
    default: Default | None = None
    indexed: bool = False
    long_doc: str = ""
    quirk: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class ParamExcluded:
    id: str
    name: str
    part: str
    optimizers: list[str]
    why: str
    doc: str = ""
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class ParamPolicy:
    id: str
    part: str
    optimizers: list[str]
    doc: str
    msg: str = ""
    quirk: str = ""
    checks: list[Check] = field(default_factory=list)
    untested: list[Untested] = field(default_factory=list)
    examples: list[Case] = field(default_factory=list)
    invalid: list[Case] = field(default_factory=list)


@dataclass
class Wrapper:
    name: str
    part: str
    text: str
    doc: str
    block: str = ""             # the block its slot is in, or ROOT
    optimizer: str = ""         # LIB:ALGO of the STRATEGY-level optimiser
    context: str = "default"


@dataclass
class TrainingContext:
    name: str
    part: str
    config: str
    doc: str
    has_train_db: bool = True
    has_validation: bool = True


@dataclass
class Fixture:
    name: str
    part: str
    path: str
    text: str
    doc: str = ""


@dataclass
class OptimizerDocs:
    """The descriptions of meta.optimizer_docs: a sentence or two per
    library (LIB) and per registered optimiser (LIB:ALGO)."""
    libraries: dict[str, str]
    optimizers: dict[str, str]


@dataclass
class Registry:
    directory: pathlib.Path
    files: list[str]
    full: bool
    meta: Meta
    lexing: Section
    reading: Section
    structure: Section
    optim: Section | None
    groups: dict[str, Group]
    blocks: list[Block]
    keys: list[Key]
    line_options: list[LineOption]
    forms: list[Form]
    removed: list[Removed]
    rules: list[Rule]
    envs: list[Env]
    loss_functions: list[LossFunction]
    strategy_types: list[StrategyType]
    init_methods: list[InitMethod]
    optim_targets: list[OptimTarget]
    params: list[Param]
    param_excluded: list[ParamExcluded]
    param_policies: list[ParamPolicy]
    wrappers: dict[str, Wrapper]
    contexts: dict[str, TrainingContext]
    fixtures: dict[str, Fixture]
    lammps_script: Section | None = None
    abbreviations: dict[str, str] = field(default_factory=dict)
    optimizer_docs: OptimizerDocs | None = None
    notes: list[str] = field(default_factory=list)

    def entries(self) -> Iterator[Any]:
        """Every entry that has an id, in file order per kind."""
        for s in (self.lexing, self.reading, self.structure, self.optim,
                  self.lammps_script):
            if s is not None:
                yield s
        yield from self.groups.values()
        for kind in ("blocks", "keys", "line_options", "forms", "removed",
                     "rules", "envs", "loss_functions", "strategy_types",
                     "init_methods", "optim_targets", "params",
                     "param_excluded", "param_policies"):
            yield from getattr(self, kind)

    def by_id(self) -> dict[str, Any]:
        return {e.id: e for e in self.entries()}

    def cases(self, part: str | None = None) -> list[Case]:
        """The examples and invalid entries, of one file or of all."""
        out: list[Case] = []
        for e in self.entries():
            for c in list(e.examples) + list(e.invalid):
                if part is None or c.part == part:
                    out.append(c)
        return out

    def block(self, name: str) -> Block:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    def keys_in_block(self, name: str) -> list[Key]:
        """The keys valid in the block `name` ("ROOT" for the top level)."""
        out = []
        for k in self.keys:
            where = (self.groups[k.group].blocks if k.group in self.groups
                     else k.blocks)
            if name in where:
                out.append(k)
        return out

    def vocabulary(self, name: str) -> set[str]:
        """The names of one kind the registry declares (the coverage check
        compares them with the literals of the parser's sources)."""
        if name == "block":
            return {b.name for b in self.blocks}
        if name == "block_end":
            return {b.end for b in self.blocks}
        if name == "alias":
            return {w for b in self.blocks for w in b.aliases}
        if name.startswith("key:"):
            scope = name[4:]
            return {k.name for k in self.keys
                    if k.scope == scope and not k.group}
        if name == "group_key":
            return {k.name for k in self.keys if k.group}
        if name == "inner_refused":
            return {r.key for b in self.blocks for r in b.refuse}
        if name == "removed":
            return {r.name for r in self.removed}
        if name == "line_option":
            return {o.option for o in self.line_options}
        if name.startswith("line_option:"):
            line = name.split(":", 1)[1]
            return {o.option for o in self.line_options if line in o.lines}
        if name == "form":
            return {f.word for f in self.forms if f.word}
        if name == "loss":
            return {f.name for f in self.loss_functions}
        if name == "init_method":
            return {m.name for m in self.init_methods}
        if name == "strategy_type":
            return {t.name for t in self.strategy_types}
        if name == "strategy_type_key":
            ids = {i for t in self.strategy_types for i in t.keys}
            return {k.name for k in self.keys if k.id in ids}
        if name.startswith("strategy_type_key:"):
            t = name.split(":", 1)[1]
            ids = {i for st in self.strategy_types if st.name == t
                   for i in st.keys}
            return {k.name for k in self.keys if k.id in ids}
        if name == "optim_target":
            return {t.key for t in self.optim_targets}
        if name == "param":
            return {p.name for p in self.params}
        if name == "param_excluded":
            return {p.name for p in self.param_excluded}
        if name == "env":
            return {e.name for e in self.envs}
        if name == "registry_name":
            # Every name the registry declares, of any kind.
            out: set[str] = set()
            for v in ("block", "block_end", "alias", "group_key",
                      "inner_refused", "removed", "line_option", "form",
                      "loss", "init_method", "strategy_type", "optim_target",
                      "param", "param_excluded", "env", "word"):
                out |= self.vocabulary(v)
            out |= {k.name for k in self.keys}
            return out
        if name == "word":
            words: set[str] = set()
            for e in self.entries():
                for a in getattr(e, "args", []):
                    words.update(a.words)
                    words.update(a.refused_words)
            for e in self.envs:
                words.update(e.values)
            lx = self.lexing.fields
            for f in ("bool_true", "bool_false", "timespec_words"):
                words.update(lx.get(f, []))
            return words
        raise KeyError(f"unknown vocabulary {name!r}")


# ------------------------------------------------------ file layout ----
# A table header at the start of a line: [name] or [[name]].
TABLE_HEADER = re.compile(r"^\s*(\[\[?)\s*([A-Za-z0-9_.\"-]+)\s*\]\]?\s*(#.*)?$")


def table_layout(text: str) -> list[str]:
    """The tables of an entry that do not follow it: TOML gives a table
    [[X.Y]] (or [X.Y...]) to the last [[X]] above it, wherever it sits, so
    one placed after a table of another kind belongs to an entry a reader
    of the file does not see it in, and a [[X]] added between would take it
    over. Each such table is a problem naming its line, the entry and the
    table it follows. Lines inside a multi-line string are not headers."""
    problems: list[str] = []
    arrays: dict[str, int] = {}   # [[X]] -> the line of the last one
    last: tuple[str, int] | None = None   # the last table of another kind
    in_string: str | None = None
    for no, line in enumerate(text.splitlines(), 1):
        if in_string is None:
            m = TABLE_HEADER.match(line)
            if m:
                name = m.group(2)
                first = name.split(".")[0]
                header = f"{m.group(1)}{name}{']]' if m.group(1) == '[[' else ']'}"
                if m.group(1) == "[[" and "." not in name:
                    arrays[first] = no
                    last = (header, no)
                elif first in arrays and "." in name:
                    if last is not None and last[0] != f"[[{first}]]" and \
                            last[1] > arrays[first]:
                        problems.append(
                            f"line {no}: {header} belongs to the [[{first}]] "
                            f"entry of line {arrays[first]}, but follows "
                            f"{last[0]} (line {last[1]}); the tables of an "
                            f"entry follow it, before the next entry")
                else:
                    last = (header, no)
                continue
        # Track the multi-line strings that open and close on this line.
        i = 0
        while True:
            if in_string is None:
                found = [(line.find(q, i), q) for q in ('"""', "'''")]
                found = [(k, q) for k, q in found if k >= 0]
                if not found:
                    break
                k, q = min(found)
                in_string, i = q, k + 3
            else:
                k = line.find(in_string, i)
                if k < 0:
                    break
                in_string, i = None, k + 3
    return problems


# ------------------------------------------------- target-file helpers ----
# The target file as the parser reads it, for the checks below and for
# tests/check_hpotarget_generated.py: a case put into its wrapper, its
# logical lines, the block each line is in, and the messages of an entry.

@dataclass
class Logical:
    """A logical line: its first and last physical lines (from 1), its text
    (the comment cut, trailing blanks removed, continued lines joined) and
    its key, the first word."""
    first: int
    last: int
    text: str
    key: str


def logical_lines(file: str) -> list[Logical]:
    out: list[Logical] = []
    lines = file.split("\n")
    if lines and lines[-1] == "":
        lines.pop()

    def cut(s: str) -> str:
        p = s.find("#")
        return (s[:p] if p >= 0 else s).rstrip()

    i = 0
    while i < len(lines):
        first = i + 1
        text = cut(lines[i])
        while text.endswith("\\") and i + 1 < len(lines):
            i += 1
            text = text[:-1] + cut(lines[i])
        i += 1
        words = text.split()
        if words:
            out.append(Logical(first, i, text, words[0]))
    return out


@dataclass
class Composed:
    file: str
    slot: int        # the physical line the case's text starts on
    context: str


def compose(reg: "Registry", wrap: str, text: str,
            context: str | None = None) -> Composed:
    """The target file of a case: `text` in the slot of the wrapper `wrap`
    (the whole file for wrap = none)."""
    if wrap == "none":
        f = text
        slot = 1
        ctx = context or "default"
    else:
        w = reg.wrappers[wrap]
        lines = w.text.strip("\n").split("\n")
        slot = lines.index("{line}") + 1
        lines[slot - 1] = text.strip("\n")
        f = "\n".join(lines)
        ctx = context or w.context
    if not f.endswith("\n"):
        f += "\n"
    return Composed(f, slot, ctx)


def line_blocks(reg: "Registry", file: str) -> list[tuple[Logical, str]]:
    """Each logical line of `file` with the block it is in (ROOT at the top
    level): a block line opens its block, its END line closes it, and a
    word a block takes as another key (INIT's STRATEGY) opens nothing."""
    names = {b.name for b in reg.blocks}
    ends = {b.end: b.name for b in reg.blocks}
    aliases = {b.name: set(b.aliases) for b in reg.blocks}
    stack: list[str] = []
    out = []
    for ln in logical_lines(file):
        here = stack[-1] if stack else "ROOT"
        if ln.key in ends:
            out.append((ln, here))
            if stack:
                stack.pop()
            continue
        if ln.key in names and ln.key not in aliases.get(here, set()):
            out.append((ln, here))
            stack.append(ln.key)
            continue
        out.append((ln, here))
    return out


def block_wrappers(reg: "Registry") -> dict[str, str]:
    """The wrapper of each place a line can be, ROOT and each block, named
    after it in lower case: the generated checks put a line into each
    (tests/check_hpotarget_generated.py)."""
    return {p: p.lower() for p in ["ROOT"] + [b.name for b in reg.blocks]}


def key_line(reg: "Registry", c: Case, name: str
             ) -> tuple[Composed, Logical, str] | None:
    """The first line of the key `name` in the case `c`, from the line its
    text starts on (a word a block reads as the key, as INIT's STRATEGY for
    METHOD, counts), with the composed file and the block the line is
    in."""
    comp = compose(reg, c.wrap, c.text, c.context)
    alias = {b.name: {w for w, ref in b.aliases.items()
                      if ref.rsplit(".", 1)[-1] == name}
             for b in reg.blocks}
    for ln, block in line_blocks(reg, comp.file):
        if ln.first >= comp.slot and (
                ln.key == name or ln.key in alias.get(block, set())):
            return comp, ln, block
    return None


def word_positions(args: list[Arg]) -> list[tuple[int, Arg]]:
    """(token position after the key or option, argument) for each word
    argument whose place is fixed: every argument before it is one required
    token."""
    out = []
    for i, a in enumerate(args):
        if a.kind == "word":
            out.append((i, a))
        if a.optional or a.repeat or a.kind in ("tokens", "loss", "indices"):
            break
    return out


def word_of(token: str, words: list[str]) -> str | None:
    """The word of `words` that `token` writes: the word itself, or the one
    word it matches in any case. An example is accepted by the parser (the
    conformance test runs it), so a token in another case is a word the
    parser reads in any case; comparing in any case can only miss a word
    left out of `words` when another word differs from it in case alone."""
    if token in words:
        return token
    folded = [w for w in words if w.casefold() == token.casefold()]
    return folded[0] if len(folded) == 1 else None


PLACEHOLDER_ANY = "(?s:.*?)"


def template_regex(template: str, fill: dict[str, str] | None = None
                   ) -> re.Pattern:
    """A message template as a regular expression: each placeholder is
    any text, or the text `fill` gives it."""
    out, pos = "", 0
    for m in PLACEHOLDER.finditer(template):
        out += re.escape(template[pos:m.start()])
        name = m.group(1)
        out += (re.escape(fill[name]) if fill and name in fill
                else PLACEHOLDER_ANY)
        pos = m.end()
    out += re.escape(template[pos:])
    return re.compile("^" + out + "$")


def entry_templates(e: Any) -> list[tuple[str, str, dict[str, str]]]:
    """The message templates an entry holds: (field, template, fill), fill
    the placeholders the entry itself gives (a block's refusals)."""
    out: list[tuple[str, str, dict[str, str]]] = []
    if isinstance(e, Section):
        for k, v in e.fields.items():
            if isinstance(v, str) and (k.endswith("_msg") or k == "msg"):
                out.append((k, v, {}))
    for f in ("repeat_msg", "misplaced_msg", "msg", "unknown_msg",
              "leftover_msg"):
        v = getattr(e, f, "")
        if isinstance(v, str) and v:
            out.append((f, v, {}))
    for kind in ("checks", "close_checks", "requires"):
        lst = getattr(e, kind, None)
        if lst is None and isinstance(e, Section):
            lst = e.fields.get(kind)
        for c in lst or []:
            out.append((kind, c.msg, {}))
    for a in getattr(e, "args", []) or []:
        if a.msg:
            out.append(("args.msg", a.msg, {}))
        for msg in a.refused_words.values():
            out.append(("args.refused_words", msg, {}))
    if isinstance(e, Block) and e.refuse_msg:
        for r in e.refuse:
            out.append(("refuse_msg", e.refuse_msg,
                        {"key": r.key, "why": r.why}))
    return out


def shared_templates(reg: "Registry"
                     ) -> list[tuple[str, str, str, dict[str, str]]]:
    """The templates any case may show: those of [lexing], [reading],
    [structure], the blocks, the groups, and the removal message with each
    hint: (id, field, template, fill)."""
    out = []
    for e in [reg.lexing, reg.reading, reg.structure] + list(reg.blocks) + \
            list(reg.groups.values()):
        out += [(e.id, f, t, fill) for f, t, fill in entry_templates(e)]
    removed = reg.structure.fields.get("removed_msg", "")
    for rm in reg.removed:
        out.append((reg.structure.id, "removed_msg", removed,
                    {"hint": rm.hint, "key": rm.name}))
    return out


def own_templates(reg: "Registry", e: Any
                  ) -> list[tuple[str, str, str, dict[str, str]]]:
    """The templates of an entry, (id, field, template, fill); an excluded
    PARAM key takes those of the PARAM policies of its optimisers, with its
    name and its reason filled in."""
    out = [(e.id, f, t, fill) for f, t, fill in entry_templates(e)]
    if isinstance(e, ParamExcluded):
        for p in reg.param_policies:
            if set(p.optimizers) & set(e.optimizers):
                for f, t, _ in entry_templates(p):
                    if "{why}" in t:
                        out.append((p.id, f, t,
                                    {"why": e.why, "key": e.name}))
    return out


def sentences(text: str) -> int:
    """The number of sentences of a documentation text."""
    t = re.sub(r"`[^`]*`", "x", text)
    n = 0
    for m in re.finditer(r"[.!?](?=\s+[A-Z(`'\"]|\s*$)", t):
        if re.search(r"\b(e\.g|i\.e|etc|vs|cf)\.$", t[:m.end()]):
            continue
        n += 1
    return max(n, 1 if t.strip() else 0)


# -------------------------------------------------------------- schema ----
# Per kind: field -> (type, required). Types: str, int, bool, number,
# list[str], list[int], dict[str,str], int|str, and the structured ones
# (cases, args, checks, refuse, default, params) read by their builders.

S, R = "str", True
O = False
CASE_FIELDS = {"wrap": ("str", R), "text": ("str", R), "check": ("str", O),
               "context": ("str", O), "env": ("dict[str,str]", O),
               "doc": ("str", O)}
INVALID_FIELDS = dict(CASE_FIELDS, error=("str", R), at=("int|str", O),
                      msg_of=("str", O))
CHECK_FIELDS = {"msg": ("str", R), "doc": ("str", R)}
ARG_FIELDS = {"name": ("str", R), "kind": ("str", R),
              "optional": ("bool", O), "repeat": ("bool", O),
              "min": ("number|str", O), "max": ("number|str", O),
              "min_open": ("bool", O), "integral": ("bool", O),
              "words": ("list[str]", O), "refused_words": ("dict[str,str]", O),
              "what": ("str", O), "msg": ("str", O), "noun": ("str", O),
              "default": ("any", O), "doc": ("str", O)}
DOC_FIELDS = {"examples": ("cases", O), "invalid": ("invalid", O),
              "untested": ("untested", O)}

SCHEMA: dict[str, dict[str, tuple[str, bool]]] = {
    "meta": {"version": ("int", R), "api": ("int", R),
             "parts": ("list[str]", R), "anneal_keys": ("str", R),
             "optimizers": ("str", R), "optimizer_docs": ("str", R)},
    "lexing": dict(DOC_FIELDS, doc=("str", R), long_doc=("str", O),
                   comment=("str", R),
                   continuation=("str", R), key_case=("str", R),
                   error_frame=("str", R), unfinished_msg=("str", R),
                   missing_file_msg=("str", R), bool_true=("list[str]", R),
                   bool_false=("list[str]", R),
                   timespec_words=("list[str]", R)),
    "reading": dict(DOC_FIELDS, doc=("str", R), long_doc=("str", O),
                    integer_msg=("str", R),
                    integer_range_msg=("str", R), note_msg=("str", R),
                    finite_msg=("str", R), positive_msg=("str", R),
                    underflow_note=("str", R), requires_value_msg=("str", R),
                    single_value_msg=("str", R), bool_msg=("str", R),
                    timespec_msg=("str", R), script_timeout_msg=("str", R),
                    script_timeout_large_msg=("str", R)),
    "structure": dict(DOC_FIELDS, doc=("str", R), long_doc=("str", O),
                      check_order=("list[str]", R),
                      end_without_open_msg=("str", R),
                      end_mismatch_msg=("str", R),
                      end_takes_no_value_msg=("str", R),
                      open_takes_no_value_msg=("str", R),
                      unclosed_msg=("str", R), unknown_root_msg=("str", R),
                      root_key_in_block_msg=("str", R),
                      removed_msg=("str", R)),
    "optim": dict(DOC_FIELDS, doc=("str", R), long_doc=("str", O),
                  synopsis=("list[str]", R),
                  positive_only=("list[str]", O),
                  auto_excluded_prefix=("str", O), checks=("checks", O)),
    "lammps_script": dict(DOC_FIELDS, doc=("str", R), long_doc=("str", O),
                          fixture=("str", R)),
    "group": dict(DOC_FIELDS, blocks=("list[str]", R),
                  leftover_msg=("str", R), doc=("str", R),
                  long_doc=("str", O)),
    "block": dict(DOC_FIELDS, name=("str", R), end=("str", R),
                  parents=("list[str]", R), repeat=("str", R),
                  repeat_msg=("str", O), misplaced_msg=("str", R),
                  unknown_msg=("str", R), unknown_first=("bool", O),
                  aliases=("dict[str,str]", O), refuse=("refuse", O),
                  refuse_msg=("str", O), unreachable_msgs=("list[str]", O),
                  requires=("checks", O), close_checks=("checks", O),
                  quirk=("str", O), doc=("str", R), long_doc=("str", O)),
    "key": dict(DOC_FIELDS, id=("str", R), name=("str", R),
                group=("str", O), blocks=("list[str]", O),
                synopsis=("str", R), args=("args", O), repeat=("str", R),
                repeat_msg=("str", O), repeat_doc=("str", O),
                misplaced_msg=("str", O),
                default=("default", O), doc=("str", R), long_doc=("str", O),
                alias_of=("str", O), refused=("str", O), quirk=("str", O),
                env=("str", O), checks=("checks", O)),
    "line_option": dict(DOC_FIELDS, id=("str", R), option=("str", R),
                        lines=("list[str]", R), synopsis=("str", R),
                        args=("args", O), repeat=("str", O),
                        repeat_msg=("str", O), default=("default", O),
                        doc=("str", R), long_doc=("str", O),
                        refused=("str", O), quirk=("str", O),
                        checks=("checks", O)),
    "form": dict(DOC_FIELDS, id=("str", R), line=("str", R),
                 word=("str", O), synopsis=("str", R), args=("args", O),
                 doc=("str", R), long_doc=("str", O), refused=("str", O),
                 quirk=("str", O), checks=("checks", O)),
    "removed": dict(DOC_FIELDS, name=("str", R), hint=("str", R),
                    doc=("str", O)),
    "rule": dict(DOC_FIELDS, id=("str", R), title=("str", R),
                 doc=("str", R), check=("str", R), msg=("str", O),
                 msg_doc=("str", O), keys=("list[str]", O),
                 quirk=("str", O)),
    "env": dict(DOC_FIELDS, name=("str", R), key=("str", R),
                doc=("str", R), values=("list[str]", O), msg=("str", O),
                checks=("checks", O)),
    "loss_function": dict(DOC_FIELDS, id=("str", R), name=("str", R),
                          params=("params", O), doc=("str", R)),
    "strategy_type": dict(DOC_FIELDS, id=("str", R), name=("str", R),
                          keys=("list[str]", O), doc=("str", R),
                          quirk=("str", O), checks=("checks", O)),
    "init_method": dict(DOC_FIELDS, id=("str", R), name=("str", R),
                        args=("args", O), doc=("str", R),
                        refused=("str", O), quirk=("str", O),
                        checks=("checks", O)),
    "optim_target": dict(DOC_FIELDS, id=("str", R), key=("str", R),
                         status=("str", R), indices=("list[int]", O),
                         msg=("str", O), doc=("str", R),
                         condition=("str", O)),
    "param": dict(DOC_FIELDS, id=("str", R), name=("str", R),
                  optimizers=("list[str]", R), args=("args", O),
                  default=("default", O), indexed=("bool", O),
                  doc=("str", R), long_doc=("str", O), quirk=("str", O),
                  checks=("checks", O)),
    "param_excluded": dict(DOC_FIELDS, id=("str", R), name=("str", R),
                           optimizers=("list[str]", R), why=("str", R),
                           doc=("str", O)),
    "param_policy": dict(DOC_FIELDS, id=("str", R),
                         optimizers=("list[str]", R), doc=("str", R),
                         msg=("str", O), quirk=("str", O),
                         checks=("checks", O)),
    "wrapper": {"text": ("str", R), "context": ("str", O),
                "doc": ("str", R), "block": ("str", R),
                "optimizer": ("str", O)},
    "context": {"config": ("str", R), "has_train_db": ("bool", O),
                "has_validation": ("bool", O), "doc": ("str", R)},
    "fixture": {"path": ("str", R), "text": ("str", R), "doc": ("str", O)},
}
# Top-level tables: where each may be written and its TOML shape.
MAIN_ONLY = {"meta", "lexing", "reading", "structure", "group", "block",
             "abbreviations"}
SINGLE = {"meta", "lexing", "reading", "structure", "optim",
          "lammps_script"}
# [abbreviations]: a table of abbreviation = what it stands for.
WORD_TABLES = {"abbreviations"}
NAMED = {"group", "wrapper", "context", "fixture"}
ARRAYS = {"block", "key", "line_option", "form", "removed", "rule", "env",
          "loss_function", "strategy_type", "init_method", "optim_target",
          "param", "param_excluded", "param_policy"}
DOC_TEXT_FIELDS = {"doc", "long_doc", "quirk", "condition", "title",
                   "repeat_doc", "msg_doc"}
ABBREVIATION_RE = re.compile(r"^[A-Za-z][A-Za-z0-9+\-]*$")
TITLE_WIDTH = 80


# -------------------------------------------------------------- loader ----

class _Loader:
    def __init__(self, directory: pathlib.Path, parts: list[str] | None,
                 resolve: bool = True, templates: bool = True):
        self.dir = directory
        self.must_resolve = resolve
        self.templates = templates
        self.errors: list[str] = []
        self.notes: list[str] = []
        self.only_parts = parts
        self.ids: dict[str, str] = {}

    # ---- reporting -------------------------------------------------------
    def err(self, where: str, msg: str) -> None:
        self.errors.append(f"{where}: {msg}")

    # ---- typed field reading ---------------------------------------------
    def fields(self, where: str, kind: str, table: Any) -> dict[str, Any]:
        """The fields of `table` as the schema of `kind` declares them."""
        if not isinstance(table, dict):
            self.err(where, f"is not a table")
            return {}
        schema = SCHEMA[kind]
        for name in table:
            if name not in schema:
                self.err(where, f"unknown field '{name}' (a {kind} has "
                                f"{', '.join(sorted(schema))})")
        out: dict[str, Any] = {}
        for name, (typ, required) in schema.items():
            if name not in table:
                if required:
                    self.err(where, f"required field '{name}' is missing")
                continue
            value = table[name]
            if self.type_ok(typ, value):
                out[name] = value
            else:
                self.err(where, f"field '{name}' is not {typ}")
        return out

    @staticmethod
    def type_ok(typ: str, v: Any) -> bool:
        if typ == "str":
            return isinstance(v, str)
        if typ == "int":
            return isinstance(v, int) and not isinstance(v, bool)
        if typ == "bool":
            return isinstance(v, bool)
        if typ == "number|str":
            return (isinstance(v, (int, float, str))
                    and not isinstance(v, bool))
        if typ == "int|str":
            return (isinstance(v, (int, str)) and not isinstance(v, bool))
        if typ == "list[str]":
            return isinstance(v, list) and all(isinstance(x, str) for x in v)
        if typ == "list[int]":
            return isinstance(v, list) and all(
                isinstance(x, int) and not isinstance(x, bool) for x in v)
        if typ == "dict[str,str]":
            return isinstance(v, dict) and all(
                isinstance(x, str) for x in v.values())
        if typ in ("cases", "invalid", "args", "checks", "refuse", "params",
                   "untested"):
            return isinstance(v, list) and all(isinstance(x, dict) for x in v)
        if typ == "default":
            return isinstance(v, dict)
        if typ == "any":
            return True
        raise AssertionError(typ)

    # ---- structured fields -------------------------------------------------
    def cases(self, where: str, owner: str, part: str, f: dict) -> tuple:
        out = []
        for kind, name in (("example", "examples"), ("invalid", "invalid")):
            got = []
            for i, t in enumerate(f.get(name, [])):
                w = f"{where}.{name}[{i}]"
                c = self.fields(w, "_invalid" if kind == "invalid"
                                else "_case", t)
                if not c:
                    continue
                check = c.get("check", "startup" if kind == "example"
                              else "parse")
                if kind == "example" and check not in CHECKS:
                    self.err(w, f"check '{check}' is not parse or startup")
                if kind == "invalid" and check not in INVALID_CHECKS:
                    self.err(w, f"check '{check}' is not parse, startup or "
                                "check")
                if check == "check" and c.get("at", 1) != "none":
                    self.err(w, "check = \"check\" shows a refusal of the "
                                "run, which names no line: at = \"none\"")
                at = c.get("at", 1)
                if isinstance(at, int) and at < 1:
                    self.err(w, "at counts logical lines from 1")
                if isinstance(at, str) and at != "none" and not WORD_RE.match(at):
                    self.err(w, f"at '{at}' is not a line number, a key or none")
                for fld in ("text", "error"):
                    if fld in c:
                        self.no_placeholders(w, fld, c[fld])
                self.doc_text(w, "doc", c.get("doc", ""))
                got.append(Case(owner=owner, part=part, kind=kind, index=i,
                                wrap=c.get("wrap", ""), text=c.get("text", ""),
                                check=check, context=c.get("context"),
                                env=dict(c.get("env", {})),
                                doc=c.get("doc", ""), error=c.get("error"),
                                at=at, msg_of=c.get("msg_of", "")))
            out.append(got)
        return tuple(out)

    def args(self, where: str, f: dict) -> list[Arg]:
        out = []
        for i, t in enumerate(f.get("args", [])):
            w = f"{where}.args[{i}]"
            a = self.fields(w, "_arg", t)
            if not a:
                continue
            kind = a.get("kind", "")
            if kind not in ARG_KINDS:
                self.err(w, f"kind '{kind}' is not one of "
                            f"{', '.join(sorted(ARG_KINDS))}")
            if ("min" in a or "max" in a) and kind not in (
                    "integer", "number", "positive"):
                self.err(w, f"min and max belong to a number, not a {kind}")
            for lim in ("min", "max"):
                v = a.get(lim)
                if isinstance(v, str) and v not in INT_LIMITS:
                    self.err(w, f"{lim} '{v}' is not a number or one of "
                                f"{', '.join(INT_LIMITS)}")
            if a.get("min_open") and "min" not in a:
                self.err(w, "min_open needs a min")
            if "integral" in a and kind != "number":
                self.err(w, "integral belongs to an argument of kind number")
            if ("words" in a or "refused_words" in a) and kind != "word":
                self.err(w, "words belong to an argument of kind word")
            if kind == "word" and not a.get("words"):
                self.err(w, "an argument of kind word lists its words")
            if "noun" in a and kind != "positive":
                self.err(w, "noun belongs to an argument of kind positive")
            for fld in ("what", "msg"):
                if fld in a:
                    self.placeholders(w, fld, a[fld], PLACEHOLDERS)
            for word, msg in a.get("refused_words", {}).items():
                self.placeholders(w, f"refused_words.{word}", msg,
                                  PLACEHOLDERS)
            self.doc_text(w, "doc", a.get("doc", ""))
            out.append(Arg(**a))
        return out

    def checks(self, where: str, f: dict, name: str) -> list[Check]:
        out = []
        for i, t in enumerate(f.get(name, [])):
            w = f"{where}.{name}[{i}]"
            c = self.fields(w, "_check", t)
            if not c:
                continue
            self.placeholders(w, "msg", c.get("msg", ""), PLACEHOLDERS)
            # A check's doc says when its message comes and what the
            # message's placeholders hold, so it may name them.
            self.voice(w, "doc", c.get("doc", ""))
            out.append(Check(**c))
        return out

    def untested(self, where: str, f: dict) -> list[Untested]:
        out = []
        for i, t in enumerate(f.get("untested", [])):
            w = f"{where}.untested[{i}]"
            u = self.fields(w, "_untested", t)
            if not u:
                continue
            self.placeholders(w, "msg", u.get("msg", ""), PLACEHOLDERS)
            self.voice(w, "why", u.get("why", ""))
            out.append(Untested(**u))
        return out

    def default(self, where: str, f: dict) -> Default | None:
        d = f.get("default")
        if d is None:
            return None
        extra = set(d) - {"literal", "text", "unchecked"}
        kinds = {"literal", "text"} & set(d)
        if extra or len(kinds) != 1 or not all(isinstance(v, str)
                                               for v in d.values()):
            self.err(where, "default is { literal = \"...\" } or "
                            "{ text = \"...\" }, with unchecked = "
                            "\"...\" for a text")
            return None
        if "unchecked" in d and "text" not in d:
            self.err(where, "default.unchecked goes with a text default")
        self.doc_text(where, "default.text", d.get("text", ""))
        self.voice(where, "default.unchecked", d.get("unchecked", ""))
        return Default(**d)

    # ---- text checks -------------------------------------------------------
    def placeholders(self, where: str, fld: str, text: str,
                     allowed: set[str]) -> None:
        for name in PLACEHOLDER.findall(text):
            if name not in allowed:
                self.err(where, f"{fld}: placeholder {{{name}}} is not one "
                                f"of {', '.join(sorted(allowed))}")

    def no_placeholders(self, where: str, fld: str, text: str) -> None:
        for name in PLACEHOLDER.findall(text):
            self.err(where, f"{fld}: holds the placeholder {{{name}}}; an "
                            "example or an error is the text itself")

    def voice(self, where: str, fld: str, text: str) -> None:
        m = READER_WORDS.search(text)
        if m:
            self.err(where, f"{fld}: '{m.group(0)}' addresses the reader; "
                            "documentation text is impersonal")

    def doc_text(self, where: str, fld: str, text: str) -> None:
        """Documentation text: impersonal, and without the placeholders of
        a message (the reference pages show it before the message it may
        describe)."""
        self.voice(where, fld, text)
        for name in PLACEHOLDER.findall(text):
            self.err(where, f"{fld}: holds the placeholder {{{name}}}; "
                            "documentation text says in words what a "
                            "message holds")

    def message_fields(self, where: str, kind: str, f: dict) -> None:
        for name, value in f.items():
            if not isinstance(value, str):
                continue
            if name in DOC_TEXT_FIELDS:
                self.doc_text(where, name, value)
            if not (name == "msg" or name.endswith("_msg")
                    or name in ("error_frame", "underflow_note")):
                continue
            allowed = FIXED_PLACEHOLDERS.get((kind, name))
            if allowed is None:
                allowed = (READING_PLACEHOLDERS if kind == "reading"
                           else PLACEHOLDERS)
            self.placeholders(where, name, value, allowed)

    def register_id(self, where: str, ident: str) -> None:
        if not ID_RE.match(ident):
            self.err(where, f"id '{ident}' has a character outside "
                            "A-Z a-z 0-9 _ . : - [ ] ( )")
        if ident in self.ids:
            self.err(where, f"id '{ident}' is also the id of an entry of "
                            f"{self.ids[ident]}")
        else:
            self.ids[ident] = where

    # ---- files -------------------------------------------------------------
    def read(self, name: str) -> dict:
        path = self.dir / name
        try:
            with open(path, "rb") as fh:
                data = tomllib.load(fh)
        except FileNotFoundError:
            self.err(name, "file not found")
            return {}
        except tomllib.TOMLDecodeError as e:
            self.err(name, f"not TOML: {e}")
            return {}
        for problem in table_layout(path.read_text(encoding="utf-8")):
            self.err(name, problem)
        return data

    def run(self) -> Registry:
        main = self.read(MAIN_FILE)
        if self.errors:
            raise RegistryError(self.errors)
        meta_f = self.fields(f"{MAIN_FILE} [meta]", "meta",
                             main.get("meta", {}))
        if meta_f.get("api") not in (None, API):
            self.err(f"{MAIN_FILE} [meta]", f"api {meta_f.get('api')} is not "
                                            f"the loader's {API}")
        parts = meta_f.get("parts", [])
        if len(set(parts)) != len(parts) or MAIN_FILE in parts:
            self.err(f"{MAIN_FILE} [meta]", "parts lists a file twice or "
                                            f"lists {MAIN_FILE}")
        selected = parts
        if self.only_parts is not None:
            for p in self.only_parts:
                if p != MAIN_FILE and p not in parts:
                    self.err(p, f"is not a part file of meta.parts "
                                f"({', '.join(parts)})")
            selected = [p for p in parts if p in self.only_parts]
        if self.errors:
            raise RegistryError(self.errors)
        docs = [(MAIN_FILE, main)] + [(p, self.read(p)) for p in selected]
        full = selected == parts
        self.full = full

        acc: dict[str, Any] = {k: [] for k in ARRAYS}
        acc.update({k: {} for k in NAMED})
        singles: dict[str, tuple[str, dict]] = {}
        words: dict[str, dict] = {}
        for part, doc in docs:
            for top, value in doc.items():
                where = f"{part} [{top}]"
                if top in WORD_TABLES:
                    if part != MAIN_FILE:
                        self.err(where, f"belongs in {MAIN_FILE}")
                    elif not isinstance(value, dict):
                        self.err(where, "is not a table")
                    else:
                        words[top] = value
                    continue
                if top not in SCHEMA or top.startswith("_"):
                    self.err(where, "unknown table (the registry has "
                                    f"{', '.join(sorted(SCHEMA))})")
                    continue
                if top in MAIN_ONLY and part != MAIN_FILE:
                    self.err(where, f"belongs in {MAIN_FILE}")
                    continue
                if top in SINGLE:
                    if top in singles:
                        self.err(where, f"is also in {singles[top][0]}")
                    elif not isinstance(value, dict):
                        self.err(where, "is not a table")
                    else:
                        singles[top] = (part, value)
                elif top in NAMED:
                    if not isinstance(value, dict):
                        self.err(where, "is not a table of named tables")
                        continue
                    for name, t in value.items():
                        if name in acc[top]:
                            self.err(f"{part} [{top}.{name}]",
                                     f"is also in {acc[top][name][0]}")
                        else:
                            acc[top][name] = (part, t)
                else:
                    if not isinstance(value, list):
                        self.err(where, f"is not an array of tables "
                                        f"([[{top}]])")
                        continue
                    for i, t in enumerate(value):
                        acc[top].append((part, i, t))
        for top in ("meta", "lexing", "reading", "structure"):
            if top not in singles:
                self.err(MAIN_FILE, f"[{top}] is missing")
        if self.errors:
            raise RegistryError(self.errors)
        return self.build(meta_f, singles, acc, full, [d[0] for d in docs],
                          words)

    # ---- building ----------------------------------------------------------
    def section(self, top: str, part: str, t: dict) -> Section:
        where = f"{part} [{top}]"
        f = self.fields(where, top, t)
        ident = top.upper()
        self.register_id(where, ident)
        self.message_fields(where, top, f)
        ex, inv = self.cases(where, ident, part, f)
        extra = {k: v for k, v in f.items()
                 if k not in ("doc", "long_doc", "examples", "invalid",
                              "untested")}
        if top == "optim":
            extra["checks"] = self.checks(where, f, "checks")
        return Section(id=ident, part=part, fields=extra,
                       doc=f.get("doc", ""), long_doc=f.get("long_doc", ""),
                       untested=self.untested(where, f), examples=ex,
                       invalid=inv)

    def build(self, meta_f, singles, acc, full, files, words) -> Registry:
        meta = Meta(**meta_f)
        lexing = self.section("lexing", *singles["lexing"])
        reading = self.section("reading", *singles["reading"])
        structure = self.section("structure", *singles["structure"])
        optim = (self.section("optim", *singles["optim"])
                 if "optim" in singles else None)
        lammps_script = (self.section("lammps_script",
                                      *singles["lammps_script"])
                         if "lammps_script" in singles else None)
        abbreviations = self.abbreviations(words.get("abbreviations", {}))
        order = structure.fields.get("check_order", [])
        if order != CHECK_STAGES:
            self.err(f"{MAIN_FILE} [structure]",
                     f"check_order is not {CHECK_STAGES}")

        groups: dict[str, Group] = {}
        for name, (part, t) in acc["group"].items():
            where = f"{part} [group.{name}]"
            f = self.fields(where, "group", t)
            ident = f"GROUP.{name}"
            self.register_id(where, ident)
            self.message_fields(where, "group", f)
            ex, inv = self.cases(where, ident, part, f)
            groups[name] = Group(id=ident, name=name, part=part,
                                 blocks=f.get("blocks", []),
                                 leftover_msg=f.get("leftover_msg", ""),
                                 doc=f.get("doc", ""),
                                 long_doc=f.get("long_doc", ""),
                                 untested=self.untested(where, f),
                                 examples=ex, invalid=inv)

        blocks: list[Block] = []
        for part, i, t in acc["block"]:
            where = f"{part} [[block]] {t.get('name', i)}"
            f = self.fields(where, "block", t)
            ident = f"BLOCK.{f.get('name', i)}"
            self.register_id(where, ident)
            self.message_fields(where, "block", f)
            ex, inv = self.cases(where, ident, part, f)
            refuse = []
            for j, r in enumerate(f.get("refuse", [])):
                w = f"{where}.refuse[{j}]"
                rf = self.fields(w, "_refuse", r)
                if rf:
                    refuse.append(Refusal(**rf))
            kw = {k: v for k, v in f.items()
                  if k not in ("examples", "invalid", "refuse", "requires",
                               "close_checks", "untested")}
            b = Block(id=ident, part=part, refuse=refuse,
                      untested=self.untested(where, f),
                      requires=self.checks(where, f, "requires"),
                      close_checks=self.checks(where, f, "close_checks"),
                      examples=ex, invalid=inv, **kw)
            if b.repeat not in BLOCK_REPEAT:
                self.err(where, f"repeat '{b.repeat}' is not one of "
                                f"{', '.join(sorted(BLOCK_REPEAT))}")
            if b.repeat.startswith("error") != bool(b.repeat_msg):
                self.err(where, "repeat_msg goes with repeat error or "
                                "error_per_parent, and only with them")
            if bool(b.refuse) != bool(b.refuse_msg):
                self.err(where, "refuse and refuse_msg go together")
            if b.end != "END" + b.name:
                self.err(where, f"end is END{b.name}")
            blocks.append(b)
        block_names = [b.name for b in blocks]
        if len(set(block_names)) != len(block_names):
            self.err(MAIN_FILE, "a block name is given twice")
        places = set(block_names) | {"ROOT"}
        for b in blocks:
            for p in b.parents:
                if p not in places:
                    self.err(b.id, f"parent '{p}' is not ROOT or a block")
        for g in groups.values():
            for bn in g.blocks:
                if bn not in block_names:
                    self.err(g.id, f"block '{bn}' is not a block")

        def entry_list(kind: str, cls, ident_of=None, extra_build=None):
            out = []
            for part, i, t in acc[kind]:
                label = t.get("id") or t.get("name") or i
                where = f"{part} [[{kind}]] {label}"
                f = self.fields(where, kind, t)
                if not f:
                    continue
                ident = ident_of(f) if ident_of else f.get("id", "")
                self.register_id(where, ident)
                self.message_fields(where, kind, f)
                ex, inv = self.cases(where, ident, part, f)
                kw = {k: v for k, v in f.items()
                      if k not in ("examples", "invalid", "args", "checks",
                                   "default", "params", "id", "untested")}
                more: dict[str, Any] = {"untested": self.untested(where, f)}
                if "args" in SCHEMA[kind]:
                    more["args"] = self.args(where, f)
                if "checks" in SCHEMA[kind]:
                    more["checks"] = self.checks(where, f, "checks")
                if "default" in SCHEMA[kind]:
                    more["default"] = self.default(where, f)
                if extra_build:
                    more.update(extra_build(where, f))
                try:
                    e = cls(id=ident, part=part, examples=ex, invalid=inv,
                            **kw, **more)
                except TypeError as exc:
                    self.err(where, f"cannot be built: {exc}")
                    continue
                e._where = where  # type: ignore[attr-defined]
                out.append(e)
            return out

        def loss_params(where, f):
            out = []
            for j, p in enumerate(f.get("params", [])):
                pf = self.fields(f"{where}.params[{j}]", "_param", p)
                if pf:
                    out.append(LossParam(**pf))
            return {"params": out}

        keys = entry_list("key", Key)
        line_options = entry_list("line_option", LineOption)
        forms = entry_list("form", Form)
        removed = entry_list("removed", Removed,
                             ident_of=lambda f: f"REMOVED.{f.get('name')}")
        rules = entry_list("rule", Rule)
        envs = entry_list("env", Env,
                          ident_of=lambda f: f"ENV.{f.get('name')}")
        loss_functions = entry_list("loss_function", LossFunction,
                                    extra_build=loss_params)
        strategy_types = entry_list("strategy_type", StrategyType)
        init_methods = entry_list("init_method", InitMethod)
        optim_targets = entry_list("optim_target", OptimTarget)
        params = entry_list("param", Param)
        param_excluded = entry_list("param_excluded", ParamExcluded)
        param_policies = entry_list("param_policy", ParamPolicy)

        wrappers = {}
        for name, (part, t) in acc["wrapper"].items():
            where = f"{part} [wrapper.{name}]"
            f = self.fields(where, "wrapper", t)
            if f:
                self.voice(where, "doc", f.get("doc", ""))
                wrappers[name] = Wrapper(name=name, part=part, **f)
        contexts = {}
        for name, (part, t) in acc["context"].items():
            where = f"{part} [context.{name}]"
            f = self.fields(where, "context", t)
            if f:
                self.voice(where, "doc", f.get("doc", ""))
                contexts[name] = TrainingContext(name=name, part=part, **f)
        fixtures = {}
        for name, (part, t) in acc["fixture"].items():
            where = f"{part} [fixture.{name}]"
            f = self.fields(where, "fixture", t)
            if f:
                p = f.get("path", "")
                if (not p or p.startswith("/") or ".." in p.split("/")):
                    self.err(where, f"path '{p}' is not a relative path "
                                    "inside the working directory")
                fixtures[name] = Fixture(name=name, part=part, **f)

        reg = Registry(
            directory=self.dir, files=files, full=full, meta=meta,
            lexing=lexing, reading=reading, structure=structure, optim=optim,
            groups=groups, blocks=blocks, keys=keys,
            line_options=line_options, forms=forms, removed=removed,
            rules=rules, envs=envs, loss_functions=loss_functions,
            strategy_types=strategy_types, init_methods=init_methods,
            optim_targets=optim_targets, params=params,
            param_excluded=param_excluded, param_policies=param_policies,
            wrappers=wrappers, contexts=contexts, fixtures=fixtures,
            lammps_script=lammps_script, abbreviations=abbreviations)
        if lammps_script is not None:
            name = lammps_script.fields.get("fixture", "")
            if name not in fixtures:
                self.err(f"{lammps_script.part} [lammps_script]",
                         f"fixture '{name}' is not a [fixture.*] table")
        reg.optimizer_docs = self.optimizer_docs(reg)
        self.check_entries(reg)
        self.check_references(reg)
        self.check_cases(reg)
        self.check_harness(reg)
        self.check_words(reg)
        reg.notes = self.notes
        if self.errors:
            raise RegistryError(self.errors)
        return reg

    def abbreviations(self, table: dict) -> dict[str, str]:
        where = f"{MAIN_FILE} [abbreviations]"
        out = {}
        for word, text in table.items():
            if not ABBREVIATION_RE.match(word):
                self.err(where, f"abbreviation '{word}' is not one word")
            if not isinstance(text, str) or not text.strip():
                self.err(where, f"abbreviation '{word}' is not spelled out "
                                "as text")
                continue
            self.doc_text(where, word, text)
            out[word] = text
        return out

    def optimizer_docs(self, reg: Registry) -> OptimizerDocs | None:
        """The descriptions of meta.optimizer_docs, checked against the
        optimiser list: one per library and per registered optimiser."""
        name = reg.meta.optimizer_docs
        if not (self.dir / name).exists():
            self.err(name, "file not found")
            return None
        data = self.read(name)
        docs = OptimizerDocs(libraries={}, optimizers={})
        for top, value in data.items():
            if top not in ("library", "optimizer") or not isinstance(value,
                                                                     dict):
                self.err(f"{name} [{top}]", "unknown table (the file has "
                                             "[library.<LIB>] and "
                                             "[optimizer.<LIB:ALGO>])")
                continue
            target = docs.libraries if top == "library" else docs.optimizers
            for key, t in value.items():
                where = f"{name} [{top}.{key}]"
                f = self.fields(where, "_described", t)
                if "doc" in f:
                    self.doc_text(where, "doc", f["doc"])
                    target[key] = f["doc"]
        registered = self.optimiser_names(reg)
        if registered is None:
            return docs
        libraries = {o.split(":")[0] for o in registered}
        for key in docs.optimizers:
            if key not in registered:
                self.err(f"{name} [optimizer.{key}]", "is not a registered "
                         f"optimiser of {reg.meta.optimizers}")
        for key in docs.libraries:
            if key not in libraries:
                self.err(f"{name} [library.{key}]", "is not a library of "
                         f"{reg.meta.optimizers}")
        for key in sorted(registered - set(docs.optimizers)):
            self.err(name, f"{key}, a registered optimiser, has no doc")
        for key in sorted(libraries - set(docs.libraries)):
            self.err(name, f"the library {key} has no doc")
        return docs

    # ---- cross-entry checks ------------------------------------------------
    def check_entries(self, reg: Registry) -> None:
        places = {b.name for b in reg.blocks} | {"ROOT"}
        for k in reg.keys:
            w = k._where  # type: ignore[attr-defined]
            if bool(k.group) == bool(k.blocks):
                self.err(w, "a key has a group or blocks, one of them")
            if k.group and k.group not in reg.groups:
                self.err(w, f"group '{k.group}' is not a group")
            for bn in k.blocks:
                if bn not in places:
                    self.err(w, f"block '{bn}' is not ROOT or a block")
            scope = k.group or (k.blocks[0] if k.blocks else "")
            if not (k.id == f"{scope}.{k.name}" or
                    any(k.id == f"{b}.{k.name}" for b in k.blocks)):
                self.err(w, f"id is <scope>.{k.name}, the scope its group "
                            "or one of its blocks")
            if k.repeat not in KEY_REPEAT:
                self.err(w, f"repeat '{k.repeat}' is not one of "
                            f"{', '.join(sorted(KEY_REPEAT))}")
            if (k.repeat == "error") != bool(k.repeat_msg):
                self.err(w, "repeat_msg goes with repeat error, and only "
                            "with it")
        # Two keys of one name valid in a common block.
        seen: dict[tuple[str, str], str] = {}
        for k in reg.keys:
            where = (reg.groups[k.group].blocks if k.group in reg.groups
                     else k.blocks)
            for bn in where:
                if (bn, k.name) in seen:
                    self.err(k._where, f"{k.name} is also valid in {bn} "  # type: ignore[attr-defined]
                                       f"as {seen[(bn, k.name)]}")
                else:
                    seen[(bn, k.name)] = k.id
        for o in reg.line_options:
            w = o._where  # type: ignore[attr-defined]
            if not o.id.endswith("." + o.option) or o.id == "." + o.option:
                self.err(w, f"id is <line>.{o.option}")
            if o.repeat not in KEY_REPEAT:
                self.err(w, f"repeat '{o.repeat}' is not one of "
                            f"{', '.join(sorted(KEY_REPEAT))}")
            if (o.repeat == "error") != bool(o.repeat_msg):
                self.err(w, "repeat_msg goes with repeat error, and only "
                            "with it")
        seen_opt: dict[tuple[str, str], str] = {}
        for o in reg.line_options:
            for line in o.lines:
                if (line, o.option) in seen_opt:
                    self.err(o._where, f"{o.option} is also an option of "  # type: ignore[attr-defined]
                                       f"{line} as {seen_opt[(line, o.option)]}")
                seen_opt[(line, o.option)] = o.id
        for fm in reg.forms:
            if not fm.id.startswith(fm.line + "."):
                self.err(fm._where, f"id is {fm.line}.<form>")  # type: ignore[attr-defined]
        prefixed = [("loss_functions", "LOSS."), ("init_methods", "METHOD."),
                    ("strategy_types", "STRATEGY.TYPE.")]
        for kind, prefix in prefixed:
            for e in getattr(reg, kind):
                if e.id != prefix + e.name:
                    self.err(e._where, f"id is {prefix}{e.name}")  # type: ignore[attr-defined]
        for t in reg.optim_targets:
            if t.id != "OPTIM.TARGET." + t.key:
                self.err(t._where, f"id is OPTIM.TARGET.{t.key}")  # type: ignore[attr-defined]
            if t.status not in ("refused", "special"):
                self.err(t._where, "status is refused or special")  # type: ignore[attr-defined]
        for p in reg.params:
            if not (p.id.startswith("PARAM.") and p.id.endswith("." + p.name)
                    and len(p.id) > len("PARAM..") + len(p.name)):
                self.err(p._where, f"id is PARAM.<optimiser>.{p.name}")  # type: ignore[attr-defined]
        for p in reg.param_excluded:
            if not (p.id.startswith("PARAM.") and p.id.endswith("." + p.name)):
                self.err(p._where, f"id is PARAM.<optimiser>.{p.name}")  # type: ignore[attr-defined]
        for p in reg.param_policies:
            if not p.id.startswith("PARAM."):
                self.err(p._where, "id is PARAM.<optimiser or library>")  # type: ignore[attr-defined]
        for r in reg.rules:
            if not r.id.startswith("RULE.") or r.id == "RULE.":
                self.err(r._where, "id is RULE.<name>")  # type: ignore[attr-defined]
            if r.check not in CHECKS:
                self.err(r._where, f"check '{r.check}' is not parse or "  # type: ignore[attr-defined]
                                   "startup")
            if r.title.rstrip().endswith("."):
                self.err(r._where, "title: a title is a phrase without a "  # type: ignore[attr-defined]
                                   "full stop at its end")
            if len(r.title) > TITLE_WIDTH:
                self.err(r._where, f"title: longer than {TITLE_WIDTH} "  # type: ignore[attr-defined]
                                   "characters")

    def resolve(self, where: str, ref: str, known: set[str], what: str) -> None:
        if ref in known:
            return
        if self.full and self.must_resolve:
            self.err(where, f"{what} '{ref}' is not in the registry")
        else:
            self.notes.append(f"{where}: {what} '{ref}' not checked (not in "
                              "the files loaded)")

    def check_references(self, reg: Registry) -> None:
        key_ids = {k.id for k in reg.keys}
        all_ids = set(reg.by_id())
        for b in reg.blocks:
            for word, ref in b.aliases.items():
                if not WORD_RE.match(word):
                    self.err(b.id, f"alias '{word}' is not a key word")
                self.resolve(b.id, ref, key_ids, "alias key")
            group_keys = {k.name for k in reg.keys
                          if k.group in reg.groups and
                          b.name in reg.groups[k.group].blocks}
            for r in b.refuse:
                self.resolve(b.id, r.key, group_keys,
                             "refused key of the group")
        for k in reg.keys:
            if k.alias_of:
                self.resolve(k.id, k.alias_of, key_ids, "alias_of")
            if k.env:
                self.resolve(k.id, "ENV." + k.env, all_ids, "env")
        for t in reg.strategy_types:
            for ref in t.keys:
                self.resolve(t.id, ref, key_ids, "key")
        for r in reg.rules:
            for ref in r.keys:
                self.resolve(r.id, ref, all_ids, "entry")
        for e in reg.envs:
            self.resolve(e.id, e.key, key_ids, "key")
        root_names = {k.name for k in reg.keys if "ROOT" in k.blocks}
        form_ids = {f.id for f in reg.forms}
        for o in reg.line_options:
            for line in o.lines:
                self.resolve(o.id, line, root_names | form_ids,
                             "line (a top-level key or a form)")
        for fm in reg.forms:
            self.resolve(fm.id, fm.line, root_names, "top-level key")
        optimizers = self.optimiser_names(reg)
        for kind in ("params", "param_excluded", "param_policies"):
            for p in getattr(reg, kind):
                for name in p.optimizers:
                    if name == "*" or optimizers is None:
                        continue
                    libs = {o.split(":")[0] for o in optimizers}
                    if name not in (optimizers if ":" in name else libs):
                        self.err(p.id, f"optimiser '{name}' is not '*', a "
                                       "LIB or a registered LIB:ALGO of "
                                       f"{reg.meta.optimizers}")
        for name, w in reg.wrappers.items():
            where = f"{w.part} [wrapper.{name}]"
            lines = w.text.split("\n")
            if lines.count("{line}") != 1 or w.text.count("{line}") != 1:
                self.err(where, "a wrapper has one slot, a line that is "
                                "{line} alone")
            for ph in PLACEHOLDER.findall(w.text):
                if ph != "line":
                    self.err(where, f"text holds the placeholder {{{ph}}}")
            self.context_ref(where, w.part, w.context, reg)

    @staticmethod
    def optimiser_names(reg: Registry) -> set[str] | None:
        path = reg.directory / reg.meta.optimizers
        if not path.exists():
            return None
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
        return {f"{o['lib']}:{o['algo']}" for o in data.get("optimizer", [])
                if o.get("status") == "registered"}

    def context_ref(self, where: str, part: str, name: str,
                    reg: Registry) -> None:
        c = reg.contexts.get(name)
        if c is None:
            self.err(where, f"context '{name}' is not a [context.*] table")
        elif c.part not in (MAIN_FILE, part):
            self.err(where, f"context '{name}' is in {c.part}: a part file "
                            f"uses those of {MAIN_FILE} and its own")

    def check_harness(self, reg: Registry) -> None:
        """What the generated checks and the conformance test rely on, and
        what an entry needs: the version, the wrappers' blocks, the cases
        of each entry, the block of each example of a key, the arguments
        in the synopsis, the messages of the invalid cases, and the length
        of a doc."""
        if reg.meta.version != REGISTRY_VERSION:
            self.err(f"{MAIN_FILE} [meta]", f"version {reg.meta.version} is "
                     f"not the layout this loader reads, {REGISTRY_VERSION}")
        places = {b.name for b in reg.blocks} | {"ROOT"}
        optimizers = self.optimiser_names(reg)
        for name, w in reg.wrappers.items():
            where = f"{w.part} [wrapper.{name}]"
            if w.block not in places:
                self.err(where, f"block '{w.block}' is not ROOT or a block")
            elif w.text.split("\n").count("{line}") == 1:
                marker = "__SLOT__"
                comp = compose(reg, name, marker)
                got = [b for ln, b in line_blocks(reg, comp.file)
                       if ln.key == marker]
                if got and got[0] != w.block:
                    self.err(where, f"block is '{w.block}', but the slot is "
                                    f"in {got[0]}")
            if w.optimizer and optimizers is not None and \
                    w.optimizer not in optimizers:
                self.err(where, f"optimizer '{w.optimizer}' is not a "
                                "registered LIB:ALGO")
        for block, wname in block_wrappers(reg).items():
            w = reg.wrappers.get(wname)
            if w is None or w.part != MAIN_FILE or w.block != block:
                self.err(MAIN_FILE, f"[wrapper.{wname}] is the wrapper of "
                                    f"the block {block} (block = "
                                    f"\"{block}\"), in {MAIN_FILE}")

        # The cases of an entry, and the block of each example of a key.
        for k in reg.keys:
            w = k._where  # type: ignore[attr-defined]
            if not k.refused and not k.examples:
                self.err(w, "a key that is not refused has an example")
            if not k.invalid:
                self.err(w, "a key has an invalid line")
            if k.group in reg.groups:
                allowed = {b for b in reg.groups[k.group].blocks
                           if k.name not in {r.key for r in
                                             reg.block(b).refuse}}
            else:
                allowed = set(k.blocks)
            for c in k.examples:
                if c.wrap != "none" and c.wrap not in reg.wrappers:
                    continue
                found = key_line(reg, c, k.name)
                if found is None:
                    self.err(w, f"example {c.index} has no {k.name} line")
                elif found[2] not in allowed:
                    self.err(w, f"example {c.index} puts {k.name} in "
                                f"{found[2]}, which is not one of its blocks "
                                f"({', '.join(sorted(allowed))})")
        for kind in ("line_options", "forms", "params"):
            for e in getattr(reg, kind):
                if not getattr(e, "refused", "") and not e.examples:
                    self.err(e._where, "an entry that is not refused has "  # type: ignore[attr-defined]
                                       "an example")
                if not e.invalid:
                    self.err(e._where, "an entry has an invalid line")  # type: ignore[attr-defined]

        # Every argument but a word is a <name> of the synopsis.
        for e in reg.entries():
            syn = getattr(e, "synopsis", None)
            if not isinstance(syn, str):
                continue
            for a in getattr(e, "args", []) or []:
                if a.kind not in ("word", "tokens", "loss") and \
                        f"<{a.name}>" not in syn:
                    self.err(e.id, f"the argument <{a.name}> is not in the "
                                   f"synopsis '{syn}'")

        # A doc is one or two sentences; the detail is in long_doc.
        for e in reg.entries():
            d = getattr(e, "doc", "")
            if isinstance(d, str) and d and sentences(d) > DOC_SENTENCES:
                self.err(e.id, f"doc: {sentences(d)} sentences; a doc has "
                               f"at most {DOC_SENTENCES}, the detail goes "
                               "into long_doc")

        # The message of each invalid case is one the registry holds: of
        # its entry, of the entry msg_of names, or of [lexing], [reading],
        # [structure], a block or a group. Each message of an entry is
        # shown by an invalid case, or listed in untested with the reason.
        byid = reg.by_id()
        shared = [(i, f, t, fill, template_regex(t, fill))
                  for i, f, t, fill in shared_templates(reg)]
        matched: set[tuple[str, str, str]] = set()
        for c in reg.cases():
            if c.kind != "invalid" or c.error is None:
                continue
            where = f"{c.part} {c.owner} invalid {c.index}"
            e = byid.get(c.owner)
            cands = own_templates(reg, e) if e is not None else []
            if c.msg_of:
                m = byid.get(c.msg_of)
                if m is None:
                    self.err(where, f"msg_of '{c.msg_of}' is not an entry")
                else:
                    cands = cands + own_templates(reg, m)
            hit = False
            for i, f, t, fill in cands:
                if template_regex(t, fill).match(c.error):
                    matched.add((i, f, t))
                    hit = True
            for i, f, t, fill, rx in shared:
                if rx.match(c.error):
                    matched.add((i, f, t))
                    hit = True
            if not hit:
                self.err(where, "the error is no message of the entry, of "
                                "the entry msg_of names, or of [lexing], "
                                "[reading], [structure], a block or a "
                                "group: add it to the entry's checks (or "
                                "name the entry whose message it is with "
                                f"msg_of): '{c.error[:100]}'")
        if not (self.full and self.templates):
            return
        for e in reg.entries():
            own = {(f, t) for f, t, _ in entry_templates(e)}
            excused = {u.msg for u in getattr(e, "untested", [])}
            excused |= set(getattr(e, "unreachable_msgs", []) or [])
            for u in getattr(e, "untested", []):
                if u.msg not in {t for _, t in own}:
                    self.err(e.id, f"untested: '{u.msg[:80]}' is no message "
                                   "of the entry")
            for f, t in sorted(own):
                if (e.id, f, t) in matched or t in excused:
                    continue
                self.err(e.id, f"{f}: no invalid case shows the message "
                               f"'{t[:90]}': add one, or list it in "
                               "untested with the reason")

    def check_words(self, reg: Registry) -> None:
        """The word of an example at the place of a word argument is one of
        its words, and, in a full load, each word is shown by an example:
        so a word left out of `words` fails while an example shows it, and
        the parser's own words are compared through the examples, which the
        conformance test runs. Every example of the registry counts, its
        own lines, not those of its wrapper: a line of a key in one of its
        blocks, a line that holds an option on one of its lines (an option
        word in any case)."""
        keys = []
        for k in reg.keys:
            if k.refused:
                continue
            for pos, a in word_positions(k.args):
                keys.append((k, pos, a))
        options = []
        for o in reg.line_options:
            if o.refused:
                continue
            for pos, a in word_positions(o.args):
                options.append((o, pos, a))
        if not keys and not options:
            return
        forms = {f.id: f for f in reg.forms}
        shown: dict[tuple[str, int], set[str]] = {}

        def allowed(k: Key) -> set[str]:
            if k.group in reg.groups:
                return {b for b in reg.groups[k.group].blocks
                        if k.name not in {r.key for r in
                                          reg.block(b).refuse}}
            return set(k.blocks)

        def on_line(o: LineOption, toks: list[str]) -> bool:
            for name in o.lines:
                f = forms.get(name)
                if f is None:
                    if toks[0] == name:
                        return True
                elif toks[0] == f.line and (
                        not f.word or f.word.casefold() in
                        {t.casefold() for t in toks[1:]}):
                    return True
            return False

        def see(e: Any, pos: int, a: Arg, tok: str, c: Case) -> None:
            word = word_of(tok, a.words)
            if word is None:
                self.err(e.id, f"{c.part} {c.owner} example {c.index} "
                               f"writes {tok} where the argument <{a.name}> "
                               f"takes one of its words "
                               f"({', '.join(a.words)})")
            else:
                shown.setdefault((e.id, pos), set()).add(word)

        for c in reg.cases():
            if c.kind != "example" or \
                    (c.wrap != "none" and c.wrap not in reg.wrappers):
                continue
            comp = compose(reg, c.wrap, c.text, c.context)
            last = comp.slot + len(c.text.strip("\n").split("\n")) - 1
            for ln, block in line_blocks(reg, comp.file):
                if ln.first < comp.slot or ln.last > last:
                    continue
                toks = ln.text.split()
                for k, pos, a in keys:
                    if toks[0] == k.name and block in allowed(k) and \
                            len(toks) > 1 + pos:
                        see(k, pos, a, toks[1 + pos], c)
                up = [t if t.startswith("--") else t.upper() for t in toks]
                for o, pos, a in options:
                    if o.option not in up[1:] or not on_line(o, toks):
                        continue
                    j = up.index(o.option, 1)
                    if len(toks) > j + 1 + pos:
                        see(o, pos, a, toks[j + 1 + pos], c)
        if not self.full:
            return
        for e, pos, a in keys + options:
            missing = [w for w in a.words
                       if w not in shown.get((e.id, pos), set())]
            if missing:
                self.err(e.id, f"no example shows the word"
                               f"{'s' if len(missing) > 1 else ''} "
                               f"{', '.join(missing)} of the argument "
                               f"<{a.name}>: add an example that writes it")

    def check_cases(self, reg: Registry) -> None:
        for c in reg.cases():
            where = f"{c.part} {c.owner} {c.kind} {c.index}"
            if c.wrap != "none":
                w = reg.wrappers.get(c.wrap)
                if w is None:
                    self.err(where, f"wrap '{c.wrap}' is not a wrapper "
                                    "(or none)")
                elif w.part not in (MAIN_FILE, c.part):
                    self.err(where, f"wrapper '{c.wrap}' is in {w.part}: a "
                                    f"part file uses those of {MAIN_FILE} "
                                    "and its own")
            elif not c.text.strip():
                self.err(where, "wrap none takes the whole file as text")
            if c.context is not None:
                self.context_ref(where, c.part, c.context, reg)
            if c.kind == "example" and c.wrap != "none" and "{line}" in c.text:
                self.err(where, "text holds the slot {line}")
            if c.kind == "example" and c.index > 0 and not c.doc.strip():
                self.err(where, "an example after the first of an entry has "
                                "a doc")


# Sub-table schemas, kept out of the top-level list.
SCHEMA["_case"] = CASE_FIELDS
SCHEMA["_invalid"] = INVALID_FIELDS
SCHEMA["_check"] = CHECK_FIELDS
SCHEMA["_untested"] = {"msg": ("str", R), "why": ("str", R)}
SCHEMA["_arg"] = ARG_FIELDS
SCHEMA["_refuse"] = {"key": ("str", R), "why": ("str", R)}
SCHEMA["_described"] = {"doc": ("str", R)}
SCHEMA["_param"] = {"name": ("str", R), "default": ("number|str", R),
                    "gt": ("number|str", O), "doc": ("str", O)}


def load(directory: pathlib.Path | str | None = None,
         parts: list[str] | None = None, resolve: bool = True,
         templates: bool = True) -> Registry:
    """Load the registry in `directory` (this script's by default): the
    main file and every part of meta.parts, or only the parts in `parts`.
    With resolve=False a reference that names no entry is a note, not an
    error (for a check that reads an incomplete registry). With
    templates=False a message template that no invalid case shows is not
    an error (for the tests of the other checks). Raises RegistryError
    listing every problem found."""
    d = pathlib.Path(directory) if directory is not None else SCRIPT_DIR
    return _Loader(d, parts, resolve, templates).run()


def summary(reg: Registry) -> str:
    cases = reg.cases()
    n_ex = sum(1 for c in cases if c.kind == "example")
    counts = [("blocks", len(reg.blocks)), ("keys", len(reg.keys)),
              ("line options", len(reg.line_options)),
              ("forms", len(reg.forms)), ("removed", len(reg.removed)),
              ("rules", len(reg.rules)), ("env", len(reg.envs)),
              ("loss functions", len(reg.loss_functions)),
              ("strategy types", len(reg.strategy_types)),
              ("init methods", len(reg.init_methods)),
              ("OPTIM targets", len(reg.optim_targets)),
              ("PARAM keys", len(reg.params)),
              ("excluded PARAM keys", len(reg.param_excluded)),
              ("PARAM policies", len(reg.param_policies))]
    text = ", ".join(f"{n} {what}" for what, n in counts)
    return (f"{', '.join(reg.files)}: {text}; {n_ex} examples, "
            f"{len(cases) - n_ex} invalid lines")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Check the target-file registry (hpotarget.toml and "
                    "its part files).")
    ap.add_argument("--check", action="store_true", required=True,
                    help="load the files and check them")
    ap.add_argument("--part", action="append", metavar="FILE",
                    help="check hpotarget.toml and this part file only "
                         "(repeatable); references into the other parts "
                         "are listed, not checked")
    ap.add_argument("--dir", type=pathlib.Path, default=SCRIPT_DIR,
                    help="the directory of the registry files "
                         "(default: this script's)")
    args = ap.parse_args(argv)
    try:
        reg = load(args.dir, args.part)
    except RegistryError as e:
        for line in e.errors:
            print(f"error: {line}", file=sys.stderr)
        print(f"hpotarget_registry.py: {len(e.errors)} error(s)",
              file=sys.stderr)
        return 1
    for note in reg.notes:
        print(f"note: {note}")
    print(f"hpotarget_registry.py: OK: {summary(reg)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
