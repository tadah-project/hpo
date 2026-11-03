#!/usr/bin/env python3
"""The corpus of target files: the target files of the Tadah! repositories,
the inputs of the frozen corpus (tests/data/hpotarget_corpus) that pins
what the parser of the target file gives for each.

    hpotarget_corpus.py list  [SOURCES]
    hpotarget_corpus.py cases OUT [SOURCES]
    hpotarget_corpus.py check --runner RUNNER [--keep DIR] [SOURCES]
    hpotarget_corpus.py write OUT --runner RUNNER [SOURCES]

SOURCES: --hpo DIR (the HPO source tree; default: the one this file is in),
--tadah-mlip DIR (the Tadah.MLIP source tree; without it, or empty, the
inputs of Tadah.MLIP are not read), --registry DIR (the registry; default:
<hpo>/blueprint).

`list` prints one line per input, `<id>  <origin>`, then each string of
the tests of Tadah.MLIP that is a template the corpus cannot fill
(`skipped`), and the number of inputs of each source. `cases` writes the
inputs as the cases of hpotarget_case_runner (tests/tools/
hpotarget_case_runner.cpp), mode "describe": the runner reads each with
the parser of `tadah hpo` and reports what the parse set
(tadah::hpo::testing::describe_parse) or the message that refused the
file. `check` runs the cases twice with the runner and exits 1 when a case gives neither a description nor a refusal, or both,
when the two runs differ, when a source that is read has no input, or when
a target file has no training configuration; it prints how many inputs
each source has and how many of them parse. `write` freezes the corpus
into the directory OUT (below).

The sources, and the id of each input
-------------------------------------
hpo-data    hpo-data:<path>: every *.hpo under HPO's tests/data (but the
            frozen corpus, tests/data/hpotarget_corpus), read in a
            copy of its directory tree, in its own directory, against the
            training configuration tests/test_ta_target_files.cpp reads it
            with (HPO_DATA_DIRS and HPO_DATA_FILES below: a <stem>.tadah
            beside the file, else the directory's configuration, with the
            changes the C++ test makes). A file under a directory that
            HPO_DATA_DIRS does not list has no configuration (an error).
registry    registry:<entry id>:<n>: each example of the target-file
            registry (blueprint/hpotarget*.toml), in its wrapper, with the
            training configuration of its context, the registry's fixture
            files and the environment the example sets (as
            tests/test_hpotarget_registry_conformance.cpp composes it).
mlip-tests  mlip-tests:<path>:<line>: each target file the tests of
            Tadah.MLIP write, read from their sources without running them:
            in a Python test a string made of literals, module-level string
            constants, `+` and `.format()` with literal arguments; in a C++
            test a run of adjacent string literals that ends an expression
            (followed by `;`, `)` or `,`). A string counts as a target file
            when it has two logical lines or more, each line starts with a
            key, block or removed keyword of the registry, a line that opens
            or closes a block holds its key alone, and a line at the top
            level is OPTIMIZER or a top-level key (so a refit file, whose
            OPTIMIZER block stands inside a stage block, and a message that
            starts with a key do not count). Each is read with the training
            configuration MLIP_TESTS_CONFIG (below) and the registry's
            fixture files.
docs        docs:<path>: each *.hpo of the documentation's files
            (docs/source/nfp/files): for each line of its manifest, the
            `tadah hpo` commands of the page's bash code blocks name the
            target file (--hpotarget) and the training configuration
            (-c, --config), read in a copy of the files in the line's
            directory; a target file no command names is read with the
            config.train of its directory (none: an error). Each *.hpo of
            docs/source/nfp_examples is read with the training
            configuration of the registry's default context.

The frozen corpus
-----------------
`write OUT` writes the frozen corpus that tests/test_hpotarget_corpus.cpp
reads (tests/data/hpotarget_corpus): every input with the files it reads,
and what the parser of `tadah hpo` gives for it, read by the runner (mode
"describe"). OUT holds:

corpus.toml   one [[input]] per input, in the order of the sources:
              id, origin, the files of the input and `expected`;
targets/      the target file of each input given as text,
              targets/<source>/<name>.hpo (`target`);
configs/      each training configuration given as text, once
              (`config`): a context of the registry under its name,
              MLIP_TESTS_CONFIG as mlip-tests.tadah;
fixtures/     the fixture files of the registry, copied into the
              directory of each input that writes them (`tree`; a
              second set of fixtures would be fixtures-2/);
trees/        each directory an input is read in, copied once,
              trees/<source>/<name> (`tree`), without the files
              TREE_EXCLUDE names (datasets, which no parse opens, and
              READMEs);
expected/     what the parser gives, expected/<source>/<name>.dump
              (the description of the parse) or <name>.error (the
              message that refuses the file), byte for byte.

<name> is the id of the input without its source, each run of characters
other than letters, digits, '.', '_' and '-' written as '_'. An input read
in place (`tree`) keeps target_file, config_file, cwd, copies,
config_drop and config_add; every input keeps env. Before it writes, the
command reads every input twice, from the sources and from the frozen
files, without the environment overrides of the registry (an input sets
its own, env), and stops when the two differ (the freezing left out a file the
parse reads) or when an input gives neither a description nor a refusal.
OUT is replaced whole; a directory that is not empty and holds no
corpus.toml is refused.

What it cannot see
------------------
- a target file a test builds from values known only when it runs (a
  Python template filled with a variable, a C++ string with a variable in
  it): `list` names the templates it skips; a C++ string whose last run of
  literals follows a variable is read from that run only;
- a test's target file with a key the registry does not know (written to
  be refused), and a target file of one line;
- target files written by a test of another repository (HPO's C++ tests
  build theirs in code), and the target-file fragments of the
  documentation's pages.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field

HERE = pathlib.Path(__file__).resolve().parent
DEFAULT_HPO = HERE.parent
SOURCES = ("hpo-data", "registry", "mlip-tests", "docs")

# The frozen corpus under HPO's tests/data (`write`), whose target files
# are copies of the inputs and not inputs themselves.
FROZEN_DIR = "hpotarget_corpus"
# The directories of HPO's tests/data that hold target files: the tree the
# case copies, the directory it works in and the training configuration of
# its files (paths relative to that directory), as
# tests/test_ta_target_files.cpp reads them.
HPO_DATA_DIRS = {
    "ta/target_files": {"tree": "ta", "cwd": "target_files",
                        "config": "../common/config.tadah"},
}
# The files read otherwise (test_ta_target_files.cpp, parse_case()).
HPO_DATA_FILES = {
    "ta/target_files/init_warm.hpo": {
        "copies": [{"from": "../common/config.tadah",
                    "to": "seed_pot.tadah"}]},
    "ta/target_files/init_chain.hpo": {
        "copies": [{"from": "../common/config.tadah",
                    "to": "seed_pot.tadah"}]},
    "ta/target_files/fixweight_no_dbfile.hpo": {
        "config_drop": ["DBFILE"],
        "config_add": {"ATOM": ["Ta"], "FIXINDEX": ["1-14"],
                       "FIXWEIGHT": ["0"] * 14}},
    "ta/target_files/dm_ream_embf0.hpo": {
        "config": "../ta2_refit_seed.tadah"},
}
PLACEHOLDER = re.compile(r"\{[A-Za-z_][A-Za-z_0-9]*\}")
# The training configuration of the target files of the tests of
# Tadah.MLIP, whose own configurations the tests build when they run: the
# model of the registry's default context with four functions per
# radial grid and the first coefficient pinned (FIXINDEX, FIXWEIGHT), so the
# OPTIM lines of those files on CGRID2B, SGRID2B (1-4) and FIXWEIGHT read.
MLIP_TESTS_CONFIG = """\
ATOM H
WATOM 1.0
DBFILE dummy.db
INIT2B true
INITMB false
TYPE2B D2_BP 4 4 H H
CGRID2B 1.0 1.5 2.0 2.5
SGRID2B 0.5 1.0 4.5 5.0
RCTYPE2B Cut_Cos
RCUT2B 5.0
VERBOSE 0
MODEL M_BLR BF_Linear
OALGO 4
LAMBDA 1e-3
FIXINDEX 1
FIXWEIGHT 0
"""


@dataclass
class Input:
    id: str
    source: str
    origin: str
    case: dict


@dataclass
class Corpus:
    inputs: list[Input] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    problems: list[str] = field(default_factory=list)
    read: list[str] = field(default_factory=list)   # the sources read

    def add(self, source: str, name: str, origin: str, case: dict) -> None:
        base = f"{source}:{name}"
        ids = {i.id for i in self.inputs}
        iid, n = base, 2
        while iid in ids:
            iid, n = f"{base}#{n}", n + 1
        self.inputs.append(Input(iid, source, origin, case))


# ------------------------------------------------------------ registry ----

def load_registry(registry: pathlib.Path):
    sys.path.insert(0, str(registry))
    import hpotarget_registry as hr  # noqa: E402
    return hr, hr.load(registry)


def default_context(reg) -> dict:
    c = reg.contexts["default"]
    return {"config": c.config, "fixtures": fixtures(reg)}


def fixtures(reg) -> list[dict]:
    return [{"path": f.path, "text": f.text} for f in reg.fixtures.values()]


def read_registry(corpus: Corpus, hr, reg) -> None:
    corpus.read.append("registry")
    for c in reg.cases():
        if c.kind != "example":
            continue
        comp = hr.compose(reg, c.wrap, c.text, c.context)
        ctx = reg.contexts[comp.context]
        case = {"file": comp.file, "config": ctx.config,
                "fixtures": fixtures(reg)}
        if c.env:
            case["env"] = dict(c.env)
        corpus.add("registry", f"{c.owner}:{c.index}",
                   f"{c.part} ({c.owner}, example {c.index + 1})", case)


# ------------------------------------------------------------ hpo-data ----

def read_hpo_data(corpus: Corpus, hpo: pathlib.Path) -> None:
    corpus.read.append("hpo-data")
    data = hpo / "tests" / "data"
    for f in sorted(data.rglob("*.hpo")):
        rel = f.relative_to(data).as_posix()
        if rel.split("/", 1)[0] == FROZEN_DIR:
            continue
        d = HPO_DATA_DIRS.get(f.parent.relative_to(data).as_posix())
        if d is None:
            corpus.problems.append(
                f"hpo-data: {rel} has no training configuration (its "
                f"directory is not in HPO_DATA_DIRS of {pathlib.Path(__file__).name})")
            continue
        rule = dict(HPO_DATA_FILES.get(rel, {}))
        config = rule.pop("config", None)
        if config is None:
            sibling = f.with_suffix(".tadah")
            config = sibling.name if sibling.exists() else d["config"]
        case = {"tree": str(data / d["tree"]), "cwd": d["cwd"],
                "target_file": f.name, "config_file": config, **rule}
        corpus.add("hpo-data", rel, f"HPO/tests/data/{rel}", case)


# ---------------------------------------------------------- mlip-tests ----

def looks_like_target(hr, reg, text: str) -> bool:
    """Whether `text` is a target file (the module docstring, mlip-tests)."""
    lines = hr.logical_lines(text)
    if len(lines) < 2:
        return False
    delimiters = ({b.name for b in reg.blocks} | {b.end for b in reg.blocks})
    known = ({k.name for k in reg.keys} | delimiters |
             {r.name for r in reg.removed})
    for b in reg.blocks:
        known |= set(b.aliases)
    for ln in lines:
        if ln.key not in known:
            return False
    root_keys = {k.name for k in reg.keys_in_block("ROOT")}
    top = False
    for ln, block in hr.line_blocks(reg, text):
        aliased = ln.key in reg.block(block).aliases if block != "ROOT" \
            else False
        if ln.key in delimiters and not aliased and len(ln.text.split()) != 1:
            return False
        if block == "ROOT" and (ln.key == "OPTIMIZER" or ln.key in root_keys):
            top = True
    return top


class PyFolder:
    """The strings of a Python module made of literals, module-level string
    constants, `+` and `.format()` with literal arguments."""

    def __init__(self, tree: ast.Module):
        self.consts: dict[str, ast.AST] = {}
        for node in tree.body:
            if (isinstance(node, ast.Assign) and len(node.targets) == 1
                    and isinstance(node.targets[0], ast.Name)):
                self.consts[node.targets[0].id] = node.value

    def fold(self, node: ast.AST, depth: int = 0) -> str | None:
        if depth > 50:
            return None
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            return node.value
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add):
            a = self.fold(node.left, depth + 1)
            b = self.fold(node.right, depth + 1)
            return a + b if a is not None and b is not None else None
        if isinstance(node, ast.Name) and node.id in self.consts:
            return self.fold(self.consts[node.id], depth + 1)
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "format"):
            base = self.fold(node.func.value, depth + 1)
            if base is None:
                return None
            args = [a.value for a in node.args
                    if isinstance(a, ast.Constant)]
            kwargs = {k.arg: k.value.value for k in node.keywords
                      if k.arg and isinstance(k.value, ast.Constant)}
            if (len(args) != len(node.args)
                    or len(kwargs) != len(node.keywords)):
                return None
            try:
                return base.format(*args, **kwargs)
            except (IndexError, KeyError, ValueError):
                return None
        return None

    @staticmethod
    def is_part(node: ast.AST, parent: ast.AST | None) -> bool:
        """Whether `node` is a part of a larger string expression."""
        if isinstance(parent, ast.BinOp) and isinstance(parent.op, ast.Add):
            return True
        return (isinstance(parent, ast.Attribute) and parent.attr == "format")


def python_strings(path: pathlib.Path) -> tuple[list[tuple[int, str]],
                                                  list[tuple[int, str]]]:
    """The strings of a Python source a test writes whole, and the
    templates it fills with values known when it runs: (line, text)."""
    tree = ast.parse(path.read_text(encoding="utf-8"), str(path))
    folder = PyFolder(tree)
    parents: dict[ast.AST, ast.AST] = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    # Module-level constants used as parts of other strings.
    parts = {n.id for n in ast.walk(tree)
             if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)
             and PyFolder.is_part(n, parents.get(n))}
    whole: list[tuple[int, str]] = []
    templates: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if not isinstance(node, (ast.Constant, ast.BinOp, ast.Call)):
            continue
        parent = parents.get(node)
        if PyFolder.is_part(node, parent):
            continue
        if (isinstance(parent, ast.Assign) and parent in tree.body
                and len(parent.targets) == 1
                and isinstance(parent.targets[0], ast.Name)
                and parent.targets[0].id in parts):
            continue
        text = folder.fold(node)
        if text is None:
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr == "format"):
                base = folder.fold(node.func.value)
                if base is not None:
                    templates.append((node.lineno, base))
            continue
        (templates if PLACEHOLDER.search(text) else whole).append(
            (node.lineno, text))
    return whole, templates


_CPP_ESCAPES = {"n": "\n", "t": "\t", "r": "\r", "0": "\0", "\\": "\\",
                '"': '"', "'": "'", "?": "?", "a": "\a", "b": "\b",
                "f": "\f", "v": "\v"}


def _cpp_unescape(body: str) -> str:
    out, i = [], 0
    while i < len(body):
        c = body[i]
        if c != "\\" or i + 1 >= len(body):
            out.append(c)
            i += 1
            continue
        e = body[i + 1]
        if e == "x":
            m = re.match(r"[0-9A-Fa-f]+", body[i + 2:])
            out.append(chr(int(m.group(0), 16)) if m else "x")
            i += 2 + (len(m.group(0)) if m else 0)
        elif e in "01234567" and re.match(r"[0-7]{1,3}", body[i + 1:]):
            m = re.match(r"[0-7]{1,3}", body[i + 1:])
            out.append(chr(int(m.group(0), 8)))
            i += 1 + len(m.group(0))
        else:
            out.append(_CPP_ESCAPES.get(e, e))
            i += 2
    return "".join(out)


def cpp_tokens(src: str) -> list[tuple[str, str, int]]:
    """(kind, value, line) of a C++ source: kind "str" for a string literal
    (its text), "chr" for a character literal, "id" for a word, "op" for
    any other character; comments and blanks are dropped."""
    toks: list[tuple[str, str, int]] = []
    i, n, line = 0, len(src), 1
    while i < n:
        c = src[i]
        if c == "\n":
            line += 1
            i += 1
        elif c.isspace():
            i += 1
        elif src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
        elif src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            line += src.count("\n", i, j)
            i = j
        elif c.isalpha() or c == "_":
            j = i
            while j < n and (src[j].isalnum() or src[j] == "_"):
                j += 1
            word = src[i:j]
            if j < n and src[j] == '"' and word in (
                    "R", "u8R", "uR", "UR", "LR"):
                m = re.match(r'"([^()\\\s]{0,16})\(', src[j:])
                if m:
                    end = src.find(")" + m.group(1) + '"', j + m.end())
                    end = n if end < 0 else end
                    toks.append(("str", src[j + m.end():end], line))
                    line += src.count("\n", i, end)
                    i = end + len(m.group(1)) + 2
                    continue
            if j < n and src[j] == '"' and word in ("u8", "u", "U", "L"):
                i = j
                continue
            toks.append(("id", word, line))
            i = j
        elif c == "'" and i > 0 and src[i - 1].isalnum():
            toks.append(("op", c, line))      # a digit separator (1'000)
            i += 1
        elif c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                j += 2 if src[j] == "\\" else 1
            body = src[i + 1:j]
            toks.append(("str" if c == '"' else "chr", _cpp_unescape(body),
                         line))
            line += body.count("\n")
            i = j + 1
        else:
            toks.append(("op", c, line))
            i += 1
    return toks


def cpp_strings(path: pathlib.Path) -> list[tuple[int, str]]:
    """The runs of adjacent string literals of a C++ source that end an
    expression (followed by `;`, `)` or `,`, not after `+`): (line, text)."""
    toks = cpp_tokens(path.read_text(encoding="utf-8", errors="replace"))
    out: list[tuple[int, str]] = []
    i = 0
    while i < len(toks):
        if toks[i][0] != "str":
            i += 1
            continue
        j = i
        while j < len(toks) and toks[j][0] == "str":
            j += 1
        prev = toks[i - 1] if i > 0 else ("op", ";", 0)
        nxt = toks[j] if j < len(toks) else ("op", ";", 0)
        if (nxt[0] == "op" and nxt[1] in ";)," and
                not (prev[0] == "op" and prev[1] == "+")):
            out.append((toks[i][2], "".join(t[1] for t in toks[i:j])))
        i = j
    return out


def read_mlip_tests(corpus: Corpus, hr, reg, tm: pathlib.Path) -> None:
    corpus.read.append("mlip-tests")
    tests = tm / "tests"
    ctx = {"config": MLIP_TESTS_CONFIG, "fixtures": fixtures(reg)}
    seen: set[str] = set()
    for f in sorted(tests.rglob("*")):
        if "__pycache__" in f.parts or not f.is_file():
            continue
        rel = f.relative_to(tm).as_posix()
        if f.suffix == ".py":
            whole, templates = python_strings(f)
        elif f.suffix in (".cpp", ".h", ".hpp"):
            whole, templates = cpp_strings(f), []
        else:
            continue
        for line, text in whole:
            if text in seen or not looks_like_target(hr, reg, text):
                continue
            seen.add(text)
            corpus.add("mlip-tests", f"{rel}:{line}",
                       f"Tadah.MLIP/{rel}, line {line}",
                       {"file": text, **ctx})
        for line, text in templates:
            if looks_like_target(hr, reg, PLACEHOLDER.sub("1", text)):
                corpus.skipped.append(
                    f"mlip-tests: Tadah.MLIP/{rel}, line {line}: a "
                    f"template filled when the test runs")


# ---------------------------------------------------------------- docs ----

def _doc_commands(tm: pathlib.Path):
    sys.path.insert(0, str(tm / "tests" / "doc_commands"))
    from doc_commands import bash_commands  # noqa: E402
    return bash_commands


def _hpo_command_files(argv: list[str]) -> tuple[str | None, str | None]:
    """The training configuration and the target file of a `tadah hpo`
    command (-c/--config, --hpotarget)."""
    config = target = None
    i = 2
    while i < len(argv):
        a = argv[i]
        for opt in ("--config", "--hpotarget"):
            if a.startswith(opt + "="):
                value = a.split("=", 1)[1]
                if opt == "--config":
                    config = value
                else:
                    target = value
        if a in ("-c", "--config") and i + 1 < len(argv):
            config = argv[i + 1]
            i += 1
        elif a == "--hpotarget" and i + 1 < len(argv):
            target = argv[i + 1]
            i += 1
        i += 1
    return config, target


def read_docs(corpus: Corpus, reg, tm: pathlib.Path) -> None:
    corpus.read.append("docs")
    source = tm / "docs" / "source"
    files = source / "nfp" / "files"
    named: set[pathlib.Path] = set()
    # A run and its check (`tadah hpo --check`) read one pair of files.
    pairs: set[tuple[str, str, str]] = set()
    manifest = files / "manifest.txt"
    if manifest.exists():
        bash_commands = _doc_commands(tm)
        for raw in manifest.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            fields = [x.strip() for x in line.split("|")]
            if len(fields) < 3:
                corpus.problems.append(f"docs: manifest line '{line}' has "
                                       f"no page or directory")
                continue
            page, run_dir = source / fields[1], fields[2]
            if not page.exists():
                continue
            for cmd in bash_commands(str(page)):
                if cmd.argv[:2] != ["tadah", "hpo"]:
                    continue
                config, target = _hpo_command_files(cmd.argv)
                if not target or not config or \
                        (run_dir, target, config) in pairs:
                    continue
                pairs.add((run_dir, target, config))
                tpath = (files / run_dir / target).resolve()
                named.add(tpath)
                name = tpath.relative_to(files.resolve()).as_posix() \
                    if tpath.is_relative_to(files.resolve()) else target
                corpus.add("docs", f"nfp/files/{name}",
                           f"{fields[1]}, line {cmd.line}",
                           {"tree": str(files), "cwd": run_dir,
                            "target_file": target, "config_file": config})
    if files.exists():
        for f in sorted(files.rglob("*.hpo")):
            if f.resolve() in named:
                continue
            rel = f.relative_to(files).as_posix()
            if not (f.parent / "config.train").exists():
                corpus.problems.append(
                    f"docs: nfp/files/{rel}: no `tadah hpo` command of a "
                    f"page of the manifest names it and its directory has "
                    f"no config.train")
                continue
            corpus.add("docs", f"nfp/files/{rel}",
                       f"docs/source/nfp/files/{rel}",
                       {"tree": str(files),
                        "cwd": f.parent.relative_to(files).as_posix(),
                        "target_file": f.name, "config_file": "config.train"})
    examples = source / "nfp_examples"
    if examples.exists():
        for f in sorted(examples.glob("*.hpo")):
            corpus.add("docs", f"nfp_examples/{f.name}",
                       f"docs/source/nfp_examples/{f.name}",
                       {"file": f.read_text(encoding="utf-8"),
                        **default_context(reg)})


# -------------------------------------------------------------- corpus ----

def collect(hpo: pathlib.Path, registry: pathlib.Path,
            tm: pathlib.Path | None) -> Corpus:
    hr, reg = load_registry(registry)
    corpus = Corpus()
    read_hpo_data(corpus, hpo)
    read_registry(corpus, hr, reg)
    if tm is not None:
        read_mlip_tests(corpus, hr, reg, tm)
        read_docs(corpus, reg, tm)
    return corpus


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


def toml_value(v) -> str:
    if isinstance(v, str):
        return toml_string(v)
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, list):
        return "[" + ", ".join(toml_value(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{ " + ", ".join(f"{toml_string(k)} = {toml_value(x)}"
                                for k, x in sorted(v.items())) + " }"
    raise TypeError(type(v))


def cases_text(corpus: Corpus) -> str:
    lines = []
    for i in corpus.inputs:
        lines.append("[[case]]")
        lines.append(f"id = {toml_string(i.id)}")
        lines.append('mode = "describe"')
        for k, v in i.case.items():
            lines.append(f"{k} = {toml_value(v)}")
        lines.append("")
    return "\n".join(lines)


def run_runner(runner: pathlib.Path, cases: pathlib.Path,
               results: pathlib.Path, env: dict[str, str] | None = None
               ) -> tuple[list[dict] | None, str]:
    """The results of the runner on `cases`, run in the environment `env`
    (None: this process's)."""
    proc = subprocess.run([str(runner), str(cases), str(results)],
                          capture_output=True, text=True, env=env)
    if proc.returncode != 0:
        return None, (proc.stdout + proc.stderr).strip() + \
            f"\n(the runner exited with {proc.returncode})"
    return json.loads(results.read_text(encoding="utf-8")), ""


def check(corpus: Corpus, runner: pathlib.Path,
          keep: pathlib.Path | None) -> list[str]:
    problems = list(corpus.problems)
    for s in corpus.read:
        if not any(i.source == s for i in corpus.inputs):
            problems.append(f"{s}: no input")
    with tempfile.TemporaryDirectory() as tmp:
        d = keep or pathlib.Path(tmp)
        d.mkdir(parents=True, exist_ok=True)
        cases = d / "cases.toml"
        cases.write_text(cases_text(corpus), encoding="utf-8")
        runs = []
        for k in (1, 2):
            res, err = run_runner(runner, cases,
                                  d / f"results{k}.json")
            if res is None:
                return problems + [err]
            runs.append({r["id"]: r for r in res})
    first, second = runs
    parsed = {s: 0 for s in corpus.read}
    for i in corpus.inputs:
        r = first.get(i.id)
        if r is None:
            problems.append(f"{i.id}: no result")
            continue
        if bool(r["parse_error"]) == bool(r["dump"]):
            problems.append(f"{i.id} ({i.origin}): the runner gave "
                            + ("both a description and a refusal"
                               if r["dump"] else
                               "neither a description nor a refusal"))
        if r != second.get(i.id):
            problems.append(f"{i.id} ({i.origin}): two runs differ")
        if r["dump"]:
            parsed[i.source] += 1
    for s in corpus.read:
        n = sum(1 for i in corpus.inputs if i.source == s)
        print(f"{s}: {n} inputs, {parsed[s]} parse, {n - parsed[s]} refused")
    return problems


# -------------------------------------------------------------- frozen ----

# The files of a tree an input is read in that the frozen corpus leaves
# out: the datasets, which the parse of a target file does not open, and
# the READMEs, which describe the tree, not the corpus.
TREE_EXCLUDE = ("*.tadahdb", "README.md")
NAME_RUN = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class Frozen:
    """The frozen corpus before it is written: the text of each file
    (corpus path -> text), each tree (corpus path -> source directory) and
    the [[input]] tables, without `expected`."""
    files: dict[str, str] = field(default_factory=dict)
    trees: dict[str, pathlib.Path] = field(default_factory=dict)
    inputs: list[dict] = field(default_factory=list)


def frozen_name(iid: str, source: str) -> str:
    rest = iid[len(source) + 1:]
    return NAME_RUN.sub("_", rest).strip("_") or "input"


def freeze(corpus: Corpus, reg) -> Frozen:
    """The frozen corpus of `corpus`'s inputs (the module docstring, "The
    frozen corpus"); the training configurations named after the context
    of the registry whose configuration they are."""
    fz = Frozen()
    config_names = {MLIP_TESTS_CONFIG: "mlip-tests"}
    for name, c in sorted(reg.contexts.items()):
        config_names.setdefault(c.config, name)
    configs: dict[str, str] = {}           # text -> corpus path
    fixture_sets: dict[tuple, str] = {}    # fixtures -> corpus directory
    trees: dict[str, str] = {}             # source directory -> corpus path
    names: set[str] = set()
    for i in corpus.inputs:
        case = i.case
        base = f"{i.source}/{frozen_name(i.id, i.source)}"
        name, n = base, 2
        while name in names:
            name, n = f"{base}-{n}", n + 1
        names.add(name)
        entry: dict = {"id": i.id, "origin": i.origin}
        if "tree" in case:
            src = pathlib.Path(case["tree"])
            tree = trees.get(str(src))
            if tree is None:
                stem = f"trees/{i.source}/{NAME_RUN.sub('_', src.name)}"
                tree, k = stem, 2
                while tree in fz.trees:
                    tree, k = f"{stem}-{k}", k + 1
                trees[str(src)] = tree
                fz.trees[tree] = src
            entry["tree"] = tree
            for k in ("cwd", "target_file", "config_file", "copies",
                      "config_drop", "config_add"):
                if k in case:
                    entry[k] = case[k]
        else:
            entry["target"] = f"targets/{name}.hpo"
            fz.files[entry["target"]] = case["file"]
            cpath = configs.get(case["config"])
            if cpath is None:
                stem = config_names.get(case["config"],
                                        f"config-{len(configs) + 1}")
                cpath = f"configs/{stem}.tadah"
                configs[case["config"]] = cpath
                fz.files[cpath] = case["config"]
            entry["config"] = cpath
            fixtures_ = tuple((f["path"], f["text"])
                              for f in case.get("fixtures", []))
            if fixtures_:
                d = fixture_sets.get(fixtures_)
                if d is None:
                    d = "fixtures" if not fixture_sets else \
                        f"fixtures-{len(fixture_sets) + 1}"
                    fixture_sets[fixtures_] = d
                    for path, text in fixtures_:
                        fz.files[f"{d}/{path}"] = text
                entry["tree"] = d
        if "env" in case:
            entry["env"] = case["env"]
        entry["name"] = name
        fz.inputs.append(entry)
    return fz


def stage(fz: Frozen, out: pathlib.Path) -> None:
    """Writes the files and the trees of `fz` under `out`."""
    for rel, text in fz.files.items():
        p = out / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(text.encode("utf-8"))
    for rel, src in fz.trees.items():
        shutil.copytree(src, out / rel,
                        ignore=shutil.ignore_patterns(*TREE_EXCLUDE))


def frozen_cases_text(fz: Frozen, root: pathlib.Path) -> str:
    """The cases of hpotarget_case_runner (mode describe) of the frozen
    inputs staged under `root`."""
    lines = []
    for e in fz.inputs:
        lines += ["[[case]]", f"id = {toml_string(e['id'])}",
                  'mode = "describe"']
        if "target" in e:
            lines.append(f"file = {toml_string(fz.files[e['target']])}")
            lines.append(f"config = {toml_string(fz.files[e['config']])}")
        if "tree" in e:
            lines.append(f"tree = {toml_string(str(root / e['tree']))}")
        for k in ("cwd", "target_file", "config_file", "copies",
                  "config_drop", "config_add", "env"):
            if k in e:
                lines.append(f"{k} = {toml_value(e[k])}")
        lines.append("")
    return "\n".join(lines)


CORPUS_HEADER = """\
# The frozen corpus of target files: each input of tests/hpotarget_corpus.py
# with the files it reads and what the parser of `tadah hpo` gives for it
# (expected/: <name>.dump the description of the parse, <name>.error the
# refusal). tests/test_hpotarget_corpus.cpp reads every input and compares.
# Written by `tests/hpotarget_corpus.py write` (its docstring describes the
# fields); a change of the parser that changes a result is a change of
# behaviour, and the corpus is written again only with it.
"""


def corpus_toml(fz: Frozen, expected: dict[str, str]) -> str:
    lines = [CORPUS_HEADER]
    for e in fz.inputs:
        lines.append("[[input]]")
        for k in ("id", "origin", "target", "config", "tree", "cwd",
                  "target_file", "config_file", "copies", "config_drop",
                  "config_add", "env"):
            if k in e:
                lines.append(f"{k} = {toml_value(e[k])}")
        lines.append(f"expected = {toml_string(expected[e['id']])}")
        lines.append("")
    return "\n".join(lines)


def write(corpus: Corpus, reg, runner: pathlib.Path,
          out: pathlib.Path) -> list[str]:
    """Writes the frozen corpus of `corpus` into `out` (the module
    docstring); returns the problems that stopped it, and writes nothing
    then."""
    problems = list(corpus.problems)
    for s in corpus.read:
        if not any(i.source == s for i in corpus.inputs):
            problems.append(f"{s}: no input")
    if out.exists() and any(out.iterdir()) and \
            not (out / "corpus.toml").exists():
        problems.append(f"{out}: a directory that is not empty and holds "
                        "no corpus.toml is not replaced")
    if problems:
        return problems
    fz = freeze(corpus, reg)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out.parent) as tmp:
        work = pathlib.Path(tmp)
        staged = work / "corpus"
        stage(fz, staged)
        (work / "live.toml").write_text(cases_text(corpus), encoding="utf-8")
        (work / "frozen.toml").write_text(frozen_cases_text(fz, staged),
                                          encoding="utf-8")
        # The environment overrides of the registry reach an input only
        # through its env.
        overrides = {e.name for e in reg.envs}
        env = {k: v for k, v in os.environ.items() if k not in overrides}
        live, err = run_runner(runner, work / "live.toml",
                               work / "live.json", env)
        if live is None:
            return [err]
        frozen, err = run_runner(runner, work / "frozen.toml",
                                 work / "frozen.json", env)
        if frozen is None:
            return [err]
        live_by_id = {r["id"]: r for r in live}
        frozen_by_id = {r["id"]: r for r in frozen}
        expected: dict[str, str] = {}
        for e, i in zip(fz.inputs, corpus.inputs):
            r = frozen_by_id.get(e["id"])
            if r is None:
                problems.append(f"{e['id']}: no result")
                continue
            if bool(r["parse_error"]) == bool(r["dump"]):
                problems.append(f"{e['id']} ({i.origin}): the runner gave "
                                + ("both a description and a refusal"
                                   if r["dump"] else
                                   "neither a description nor a refusal"))
                continue
            if r != live_by_id.get(e["id"]):
                problems.append(f"{e['id']} ({i.origin}): the frozen input "
                                "does not read as the input of its source "
                                "(a file the parse reads is not in the "
                                "corpus)")
                continue
            kind, text = (".dump", r["dump"]) if r["dump"] else \
                (".error", r["parse_error"])
            expected[e["id"]] = f"expected/{e['name']}{kind}"
            fz.files[expected[e["id"]]] = text
        if problems:
            return problems
        for rel in expected.values():
            p = staged / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(fz.files[rel].encode("utf-8"))
        (staged / "corpus.toml").write_text(corpus_toml(fz, expected),
                                            encoding="utf-8")
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(staged), str(out))
    return []


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("list", "cases", "check", "write"))
    ap.add_argument("out", nargs="?", type=pathlib.Path,
                    help="the cases file (cases), the corpus directory "
                         "(write)")
    ap.add_argument("--hpo", type=pathlib.Path, default=DEFAULT_HPO)
    ap.add_argument("--tadah-mlip", default="")
    ap.add_argument("--registry", type=pathlib.Path)
    ap.add_argument("--runner", type=pathlib.Path)
    ap.add_argument("--keep", type=pathlib.Path)
    a = ap.parse_args(argv)
    tm = pathlib.Path(a.tadah_mlip) if a.tadah_mlip else None
    registry = a.registry or a.hpo / "blueprint"
    corpus = collect(a.hpo, registry, tm)
    if a.command == "list":
        for i in corpus.inputs:
            print(f"{i.id}  {i.origin}")
        for s in corpus.skipped:
            print(f"skipped  {s}")
        for p in corpus.problems:
            print(f"problem  {p}")
        for s in corpus.read:
            print(f"{s}: {sum(1 for i in corpus.inputs if i.source == s)} "
                  f"inputs")
        return 1 if corpus.problems else 0
    if a.command == "cases":
        if a.out is None:
            ap.error("cases needs the file to write")
        a.out.write_text(cases_text(corpus), encoding="utf-8")
        for p in corpus.problems:
            print(f"problem  {p}")
        return 1 if corpus.problems else 0
    if a.runner is None:
        ap.error(f"{a.command} needs --runner")
    if a.command == "write":
        if a.out is None:
            ap.error("write needs the corpus directory")
        _, reg = load_registry(registry)
        problems = write(corpus, reg, a.runner, a.out)
        for p in problems:
            print(p)
        if problems:
            print(f"hpotarget_corpus.py: {len(problems)} problem(s); "
                  f"{a.out} is not written")
            return 1
        n = len(corpus.inputs)
        refused = sum(1 for f in a.out.glob("expected/*/*.error"))
        print(f"hpotarget_corpus.py: wrote {a.out}: {n} inputs, "
              f"{n - refused} read, {refused} refused")
        return 0
    problems = check(corpus, a.runner, a.keep)
    for p in problems:
        print(p)
    n = len(corpus.inputs)
    if problems:
        print(f"hpotarget_corpus.py: {len(problems)} problem(s) in {n} "
              f"inputs")
        return 1
    print(f"hpotarget_corpus.py: OK: {n} inputs")
    return 0


if __name__ == "__main__":
    sys.exit(main())
