#!/usr/bin/env python3
"""Generate the C++ tables of the target-file registry.

    gen_hpotarget.py --header [--registry DIR] [-o DIR]
    gen_hpotarget.py --check  [--registry DIR] [-o DIR]

Reads the target-file registry (hpotarget.toml and the part files of its
meta.parts) with its loader, hpotarget_registry.py, and writes two headers
into include/tadah/hpo/ of this repository (or the directory -o names):

    hpotarget_registry_gen.h   the tables of the target-file registry
    hpo_param_registry_gen.h   its PARAM keys

blueprint/README.md, "The generated tables", lists what the headers hold,
what stays out of them and how the ids are named.

The output depends on the registry, on its loader (the headers also hold
its placeholder sets, the words and the field order of its schema and its
named integer limits) and on this generator, and on nothing else (no
date, no path), so two runs write the same files. The headers are checked
in, and regenerated after a change to any of the three; CTest
hpo_hpotarget_registry_sync runs --check, which fails when a header is
missing or differs from what the registry and its loader give, and the
build target hpo_regenerate_hpotarget_registry runs --header.

Exit status: 0; 1 with the reason (a registry the loader refuses, a loader
of another API, a name the generator cannot give, a header that is missing
or differs with --check); 2 for a usage error.
"""

from __future__ import annotations

import argparse
import math
import pathlib
import re
import shlex
import sys
from typing import Any, Iterable

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

import hpotarget_registry as hr  # noqa: E402

# The loader API the generator is written for: a loader of another API has
# another schema, which the generator must follow first.
LOADER_API = 1

# The registry and the header directory of this repository: the build
# target REGENERATE runs --header with these two.
DEFAULT_REGISTRY = SCRIPT_DIR
DEFAULT_OUT = SCRIPT_DIR.parent / "include" / "tadah" / "hpo"
REGISTRY_HEADER = "hpotarget_registry_gen.h"
PARAM_HEADER = "hpo_param_registry_gen.h"
REGENERATE_TARGET = "hpo_regenerate_hpotarget_registry"
REGENERATE = ("cmake --build <build directory> --target "
              f"{REGENERATE_TARGET}")

# The statuses of an OPTIM target the loader accepts.
OPTIM_TARGET_STATUS = {"refused": "Refused", "special": "Special"}
# Fields of a section that are documentation or cases, never data.
DOC_FIELDS = {"doc", "long_doc", "examples", "invalid", "untested",
              "synopsis"}
IDENT = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")


class GenerationError(Exception):
    """The registry gives no headers: the reason."""


# ------------------------------------------------------------ helpers ----

def cxx_name(ident: str) -> str:
    """The C++ name of a registry id: '.' and ':' become '_', '-' is left
    out. Refuses an id that gives no identifier, or one with '__' (a name
    the C++ implementation reserves)."""
    name = ident.replace(".", "_").replace(":", "_").replace("-", "")
    if not IDENT.match(name) or "__" in name:
        raise GenerationError(f"the id {ident!r} gives no C++ name "
                              f"({name!r} is not an identifier of letters, "
                              "digits and single underscores)")
    return name


def cxx_names(idents: Iterable[str], what: str) -> dict[str, str]:
    """The C++ name of each id; two ids of one name stop the generator."""
    out: dict[str, str] = {}
    seen: dict[str, str] = {}
    for ident in idents:
        name = cxx_name(ident)
        if name in seen and seen[name] != ident:
            raise GenerationError(f"{what}: the ids {seen[name]!r} and "
                                  f"{ident!r} both give the C++ name {name}")
        seen[name] = ident
        out[ident] = name
    return out


def camel(word: str) -> str:
    """An enumerator of a schema word: last_wins -> LastWins."""
    return "".join(p[:1].upper() + p[1:] for p in word.split("_"))


def cstr(text: str) -> str:
    """A C++ string literal of `text`: a backslash, a double quote and a
    control character are escaped (octal, which takes at most three digits,
    so no following character joins it); other characters, UTF-8 included,
    are written as they are."""
    out = ['"']
    for ch in text:
        if ch == "\\":
            out.append("\\\\")
        elif ch == '"':
            out.append('\\"')
        elif ch == "\n":
            out.append("\\n")
        elif ch == "\t":
            out.append("\\t")
        elif ord(ch) < 0x20 or ord(ch) == 0x7F:
            out.append(f"\\{ord(ch):03o}")
        else:
            out.append(ch)
    out.append('"')
    return "".join(out)


def fmt_double(value: float) -> str:
    """A C++ double literal that reads back as `value`."""
    if not math.isfinite(value):
        raise GenerationError(f"the number {value!r} is not finite")
    text = repr(float(value))
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def lit(value: Any) -> str:
    """A registry value as text: a number as the registry writes it."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return repr(value)
    if isinstance(value, str):
        return value
    raise GenerationError(f"the value {value!r} is neither a number nor a "
                          "text")


def bool_lit(value: bool) -> str:
    return "true" if value else "false"


def sorted_set(values: Iterable[str]) -> list[str]:
    return sorted(set(values))


def split_fields(row: str) -> tuple[str, list[str], str]:
    """`Type{a, b, c}` as ("Type{", ["a", "b", "c"], "}"): the fields of a
    braced row, split at the commas outside string literals and inner
    braces."""
    brace, quote = row.find("{"), row.find('"')
    if brace < 0 or (0 <= quote < brace) or not row.endswith("}"):
        return row, [], ""
    start = brace + 1
    parts, depth, quote, cur = [], 0, False, ""
    i = start
    while i < len(row) - 1:
        ch = row[i]
        cur += ch
        if quote:
            if ch == "\\":
                cur += row[i + 1]
                i += 1
            elif ch == '"':
                quote = False
        elif ch == '"':
            quote = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        elif ch == "," and depth == 0:
            parts.append(cur[:-1].strip())
            cur = ""
        i += 1
    if cur.strip():
        parts.append(cur.strip())
    return row[:start], parts, "}"


def wrap_row(row: str, indent: str, width: int = 79) -> list[str]:
    """The lines of a table row: its fields packed into lines of at most
    `width` characters, a longer field (a message) on a line of its own;
    the continuation lines are indented by four more spaces."""
    if len(indent) + len(row) + 1 <= width:
        return [f"{indent}{row},"]
    head, parts, tail = split_fields(row)
    if not parts:
        return [f"{indent}{row},"]
    lines, cur, cont = [], indent + head, indent + "    "
    for j, part in enumerate(parts):
        text = part + ("," if j + 1 < len(parts) else tail + ",")
        sep = "" if cur.endswith("{") else " "
        if cur.endswith("{") or len(cur) + len(sep) + len(text) <= width:
            cur += sep + text
        else:
            lines.append(cur)
            cur = cont + text
    lines.append(cur)
    return lines


class Writer:
    """The lines of a header, and the arrays its tables point to."""

    def __init__(self) -> None:
        self.lines: list[str] = []
        self.detail: list[str] = []
        self.names: set[str] = set()

    def out(self, line: str = "") -> None:
        self.lines.append(line)

    def array(self, name: str, ctype: str, items: list[str]) -> str:
        """A Span over the array `name` of `items` (in namespace detail),
        or an empty Span for no item."""
        if not items:
            return f"Span<{ctype}>{{nullptr, 0}}"
        if name in self.names:
            raise GenerationError(f"two arrays are named {name}")
        self.names.add(name)
        self.detail.append(f"inline constexpr {ctype} {name}[] = {{")
        for item in items:
            self.detail.extend(wrap_row(item, "    "))
        self.detail.append("};")
        return f"Span<{ctype}>{{detail::{name}, {len(items)}}}"

    def strings(self, name: str, items: list[str]) -> str:
        return self.array(name, "std::string_view", [cstr(i) for i in items])


# ------------------------------------------------------------ registry ----

class Model:
    """The registry as the headers need it: the C++ name of every entry
    and the derived tables."""

    def __init__(self, reg: hr.Registry):
        if reg.meta.api != LOADER_API or hr.API != LOADER_API:
            raise GenerationError(
                f"the registry has API {reg.meta.api} and the loader "
                f"{hr.API}; the generator reads API {LOADER_API}")
        self.reg = reg
        ids = [e.id for e in reg.entries()]
        self.name = cxx_names(ids, "the registry")
        self.blocks = ["ROOT"] + [b.name for b in reg.blocks]
        cxx_names(self.blocks, "the blocks")
        self.key_ids = [k.id for k in reg.keys]
        for t in reg.optim_targets:
            if t.status not in OPTIM_TARGET_STATUS:
                raise GenerationError(f"{t.id}: status {t.status!r} is not "
                                      f"one of {sorted(OPTIM_TARGET_STATUS)}")

    def key_blocks(self, k: hr.Key) -> list[str]:
        g = self.reg.groups.get(k.group)
        return list(g.blocks) if g is not None else list(k.blocks)

    def block_mask(self, places: Iterable[str]) -> str:
        bits = [f"block_bit(BlockId::{p})" for p in self.blocks if p in
                set(places)]
        return " | ".join(bits) if bits else "0u"

    def key_ref(self, ident: str) -> str:
        if ident not in self.key_ids:
            raise GenerationError(f"{ident!r} is no key of the registry")
        return f"KeyId::{self.name[ident]}"

    def block_keys(self, b: hr.Block) -> list[str]:
        """The ids of the keys the block takes: those valid in it, less the
        keys it refuses, in registry order."""
        refused = {r.key for r in b.refuse}
        return [k.id for k in self.reg.keys_in_block(b.name)
                if k.name not in refused]

    def top_level_keys(self) -> list[str]:
        """The names of the top-level keys a block refuses as such: the
        keys valid at the top level, less the names of the keys of a
        group (MAXTIME at the top level is PC_SCRIPT_TIMEOUT; inside a
        block it is the optimiser's)."""
        group = {k.name for k in self.reg.keys if k.group}
        out: list[str] = []
        for k in self.reg.keys_in_block("ROOT"):
            if k.name not in group and k.name not in out:
                out.append(k.name)
        return out


def limit(value: Any) -> str:
    """A Limit: the registry's text, its value and, for a whole number of
    0 or more, that number exactly."""
    if value is None:
        return "Limit{}"
    if isinstance(value, str):
        if value not in hr.INT_LIMITS:
            raise GenerationError(f"the limit {value!r} is not one of "
                                  f"{sorted(hr.INT_LIMITS)}")
        whole = hr.INT_LIMITS[value]
        return (f"Limit{{true, {cstr(value)}, {fmt_double(float(whole))}, "
                f"true, {whole}ull}}")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise GenerationError(f"the limit {value!r} is not a number")
    whole = (isinstance(value, int) or float(value).is_integer()) and \
        0 <= value <= 2**64 - 1
    return (f"Limit{{true, {cstr(lit(value))}, {fmt_double(float(value))}, "
            f"{bool_lit(whole)}, {int(value) if whole else 0}ull}}")


def args_span(w: Writer, prefix: str, args: list[hr.Arg]) -> str:
    rows = []
    for i, a in enumerate(args):
        if a.kind not in hr.ARG_KINDS:
            raise GenerationError(f"{prefix}: argument kind {a.kind!r}")
        words = w.strings(f"{prefix}_arg{i}_words", a.words)
        refused = w.array(f"{prefix}_arg{i}_refused", "RefusedWord",
                          [f"RefusedWord{{{cstr(k)}, {cstr(v)}}}"
                           for k, v in a.refused_words.items()])
        default = "" if a.default is None else lit(a.default)
        rows.append(
            f"Arg{{{cstr(a.name)}, ArgKind::{camel(a.kind)}, "
            f"{bool_lit(a.optional)}, {bool_lit(a.repeat)}, "
            f"{limit(a.min)}, {limit(a.max)}, {bool_lit(a.min_open)}, "
            f"{bool_lit(a.integral)}, {words}, {refused}, {cstr(a.what)}, "
            f"{cstr(a.msg)}, {cstr(a.noun)}, "
            f"{bool_lit(a.default is not None)}, "
            f"{cstr(default)}}}")
    return w.array(f"{prefix}_args", "Arg", rows)


def checks_span(w: Writer, name: str, checks: list[hr.Check]) -> str:
    return w.strings(name, [c.msg for c in checks])


def default_literal(d: hr.Default | None) -> tuple[str, str]:
    if d is None or d.literal is None:
        return "false", cstr("")
    return "true", cstr(d.literal)


def enum(w: Writer, name: str, base: str, members: list[str],
         comment: str) -> None:
    w.out(f"/* {comment} */")
    w.out(f"enum class {name} : {base} {{")
    for m in members:
        w.out(f"  {m},")
    w.out("};")
    w.out()


def table(w: Writer, ctype: str, name: str, rows: list[str]) -> None:
    """The table `name` of `rows`: an array, or an empty Span for no row
    (C++ has no array of no element). Both are ranges."""
    if not rows:
        w.out(f"inline constexpr Span<{ctype}> {name}{{nullptr, 0}};")
        w.out()
        return
    w.out(f"inline constexpr {ctype} {name}[] = {{")
    for r in rows:
        for line in wrap_row(r, "  "):
            w.out(line)
    w.out("};")
    w.out()


def section_fields(w: Writer, sec: hr.Section, schema: str,
                   skip: Iterable[str] = ()) -> None:
    """The data fields of a section, in the order of the loader's schema:
    a text as a std::string_view, a list of texts as a Span."""
    for field in hr.SCHEMA[schema]:
        if field in DOC_FIELDS or field in skip or field not in sec.fields:
            continue
        value = sec.fields[field]
        if isinstance(value, str):
            w.out(f"inline constexpr std::string_view {field} = "
                  f"{cstr(value)};")
        elif isinstance(value, list) and all(isinstance(v, str)
                                             for v in value):
            span = w.strings(f"{schema}_{field}", value)
            w.out(f"inline constexpr Span<std::string_view> {field} = "
                  f"{span};")
        else:
            raise GenerationError(f"[{schema}] {field}: a field of type "
                                  f"{type(value).__name__}, which the "
                                  "generator does not write")


# ------------------------------------------------------------- headers ----

PRELUDE = """\
/* A view of the rows of a table (C++17 has no std::span). */
template <class T>
struct Span {
  const T* data;
  std::size_t size;
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
  constexpr const T& operator[](std::size_t i) const { return data[i]; }
  constexpr bool empty() const { return size == 0; }
};

/* A limit of the range of an argument: the registry's text ("0", "0.5",
 * "INT_MAX"), its value, and, for a whole number of 0 or more (INT_MAX,
 * UINT_MAX and UINT64_MAX are), that number exactly. */
struct Limit {
  bool set;
  std::string_view text;
  double value;
  bool whole;
  std::uint64_t whole_value;
};

/* A word an argument refuses, with its message. */
struct RefusedWord {
  std::string_view word;
  std::string_view msg;
};
"""

FILL = """\
/* A value of a placeholder: fill() puts `text` where a template writes
 * {name}. */
struct Value {
  std::string_view name;
  std::string_view text;
};

/* The message of the template `tmpl`: each placeholder {name} (a letter or
 * '_', then letters, digits and '_') replaced by the text of the value of
 * that name; ${name}, a LAMMPS variable reference, and every other brace
 * are text. Throws std::invalid_argument for a placeholder without a
 * value. */
inline std::string fill(std::string_view tmpl,
                        std::initializer_list<Value> values)
{
  const auto head = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
  };
  const auto tail = [&](char c) { return head(c) || (c >= '0' && c <= '9'); };
  std::string out;
  std::size_t i = 0;
  while (i < tmpl.size()) {
    if (tmpl[i] == '{' && (i == 0 || tmpl[i - 1] != '$') &&
        i + 1 < tmpl.size() && head(tmpl[i + 1])) {
      std::size_t j = i + 2;
      while (j < tmpl.size() && tail(tmpl[j])) ++j;
      if (j < tmpl.size() && tmpl[j] == '}') {
        const std::string_view name = tmpl.substr(i + 1, j - i - 1);
        const Value* found = nullptr;
        for (const Value& v : values)
          if (v.name == name) { found = &v; break; }
        if (!found)
          throw std::invalid_argument(
              "fill: no value for the placeholder {" + std::string(name) +
              "} of \\"" + std::string(tmpl) + "\\"");
        out += found->text;
        i = j + 1;
        continue;
      }
    }
    out += tmpl[i];
    ++i;
  }
  return out;
}
"""


def header_top(w: Writer, sources: list[str],
               description: list[str]) -> None:
    w.out("// GENERATED FILE -- DO NOT EDIT.")
    w.out("//")
    w.out(f"// Source of truth: {sources[0]}")
    for line in sources[1:]:
        w.out(f"//                  {line}")
    w.out("// Generator:       HPO/blueprint/gen_hpotarget.py --header")
    w.out(f"//                  (build target {REGENERATE_TARGET})")
    w.out("// Drift gate:      ctest hpo_hpotarget_registry_sync "
          "(gen_hpotarget.py --check)")
    w.out("//")
    for line in description:
        w.out(f"// {line}" if line else "//")
    w.out("#pragma once")
    w.out()


def finish(w: Writer, body_at: int) -> str:
    """The header text: the arrays (namespace detail) before the tables
    that point to them."""
    detail = (["namespace detail {", ""] + w.detail +
              ["", "}  // namespace detail", ""])
    lines = w.lines[:body_at] + detail + w.lines[body_at:]
    return "\n".join(lines) + "\n"


def emit_registry(reg: hr.Registry) -> str:
    m = Model(reg)
    w = Writer()
    header_top(w, [
        "HPO/blueprint/hpotarget.toml and the part files of its",
        "meta.parts, read by HPO/blueprint/hpotarget_registry.py"], [
        "The tables of the target-file registry (blueprint/README.md, \"The",
        "generated tables\"), in the order of the registry (the placeholder",
        "sets sorted). Messages are templates: fill() puts the values of "
        "their",
        "{name} placeholders in."])
    for inc in ("cstddef", "cstdint", "initializer_list", "stdexcept",
                "string", "string_view"):
        w.out(f"#include <{inc}>")
    w.out()
    w.out("namespace tadah {")
    w.out("namespace hpo {")
    w.out("namespace hpotarget_registry {")
    w.out()
    w.out(PRELUDE)
    w.out(FILL)

    # ---- enums -----------------------------------------------------------
    enum(w, "BlockId", "unsigned char", m.blocks,
         "ROOT, the top level, and the blocks, in registry order.")
    w.out("constexpr unsigned block_bit(BlockId b)")
    w.out("{")
    w.out("  return 1u << static_cast<unsigned>(b);")
    w.out("}")
    w.out()
    enum(w, "ArgKind", "unsigned char",
         [camel(k) for k in sorted(hr.ARG_KINDS)],
         "The reader of an argument (blueprint/README.md, `args`).")
    enum(w, "Repeat", "unsigned char",
         [camel(k) for k in sorted(hr.KEY_REPEAT)],
         "What a key or an option given again does.")
    enum(w, "BlockRepeat", "unsigned char",
         [camel(k) for k in sorted(hr.BLOCK_REPEAT)],
         "What a block opened again does.")
    enum(w, "CheckStage", "unsigned char",
         [camel(k) for k in hr.CHECK_STAGES],
         "A stage of the order in which a line is tried "
         "(structure::check_order).")
    enum(w, "RuleCheck", "unsigned char",
         [camel(k) for k in sorted(hr.CHECKS)],
         "When a rule is checked: as the file is read, or at startup.")
    enum(w, "OptimTargetStatus", "unsigned char",
         list(OPTIM_TARGET_STATUS.values()),
         "An OPTIM target the parser refuses, or one it reads with a rule "
         "of its own.")
    enum(w, "KeyId", "unsigned short", [m.name[k.id] for k in reg.keys],
         "The keys, in registry order: kKeys[i].id is KeyId(i).")
    enum(w, "LineOptionId", "unsigned short",
         [m.name[o.id] for o in reg.line_options],
         "The options of the lines, in registry order.")
    enum(w, "FormId", "unsigned char", [m.name[f.id] for f in reg.forms],
         "The forms of the lines, in registry order.")

    # ---- structs ---------------------------------------------------------
    w.out("""\
struct Arg {
  std::string_view name;
  ArgKind kind;
  bool optional;
  bool repeat;
  Limit min;
  Limit max;
  bool min_open;     // min itself is refused
  bool integral;     // a number with no fractional part
  Span<std::string_view> words;
  Span<RefusedWord> refused_words;
  std::string_view what;   // the {what} of the reader's message
  std::string_view msg;    // the key's own message for a refused value
  std::string_view noun;   // the {noun} of positive_msg
  bool has_default;
  std::string_view default_text;
};

/* A word a block reads as another key (INIT's STRATEGY is METHOD). */
struct Alias {
  std::string_view word;
  KeyId key;
};

/* A key of the group a block refuses, and why. */
struct Refusal {
  std::string_view key;
  std::string_view why;
};

struct Block {
  BlockId id;
  std::string_view name;
  std::string_view end;
  unsigned parents;            // block_bit() of each place it opens in
  BlockRepeat repeat;
  std::string_view repeat_msg;
  std::string_view misplaced_msg;
  std::string_view unknown_msg;
  bool unknown_first;
  Span<Alias> aliases;
  Span<Refusal> refuse;
  std::string_view refuse_msg;
  Span<std::string_view> unreachable_msgs;
  Span<std::string_view> requires_checks;   // at the opening line
  Span<std::string_view> close_checks;
  Span<KeyId> keys;            // the keys it takes: valid in it, not refused
};

struct Group {
  std::string_view name;
  unsigned blocks;
  std::string_view leftover_msg;
};

struct Key {
  KeyId id;
  std::string_view id_text;
  std::string_view name;
  std::string_view group;      // empty: valid where `blocks` says
  unsigned blocks;             // block_bit() of each place it is valid in
  Span<Arg> args;
  Repeat repeat;
  std::string_view repeat_msg;
  std::string_view misplaced_msg;
  bool has_default_literal;
  std::string_view default_literal;   // the value as a file writes it
  bool has_alias;
  KeyId alias_of;
  bool refused;                // read only to be refused
  std::string_view env;        // the environment variable that overrides it
  Span<std::string_view> checks;
};

struct LineOption {
  LineOptionId id;
  std::string_view id_text;
  std::string_view option;
  Span<std::string_view> lines;   // the keys and forms whose lines take it
  Span<Arg> args;
  Repeat repeat;
  std::string_view repeat_msg;
  bool has_default_literal;
  std::string_view default_literal;
  bool refused;
  Span<std::string_view> checks;
};

struct Form {
  FormId id;
  std::string_view id_text;
  std::string_view line;
  std::string_view word;       // empty: the form without a word
  Span<Arg> args;
  bool refused;
  Span<std::string_view> checks;
};

struct Removed {
  std::string_view name;
  std::string_view hint;
};

struct Rule {
  std::string_view id_text;
  RuleCheck check;
  std::string_view msg;
};

struct Env {
  std::string_view name;
  KeyId key;
  Span<std::string_view> values;
  std::string_view msg;
  Span<std::string_view> checks;
};

struct LossParam {
  std::string_view name;
  double default_value;
  bool has_gt;
  double gt;                   // a value must be above it
};

struct LossFunction {
  std::string_view name;
  Span<LossParam> params;
};

struct StrategyType {
  std::string_view name;
  Span<KeyId> keys;
  Span<std::string_view> checks;
};

struct InitMethod {
  std::string_view name;
  Span<Arg> args;
  bool refused;
  Span<std::string_view> checks;
};

struct OptimTarget {
  std::string_view key;
  OptimTargetStatus status;
  Span<int> indices;
  std::string_view msg;
};
""")
    body_at = len(w.lines)

    # ---- sections --------------------------------------------------------
    w.out("namespace lexing {")
    section_fields(w, reg.lexing, "lexing")
    w.out("}  // namespace lexing")
    w.out()
    w.out("namespace reading {")
    section_fields(w, reg.reading, "reading")
    w.out("}  // namespace reading")
    w.out()
    w.out("namespace structure {")
    stages = [f"CheckStage::{camel(s)}" for s in
              reg.structure.fields.get("check_order", [])]
    w.out("inline constexpr Span<CheckStage> check_order = "
          f"{w.array('structure_check_order', 'CheckStage', stages)};")
    section_fields(w, reg.structure, "structure", skip=("check_order",))
    w.out("}  // namespace structure")
    w.out()

    # ---- placeholders ----------------------------------------------------
    w.out("/* The placeholders a message may hold: those of the messages")
    w.out(" * whose set the loader fixes (FIXED_PLACEHOLDERS), of the")
    w.out(" * readers' messages, and of the messages of the entries. */")
    w.out("namespace placeholders {")
    for (kind, field), names in sorted(hr.FIXED_PLACEHOLDERS.items()):
        span = w.strings(f"placeholders_{kind}_{field}", sorted_set(names))
        w.out(f"inline constexpr Span<std::string_view> {kind}_{field} = "
              f"{span};")
    span = w.strings("placeholders_reading",
                     sorted_set(hr.READING_PLACEHOLDERS))
    w.out(f"inline constexpr Span<std::string_view> reading = {span};")
    span = w.strings("placeholders_entry", sorted_set(hr.PLACEHOLDERS))
    w.out(f"inline constexpr Span<std::string_view> entry = {span};")
    w.out("}  // namespace placeholders")
    w.out()

    # ---- groups and blocks -----------------------------------------------
    table(w, "Group", "kGroups",
          [f"Group{{{cstr(g.name)}, {m.block_mask(g.blocks)}, "
           f"{cstr(g.leftover_msg)}}}" for g in reg.groups.values()])
    rows = []
    for b in reg.blocks:
        n = m.name[b.id]
        aliases = w.array(f"{n}_aliases", "Alias",
                          [f"Alias{{{cstr(word)}, {m.key_ref(ref)}}}"
                           for word, ref in b.aliases.items()])
        refuse = w.array(f"{n}_refuse", "Refusal",
                         [f"Refusal{{{cstr(r.key)}, {cstr(r.why)}}}"
                          for r in b.refuse])
        keys = w.array(f"{n}_keys", "KeyId",
                       [m.key_ref(i) for i in m.block_keys(b)])
        rows.append(f"Block{{BlockId::{b.name}, {cstr(b.name)}, "
                    f"{cstr(b.end)}, {m.block_mask(b.parents)}, "
                    f"BlockRepeat::{camel(b.repeat)}, {cstr(b.repeat_msg)}, "
                    f"{cstr(b.misplaced_msg)}, {cstr(b.unknown_msg)}, "
                    f"{bool_lit(b.unknown_first)}, {aliases}, {refuse}, "
                    f"{cstr(b.refuse_msg)}, "
                    f"{w.strings(f'{n}_unreachable', b.unreachable_msgs)}, "
                    f"{checks_span(w, f'{n}_requires', b.requires)}, "
                    f"{checks_span(w, f'{n}_close_checks', b.close_checks)}, "
                    f"{keys}}}")
    table(w, "Block", "kBlocks", rows)

    # ---- keys ------------------------------------------------------------
    rows = []
    for k in reg.keys:
        n = m.name[k.id]
        has_def, def_lit = default_literal(k.default)
        alias = m.key_ref(k.alias_of) if k.alias_of else f"KeyId::{n}"
        rows.append(f"Key{{KeyId::{n}, {cstr(k.id)}, {cstr(k.name)}, "
                    f"{cstr(k.group)}, {m.block_mask(m.key_blocks(k))}, "
                    f"{args_span(w, n, k.args)}, Repeat::{camel(k.repeat)}, "
                    f"{cstr(k.repeat_msg)}, {cstr(k.misplaced_msg)}, "
                    f"{has_def}, {def_lit}, {bool_lit(bool(k.alias_of))}, "
                    f"{alias}, {bool_lit(bool(k.refused))}, {cstr(k.env)}, "
                    f"{checks_span(w, f'{n}_checks', k.checks)}}}")
    table(w, "Key", "kKeys", rows)
    w.out("/* The top-level keys a block refuses as such: the keys valid at "
          "the top")
    w.out(" * level, less the names of the keys of a group. */")
    span = w.strings("top_level_keys", m.top_level_keys())
    w.out(f"inline constexpr Span<std::string_view> kTopLevelKeys = {span};")
    w.out()

    # ---- options and forms -----------------------------------------------
    rows = []
    for o in reg.line_options:
        n = m.name[o.id]
        has_def, def_lit = default_literal(o.default)
        rows.append(f"LineOption{{LineOptionId::{n}, {cstr(o.id)}, "
                    f"{cstr(o.option)}, {w.strings(f'{n}_lines', o.lines)}, "
                    f"{args_span(w, n, o.args)}, Repeat::{camel(o.repeat)}, "
                    f"{cstr(o.repeat_msg)}, {has_def}, {def_lit}, "
                    f"{bool_lit(bool(o.refused))}, "
                    f"{checks_span(w, f'{n}_checks', o.checks)}}}")
    table(w, "LineOption", "kLineOptions", rows)
    table(w, "Form", "kForms",
          [f"Form{{FormId::{m.name[f.id]}, {cstr(f.id)}, {cstr(f.line)}, "
           f"{cstr(f.word)}, {args_span(w, m.name[f.id], f.args)}, "
           f"{bool_lit(bool(f.refused))}, "
           f"{checks_span(w, m.name[f.id] + '_checks', f.checks)}}}"
           for f in reg.forms])

    # ---- the other tables ------------------------------------------------
    table(w, "Removed", "kRemoved",
          [f"Removed{{{cstr(r.name)}, {cstr(r.hint)}}}" for r in reg.removed])
    table(w, "Rule", "kRules",
          [f"Rule{{{cstr(r.id)}, RuleCheck::{camel(r.check)}, "
           f"{cstr(r.msg)}}}" for r in reg.rules])
    table(w, "Env", "kEnvs",
          [f"Env{{{cstr(e.name)}, {m.key_ref(e.key)}, "
           f"{w.strings(m.name[e.id] + '_values', e.values)}, "
           f"{cstr(e.msg)}, "
           f"{checks_span(w, m.name[e.id] + '_checks', e.checks)}}}"
           for e in reg.envs])
    loss_rows = []
    for f in reg.loss_functions:
        params = w.array(
            f"{m.name[f.id]}_params", "LossParam",
            [f"LossParam{{{cstr(p.name)}, {fmt_double(float(p.default))}, "
             f"{bool_lit(p.gt is not None)}, "
             f"{fmt_double(float(p.gt)) if p.gt is not None else '0.0'}}}"
             for p in f.params])
        loss_rows.append(f"LossFunction{{{cstr(f.name)}, {params}}}")
    table(w, "LossFunction", "kLossFunctions", loss_rows)
    rows = []
    for t in reg.strategy_types:
        keys = w.array(m.name[t.id] + "_keys", "KeyId",
                       [m.key_ref(i) for i in t.keys])
        rows.append(f"StrategyType{{{cstr(t.name)}, {keys}, "
                    f"{checks_span(w, m.name[t.id] + '_checks', t.checks)}}}")
    table(w, "StrategyType", "kStrategyTypes", rows)
    table(w, "InitMethod", "kInitMethods",
          [f"InitMethod{{{cstr(i.name)}, "
           f"{args_span(w, m.name[i.id], i.args)}, "
           f"{bool_lit(bool(i.refused))}, "
           f"{checks_span(w, m.name[i.id] + '_checks', i.checks)}}}"
           for i in reg.init_methods])

    w.out("namespace optim {")
    o = reg.optim.fields if reg.optim is not None else {}
    span = w.strings("optim_positive_only", o.get("positive_only", []))
    w.out(f"inline constexpr Span<std::string_view> positive_only = {span};")
    w.out("inline constexpr std::string_view auto_excluded_prefix = "
          f"{cstr(o.get('auto_excluded_prefix', ''))};")
    span = checks_span(w, "optim_checks", o.get("checks", []))
    w.out(f"inline constexpr Span<std::string_view> checks = {span};")
    w.out("}  // namespace optim")
    w.out()
    rows = []
    for t in reg.optim_targets:
        indices = w.array(m.name[t.id] + "_indices", "int",
                          [str(i) for i in t.indices])
        rows.append(f"OptimTarget{{{cstr(t.key)}, "
                    f"OptimTargetStatus::{OPTIM_TARGET_STATUS[t.status]}, "
                    f"{indices}, {cstr(t.msg)}}}")
    table(w, "OptimTarget", "kOptimTargets", rows)

    # ---- lookups ---------------------------------------------------------
    w.out("""\
/* The block of the name `name`; nullptr for none. */
constexpr const Block* find_block(std::string_view name)
{
  for (const Block& b : kBlocks)
    if (b.name == name) return &b;
  return nullptr;
}

/* The key of the name `name` valid at `place`; nullptr for none. */
constexpr const Key* find_key(std::string_view name, BlockId place)
{
  for (const Key& k : kKeys)
    if (k.name == name && (k.blocks & block_bit(place))) return &k;
  return nullptr;
}

/* The option `option` of the line of the key or form `line`; nullptr for
 * none. */
constexpr const LineOption* find_line_option(std::string_view line,
                                             std::string_view option)
{
  for (const LineOption& o : kLineOptions) {
    if (o.option != option) continue;
    for (std::string_view l : o.lines)
      if (l == line) return &o;
  }
  return nullptr;
}
""")
    w.out("}  // namespace hpotarget_registry")
    w.out("}  // namespace hpo")
    w.out("}  // namespace tadah")
    return finish(w, body_at)


def emit_params(reg: hr.Registry) -> str:
    m = Model(reg)
    w = Writer()
    header_top(w, [
        "HPO/blueprint/hpotarget_params.toml (a part file of",
        "HPO/blueprint/hpotarget.toml), read by",
        "HPO/blueprint/hpotarget_registry.py"], [
        "The PARAM keys of the target-file registry: [[param]],",
        "[[param_excluded]] and [[param_policy]] (blueprint/README.md, \"The",
        "generated tables\"). The PARAM keys of the annealers are",
        "anneal_keys_gen.h's."])
    w.out("#include <tadah/hpo/hpotarget_registry_gen.h>")
    w.out()
    w.out("#include <cstddef>")
    w.out("#include <string_view>")
    w.out()
    w.out("namespace tadah {")
    w.out("namespace hpo {")
    w.out("namespace param_registry {")
    w.out()
    w.out("using hpotarget_registry::Arg;")
    w.out("using hpotarget_registry::ArgKind;")
    w.out("using hpotarget_registry::Limit;")
    w.out("using hpotarget_registry::RefusedWord;")
    w.out("using hpotarget_registry::Span;")
    w.out()
    enum(w, "ParamId", "unsigned short", [m.name[p.id] for p in reg.params],
         "The PARAM keys, in registry order: kParams[i].id is ParamId(i).")
    w.out("""\
struct Param {
  ParamId id;
  std::string_view id_text;
  std::string_view name;
  Span<std::string_view> optimizers;   // LIB:ALGO, LIB, or * for every one
  Span<Arg> args;
  bool has_default_literal;
  std::string_view default_literal;
  bool indexed;                        // the name takes [<i>]
  Span<std::string_view> checks;
};

/* A PARAM key an optimiser names but does not take, and why. */
struct ParamExcluded {
  std::string_view name;
  Span<std::string_view> optimizers;
  std::string_view why;
};

/* What an optimiser does with a PARAM key no [[param]] entry names. */
struct ParamPolicy {
  std::string_view id_text;
  Span<std::string_view> optimizers;
  std::string_view msg;
  Span<std::string_view> checks;
};

/* The PARAM keys of one optimiser as the entries name it (LIB:ALGO, LIB or
 * *), in registry order. */
struct OptimiserParams {
  std::string_view optimizer;
  Span<ParamId> params;
};
""")
    body_at = len(w.lines)
    rows = []
    for p in reg.params:
        n = m.name[p.id]
        has_def, def_lit = default_literal(p.default)
        rows.append(f"Param{{ParamId::{n}, {cstr(p.id)}, {cstr(p.name)}, "
                    f"{w.strings(n + '_optimizers', p.optimizers)}, "
                    f"{args_span(w, n, p.args)}, {has_def}, {def_lit}, "
                    f"{bool_lit(p.indexed)}, "
                    f"{checks_span(w, n + '_checks', p.checks)}}}")
    order: list[str] = []
    for p in reg.params:
        for o in p.optimizers:
            if o not in order:
                order.append(o)
    by_opt = []
    for i, o in enumerate(order):
        ids = [f"ParamId::{m.name[p.id]}" for p in reg.params
               if o in p.optimizers]
        by_opt.append(f"OptimiserParams{{{cstr(o)}, "
                      f"{w.array(f'optimiser{i}_params', 'ParamId', ids)}}}")
    excluded = [f"ParamExcluded{{{cstr(x.name)}, "
                f"{w.strings(m.name[x.id] + '_optimizers', x.optimizers)}, "
                f"{cstr(x.why)}}}" for x in reg.param_excluded]
    policies = [f"ParamPolicy{{{cstr(p.id)}, "
                f"{w.strings(m.name[p.id] + '_optimizers', p.optimizers)}, "
                f"{cstr(p.msg)}, "
                f"{checks_span(w, m.name[p.id] + '_checks', p.checks)}}}"
                for p in reg.param_policies]
    table(w, "Param", "kParams", rows)
    table(w, "OptimiserParams", "kParamsByOptimiser", by_opt)
    table(w, "ParamExcluded", "kParamExcluded", excluded)
    table(w, "ParamPolicy", "kParamPolicies", policies)
    w.out("}  // namespace param_registry")
    w.out("}  // namespace hpo")
    w.out("}  // namespace tadah")
    return finish(w, body_at)


def headers(reg: hr.Registry) -> dict[str, str]:
    """The text of each header, by file name."""
    return {REGISTRY_HEADER: emit_registry(reg),
            PARAM_HEADER: emit_params(reg)}


# ---------------------------------------------------------------- main ----

def regenerate_hint(registry: pathlib.Path, out: pathlib.Path) -> str:
    """The command that rewrites the headers --check compared: the build
    target, which writes the headers of this repository from its registry,
    when both directories are the defaults; else --header with the two
    directories checked."""
    if (registry.resolve() == DEFAULT_REGISTRY.resolve()
            and out.resolve() == DEFAULT_OUT.resolve()):
        return f"`{REGENERATE}` (or gen_hpotarget.py --header)"
    return (f"`gen_hpotarget.py --header --registry "
            f"{shlex.quote(str(registry))} -o {shlex.quote(str(out))}`")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Write or check the C++ tables of the target-file "
                    "registry.")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--header", action="store_true",
                      help="write the headers")
    mode.add_argument("--check", action="store_true",
                      help="fail when a checked-in header is missing or "
                           "differs from what the registry and its loader "
                           "give")
    ap.add_argument("--registry", type=pathlib.Path,
                    default=DEFAULT_REGISTRY,
                    help="the directory of hpotarget.toml (default: this "
                         "script's)")
    ap.add_argument("-o", "--out", type=pathlib.Path, default=DEFAULT_OUT,
                    help="the directory of the headers (default: "
                         "include/tadah/hpo of this script's repository)")
    args = ap.parse_args(argv)
    try:
        texts = headers(hr.load(args.registry))
    except hr.RegistryError as e:
        for line in e.errors:
            print(f"error: {line}", file=sys.stderr)
        print(f"gen_hpotarget.py: the registry in {args.registry} has "
              f"{len(e.errors)} error(s); hpotarget_registry.py --check "
              "lists them", file=sys.stderr)
        return 1
    except GenerationError as e:
        print(f"gen_hpotarget.py: {e}", file=sys.stderr)
        return 1
    if args.header:
        args.out.mkdir(parents=True, exist_ok=True)
        for name, text in texts.items():
            path = args.out / name
            if not path.exists() or path.read_text(encoding="utf-8") != text:
                path.write_text(text, encoding="utf-8")
                print(f"gen_hpotarget.py: wrote {path}")
            else:
                print(f"gen_hpotarget.py: {path} is up to date")
        return 0
    stale = []
    for name, text in texts.items():
        path = args.out / name
        if not path.exists():
            stale.append(f"{path} does not exist")
        elif path.read_text(encoding="utf-8") != text:
            stale.append(f"{path} differs from what the registry and its "
                         "loader give")
    if stale:
        for s in stale:
            print(f"gen_hpotarget.py --check: {s}", file=sys.stderr)
        print("gen_hpotarget.py --check: regenerate with "
              f"{regenerate_hint(args.registry, args.out)} and commit the "
              "headers with the change to the registry, its loader or the "
              "generator", file=sys.stderr)
        return 1
    print(f"gen_hpotarget.py --check: {', '.join(texts)} in {args.out} "
          "match the registry and its loader")
    return 0


if __name__ == "__main__":
    sys.exit(main())
