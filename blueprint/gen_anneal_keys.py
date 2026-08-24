#!/usr/bin/env python3
"""Generate include/tadah/hpo/anneal_keys_gen.h from blueprint/anneal_keys.toml.

The TOML file is the single source of truth for the ANNEAL-family PARAM keys
(names, types, defaults, ranges, applicability, cross-key rules, docs
metadata).  The generated header is checked in; the ctest target
hpo_anneal_registry_sync runs `gen_anneal_keys.py --check` and fails when the
checked-in header drifts from the TOML.

Usage:
  gen_anneal_keys.py --header [-o PATH]   write the header (default path below)
  gen_anneal_keys.py --check  [-o PATH]   exit 1 if PATH differs from the TOML
"""

from __future__ import annotations

import argparse
import pathlib
import sys
import tomllib

SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DEFAULT_TOML = SCRIPT_DIR / "anneal_keys.toml"
DEFAULT_HEADER = SCRIPT_DIR.parent / "include" / "tadah" / "hpo" / "anneal_keys_gen.h"

KINDS = {"serial": "KIND_SERIAL", "pt": "KIND_PT", "pa": "KIND_PA"}
TYPES = {"bool": "AnnealKeyType::Bool", "count": "AnnealKeyType::Count",
         "float": "AnnealKeyType::Float"}
GROUPS = {"budget", "temperature", "steps", "ladder", "swaps", "init", "stopping"}
RULE_TYPES = {"requires_flag", "conflicts_flag", "unused_when_set",
              "unused_when_flag_off", "kind_reject"}

KEY_FIELDS = {"name", "type", "default", "default_pt", "default_pa", "kinds",
              "group", "rank", "doc", "min", "max", "min_open", "max_open",
              "range_msg", "range_note", "report_value", "custom_range"}
RULE_FIELDS = {"type", "key", "flag", "other", "kind", "msg"}


def fail(msg: str) -> None:
    raise SystemExit(f"gen_anneal_keys.py: error: {msg}")


def fmt_double(value: float) -> str:
    """A C++ double literal that round-trips the value exactly."""
    text = repr(float(value))
    if text in ("inf", "-inf", "nan"):
        fail(f"non-finite literal {text!r} is not representable")
    if "e" not in text and "." not in text:
        text += ".0"
    return text


def cstr(text: str) -> str:
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"') + '"'


def number_text(value: float) -> str:
    """The way a bound reads in an error message: integral values print bare."""
    fvalue = float(value)
    if fvalue == int(fvalue):
        return str(int(fvalue))
    return repr(fvalue)


def build_range_msg(key: dict) -> str | None:
    """Compose the 'must be ...' message from the declared bounds."""
    if key.get("custom_range"):
        if any(f in key for f in ("min", "max", "range_msg")):
            fail(f"key {key['name']}: custom_range excludes min/max/range_msg")
        return None
    msg = key.get("range_msg")
    has_min, has_max = "min" in key, "max" in key
    if msg is None:
        if not has_min and not has_max:
            return None
        if has_min and has_max:
            lo = "(" if key.get("min_open") else "["
            hi = ")" if key.get("max_open") else "]"
            msg = (f"must be in {lo}{number_text(key['min'])},"
                   f"{number_text(key['max'])}{hi}")
        elif has_min:
            op = ">" if key.get("min_open") else ">="
            msg = f"must be {op} {number_text(key['min'])}"
        else:
            op = "<" if key.get("max_open") else "<="
            msg = f"must be {op} {number_text(key['max'])}"
    note = key.get("range_note")
    if note:
        msg = f"{msg} {note}"
    return msg


def load(toml_path: pathlib.Path) -> dict:
    with open(toml_path, "rb") as fh:
        data = tomllib.load(fh)
    if set(data) - {"meta", "key", "rule"}:
        fail(f"unknown top-level tables: {sorted(set(data) - {'meta', 'key', 'rule'})}")
    if data.get("meta", {}).get("version") != 1:
        fail("meta.version must be 1")

    names = set()
    for key in data.get("key", []):
        extra = set(key) - KEY_FIELDS
        if extra:
            fail(f"key {key.get('name', '?')}: unknown fields {sorted(extra)}")
        for field in ("name", "type", "kinds", "group", "rank", "doc"):
            if field not in key:
                fail(f"key {key.get('name', '?')}: missing field '{field}'")
        if key["name"] in names:
            fail(f"duplicate key {key['name']}")
        names.add(key["name"])
        if key["type"] not in TYPES:
            fail(f"key {key['name']}: bad type {key['type']!r}")
        if key["group"] not in GROUPS:
            fail(f"key {key['name']}: bad group {key['group']!r}")
        if not key["kinds"] or any(k not in KINDS for k in key["kinds"]):
            fail(f"key {key['name']}: bad kinds {key['kinds']!r}")
        for field in ("default", "default_pt", "default_pa"):
            if field in key:
                float(key[field])  # must parse
        if "default_pt" in key and "default" not in key:
            fail(f"key {key['name']}: default_pt without default")
        if key["type"] == "bool" and ("min" in key or "max" in key):
            fail(f"key {key['name']}: bool keys take no range")
        key["_range_msg"] = build_range_msg(key)

    by_name = {k["name"]: k for k in data.get("key", [])}
    for rule in data.get("rule", []):
        extra = set(rule) - RULE_FIELDS
        if extra:
            fail(f"rule on {rule.get('key', '?')}: unknown fields {sorted(extra)}")
        if rule.get("type") not in RULE_TYPES:
            fail(f"rule on {rule.get('key', '?')}: bad type {rule.get('type')!r}")
        if "msg" not in rule:
            fail(f"rule on {rule.get('key', '?')}: missing msg")
        if rule.get("key") not in by_name:
            fail(f"rule references unknown key {rule.get('key')!r}")
        if rule["type"] in ("requires_flag", "conflicts_flag", "unused_when_flag_off"):
            flag = rule.get("flag")
            if flag not in by_name or by_name[flag]["type"] != "bool":
                fail(f"rule on {rule['key']}: flag {flag!r} is not a bool key")
        elif rule["type"] == "unused_when_set":
            if rule.get("other") not in by_name:
                fail(f"rule on {rule['key']}: other {rule.get('other')!r} unknown")
        elif rule["type"] == "kind_reject":
            if rule.get("kind") not in KINDS:
                fail(f"rule on {rule['key']}: bad kind {rule.get('kind')!r}")
    return data


def emit(data: dict) -> str:
    lines: list[str] = []
    out = lines.append
    out("// GENERATED FILE -- DO NOT EDIT.")
    out("//")
    out("// Source of truth: HPO/blueprint/anneal_keys.toml")
    out("// Generator:       HPO/blueprint/gen_anneal_keys.py --header")
    out("// Drift gate:      ctest hpo_anneal_registry_sync (gen_anneal_keys.py --check)")
    out("#pragma once")
    out("")
    out("#include <cstddef>")
    out("#include <string_view>")
    out("")
    out("namespace tadah {")
    out("namespace hpo {")
    out("namespace anneal_registry {")
    out("")
    out("inline constexpr unsigned KIND_SERIAL = 1u;")
    out("inline constexpr unsigned KIND_PT     = 2u;")
    out("inline constexpr unsigned KIND_PA     = 4u;")
    out("")
    out("enum class AnnealKeyType : unsigned char { Bool, Count, Float };")
    out("")
    out("enum class AnnealRuleType : unsigned char {")
    out("  RequiresFlag,      // key valid only when `flag` parses to 1")
    out("  ConflictsFlag,     // key invalid when `flag` parses to 1")
    out("  UnusedWhenSet,     // key invalid when `other` is explicitly set")
    out("  UnusedWhenFlagOff, // key invalid when `flag` parses to 0")
    out("  KindReject,        // key invalid for `kind`, custom message")
    out("};")
    out("")
    out("struct AnnealKeyInfo {")
    out("  const char* name;")
    out("  AnnealKeyType type;")
    out("  unsigned kinds;          // KIND_* bitmask")
    out("  const char* group;")
    out("  int rank;                // importance order within the group")
    out("  const char* doc;")
    out("  bool has_default;        // false = presence-tested optional key")
    out("  double def;              // base default (bool: 0/1)")
    out("  bool has_def_pt, has_def_pa;")
    out("  double def_pt, def_pa;   // per-kind default overrides")
    out("  bool has_min, min_open, has_max, max_open;")
    out("  double min, max;")
    out("  const char* range_msg;   // \"must be ...\"; nullptr = no table range check")
    out("  bool report_value;       // append \" (got <v>)\" to the range message")
    out("};")
    out("")
    out("struct AnnealKeyRule {")
    out("  AnnealRuleType type;")
    out("  const char* key;")
    out("  const char* arg;         // flag / other key; kind name for KindReject")
    out("  unsigned kind_mask;      // KindReject only")
    out("  const char* msg;         // full message suffix after \"TADAH:<ALGO>: \"")
    out("};")
    out("")
    out("inline constexpr AnnealKeyInfo kKeys[] = {")
    for key in data["key"]:
        kinds = " | ".join(KINDS[k] for k in key["kinds"])
        has_default = "default" in key
        def_v = fmt_double(float(key["default"])) if has_default else "0.0"
        has_pt, has_pa = "default_pt" in key, "default_pa" in key
        def_pt = fmt_double(float(key["default_pt"])) if has_pt else "0.0"
        def_pa = fmt_double(float(key["default_pa"])) if has_pa else "0.0"
        has_min, has_max = "min" in key, "max" in key
        min_v = fmt_double(float(key["min"])) if has_min else "0.0"
        max_v = fmt_double(float(key["max"])) if has_max else "0.0"
        range_msg = cstr(key["_range_msg"]) if key["_range_msg"] else "nullptr"
        out(f"  {{{cstr(key['name'])}, {TYPES[key['type']]}, {kinds},")
        out(f"   {cstr(key['group'])}, {key['rank']}, {cstr(key['doc'])},")
        out(f"   {bool_lit(has_default)}, {def_v}, "
            f"{bool_lit(has_pt)}, {bool_lit(has_pa)}, {def_pt}, {def_pa},")
        out(f"   {bool_lit(has_min)}, {bool_lit(bool(key.get('min_open')))}, "
            f"{bool_lit(has_max)}, {bool_lit(bool(key.get('max_open')))}, "
            f"{min_v}, {max_v},")
        out(f"   {range_msg}, {bool_lit(bool(key.get('report_value')))}}},")
    out("};")
    out("")
    rule_cpp = {"requires_flag": "RequiresFlag", "conflicts_flag": "ConflictsFlag",
                "unused_when_set": "UnusedWhenSet",
                "unused_when_flag_off": "UnusedWhenFlagOff",
                "kind_reject": "KindReject"}
    out("inline constexpr AnnealKeyRule kRules[] = {")
    for rule in data.get("rule", []):
        arg = rule.get("flag") or rule.get("other") or rule.get("kind") or ""
        mask = KINDS[rule["kind"]] if rule["type"] == "kind_reject" else "0u"
        out(f"  {{AnnealRuleType::{rule_cpp[rule['type']]}, {cstr(rule['key'])}, "
            f"{cstr(arg)}, {mask}, {cstr(rule['msg'])}}},")
    out("};")
    out("")
    out("inline constexpr const AnnealKeyInfo* find_key(std::string_view name)")
    out("{")
    out("  for (const auto& k : kKeys)")
    out("    if (name == k.name) return &k;")
    out("  return nullptr;")
    out("}")
    out("")
    out("}  // namespace anneal_registry")
    out("")
    out("/* Default values, one constexpr per key with a static default.  The")
    out(" * anneal::Knobs member initialisers consume these; nothing else may")
    out(" * restate a default. */")
    out("namespace anneal_defaults {")
    for key in data["key"]:
        if "default" not in key:
            continue
        out(f"inline constexpr double {key['name']} = "
            f"{fmt_double(float(key['default']))};")
    for key in data["key"]:
        for field, suffix in (("default_pt", "_PT"), ("default_pa", "_PA")):
            if field in key:
                out(f"inline constexpr double {key['name']}{suffix} = "
                    f"{fmt_double(float(key[field]))};")
    out("}  // namespace anneal_defaults")
    out("")
    out("}  // namespace hpo")
    out("}  // namespace tadah")
    out("")
    return "\n".join(lines)


def bool_lit(value: bool) -> str:
    return "true" if value else "false"


def main() -> int:
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--header", action="store_true", help="write the header")
    mode.add_argument("--check", action="store_true",
                      help="fail if the checked-in header drifts from the TOML")
    parser.add_argument("--toml", type=pathlib.Path, default=DEFAULT_TOML)
    parser.add_argument("-o", "--out", type=pathlib.Path, default=DEFAULT_HEADER)
    args = parser.parse_args()

    text = emit(load(args.toml))
    if args.header:
        args.out.write_text(text)
        print(f"wrote {args.out}")
        return 0
    on_disk = args.out.read_text() if args.out.exists() else "<missing>"
    if on_disk != text:
        sys.stderr.write(
            f"gen_anneal_keys.py --check: {args.out} is out of date with "
            f"{args.toml}; run gen_anneal_keys.py --header\n")
        return 1
    print(f"{args.out} is in sync")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
