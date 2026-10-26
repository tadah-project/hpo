#!/usr/bin/env python3
"""Generate include/tadah/hpo/anneal_keys_gen.h from blueprint/anneal_keys.toml.

The TOML file is the single source of truth for the ANNEAL-family PARAM keys
(names, types, defaults, ranges, applicability, enum values, cross-key rules,
rename hints, docs metadata).  The generated header is checked in; the ctest
target hpo_anneal_registry_sync runs `gen_anneal_keys.py --check` and fails
when the checked-in header drifts from the TOML.

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
KIND_ORDER = ["serial", "pt", "pa"]
TYPES = {"bool": "AnnealKeyType::Bool", "count": "AnnealKeyType::Count",
         "float": "AnnealKeyType::Float", "enum": "AnnealKeyType::Enum"}
GROUPS = {"budget", "temperature", "steps", "ladder", "swaps", "init", "stopping"}
RULE_TYPES = {"requires_flag", "conflicts_flag", "unused_when_set",
              "unused_when_flag_off", "requires_enum", "kind_reject"}

KEY_FIELDS = {"name", "type", "default", "default_serial", "default_pt",
              "default_pa", "values", "kinds", "group", "rank", "doc",
              "long_doc", "range_doc",
              "min", "max", "min_open", "max_open", "range_msg", "range_note",
              "report_value", "custom_range", "accepts_auto", "auto_kinds",
              "accepts_off", "default_doc", "default_doc_serial",
              "default_doc_pt", "default_doc_pa"}
RULE_FIELDS = {"type", "key", "flag", "other", "kind", "enum_key", "value", "msg"}
RENAMED_FIELDS = {"old", "kinds", "msg"}


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


def default_literal(key: dict, raw: str) -> float:
    """Numeric form of a default: enum defaults resolve to their index."""
    if key["type"] == "enum":
        if raw not in key["values"]:
            fail(f"key {key['name']}: default {raw!r} not in values")
        return float(key["values"].index(raw))
    return float(raw)


def load(toml_path: pathlib.Path) -> dict:
    with open(toml_path, "rb") as fh:
        data = tomllib.load(fh)
    if set(data) - {"meta", "key", "rule", "renamed"}:
        fail("unknown top-level tables: "
             f"{sorted(set(data) - {'meta', 'key', 'rule', 'renamed'})}")
    if data.get("meta", {}).get("version") != 2:
        fail("meta.version must be 2")

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
        if key["type"] == "enum":
            values = key.get("values")
            if not values or not all(
                    v.isupper() and v.isidentifier() for v in values):
                fail(f"key {key['name']}: enum needs uppercase identifier values")
            if any(f in key for f in ("min", "max", "range_msg", "custom_range")):
                fail(f"key {key['name']}: enum keys take no range")
        elif "values" in key:
            fail(f"key {key['name']}: 'values' is enum-only")
        if key["type"] == "bool" and ("min" in key or "max" in key):
            fail(f"key {key['name']}: bool keys take no range")
        if "auto_kinds" in key:
            if "accepts_auto" not in key:
                fail(f"key {key['name']}: auto_kinds without accepts_auto")
            if (not key["auto_kinds"] or
                    any(k not in key["kinds"] for k in key["auto_kinds"])):
                fail(f"key {key['name']}: auto_kinds {key['auto_kinds']!r} "
                     f"is not a subset of kinds {key['kinds']!r}")
        # Resolve per-kind defaults: override > base > absent.
        key["_def"] = {}
        for kind in key["kinds"]:
            raw = key.get(f"default_{kind}", key.get("default"))
            if raw is not None:
                key["_def"][kind] = default_literal(key, str(raw))
        for kind in KIND_ORDER:
            if f"default_{kind}" in key and kind not in key["kinds"]:
                fail(f"key {key['name']}: default_{kind} but not a {kind} key")
            if f"default_doc_{kind}" in key and kind not in key["kinds"]:
                fail(f"key {key['name']}: default_doc_{kind} but not a "
                     f"{kind} key")
        # The default in words of each kind: override > base > absent.
        key["_def_doc"] = {}
        for kind in key["kinds"]:
            text = key.get(f"default_doc_{kind}", key.get("default_doc"))
            if text is not None:
                key["_def_doc"][kind] = text
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
        elif rule["type"] == "requires_enum":
            ek = rule.get("enum_key")
            if ek not in by_name or by_name[ek]["type"] != "enum":
                fail(f"rule on {rule['key']}: enum_key {ek!r} is not an enum key")
            if rule.get("value") not in by_name[ek]["values"]:
                fail(f"rule on {rule['key']}: value {rule.get('value')!r} "
                     f"not in {ek}'s values")
        elif rule["type"] == "kind_reject":
            if rule.get("kind") not in KINDS:
                fail(f"rule on {rule['key']}: bad kind {rule.get('kind')!r}")

    # Every kind that takes a key states its default: a value (default,
    # default_<kind>) or words (default_doc, default_doc_<kind>), which the
    # help (`tadah explain`) and the documentation print. A kind that
    # refuses the key (a kind_reject rule) lists it nowhere.
    refused = {(r["key"], r["kind"]) for r in data.get("rule", [])
               if r["type"] == "kind_reject"}
    for key in data.get("key", []):
        for kind in key["kinds"]:
            if (key["name"], kind) in refused:
                if kind in key["_def"] or kind in key["_def_doc"]:
                    fail(f"key {key['name']}: a default for {kind}, which "
                         "refuses the key")
                continue
            if kind not in key["_def"] and kind not in key["_def_doc"]:
                fail(f"key {key['name']}: no default and no default_doc "
                     f"for {kind}")

    for ren in data.get("renamed", []):
        extra = set(ren) - RENAMED_FIELDS
        if extra:
            fail(f"renamed {ren.get('old', '?')}: unknown fields {sorted(extra)}")
        for field in ("old", "kinds", "msg"):
            if field not in ren:
                fail(f"renamed {ren.get('old', '?')}: missing '{field}'")
        if not ren["kinds"] or any(k not in KINDS for k in ren["kinds"]):
            fail(f"renamed {ren['old']}: bad kinds {ren['kinds']!r}")
        # A rename hint must not shadow a key that is live for that kind.
        live = by_name.get(ren["old"])
        if live and any(k in live["kinds"] for k in ren["kinds"]):
            fail(f"renamed {ren['old']}: shadows a live key for {ren['kinds']}")
    return data


def bool_lit(value: bool) -> str:
    return "true" if value else "false"


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
    out("enum class AnnealKeyType : unsigned char { Bool, Count, Float, Enum };")
    out("")
    out("enum class AnnealRuleType : unsigned char {")
    out("  RequiresFlag,      // key valid only when bool `arg` parses to 1")
    out("  ConflictsFlag,     // key invalid when bool `arg` parses to 1")
    out("  UnusedWhenSet,     // key invalid when `arg` is explicitly set")
    out("  UnusedWhenFlagOff, // key invalid when bool `arg` parses to 0")
    out("  RequiresEnum,      // key valid only when enum `arg` == `arg2`")
    out("  KindReject,        // key invalid for `kind_mask`, custom message")
    out("};")
    out("")
    out("struct AnnealKeyInfo {")
    out("  const char* name;")
    out("  AnnealKeyType type;")
    out("  unsigned kinds;          // KIND_* bitmask")
    out("  const char* group;")
    out("  int rank;                // importance order within the group")
    out("  const char* doc;")
    out("  const char* long_doc;     // detail after doc; nullptr = none")
    out("  // Per-kind defaults (absent = presence-tested or run-time AUTO).")
    out("  bool has_def_serial, has_def_pt, has_def_pa;")
    out("  double def_serial, def_pt, def_pa;   // enum defaults = value index")
    out("  bool has_min, min_open, has_max, max_open;")
    out("  double min, max;")
    out("  const char* range_msg;   // \"must be ...\"; nullptr = no table range check")
    out("  bool report_value;       // append \" (got <v>)\" to the range message")
    out("  const char* const* enum_values;  // Enum type only, else nullptr")
    out("  std::size_t enum_count;")
    out("  // 1.5.0 value vocabulary: a numeric key may also accept the token")
    out("  // AUTO (\"derive at run time\") and/or OFF (\"feature disabled\"),")
    out("  // each resolving to the internal value below. Keys where 0 is a")
    out("  // literal value (V_FLOOR_FRAC, LADDER_MIN_GAP) declare neither.")
    out("  bool accepts_auto, accepts_off;")
    out("  double auto_value, off_value;")
    out("  unsigned auto_kinds;     // KIND_* bitmask of the kinds that accept AUTO")
    out("  // The default in words of each kind (default_doc): a rule rather")
    out("  // than a value, or nullptr when the kind has a value default.")
    out("  const char* def_doc_serial;")
    out("  const char* def_doc_pt;")
    out("  const char* def_doc_pa;")
    out("};")
    out("")
    out("struct AnnealKeyRule {")
    out("  AnnealRuleType type;")
    out("  const char* key;")
    out("  const char* arg;         // flag / other key / enum key")
    out("  const char* arg2;        // RequiresEnum: required enum token")
    out("  unsigned kind_mask;      // KindReject only")
    out("  const char* msg;         // full message suffix after \"TADAH:<ALGO>: \"")
    out("};")
    out("")
    out("struct AnnealKeyRename {")
    out("  const char* old_name;")
    out("  unsigned kinds;          // where the hint applies")
    out("  const char* msg;         // full message suffix after \"TADAH:<ALGO>: \"")
    out("};")
    out("")
    for key in data["key"]:
        if key["type"] != "enum":
            continue
        vals = ", ".join(cstr(v) for v in key["values"])
        out(f"inline constexpr const char* k{key['name']}_values[] = {{{vals}}};")
    out("")
    out("inline constexpr AnnealKeyInfo kKeys[] = {")
    for key in data["key"]:
        kinds = " | ".join(KINDS[k] for k in key["kinds"])
        defs = key["_def"]
        has = {k: k in defs for k in KIND_ORDER}
        val = {k: fmt_double(defs[k]) if k in defs else "0.0" for k in KIND_ORDER}
        has_min, has_max = "min" in key, "max" in key
        min_v = fmt_double(float(key["min"])) if has_min else "0.0"
        max_v = fmt_double(float(key["max"])) if has_max else "0.0"
        range_msg = cstr(key["_range_msg"]) if key["_range_msg"] else "nullptr"
        if key["type"] == "enum":
            enum_vals = f"k{key['name']}_values"
            enum_count = str(len(key["values"]))
        else:
            enum_vals, enum_count = "nullptr", "0"
        out(f"  {{{cstr(key['name'])}, {TYPES[key['type']]}, {kinds},")
        long_doc = cstr(key["long_doc"]) if key.get("long_doc") else "nullptr"
        out(f"   {cstr(key['group'])}, {key['rank']}, {cstr(key['doc'])},")
        out(f"   {long_doc},")
        out(f"   {bool_lit(has['serial'])}, {bool_lit(has['pt'])}, "
            f"{bool_lit(has['pa'])}, {val['serial']}, {val['pt']}, {val['pa']},")
        out(f"   {bool_lit(has_min)}, {bool_lit(bool(key.get('min_open')))}, "
            f"{bool_lit(has_max)}, {bool_lit(bool(key.get('max_open')))}, "
            f"{min_v}, {max_v},")
        has_auto = "accepts_auto" in key
        has_off = "accepts_off" in key
        auto_v = fmt_double(float(key["accepts_auto"])) if has_auto else "0.0"
        off_v = fmt_double(float(key["accepts_off"])) if has_off else "0.0"
        out(f"   {range_msg}, {bool_lit(bool(key.get('report_value')))}, "
            f"{enum_vals}, {enum_count},")
        auto_kinds = (" | ".join(KINDS[k] for k in key.get("auto_kinds", key["kinds"]))
                      if has_auto else "0u")
        out(f"   {bool_lit(has_auto)}, {bool_lit(has_off)}, {auto_v}, {off_v},")
        def_doc = {k: cstr(key["_def_doc"][k]) if k in key["_def_doc"]
                   else "nullptr" for k in KIND_ORDER}
        out(f"   {auto_kinds},")
        out(f"   {def_doc['serial']}, {def_doc['pt']}, {def_doc['pa']}}},")
    out("};")
    out("")
    rule_cpp = {"requires_flag": "RequiresFlag", "conflicts_flag": "ConflictsFlag",
                "unused_when_set": "UnusedWhenSet",
                "unused_when_flag_off": "UnusedWhenFlagOff",
                "requires_enum": "RequiresEnum",
                "kind_reject": "KindReject"}
    out("inline constexpr AnnealKeyRule kRules[] = {")
    for rule in data.get("rule", []):
        arg = rule.get("flag") or rule.get("other") or rule.get("enum_key") or ""
        arg2 = rule.get("value") or ""
        mask = KINDS[rule["kind"]] if rule["type"] == "kind_reject" else "0u"
        out(f"  {{AnnealRuleType::{rule_cpp[rule['type']]}, {cstr(rule['key'])}, "
            f"{cstr(arg)}, {cstr(arg2)}, {mask}, {cstr(rule['msg'])}}},")
    out("};")
    out("")
    out("inline constexpr AnnealKeyRename kRenamed[] = {")
    for ren in data.get("renamed", []):
        kinds = " | ".join(KINDS[k] for k in ren["kinds"])
        out(f"  {{{cstr(ren['old'])}, {kinds}, {cstr(ren['msg'])}}},")
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
    out("/* Default values, one constexpr per key kind-variant with a static")
    out(" * default (enum defaults are value indices).  The anneal::Knobs")
    out(" * member initialisers consume these; nothing else may restate a")
    out(" * default. */")
    out("namespace anneal_defaults {")
    for key in data["key"]:
        defs = key["_def"]
        if not defs:
            continue
        distinct = set(defs.values())
        if len(distinct) == 1 and set(defs) == set(key["kinds"]):
            out(f"inline constexpr double {key['name']} = "
                f"{fmt_double(next(iter(distinct)))};")
        else:
            for kind in KIND_ORDER:
                if kind in defs:
                    out(f"inline constexpr double {key['name']}_{kind.upper()} = "
                        f"{fmt_double(defs[kind])};")
    out("}  // namespace anneal_defaults")
    out("")
    out("/* Named enum value indices for readable driver code, e.g.")
    out(" *   kb.step_mode == anneal_enums::STEP_MODE::FROZEN */")
    out("namespace anneal_enums {")
    for key in data["key"]:
        if key["type"] != "enum":
            continue
        out(f"namespace {key['name']} {{")
        for idx, v in enumerate(key["values"]):
            out(f"inline constexpr int {v} = {idx};")
        out("}")
    out("}  // namespace anneal_enums")
    out("")
    out("}  // namespace hpo")
    out("}  // namespace tadah")
    out("")
    return "\n".join(lines)


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
