#!/usr/bin/env python3
"""Coverage of the target-file registry against the parser's sources.

    check_hpotarget_coverage.py <HPO source directory> [--registry DIR]

Reads the C++ sources of the target-file parser and extracts every name it
recognises as a string literal, then compares the names with those the
registry (blueprint/hpotarget.toml and its part files, loaded with
blueprint/hpotarget_registry.py) declares, per vocabulary:

  - every name the parser recognises has a registry entry ("missing from
    the registry");
  - every name of the registry is one the parser recognises ("not in the
    parser"): an entry for a key, an option, a word or a PARAM key the
    parser does not know fails.

The vocabularies of the options follow their lines: line_option:<line>,
for each line or form the registry gives options (PC_LAMMPS,
PC_FAILPROBE, the metric lines, RHOGATE, RHOBARGATE, OPTIM, OPTIM.AUTO,
PC_OBSERVABLE.cohesive_energy), and the STRATEGY keys of each TYPE,
strategy_type_key:<TYPE>. An option the registry gives a line the parser
does not take it on fails as "not in the parser", where the sources say
which line takes the option: the table of lammps_sim.cpp (its on_lammps
and on_failprobe fields) and the functions of each line. The keys of a
TYPE come from the second field of kStrategyTypeKeys.

Exits 1 and lists each difference, 0 when there is none.

How the names are extracted
---------------------------
Comments are removed first (string literals kept), so a name in a comment
or in commented-out code does not count. Each rule of EXTRACTION below
names a source file and, mostly, one function in it (found by a regular
expression on its definition, which must match exactly once), and reads:

  - comparisons `<operand> == "<literal>"` and `!=` in the function; each
    operand that occurs is mapped to the vocabularies its literals belong
    to, or ignored with a reason. A comparison whose operand the rule does
    not map is an error ("unclassified comparison"), and so is one with
    the literal on the left;
  - tables: the string literals of a braced initializer (`name = {...}` or
    `name[] = {...}`), each one or the first of each element, whose
    vocabulary may depend on the element's other fields;
  - calls: the literal argument of a call pattern (`param.find("...")`,
    `ceres_param(sp, "...")`, `std::getenv("TADAH_...")`).

In a function a rule names, two more things are errors, so that a name
read in another way is seen:
  - a comparison the rule cannot read: the operand compared with a named
    constant (`key == kNewKey`), with a constructed string
    (`key == std::string(...)`), or compare(), unless the rule's
    unread_ok gives the reason ("unread comparison");
  - a literal shaped like a name of the target file (an upper-case word
    or an option --name) that no rule reads and the registry does not
    know, as a name kept in a set or a map, unless the rule's not_names
    gives the reason ("unclassified literal").

A rule that finds nothing is an error: the code moved and the rule needs
updating.

What it cannot see
------------------
  - where a key of HPOTargets::read is valid: every comparison there is on
    one variable, `key`, so a name found there counts for a block, an END
    line, a key of the optimiser group (PARAM and STEP, compared to name
    what a line sets once) or a key of ROOT, OPTIMIZER, STRATEGY, INIT
    (METHOD, which adds to its chain), NOISE or RESTART alike. The blocks
    of a key are checked by tests/check_hpotarget_generated.py, which puts
    each key into every block;
  - the line of an option a function shared by two lines reads
    (LammpsSim::parse_tokens for PC_LAMMPS and PC_FAILPROBE,
    parse_gate_options for RHOGATE and RHOBARGATE): a name found there
    counts for both lines. check_hpotarget_generated.py puts each option
    on every line;
  - a literal that is not shaped like a name (a lower-case word, as a
    LOSS name or a lattice), outside the comparisons, tables and calls the
    rules read;
  - a name the parser builds instead of writing it: END lines are matched
    as literals in HPOTargets::read, but block_ended_by() builds "END" +
    name; a rule that compares a computed string;
  - prefix and substring tests: `k.rfind("RCUT", 0) == 0` (AUTO refused on
    the cutoff keys), `find(".DENSITY.")`, `.substr(0, 2) != "--"`;
  - names taken from data at run time: the internal keys of the training
    blueprint (OPTIM refuses each), the integer values of a key's formats
    (OPTIM refuses their indices), the DM_Embed component paths, the
    optimiser list (LIB and ALGO values; optimizers_gen.toml and its own
    sync test), the PARAM keys of the annealers (anneal_keys.toml and its
    own sync test) and NLopt's algorithm parameters (passed to NLopt);
  - LAMMPS command-line options (single '-'), which go to LAMMPS, and the
    reading of the LAMMPS scripts a PC_LAMMPS line names (lammps_sim.cpp:
    their variable, label, jump, if, clear, include and timer commands),
    which are LAMMPS input, not target-file text;
  - the words the number grammar refuses (inf, infinity, nan in
    strict_parse.cpp), which [reading] in hpotarget.toml describes;
  - names an INIT method compares while it draws its points
    (init_strategies.cpp: the DESCRIPTOR line of a WARM file, the density
    keys AMPGRIDMB, CGRIDMB and SGRIDMB of a WARM start), which are not
    target-file keys;
  - case rules: the extraction sees a literal, not whether the parser
    upper-cases the token first (OPTIM keys, LOSS names, gate keywords,
    OPTIM_SCALE values);
  - functions and files no rule names: the startup checks
    (startup_validation.cpp, rho_gates.cpp, scale_validation.cpp, the
    backends' run code), the `tadah refit` reader (read_block_until,
    refit_parser.cpp), the environment overrides' values in hpo.cpp, and
    density_gate_keys() (the keys of the RHOGATE arming rule, named in its
    message);
  - messages: the conformance test compares each invalid case's error with
    the parser's, and hpotarget_registry.py --check requires each message
    template of an entry to match an invalid case (or to give the reason
    it has none), so a template is compared with the parser through the
    cases that match it.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import re
import sys

# --------------------------------------------------------- C++ reading ----

def strip_comments(src: str) -> str:
    """`src` with every comment replaced by blanks (newlines kept), string
    and character literals left as they are."""
    out: list[str] = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if c == '"' and i > 0 and src[i - 1] == "R" and (
                i < 2 or not (src[i - 2].isalnum() or src[i - 2] == "_")):
            m = re.match(r'"([^()\\ ]{0,16})\(', src[i:])
            if m:
                close = ")" + m.group(1) + '"'
                end = src.find(close, i)
                end = n if end < 0 else end + len(close)
                out.append(src[i:end])
                i = end
                continue
        if c in "\"'":
            j = i + 1
            while j < n and src[j] != c:
                if src[j] == "\\":
                    j += 1
                j += 1
            out.append(src[i:j + 1])
            i = j + 1
            continue
        if src.startswith("//", i):
            j = src.find("\n", i)
            j = n if j < 0 else j
            out.append(" " * (j - i))
            i = j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            j = n if j < 0 else j + 2
            out.append("".join("\n" if ch == "\n" else " "
                               for ch in src[i:j]))
            i = j
            continue
        out.append(c)
        i += 1
    return "".join(out)


def braced(src: str, start: int) -> tuple[int, int]:
    """The span of the first {...} at or after `start`, braces inside
    string and character literals skipped."""
    i = src.index("{", start)
    depth, j, n = 0, i, len(src)
    while j < n:
        c = src[j]
        if c in "\"'":
            k = j + 1
            while k < n and src[k] != c:
                if src[k] == "\\":
                    k += 1
                k += 1
            j = k + 1
            continue
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return i, j + 1
        j += 1
    raise ValueError("unbalanced braces")


STRING = re.compile(r'"((?:[^"\\]|\\.)*)"')
COMPARISON = re.compile(
    r'([A-Za-z_][\w.:\->\[\]+]*(?:\((?:[^()]|\([^()]*\))*\))?'
    r'(?:\.[A-Za-z_]\w*\(\))?(?:\[[^\]]*\])?)\s*(==|!=)\s*'
    r'"((?:[^"\\]|\\.)*)"')
LITERAL_LEFT = re.compile(r'"((?:[^"\\]|\\.)*)"\s*(==|!=)')


@dataclasses.dataclass
class Found:
    name: str
    vocabularies: tuple[str, ...]
    where: str           # "file:line (rule)"


@dataclasses.dataclass
class Rule:
    """One extraction rule: a file, optionally one function of it, and
    what to read there."""
    label: str
    file: str
    function: str | None = None          # regex of the definition
    operands: dict[str, tuple[str, ...]] = dataclasses.field(
        default_factory=dict)            # operand -> vocabularies
    ignore: dict[str, str] = dataclasses.field(
        default_factory=dict)            # operand -> why it is not a name
    ignore_literals: dict[str, str] = dataclasses.field(
        default_factory=dict)            # literal -> why
    compare: bool = True                 # read the comparisons
    tables: list[tuple[str, str, tuple[str, ...]]] = dataclasses.field(
        default_factory=list)            # (name regex, all|first, vocab)
    calls: list[tuple[str, tuple[str, ...]]] = dataclasses.field(
        default_factory=list)            # (regex with one group, vocab)
    finds_names: bool = True             # False: it only guards against
                                         # unclassified comparisons
    unread_ok: dict[str, str] = dataclasses.field(
        default_factory=dict)            # comparison start -> why it is
                                         # no name of the target file
    not_names: dict[str, str] = dataclasses.field(
        default_factory=dict)            # literal -> why it is no name


IGNORE_OPTIMISERS = "an optimiser name (LIB or ALGO value: the optimiser list)"
IGNORE_INIT_MODE = ("the INIT_MODE grammar of the annealers' PARAM keys "
                    "(anneal_keys.toml)")
# The metric lines that take --skip-above and --skip-rel (PC_ERSQ takes
# neither: a cap would reject the better fits).
METRIC_OPTIONS = ("line_option:PC_EMAE", "line_option:PC_ERMSE",
                  "line_option:PC_ErRMSE", "line_option:PC_SRMSE_GPA",
                  "line_option:PC_FRMSE")


def pc_line_option_of(element: str) -> tuple[str, ...]:
    """The lines an element {"--opt", on_lammps, on_failprobe} of
    lammps_sim.cpp's line_options() table gives its option to."""
    flags = [f.strip() for f in element.strip("{} \n").split(",")[1:3]]
    out = []
    if flags and flags[0] == "true":
        out.append("line_option:PC_LAMMPS")
    if len(flags) > 1 and flags[1] == "true":
        out.append("line_option:PC_FAILPROBE")
    return tuple(out)


def strategy_type_key_of(element: str) -> tuple[str, ...]:
    """The TYPE of an element {"KEY", StrategyType::TYPE, ...} of
    strategy_rules.cpp's kStrategyTypeKeys."""
    m = re.search(r"StrategyType::(\w+)", element)
    return (f"strategy_type_key:{m.group(1)}",) if m else ()


READ_KEY = ("block", "block_end", "group_key", "key:ROOT", "key:OPTIMIZER",
            "key:STRATEGY", "key:INIT", "key:NOISE", "key:RESTART")
PARAM_FIND = (r'param(?:_raw)?\s*\.\s*(?:find|count)\(\s*"([^"]+)"\s*\)',
              ("param",))

EXTRACTION: list[Rule] = [
    Rule("the target-file reader", "src/hpo_targets.cpp",
         r"^void HPOTargets::read\(",
         operands={"key": READ_KEY, "t": ("strategy_type", "word"),
                   "v": ("word",), "PC_ISOLATION": ("word",),
                   "flag": METRIC_OPTIONS,
                   "toks[i2]": ("line_option:PC_FAILPROBE",),
                   "e.method": ("init_method",)},
         ignore={"sp.lib": IGNORE_OPTIMISERS, "sp.algo": IGNORE_OPTIMISERS,
                 "sp.inner->lib": IGNORE_OPTIMISERS,
                 "b.sp->inner->lib": IGNORE_OPTIMISERS},
         tables=[(r"kBlockNames\s*\[\]", "all", ("block",))],
         unread_ok={"key == std::string(": "an END line: END and a block "
                                           "name of kBlockNames, which the "
                                           "table is read for"},
         not_names={"END": "the prefix of an END line, joined to a block "
                           "name of kBlockNames",
                    "NLOPT": IGNORE_OPTIMISERS, "DLIB": IGNORE_OPTIMISERS,
                    "BFGS": IGNORE_OPTIMISERS}),
    Rule("the INIT and RESTART keys", "src/hpo_targets.cpp",
         r"^static bool dispatch_init_attr\(",
         operands={"key": ("key:INIT", "alias"),
                   "e.method": ("init_method",), "v": ("word",)}),
    Rule("the INNER refusals", "src/hpo_targets.cpp",
         r"^static const char\* inner_rejects_attr\(",
         operands={"key": ("inner_refused",)}),
    Rule("the removed keywords", "src/hpo_targets.cpp",
         r"^static bool reject_old_keyword\(", compare=False,
         tables=[(r"\bmap\s*\[\]", "first", ("removed",))],
         unread_ok={"key == r": "a keyword of the table, which is read as "
                                "a table"}),
    Rule("the OPTIM line", "src/hpo_targets.cpp",
         r"^void HPOTargets::process_optim_config\(",
         operands={"k": ("optim_target",),
                   "opt": ("line_option:OPTIM.AUTO",),
                   "tag": ("line_option:OPTIM",),
                   "upper(toks[toks.size() - 2])": ("line_option:OPTIM",),
                   "upper(tokens.front())": ("form",)},
         unread_ok={".compare(": "the suffix .AMPLITUDES of a DM_Embed "
                                 "component path",
                    "key != k": "a value of the potential left out, found "
                                "by its OPTIM key"},
         not_names={"RCUT": "the prefix of the cutoff keys that AUTO "
                            "refuses ([optim] auto_excluded_prefix)"}),
    Rule("the OPTIM keys of the fitted coefficients and of the embedding "
         "reference", "src/hpo_targets.cpp",
         r"^std::string HPOTargets::optim_fitted_key_error\(",
         operands={"k": ("optim_target",)}),
    Rule("the OPTIM structure keys", "src/hpo_targets.cpp",
         r"^std::string HPOTargets::optim_structure_key_error\(",
         tables=[(r"\bstructure\s*\[\]", "first", ("optim_target",))],
         unread_ok={"k == e": "a key of the table, which is read as a "
                              "table"}),
    Rule("the OPTIM keys applied once at startup", "src/hpo_targets.cpp",
         r"^std::string HPOTargets::optim_startup_only_key_error\(",
         tables=[(r"\bonce\s*\[\]", "all", ("optim_target",))],
         unread_ok={"k == o": "a key of the table, which is read as a "
                              "table"}),
    Rule("the OPTIM values of the fit", "src/hpo_targets.cpp",
         r"^std::optional<HPOTargets::FitValueRule> HPOTargets::optim_fit_value_rule\(",
         operands={"k": ("optim_target",)}),
    Rule("the OPTIM choices and switches of the fit", "src/hpo_targets.cpp",
         r"^std::string HPOTargets::optim_switch_key_error\(",
         operands={"k": ("optim_target",)}),
    Rule("the OPTIM retired keys", "src/hpo_targets.cpp",
         r"^std::string HPOTargets::optim_retired_key_error\(",
         tables=[(r"\bretired\s*\[\]", "all", ("optim_target",))],
         unread_ok={"k != r": "a key of the table, which is read as a "
                              "table"}),
    Rule("the OPTIM discrete index", "src/hpo_targets.cpp",
         r"^std::string HPOTargets::optim_discrete_index_error\(",
         operands={"core::to_upper(key)": ("optim_target",)}),
    Rule("the OPTIM keys of a DM_Embed potential", "src/hpo_targets.cpp",
         r"^std::string HPOTargets::optim_component_key_error\(",
         ignore={"field": "a DM_Embed component path in the message",
                 "s.cutoff": "the CUTOFF TYPE of the training "
                             "configuration, in the message"},
         tables=[(r"\blegacy\s*\[\]", "first", ("optim_target",))],
         unread_ok={"k == r": "a key of the table, which is read as a "
                              "table"},
         not_names={f: "a field of a DM_Embed component path"
                    for f in ("CENTRES", "AMPLITUDES", "WIDTHS", "ETA")}),
    Rule("the gate options", "src/hpo_targets.cpp",
         r"^void parse_gate_options\(",
         operands={"kw": ("line_option:RHOGATE", "line_option:RHOBARGATE"),
                   "m": ("word",)}),
    Rule("the RHOBARGATE lattice", "src/hpo_targets.cpp",
         r"^RhoBarGateSpec parse_rhobargate\(",
         operands={"s.lattice": ("word",)}),
    Rule("the OPTIM_SCALE override", "src/hpo_targets.cpp",
         r"^HPOTargets::HPOTargets\(",
         operands={"v": ("word",)},
         ignore={"s.lib": IGNORE_OPTIMISERS},
         calls=[(r'std::getenv\(\s*"(TADAH_[A-Z_]+)"\s*\)', ("env",))],
         not_names={"TADAH": IGNORE_OPTIMISERS,
                    "ANNEAL": IGNORE_OPTIMISERS, "ASA": IGNORE_OPTIMISERS,
                    "NLOPT": IGNORE_OPTIMISERS,
                    "OUTER": "the name of the STRATEGY-level phase in a "
                             "message"}),
    Rule("the booleans", "src/optimizer_block.cpp", r"^bool parse_bool\(",
         operands={"lo": ("word",)}),
    Rule("the optimiser keys", "src/optimizer_block.cpp",
         r"^const std::vector<std::string> &opt_attr_keys\(",
         tables=[(r"\bkeys", "all", ("group_key",))]),
    Rule("the optimiser key reader", "src/optimizer_block.cpp",
         r"^bool dispatch_opt_attr\(",
         operands={"key": ("group_key",), "v": ("word",)},
         ignore={"ukey": IGNORE_INIT_MODE, "up(vals[i])": IGNORE_INIT_MODE},
         not_names={"INIT_MODE": IGNORE_INIT_MODE}),
    Rule("the PARAM value grammar", "src/optimizer_block.cpp",
         r"^bool param_takes_another_value\(",
         ignore={"key": IGNORE_INIT_MODE, "up(next)": IGNORE_INIT_MODE,
                 "up(sofar[0])": IGNORE_INIT_MODE,
                 "up(sofar[1])": IGNORE_INIT_MODE},
         not_names={"JITTER": IGNORE_INIT_MODE,
                    "INIT_MODE": IGNORE_INIT_MODE},
         finds_names=False),
    Rule("the LAMMPS time spec", "src/strict_parse.cpp",
         r"^std::optional<double> parse_lammps_timespec\(",
         operands={"tok": ("word",)}),
    Rule("the STRATEGY keys of one TYPE", "src/strategy_rules.cpp",
         compare=False,
         tables=[(r"kStrategyTypeKeys\s*\[\]", "first",
                  strategy_type_key_of)]),
    Rule("the PC_LAMMPS and PC_FAILPROBE options", "src/lammps_sim.cpp",
         r"^static const std::vector<LineOption>& line_options\(",
         tables=[(r"\bk", "first", pc_line_option_of)]),
    Rule("the PC_LAMMPS reader", "src/lammps_sim.cpp",
         r"^void LammpsSim::parse_tokens\(",
         operands={"tokens[i]": ("line_option:PC_LAMMPS",
                                 "line_option:PC_FAILPROBE")},
         ignore={"name": "the LAMMPS variable maxtime, which HPO reserves "
                         "for the script timeout"},
         ignore_literals={"-var": "a LAMMPS command-line option, passed to "
                                  "LAMMPS (checked for the reserved name)",
                          "-v": "a LAMMPS command-line option, passed to "
                                "LAMMPS (checked for the reserved name)",
                          "-sc": "LAMMPS' short spelling of -screen, passed "
                                 "to LAMMPS as -screen",
                          "-l": "LAMMPS' short spelling of -log, passed to "
                                "LAMMPS as -log"}),
    Rule("the observable kinds", "src/observable_parser.cpp",
         r"^std::unique_ptr<Observable> parse_pc_observable_line\(",
         operands={"kind": ("form",)}),
    Rule("the cohesive_energy options", "src/observable_parser.cpp",
         r"^parse_cohesive_energy\(",
         operands={"t": ("line_option:PC_OBSERVABLE.cohesive_energy",),
                   "kind": ("word",),
                   "lattice": ("word",), "name": ("word",)},
         ignore={"substr(0, 2)": "the option prefix --"}),
    Rule("the loss functions", "src/loss_function.cpp",
         r"^LossFunction::Type LossFunction::from_string\(",
         operands={"u": ("loss",)}),
    Rule("the INIT methods", "src/init_strategies.cpp", r"^make_one\(",
         operands={"e.method": ("init_method",)}),
    Rule("the ASA options", "src/asa_options.cpp",
         operands={"n": ("param",), "base": ("param",)},
         tables=[(r"\bkDescs\s*\[\]", "first", ("param",)),
                 (r"\bkExcluded", "first", ("param_excluded",))]),
    Rule("the TADAH:GRID PARAM key", "src/backend_rules.cpp", compare=False,
         calls=[PARAM_FIND,
                (r'reject_unknown_params\(\s*\w+\s*,\s*\{\s*"([^"]+)"',
                 ("param",))]),
    Rule("the DLIB:BFGS PARAM keys", "include/tadah/hpo/dlib_adapter.h",
         r"^inline void validate_dlib_bfgs_spec\(",
         operands={"kv.first": ("param",)},
         ignore={"sp.lib": IGNORE_OPTIMISERS, "sp.algo": IGNORE_OPTIMISERS},
         not_names={"DLIB": IGNORE_OPTIMISERS, "BFGS": IGNORE_OPTIMISERS},
         calls=[(r'dlib_param\(\s*\w+\s*,\s*"([^"]+)"\s*\)', ("param",))]),
    Rule("the CERES PARAM keys", "include/tadah/hpo/ceres_adapter.h",
         r"^inline void validate_ceres_common_\(",
         ignore={"sp.lib": IGNORE_OPTIMISERS},
         tables=[(r"\bcommon_params", "all", ("param",))],
         calls=[PARAM_FIND,
                (r'validate_ceres_int_param_\(\s*\w+\s*,\s*\w+\s*,\s*"([^"]+)"',
                 ("param",))]),
    Rule("the CERES PARAM reads", "include/tadah/hpo/ceres_adapter.h",
         compare=False,
         calls=[(r'ceres_param\(\s*\w+\s*,\s*"([^"]+)"\s*\)', ("param",))]),
    Rule("the finite-difference PARAM keys", "include/tadah/hpo/fd_stencil.h",
         r"^inline bool is_fd_param\(", operands={"key": ("param",)}),
    Rule("the finite-difference settings", "src/fd_stencil.cpp",
         r"^FdSettings fd_settings_of\(",
         operands={"it->second": ("word",)}, calls=[PARAM_FIND]),
    Rule("the environment overrides", "src/hpo.cpp", compare=False,
         calls=[(r'std::getenv\(\s*"(TADAH_(?:FAILSCORE_GRADING|BUMPGATE|'
                 r'PC_ISOLATION|PC_WORKER_TIMEOUT))"\s*\)', ("env",))]),
]

# Every vocabulary of the registry this check compares (the loader's
# Registry.vocabulary names). A key scope the registry uses must be here.
VOCABULARIES = ("block", "block_end", "alias", "key:ROOT", "key:OPTIMIZER",
                "key:STRATEGY", "key:INIT", "key:NOISE", "key:RESTART",
                "group_key", "inner_refused", "removed", "form", "loss",
                "init_method", "strategy_type", "optim_target", "param",
                "param_excluded", "env", "word")


def vocabularies(reg) -> tuple[str, ...]:
    """VOCABULARIES and the scoped ones of the registry: the options of
    each line (line_option:<line>) and the STRATEGY keys of each TYPE
    (strategy_type_key:<TYPE>)."""
    lines = sorted({ln for o in reg.line_options for ln in o.lines})
    types = sorted(t.name for t in reg.strategy_types)
    return VOCABULARIES + tuple(f"line_option:{ln}" for ln in lines) + \
        tuple(f"strategy_type_key:{t}" for t in types)


def line_of(src: str, pos: int) -> int:
    return src.count("\n", 0, pos) + 1


def table_literals(text: str, how: str) -> list[tuple[str, str]]:
    """The names of a braced table: each literal ("all"), or the first
    literal of each element ("first"), with the text of its element."""
    if how == "all":
        return [(m.group(1), m.group(0)) for m in STRING.finditer(text)]
    out = []
    inner = text[1:-1]
    pos = 0
    while True:
        k = inner.find("{", pos)
        if k < 0:
            break
        a, b = braced(inner, k)
        m = STRING.search(inner[a:b])
        if m:
            out.append((m.group(1), inner[a:b]))
        pos = b
    return out


# A comparison of a scanned function that the rules cannot read: the key
# compared with a named constant or a constructed string, or compare().
UNREAD_COMPARISON = re.compile(
    r"\b(?:key|k|kw|opt|tag|flag|t|v)\s*(?:==|!=)\s*(?:std::string\s*\(|"
    r"[A-Za-z_][\w:]*\b(?!\s*\())|\.compare\s*\(")
# A literal shaped like a name of the target file: a key, a word or an
# option.
NAME_SHAPED = re.compile(r'"([A-Z][A-Z0-9_]+|--[a-z][a-z0-9-]*)"')


def unread_comparisons(rule: Rule, src: str, offset: int,
                       body: str) -> list[str]:
    out = []
    for m in UNREAD_COMPARISON.finditer(body):
        text = m.group(0)
        if any(text.startswith(f"{op} ") or text.startswith(op)
               for op in rule.unread_ok):
            continue
        where = f"{rule.file}:{line_of(src, offset + m.start())}"
        out.append(f"{where}: comparison in {rule.label} that this check "
                   f"cannot read: '{text.strip()}' (compare with a literal, "
                   "or list the operand in the rule's unread_ok with the "
                   "reason)")
    return out


def name_literals(rule: Rule, src: str, offset: int, body: str,
                  read: set[str]) -> list[tuple[str, str]]:
    """The name-shaped literals of a scanned function that no rule reads:
    (literal, place). compare() reports those the registry does not know
    and the rule does not ignore."""
    out = []
    for m in NAME_SHAPED.finditer(body):
        lit = m.group(1)
        if lit in read or lit in rule.ignore_literals or \
                lit in rule.not_names:
            continue
        out.append((lit, f"{rule.file}:{line_of(src, offset + m.start())} "
                         f"({rule.label})"))
    return out


def extract(hpo: pathlib.Path,
            unread: list | None = None) -> tuple[list[Found], list[str]]:
    """The names the rules find, and the problems of the rules. `unread`
    receives the name-shaped literals of the scanned functions that no
    rule reads (name_literals), for compare() to look up."""
    found: list[Found] = []
    problems: list[str] = []
    if unread is None:
        unread = []
    cache: dict[str, str] = {}
    for rule in EXTRACTION:
        path = hpo / rule.file
        if rule.file not in cache:
            try:
                cache[rule.file] = strip_comments(path.read_text())
            except OSError as e:
                problems.append(f"{rule.file}: cannot read ({e})")
                cache[rule.file] = ""
        src = cache[rule.file]
        offset, body = 0, src
        if rule.function:
            hits = list(re.finditer(rule.function, src, re.M))
            if len(hits) != 1:
                problems.append(
                    f"{rule.file}: {rule.label}: the definition "
                    f"/{rule.function}/ is found {len(hits)} times, not once: "
                    "the extraction rule needs updating")
                continue
            a, b = braced(src, hits[0].end())
            offset, body = a, src[a:b]
        n_found = 0
        rule_names: set[str] = set()
        if rule.compare:
            for m in COMPARISON.finditer(body):
                operand, literal = m.group(1), m.group(3)
                where = f"{rule.file}:{line_of(src, offset + m.start())}"
                if literal in rule.ignore_literals or operand in rule.ignore:
                    continue
                vocab = rule.operands.get(operand)
                if vocab is None:
                    problems.append(
                        f"{where}: unclassified comparison in {rule.label}: "
                        f'{operand} {m.group(2)} "{literal}" (map the operand '
                        "to a vocabulary, or ignore it with a reason)")
                    continue
                found.append(Found(literal, vocab, f"{where} ({rule.label})"))
                rule_names.add(literal)
                n_found += 1
            for m in LITERAL_LEFT.finditer(body):
                where = f"{rule.file}:{line_of(src, offset + m.start())}"
                problems.append(
                    f"{where}: unclassified comparison in {rule.label}: the "
                    f'literal "{m.group(1)}" is on the left, which this check '
                    "does not read")
        for name_re, how, vocab in rule.tables:
            m = re.search(name_re + r"\s*=\s*\{", body)
            if not m:
                problems.append(f"{rule.file}: {rule.label}: no table "
                                f"/{name_re}/: the extraction rule needs "
                                "updating")
                continue
            a, b = braced(body, m.end() - 1)
            names = table_literals(body[a:b], how)
            where = f"{rule.file}:{line_of(src, offset + a)}"
            for name, element in names:
                vocabs = vocab(element) if callable(vocab) else vocab
                if vocabs:
                    found.append(Found(name, vocabs,
                                       f"{where} ({rule.label})"))
                    rule_names.add(name)
            n_found += len(names)
        for call_re, vocab in rule.calls:
            for m in re.finditer(call_re, body):
                where = f"{rule.file}:{line_of(src, offset + m.start())}"
                found.append(Found(m.group(1), vocab,
                                   f"{where} ({rule.label})"))
                rule_names.add(m.group(1))
                n_found += 1
        if rule.function:
            problems += unread_comparisons(rule, src, offset, body)
            unread.extend(name_literals(rule, src, offset, body,
                                        rule_names))
        if n_found == 0 and rule.finds_names:
            problems.append(f"{rule.file}: {rule.label}: the rule finds no "
                            "name: the extraction rule needs updating")
    return found, problems


def compare(found: list[Found], reg,
            unread: list[tuple[str, str]] = ()) -> list[str]:
    """The differences between the names found and the registry's, and
    the name-shaped literals no rule reads that the registry does not
    know."""
    problems: list[str] = []
    names = vocabularies(reg) if hasattr(reg, "line_options") else \
        VOCABULARIES
    vocab = {v: reg.vocabulary(v) for v in names}
    if unread:
        known = reg.vocabulary("registry_name")
        reported = set()
        for lit, where in unread:
            if lit in known or lit in reported:
                continue
            reported.add(lit)
            problems.append(f"unclassified literal: '{lit}' at {where} is "
                            "shaped like a name of the target file, no rule "
                            "reads it and the registry does not know it "
                            "(read it in a rule, or give the reason in the "
                            "rule's not_names)")
    missing: dict[tuple[str, tuple[str, ...]], list[str]] = {}
    for f in found:
        if not any(f.name in vocab[v] for v in f.vocabularies):
            missing.setdefault((f.name, f.vocabularies), []).append(f.where)
    for (name, vocabs), places in missing.items():
        more = (f" and {len(places) - 1} more place(s)"
                if len(places) > 1 else "")
        problems.append(f"missing from the registry: '{name}' "
                        f"({' or '.join(vocabs)}), {places[0]}{more}")
    seen: dict[str, set[str]] = {v: set() for v in names}
    for f in found:
        for v in f.vocabularies:
            seen[v].add(f.name)
    used = {v for f in found for v in f.vocabularies}
    for v in names:
        if v not in used and vocab[v]:
            problems.append(f"no extraction rule reads the vocabulary {v}")
        for name in sorted(vocab[v] - seen[v]):
            problems.append(f"not in the parser: '{name}' ({v}) is declared "
                            "in the registry, and the parser's sources do "
                            "not recognise it")
    scopes = {k.scope for k in reg.keys if not k.group}
    for s in sorted(scopes):
        if f"key:{s}" not in VOCABULARIES:
            problems.append(f"not in the parser: the registry has keys of "
                            f"scope {s}, a scope no extraction rule reads")
    return problems


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("hpo", type=pathlib.Path, help="the HPO source directory")
    ap.add_argument("--registry", type=pathlib.Path,
                    help="the registry directory (default: <hpo>/blueprint)")
    args = ap.parse_args(argv)
    registry_dir = args.registry or args.hpo / "blueprint"
    loader = registry_dir / "hpotarget_registry.py"
    if not loader.is_file():
        print(f"check_hpotarget_coverage.py: {loader} not found: there is "
              "no registry to compare with the parser's sources")
        return 1
    sys.path.insert(0, str(registry_dir))
    import hpotarget_registry  # noqa: E402  (the registry's own loader)

    unread: list[tuple[str, str]] = []
    found, problems = extract(args.hpo, unread)
    try:
        reg = hpotarget_registry.load(registry_dir, resolve=False)
    except hpotarget_registry.RegistryError as e:
        for line in e.errors:
            print(f"registry error: {line}")
        print(f"check_hpotarget_coverage.py: the registry does not load "
              f"({len(e.errors)} error(s))")
        return 1
    problems += compare(found, reg, unread)
    for p in problems:
        print(p)
    names = len({(f.name, f.vocabularies) for f in found})
    if problems:
        print(f"check_hpotarget_coverage.py: {len(problems)} problem(s); "
              f"{names} names found in the parser's sources")
        return 1
    print(f"check_hpotarget_coverage.py: OK: the registry and the parser's "
          f"sources name the same {names} names")
    return 0


if __name__ == "__main__":
    sys.exit(main())
