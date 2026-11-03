#!/usr/bin/env python3
"""The cases on which the two parsers of the target file are compared.

    hpotarget_differential.py cases OUT [--hpo DIR] [--tadah-mlip DIR]
                                        [--registry DIR]

Writes to OUT the cases of hpotarget_case_runner (tests/tools/
hpotarget_case_runner.cpp; tests/tools/hpotarget_cases.h reads them) that
test_hpotarget_differential.cpp reads with ParsedTarget::read and with the
registry-driven parser, comparing the description of each parse
(tadah::hpo::testing::describe_parse), the refusal of the file and the
refusal of the startup checks. Prints how many cases each source has, and
exits 1 when a source has none or the corpus has a problem.

The sources, and the id of each case (<source>:<id in the source>)
--------------------------------------------------------------------
registry-example  each example of the target-file registry
registry-invalid  and each invalid entry, in its wrapper, with the
                  training configuration of its context, the registry's
                  fixture files and the environment it sets (as
                  test_hpotarget_registry_conformance.cpp composes it):
                  mode parse or startup as the entry's check says; an
                  invalid entry checked by `tadah hpo --check` runs
                  through the startup checks (validate_hpo_setup), which
                  the comparison reads.
generated         each case of the checks of check_hpotarget_generated.py
                  (blocks, repeat, ranges, case, words, params, lines,
                  defaults), in its mode (defaults as parse: the
                  description holds what the parse set).
corpus            each input of tests/hpotarget_corpus.py (hpo-data,
                  registry, mlip-tests and, with --tadah-mlip, docs).
"""

from __future__ import annotations

import argparse
import pathlib
import sys

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_hpotarget_generated as generated  # noqa: E402
import hpotarget_corpus as corpus  # noqa: E402

SOURCES = ("registry-example", "registry-invalid", "generated", "corpus")
# The mode each check of a registry entry runs in.
REGISTRY_MODES = {"parse": "parse", "startup": "startup", "check": "startup"}


def registry_cases(hr, reg) -> list[tuple[str, dict]]:
    out = []
    for c in reg.cases():
        comp = hr.compose(reg, c.wrap, c.text, c.context)
        ctx = reg.contexts[comp.context]
        case = {"mode": REGISTRY_MODES[c.check], "file": comp.file,
                "config": ctx.config, "has_train_db": ctx.has_train_db,
                "has_validation": ctx.has_validation,
                "fixtures": corpus.fixtures(reg)}
        if c.env:
            case["env"] = dict(c.env)
        out.append((f"registry-{c.kind}:{c.owner}:{c.index}", case))
    return out


def generated_cases(hr, reg) -> list[tuple[str, dict]]:
    g = generated.Generator(hr, reg)
    for name in generated.CHECK_NAMES:
        getattr(g, f"gen_{name}")()
    out = []
    for c in g.cases:
        case = {k: v for k, v in c.case.items() if k != "id"}
        if case["mode"] == "defaults":
            case["mode"] = "parse"
        out.append((f"generated:{c.case['id']}", case))
    return out


def corpus_cases(hpo: pathlib.Path, registry: pathlib.Path,
                 tm: pathlib.Path | None) -> tuple[list[tuple[str, dict]],
                                                   list[str]]:
    c = corpus.collect(hpo, registry, tm)
    return ([(f"corpus:{i.id}", {"mode": "parse", **i.case})
             for i in c.inputs], list(c.problems))


def all_cases(hpo: pathlib.Path, registry: pathlib.Path,
              tm: pathlib.Path | None) -> tuple[list[tuple[str, dict]],
                                                list[str]]:
    hr, reg = corpus.load_registry(registry)
    cases = registry_cases(hr, reg) + generated_cases(hr, reg)
    more, problems = corpus_cases(hpo, registry, tm)
    cases += more
    ids: set[str] = set()
    for cid, _ in cases:
        if cid in ids:
            problems.append(f"{cid}: two cases have this id")
        ids.add(cid)
    for s in SOURCES:
        if not any(cid.split(":", 1)[0] == s for cid, _ in cases):
            problems.append(f"{s}: no case")
    return cases, problems


def cases_text(cases: list[tuple[str, dict]]) -> str:
    lines = []
    for cid, case in cases:
        lines.append("[[case]]")
        lines.append(f"id = {corpus.toml_string(cid)}")
        for k, v in case.items():
            lines.append(f"{k} = {corpus.toml_value(v)}")
        lines.append("")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("command", choices=("cases",))
    ap.add_argument("out", type=pathlib.Path)
    ap.add_argument("--hpo", type=pathlib.Path, default=corpus.DEFAULT_HPO)
    ap.add_argument("--tadah-mlip", default="")
    ap.add_argument("--registry", type=pathlib.Path)
    a = ap.parse_args(argv)
    tm = pathlib.Path(a.tadah_mlip) if a.tadah_mlip else None
    registry = a.registry or a.hpo / "blueprint"
    cases, problems = all_cases(a.hpo, registry, tm)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(cases_text(cases), encoding="utf-8")
    for s in SOURCES:
        n = sum(1 for cid, _ in cases if cid.split(":", 1)[0] == s)
        print(f"{s}: {n} cases")
    for p in problems:
        print(f"problem  {p}")
    if problems:
        print(f"hpotarget_differential.py: {len(problems)} problem(s)")
        return 1
    print(f"hpotarget_differential.py: {len(cases)} cases written to {a.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
