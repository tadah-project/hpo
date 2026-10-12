#!/usr/bin/env python3
"""Check HPO's Catch2 test sources for the two ways a test case fails only in
the MPI build (TADAH_HPO_MPI), whose suite the serial checks never run.

  check_mpi_build_tests.py <tests_dir> [<tests_dir> ...]

Every *.cpp directly under each directory is read as the MPI build compiles
it: the branches of ``#ifdef TADAH_HPO_MPI``, ``#ifndef TADAH_HPO_MPI`` and
``#if [!]defined(TADAH_HPO_MPI)`` that build does not compile are dropped,
every other conditional keeps all its branches, and directive lines (a
TEST_CASE in a macro definition) are not code.

Rule PROCESS. The MPI build has no PC_ISOLATION PROCESS workers: hpo_run
refuses an explicit PROCESS ("PC_ISOLATION PROCESS is not supported in the
MPI build (TADAH_HPO_MPI) ...") and the worker entry refuses to start
("tadah --tadah-pc-worker: not supported in the MPI build"). A TEST_CASE
that starts such workers there must be Skipped there: the first statement
of its body is SKIP_WITHOUT_PC_PROCESS_WORKERS() (tests/pc_process_workers.h).
A case starts PROCESS workers when its body

  * builds a worker pool: a PcWorkerPool object it names or allocates
    (``PcWorkerPool pool(...)``, ``std::make_unique<PcWorkerPool>(...)``,
    ``new PcWorkerPool(...)``), or calls pc_worker_pool_init(); or
  * requests PROCESS for PC_LAMMPS lines: a string literal of the body
    contains "PC_ISOLATION PROCESS", or one statement names the literals
    "TADAH_PC_ISOLATION" and "PROCESS" (an environment override), and a
    string literal of the body contains "PC_LAMMPS " (hpo_run builds the
    pool for PC_LAMMPS lines). Such a case may assert the MPI build's
    refusal instead: a string literal of its body contains "is not
    supported in the MPI build".

Not seen: a pool or a request in a helper the case calls, a mode handed to a
helper as a value (``run_tadah_hpo(..., mode, ...)``: such cases list their
modes under ``#ifdef TADAH_HPO_MPI``, THREAD alone there, as
test_hpo_pc_readback_e2e.cpp does), and a temporary ``PcWorkerPool(...)``
(the width check, which throws before a worker is spawned).

Rule LAMMPS. Every LAMMPS instance of the MPI build needs that build's
communicator, MPI_COMM_SELF; the serial build's, the LAMMPS STUBS' 0, is a
null communicator under Open MPI (MPI_ERR_COMM). The test sources create
instances only through LammpsRunner::create_instance, which takes no
communicator and uses the build's (tadah::hpo::hpo_lammps_comm()).
Constructing one directly -- ``new LAMMPS_NS::LAMMPS(...)``, a
``LAMMPS_NS::LAMMPS`` object, ``make_unique<LAMMPS_NS::LAMMPS>``,
``new LMP(...)``, ``lammps_open...(...)`` -- is refused, in every build's
view of the file.

Exit status: 0 when every case follows both rules; 1 with one line per
violation; 2 for a usage error or when no TEST_CASE is found.
"""

from __future__ import annotations

import pathlib
import re
import sys
from dataclasses import dataclass, field

SKIP_MACRO = "SKIP_WITHOUT_PC_PROCESS_WORKERS"
MPI_SYMBOL = "TADAH_HPO_MPI"
REFUSAL_TEXT = "is not supported in the MPI build"


@dataclass
class Scanned:
    """A source as the rules read it, offsets and newlines kept.

    ``code``: comments and the contents of every string and character
    literal blanked (the quotes kept), so brackets and words are code.
    ``literals``: (start, end, value) of each string literal, adjacent
    literals joined ("a" "b" is "ab"), ``value`` the characters between
    the quotes as written."""
    code: str
    literals: list = field(default_factory=list)


def scan(text: str) -> Scanned:
    out = list(text)
    literals = []
    i, n = 0, len(text)

    def blank(a: int, b: int) -> None:
        for k in range(a, b):
            if out[k] != "\n":
                out[k] = " "

    def ident_before(k: int) -> bool:
        return k > 0 and (text[k - 1].isalnum() or text[k - 1] == "_")

    while i < n:
        c = text[i]
        if text.startswith("//", i):
            j = text.find("\n", i)
            j = n if j < 0 else j
            blank(i, j)
            i = j
        elif text.startswith("/*", i):
            j = text.find("*/", i + 2)
            j = n if j < 0 else j + 2
            blank(i, j)
            i = j
        elif text.startswith('R"', i) and (
                not ident_before(i) or text[i - 1] in "uUL"
                or text[max(0, i - 2):i] == "u8"):
            k = text.find("(", i + 2)
            if k < 0:
                i += 2
                continue
            close = ")" + text[i + 2:k] + '"'
            end = text.find(close, k + 1)
            end = n if end < 0 else end
            literals.append((i, end + len(close), text[k + 1:end]))
            blank(k + 1, end)
            i = end + len(close)
        elif c == "'" and i > 0 and text[i - 1].isdigit():
            i += 1   # a digit separator: 1'000
        elif c in "\"'":
            j = i + 1
            while j < n and text[j] != c and text[j] != "\n":
                j += 2 if text[j] == "\\" else 1
            if c == '"':
                literals.append((i, j + 1, text[i + 1:j]))
            blank(i + 1, j)
            i = j + 1
        else:
            i += 1
    code = "".join(out)
    joined = []
    for a, b, v in literals:
        if joined and not code[joined[-1][1]:a].strip():
            pa, _, pv = joined[-1]
            joined[-1] = (pa, b, pv + v)
        else:
            joined.append((a, b, v))
    return Scanned(code, joined)


_DIRECTIVE = re.compile(r"^\s*#\s*(\w*)(.*)$")


def _mpi_branch(kind: str, rest: str):
    """For a conditional on TADAH_HPO_MPI alone, whether its first branch
    is compiled in the MPI build; None for any other condition."""
    rest = rest.strip()
    if kind in ("ifdef", "ifndef") and rest == MPI_SYMBOL:
        return kind == "ifdef"
    if kind == "if":
        m = re.fullmatch(r"(!?)\s*defined\s*(?:\(\s*" + MPI_SYMBOL +
                         r"\s*\)|\s" + MPI_SYMBOL + ")", rest)
        if m:
            return not m.group(1)
    return None


def mpi_view(sc: Scanned) -> tuple[Scanned, list]:
    """@p sc as the MPI build compiles it: every line it does not compile
    and every directive line blanked, their literals dropped. Also returns,
    per line, whether it is code of that build."""
    lines = sc.code.split("\n")
    frames = []       # [mpi_known, taken, parent_active]
    active = True
    kept = []
    continued = False   # the line before ended a directive line with '\\'
    for line in lines:
        starts_directive = not continued and line.lstrip().startswith("#")
        directive = continued or starts_directive
        continued = directive and line.rstrip().endswith("\\")
        m = _DIRECTIVE.match(line) if starts_directive else None
        if m:
            kind, rest = m.group(1), m.group(2)
            if kind in ("if", "ifdef", "ifndef"):
                known = _mpi_branch(kind, rest)
                branch = True if known is None else known
                frames.append([known is not None, branch, active])
                active = active and branch
            elif kind in ("elif", "else") and frames:
                known, taken, parent = frames[-1]
                if not known:
                    branch = True
                elif kind == "else":
                    branch = not taken
                elif taken:
                    branch = False
                else:
                    # An #elif after a TADAH_HPO_MPI branch the MPI build
                    # does not compile: its own condition decides, unknown
                    # here, so it and every later branch are kept.
                    branch = True
                    frames[-1][0] = False
                frames[-1][1] = taken or branch
                active = parent and branch
            elif kind == "endif" and frames:
                active = frames.pop()[2]
        kept.append(active and not directive)
    out = []
    for line, keep in zip(lines, kept):
        out.append(line if keep else " " * len(line))
    code = "\n".join(out)
    literals = []
    for a, b, v in sc.literals:
        line = sc.code.count("\n", 0, a)
        if kept[line]:
            literals.append((a, b, v))
    return Scanned(code, literals), kept


@dataclass
class Case:
    name: str
    line: int
    body_start: int   # offset just after the body's '{'
    body_end: int     # offset of the body's closing '}'


def _matching(code: str, i: int, open_c: str, close_c: str) -> int:
    depth = 0
    for k in range(i, len(code)):
        if code[k] == open_c:
            depth += 1
        elif code[k] == close_c:
            depth -= 1
            if depth == 0:
                return k
    return -1


def test_cases(sc: Scanned) -> list:
    cases = []
    for m in re.finditer(r"\bTEST_CASE(?:_METHOD)?\s*\(", sc.code):
        open_p = m.end() - 1
        close_p = _matching(sc.code, open_p, "(", ")")
        if close_p < 0:
            continue
        brace = sc.code.find("{", close_p)
        if brace < 0 or sc.code[close_p + 1:brace].strip():
            continue
        end = _matching(sc.code, brace, "{", "}")
        if end < 0:
            continue
        name = next((v for a, _, v in sc.literals if open_p < a < close_p),
                    "")
        cases.append(Case(name, _line_of(sc.code, m.start()), brace + 1, end))
    return cases


_POOL = re.compile(
    r"\bPcWorkerPool\s+[A-Za-z_]\w*\s*[({]"
    r"|\bmake_(?:unique|shared)\s*<\s*(?:\w+\s*::\s*)*PcWorkerPool\s*>"
    r"|\bnew\s+(?:\w+\s*::\s*)*PcWorkerPool\b"
    r"|\bpc_worker_pool_init\s*\(")
_SKIP_FIRST = re.compile(r"\s*" + SKIP_MACRO + r"\s*\(\s*\)\s*;")
_LAMMPS = re.compile(
    r"\bnew\s+(?:LAMMPS_NS\s*::\s*)?LAMMPS\s*\("
    r"|\bLAMMPS_NS\s*::\s*LAMMPS\s+[A-Za-z_]\w*\s*[({]"
    r"|\bmake_(?:unique|shared)\s*<\s*(?:LAMMPS_NS\s*::\s*)?LAMMPS\s*>"
    r"|\bnew\s+(?:LammpsRunner\s*::\s*)?LMP\s*\("
    r"|\blammps_open\w*\s*\(")


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _flat(s: str) -> str:
    return " ".join(s.split())


def process_reasons(sc: Scanned, case: Case) -> list:
    """Why @p case starts PC_ISOLATION PROCESS workers in the MPI build;
    empty when it does not (or asserts that build's refusal)."""
    body = sc.code[case.body_start:case.body_end]
    lits = [(a, v) for a, _, v in sc.literals
            if case.body_start <= a < case.body_end]
    reasons = []
    m = _POOL.search(body)
    if m:
        at = _line_of(sc.code, case.body_start + m.start())
        reasons.append(f"builds a worker pool ('{_flat(m.group(0))}', "
                       f"line {at})")
    request = next((a for a, v in lits if "PC_ISOLATION PROCESS" in v), None)
    if request is None:
        for a, v in lits:
            if v != "TADAH_PC_ISOLATION":
                continue
            stmt_end = sc.code.find(";", a)
            if any(v2 == "PROCESS" and a < a2 < stmt_end for a2, v2 in lits):
                request = a
                break
    if (request is not None and any("PC_LAMMPS " in v for _, v in lits)
            and not any(REFUSAL_TEXT in v for _, v in lits)):
        reasons.append("requests PC_ISOLATION PROCESS for its PC_LAMMPS "
                       f"lines (line {_line_of(sc.code, request)})")
    return reasons


def check_text(text: str, where: str) -> tuple[list, int]:
    """The violations in the source @p text (named @p where in them) and
    the number of test cases of the MPI build's view."""
    problems = []
    whole = scan(text)
    for m in _LAMMPS.finditer(whole.code):
        problems.append(
            f"{where}:{_line_of(text, m.start())}: constructs a LAMMPS "
            f"instance directly ('{_flat(m.group(0))}'): create it with "
            "LammpsRunner::create_instance(tokens), which runs it on the "
            "build's communicator (hpo_lammps_comm(): MPI_COMM_SELF in the "
            "MPI build, where the serial build's 0 is a null communicator)")
    view, _ = mpi_view(whole)
    cases = test_cases(view)
    for case in cases:
        reasons = process_reasons(view, case)
        if not reasons or _SKIP_FIRST.match(view.code, case.body_start):
            continue
        problems.append(
            f"{where}:{case.line}: TEST_CASE \"{case.name}\" "
            f"{'; '.join(reasons)}, and the MPI build (TADAH_HPO_MPI) has "
            f"no PC_ISOLATION PROCESS workers: its first statement must be "
            f"{SKIP_MACRO}() (tests/pc_process_workers.h)")
    return problems, len(cases)


def main(argv: list) -> int:
    if len(argv) < 2:
        print(__doc__.split("\n\n")[1], file=sys.stderr)
        return 2
    problems, n_files, n_cases = [], 0, 0
    for d in argv[1:]:
        root = pathlib.Path(d)
        if not root.is_dir():
            print(f"check_mpi_build_tests: {d}: not a directory",
                  file=sys.stderr)
            return 2
        for path in sorted(root.glob("*.cpp")):
            p, n = check_text(path.read_text(encoding="utf-8"), str(path))
            problems += p
            n_files += 1
            n_cases += n
    if n_cases == 0:
        print("check_mpi_build_tests: no TEST_CASE in " + ", ".join(argv[1:]),
              file=sys.stderr)
        return 2
    for p in problems:
        print(p)
    print(f"check_mpi_build_tests: {n_files} files, {n_cases} test cases in "
          f"the MPI build's view, {len(problems)} violation(s)")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
