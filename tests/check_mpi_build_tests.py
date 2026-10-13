#!/usr/bin/env python3
"""Check HPO's Catch2 test sources for the two ways a test case fails only in
the MPI build (TADAH_HPO_MPI), whose suite the serial checks never run.

  check_mpi_build_tests.py <tests_dir> [<tests_dir> ...]

Every *.cpp, *.h and *.hpp under each directory (subdirectories included) is
read twice. The LAMMPS rule reads the whole file. The PROCESS rule reads it
as the MPI build compiles it: the branches of ``#ifdef TADAH_HPO_MPI``,
``#ifndef TADAH_HPO_MPI`` and ``#if [!]defined(TADAH_HPO_MPI)`` that build
does not compile are dropped, every other conditional keeps all its
branches, and the directive lines that build reads are checked as well (a
macro's replacement: a TEST_CASE in a macro is no test case). Comments and
the insides of string literals are not code.

Rule PROCESS. The MPI build has no PC_ISOLATION PROCESS workers: hpo_run
refuses every explicit PROCESS request, whatever lines the configuration has
("PC_ISOLATION PROCESS is not supported in the MPI build (TADAH_HPO_MPI)
..."), and the worker entry refuses to start ("tadah --tadah-pc-worker: not
supported in the MPI build"). A TEST_CASE that builds a worker pool or makes
such a request is Skipped there: the first statement of its body is
SKIP_WITHOUT_PC_PROCESS_WORKERS() (tests/pc_process_workers.h).

  * Building a pool: any use of the class PcWorkerPool other than as a
    pointer or reference type, a unique_ptr/shared_ptr/weak_ptr holder or a
    scope (``PcWorkerPool::PROTOCOL_VERSION``) -- a named pool, a temporary,
    ``auto p = PcWorkerPool(...)``, ``make_unique<PcWorkerPool>``,
    ``std::optional<PcWorkerPool>``, ``new PcWorkerPool`` -- and a call of
    pc_worker_pool_init().
  * A PROCESS request: a string literal (adjacent literals joined, \\t \\n
    \\r \\v \\f read as whitespace) that holds the configuration line
    ``PC_ISOLATION PROCESS`` (any blanks between the words, an optional
    ``#`` comment after it, at the literal's start or after a newline), or
    the assignment ``TADAH_PC_ISOLATION=PROCESS`` (a command line, a
    ``K=V`` list), or that is the mode ``"PROCESS"`` itself (blanks around
    it ignored: a value for ``setenv``, a mode list, a helper's mode
    argument) -- except as an operand of ``==`` or ``!=``. The literals of
    TEST_CASE, SECTION, INFO, WARN, FAIL and SUCCEED are names and messages,
    never a request. A request may instead sit in a brace block of the case
    that also holds a literal with "is not supported in the MPI build" (that
    block and the blocks in it): a check of the MPI build's refusal.

Outside every TEST_CASE body -- a helper function, a file-scope constant, a
macro, a test header -- a pool or a request is refused in the MPI build's
view whatever calls it (a refusal block exempts a request there too): this
check sees them only in the case that makes them, so a helper takes the
mode from its caller (``run_tadah_hpo(..., mode, ...)``) and a pool by
reference, or is compiled in the serial build only. The run-time check of
test_main.cpp backs the pool part in the serial build: there the worker of a
case that did not call SKIP_WITHOUT_PC_PROCESS_WORKERS() first refuses to
start, as every worker of the MPI build does, whatever code built the pool.

Not seen: a value assembled from pieces (``std::string("PROC") + "ESS"``),
a request in a file a case reads (a fixture or an example configuration),
and a case that relies on the serial build's default PROCESS through the
tadah executable (the MPI build's default is THREAD; in-process, its
workers meet the run-time check).

Rule LAMMPS. Every LAMMPS instance of the MPI build needs that build's
communicator, MPI_COMM_SELF; the serial build's, the LAMMPS STUBS' 0, is a
null communicator under Open MPI (MPI_ERR_COMM). The test sources create
instances only through LammpsRunner::create_instance, which takes no
communicator and uses the build's (tadah::hpo::hpo_lammps_comm()). So the
LAMMPS class -- ``LAMMPS`` unqualified or qualified by ``LAMMPS_NS::``, and
its LammpsRunner alias ``LMP`` unqualified or qualified by
``LammpsRunner::`` -- is named only as a pointer or reference type, a
unique_ptr/shared_ptr/weak_ptr holder or a scope; any other use (an object,
a temporary, ``new``, ``make_unique``/``make_shared``, a typedef or alias of
it, ``std::optional<LMP>``) and a call of a ``lammps_open*`` function is
refused, in every build's view of the file. Not seen: the class reached
through a trait (``LMPPtr::element_type``, ``decltype``).

Exit status: 0 when every file follows both rules; 1 with one line per
violation; 2 for a usage error or when no TEST_CASE is found.
"""

from __future__ import annotations

import bisect
import pathlib
import re
import sys
from dataclasses import dataclass, field

SKIP_MACRO = "SKIP_WITHOUT_PC_PROCESS_WORKERS"
MPI_SYMBOL = "TADAH_HPO_MPI"
REFUSAL_TEXT = "is not supported in the MPI build"
SOURCE_SUFFIXES = (".cpp", ".h", ".hpp")


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


@dataclass
class View:
    """@p sc as the MPI build compiles it. ``code``: every line that build
    does not compile and every directive line blanked; ``literals``: the
    literals of the kept lines; ``macros``: the code and literals of the
    directive lines the MPI build reads (a ``#define`` it compiles), as
    their own Scanned with the same offsets."""
    code: str
    literals: list
    macros: Scanned


def mpi_view(sc: Scanned) -> View:
    lines = sc.code.split("\n")
    frames = []       # [mpi_known, taken, parent_active]
    active = True
    kept, kept_directive = [], []
    continued = False   # the line before ended a directive line with '\\'
    for line in lines:
        starts_directive = not continued and line.lstrip().startswith("#")
        directive = continued or starts_directive
        continued = directive and line.rstrip().endswith("\\")
        active_here = active
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
                active_here = parent
            elif kind == "endif" and frames:
                active = frames.pop()[2]
                active_here = active
        kept.append(active and not directive)
        kept_directive.append(active_here and directive)

    starts = [0]
    for line in lines[:-1]:
        starts.append(starts[-1] + len(line) + 1)

    def keep(which):
        out = [line if k else " " * len(line) for line, k in zip(lines, which)]
        lits = [(a, b, v) for a, b, v in sc.literals
                if which[bisect.bisect_right(starts, a) - 1]]
        return "\n".join(out), lits

    code, literals = keep(kept)
    mcode, mlits = keep(kept_directive)
    # A directive's own tokens (#define NAME, #include) are not code.
    mcode = re.sub(r"(?m)^[ \t]*#[ \t]*(?:define[ \t]+\w+(?:\([^)\n]*\))?|\w+)",
                   lambda m: " " * len(m.group(0)), mcode)
    return View(code, literals, Scanned(mcode, mlits))


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


def test_cases(code: str, literals: list) -> list:
    cases = []
    for m in re.finditer(r"\bTEST_CASE(?:_METHOD)?\s*\(", code):
        open_p = m.end() - 1
        close_p = _matching(code, open_p, "(", ")")
        if close_p < 0:
            continue
        brace = code.find("{", close_p)
        if brace < 0 or code[close_p + 1:brace].strip():
            continue
        end = _matching(code, brace, "{", "}")
        if end < 0:
            continue
        name = next((v for a, _, v in literals if open_p < a < close_p), "")
        cases.append(Case(name, _line_of(code, m.start()), brace + 1, end))
    return cases


def _line_of(text: str, offset: int) -> int:
    return text.count("\n", 0, offset) + 1


def _flat(s: str) -> str:
    return " ".join(s.split())


# --- uses of a class --------------------------------------------------------

_HOLDER = re.compile(r"\b(?:unique_ptr|shared_ptr|weak_ptr)\s*<\s*$")
_LOOK_BEHIND = 256   # characters read before a name: its qualifiers
_CV = re.compile(r"\s*(?:\b(?:const|volatile)\b\s*)*")


def _constructing_uses(code: str, name_re: str, qualifier_ok) -> list:
    """Offsets (start, end) of the uses of a class named by @p name_re
    other than as a pointer or reference type, a smart-pointer holder or a
    scope. @p qualifier_ok(q) says whether a use qualified by ``q::``
    (None: unqualified) names the class."""
    uses = []
    for m in re.finditer(r"\b(?:" + name_re + r")\b", code):
        a, b = m.start(), m.end()
        w0 = max(0, a - _LOOK_BEHIND)
        window = code[w0:a]               # what precedes the name
        before = window.rstrip()
        if before.endswith(".") or before.endswith("->"):
            continue                       # a member, not the class
        start = a
        q = re.search(r"([A-Za-z_]\w*)\s*::\s*$", window)
        if q:
            if not qualifier_ok(q.group(1)):
                continue
            # the whole qualified name: tadah::hpo::PcWorkerPool
            head = re.search(r"(?:(?:::\s*)?[A-Za-z_]\w*\s*::\s*)+$",
                             window)
            start = w0 + head.start()
        elif not qualifier_ok(None):   # unqualified, or ::name
            continue
        after = _CV.match(code, b).end()
        nxt = code[after:after + 2]
        if nxt[:1] in ("*", "&") or nxt == "::":
            continue
        if nxt[:1] == ">" and _HOLDER.search(
                code[max(0, start - _LOOK_BEHIND):start]):
            continue
        uses.append((start, after))
    return uses


def _pool_uses(code: str) -> list:
    uses = _constructing_uses(code, "PcWorkerPool", lambda q: True)
    uses += [(m.start(), m.end()) for m in
             re.finditer(r"\bpc_worker_pool_init\s*\(", code)]
    return sorted(uses)


def _lammps_uses(code: str) -> list:
    uses = _constructing_uses(
        code, "LAMMPS",
        lambda q: q is None or q == "LAMMPS_NS")
    uses += _constructing_uses(
        code, "LMP",
        lambda q: q is None or q == "LammpsRunner")
    uses += [(m.start(), m.end()) for m in
             re.finditer(r"\blammps_open\w*\s*\(", code)]
    return sorted(uses)


# --- PROCESS requests -------------------------------------------------------

_ESCAPED_BLANK = re.compile(r"\\[tnrvf]")
_CONFIG_LINE = re.compile(
    r"(?:^|\n)[ \t\r\v\f]*PC_ISOLATION[ \t\r\v\f]+PROCESS[ \t\r\v\f]*"
    r"(?:#[^\n]*)?(?:\n|$)")
_ENV_ASSIGNMENT = re.compile(r"\bTADAH_PC_ISOLATION\s*=\s*PROCESS\b")


def _request_value(code: str, a: int, b: int, value: str) -> bool:
    """Whether the literal at [a, b) with @p value requests PROCESS."""
    text = _ESCAPED_BLANK.sub(lambda m: {"n": "\n"}.get(m.group(0)[1], " "),
                              value)
    if _CONFIG_LINE.search(text) or _ENV_ASSIGNMENT.search(text):
        return True
    if text.strip() != "PROCESS":
        return False
    before = code[max(0, a - _LOOK_BEHIND):a].rstrip()
    after = code[b:b + _LOOK_BEHIND].lstrip()
    return not (before.endswith(("==", "!=")) or after.startswith(("==", "!=")))


def _blocks(code: str) -> list:
    """(open, close) of every brace pair that is not a namespace's or an
    extern "C" block's."""
    out, stack = [], []
    for k, ch in enumerate(code):
        if ch == "{":
            head = code[max(0, k - 200):k]
            transparent = bool(re.search(
                r"\bnamespace\b(?:\s*[A-Za-z_]\w*(?:\s*::\s*[A-Za-z_]\w*)*)?"
                r"\s*$|\bextern\s*\"\s*\"\s*$", head))
            stack.append((k, transparent))
        elif ch == "}" and stack:
            k0, transparent = stack.pop()
            if not transparent:
                out.append((k0, k))
    return out


def _refusal_exempt(code: str, literals: list, blocks: list) -> callable:
    """A predicate: whether an offset lies in a brace block whose innermost
    literals include the MPI build's refusal."""
    spans = []
    for a, _, v in literals:
        if REFUSAL_TEXT not in v:
            continue
        inner = [(o, c) for o, c in blocks if o < a < c]
        if inner:
            spans.append(max(inner))   # the innermost: the latest opening
    return lambda off: any(o < off < c for o, c in spans)


# Catch macros whose literals are names and messages, never a request.
_TEXT_MACROS = re.compile(
    r"\b(?:TEST_CASE(?:_METHOD)?|SCENARIO|(?:DYNAMIC_)?SECTION|"
    r"(?:UNSCOPED_)?INFO|WARN|FAIL(?:_CHECK)?|SUCCEED)\s*\(")


def _requests(code: str, literals: list) -> list:
    text_spans = []
    for m in _TEXT_MACROS.finditer(code):
        close = _matching(code, m.end() - 1, "(", ")")
        if close > 0:
            text_spans.append((m.end(), close))
    return [(a, b, v) for a, b, v in literals
            if not any(o <= a < c for o, c in text_spans)
            and _request_value(code, a, b, v)]


def _shown(text: str) -> str:
    """A literal's value as one short line."""
    flat = _flat(_ESCAPED_BLANK.sub(" ", text))
    return flat if len(flat) <= 80 else flat[:77] + "..."


def _code_line(code: str, off: int) -> str:
    a = code.rfind("\n", 0, off) + 1
    b = code.find("\n", off)
    line = _flat(code[a:] if b < 0 else code[a:b])
    return line if len(line) <= 70 else line[:67] + "..."


def check_text(text: str, where: str) -> tuple[list, int]:
    """The violations in the source @p text (named @p where in them) and
    the number of test cases of the MPI build's view."""
    problems = []
    whole = scan(text)
    for a, b in _lammps_uses(whole.code):
        problems.append(
            f"{where}:{_line_of(text, a)}: names the LAMMPS class other "
            f"than as a pointer, a reference or a smart-pointer holder "
            f"('{_code_line(whole.code, a)}'): create the instance with "
            "LammpsRunner::create_instance(tokens), which runs it on the "
            "build's communicator (hpo_lammps_comm(): MPI_COMM_SELF in the "
            "MPI build, where the serial build's 0 is a null communicator)")

    view = mpi_view(whole)
    cases = test_cases(view.code, view.literals)
    found = []      # (offset, what, code) of every pool and request
    for code, literals in ((view.code, view.literals),
                           (view.macros.code, view.macros.literals)):
        exempt = _refusal_exempt(code, literals, _blocks(code))
        for a, b in _pool_uses(code):
            found.append((a, f"builds a worker pool "
                             f"('{_code_line(code, a)}', line "
                             f"{_line_of(code, a)})"))
        for a, b, v in _requests(code, literals):
            if not exempt(a):
                found.append((a, "requests PC_ISOLATION PROCESS "
                                 f"(\"{_shown(v)}\", line "
                                 f"{_line_of(code, a)})"))
    found.sort()
    in_case = {}
    outside = []
    for off, what in found:
        case = next((c for c in cases
                     if c.body_start <= off < c.body_end), None)
        if case is None:
            outside.append((off, what))
        else:
            in_case.setdefault(id(case), (case, []))[1].append(what)
    for case, reasons in in_case.values():
        if re.match(r"\s*" + SKIP_MACRO + r"\s*\(\s*\)\s*;", view.code[
                case.body_start:]):
            continue
        problems.append(
            f"{where}:{case.line}: TEST_CASE \"{case.name}\" "
            f"{'; '.join(reasons)}, and the MPI build (TADAH_HPO_MPI) has "
            f"no PC_ISOLATION PROCESS workers: its first statement must be "
            f"{SKIP_MACRO}() (tests/pc_process_workers.h)")
    for off, what in outside:
        problems.append(
            f"{where}:{_line_of(text, off)}: {what} outside every TEST_CASE "
            "body in the MPI build's view (TADAH_HPO_MPI), where this check "
            "cannot tell which case reaches it: do it in the case, after "
            f"{SKIP_MACRO}() (a helper takes the mode or the pool from its "
            "caller), or compile it in the serial build only")
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
        for path in sorted(p for p in root.rglob("*")
                           if p.suffix in SOURCE_SUFFIXES and p.is_file()):
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
