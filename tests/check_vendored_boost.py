#!/usr/bin/env python3
"""Every Boost header HPO includes must come from the Boost that Tadah
vendors (MLIP's external/boost, which HPO compiles against through
tadah.mlip), not from a system Boost that happens to be installed.

Usage: check_vendored_boost.py <HPO source dir> <include dir>[:<include dir>...]
                               [--cxx <compiler>]

Scans HPO's src/, include/ and tests/ for '#include <boost/...>' and
'#include "boost/..."' and checks that each header exists in one of the
include directories. With --cxx, it also follows each header's own includes
as that compiler resolves them, in the current environment (CPATH
included): every header under a 'boost' directory must come from the
include directories. Prints each header that is missing or comes from
elsewhere, with the files that include it, and exits 1 if there is any.
"""
import os
import re
import subprocess
import sys
from pathlib import Path

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](boost/[^>"]+)[>"]')


def dependencies(make_rule):
    """The prerequisites of a 'cxx -M' rule."""
    text = make_rule.replace("\\\n", " ").replace("\\ ", "\0")
    words = text.split()
    if words and words[0].endswith(":"):
        words = words[1:]
    return [w.replace("\0", " ") for w in words]


def under(path, dirs):
    p = Path(os.path.realpath(path))
    return any(p.is_relative_to(Path(os.path.realpath(d))) for d in dirs)


def transitive_problems(cxx, header, dirs):
    """What is wrong with the headers 'header' pulls in, as 'cxx' resolves
    them: [] when every Boost header comes from 'dirs'."""
    cmd = [cxx, "-std=c++17", "-M", "-x", "c++", "-"] + [f"-I{d}" for d in dirs]
    r = subprocess.run(cmd, input=f"#include <{header}>\n", capture_output=True,
                       text=True)
    if r.returncode != 0:
        errors = [l for l in r.stderr.splitlines() if "error" in l]
        why = errors[0].strip() if errors else r.stderr.strip()
        return [f"{header}: cannot be preprocessed with the vendored Boost: {why}"]
    return [f"{header}: pulls in {dep}, which is not in the vendored Boost "
            f"({', '.join(str(d) for d in dirs)})"
            for dep in dependencies(r.stdout)
            if "boost" in Path(dep).parts and not under(dep, dirs)]


def main(argv):
    argv = list(argv)
    cxx = None
    if "--cxx" in argv:
        i = argv.index("--cxx")
        if i + 1 >= len(argv):
            print(__doc__)
            return 2
        cxx = argv[i + 1]
        del argv[i:i + 2]
    if len(argv) != 3:
        print(__doc__)
        return 2
    root = Path(argv[1])
    dirs = [Path(d) for d in argv[2].split(":") if d]
    wanted = {}
    for sub in ("src", "include", "tests"):
        for path in sorted((root / sub).rglob("*")):
            if path.suffix not in {".h", ".hpp", ".cpp", ".cc", ".hxx", ".cxx"}:
                continue
            for line in path.read_text(errors="replace").splitlines():
                m = INCLUDE.match(line)
                if m:
                    wanted.setdefault(m.group(1), []).append(str(path.relative_to(root)))
    missing = {h: files for h, files in wanted.items()
               if not any((d / h).is_file() for d in dirs)}
    for header, files in sorted(missing.items()):
        print(f"{header}: not in the vendored Boost ({', '.join(str(d) for d in dirs)}); "
              f"included by {', '.join(sorted(set(files)))}")
    print(f"{len(wanted)} Boost headers included by HPO, {len(missing)} missing "
          f"from the vendored Boost")
    elsewhere = 0
    if cxx:
        for header in sorted(set(wanted) - set(missing)):
            problems = transitive_problems(cxx, header, dirs)
            if problems:
                elsewhere += 1
                files = ", ".join(sorted(set(wanted[header])))
                for p in problems:
                    print(f"{p}; included by {files}")
        print(f"{len(wanted) - len(missing)} Boost headers followed with {cxx}, "
              f"{elsewhere} pulling in a header that is missing or comes from "
              f"elsewhere")
    return 1 if missing or elsewhere else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
