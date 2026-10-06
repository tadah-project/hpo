#!/usr/bin/env python3
"""Every Boost header HPO includes must come from the Boost that Tadah
vendors (MLIP's external/boost, which HPO compiles against through
tadah.mlip), not from a system Boost that happens to be installed.

Usage: check_vendored_boost.py <HPO source dir> <include dir>[:<include dir>...]

Scans HPO's src/, include/ and tests/ for '#include <boost/...>' and
'#include "boost/..."' and checks that each header exists in one of the
include directories. Prints each missing header with the files that include
it and exits 1 if any is missing.
"""
import re
import sys
from pathlib import Path

INCLUDE = re.compile(r'^\s*#\s*include\s*[<"](boost/[^>"]+)[>"]')


def main(argv):
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
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
