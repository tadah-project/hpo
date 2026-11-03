#!/usr/bin/env python3
"""The CTest entries of HPO that build in the build tree they run in hold
its lock: RESOURCE_LOCK tadah_header_check, which the header checks of
every repository (cmake-tools, TadahTesting.cmake) and the other entries
that build in the tree hold, or RUN_SERIAL.

    check_build_tree_locks.py --ctest CTEST --build-dir DIR --source DIR

An entry builds in the tree when its command runs `--build <DIR>` or
passes -DBUILD_DIR=<DIR>, DIR the build tree (--build-dir); it is an
entry of HPO when a CMake file under --source defines it (the backtrace
of `ctest --show-only=json-v1`). Two such entries run at once would build
one tree together, and one that brings a library up to date while other
entries run changes the files those entries load. Exit status 1 names
each entry without the lock; 2 for a usage error or a ctest that fails.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import subprocess
import sys

LOCK = "tadah_header_check"


def builds_in(command: list[str], build: pathlib.Path) -> bool:
    """Whether a test command builds in the tree `build`."""
    def same(arg: str) -> bool:
        try:
            return pathlib.Path(arg).resolve() == build
        except (OSError, ValueError):
            return False
    for i, arg in enumerate(command):
        if arg == "--build" and i + 1 < len(command) and same(command[i + 1]):
            return True
        if arg.startswith("-DBUILD_DIR=") and same(arg[len("-DBUILD_DIR="):]):
            return True
    return False


def defined_under(doc: dict, test: dict, source: pathlib.Path) -> bool:
    """Whether a CMake file under `source` is in the backtrace of `test`."""
    graph = doc.get("backtraceGraph", {})
    nodes, files = graph.get("nodes", []), graph.get("files", [])
    node = test.get("backtrace")
    while node is not None and 0 <= node < len(nodes):
        f = pathlib.Path(files[nodes[node]["file"]]).resolve()
        if f == source or source in f.parents:
            return True
        node = nodes[node].get("parent")
    return False


def unlocked(doc: dict, build: pathlib.Path,
             source: pathlib.Path) -> list[str]:
    """The entries of `doc` defined under `source` that build in `build`
    and hold neither the lock nor RUN_SERIAL."""
    out = []
    for t in doc.get("tests", []):
        if not builds_in(t.get("command") or [], build):
            continue
        if not defined_under(doc, t, source):
            continue
        props = {p["name"]: p["value"] for p in t.get("properties", [])}
        if props.get("RUN_SERIAL") is True:
            continue
        if LOCK in (props.get("RESOURCE_LOCK") or []):
            continue
        out.append(t["name"])
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--ctest", required=True)
    ap.add_argument("--build-dir", required=True, type=pathlib.Path)
    ap.add_argument("--source", required=True, type=pathlib.Path)
    a = ap.parse_args(argv)
    build, source = a.build_dir.resolve(), a.source.resolve()
    proc = subprocess.run([a.ctest, "--show-only=json-v1"], cwd=build,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"check_build_tree_locks.py: ctest --show-only=json-v1 in "
              f"{build} failed (exit {proc.returncode}): "
              f"{proc.stderr.strip()}")
        return 2
    names = unlocked(json.loads(proc.stdout), build, source)
    for n in names:
        print(f"{n}: builds in the build tree {build} and holds neither "
              f"RESOURCE_LOCK {LOCK} nor RUN_SERIAL")
    if names:
        return 1
    print(f"check_build_tree_locks.py: OK: every entry of {source} that "
          f"builds in {build} holds RESOURCE_LOCK {LOCK} or RUN_SERIAL")
    return 0


if __name__ == "__main__":
    sys.exit(main())
