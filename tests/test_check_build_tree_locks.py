#!/usr/bin/env python3
"""Unit tests of tests/check_build_tree_locks.py on synthetic output of
`ctest --show-only=json-v1`: which commands build in the build tree, which
entries the source directory defines (through the backtrace) and which of
them lack the lock."""

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import check_build_tree_locks as locks  # noqa: E402


class Locks(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        root = pathlib.Path(self._tmp.name).resolve()
        self.build = root / "build"
        self.source = root / "src" / "HPO"
        self.build.mkdir()
        self.source.mkdir(parents=True)

    def tearDown(self):
        self._tmp.cleanup()

    def doc(self, tests):
        """A json-v1 document: each test (name, command, properties, file
        that defines it, called from the file of the parent frame)."""
        files = [str(self.source / "tests" / "CMakeLists.txt"),
                 str(self.source.parent / "MLIP" / "CMakeLists.txt"),
                 str(self.source.parent / "Other" / "CMakeLists.txt")]
        nodes, out = [], []
        for name, command, props, chain in tests:
            parent = None
            for f in chain:
                node = {"file": f}
                if parent is not None:
                    node["parent"] = parent
                nodes.append(node)
                parent = len(nodes) - 1
            out.append({"name": name, "command": command,
                        "backtrace": parent,
                        "properties": [{"name": k, "value": v}
                                       for k, v in props.items()]})
        return {"backtraceGraph": {"files": files, "nodes": nodes,
                                   "commands": []},
                "tests": out, "kind": "ctestInfo",
                "version": {"major": 1, "minor": 0}}

    # Pins: an entry of the source directory (a frame of its backtrace
    # is a file under it, even when another repository's add_test()
    # wrapper is the innermost frame) whose command builds in the tree
    # (--build <dir> or -DBUILD_DIR=<dir>) is reported without
    # RESOURCE_LOCK tadah_header_check or RUN_SERIAL, and not with
    # either; an entry that builds elsewhere or another directory's entry
    # is not.
    def test_unlocked(self):
        b = str(self.build)
        doc = self.doc([
            ("bare", ["cmake", f"-DBUILD_DIR={b}", "-P", "x.cmake"], {},
             [0, 1]),
            ("built", ["cmake", "--build", b, "--target", "t"], {}, [0]),
            ("locked", ["cmake", "--build", b],
             {"RESOURCE_LOCK": ["other", "tadah_header_check"]}, [0]),
            ("serial", ["cmake", f"-DBUILD_DIR={b}/."],
             {"RUN_SERIAL": True}, [0]),
            ("elsewhere", ["cmake", "--build", str(self.source)], {}, [0]),
            ("other_lock", ["cmake", "--build", b],
             {"RESOURCE_LOCK": ["other"]}, [0]),
            ("not_ours", ["cmake", "--build", b], {}, [2, 1]),
            ("no_command", [], {}, [0]),
        ])
        self.assertEqual(locks.unlocked(doc, self.build, self.source),
                         ["bare", "built", "other_lock"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
