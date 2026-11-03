#!/usr/bin/env python3
"""The configure messages of HPO's CMake files (CTest hpo_cmake_messages).

Every message(FATAL_ERROR ...), message(SEND_ERROR ...), message(WARNING
...) and message(AUTHOR_WARNING ...) of HPO's CMakeLists.txt files and of
its cmake/*.cmake files states what the configure found and what it needs,
without addressing the reader: none of their text has the words you, your,
yours, yourself or please (any case), and none begins a sentence with an
instruction to the reader ("Point TADAH_LAMMPS_LIB at ..."). The words of a
message are those of its quoted strings; ${...} references and the words
outside the quotes (FATAL_ERROR) are not text.

The case on synthetic text shows that the extraction finds the words of a
message of several quoted strings and lines, and that a STATUS message is
not checked (it reports, it refuses nothing).
"""
from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

KINDS = ("FATAL_ERROR", "SEND_ERROR", "WARNING", "AUTHOR_WARNING")
WORDS = re.compile(r"\b(you|your|yours|yourself|please)\b", re.IGNORECASE)
# A sentence that opens with an instruction: the verb in the base form at
# the start of the text or after a full stop.
IMPERATIVE = re.compile(r"(?:^|[.!?]\s+)(Point|Set|Use|Rebuild|Link|Run|"
                        r"Add|Remove|Pass|Install|Reconfigure)\b")


def cmake_files(root: Path) -> list[Path]:
    """HPO's CMakeLists.txt files and cmake/*.cmake, not those of external/
    or of a build tree."""
    out = []
    for p in sorted(root.rglob("CMakeLists.txt")):
        rel = p.relative_to(root).parts
        if rel[0] in ("external",) or rel[0].startswith("build"):
            continue
        out.append(p)
    out.extend(sorted((root / "cmake").glob("*.cmake")))
    return out


def messages(text: str) -> list[tuple[int, str, str]]:
    """(line, kind, text) of each message() call of a kind in KINDS: the
    text is its quoted strings joined, ${...} references removed."""
    out = []
    for m in re.finditer(r"\bmessage\s*\(\s*([A-Z_]+)", text):
        kind = m.group(1)
        # The call ends at the parenthesis that balances the opening one,
        # outside quotes.
        i, depth, quoted = m.end(), 1, False
        body_start = i
        while i < len(text) and depth:
            c = text[i]
            if c == "\\" and quoted:
                i += 2
                continue
            if c == '"':
                quoted = not quoted
            elif not quoted:
                if c == "(":
                    depth += 1
                elif c == ")":
                    depth -= 1
            i += 1
        if kind not in KINDS:
            continue
        body = text[body_start:i - 1]
        strings = re.findall(r'"((?:[^"\\]|\\.)*)"', body, re.DOTALL)
        words = " ".join(re.sub(r"\$\{[^}]*\}", "", s) for s in strings)
        words = re.sub(r"\\n", " ", words)
        words = " ".join(words.split())
        out.append((text.count("\n", 0, m.start()) + 1, kind, words))
    return out


def findings(path: Path) -> list[str]:
    out = []
    for line, kind, words in messages(path.read_text()):
        for w in WORDS.finditer(words):
            out.append(f"{path.relative_to(ROOT)}:{line}: message({kind}) "
                       f"has '{w.group(0)}': {words}")
        for w in IMPERATIVE.finditer(words):
            out.append(f"{path.relative_to(ROOT)}:{line}: message({kind}) "
                       f"instructs the reader ('{w.group(1)}'): {words}")
    return out


class CMakeMessages(unittest.TestCase):
    def test_extraction(self):
        text = ('message(STATUS "Please wait")\n'
                'message(FATAL_ERROR\n  "The lib ${X} is static. "\n'
                '  "Please link your shared one.")\n'
                'message(WARNING "A (bracketed) note. Point X at Y.")\n')
        got = messages(text)
        self.assertEqual([(k, w) for _, k, w in got],
                         [("FATAL_ERROR",
                           "The lib is static. Please link your shared one."),
                          ("WARNING", "A (bracketed) note. Point X at Y.")])
        self.assertEqual(got[0][0], 2)

    def test_hpo_messages_address_no_reader(self):
        bad = []
        for p in cmake_files(ROOT):
            bad.extend(findings(p))
        self.assertTrue(cmake_files(ROOT))
        self.assertEqual(bad, [], "\n" + "\n".join(bad))


if __name__ == "__main__":
    sys.exit(unittest.main())
