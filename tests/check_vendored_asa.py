#!/usr/bin/env python3
"""The upstream files of the vendored ASA keep their bytes, upstream's CRLF
line endings included: the TADAH-PATCH hunks of asa.c are written in them,
so that a patch's diff shows only its hunks (external/asa/TADAH-PATCHES.md).

Usage: check_vendored_asa.py <HPO source dir> [--git=<git>]

Without --git: every line of each upstream file of external/asa (UPSTREAM
below) ends in CRLF, the last one too, and no CR stands alone. Prints each
file that breaks it, with the number of such lines and the first one, and
exits 1.

With --git=<git>: the repository's attributes keep those bytes under the
git settings that convert line endings. The files and every .gitattributes
on their path in the HPO tree are copied into a scratch repository and
committed as they are; then, under core.autocrlf=input, core.autocrlf=true
and a global attributes file (core.attributesFile) with '* text eol=lf',
`git add --renormalize` and `git add` of a copy with a CRLF line appended
must store each file's bytes unchanged. The user's and the system's git
configuration are not read. Prints each file and setting that changes them
and exits 1. Exits 77 (a CTest skip) when <git> is empty or cannot be run.
"""
import os
import subprocess
import sys
import tempfile
from pathlib import Path

ASA_DIR = Path("external/asa")
UPSTREAM = ["asa.c", "asa.h", "asa_usr_asa.h",
            "ASA-LICENSE", "ASA-NOTES", "ASA-README.txt"]
SETTINGS = [
    ("core.autocrlf=input", []),
    ("core.autocrlf=true", []),
    ("core.attributesFile with '* text eol=lf'", ["global-attributes"]),
]
SKIP = 77


def line_ending_problems(path):
    """[] when every line of @p path ends in CRLF and no CR stands alone,
    else one message naming the file."""
    data = path.read_bytes()
    lines = data.split(b"\n")
    tail = lines.pop()  # after the last LF: empty when the file ends in LF
    lf_only = [i + 1 for i, l in enumerate(lines) if not l.endswith(b"\r")]
    lone_cr = [i + 1 for i, l in enumerate(lines) if b"\r" in l[:-1]]
    problems = []
    if lf_only:
        problems.append(f"{len(lf_only)} line(s) end in LF alone, the first "
                        f"line {lf_only[0]}")
    if lone_cr:
        problems.append(f"{len(lone_cr)} line(s) hold a CR that ends no "
                        f"line, the first line {lone_cr[0]}")
    if tail:
        problems.append("the last line has no CRLF")
    if not problems:
        return []
    return [f"{path}: " + "; ".join(problems) +
            " (upstream's CRLF endings are lost: TADAH-PATCHES.md)"]


def check_bytes(hpo):
    problems = []
    for name in UPSTREAM:
        path = hpo / ASA_DIR / name
        if not path.is_file():
            problems.append(f"{path}: missing")
            continue
        problems.extend(line_ending_problems(path))
    return problems


def attribute_files(hpo):
    """The .gitattributes files on the path from the HPO tree to
    external/asa, relative to the tree."""
    out = []
    rel = Path(".")
    for part in [None] + list(ASA_DIR.parts):
        if part is not None:
            rel = rel / part
        f = hpo / rel / ".gitattributes"
        if f.is_file():
            out.append(rel / ".gitattributes")
    return out


def git_env(scratch):
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(HOME=str(scratch), XDG_CONFIG_HOME=str(scratch),
               GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull,
               GIT_ATTR_NOSYSTEM="1", LC_ALL="C")
    return env


def run_git(git, repo, env, *args):
    return subprocess.run([git, "-c", "user.name=check",
                           "-c", "user.email=check@localhost", *args],
                          cwd=repo, env=env, capture_output=True, check=True)


def index_blob(git, repo, env, rel):
    return run_git(git, repo, env, "cat-file", "blob", f":{rel}").stdout


def check_attributes(hpo, git):
    problems = []
    with tempfile.TemporaryDirectory() as tmp:
        scratch = Path(tmp)
        repo = scratch / "repo"
        env = git_env(scratch)
        (scratch / "global-attributes").write_bytes(b"* text eol=lf\n")
        repo.mkdir()
        run_git(git, repo, env, "init", "-q")
        copied = []
        for rel in attribute_files(hpo):
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            (repo / rel).write_bytes((hpo / rel).read_bytes())
            copied.append(str(rel))
        originals = {}
        for name in UPSTREAM:
            rel = (ASA_DIR / name).as_posix()
            (repo / rel).parent.mkdir(parents=True, exist_ok=True)
            originals[rel] = (hpo / rel).read_bytes()
            (repo / rel).write_bytes(originals[rel])
        # Committed as they are, as a checkout that converts nothing has them.
        run_git(git, repo, env, "-c", "core.autocrlf=false", "add", "-A")
        run_git(git, repo, env, "-c", "core.autocrlf=false",
                "commit", "-q", "-m", "vendored")
        for label, attr in SETTINGS:
            if attr:
                opt = f"core.attributesFile={scratch / attr[0]}"
            else:
                opt = label
            for rel, want in originals.items():
                run_git(git, repo, env, "-c", opt,
                        "add", "--renormalize", "--", rel)
                if index_blob(git, repo, env, rel) != want:
                    problems.append(
                        f"{rel}: `git add --renormalize` under {label} changes "
                        f"its bytes")
                run_git(git, repo, env, "reset", "-q")
                edited = want + b"/* edit */\r\n"
                (repo / rel).write_bytes(edited)
                run_git(git, repo, env, "-c", opt, "add", "--", rel)
                if index_blob(git, repo, env, rel) != edited:
                    problems.append(
                        f"{rel}: `git add` of an edit under {label} changes "
                        f"its bytes")
                run_git(git, repo, env, "reset", "-q")
                (repo / rel).write_bytes(want)
        if problems:
            problems.append(
                "attributes read: " + (", ".join(copied) if copied else
                                       "none (no .gitattributes on the path "
                                       "to external/asa)") +
                "; an attribute '-text' for these files keeps their bytes")
    return problems


def main(argv):
    git = None
    args = []
    for a in argv[1:]:
        if a.startswith("--git="):
            git = a[len("--git="):]
        else:
            args.append(a)
    if len(args) != 1:
        print(__doc__, file=sys.stderr)
        return 2
    hpo = Path(args[0])
    if git is None:
        problems = check_bytes(hpo)
        what = "line endings of the vendored ASA"
    else:
        if not git:
            print("no git executable was found at configure time (FindGit): "
                  "the attributes cannot be checked")
            return SKIP
        try:
            subprocess.run([git, "--version"], capture_output=True,
                           check=True)
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"{git} cannot be run ({e}): the attributes cannot be "
                  "checked")
            return SKIP
        problems = check_attributes(hpo, git)
        what = "git attributes of the vendored ASA"
    for p in problems:
        print(p)
    if problems:
        print(f"FAILED: {what}")
        return 1
    print(f"OK: {what} ({len(UPSTREAM)} files)")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
