#!/usr/bin/env python3
"""Multi-rank `tadah hpo` end to end (a TADAH_HPO_MPI build).

Runs the built `tadah` under the MPI launcher of the MPI it links
(MPIEXEC_EXECUTABLE), on 1 and on 2 ranks, in a scratch copy of
HPO/examples/common and of example 27's DM_REAM seed, and checks one case:

  restart_refusal   ANNEAL_PT with an active RESTART block.
                    Pins: on 2 ranks the launch fails on every rank before
                    any evaluation (exit status 1; "multi-rank HPO does not
                    support RESTART (AFTER_STAGNATION 4, MAX 1 on 2 ranks)"
                    on stderr and on an "ERROR     startup:" line of
                    hpo_run.log.r0 and hpo_run.log.r1; no loss.tadah.r*);
                    on 1 rank the same configuration runs (status 0,
                    loss.tadah written).
                    Before the fix: passes as well (a pin of the refusal,
                    tested at spec level only so far).
  gradient_refusal  The finite-difference counters: the finite differences
                    run only in gradient-based searches, which a multi-rank
                    launch refuses, so no multi-rank run can count them.
                    Pins: example 01 (DLIB BFGS, numerical gradients) on 2
                    ranks fails on every rank before any evaluation
                    ("multi-rank HPO supports only STRATEGY TYPE PLAIN with
                    LIB TADAH and ALGO ANNEAL_PT or ANNEAL_PA; got DLIB:BFGS
                    on 2 ranks"); on 1 rank of the MPI build it runs and its
                    DIAGNOSTICS count one-sided differences ("One-sided
                    differences : N", N > 0).
                    Before the fix: passes as well (a pin).
  eamfast           The LAMMPS fast kernels on several ranks: example 27's
                    D2_Knot5 + DM_REAM seed (EAMFAST is on by default) under
                    ANNEAL_PT, PC_LAMMPS with -log lmp.log.
                    Pins: on 1 and 2 ranks every rank's journal ends
                    "TADAH:ANNEAL_PT: MAXEVAL reached"; every evaluation
                    instance's LAMMPS log (lmp.log.t<slot>, on 2 ranks
                    lmp.log.r<rank>.t<slot>, both ranks) states "Tadah!
                    EAM-collapse fast path ON" and none "collapse fast path
                    unavailable"; the evaluated set -- (slot, parameters,
                    loss) of every row of params.tadah and loss.tadah, the
                    2 ranks' rows together -- is the 1-rank run's; the global
                    best_pot.tadah has the same HPO_BEST_LOSS and the same
                    key lines; the log of the instance tadah creates at
                    startup to read the script's variables is lmp.log on 1
                    rank and lmp.log.r0 / lmp.log.r1 on 2 ranks, with no
                    lmp.log.
                    Before the fix: on 2 ranks both ranks' startup instances
                    wrote the one file lmp.log (no lmp.log.r0, lmp.log.r1);
                    the rest passes as well (a pin).
  rhogates          The RHOFORM density gates and the density report on
                    several ranks: the seed with OPTIM AMPGRIDMB boxes, once
                    with RHOGATE and once with RHOBARGATE.
                    Pins: the 2 ranks' "of which rho-sign gate rejects" /
                    "of which lattice-rhobar gate rejects" counts add up to
                    the 1-rank run's, which is > 0; every journal's POTENTIAL
                    SETUP density rows ("Density form", "Density rho(r), -c
                    pot", "Density rhobar, -c pot") are the 1-rank run's; the
                    RESULT density rows of the rank whose best is the global
                    best are the 1-rank RESULT's but for the rank-local
                    evaluation number; the evaluated set is the 1-rank
                    run's.
                    Before the fix: passes as well (a pin).

Exit status 0 when the case passes, 1 when it fails, 77 (Skipped) when the
build has no `tadah` executable. TADAH_PC_ISOLATION is removed from the
runs' environment: the MPI build has PC_ISOLATION THREAD only, and an
inherited PROCESS would refuse every run.
"""
import argparse
import collections
import os
import pathlib
import re
import shlex
import shutil
import subprocess
import sys

SKIP = 77

PT_HEAD = """OPTIMIZER
  STRATEGY
    TYPE PLAIN
    LIB TADAH
    ALGO ANNEAL_PT
    THREADS 2
    SEED 77
    MAXEVAL 40
    PARAM CORANA ON
    PARAM NS 2
    PARAM REPLICAS 4
    PARAM T_HOT 1.0
    PARAM T_COLD 0.01
{restart}  ENDSTRATEGY
ENDOPTIMIZER

LOSS L2
FAILSCORE 1e16
"""

RESTART = """    RESTART
      AFTER_STAGNATION 4
      MAX 1
      STRATEGY RANDOM 1
    ENDRESTART
"""

ECOH_27 = ("PC_LAMMPS --script ../common/lammps/Ecoh.in --hotvar hot_a0 2.5 "
           "4.5 3.304 --varloss Ecoh 8.10 800 -log lmp.log -screen none\n")

ECOH_COMMON = ("PC_LAMMPS --script ../common/lammps/Ecoh.in --hotvar hot_a0 "
               "2.5 4.5 3.302 --varloss Ecoh 8.10 800\n")

TAIL = "\nOUTPUT 1\nBOUTPUT 1\n"


class Failure(Exception):
    pass


def check(cond, what):
    if not cond:
        raise Failure(what)


class Runner:
    def __init__(self, args):
        self.mpiexec = args.mpiexec
        self.np_flag = args.numproc_flag
        self.preflags = shlex.split(args.preflags or "")
        self.postflags = shlex.split(args.postflags or "")
        self.tadah = args.tadah
        self.examples = pathlib.Path(args.examples).resolve()
        self.root = pathlib.Path(args.workdir).resolve() / args.case
        if self.root.exists():
            shutil.rmtree(self.root)
        self.root.mkdir(parents=True)
        shutil.copytree(self.examples / "common", self.root / "common")
        shutil.copy(self.examples / "27_dm_ream_embf0" / "ta2_refit_seed.tadah",
                    self.root / "ta2_refit_seed.tadah")
        self.env = dict(os.environ)
        self.env.pop("TADAH_PC_ISOLATION", None)
        self.env["OMP_NUM_THREADS"] = "1"

    def run(self, name, nranks, config, tadah_args):
        """Run `tadah hpo` in <root>/<name>_n<nranks>; returns the run
        directory and the completed process."""
        d = self.root / f"{name}_n{nranks}"
        d.mkdir()
        (d / "config.hpo").write_text(config)
        cmd = ([self.mpiexec, self.np_flag, str(nranks)] + self.preflags +
               [self.tadah] + self.postflags +
               ["hpo", "--hpotarget", "config.hpo"] + tadah_args)
        p = subprocess.run(cmd, cwd=d, env=self.env, capture_output=True,
                           text=True, timeout=240)
        (d / "run.stdout").write_text(p.stdout)
        (d / "run.stderr").write_text(p.stderr)
        print(f"[{name} n={nranks}] exit {p.returncode}: {' '.join(cmd)}")
        return d, p


def journals(d, nranks):
    if nranks == 1:
        return [d / "hpo_run.log"]
    return [d / f"hpo_run.log.r{r}" for r in range(nranks)]


def suffixes(nranks):
    return [""] if nranks == 1 else [f".r{r}" for r in range(nranks)]


def journal_status(text):
    m = re.search(r"Optimiser status\s*: (.*)", text)
    return m.group(1).strip() if m else ""


def table(path):
    rows = {}
    for line in path.read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tok = line.split()
        rows[tok[0]] = tok[1:]
    return rows


def evaluated_set(d, nranks):
    """(slot and parameters, total loss) of every evaluation, as a
    multiset; the ranks' rows together."""
    out = collections.Counter()
    for s in suffixes(nranks):
        params = table(d / f"params.tadah{s}")
        loss = table(d / f"loss.tadah{s}")
        check(params.keys() == loss.keys(),
              f"{d.name}: params.tadah{s} and loss.tadah{s} list different "
              "evaluations")
        for it, values in params.items():
            out[(tuple(values), loss[it][-1])] += 1
    return out


def best_pot(d):
    text = (d / "best_pot.tadah").read_text()
    m = re.search(r"^# HPO_BEST_LOSS (\S+)$", text, re.M)
    check(m is not None, f"{d.name}: best_pot.tadah has no HPO_BEST_LOSS")
    keys = [l for l in text.splitlines() if l.strip() and not l.startswith("#")]
    return m.group(1), keys


def check_completed(d, nranks):
    for j in journals(d, nranks):
        check(j.exists(), f"{d.name}: {j.name} is missing")
        status = journal_status(j.read_text())
        check(status == "TADAH:ANNEAL_PT: MAXEVAL reached",
              f"{d.name}: {j.name} ends '{status}', not "
              "'TADAH:ANNEAL_PT: MAXEVAL reached'")


def check_refused(d, p, nranks, message):
    check(p.returncode == 1,
          f"{d.name}: exit status {p.returncode}, expected 1 (refused)")
    check(message in p.stderr, f"{d.name}: stderr lacks '{message}'")
    for j in journals(d, nranks):
        check(j.exists(), f"{d.name}: {j.name} is missing")
        check("ERROR     startup: " + message in j.read_text(),
              f"{d.name}: {j.name} has no 'ERROR     startup: {message}' "
              "line")
    for s in suffixes(nranks):
        check(not (d / f"loss.tadah{s}").exists(),
              f"{d.name}: loss.tadah{s} exists: an evaluation ran")


def case_restart_refusal(r):
    config = PT_HEAD.format(restart=RESTART) + ECOH_27 + \
        "\nOPTIM CEMBFUNC (1) -2.19 -2.15\n" + TAIL
    args = ["-c", "../ta2_refit_seed.tadah"]
    d, p = r.run("restart", 2, config, args)
    check_refused(d, p, 2,
                  "multi-rank HPO does not support RESTART (AFTER_STAGNATION "
                  "4, MAX 1 on 2 ranks): stagnation is tracked per rank; run "
                  "on one rank or remove RESTART")
    d, p = r.run("restart", 1, config, args)
    check(p.returncode == 0, f"{d.name}: exit status {p.returncode}, "
          "expected 0 (RESTART runs on one rank)")
    check((d / "loss.tadah").exists(), f"{d.name}: no loss.tadah")


def case_gradient_refusal(r):
    config = (r.examples / "01_minimal" / "config.hpo").read_text()
    args = ["--validation", "../common/config.val",
            "-c", "../common/config.tadah"]
    d, p = r.run("bfgs", 2, config, args)
    check_refused(d, p, 2,
                  "multi-rank HPO supports only STRATEGY TYPE PLAIN with LIB "
                  "TADAH and ALGO ANNEAL_PT or ANNEAL_PA; got DLIB:BFGS on 2 "
                  "ranks")
    d, p = r.run("bfgs", 1, config, args)
    check(p.returncode == 0, f"{d.name}: exit status {p.returncode}")
    m = re.search(r"One-sided differences\s*: (\d+)",
                  (d / "hpo_run.log").read_text())
    check(m is not None and int(m.group(1)) > 0,
          f"{d.name}: DIAGNOSTICS count no one-sided differences")


def case_eamfast(r):
    config = PT_HEAD.format(restart="") + ECOH_27 + \
        "\nOPTIM CEMBFUNC (1) -2.19 -2.15\nOPTIM CEMBFUNC (4) -0.1665 -0.1630\n" \
        + TAIL
    args = ["-c", "../ta2_refit_seed.tadah"]
    runs = {}
    for n in (1, 2):
        d, p = r.run("eamfast", n, config, args)
        check(p.returncode == 0, f"{d.name}: exit status {p.returncode}")
        check_completed(d, n)
        for s in suffixes(n):
            logs = sorted(d.glob(f"lmp.log{s}.t*"))
            check(logs, f"{d.name}: no lmp.log{s}.t<slot> evaluation log")
            for lg in logs:
                text = lg.read_text()
                check("Tadah! EAM-collapse fast path ON" in text,
                      f"{d.name}: {lg.name} does not state the EAM-collapse "
                      "fast path")
                check("collapse fast path unavailable" not in text,
                      f"{d.name}: {lg.name} fell back to the generic kernel")
        startup = ["lmp.log"] if n == 1 else [f"lmp.log.r{q}" for q in range(n)]
        for name in startup:
            check((d / name).exists(),
                  f"{d.name}: no {name} (the log of the startup instance)")
        if n > 1:
            check(not (d / "lmp.log").exists(),
                  f"{d.name}: lmp.log exists: the ranks' startup instances "
                  "shared one log file")
        runs[n] = d
    check(evaluated_set(runs[1], 1) == evaluated_set(runs[2], 2),
          "eamfast: the 2-rank run evaluated another set than the 1-rank run")
    check(best_pot(runs[1]) == best_pot(runs[2]),
          "eamfast: the global best_pot.tadah differs between 1 and 2 ranks")


def density_rows(text, section):
    """The 'Density ...' rows of a journal section."""
    m = re.search(r"^-- " + section + r" -+\n(.*?)(?=^-- |\Z)", text,
                  re.M | re.S)
    check(m is not None, f"no {section} section")
    return [l.strip() for l in m.group(1).splitlines()
            if l.strip().startswith("Density")]


def gate_rejects(text, label):
    m = re.search(r"of which " + label + r" gate rejects: (\d+)", text)
    return int(m.group(1)) if m else 0


def best_loss(text):
    m = re.search(r"Best total loss\s*: (\S+)", text)
    check(m is not None, "a journal has no 'Best total loss'")
    return float(m.group(1))


def case_rhogates(r):
    boxes = ("\nOPTIM CEMBFUNC (1) -2.19 -2.15\n"
             "OPTIM AMPGRIDMB (5) -0.6 -0.05\n"
             "OPTIM AMPGRIDMB (7) -0.3 0.3\n")
    args = ["-c", "../ta2_refit_seed.tadah"]
    for gate, label in (("RHOGATE 1e-6\n", "rho-sign"),
                        ("RHOBARGATE bcc 3.304 0.9 1.1\n", "lattice-rhobar")):
        config = PT_HEAD.format(restart="") + ECOH_27 + boxes + gate + TAIL
        name = label.replace("-", "_")
        runs = {}
        for n in (1, 2):
            d, p = r.run(name, n, config, args)
            check(p.returncode == 0, f"{d.name}: exit status {p.returncode}")
            check_completed(d, n)
            runs[n] = d
        one = (runs[1] / "hpo_run.log").read_text()
        two = [j.read_text() for j in journals(runs[2], 2)]
        n_one = gate_rejects(one, label)
        check(n_one > 0, f"{name}: the 1-rank run has no {label} gate reject")
        n_two = sum(gate_rejects(t, label) for t in two)
        check(n_two == n_one,
              f"{name}: the 2 ranks count {n_two} {label} gate rejects, the "
              f"1-rank run {n_one}")
        setup = density_rows(one, "POTENTIAL SETUP")
        check(len(setup) >= 2, f"{name}: no density report in POTENTIAL SETUP")
        for q, t in enumerate(two):
            check(density_rows(t, "POTENTIAL SETUP") == setup,
                  f"{name}: rank {q}'s POTENTIAL SETUP density rows differ "
                  "from the 1-rank run's")
        owner = min(range(2), key=lambda q: best_loss(two[q]))
        drop_eval = lambda rows: [re.sub(r"\(eval \d+\)", "(eval N)", x)
                                  for x in rows]
        result = drop_eval(density_rows(one, "RESULT"))
        check(result, f"{name}: no density rows in RESULT")
        check(drop_eval(density_rows(two[owner], "RESULT")) == result,
              f"{name}: the RESULT density rows of rank {owner} (the global "
              "best) differ from the 1-rank run's")
        check(evaluated_set(runs[1], 1) == evaluated_set(runs[2], 2),
              f"{name}: the 2-rank run evaluated another set than the 1-rank "
              "run")


CASES = {
    "restart_refusal": case_restart_refusal,
    "gradient_refusal": case_gradient_refusal,
    "eamfast": case_eamfast,
    "rhogates": case_rhogates,
}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", required=True, choices=sorted(CASES))
    ap.add_argument("--mpiexec", required=True)
    ap.add_argument("--numproc-flag", default="-n")
    ap.add_argument("--preflags", default="")
    ap.add_argument("--postflags", default="")
    ap.add_argument("--tadah", default="")
    ap.add_argument("--examples", required=True)
    ap.add_argument("--workdir", required=True)
    args = ap.parse_args()
    if not args.tadah:
        print("TADAH-SKIP: this build has no tadah executable")
        return SKIP
    try:
        CASES[args.case](Runner(args))
    except (Failure, subprocess.TimeoutExpired) as e:
        print(f"FAILED ({args.case}): {e}", file=sys.stderr)
        return 1
    print(f"PASSED ({args.case})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
