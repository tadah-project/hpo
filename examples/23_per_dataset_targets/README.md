# 23 -- per-dataset validation targets

Example 01 without `PC_LAMMPS`, validated on two datasets, each with its
own target and weight.

What this shows:
- A local `config.val` with two `DBFILE` lines (the elastic set and the
  bcc equation-of-state set from `../common/data/`); `common/run.sh`
  uses it in place of `../common/config.val` and prints which file it
  uses.
- `PC_ERMSE (1) 0 1e-4` and `PC_ERMSE (2) 0 1e-3`: the index names the
  `DBFILE` line of `config.val`. The equation-of-state errors are about
  2.5x smaller than the elastic ones, so that set is weighted 10x, which
  makes its term comparable to the elastic one in the L2 loss (about
  1.5x it at the optimum).
- One column per dataset: `L(ERMSE)[1]`, `L(ERMSE)[2]` in `loss.tadah`
  and `ERMSE[1]`, `ERMSE[2]` in `stats.tadah`. With indexed targets
  `L(ERMSE)[avg]` is the average of the dataset losses (the journal's
  loss section says `indexed (average of dataset losses)`).
- `--skip-above 5` on dataset 2 only: a per-dataset bad-potential
  filter. A candidate whose equation-of-state error exceeds 5 meV/atom
  scores `FAILSCORE` and skips the rest of its evaluation; the journal
  warns `skipping downstream (obs+LAMMPS) -- ermse[2] ... > skip-above=5`.
  In this box that is `RCUT2B` below about 4.24 A (the error is
  6.8 meV/atom at 4.0 A, 5.2 at 4.2 A, 4.7 at 4.3 A). This run's path
  (`RCUT2B` 5.6 to 6.0, about 2 meV/atom) never goes there, so the
  filter does not fire here.

Run:

    bash ../common/run.sh
