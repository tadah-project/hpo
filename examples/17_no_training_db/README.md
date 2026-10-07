# 17 -- no training DBFILE: coefficient search on a fitted pot

`tadah hpo` without any training data: every linear coefficient is
pinned by `FIXINDEX 1-14` / `FIXWEIGHT`, so there is nothing for the
least-squares stage to fit and the training `DBFILE` can be dropped.
`--validation` is omitted too -- the loss is a single LAMMPS-computed
observable (cohesive energy of BCC Ta).

What this shows:
- the seed workflow: `tadah train` -> `pot.tadah` -> pin `WEIGHTS` as
  `FIXINDEX`/`FIXWEIGHT` (see `run.sh` step 2 -- this is what `tadah
  refit` emits for its coefficient-polish stage); a potential records no
  `DBFILE`, so the seed has no training data (step 2 drops the `DBFILE`
  line an `M_BLR` potential written before 1.6.0 has);
- a run whose only loss source is `PC_LAMMPS` (no validation metrics,
  hence no `--validation` on the command line);
- `OPTIM FIXWEIGHT (2-4)`: search three coefficients, hold the rest;
- the element set comes from the `ATOM` key carried by the pot header
  (required -- with neither training nor validation data there is
  nothing to derive it from).

Run:

    bash ./run.sh
