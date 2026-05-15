# 01 -- minimal HPO run

The smallest config that exercises the full pipeline.

What this shows:
- The required top-level blocks: `OPTIMIZER { STRATEGY { ... } }`,
  `LOSS`, `PC_LAMMPS`, `OPTIM`.
- A single free hyperparameter (`RCUT2B`) optimised with `DLIB:BFGS`.
- A LAMMPS-driven loss term: cohesive energy of BCC Ta.
- Implicit starting point: the value in `../common/config.tadah` is
  evaluated as iteration 0.

Run:

    bash ../common/run.sh

The optimum lands in `best_pot.tadah`; per-iter logs are in
`loss.tadah`, `params.tadah`, `outvar.tadah`.
