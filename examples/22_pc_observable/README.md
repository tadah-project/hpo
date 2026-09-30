# 22 -- a typed observable: PC_OBSERVABLE instead of LAMMPS

Example 01's search with the LAMMPS cohesive-energy script replaced by
`PC_OBSERVABLE cohesive_energy`: Tadah builds and relaxes a 4x4x4 bcc Ta
supercell in process (the relaxation limits of
`../common/lammps/Ecoh.in`: `--maxiter 500 --maxeval 500 --etol 1e-12`)
and differentiates the cohesive energy with respect to the
hyperparameters analytically. No LAMMPS runs, so the example also runs
where LAMMPS worker processes cannot.

What this shows:
- `PC_OBSERVABLE cohesive_energy --species Ta --target 8.10 --weight 800
  --hotvar a0 2.5 4.5 3.302` -- the same target, weight and lattice
  window as example 01's `PC_LAMMPS` line.
- No `GRAD_STRATEGY` line: with `PC_OBSERVABLE` it resolves to
  `ANALYTICAL` for `DLIB:BFGS`, and the OPTIMISER section of
  `hpo_run.log` says so (`GRAD_STRATEGY : ANALYTICAL (resolved for
  PC_OBSERVABLE)`). BFGS reaches the optimum (`RCUT2B` at its upper
  bound, as in 01) in 4 evaluations instead of 01's 17 or more (30 on
  the reference machine; see 05 for why that count varies), because its
  gradient costs no finite-difference evaluations.
- A local `config.tadah` (picked up by `common/run.sh`): the shared model
  with `OALGO 4` and `LAMBDA 1e-6`. The analytical gradient
  differentiates the fit through its Cholesky factor and needs both;
  without them the run stops at startup with the reason, or use
  `GRAD_STRATEGY HYBRID` for finite differences.
- The observable's columns: `L(Ecoh)[1]` in `loss.tadah`, `Ecoh[1]` in
  `outvar.tadah`; the journal names the term `cohesive_energy[1]`.

Run:

    bash ../common/run.sh
