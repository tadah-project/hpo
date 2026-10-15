# 27 -- DM_REAM refit potential: F(0) = 0 by construction (EMBF0 ZERO)

`tadah hpo` on a native EAM potential written by `tadah refit`, searching
embedding-shape parameters only. The seed's embedding is zero-referenced
(`EMBF0 ZERO`, the default): every candidate's DM_REAM subtracts its own
g_m(0), so the isolated-atom energy is exactly 0 for every evaluation --
no penalty term on E_iso is needed.

What this shows:
- a DM_REAM seed (`D2_Knot5` + `DM_REAM` knot family) with every
  coefficient pinned (`FIXINDEX`/`FIXWEIGHT`): no training `DBFILE`, no
  `--validation`, the loss is the LAMMPS cohesive energy;
- the mandatory density form `RHOFORM SIGNED` (new in tadah 1.6.0): 4 of
  the seed's 8 density amplitudes are negative, so it keeps the signed
  density of every tadah before 1.6.0;
- `OPTIM CEMBFUNC (1)` / `(4)`: the two NEGATIVE embedding knots, the
  parameters that change the raw F(0) = sum_m w_m g_m(0) (the journal
  prints it: about -6.07 eV for this seed);
- `--outvar Eiso` reads the isolated-atom energy back from
  `common/lammps/Ecoh.in`; `run.sh` asserts it is exactly 0.0 for every
  evaluated candidate (rejected candidates report FAILSCORE instead) and
  that `best_pot.tadah` declares `EMBF0 ZERO`.

Seed provenance (`ta2_refit_seed.tadah`): `tadah refit` with

    EAM          MODELS/tests/tests_data/eam.alloy   (Ta2, Ravelo 2013)
    RHOFORM      SIGNED
    REFIT_MODE   compact

(default `EMBF0 ZERO`; the seed was written with `REFIT_LEVEL coarse` too,
which the compact fit never read and which `tadah refit` now refuses under
`REFIT_MODE compact`). Add `EMBF0 FREE` to the seed to see the raw
embedding instead: Eiso is then the candidate's raw F(0).

The seed was written by `tadah refit` before 1.6.0, which wrote the
many-body term as `DM_REAM`; this example keeps it to show a DM_REAM seed.
`tadah refit` 1.6.0 writes a `DM_Embed` component instead (`DESCRIPTOR
eam DM_Embed` and `TYPEMB @eam`, with no `CEMBFUNC`, `RHOFORM` or `EMBF0`
lines), so the recipe above now gives a potential for which `config.hpo`'s
`OPTIM CEMBFUNC` is refused, naming the component path to use instead
(`OPTIM CEMBFUNC: the potential has no CEMBFUNC key: ... the named path
EAM.EMBEDDING.CENTRES; ...`): with such a seed, search the negative
embedding knots as `OPTIM EAM.EMBEDDING.CENTRES (<index>)` (the starter
`refit.hpotarget` of the refit lists them), and its `best_pot.tadah`
declares `REFERENCE ZERO` in the component's `EMBEDDING` block.

Run:

    bash ./run.sh
