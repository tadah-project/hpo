# 08 -- Sobol init + DLIB:GFS

Quasi-random seeds fed to a multi-seed optimiser.

What this shows:
- `STRATEGY SOBOL K` -- low-discrepancy sequence; better space-filling
  than RANDOM for small K. `SEED s` draws block b = s mod 65536 of K
  points of the sequence (points b*K ... b*K + K - 1), so seeds that
  differ modulo 65536 draw disjoint points; without `SEED` the block is
  random.
- `DLIB:GFS` -- DLIB's global function search. Unlike BFGS it
  consumes every seed in the init chain (each becomes an initial
  function evaluation), so a longer chain genuinely speeds it up.
