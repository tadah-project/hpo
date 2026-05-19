# 08 -- Sobol init + DLIB:GFS

Quasi-random seeds fed to a multi-seed optimiser.

What this shows:
- `STRATEGY SOBOL K` -- low-discrepancy sequence; better space-filling
  than RANDOM for small K.
- `DLIB:GFS` -- DLIB's global function search. Unlike BFGS it
  consumes every seed in the init chain (each becomes an initial
  function evaluation), so a longer chain genuinely speeds it up.
