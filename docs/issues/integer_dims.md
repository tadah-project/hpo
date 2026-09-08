# Integer OPTIM dimensions are hard-wired off

`HPOTargets` sets `is_int` all-false (`hpo_targets.cpp`); `SearchSpace::build` refuses integer dims under
`OPTIM_SCALE AUTO` (a unit-box coordinate cannot be integral). dlib GFS supports integer dims natively;
NLopt does not. A real integer story needs: an `INT` tag on the OPTIM line, `NONE`-style raw
coordinates for those dims (or rounding in `to_real`), GFS pass-through, and rejection for the other
backends.
