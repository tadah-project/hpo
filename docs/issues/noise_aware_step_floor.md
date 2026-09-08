# Noise-aware floors for STEP / XTOL_ABS / fd_step (warn, never clamp)

Moré–Wild (SIAM J. Sci. Comput. 33:1292, 2011; ACM TOMS 38(3):19, 2012): the resolvable
finite-difference step is `h_FD ≈ 8^{1/4} (ε_f / ν₂)^{1/2}` (forward) / `3^{1/3}(ε_f/ν₃)^{1/3}` (central),
where `ε_f` is the objective's noise level — which `NOISE_CALIBRATION` already measures at the seed.
In the unit box every dim shares a scale, so one floor is meaningful for all of them.

**Proposal:** after `NOISE_CALIBRATION`, WARN (never clamp — memory: calibration is unreliable on early /
poor potentials) when `STEP` (fraction of width), `XTOL_ABS` or `PARAM fd_step` sits below the
resolvable floor implied by the measured `rel_spread` and a curvature estimate from the first
gradient / simplex; print the implied floor in the NOISE CALIBRATION section.

**Test:** synthetic noisy bowl with known `ε_f`: the WARNING fires for `fd_step < floor` and not above.
