# 16 -- mixed weight mode: LS + pinned + OPTIM'd coefficients

Demonstrates `FIXINDEX`/`FIXWEIGHT` pinning combined with
`OPTIM FIXWEIGHT`, on the shared 14-weight Ta model:

- weights 1-12: least-squares refitted on every evaluation (as usual),
  against the residual target `T - Phi_fixed * w_fixed`;
- weight 13: held constant at its pinned value (0.0);
- weight 14: searched by the optimiser via `OPTIM FIXWEIGHT (14)`.

What this shows:
- `FIXINDEX 13,14` + `FIXWEIGHT 0.0 0.0` in a **local** `config.tadah`
  (the i-th value pins the i-th index).
- `OPTIM FIXWEIGHT (14)` -- the index names **model weight 14**, not a
  position in the FIXWEIGHT list.
- Both pinned weights are 0 in the unconstrained fit on this dataset,
  so iteration 0 (implicit CONFIG start) reproduces the baseline
  potential exactly; the optimiser then explores whether activating
  w14 buys any loss.
- A derivative-free optimiser (`NLOPT:LN_BOBYQA`): with FIXINDEX
  present, analytical/HYBRID gradients fall back to finite differences.

Run:

    bash ./run.sh

The optimum lands in `best_pot.tadah` -- note its FIXWEIGHT values are
the winning ones, so the file re-seeds a follow-up HPO run.
