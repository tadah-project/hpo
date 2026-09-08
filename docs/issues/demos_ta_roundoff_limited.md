# HPO_REFIT_2B_EAM/demos/Ta: LN_BOBYQA stopped "nlopt roundoff-limited" at 109/200 evals

The only refit-space BOBYQA demo that did not run to MAXEVAL; loss 2.7e-3 at eval 108; Al/Cu/Fe/Ti ran to
200. With phase 2 the per-dim `rhobeg` is journaled (`Initial step` table) — re-run the demo under
`OPTIM_SCALE NONE` and `AUTO`, compare the effective rhobeg per FIXWEIGHT dim with the writer's group-floor
boxes (`refit_writer_group_floor.md`), and check whether the trust region collapsed on the 13 identical
±38.2 boxes (flat loss vs 0.25·width) or on a zero-spanning box.
