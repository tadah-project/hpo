# ASA lessons for the in-house MC family (ANNEAL / ANNEAL_PT / ANNEAL_PA)

Observations collected while vendoring Ingber's ASA-30.53 as `LIB ASA /
ALGO ASA` (2026-08-19). These are HYPOTHESES for a future session, not
prescriptions: each should be re-ranked against live evidence on the
current configs before any implementation (see the house rule: iterate
on evidence, not on ticket prescriptions). Sources: `external/asa/asa.c`
/ `ASA-README.txt` / `ASA-NOTES`, the three Ingber papers, and what the
integration surfaced.

## 1. Reannealing is a restoring force; Corana V-adaptation has none

ASA periodically RESCALES each parameter temperature from measured
finite-difference sensitivities (`tangents`, step `Delta_X`): an
insensitive dimension gets its temperature RAISED (its k_i index is
recomputed from the new temperature), so a dimension that went quiet can
wake up. The Corana V-update in our family only reacts to acceptance
ratios and has a hard floor — once V hits `V_FLOOR_FRAC` it tends to
stay there (the documented V_FLOOR absorbing state; the D27 trust region
bounds drift but still adds no restoring force). Observation: the
mechanism that prevents ASA's equivalent of the absorbing state is not
the floor but the periodic sensitivity-driven RESET. A cheap experiment:
per-rung tangents at the incumbent (D+1 evals per reanneal cadence,
ASA pays the same) driving a V (or sigma) re-initialisation.

## 2. Fat-tailed generating distribution vs bounded uniform/Gaussian steps

ASA's per-dimension generating distribution keeps polynomial tails at
low temperature — occasional box-scale jumps remain possible forever,
which is precisely what admits the exponentially fast schedule with a
sampling guarantee (Ingber 1989). CLASSIC's Gaussian and Corana's
U(-V,V) proposals lose all long-range mass once sigma/V anneal down;
run24's autopsy concluded the PT basin mechanism could not reach the
loss~1 basin by continuation. Observation: tail mass, not temperature,
is what buys basin escapes late in a run. A Corana-compatible variant
(e.g. mixing a small probability of a box-scale draw into the proposal)
is testable without touching the schedule.

## 3. Dimension-aware schedules: the k^(1/D) exponent

ASA anneals T_i(k) = T0 exp(-c k^(1/D)): the schedule EXPLICITLY slows
with dimension, and Ingber's tuning advice scales the constants with D
(`Temperature_Anneal_Scale > D`, think in Q/D units). Our RT-per-rung
geometric ladders have no D in them anywhere; the same RT that works at
D=2 examples was repeatedly too fast/slow on the D~100 potentials
(run21+ bounds-tending, plateau ladders). Observation: making the
default RT (or rung length) a function of dim is the cheapest
ASA-inspired change with a clear theoretical motivation.

## 4. Acceptance-flow stop: Accepted_To_Generated_Ratio

ASA stops (or triggers reanneal tests) when accepted/generated falls
below a threshold — a direct measure of "is the chain still doing
useful work", independent of loss values and noise floors. The MC
family stops on temperature (T_MIN), budget, or the eps-NEPS plateau;
the plateau needs a well-chosen FTOL against the measured noise floor
(NOISE_CALIBRATION caveats apply). An acceptance-flow stop would be
insensitive to the loss scale entirely.

## 5. Plateau detection: |last - best| repetition vs eps-NEPS

`Cost_Precision` x `Maximum_Cost_Repeat` is an absolute-tolerance
repeat counter on the accepted-cost stream — much cruder than
CoranaPlateau, and at the 1e-18 default it effectively never fires on
noisy LAMMPS losses. Nothing to import here; if anything this validates
the eps-NEPS design. Worth writing down so nobody "upgrades" to it.

## 6. Block-parallel generation for the SERIAL annealer

ASA_PARALLEL generates a block of B candidate states serially (RNG
stays sequential), evaluates them concurrently, sorts by cost, and
feeds them through the standard sequential acceptance. Properties worth
copying: (a) single-chain parallelism WITHOUT replicas or a ladder;
(b) results provably THREADS-invariant (evaluation order cannot matter
after the sort); (c) budget reservation per-slot in deterministic order
keeps even the truncated final block reproducible (our integration
does this; see `src/algorithms_asa.cpp`). Serial `TADAH:ANNEAL` could
gain a `BLOCK N` knob with the same contract — the sampled sequence
changes (more proposals per acceptance decision at low acceptance
rates), which is the same caveat ASA documents.

## 7. Evaluation cache with per-dim resolution (ASA_QUEUE)

ASA can cache (state -> cost) with a per-dimension match resolution and
replay hits without calling the objective. With LAMMPS-priced
evaluations, integer/quantized dims (refit knot grids!), and annealers
that re-visit, a host-side loss cache keyed on quantized theta could be
a real saver. Needs care with the loss.tadah eval stream and iteration
accounting (cached hits are not host evals) — and it changes trajectory
vs no-cache, so opt-in only.

## 8. Exit taxonomy: name the mechanism, per dimension

ASA distinguishes P_TEMP_TOO_SMALL (a PARAMETER temperature annealed
out — with the index of the dimension recorded) from C_TEMP_TOO_SMALL,
COST_REPEATING, TOO_MANY_INVALID_STATES etc. The per-dimension
temperature-underflow exit is a convergence diagnostic the MC family
cannot express (T_MIN is global). The pinned-dim audit already gathers
per-dim evidence; promoting "dim i annealed out" to a first-class stop
reason (or at least a SUMMARY WARNING) would make many "why did it
stop/stall" investigations one-line answers.

## 9. Bound every invalid-state retry loop (found live)

Upstream ASA's parallel invalid-state retry loop lacks the serial
path's in-loop `Limit_Invalid_Generated_States` bound: a block that
stays all-invalid (persistent host failures, or a stop path flagging
states invalid) spins FOREVER. We hit this within minutes of wiring
MAXEVAL (see `external/asa/TADAH-PATCHES.md`, patch 3). The MC family's
equivalent guards look healthy (BOUNDARY_MODE redraw has its 4096-draw
guard), but the audit question "can any retry loop survive a
permanently failing host?" is worth asking once per driver —
FAILSCORE storms are routine in our runs.

## 10. Stop granularity vs determinism is a real design axis

To keep block-parallel results THREADS-invariant we made MAXTIME/
STOPVAL act at block boundaries (budget stays exact via serial per-slot
reservation). PT made the same trade at epoch granularity. Writing the
rule down ("a parallel driver's stops act at its synchronisation
granularity; only budget is exact") would prevent re-deriving it per
driver — and per-eval stop checks inside concurrent regions are exactly
what broke THREADS-invariance in the first ASA integration attempt.

## 11. Sensitivity probes are cheap and reusable

ASA buys its per-dim adaptivity with D finite-difference tangents per
reanneal cycle. ANNEAL_PT's D28 warm-up calibration spends far more
evals per (slot, dim) bracketing acceptance. Observation: a
tangent-style probe (one central difference per dim at the incumbent)
gives a sensitivity ranking for a fraction of the cost and could seed
the bracket search, classify insensitive/hypersensitive dims earlier,
or drive reannealing (#1) — all from the same D+1 evals.
