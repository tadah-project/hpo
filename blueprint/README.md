# Registry files

The files in this directory describe parts of `tadah hpo` as data, for the
documentation (Tadah.MLIP `docs/`): it is built without C++, so it cannot ask
the code. The documentation reads `anneal_keys.toml`; `optimizers_gen.toml`
is written for it. A CTest entry fails when a file and the code differ.

| File | Kept in step by |
|---|---|
| `anneal_keys.toml` | `gen_anneal_keys.py`, CTest `hpo_anneal_registry_sync` |
| `optimizers_gen.toml` | `tests/tools/hpo_registry_export.cpp`, CTest `hpo_optimizer_registry_sync` |

## anneal_keys.toml

The `PARAM` keys of the `TADAH:ANNEAL` family: names, types, defaults,
ranges and help text. It is the source: `gen_anneal_keys.py --header` writes
`include/tadah/hpo/anneal_keys_gen.h` from it, and
`hpo_anneal_registry_sync` (`gen_anneal_keys.py --check`) fails when the
checked-in header differs.

## optimizers_gen.toml

The optimiser registry and the ALGO help of every optimiser, written for
the documentation. Generated from the optimiser registry
(`src/optimizer_caps.cpp`), the ALGO help (`src/optimizer_spec.cpp`,
`src/asa_options.cpp`, and `anneal_keys.toml` for the annealers' `PARAM`
keys) and the NLopt version HPO is built with. It is not edited by hand:
after a change to any of these, the command below regenerates it, and it
is committed with that change:

```bash
cmake --build <build directory> --target hpo_regenerate_optimizer_registry
```

The target builds `hpo_registry_export` (`tests/tools/hpo_registry_export.cpp`)
and runs it with `--write` on this file. CTest `hpo_optimizer_registry_sync`
(label `docs`) runs it with `--check` and fails when the file differs from
what the code produces. Both read the text back as TOML before they write or
compare it. The text is the same in every build of one
NLopt version: the ALGO help leaves out the lines that say what a build lacks
(a build without Ceres).

`[meta]`:

- `schema`: the layout of the file, 1.
- `nlopt_version`: the NLopt version HPO is built with; the `NLOPT`
  entries are its algorithms.
- `libraries`: the `LIB` names of the registered optimisers, sorted.

One `[[optimizer]]` table per `LIB:ALGO`, registered or refused, sorted by
`lib`, then `algo`. Every table has:

- `lib`, `algo`: the names, as a target file writes them.
- `status`: `"registered"` or `"refused"`.
- `explain_text`: what `tadah explain LIB:ALGO` prints.

A refused optimiser has one more field:

- `reason`: why it is refused, and what to use instead.

A registered optimiser has these fields (an empty string means none):

- `row_text`: its row of `tadah explain ALGO`, after the `ALGO` name.
- `global`: a global search; `false` for a local one.
- `gradient_based`: it uses the gradient of the loss.
- `accepts_inner`: it runs a local search, which an `INNER` block can set
  (NLopt's MLSL and AUGLAG).
- `default_inner`: the local optimiser NLopt runs without an `INNER` block;
  empty when it has none (then the block is required) or when
  `accepts_inner` is `false`.
- `accepts_multi_seed`: one run takes more than one starting point.
- `reads`: the attributes a run reads besides its stops, sorted, from
  `FTOL_ABS` (read as a setting, not a stop), `POPULATION`, `SEED`,
  `VECTOR_ARR`.
- `warns_unread_seed`: a run does not read `SEED`, and a `SEED` gets a
  warning, not a refusal.
- `stops`: the stop criteria that end a run, in the order `MAXEVAL`,
  `MAXTIME`, `STOPVAL`, `FTOL_REL`, `FTOL_ABS`, `XTOL_REL`, `XTOL_ABS`.
- `needs_stop`, `stop_note`: a run has no convergence test and no finite
  schedule, so it needs one of its stops or an active `RESTART`; the note
  says why.
- `ignores_forced_stop`: the search ignores NLopt's forced stop, so only
  `MAXEVAL` counts as its stop.
- `tolerance_note`: how a tolerance ends the run when it is costly.
- `untested_tolerance_note`: why a tolerance the run does not test has no
  effect.
- `step_use`: how the run uses `STEP`, one of `rejected`, `ignored`,
  `per_dim_vector`, `per_dim_rescale`, `min_collapsed`, `trust_half_width`,
  `forwarded_to_inner`, `grid_count`, `width_fraction`; `step_use_text`: its
  wording in the ALGO help and the run's journal; `step_consumed`: whether
  `STEP` is accepted; `step_note`: how it is used, or what to set instead.
- `starts_at_x0`: the first evaluation is at the point the search is given;
  `start_note`: where it is otherwise.
- `random_draws`: for a search that does not start at its point, whether a
  run draws random numbers: `never`, `between_sides` (only to choose between
  equally long sides of a box), `trial_points`; `not_recorded` for every
  other search.
- `ignores_bounds`, `bounds_note`: the search proposes points outside the
  `OPTIM` box; the note is the startup warning.
