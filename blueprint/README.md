# Registry files

The files in this directory describe parts of `tadah hpo` as data, for the
documentation (Tadah.MLIP `docs/`): it is built without C++, so it cannot ask
the code. The documentation reads `anneal_keys.toml`; `optimizers_gen.toml`
is written for it, and so are the target-file registry, `hpotarget.toml` and
its part files, and `optimizers_doc.toml`, what each optimiser is. A CTest
entry fails when a file and the code differ.

| File | Kept in step by |
|---|---|
| `anneal_keys.toml` | `gen_anneal_keys.py`, CTest `hpo_anneal_registry_sync` |
| `optimizers_gen.toml` | `tests/tools/hpo_registry_export.cpp`, CTest `hpo_optimizer_registry_sync` |
| `hpotarget*.toml` | `hpotarget_registry.py`, the CTest entries of "The target-file registry" |
| `optimizers_doc.toml` | `hpotarget_registry.py` (it describes every optimiser of `optimizers_gen.toml`) |

## anneal_keys.toml

The `PARAM` keys of the `TADAH:ANNEAL` family: names, types, defaults,
ranges and help text (`doc`, one or two sentences; `long_doc`, the detail;
`range_doc`, in words, the range of a key whose range the code checks by
hand; `default_doc`, in words, a default that is no value, such as a
calibration). Every kind a key lists has a default or a `default_doc`
for it, except a kind that refuses the key (a `kind_reject` rule); the
help and the documentation leave such a key out. It is the source:
`gen_anneal_keys.py --header` writes
`include/tadah/hpo/anneal_keys_gen.h` from it, and
`hpo_anneal_registry_sync` (`gen_anneal_keys.py --check`) fails when the
checked-in header differs. The example of each key is in the `PARAM` policy
of the annealers, `PARAM.TADAH:ANNEAL` in `hpotarget_params.toml`, whose
examples the conformance test runs: the last line of an example sets the
key, and the lines before it set what the key needs.

## optimizers_doc.toml

What each optimiser is, in one or two sentences: `[library.<LIB>] doc` for
each library of `optimizers_gen.toml` and `[optimizer."<LIB:ALGO>"] doc` for
each registered optimiser. The documentation shows it first in the
optimiser's entry. `hpotarget.toml` names the file (`meta.optimizer_docs`),
and `hpotarget_registry.py --check` fails on a library or a registered
optimiser without a description, on a name that is neither, and on text
that addresses the reader. A new optimiser of the registry therefore needs
its description here.

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
  `THREADS` (the `THREADS` of an `EXPLORE` block), `VECTOR_ARR`.
- `algorithm_params`: the `PARAM` keys NLopt reads for the algorithm, in
  NLopt's order; empty for an algorithm that reads none and for another
  library. A `PARAM` key of an NLopt algorithm that is neither one of them
  nor a finite-difference key is refused.
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

## The target-file registry

`hpotarget.toml` and the part files its `meta.parts` lists describe the
target file of `tadah hpo` (`--hpotarget`): how a line is read, the eight
blocks and where each opens, every key with its values, default, range and
refusals, the options of the `PC_*` lines, the `OPTIM` grammar and the
`PARAM` keys of the optimisers. The parser is the source. Every message in
the registry is the parser's text, copied from the code, and the checks at
the end of this section fail when the registry and the parser differ.
`hpotarget_registry.py` (Python 3.11 or later, standard library only) loads
the files, checks them and returns them as dataclasses.

### Files

| File | Contents |
|---|---|
| `hpotarget.toml` | `[meta]`, `[lexing]`, `[reading]`, `[structure]`, `[group.OPTIMIZER]`, the eight `[[block]]` tables; the top-level keys that `hpotarget_lines.toml` does not hold, with the options of `RHOGATE` and `RHOBARGATE`; `[[removed]]`, `[[rule]]`, `[[env]]`; the wrappers and the training configurations of the examples |
| `hpotarget_optimizer.toml` | the keys of the optimiser group, `LOG_HP`, the keys of the `STRATEGY`, `INIT`, `NOISE` and `RESTART` blocks; `[[strategy_type]]`, `[[init_method]]`; the rules of the `INNER` block |
| `hpotarget_lines.toml` | the top-level keys `PC_EMAE`, `PC_ERMSE`, `PC_ErRMSE`, `PC_ERSQ`, `PC_SRMSE_GPA`, `PC_FRMSE`, `PC_LAMMPS`, `PC_FAILPROBE`, `PC_OBSERVABLE`, `LOSS` and `OPTIM`; their options (`[[line_option]]`); the observable kinds and the `OPTIM` forms (`[[form]]`); `[[loss_function]]`, `[optim]`, `[[optim_target]]`; `[lammps_script]`, what the LAMMPS script of a `PC_LAMMPS` line provides; the rules between these lines, and between them and the optimisers |
| `hpotarget_params.toml` | `[[param]]`, `[[param_excluded]]`, `[[param_policy]]`; the `PARAM` keys of the annealers are those of `anneal_keys.toml`, which `meta.anneal_keys` names |

Only `hpotarget.toml` holds `[meta]`, `[abbreviations]`, `[lexing]`,
`[reading]`, `[structure]`, `[group.*]` and `[[block]]`. A part file holds the other
tables, and may define wrappers, training configurations and fixtures of
its own; its examples use those of `hpotarget.toml` and its own.

### Ids

Every entry has an id, unique across the files: a scope, a dot and a name,
as the target file writes the name. The documentation links to the ids.

- key: `<scope>.<NAME>`, the scope its group (`OPTIMIZER.MAXEVAL`) or the
  first block it is valid in, `ROOT` for a top-level key (`ROOT.FAILSCORE`,
  `STRATEGY.TYPE`, `INIT.METHOD`);
- line option: `<line>.<option>`, the line a top-level key or a form
  (`PC_LAMMPS.--hotvar`, `OPTIM.AUTO.RATIO`);
- form: `<line>.<form>` (`PC_OBSERVABLE.cohesive_energy`, `OPTIM.AUTO`);
- loss function `LOSS.<NAME>`, STRATEGY type `STRATEGY.TYPE.<NAME>`, INIT
  method `METHOD.<NAME>`, `OPTIM` target `OPTIM.TARGET.<KEY>`, rule
  `RULE.<name>`;
- `PARAM` key: `PARAM.<optimiser>.<name>`, the optimiser a `LIB:ALGO`, a
  `LIB`, or `FD` for the finite-difference keys
  (`PARAM.DLIB:BFGS.tol_gradinf`, `PARAM.CERES.max_iter`,
  `PARAM.FD.fd_step`); `PARAM` policy: `PARAM.<LIB or LIB:ALGO>`;
- set by the loader: `BLOCK.<NAME>`, `GROUP.<NAME>`, `REMOVED.<NAME>`,
  `ENV.<NAME>`, and `LEXING`, `READING`, `STRUCTURE`, `OPTIM` and
  `LAMMPS_SCRIPT` for the sections.

A name may have several entries when the parser reads it in different
places: `ROOT.MAXTIME` (the script timeout) and `OPTIMIZER.MAXTIME` (a stop
of the search), `OPTIMIZER.SEED` and `INIT.SEED`. Two keys of one name are
never valid in one block.

### Messages and placeholders

`error_frame`, `underflow_note`, a field named `msg` and a field whose name
ends in `_msg` are message templates: `{name}` marks a value the parser puts
there. A template uses the placeholder names of the parser's messages
(`PLACEHOLDERS` in `hpotarget_registry.py`). The messages of `[lexing]`,
`[structure]` and `[group.*]`, and a block's `repeat_msg`, `misplaced_msg`,
`unknown_msg` and `refuse_msg`, have a fixed set each
(`FIXED_PLACEHOLDERS`); those of `[reading]` hold `what`, `min`, `max`,
`tok`, `note`, `noun`, `underflow` and `key`. An example, an error and a
wrapper hold no placeholder other than a wrapper's slot: they are the text
itself. `${name}`, a LAMMPS variable reference, is text, not a placeholder
(`timer timeout ${maxtime}` in the refusal of `--invar maxtime`).

Documentation text (`doc`, `long_doc`, `quirk`, `condition`, `title`,
`repeat_doc`, `msg_doc`, the `doc` of an argument or a case, a default's
`text`) holds no placeholder: the reference page shows it before the
message, so it says in words what the message holds; a rule's `msg_doc`,
shown after its message, is the place for that. The `doc` of a check, which
says when its message comes, may name the placeholders of that message.

Every refusal the parser makes inside the file is framed by
`lexing.error_frame`, `Error on line {line}: {what}\nLine: {text}`, with the
last physical line of the logical line and its joined text; an `error` field
holds the `{what}` part.

### Examples, invalid lines, wrappers

Every entry, and `[lexing]`, `[reading]`, `[structure]`, `[optim]` and
`[group.*]`, may have `examples` and `invalid`, arrays of tables with these
fields:

- `wrap`: the wrapper the text goes into, or `none` when the text is the
  whole target file;
- `text`: the lines that take the wrapper's slot, one or more, continued
  lines included;
- `check`: `parse` (the target file is read) or `startup` (it is read, and
  passes the startup checks of `tadah hpo`); `startup` for an example and
  `parse` for an invalid line by default. An invalid line may also take
  `check` (the run of `tadah hpo --check`, with the training configuration
  as its validation configuration), for a refusal that the run makes after
  the startup checks and before it reads its data, such as that of an
  environment variable;
- `context`: the training configuration (`[context.*]`), by default the
  wrapper's;
- `env`: environment variables set while the case runs;
- `doc`: what the case shows, the caption of its code on the reference
  page (optional for the first example of an entry, required for every
  later one);
- `error` (invalid lines only): the message without its frame;
- `at` (invalid lines only): the line the frame names. A number counts the
  logical lines from the first line of the text, 1 by default; a key names
  the first logical line from there that starts with it (`ENDOPTIMIZER`);
  `none` means a message without a frame;
- `msg_of` (invalid lines only, optional): the id of the entry whose
  message the error is, when it is not the entry's own, nor one of
  `[lexing]`, `[reading]`, `[structure]`, a block or a group.

The error of an invalid line is a message of the registry: a template of
its entry, of the entry `msg_of` names, or of those sections, blocks and
groups, with its placeholders filled. In reverse, each message template of
an entry is shown by an invalid line, or listed in the entry's `untested`
with the reason no case can show it.

A wrapper, `[wrapper.<name>]` (`text`, `doc`, `block`, optional `context`
and `optimizer`), is a complete target file with one slot line, `{line}`:
with the slot left empty, the parser and the startup checks accept it, so a
case fails for its own text. `block` is the block the slot is in, `ROOT`
for a slot at the top level; `optimizer` names the `LIB:ALGO` of the
`STRATEGY`-level optimiser, for the generated checks of the `PARAM` keys.
`hpotarget.toml` has a wrapper for each place a line can be, named after
it: `root` and `optimizer`, `strategy`, `init`, `noise`, `explore`,
`refine`, `inner` and `restart`, each with the `block` of its name; `end`
has its slot at the top level after the `OPTIMIZER` block. The
`[wrapper.*]` comment says where each slot is. A training configuration,
`[context.<name>]` (`config`, `doc`, optional `has_train_db` and
`has_validation`, both `true` by default), is the text of a `--config`
file; `default` is a one-species two-body model that builds without data.
A fixture, `[fixture.<name>]` (`path`, `text`, optional `doc`), is a file
written next to the target file before each case, such as a LAMMPS script
that a `PC_LAMMPS` line names.

### Tables and fields

Fields are required unless marked optional. `doc` is one or two sentences
for the reference page and `long_doc` (optional) the detail; both, and
`quirk`, `condition` and every other `doc`, are written in the impersonal
voice, without history or source references. `quirk` (optional) states
behaviour kept as the parser has it although a reader would not expect it.
`untested` (optional, on every entry, section, group and block) is an
array of `msg` (a message template of the entry) and `why` (why no case
shows it: a refusal of the MPI build, of a directory the run cannot write,
of the data the run reads).

`[lexing]`, `[reading]`, `[structure]`, `[optim]`, `[group.*]` and
`[[block]]` take a `long_doc` too.

`[meta]`: `version` (the layout of the registry, 1; the loader refuses
another), `api` (the loader's
API, 1), `parts` (the part files, in order), `anneal_keys`, `optimizers`
and `optimizer_docs` (the files with the annealers' `PARAM` keys, the
optimiser list and the descriptions of the optimisers).

`[abbreviations]`: abbreviation = what it stands for (`MLSL =
"multi-level single-linkage"`), for the abbreviations of the registry's
text; the reference pages spell each out at its first use on a page.

`[lexing]`: `doc`; `comment`, `continuation`; `key_case` (`exact`: keys are
matched as written); `error_frame`, `unfinished_msg`, `missing_file_msg`;
`bool_true`, `bool_false` (the boolean words); `timespec_words` (the words
of a LAMMPS time spec).

`[reading]`: `doc` (the number, integer and boolean grammar) and the
messages of the value readers: `integer_msg`, `integer_range_msg`,
`note_msg`, `finite_msg`, `positive_msg`, `underflow_note`,
`requires_value_msg`, `single_value_msg`, `bool_msg`, `timespec_msg`,
`script_timeout_msg`, `script_timeout_large_msg`. An argument's `kind`
selects the reader, and its `what`, `min`, `max` and `noun` fill the
message.

`[structure]`: `doc`; `check_order`, the order in which the parser tries a
line (block lines, removed keywords, the INNER refusals, the optimiser
group, the INIT and RESTART keys, the NOISE keys, `LOG_HP`, the STRATEGY
keys, the unknown-key refusal of the block, the top-level keys; the loader
requires exactly this order); `end_without_open_msg`, `end_mismatch_msg`,
`end_takes_no_value_msg`, `open_takes_no_value_msg`, `unclosed_msg`,
`unknown_root_msg`, `removed_msg`.

`[group.<NAME>]`: `blocks` (where the keys of the group are valid),
`leftover_msg` (a value after the last one a key reads), `doc`.

`[[block]]`: `name`, `end` (`END<name>`), `parents` (`ROOT` or blocks),
`repeat` (`error`: a second block is refused with `repeat_msg`;
`error_per_parent`: once for each optimiser, the one the enclosing block
configures, so the `STRATEGY` and `REFINE` blocks share one, and so do two
`EXPLORE` blocks; `merge`: a second block continues the first), `repeat_msg`, `misplaced_msg` (opened where no parent
allows), `unknown_msg`, `unknown_first` (optional: an unknown key is refused
there before the `LOG_HP` check, so `LOG_HP` inside `INIT` is an unknown
key), `aliases` (optional: a word read as another key inside the block,
`STRATEGY = "INIT.METHOD"`), `refuse` and `refuse_msg` (optional: group keys
the block refuses, each with `key` and `why`), `unreachable_msgs` (optional:
messages the parser holds for the block that no file reaches), `requires`
(optional: conditions checked at the opening line), `close_checks`
(optional: checks at or after the END line), `quirk`, `doc`. `requires` and
`close_checks` are arrays of `msg` and `doc` (when the check is made).

`[[key]]`: `id`, `name`; `group` or `blocks` (one of them); `synopsis` (the
line as the reference shows it, `MAXEVAL <n>`); `args` (optional);
`repeat` (`last_wins`, `warn_last_wins`: last wins with a warning, `error`:
refused with `repeat_msg`, `add`: each line adds an entry); `repeat_msg`;
`repeat_doc` (optional: the repeat rule as the reference shows it, where
`repeat` alone does not say it: a key whose forms repeat differently, such
as `STEP`, a key set once for what several blocks configure, such as
`MAXEVAL`, or a key given again only after another line, such as `LOSS`);
`misplaced_msg` (optional: the refusal of the key in a block where it is not
valid, when the parser has one of its own); `default` (optional:
`{ literal = "..." }`, the value as a file writes it, or
`{ text = "..." }`, a sentence, with `unchecked` when a number of the
sentence is one no setting of the parser holds, saying why); `alias_of` (optional: the id of the key it
is another name for); `refused` (optional: the parser reads the key only to
refuse it; the text says why); `env` (optional: the `[[env]]` name that
overrides it); `checks` (optional: refusals of the key's values that no
argument reader makes, `msg` and `doc`); `doc`, `long_doc`, `quirk`.

`args`, an array of tables, one per value in the order of the line:
`name` (the placeholder of the `synopsis`, `<name>`, or, for a word the
synopsis writes out, any name); `kind`; `optional`, `repeat` (optional
booleans); `min`, `max` (numbers, or `INT_MAX`, `UINT_MAX`, `UINT64_MAX`);
`min_open` (optional: `min` itself is refused, a value above it is
taken); `integral` (optional, for a `number`: whole numbers only); `words` and
`refused_words` (a table of word = message) for a `word`; `what` (the
`{what}` of the reader's message, often `"{key}"`); `msg` (the key's own
message when the value is refused); `noun` (for `positive`); `default`;
`doc`. The kinds:

- `integer`: decimal digits only, between `min` and `max`;
- `number`: a finite decimal number;
- `positive`: a finite number > 0;
- `word`: one of `words`;
- `name`: any word, such as a file name, a LAMMPS variable, or a `LIB` or
  `ALGO` value (those come from the optimiser list, `optimizers_gen.toml`,
  so `LIB` and `ALGO` list no `words`);
- `bool`: a word of `bool_true` or `bool_false`;
- `timespec`: a LAMMPS time spec;
- `indices`: an index list in parentheses, `(1,3-5)`;
- `tokens`: the rest of the line, read by the grammar the `synopsis` and
  `doc` give (`PARAM`, `PC_LAMMPS`);
- `loss`: a loss function name and its parameters.

`[[line_option]]`: `id`, `option` (as written: `--hotvar`, `RATIO`),
`lines`, `synopsis`, `args`, `repeat`, `repeat_msg` (with `repeat = "error"`
only), `default`, `refused`,
`quirk`, `checks` (optional: refusals of the option's values, `msg` and
`doc`), `doc`, `long_doc`.

`[[form]]`: `id`, `line`, `word` (optional: the word that selects the form;
none for the form without one), `synopsis`, `args`, `refused`, `quirk`,
`checks`, `doc`, `long_doc`.

`[[removed]]`: `name`, `hint` (the parser's text after `was removed from the
grammar: `), `doc` (optional).

`[[rule]]`: `id`, `title` (a phrase of at most 80 characters without a
full stop, the heading of the rule's entry and the text of a link to it),
`check` (`parse` or `startup`), `msg` (optional), `msg_doc` (optional: what
the message holds, said in words), `keys` (optional: the ids of the entries
it concerns), `quirk`, `doc`. A rule relates keys or blocks to each other,
such as `PC_OBSERVABLE` and `PC_LAMMPS` in one file.

`[[env]]`: `name` (`TADAH_...`), `key` (the id of the key it overrides),
`values` (optional), `msg` (optional), `checks` (optional: its other
refusals, `msg` and `doc`), `doc`.

`[[loss_function]]`: `id`, `name`, `params` (optional: `name`, `default`,
`gt`, `doc`), `doc`.

`[[strategy_type]]`: `id`, `name`, `keys` (the ids of the STRATEGY keys of
this TYPE), `checks`, `quirk`, `doc`.

`[[init_method]]`: `id`, `name`, `args`, `refused`, `checks`, `quirk`,
`doc`.

`[optim]`: `doc`, `synopsis` (the forms of an `OPTIM` line),
`positive_only` (optional: the keys of positive values, whose `AUTO` box
without `RATIO` or `FRAC` is `RATIO 2`, where any other key's is
`FRAC 0.5`),
`auto_excluded_prefix` (optional: the key prefix `AUTO` refuses), `checks`
(optional).

`[lammps_script]` (in a part file): `doc`, `long_doc`, `fixture` (the
`[fixture.*]` that is the script the reference page shows), `examples`
(lines that run it). The entry `LAMMPS_SCRIPT`: what the LAMMPS script of a
`PC_LAMMPS` or `PC_FAILPROBE` line provides. The Catch2 case "hpotarget
registry: the LAMMPS script of [lammps_script] takes the candidate
potential of each evaluation" (`tests/test_hpotarget_script_contract.cpp`,
tag `[hpotarget_registry]`) runs the fixture as two evaluations of one line
run it, with two potentials, and checks the variables the examples' lines
name.

`[[optim_target]]`: `id`, `key`, `status` (`refused` or `special`),
`indices` (optional), `msg` (optional), `condition` (optional: when it
applies), `doc`.

`[[param]]`: `id`, `name`, `optimizers` (the `LIB:ALGO` or `LIB` names that
read it; `*` for a key every optimiser reads, as `doc` says), `args`,
`default`, `indexed` (optional: the name takes a dimension suffix,
`[<i>]`), `checks`, `quirk`, `doc`, `long_doc`.

`[[param_excluded]]`: `id`, `name`, `optimizers`, `why` (the parser's
text), `doc` (optional). A `PARAM` key the optimiser names but does not
take.

`[[param_policy]]`: `id`, `optimizers`, `msg` (optional), `checks`
(optional), `quirk`, `doc`. What an
optimiser does with a `PARAM` key that no `[[param]]` entry names, such as
NLopt, which passes a number to its algorithm, and the refusal `msg` of an
optimiser that reads a fixed set.

### Changing the registry

Adding a key starts with its entry, in the file of its kind: `doc`,
`default` when it has one, at least one example, and one invalid line with
the parser's message. A line option, a form and a `PARAM` key need the same:
an invalid line, and an example unless the entry is refused. Until the
parser reads the key, the coverage check reports its name as "not in the
parser", and the conformance case of the file fails on its example.

A removed key moves to `[[removed]]` with the parser's hint, and a renamed
key also gets an entry under the name that replaces it. While the registry
keeps the entry of a key the parser does not read, the coverage check
reports its name and the conformance case fails on its examples.

A change of a value, a default, a range or a message changes the entry's
`args`, `default` or message, and its examples. The conformance case fails
on an example the parser refuses and on an invalid line it refuses with
another message; a message that no invalid line shows is listed in
`untested` with the reason. The generated checks fail on a block, a repeat
rule, a range, a word, a `PARAM` key of an optimiser, an option of a line
or a default that differs from the parser's, and on the largest value of a
range when `tadah hpo` refuses it at startup in the file of an example. A
word of a word argument needs an example that writes it, and a word an
example writes there is one of `words`: so `hpotarget_registry.py --check`
fails on a word left out of `words` while an example still shows it.

### Checks

| CTest entry | What fails it |
|---|---|
| `hpo_hpotarget_registry_python` | the loader, on synthetic registries (`tests/test_hpotarget_registry.py`) |
| `hpo_hpotarget_registry_check` | `hpotarget_registry.py --check`: an unknown table or field, a table of an entry (`[[key.invalid]]`, ...) that follows a table of another kind instead of its entry, a wrong type, a missing field, a repeated id, a reference to nothing, a `version` other than 1, a placeholder a message may not hold, a wrapper without exactly one slot line or whose `block` is not that of its slot, a place a line can be without its wrapper, documentation text that addresses the reader or holds a placeholder, a `doc` of more than two sentences, a later example without a doc, a rule without a title, an optimiser or a library without a description, an entry without the cases it needs, an example that puts a key outside its blocks, an argument the synopsis does not name, an invalid line whose error is no message of the registry, a message without an invalid line or an `untested` reason, an example that writes at the place of a word argument a word that is not one of its `words` (compared in any case), a word of `words` that no example writes |
| `hpo_hpotarget_registry_coverage` | a name the parser's sources recognise without an entry, or an entry whose name they do not recognise; the keys of each block, the options of each line and the `STRATEGY` keys of each `TYPE` are compared apart, so an option given to another line fails (`tests/check_hpotarget_coverage.py`; its docstring says how the names are extracted and what the extraction cannot see) |
| `hpo_hpotarget_coverage_tool` | the extraction itself, on synthetic sources (`tests/test_check_hpotarget_coverage.py`) |
| `hpo_hpotarget_registry_conformance` | an example the parser refuses, an invalid line it accepts or refuses with another message (`tests/test_hpotarget_registry_conformance.cpp`), the script of `[lammps_script]` that does not take the candidate potential (`tests/test_hpotarget_script_contract.cpp`): every case of tag `[hpotarget_registry]` |
| `hpo_hpotarget_registry_generated` | a case derived from the structured fields that the parser answers otherwise (`tests/check_hpotarget_generated.py`, which runs the cases through `hpotarget_case_runner`, `tests/tools/hpotarget_case_runner.cpp`; its docstring lists the checks and what they cannot see): the first example of each key in every block, accepted where the key's blocks list it and refused there otherwise; each key and option given twice, as its repeat rule says; the values at and past the limits of the argument of each key, `INIT` method, `PARAM` key and option, in the file of each example (of the first, for an option), the largest value through the startup checks and a `PARAM` value at every limit, a value in the range accepted in one example's file at least; each word of a word list accepted and each refused word refused; each `PARAM` key on each optimiser a wrapper names, accepted where the key lists the optimiser and refused otherwise; each option on every other line; the defaults, against what the code gives a file without the keys |
| `hpo_hpotarget_generated_tool` | the generator, on a synthetic registry (`tests/test_check_hpotarget_generated.py`) |

All seven have the label `docs`. The conformance test reads the TOML when
it runs, so a change to the registry needs no rebuild; it has one Catch2
case per file, which the tag `[hpotarget_main]`, `[hpotarget_optimizer]`,
`[hpotarget_lines]` or `[hpotarget_params]` selects, and a part file is read
with `hpotarget.toml`. `TADAH_HPOTARGET_REGISTRY_DIR` names another
directory to read the registry from. One part file is checked with:

```bash
python3 HPO/blueprint/hpotarget_registry.py --check --part hpotarget_lines.toml
build_tadah/_deps/tadah.hpo-build/tests/test_hpo "[hpotarget_lines]"
```

and the whole registry with:

```bash
python3 HPO/blueprint/hpotarget_registry.py --check
python3 HPO/tests/check_hpotarget_coverage.py HPO
python3 HPO/tests/check_hpotarget_generated.py \
    --runner build_tadah/_deps/tadah.hpo-build/tests/hpotarget_case_runner
ctest --test-dir build_tadah -R hpotarget --output-on-failure
```

With `--part`, a reference into a part file not read is listed as a note,
not checked; the full check resolves it.
