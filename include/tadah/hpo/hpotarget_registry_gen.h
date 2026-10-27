// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: HPO/blueprint/hpotarget.toml and the part files of its
//                  meta.parts, read by HPO/blueprint/hpotarget_registry.py
// Generator:       HPO/blueprint/gen_hpotarget.py --header
//                  (build target hpo_regenerate_hpotarget_registry)
// Drift gate:      ctest hpo_hpotarget_registry_sync (gen_hpotarget.py --check)
//
// The tables of the target-file registry (blueprint/README.md, "The
// generated tables"), in the order of the registry (the placeholder
// sets sorted). Messages are templates: fill() puts the values of their
// {name} placeholders in.
#pragma once

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>
#include <string>
#include <string_view>

namespace tadah {
namespace hpo {
namespace hpotarget_registry {

/* A view of the rows of a table (C++17 has no std::span). */
template <class T>
struct Span {
  const T* data;
  std::size_t size;
  constexpr const T* begin() const { return data; }
  constexpr const T* end() const { return data + size; }
  constexpr const T& operator[](std::size_t i) const { return data[i]; }
  constexpr bool empty() const { return size == 0; }
};

/* A limit of the range of an argument: the registry's text ("0", "0.5",
 * "INT_MAX"), its value, and, for a whole number of 0 or more (INT_MAX,
 * UINT_MAX and UINT64_MAX are), that number exactly. */
struct Limit {
  bool set;
  std::string_view text;
  double value;
  bool whole;
  std::uint64_t whole_value;
};

/* A word an argument refuses, with its message. */
struct RefusedWord {
  std::string_view word;
  std::string_view msg;
};

/* A value of a placeholder: fill() puts `text` where a template writes
 * {name}. */
struct Value {
  std::string_view name;
  std::string_view text;
};

/* The message of the template `tmpl`: each placeholder {name} (a letter or
 * '_', then letters, digits and '_') replaced by the text of the value of
 * that name; ${name}, a LAMMPS variable reference, and every other brace
 * are text. Throws std::invalid_argument for a placeholder without a
 * value. */
inline std::string fill(std::string_view tmpl,
                        std::initializer_list<Value> values)
{
  const auto head = [](char c) {
    return (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || c == '_';
  };
  const auto tail = [&](char c) { return head(c) || (c >= '0' && c <= '9'); };
  std::string out;
  std::size_t i = 0;
  while (i < tmpl.size()) {
    if (tmpl[i] == '{' && (i == 0 || tmpl[i - 1] != '$') &&
        i + 1 < tmpl.size() && head(tmpl[i + 1])) {
      std::size_t j = i + 2;
      while (j < tmpl.size() && tail(tmpl[j])) ++j;
      if (j < tmpl.size() && tmpl[j] == '}') {
        const std::string_view name = tmpl.substr(i + 1, j - i - 1);
        const Value* found = nullptr;
        for (const Value& v : values)
          if (v.name == name) { found = &v; break; }
        if (!found)
          throw std::invalid_argument(
              "fill: no value for the placeholder {" + std::string(name) +
              "} of \"" + std::string(tmpl) + "\"");
        out += found->text;
        i = j + 1;
        continue;
      }
    }
    out += tmpl[i];
    ++i;
  }
  return out;
}

/* ROOT, the top level, and the blocks, in registry order. */
enum class BlockId : unsigned char {
  ROOT,
  OPTIMIZER,
  STRATEGY,
  INIT,
  NOISE,
  EXPLORE,
  REFINE,
  INNER,
  RESTART,
};

constexpr unsigned block_bit(BlockId b)
{
  return 1u << static_cast<unsigned>(b);
}

/* The reader of an argument (blueprint/README.md, `args`). */
enum class ArgKind : unsigned char {
  Bool,
  Indices,
  Integer,
  Loss,
  Name,
  Number,
  Positive,
  Timespec,
  Tokens,
  Word,
};

/* What a key or an option given again does. */
enum class Repeat : unsigned char {
  Add,
  Error,
  LastWins,
  WarnLastWins,
};

/* What a block opened again does. */
enum class BlockRepeat : unsigned char {
  Error,
  ErrorPerParent,
  Merge,
};

/* A stage of the order in which a line is tried (structure::check_order). */
enum class CheckStage : unsigned char {
  BlockLines,
  Removed,
  InnerRefusal,
  GroupKeys,
  InitRestartKeys,
  NoiseKeys,
  LogHp,
  StrategyKeys,
  BlockUnknown,
  RootKeys,
};

/* When a rule is checked: as the file is read, or at startup. */
enum class RuleCheck : unsigned char {
  Parse,
  Startup,
};

/* An OPTIM target the parser refuses, or one it reads with a rule of its own. */
enum class OptimTargetStatus : unsigned char {
  Refused,
  Special,
};

/* The keys, in registry order: kKeys[i].id is KeyId(i). */
enum class KeyId : unsigned short {
  ROOT_FAILSCORE,
  ROOT_PC_SCRIPT_TIMEOUT,
  ROOT_MAXTIME,
  ROOT_PC_ISOLATION,
  ROOT_PC_WORKER_TIMEOUT,
  ROOT_FAILSCORE_GRADING,
  ROOT_BUMPGATE,
  ROOT_RHOGATE,
  ROOT_RHOBARGATE,
  ROOT_QFLUSH,
  ROOT_OUTPUT,
  ROOT_BOUTPUT,
  ROOT_DIGITS,
  ROOT_DUMP,
  ROOT_BDUMP,
  ROOT_OPTIM_SCALE,
  OPTIMIZER_LIB,
  OPTIMIZER_ALGO,
  OPTIMIZER_MAXEVAL,
  OPTIMIZER_MAXTIME,
  OPTIMIZER_STOPVAL,
  OPTIMIZER_FTOL_REL,
  OPTIMIZER_FTOL_ABS,
  OPTIMIZER_XTOL_REL,
  OPTIMIZER_XTOL_ABS,
  OPTIMIZER_POPULATION,
  OPTIMIZER_STEP,
  OPTIMIZER_SEED,
  OPTIMIZER_VECTOR_ARR,
  OPTIMIZER_THREADS,
  OPTIMIZER_GRAD_STRATEGY,
  OPTIMIZER_PARAM,
  OPTIMIZER_LOG_HP,
  STRATEGY_TYPE,
  STRATEGY_N_STARTS,
  STRATEGY_BUDGET_PER_START,
  STRATEGY_TOPK,
  INIT_METHOD,
  INIT_K_TOTAL,
  INIT_SEED,
  INIT_WARM_ON_MISMATCH,
  NOISE_AUTO,
  NOISE_CALIBRATION,
  RESTART_AFTER_STAGNATION,
  RESTART_MAX,
  ROOT_PC_EMAE,
  ROOT_PC_ERMSE,
  ROOT_PC_ErRMSE,
  ROOT_PC_ERSQ,
  ROOT_PC_SRMSE_GPA,
  ROOT_PC_FRMSE,
  ROOT_PC_LAMMPS,
  ROOT_PC_FAILPROBE,
  ROOT_PC_OBSERVABLE,
  ROOT_LOSS,
  ROOT_OPTIM,
};

/* The options of the lines, in registry order. */
enum class LineOptionId : unsigned short {
  RHOGATE_R_LO,
  RHOGATE_NPTS,
  RHOGATE_SEED,
  RHOBARGATE_NPTS,
  RHOBARGATE_SEED,
  PC_EMAE_skipabove,
  PC_EMAE_skiprel,
  PC_LAMMPS_script,
  PC_LAMMPS_varloss,
  PC_LAMMPS_invar,
  PC_LAMMPS_outvar,
  PC_LAMMPS_failscore,
  PC_LAMMPS_maxtime,
  PC_LAMMPS_hotvar,
  PC_LAMMPS_failvar,
  PC_LAMMPS_ncpu,
  PC_FAILPROBE_gate,
  PC_FAILPROBE_s0,
  PC_OBSERVABLE_cohesive_energy_species,
  PC_OBSERVABLE_cohesive_energy_target,
  PC_OBSERVABLE_cohesive_energy_weight,
  PC_OBSERVABLE_cohesive_energy_hotvar,
  PC_OBSERVABLE_cohesive_energy_lattice,
  PC_OBSERVABLE_cohesive_energy_supercell,
  PC_OBSERVABLE_cohesive_energy_loss,
  PC_OBSERVABLE_cohesive_energy_maxiter,
  PC_OBSERVABLE_cohesive_energy_maxeval,
  PC_OBSERVABLE_cohesive_energy_maxwalltime,
  PC_OBSERVABLE_cohesive_energy_etol,
  PC_OBSERVABLE_cohesive_energy_fmaxconverged,
  PC_OBSERVABLE_cohesive_energy_unconverged,
  OPTIM_AUTO_RATIO,
  OPTIM_AUTO_FRAC,
  OPTIM_AUTO_FLOOR,
  OPTIM_AUTO_GROUP_FLOOR,
  OPTIM_LIN,
  OPTIM_LOG,
  OPTIM_ASINH,
};

/* The forms of the lines, in registry order. */
enum class FormId : unsigned char {
  PC_OBSERVABLE_cohesive_energy,
  OPTIM_range,
  OPTIM_AUTO,
};

struct Arg {
  std::string_view name;
  ArgKind kind;
  bool optional;
  bool repeat;
  Limit min;
  Limit max;
  bool min_open;     // min itself is refused
  bool integral;     // a number with no fractional part
  Span<std::string_view> words;
  Span<RefusedWord> refused_words;
  std::string_view what;   // the {what} of the reader's message
  std::string_view msg;    // the key's own message for a refused value
  std::string_view noun;   // the {noun} of positive_msg
  bool has_default;
  std::string_view default_text;
};

/* A word a block reads as another key (INIT's STRATEGY is METHOD). */
struct Alias {
  std::string_view word;
  KeyId key;
};

/* A key of the group a block refuses, and why. */
struct Refusal {
  std::string_view key;
  std::string_view why;
};

struct Block {
  BlockId id;
  std::string_view name;
  std::string_view end;
  unsigned parents;            // block_bit() of each place it opens in
  BlockRepeat repeat;
  std::string_view repeat_msg;
  std::string_view misplaced_msg;
  std::string_view unknown_msg;
  bool unknown_first;
  Span<Alias> aliases;
  Span<Refusal> refuse;
  std::string_view refuse_msg;
  Span<std::string_view> unreachable_msgs;
  Span<std::string_view> requires_checks;   // at the opening line
  Span<std::string_view> close_checks;
  Span<KeyId> keys;            // the keys it takes: valid in it, not refused
};

struct Group {
  std::string_view name;
  unsigned blocks;
  std::string_view leftover_msg;
};

struct Key {
  KeyId id;
  std::string_view id_text;
  std::string_view name;
  std::string_view group;      // empty: valid where `blocks` says
  unsigned blocks;             // block_bit() of each place it is valid in
  Span<Arg> args;
  Repeat repeat;
  std::string_view repeat_msg;
  std::string_view misplaced_msg;
  bool has_default_literal;
  std::string_view default_literal;   // the value as a file writes it
  bool has_alias;
  KeyId alias_of;
  bool refused;                // read only to be refused
  std::string_view env;        // the environment variable that overrides it
  Span<std::string_view> checks;
};

struct LineOption {
  LineOptionId id;
  std::string_view id_text;
  std::string_view option;
  Span<std::string_view> lines;   // the keys and forms whose lines take it
  Span<Arg> args;
  Repeat repeat;
  std::string_view repeat_msg;
  bool has_default_literal;
  std::string_view default_literal;
  bool refused;
  Span<std::string_view> checks;
};

struct Form {
  FormId id;
  std::string_view id_text;
  std::string_view line;
  std::string_view word;       // empty: the form without a word
  Span<Arg> args;
  bool refused;
  Span<std::string_view> checks;
};

struct Removed {
  std::string_view name;
  std::string_view hint;
};

struct Rule {
  std::string_view id_text;
  RuleCheck check;
  std::string_view msg;
};

struct Env {
  std::string_view name;
  KeyId key;
  Span<std::string_view> values;
  std::string_view msg;
  Span<std::string_view> checks;
};

struct LossParam {
  std::string_view name;
  double default_value;
  bool has_gt;
  double gt;                   // a value must be above it
};

struct LossFunction {
  std::string_view name;
  Span<LossParam> params;
};

struct StrategyType {
  std::string_view name;
  Span<KeyId> keys;
  Span<std::string_view> checks;
};

struct InitMethod {
  std::string_view name;
  Span<Arg> args;
  bool refused;
  Span<std::string_view> checks;
};

struct OptimTarget {
  std::string_view key;
  OptimTargetStatus status;
  Span<int> indices;
  std::string_view msg;
};

namespace detail {

inline constexpr std::string_view lexing_bool_true[] = {
    "true",
    "1",
    "yes",
    "on",
};
inline constexpr std::string_view lexing_bool_false[] = {
    "false",
    "0",
    "no",
    "off",
};
inline constexpr std::string_view lexing_timespec_words[] = {
    "off",
    "unlimited",
};
inline constexpr CheckStage structure_check_order[] = {
    CheckStage::BlockLines,
    CheckStage::Removed,
    CheckStage::InnerRefusal,
    CheckStage::GroupKeys,
    CheckStage::InitRestartKeys,
    CheckStage::NoiseKeys,
    CheckStage::LogHp,
    CheckStage::StrategyKeys,
    CheckStage::BlockUnknown,
    CheckStage::RootKeys,
};
inline constexpr std::string_view placeholders_block_misplaced_msg[] = {
    "parent",
    "parent_line",
};
inline constexpr std::string_view placeholders_block_refuse_msg[] = {
    "enclosing",
    "key",
    "why",
};
inline constexpr std::string_view placeholders_block_repeat_msg[] = {
    "first_line",
};
inline constexpr std::string_view placeholders_block_unknown_msg[] = {
    "key",
    "keys",
    "open_line",
};
inline constexpr std::string_view placeholders_group_leftover_msg[] = {
    "key",
    "tok",
};
inline constexpr std::string_view placeholders_lexing_error_frame[] = {
    "line",
    "text",
    "what",
};
inline constexpr std::string_view placeholders_lexing_missing_file_msg[] = {
    "file",
};
inline constexpr std::string_view placeholders_structure_end_mismatch_msg[] = {
    "end",
    "open",
    "open_line",
};
inline constexpr std::string_view placeholders_structure_end_takes_no_value_msg[] = {
    "end",
    "name",
    "tok",
};
inline constexpr std::string_view placeholders_structure_end_without_open_msg[] = {
    "end",
    "name",
};
inline constexpr std::string_view placeholders_structure_open_takes_no_value_msg[] = {
    "name",
    "tok",
};
inline constexpr std::string_view placeholders_structure_removed_msg[] = {
    "hint",
    "key",
};
inline constexpr std::string_view placeholders_structure_root_key_in_block_msg[] = {
    "end",
    "key",
    "name",
    "open_line",
};
inline constexpr std::string_view placeholders_structure_unclosed_msg[] = {
    "end",
    "name",
};
inline constexpr std::string_view placeholders_structure_unknown_root_msg[] = {
    "key",
};
inline constexpr std::string_view placeholders_reading[] = {
    "key",
    "max",
    "min",
    "note",
    "noun",
    "tok",
    "underflow",
    "what",
};
inline constexpr std::string_view placeholders_entry[] = {
    "algo",
    "block",
    "count",
    "default",
    "effect",
    "enclosing",
    "end",
    "env",
    "file",
    "first_line",
    "flag",
    "form",
    "high",
    "hint",
    "home",
    "index",
    "indices",
    "key",
    "keys",
    "kind",
    "lib",
    "line",
    "list",
    "low",
    "max",
    "min",
    "missing",
    "name",
    "note",
    "noun",
    "open",
    "open_line",
    "option",
    "parent",
    "parent_line",
    "phase",
    "script",
    "seed",
    "text",
    "tok",
    "type",
    "underflow",
    "value",
    "var",
    "what",
    "why",
    "width",
};
inline constexpr KeyId BLOCK_OPTIMIZER_keys[] = {
    KeyId::OPTIMIZER_LIB,
    KeyId::OPTIMIZER_ALGO,
    KeyId::OPTIMIZER_MAXEVAL,
    KeyId::OPTIMIZER_MAXTIME,
    KeyId::OPTIMIZER_STOPVAL,
    KeyId::OPTIMIZER_FTOL_REL,
    KeyId::OPTIMIZER_FTOL_ABS,
    KeyId::OPTIMIZER_XTOL_REL,
    KeyId::OPTIMIZER_XTOL_ABS,
    KeyId::OPTIMIZER_POPULATION,
    KeyId::OPTIMIZER_STEP,
    KeyId::OPTIMIZER_SEED,
    KeyId::OPTIMIZER_VECTOR_ARR,
    KeyId::OPTIMIZER_THREADS,
    KeyId::OPTIMIZER_GRAD_STRATEGY,
    KeyId::OPTIMIZER_PARAM,
    KeyId::OPTIMIZER_LOG_HP,
};
inline constexpr std::string_view BLOCK_OPTIMIZER_close_checks[] = {
    "the OPTIMIZER block names no optimiser: {missing}; LIB and ALGO in its STRATEGY block (or directly inside OPTIMIZER) name it",
    "STRATEGY TYPE MULTISTART names no optimiser to run from each start: {missing}; LIB and ALGO in the STRATEGY block name it",
    "STRATEGY TYPE PRESTAGE names no refinement optimiser: {missing}; LIB and ALGO in the STRATEGY block or in a REFINE block name it",
    "{phase}: unknown optimiser {lib}:{algo}: {why}; `tadah explain ALGO` lists the optimisers",
    "{phase}: {lib}:{algo} is refused: {why}",
};
inline constexpr KeyId BLOCK_STRATEGY_keys[] = {
    KeyId::OPTIMIZER_LIB,
    KeyId::OPTIMIZER_ALGO,
    KeyId::OPTIMIZER_MAXEVAL,
    KeyId::OPTIMIZER_MAXTIME,
    KeyId::OPTIMIZER_STOPVAL,
    KeyId::OPTIMIZER_FTOL_REL,
    KeyId::OPTIMIZER_FTOL_ABS,
    KeyId::OPTIMIZER_XTOL_REL,
    KeyId::OPTIMIZER_XTOL_ABS,
    KeyId::OPTIMIZER_POPULATION,
    KeyId::OPTIMIZER_STEP,
    KeyId::OPTIMIZER_SEED,
    KeyId::OPTIMIZER_VECTOR_ARR,
    KeyId::OPTIMIZER_THREADS,
    KeyId::OPTIMIZER_GRAD_STRATEGY,
    KeyId::OPTIMIZER_PARAM,
    KeyId::OPTIMIZER_LOG_HP,
    KeyId::STRATEGY_TYPE,
    KeyId::STRATEGY_N_STARTS,
    KeyId::STRATEGY_BUDGET_PER_START,
    KeyId::STRATEGY_TOPK,
};
inline constexpr std::string_view BLOCK_STRATEGY_close_checks[] = {
    "{key} is not accepted with STRATEGY TYPE {type}{note}: {effect}, so it would have no effect; {key} takes effect under STRATEGY TYPE {home}",
};
inline constexpr Alias BLOCK_INIT_aliases[] = {
    Alias{"STRATEGY", KeyId::INIT_METHOD},
};
inline constexpr KeyId BLOCK_INIT_keys[] = {
    KeyId::INIT_METHOD,
    KeyId::INIT_K_TOTAL,
    KeyId::INIT_SEED,
    KeyId::INIT_WARM_ON_MISMATCH,
};
inline constexpr std::string_view BLOCK_INIT_unreachable[] = {
    "INIT block requires an active OptimizerSpec",
};
inline constexpr KeyId BLOCK_NOISE_keys[] = {
    KeyId::NOISE_AUTO,
    KeyId::NOISE_CALIBRATION,
};
inline constexpr KeyId BLOCK_EXPLORE_keys[] = {
    KeyId::OPTIMIZER_LIB,
    KeyId::OPTIMIZER_ALGO,
    KeyId::OPTIMIZER_MAXEVAL,
    KeyId::OPTIMIZER_MAXTIME,
    KeyId::OPTIMIZER_STOPVAL,
    KeyId::OPTIMIZER_FTOL_REL,
    KeyId::OPTIMIZER_FTOL_ABS,
    KeyId::OPTIMIZER_XTOL_REL,
    KeyId::OPTIMIZER_XTOL_ABS,
    KeyId::OPTIMIZER_POPULATION,
    KeyId::OPTIMIZER_STEP,
    KeyId::OPTIMIZER_SEED,
    KeyId::OPTIMIZER_VECTOR_ARR,
    KeyId::OPTIMIZER_THREADS,
    KeyId::OPTIMIZER_GRAD_STRATEGY,
    KeyId::OPTIMIZER_PARAM,
};
inline constexpr std::string_view BLOCK_EXPLORE_requires[] = {
    "EXPLORE block is only valid with TYPE PRESTAGE (declare TYPE before EXPLORE)",
};
inline constexpr std::string_view BLOCK_EXPLORE_close_checks[] = {
    "the EXPLORE block names no optimiser: {missing}; LIB and ALGO inside the EXPLORE block name the optimiser of the exploration of STRATEGY TYPE PRESTAGE",
    "the EXPLORE block: unknown optimiser {lib}:{algo}: {why}; `tadah explain ALGO` lists the optimisers",
    "the EXPLORE block: {lib}:{algo} is refused: {why}",
};
inline constexpr KeyId BLOCK_REFINE_keys[] = {
    KeyId::OPTIMIZER_LIB,
    KeyId::OPTIMIZER_ALGO,
    KeyId::OPTIMIZER_MAXEVAL,
    KeyId::OPTIMIZER_MAXTIME,
    KeyId::OPTIMIZER_STOPVAL,
    KeyId::OPTIMIZER_FTOL_REL,
    KeyId::OPTIMIZER_FTOL_ABS,
    KeyId::OPTIMIZER_XTOL_REL,
    KeyId::OPTIMIZER_XTOL_ABS,
    KeyId::OPTIMIZER_POPULATION,
    KeyId::OPTIMIZER_STEP,
    KeyId::OPTIMIZER_SEED,
    KeyId::OPTIMIZER_VECTOR_ARR,
    KeyId::OPTIMIZER_THREADS,
    KeyId::OPTIMIZER_GRAD_STRATEGY,
    KeyId::OPTIMIZER_PARAM,
};
inline constexpr std::string_view BLOCK_REFINE_requires[] = {
    "REFINE block is only valid with TYPE PRESTAGE (declare TYPE before REFINE)",
};
inline constexpr Refusal BLOCK_INNER_refuse[] = {
    Refusal{"SEED",
        "NLopt draws the INNER search from the enclosing run's random stream"},
    Refusal{"GRAD_STRATEGY",
        "the enclosing block's GRAD_STRATEGY applies to its local search"},
    Refusal{"STOPVAL",
        "NLopt's MLSL and AUGLAG replace the INNER search's STOPVAL with the enclosing block's"},
};
inline constexpr KeyId BLOCK_INNER_keys[] = {
    KeyId::OPTIMIZER_LIB,
    KeyId::OPTIMIZER_ALGO,
    KeyId::OPTIMIZER_MAXEVAL,
    KeyId::OPTIMIZER_MAXTIME,
    KeyId::OPTIMIZER_FTOL_REL,
    KeyId::OPTIMIZER_FTOL_ABS,
    KeyId::OPTIMIZER_XTOL_REL,
    KeyId::OPTIMIZER_XTOL_ABS,
    KeyId::OPTIMIZER_POPULATION,
    KeyId::OPTIMIZER_STEP,
    KeyId::OPTIMIZER_VECTOR_ARR,
    KeyId::OPTIMIZER_THREADS,
    KeyId::OPTIMIZER_PARAM,
};
inline constexpr std::string_view BLOCK_INNER_unreachable[] = {
    "INNER block has no parent OptimizerSpec",
};
inline constexpr std::string_view BLOCK_INNER_close_checks[] = {
    "the INNER block inside {enclosing} names no optimiser: {missing}; LIB and ALGO inside the INNER block name the local optimiser that NLopt's MLSL and AUGLAG run",
    "the INNER block inside {enclosing}: unknown optimiser {lib}:{algo}: {why}; `tadah explain ALGO` lists the optimisers",
    "the INNER block inside {enclosing}: {lib}:{algo} is refused: {why}",
};
inline constexpr Alias BLOCK_RESTART_aliases[] = {
    Alias{"STRATEGY", KeyId::INIT_METHOD},
};
inline constexpr KeyId BLOCK_RESTART_keys[] = {
    KeyId::INIT_METHOD,
    KeyId::INIT_K_TOTAL,
    KeyId::INIT_SEED,
    KeyId::INIT_WARM_ON_MISMATCH,
    KeyId::RESTART_AFTER_STAGNATION,
    KeyId::RESTART_MAX,
};
inline constexpr std::string_view BLOCK_RESTART_close_checks[] = {
    "the RESTART block never restarts the search: a restart needs AFTER_STAGNATION and MAX, and the block has {missing}; a search without restarts has no RESTART block",
};
inline constexpr Arg ROOT_FAILSCORE_args[] = {
    Arg{"loss", ArgKind::Positive, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "finite number", false,
        ""},
};
inline constexpr Arg ROOT_PC_SCRIPT_TIMEOUT_args[] = {
    Arg{"time", ArgKind::Timespec, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view ROOT_PC_SCRIPT_TIMEOUT_checks[] = {
    "{key} sets the script timeout of the PC_LAMMPS and PC_FAILPROBE lines after it that give no --maxtime, and no such line follows it; it would have no effect",
};
inline constexpr Arg ROOT_MAXTIME_args[] = {
    Arg{"time", ArgKind::Timespec, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view ROOT_MAXTIME_checks[] = {
    "{key} sets the script timeout of the PC_LAMMPS and PC_FAILPROBE lines after it that give no --maxtime, and no such line follows it; it would have no effect",
};
inline constexpr std::string_view ROOT_PC_ISOLATION_arg0_words[] = {
    "THREAD",
    "PROCESS",
};
inline constexpr Arg ROOT_PC_ISOLATION_args[] = {
    Arg{"mode", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::ROOT_PC_ISOLATION_arg0_words, 2},
        Span<RefusedWord>{nullptr, 0}, "",
        "PC_ISOLATION must be THREAD or PROCESS, got '{tok}'", "", false, ""},
};
inline constexpr std::string_view ROOT_PC_ISOLATION_checks[] = {
    "PC_ISOLATION PROCESS is not supported in the MPI build (TADAH_HPO_MPI): the worker processes cannot initialize MPI. Use the serial build, or PC_ISOLATION THREAD.",
};
inline constexpr Arg ROOT_PC_WORKER_TIMEOUT_args[] = {
    Arg{"seconds", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_WORKER_TIMEOUT must be a finite number of seconds >= 0 (0 derives the deadline from each script's timeout), got '{tok}'{underflow}",
        "", false, ""},
};
inline constexpr std::string_view ROOT_PC_WORKER_TIMEOUT_checks[] = {
    "PC_WORKER_TIMEOUT is too large for the worker deadline, got '{tok}' (it is kept in 64-bit milliseconds: at most about 9.2e15 s)",
    "PC_WORKER_TIMEOUT sets the deadline of a script task in a worker process (PC_ISOLATION PROCESS); under PC_ISOLATION THREAD, {why}, the scripts run inside the tadah process without a worker deadline, so it would have no effect",
};
inline constexpr std::string_view ROOT_FAILSCORE_GRADING_arg0_words[] = {
    "ON",
    "OFF",
};
inline constexpr Arg ROOT_FAILSCORE_GRADING_args[] = {
    Arg{"mode", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::ROOT_FAILSCORE_GRADING_arg0_words, 2},
        Span<RefusedWord>{nullptr, 0}, "",
        "FAILSCORE_GRADING must be ON or OFF, got '{tok}'", "", false, ""},
};
inline constexpr std::string_view ROOT_FAILSCORE_GRADING_checks[] = {
    "{phase}: ASA:ASA: under FAILSCORE_GRADING ON a failed evaluation scores from 0.999 FAILSCORE to below FAILSCORE ({value}), and ASA:ASA takes a cost of about 1e+18 or more for an invalid state and draws another, so it would not see the graded failures at or above that cost; a FAILSCORE of 1e+18 or less keeps every graded failure below it",
};
inline constexpr std::string_view ROOT_BUMPGATE_arg0_words[] = {
    "CANDIDATE",
    "STRICT",
};
inline constexpr Arg ROOT_BUMPGATE_args[] = {
    Arg{"mode", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::ROOT_BUMPGATE_arg0_words, 2},
        Span<RefusedWord>{nullptr, 0}, "",
        "BUMPGATE must be CANDIDATE or STRICT, got '{tok}'", "", false, ""},
};
inline constexpr std::string_view ROOT_BUMPGATE_checks[] = {
    "bump-support gate: with a hard (Cut_Dummy) cutoff every compact bump must vanish at or before RCUT (c + w/eta <= RCUT) — a truncated overhang is an energy/force STEP at the cutoff. Violations:{list}\n  To search longer range, raise RCUT2B/RCUTMB in the config (the potential stays smooth: bumps still vanish at their own support edge).",
};
inline constexpr Arg ROOT_RHOGATE_args[] = {
    Arg{"tol", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "RHOGATE tol", "RHOGATE tol must be >= 0, got '{tok}'", "", false, ""},
};
inline constexpr std::string_view ROOT_RHOGATE_checks[] = {
    "RHOGATE: the tolerance is missing; expected RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]",
    "RHOGATE: unexpected token '{tok}'; expected RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]",
    "{keys} {what} the density of one DM_REAM many-body term or of one DM_Embed component, and {why}",
    "RHOGATE: under {form} every density rho(r) is >= 0, so the sign condition of RHOGATE holds for every candidate; RHOGATE is taken under RHOFORM SIGNED only (RHOBARGATE under every form)",
    "{keys}: the many-body cutoff RCUTMB is not set.",
    "{keys}: {why} on every candidate (WATOM has no OPTIM line, or its OPTIM box does not reach above 0), so every evaluation would be refused before training, under SEED ERROR and SEED WARN alike. The checks of {keys} need every density pair weight > 0, which a WATOM above 0 gives (under RHOFORM SIGNED the sign of rho is that of the AMPGRIDMB amplitudes); without the {keys} line(s) the run makes no such check.",
    "RHOGATE: seed {name} fails the rho(r) sign gate: {why}. A seed that fails is refused under SEED ERROR, the default, and is evaluated as a rejected start under SEED WARN. tadah refit --edit changes the density of a potential file; a WARM seed takes the values of its search parameters from its file and the others from the training configuration.",
};
inline constexpr std::string_view ROOT_RHOBARGATE_arg0_words[] = {
    "bcc",
    "fcc",
    "hcp",
};
inline constexpr Arg ROOT_RHOBARGATE_args[] = {
    Arg{"lattice", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::ROOT_RHOBARGATE_arg0_words, 3},
        Span<RefusedWord>{nullptr, 0}, "",
        "RHOBARGATE lattice must be bcc, fcc or hcp, got '{tok}'", "", false,
        ""},
    Arg{"a0", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "RHOBARGATE a0", "", "", false, ""},
    Arg{"f_lo", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "RHOBARGATE f_lo", "", "", false, ""},
    Arg{"f_hi", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "RHOBARGATE f_hi", "", "", false, ""},
};
inline constexpr std::string_view ROOT_RHOBARGATE_checks[] = {
    "RHOBARGATE: the lattice, a0, f_lo and f_hi are required; expected RHOBARGATE <bcc|fcc|hcp> <a0> <f_lo> <f_hi> [NPTS <n>] [SEED ERROR|WARN]",
    "RHOBARGATE a0 must be > 0, got '{tok}'",
    "RHOBARGATE needs 0 < f_lo < f_hi, got f_lo '{low}', f_hi '{high}'",
    "RHOBARGATE: unexpected token '{tok}'; expected RHOBARGATE <bcc|fcc|hcp> <a0> <f_lo> <f_hi> [NPTS <n>] [SEED ERROR|WARN]",
    "{keys} {what} the density of one DM_REAM many-body term or of one DM_Embed component, and {why}",
    "RHOBARGATE: f_lo = {low} puts the largest reachable RCUTMB, {max} A, at {value} lattice constants of the smallest lattice (a = f_lo a0 = {min} A); the limit is 20. f_lo and f_hi are fractions of a0 (0.75 scans down to 25 % compression), and a larger f_lo puts that RCUTMB at fewer lattice constants.",
    "RHOBARGATE: seed {name} fails the lattice rhobar(a) gate: {why}. A seed that fails is refused under SEED ERROR, the default, and is evaluated as a rejected start under SEED WARN. tadah refit --edit changes the density of a potential file; a WARM seed takes the values of its search parameters from its file and the others from the training configuration.",
    "RHOBARGATE: the first {name} shell at a = f_hi a0 = {high} A lies at r = {value} A, beyond the smallest cutoff a candidate can have, RCUTMB = {max} A{note}: rhobar(a) would lose its nearest neighbours; a smaller f_hi or a0, or a larger RCUTMB, keeps the first shell inside the cutoff.",
};
inline constexpr Arg ROOT_QFLUSH_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr Arg ROOT_OUTPUT_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}",
        "OUTPUT must be an integer >= 1, got '{tok}' (it is a cadence: write every Nth evaluation; to write as little as possible, use a large value such as OUTPUT 1000000)",
        "", false, ""},
};
inline constexpr Arg ROOT_BOUTPUT_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}",
        "BOUTPUT must be an integer >= 1, got '{tok}' (it is a cadence: write every Nth new best)",
        "", false, ""},
};
inline constexpr Arg ROOT_DIGITS_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "15", 15.0, true, 15ull}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key}", "", "", false, ""},
};
inline constexpr Arg ROOT_DUMP_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} cadence", "", "", false, ""},
    Arg{"directory", ArgKind::Name, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view ROOT_DUMP_checks[] = {
    "{key} requires a cadence and a directory",
    "{key} takes a cadence and a directory (unexpected '{tok}')",
    "{key} directory '{name}' exists and is not a directory",
    "{key} directory '{name}' is not writable",
    "{key} directory '{name}' is a symbolic link to '{file}', which does not exist",
    "{key} directory '{name}' cannot be created: the directory '{parent}' does not exist (tadah hpo creates only the last directory of the path)",
    "{key} directory '{name}' cannot be created: '{parent}' is not a directory",
    "{key} directory '{name}' cannot be created: the directory '{parent}' is not writable",
    "{key} directory '{name}' cannot be read: {why}",
    "{key} directory '{name}' cannot be resolved: {why}",
};
inline constexpr Arg ROOT_BDUMP_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} cadence", "", "", false, ""},
    Arg{"directory", ArgKind::Name, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view ROOT_BDUMP_checks[] = {
    "{key} requires a cadence and a directory",
    "{key} takes a cadence and a directory (unexpected '{tok}')",
    "{key} directory '{name}' exists and is not a directory",
    "{key} directory '{name}' is not writable",
    "{key} directory '{name}' is a symbolic link to '{file}', which does not exist",
    "{key} directory '{name}' cannot be created: the directory '{parent}' does not exist (tadah hpo creates only the last directory of the path)",
    "{key} directory '{name}' cannot be created: '{parent}' is not a directory",
    "{key} directory '{name}' cannot be created: the directory '{parent}' is not writable",
    "{key} directory '{name}' cannot be read: {why}",
    "{key} directory '{name}' cannot be resolved: {why}",
};
inline constexpr std::string_view ROOT_OPTIM_SCALE_arg0_words[] = {
    "AUTO",
    "NONE",
};
inline constexpr Arg ROOT_OPTIM_SCALE_args[] = {
    Arg{"mode", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::ROOT_OPTIM_SCALE_arg0_words, 2},
        Span<RefusedWord>{nullptr, 0}, "",
        "OPTIM_SCALE must be AUTO (every OPTIM dimension is mapped to [0, 1]) or NONE (each dimension stays in its OPTIM units: the value for LIN, its natural logarithm for LOG, asinh(x/c) for ASINH <c>), got '{tok}'",
        "", false, ""},
};
inline constexpr Arg OPTIMIZER_LIB_args[] = {
    Arg{"library", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr Arg OPTIMIZER_ALGO_args[] = {
    Arg{"algorithm", ArgKind::Name, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr Arg OPTIMIZER_MAXEVAL_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_MAXEVAL_checks[] = {
    "{phase}: {lib}:{algo} needs a stop: {why}; set {list}",
    "{phase}: TADAH:ANNEAL: CHAIN_LENGTH AUTO spreads MAXEVAL over the temperature steps, and this search has no MAXEVAL; without MAXEVAL, PARAM CHAIN_LENGTH gives the number of proposals at each temperature step",
    "{phase}: TADAH:ANNEAL: NT AUTO follows from MAXEVAL, and this search has no MAXEVAL; without MAXEVAL, PARAM NT gives the number of step adjustments of the Corana kernel at each temperature step",
};
inline constexpr Arg OPTIMIZER_MAXTIME_args[] = {
    Arg{"seconds", ArgKind::Positive, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "number of seconds", false,
        ""},
};
inline constexpr std::string_view OPTIMIZER_MAXTIME_checks[] = {
    "{phase}: {lib}:{algo} does not support MAXTIME",
};
inline constexpr Arg OPTIMIZER_STOPVAL_args[] = {
    Arg{"loss", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_STOPVAL_checks[] = {
    "{phase}: {lib}:{algo} does not support STOPVAL",
};
inline constexpr Arg OPTIMIZER_FTOL_REL_args[] = {
    Arg{"tolerance", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_FTOL_REL_checks[] = {
    "{phase}: {lib}:{algo} does not support FTOL_REL with an INNER block: {why}; set it in the INNER block, whose local search tests it",
    "{phase}: CERES:{algo} requires FTOL_REL >= 0 (Ceres' function_tolerance), got {value}",
    "{phase}: DLIB:BFGS does not support FTOL_REL; use FTOL_ABS",
    "{phase}: ASA:ASA does not take FTOL_REL or FTOL_ABS: ASA's own stops end the search, PARAM Cost_Precision with Maximum_Cost_Repeat (a plateau of the loss) or Accepted_To_Generated_Ratio",
    "{phase}: TADAH:{algo}: FTOL_REL must be finite and non-negative",
    "{phase}: {lib}:{algo} does not support FTOL_REL{note}",
};
inline constexpr Arg OPTIMIZER_FTOL_ABS_args[] = {
    Arg{"tolerance", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_FTOL_ABS_checks[] = {
    "{phase}: CERES:{algo} uses relative tolerance; use FTOL_REL (mapped to function_tolerance) instead of FTOL_ABS",
    "{phase}: ASA:ASA does not take FTOL_REL or FTOL_ABS: ASA's own stops end the search, PARAM Cost_Precision with Maximum_Cost_Repeat (a plateau of the loss) or Accepted_To_Generated_Ratio",
    "{phase}: TADAH:{algo}: FTOL_ABS must be finite and non-negative",
    "{phase}: DLIB:GFS: FTOL_ABS is the accuracy of dlib's solver and must be >= 0 (got {value})",
    "{phase}: {lib}:{algo} does not support FTOL_ABS{note}",
    "{phase}: {lib}:{algo} does not support FTOL_ABS with an INNER block: {why}; set it in the INNER block, whose local search tests it",
};
inline constexpr Arg OPTIMIZER_XTOL_REL_args[] = {
    Arg{"tolerance", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_XTOL_REL_checks[] = {
    "{phase}: CERES:{algo} requires XTOL_REL >= 0 (Ceres' parameter_tolerance), got {value}",
    "{phase}: DLIB:BFGS does not support XTOL_REL",
    "{phase}: ASA:ASA does not take XTOL_REL or XTOL_ABS",
    "{phase}: TADAH:{algo} does not use XTOL_REL or XTOL_ABS: it has no convergence test on the point; FTOL_REL and FTOL_ABS end it on a plateau of the loss (the eps-NEPS test)",
    "{phase}: {lib}:{algo} does not support XTOL_REL{note}",
    "{phase}: {lib}:{algo} does not support XTOL_REL with an INNER block: {why}; set it in the INNER block, whose local search tests it",
};
inline constexpr Arg OPTIMIZER_XTOL_ABS_args[] = {
    Arg{"tolerance", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_XTOL_ABS_checks[] = {
    "{phase}: CERES:{algo} uses relative tolerance; use XTOL_REL (mapped to parameter_tolerance) instead of XTOL_ABS",
    "{phase}: DLIB:BFGS does not support XTOL_ABS",
    "{phase}: ASA:ASA does not take XTOL_REL or XTOL_ABS",
    "{phase}: TADAH:{algo} does not use XTOL_REL or XTOL_ABS: it has no convergence test on the point; FTOL_REL and FTOL_ABS end it on a plateau of the loss (the eps-NEPS test)",
    "{phase}: {lib}:{algo} does not support XTOL_ABS{note}",
    "{phase}: {lib}:{algo} does not support XTOL_ABS with an INNER block: {why}; set it in the INNER block, whose local search tests it",
};
inline constexpr Arg OPTIMIZER_POPULATION_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "UINT_MAX", 4294967295.0, true, 4294967295ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_POPULATION_checks[] = {
    "{phase}: TADAH:ANNEAL_PA: POPULATION must be >= 8",
    "{phase}: TADAH:{algo} does not use POPULATION: TADAH:ANNEAL_PA runs a population",
    "{phase}: DLIB:BFGS does not support POPULATION",
    "{phase}: CERES:{algo} does not support POPULATION",
    "{phase}: ASA:ASA does not take POPULATION: ASA runs one chain; PARAM Gener_Block sets the number of states it generates at a time at the start (the block then adapts up to Gener_Block_Max unless Gener_Mov_Avr is 0), and the THREADS workers evaluate each block together",
    "{phase}: {lib}:{algo} does not use POPULATION; remove it (it is read by {list})",
};
inline constexpr Arg OPTIMIZER_STEP_args[] = {
    Arg{"KEY", ArgKind::Name, true, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"indices", ArgKind::Indices, true, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
    Arg{"fraction", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "0.5", 0.5, false, 0ull}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_STEP_checks[] = {
    "{phase}: {lib}:{algo} does not use STEP; {why}",
    "STEP override '{text}': expected KEY(indices) <frac>",
    "STEP override '{text}': empty index list",
    "STEP override '{text}': bad index list ({why})",
    "STEP override names OPTIM key '{key}' which is not optimised (no OPTIM line for it)",
    "STEP {key}({index}): index {index} is not an OPTIM'd index of {key}",
    "STEP overrides cover only some dimensions and no block-wide STEP is set; add `STEP <frac>` or override every OPTIM'd dimension",
    "{phase}: {lib}:{algo}: STEP must be > 0 and finite (got {value})",
    "{phase}: {lib}:{algo} STEP {key}: STEP must be > 0 and finite (got {value})",
    "{phase}: {lib}:{algo}: STEP must be a fraction of the OPTIM width in (0, 0.5] (got {value}); values above 0.5 cannot satisfy BOBYQA's width >= 2*step rule and exceed NLopt's own 0.25 default",
    "{phase}: {lib}:{algo} STEP {key}: STEP must be a fraction of the OPTIM width in (0, 0.5] (got {value}); values above 0.5 cannot satisfy BOBYQA's width >= 2*step rule and exceed NLopt's own 0.25 default",
    "STEP {key} is set twice in the {block} block (first on line {first_line}); each key is given once",
};
inline constexpr Arg OPTIMIZER_SEED_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "UINT64_MAX", 1.8446744073709552e+19, true, 18446744073709551615ull},
        false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_SEED_checks[] = {
    "{phase}: {lib}:{algo} does not use SEED: it draws no random numbers",
    "{phase}: {lib}:{algo} does not use SEED: in one OPTIM dimension it draws no random numbers (it draws them only to choose between equally long sides of a box)",
    "{phase}: {lib}:{algo} does not support SEED",
    "{phase}: {lib}:{algo} does not use SEED: it draws no random numbers, and the exploration ({name}) {why}",
};
inline constexpr Arg OPTIMIZER_VECTOR_ARR_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "UINT_MAX", 4294967295.0, true, 4294967295ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_VECTOR_ARR_checks[] = {
    "{phase}: {lib}:{algo} does not use VECTOR_ARR; remove it (it is read by {list})",
    "{phase}: DLIB:BFGS does not support VECTOR_ARR; use ALGO GFS or an NLopt L-BFGS variant instead",
    "{phase}: CERES:{algo} does not support VECTOR_ARR; use DLIB:GFS or an NLopt global algorithm",
    "{phase}: ASA:ASA does not accept VECTOR_ARR",
    "{phase}: TADAH:{algo} does not use VECTOR_ARR: it keeps no gradient history",
};
inline constexpr Arg OPTIMIZER_THREADS_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_THREADS_checks[] = {
    "{phase}: {lib}:{algo} does not use THREADS in an INNER block: the THREADS of the STRATEGY-level optimiser sets the workers of the run; remove it",
    "{phase}: {lib}:{algo} does not use THREADS: the THREADS of the STRATEGY-level optimiser sets the workers of the run, and the THREADS of an EXPLORE block is read by {list}; remove it",
};
inline constexpr std::string_view OPTIMIZER_GRAD_STRATEGY_arg0_words[] = {
    "NUMERICAL",
    "ANALYTICAL",
    "HYBRID",
};
inline constexpr Arg OPTIMIZER_GRAD_STRATEGY_args[] = {
    Arg{"strategy", ArgKind::Word, false, false, Limit{}, Limit{}, false,
        false,
        Span<std::string_view>{detail::OPTIMIZER_GRAD_STRATEGY_arg0_words, 3},
        Span<RefusedWord>{nullptr, 0}, "",
        "GRAD_STRATEGY must be one of NUMERICAL, ANALYTICAL, HYBRID (got '{tok}')",
        "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_GRAD_STRATEGY_checks[] = {
    "GRAD_STRATEGY takes one value: NUMERICAL, ANALYTICAL or HYBRID",
    "{block} block: GRAD_STRATEGY ANALYTICAL{note} needs the analytical gradient of the loss, which this setup cannot form: {why}. Change that setting, or set GRAD_STRATEGY HYBRID to use finite differences where no analytical gradient is available.",
    "{block} block: GRAD_STRATEGY ANALYTICAL{note} needs an analytical derivative with respect to every search parameter, and the descriptors give none for {list}. GRAD_STRATEGY HYBRID takes finite differences for {what}.",
    "{phase}: ASA:ASA does not accept GRAD_STRATEGY: it is gradient-free, and the reannealing takes its tangents by finite differences (PARAM Delta_X)",
    "{phase}: TADAH:{algo} does not use GRAD_STRATEGY: it is gradient-free",
    "{phase} {lib}:{algo}: GRAD_STRATEGY never takes effect: {lib}:{algo} takes no gradient{note}; remove it.",
};
inline constexpr Arg OPTIMIZER_PARAM_args[] = {
    Arg{"name", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"value", ArgKind::Tokens, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view OPTIMIZER_PARAM_checks[] = {
    "PARAM requires a name and a value",
    "PARAM {key} is set twice in the {block} block (first on line {first_line}); each key is given once",
};
inline constexpr Arg OPTIMIZER_LOG_HP_args[] = {
    Arg{"ratio", ArgKind::Number, false, false,
        Limit{true, "1", 1.0, true, 1ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key}",
        "LOG_HP must be a finite ratio >= 1 (a dimension is log-scaled when high/low >= LOG_HP and low > 0), got '{tok}'",
        "", false, ""},
};
inline constexpr std::string_view STRATEGY_TYPE_arg0_words[] = {
    "PLAIN",
    "MULTISTART",
    "PRESTAGE",
};
inline constexpr Arg STRATEGY_TYPE_args[] = {
    Arg{"type", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::STRATEGY_TYPE_arg0_words, 3},
        Span<RefusedWord>{nullptr, 0}, "",
        "STRATEGY TYPE must be PLAIN, MULTISTART, or PRESTAGE (got '{tok}')",
        "", false, ""},
};
inline constexpr std::string_view STRATEGY_TYPE_checks[] = {
    "STRATEGY TYPE MULTISTART needs N_STARTS 2 or more: without N_STARTS it runs one start, which is STRATEGY TYPE PLAIN",
};
inline constexpr Arg STRATEGY_N_STARTS_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "2", 2.0, true, 2ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}",
        "N_STARTS must be an integer >= 2, got '{tok}' (one start is STRATEGY TYPE PLAIN)",
        "", false, ""},
};
inline constexpr std::string_view STRATEGY_N_STARTS_checks[] = {
    "N_STARTS must be an integer >= 2, got '{tok}'",
    "N_STARTS {value} needs {value} starting points, and the INIT chain gives {count} ({list}); {hint}",
    "N_STARTS {value} needs {value} starting points in each RESTART round, and the RESTART chain gives {count} ({list}); {hint}",
};
inline constexpr Arg STRATEGY_BUDGET_PER_START_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr Arg STRATEGY_TOPK_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "1000000", 1000000.0, true, 1000000ull}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key}", "", "", false, ""},
};
inline constexpr Arg INIT_METHOD_args[] = {
    Arg{"method", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "{key} {tok}: unknown method; the methods are CONFIG, RANDOM, LHS, SOBOL and WARM, in upper case",
        "", false, ""},
    Arg{"value", ArgKind::Tokens, true, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view INIT_METHOD_checks[] = {
    "{key} requires a method name",
};
inline constexpr Arg INIT_K_TOTAL_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view INIT_K_TOTAL_checks[] = {
    "K_TOTAL {value} caps nothing: the METHOD lines of {parent} give at most {count} starting points ({list}), and K_TOTAL keeps the first K_TOTAL starting points",
    "K_TOTAL {value} caps nothing: the METHOD lines of {parent} give at most 1 starting point ({list}), and K_TOTAL keeps the first K_TOTAL starting points",
    "K_TOTAL {value} caps nothing: a block without a METHOD line gives one starting point (the CONFIG point), and K_TOTAL keeps the first K_TOTAL starting points",
    "K_TOTAL {value} caps nothing: an INIT block inside EXPLORE without a METHOD line gives the exploration the starting points of the INIT blocks inside OPTIMIZER, STRATEGY and REFINE (the CONFIG point without one), and K_TOTAL caps only the points of the block's own METHOD lines",
};
inline constexpr Arg INIT_SEED_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "UINT64_MAX", 1.8446744073709552e+19, true, 18446744073709551615ull},
        false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view INIT_WARM_ON_MISMATCH_arg0_words[] = {
    "warn",
    "error",
    "skip",
};
inline constexpr Arg INIT_WARM_ON_MISMATCH_args[] = {
    Arg{"policy", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::INIT_WARM_ON_MISMATCH_arg0_words, 3},
        Span<RefusedWord>{nullptr, 0}, "",
        "WARM_ON_MISMATCH must be one of: warn, error, skip", "", false, ""},
};
inline constexpr Arg NOISE_AUTO_args[] = {
    Arg{"bool", ArgKind::Bool, true, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "NOISE.AUTO", "", "", false, ""},
};
inline constexpr std::string_view NOISE_AUTO_checks[] = {
    "{phase} {lib}:{algo}: NOISE AUTO has no effect: {why}; remove AUTO from the NOISE block.",
};
inline constexpr Arg NOISE_CALIBRATION_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}", "", "", false, ""},
};
inline constexpr std::string_view NOISE_CALIBRATION_checks[] = {
    "NOISE.CALIBRATION must be 0 (off) or >= 2",
};
inline constexpr Arg RESTART_AFTER_STAGNATION_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "UINT64_MAX", 1.8446744073709552e+19, true, 18446744073709551615ull},
        false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}",
        "AFTER_STAGNATION must be an integer >= 1, got '{tok}' (0 never restarts the search; a search without restarts has no RESTART block)",
        "", false, ""},
};
inline constexpr std::string_view RESTART_AFTER_STAGNATION_checks[] = {
    "AFTER_STAGNATION must be an integer >= 1, got '{tok}'",
};
inline constexpr Arg RESTART_MAX_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "UINT64_MAX", 1.8446744073709552e+19, true, 18446744073709551615ull},
        false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key}",
        "MAX must be an integer >= 1, got '{tok}' (0 never restarts the search; a search without restarts has no RESTART block)",
        "", false, ""},
};
inline constexpr std::string_view RESTART_MAX_checks[] = {
    "MAX must be an integer >= 1, got '{tok}'",
};
inline constexpr Arg ROOT_PC_EMAE_args[] = {
    Arg{"target", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} <target>", "", "", false, ""},
    Arg{"weight", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} <weight>", "{key} <weight> must be a number >= 0, got '{tok}'",
        "", false, ""},
};
inline constexpr std::string_view ROOT_PC_EMAE_checks[] = {
    "tadah hpo: {keys} on the validation set: no dataset has a structure with an energy ({list}): every structure has energy priority 0 (eweight), so the energy metrics have nothing to measure",
    "{key} requires <target> <weight>",
    "{key}: unknown flag '{tok}'",
    "{key}: indexed targets are supported only for PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA",
};
inline constexpr Arg ROOT_PC_ERMSE_args[] = {
    Arg{"indices", ArgKind::Indices, true, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
    Arg{"target", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} <target>", "", "", false, ""},
    Arg{"weight", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} <weight>", "{key} <weight> must be a number >= 0, got '{tok}'",
        "", false, ""},
};
inline constexpr std::string_view ROOT_PC_ERMSE_checks[] = {
    "{key} requires validation DBFILE entries",
    "{key}: dataset {index} on line {line} is outside validation DBFILE range 1-{count}",
    "{key}: missing target and weight for:{list}",
    "tadah hpo: {keys} on the validation set: no dataset has a structure with an energy ({list}): every structure has energy priority 0 (eweight), so the energy metrics have nothing to measure",
    "tadah hpo: PC_ERMSE ({index}) gives weight {value} to validation dataset {index} ({file}), which has no structure with an energy: every one has energy priority 0 (eweight). A weight of 0 for this dataset is accepted, and so is a validation configuration without this file.",
    "{key} requires <target> <weight>",
    "{key}: unknown flag '{tok}'",
    "{key}: repeated declaration (first declared on line {first_line})",
    "{key}: cannot mix global and indexed forms",
    "{key}: duplicate assignment for dataset {index} (first declared on line {first_line})",
    "{key}: validation indices start at 1",
    "Invalid index number '{tok}'",
    "Descending index range '{tok}'",
};
inline constexpr Arg ROOT_PC_ErRMSE_args[] = {
    Arg{"target", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} <target>", "", "", false, ""},
    Arg{"weight", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} <weight>", "{key} <weight> must be a number >= 0, got '{tok}'",
        "", false, ""},
};
inline constexpr std::string_view ROOT_PC_ErRMSE_checks[] = {
    "tadah hpo: {keys} on the validation set: no dataset has a structure with an energy ({list}): every structure has energy priority 0 (eweight), so the energy metrics have nothing to measure",
    "{key} requires <target> <weight>",
    "{key}: unknown flag '{tok}'",
    "{key}: indexed targets are supported only for PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA",
};
inline constexpr Arg ROOT_PC_ERSQ_args[] = {
    Arg{"target", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} <target>", "", "", false, ""},
    Arg{"weight", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} <weight>", "{key} <weight> must be a number >= 0, got '{tok}'",
        "", false, ""},
};
inline constexpr std::string_view ROOT_PC_ERSQ_checks[] = {
    "tadah hpo: {keys} on the validation set: no dataset has a structure with an energy ({list}): every structure has energy priority 0 (eweight), so the energy metrics have nothing to measure",
    "tadah hpo: PC_ERSQ on the validation set: the energies per atom of validation dataset {index} ({file}) are all equal, so its R^2 has no value; PC_ERSQ needs two different energies per atom in every validation dataset it measures",
    "{key} requires <target> <weight>",
    "{key}: unknown flag '{tok}'",
    "{key}: indexed targets are supported only for PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA",
    "PC_ERSQ takes no {option}: a cap rejects an evaluation whose metric is above it, and a higher R^2 is a better fit, so the cap would reject the better fits",
};
inline constexpr Arg ROOT_PC_SRMSE_GPA_args[] = {
    Arg{"indices", ArgKind::Indices, true, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
    Arg{"target", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} <target>", "", "", false, ""},
    Arg{"weight", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} <weight>", "{key} <weight> must be a number >= 0, got '{tok}'",
        "", false, ""},
};
inline constexpr std::string_view ROOT_PC_SRMSE_GPA_checks[] = {
    "{key} requires validation DBFILE entries",
    "{key}: dataset {index} on line {line} is outside validation DBFILE range 1-{count}",
    "{key}: missing target and weight for:{list}",
    "tadah hpo: PC_SRMSE_GPA on the validation set: no dataset has a stress tensor ({list}): every structure has stress priority 0 (sweight), so the stress metrics have nothing to measure",
    "tadah hpo: PC_SRMSE_GPA ({index}) gives weight {value} to validation dataset {index} ({file}), which has no structure with a stress tensor: every one has stress priority 0 (sweight). A weight of 0 for this dataset is accepted, and so is a validation configuration without this file.",
    "{key} requires <target> <weight>",
    "{key}: unknown flag '{tok}'",
    "{key}: repeated declaration (first declared on line {first_line})",
    "{key}: cannot mix global and indexed forms",
    "{key}: duplicate assignment for dataset {index} (first declared on line {first_line})",
    "{key}: validation indices start at 1",
};
inline constexpr Arg ROOT_PC_FRMSE_args[] = {
    Arg{"indices", ArgKind::Indices, true, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
    Arg{"target", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} <target>", "", "", false, ""},
    Arg{"weight", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} <weight>", "{key} <weight> must be a number >= 0, got '{tok}'",
        "", false, ""},
};
inline constexpr std::string_view ROOT_PC_FRMSE_checks[] = {
    "{key} requires validation DBFILE entries",
    "{key}: dataset {index} on line {line} is outside validation DBFILE range 1-{count}",
    "{key}: missing target and weight for:{list}",
    "tadah hpo: PC_FRMSE on the validation set: no dataset has a structure with forces ({list}): every structure has force priority 0 (fweight), so the force metrics have nothing to measure",
    "tadah hpo: PC_FRMSE ({index}) gives weight {value} to validation dataset {index} ({file}), which has no structure with forces: every one has force priority 0 (fweight). A weight of 0 for this dataset is accepted, and so is a validation configuration without this file.",
    "{key} requires <target> <weight>",
    "{key}: unknown flag '{tok}'",
    "{key}: repeated declaration (first declared on line {first_line})",
    "{key}: cannot mix global and indexed forms",
    "{key}: duplicate assignment for dataset {index} (first declared on line {first_line})",
    "{key}: validation indices start at 1",
    "Invalid index number '{tok}'",
    "Descending index range '{tok}'",
};
inline constexpr Arg ROOT_PC_LAMMPS_args[] = {
    Arg{"option", ArgKind::Tokens, true, true, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"LAMMPS option", ArgKind::Tokens, true, true, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view ROOT_PC_LAMMPS_checks[] = {
    "{key} {option} maxtime: the variable maxtime is reserved: tadah hpo passes the script's time limit to LAMMPS in it, for `timer timeout ${maxtime}`; --maxtime on this line or PC_SCRIPT_TIMEOUT sets the limit",
    "LAMMPS include file not found: {file}",
    "Circular include detected at: {file}",
    "Missing file name in include statement ({file})",
    "{key}: the script {script} has no command: every line of it, and of the files it includes, is blank or a comment",
    "{key}: LAMMPS cannot start with this line's command-line options: {why}",
    "{key}: the line has no --script option; --script <file> names the LAMMPS script the line runs",
    "PC_LAMMPS: the line has no --varloss option; a PC_LAMMPS line gives at least one --varloss <name> <target> <weight>, a loss term (--outvar records a value without one)",
    "PC_LAMMPS validation failed for script '{script}'.\nTadah requires every --varloss/--outvar/--failvar variable to be defined in the LAMMPS script (or included files)\nas an *equal-style* variable:\n  variable <name> equal ...\n\nProblems found:\n{list}",
    "{key}: unexpected '{tok}': every word after {key} is an option or a value of one (the options of {key} start with --, LAMMPS command-line options with -)",
    "{key}: unknown option '{option}'{note}: a {key} line takes the options {list}, and LAMMPS command-line options, which start with a single '-'",
    "{key}: the script {script} has a quit command, command {index} of the script with its include files read in ({text}): quit ends the process that runs the script (tadah hpo itself under PC_ISOLATION THREAD)",
    "PC_LAMMPS: the script {script} reads {name}, the --hotvar of line {line}, and this line defines {name} ({option} {name}): the run injects `variable {name} equal <value>` into the script, which LAMMPS refuses for a variable of another style",
};
inline constexpr Arg ROOT_PC_FAILPROBE_args[] = {
    Arg{"option", ArgKind::Tokens, true, true, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"LAMMPS option", ArgKind::Tokens, true, true, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view ROOT_PC_FAILPROBE_checks[] = {
    "{key}: the line has no --script option; --script <file> names the LAMMPS script the line runs",
    "PC_FAILPROBE requires --outvar <S> (the sanity scalar)",
    "PC_FAILPROBE must not declare --varloss (it grades failures; it must not shape the feasible objective)",
    "PC_FAILPROBE must not declare --hotvar/--failvar",
    "{key}: unknown option '{option}'{note}: a {key} line takes the options {list}, and LAMMPS command-line options, which start with a single '-'",
    "{key}: LAMMPS cannot start with this line's command-line options: {why}",
};
inline constexpr Arg ROOT_PC_OBSERVABLE_args[] = {
    Arg{"kind", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE: unknown observable kind '{tok}'", "", false, ""},
    Arg{"option", ArgKind::Tokens, false, true, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view ROOT_PC_OBSERVABLE_checks[] = {
    "PC_OBSERVABLE: empty line",
};
inline constexpr Arg ROOT_LOSS_args[] = {
    Arg{"function", ArgKind::Loss, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view ROOT_LOSS_checks[] = {
    "LOSS requires a loss type",
    "{what}: unknown loss function '{tok}'; the loss functions are L1, L2, LOG_COSH, MAPE, HUBER, RMSLE and TUKEY, in any case",
    "LOSS {name}: parameter '{tok}' is not a number{underflow}",
    "loss {name} takes no parameter (got {count})",
    "loss {name} takes at most {max} parameter (got {count})",
    "loss {name} parameter ({key}) must be finite and > 0 (got {value}); omit it to use the default {default}",
    "LOSS sets the loss function of the --varloss and PC_OBSERVABLE terms after it that name none, and of the metric lines (PC_EMAE, PC_ERMSE, PC_ErRMSE, PC_ERSQ, PC_FRMSE, PC_SRMSE_GPA), and the file has no metric line and no such term after it; it would have no effect",
    "{key} {value}: the metric lines take the loss function of the last LOSS line (line {line}): {why}",
};
inline constexpr Arg ROOT_OPTIM_args[] = {
    Arg{"KEY", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"indices", ArgKind::Indices, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
    Arg{"low", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"high", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view top_level_keys[] = {
    "FAILSCORE",
    "PC_SCRIPT_TIMEOUT",
    "PC_ISOLATION",
    "PC_WORKER_TIMEOUT",
    "FAILSCORE_GRADING",
    "BUMPGATE",
    "RHOGATE",
    "RHOBARGATE",
    "QFLUSH",
    "OUTPUT",
    "BOUTPUT",
    "DIGITS",
    "DUMP",
    "BDUMP",
    "OPTIM_SCALE",
    "PC_EMAE",
    "PC_ERMSE",
    "PC_ErRMSE",
    "PC_ERSQ",
    "PC_SRMSE_GPA",
    "PC_FRMSE",
    "PC_LAMMPS",
    "PC_FAILPROBE",
    "PC_OBSERVABLE",
    "LOSS",
    "OPTIM",
};
inline constexpr std::string_view RHOGATE_R_LO_lines[] = {
    "RHOGATE",
};
inline constexpr Arg RHOGATE_R_LO_args[] = {
    Arg{"r", ArgKind::Number, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "RHOGATE R_LO",
        "RHOGATE R_LO must be >= 0, got '{tok}'", "", false, ""},
};
inline constexpr std::string_view RHOGATE_R_LO_checks[] = {
    "RHOGATE: R_LO needs a value; expected RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]",
    "RHOGATE R_LO {value} A must lie below the smallest cutoff a candidate can have, RCUTMB = {max} A{note}: the window [R_LO, RCUTMB] would be empty; a smaller R_LO gives the window a width.",
};
inline constexpr std::string_view RHOGATE_NPTS_lines[] = {
    "RHOGATE",
};
inline constexpr Arg RHOGATE_NPTS_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "2", 2.0, true, 2ull},
        Limit{true, "10000000", 10000000.0, true, 10000000ull}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "RHOGATE NPTS", "", "", false, ""},
};
inline constexpr std::string_view RHOGATE_NPTS_checks[] = {
    "RHOGATE: NPTS needs a value; expected RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]",
};
inline constexpr std::string_view RHOGATE_SEED_lines[] = {
    "RHOGATE",
};
inline constexpr std::string_view RHOGATE_SEED_arg0_words[] = {
    "ERROR",
    "WARN",
};
inline constexpr Arg RHOGATE_SEED_args[] = {
    Arg{"mode", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::RHOGATE_SEED_arg0_words, 2},
        Span<RefusedWord>{nullptr, 0}, "",
        "RHOGATE SEED must be ERROR or WARN, got '{tok}'", "", false, ""},
};
inline constexpr std::string_view RHOGATE_SEED_checks[] = {
    "RHOGATE: SEED needs a value; expected RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]",
};
inline constexpr std::string_view RHOBARGATE_NPTS_lines[] = {
    "RHOBARGATE",
};
inline constexpr Arg RHOBARGATE_NPTS_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "3", 3.0, true, 3ull},
        Limit{true, "10000000", 10000000.0, true, 10000000ull}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "RHOBARGATE NPTS", "", "", false, ""},
};
inline constexpr std::string_view RHOBARGATE_NPTS_checks[] = {
    "RHOBARGATE: NPTS needs a value; expected RHOBARGATE <bcc|fcc|hcp> <a0> <f_lo> <f_hi> [NPTS <n>] [SEED ERROR|WARN]",
};
inline constexpr std::string_view RHOBARGATE_SEED_lines[] = {
    "RHOBARGATE",
};
inline constexpr std::string_view RHOBARGATE_SEED_arg0_words[] = {
    "ERROR",
    "WARN",
};
inline constexpr Arg RHOBARGATE_SEED_args[] = {
    Arg{"mode", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::RHOBARGATE_SEED_arg0_words, 2},
        Span<RefusedWord>{nullptr, 0}, "",
        "RHOBARGATE SEED must be ERROR or WARN, got '{tok}'", "", false, ""},
};
inline constexpr std::string_view RHOBARGATE_SEED_checks[] = {
    "RHOBARGATE: SEED needs a value; expected RHOBARGATE <bcc|fcc|hcp> <a0> <f_lo> <f_hi> [NPTS <n>] [SEED ERROR|WARN]",
};
inline constexpr std::string_view PC_EMAE_skipabove_lines[] = {
    "PC_EMAE",
    "PC_ERMSE",
    "PC_ErRMSE",
    "PC_SRMSE_GPA",
    "PC_FRMSE",
};
inline constexpr Arg PC_EMAE_skipabove_args[] = {
    Arg{"cap", ArgKind::Positive, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} --skip-above", "", "number", false, ""},
};
inline constexpr std::string_view PC_EMAE_skipabove_checks[] = {
    "{key} --skip-above requires a positive number",
};
inline constexpr std::string_view PC_EMAE_skiprel_lines[] = {
    "PC_EMAE",
    "PC_ERMSE",
    "PC_ErRMSE",
    "PC_SRMSE_GPA",
    "PC_FRMSE",
};
inline constexpr Arg PC_EMAE_skiprel_args[] = {
    Arg{"ratio", ArgKind::Positive, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "{key} --skip-rel", "", "number", false,
        ""},
};
inline constexpr std::string_view PC_EMAE_skiprel_checks[] = {
    "{key} --skip-rel requires a positive number",
};
inline constexpr std::string_view PC_LAMMPS_script_lines[] = {
    "PC_LAMMPS",
    "PC_FAILPROBE",
};
inline constexpr Arg PC_LAMMPS_script_args[] = {
    Arg{"file", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PC_LAMMPS_script_checks[] = {
    "{key} --script takes one value, the LAMMPS script file",
};
inline constexpr std::string_view PC_LAMMPS_varloss_lines[] = {
    "PC_LAMMPS",
};
inline constexpr Arg PC_LAMMPS_varloss_args[] = {
    Arg{"name", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"target", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "--varloss {name} <target>", "", "",
        false, ""},
    Arg{"weight", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "--varloss {name} <weight>", "", "",
        false, ""},
    Arg{"loss", ArgKind::Loss, true, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PC_LAMMPS_varloss_checks[] = {
    "{key} --varloss takes <name> <target> <weight> (the name is missing)",
    "{key} --varloss takes <name> <target> <weight> ({name} has no target)",
    "{key} --varloss takes <name> <target> <weight> ({name} has no weight)",
    "--varloss {name} {kind} parameter must be a finite number, got '{tok}'{underflow}",
    "--varloss {name} {tok}: {why}",
};
inline constexpr std::string_view PC_LAMMPS_invar_lines[] = {
    "PC_LAMMPS",
    "PC_FAILPROBE",
};
inline constexpr Arg PC_LAMMPS_invar_args[] = {
    Arg{"name", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"value", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PC_LAMMPS_invar_checks[] = {
    "{key} --invar takes two values, the name of a LAMMPS variable and its value",
    "{key} --invar maxtime: the variable maxtime is reserved: tadah hpo passes the script's time limit to LAMMPS in it, for `timer timeout ${maxtime}`; --maxtime on this line or PC_SCRIPT_TIMEOUT sets the limit",
    "{key} --invar takes <name> <value> (unexpected '{tok}')",
    "{option} {name}: the variable {name} is defined twice on this line (first by {flag} {name}); each variable is defined once",
};
inline constexpr std::string_view PC_LAMMPS_outvar_lines[] = {
    "PC_LAMMPS",
    "PC_FAILPROBE",
};
inline constexpr Arg PC_LAMMPS_outvar_args[] = {
    Arg{"name", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PC_LAMMPS_outvar_checks[] = {
    "{key} --outvar takes one value, the name of a LAMMPS variable",
};
inline constexpr std::string_view PC_LAMMPS_failscore_lines[] = {
    "PC_LAMMPS",
    "PC_FAILPROBE",
};
inline constexpr Arg PC_LAMMPS_failscore_args[] = {
    Arg{"value", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "--failscore", "", "", false, ""},
};
inline constexpr std::string_view PC_LAMMPS_failscore_checks[] = {
    "{key} --failscore takes one value, a number",
    "PC_FAILPROBE --failscore must be a positive finite number, got '{tok}'{underflow}",
};
inline constexpr std::string_view PC_LAMMPS_maxtime_lines[] = {
    "PC_LAMMPS",
    "PC_FAILPROBE",
};
inline constexpr Arg PC_LAMMPS_maxtime_args[] = {
    Arg{"time", ArgKind::Timespec, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "--maxtime", "", "", false, ""},
};
inline constexpr std::string_view PC_LAMMPS_maxtime_checks[] = {
    "{key} --maxtime takes one value, a time limit (off, unlimited or [[HH:]MM:]SS)",
};
inline constexpr std::string_view PC_LAMMPS_hotvar_lines[] = {
    "PC_LAMMPS",
};
inline constexpr Arg PC_LAMMPS_hotvar_args[] = {
    Arg{"name", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"min", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "--hotvar {name} <min>", "", "", false, ""},
    Arg{"max", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "--hotvar {name} <max>", "", "", false, ""},
    Arg{"initial", ArgKind::Number, true, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "--hotvar {name} <initial>", "", "",
        false, ""},
};
inline constexpr std::string_view PC_LAMMPS_hotvar_checks[] = {
    "--hotvar requires <name> <min> <max> [<initial>]",
    "--hotvar: <name> must be a non-empty identifier",
    "--hotvar {name}: <max> must exceed <min>",
    "{key} --hotvar takes <name> <min> <max> [<initial>] (unexpected '{tok}')",
    "--hotvar {name} is declared twice on this line; each variable is declared once",
    "--hotvar {name}: the variable {name} is defined on this line by {option} {name} too; a --hotvar variable is injected as `variable {name} equal <value>`, which LAMMPS refuses for a variable of another style",
};
inline constexpr std::string_view PC_LAMMPS_failvar_lines[] = {
    "PC_LAMMPS",
};
inline constexpr Arg PC_LAMMPS_failvar_args[] = {
    Arg{"name", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
    Arg{"scale", ArgKind::Positive, true, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "--failvar {name} <scale>", "",
        "number", false, ""},
};
inline constexpr std::string_view PC_LAMMPS_failvar_checks[] = {
    "--failvar requires <name> [<scale>]",
    "--failvar: <name> must be a non-empty identifier",
    "{key} --failvar takes <name> [<scale>] (unexpected '{tok}')",
    "--failvar {name} is declared twice on this line; each variable is declared once",
    "--failvar {name} {value}: the scale grades the failure, which FAILSCORE_GRADING ON does; under FAILSCORE_GRADING OFF a failed evaluation scores FAILSCORE, so the scale would have no effect (--failvar {name} alone fails the evaluation when {name} > 0)",
};
inline constexpr std::string_view PC_LAMMPS_ncpu_lines[] = {
    "PC_LAMMPS",
    "PC_FAILPROBE",
};
inline constexpr std::string_view PC_LAMMPS_ncpu_checks[] = {
    "{key} --ncpu: a removed option: it set the MPI ranks of one script in an MPI build of Tadah! 1.5.0; THREADS sets the number of workers that run the scripts, and PC_ISOLATION whether they run in worker processes or inside the tadah process",
};
inline constexpr std::string_view PC_FAILPROBE_gate_lines[] = {
    "PC_FAILPROBE",
};
inline constexpr Arg PC_FAILPROBE_gate_args[] = {
    Arg{"g", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "PC_FAILPROBE --gate", "", "", false, ""},
};
inline constexpr std::string_view PC_FAILPROBE_gate_checks[] = {
    "PC_FAILPROBE --gate requires a value",
};
inline constexpr std::string_view PC_FAILPROBE_s0_lines[] = {
    "PC_FAILPROBE",
};
inline constexpr Arg PC_FAILPROBE_s0_args[] = {
    Arg{"s0", ArgKind::Positive, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "PC_FAILPROBE --s0", "", "number", false, ""},
};
inline constexpr std::string_view PC_FAILPROBE_s0_checks[] = {
    "PC_FAILPROBE --s0 requires a value",
    "PC_FAILPROBE --s0 grades failed evaluations, which FAILSCORE_GRADING ON does; under FAILSCORE_GRADING OFF every failed evaluation scores FAILSCORE, so --s0 would have no effect",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_species_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_species_args[] = {
    Arg{"element", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_species_checks[] = {
    "PC_OBSERVABLE cohesive_energy: --species {name} is not the symbol of an element",
    "PC_OBSERVABLE cohesive_energy --species {name}: the potential has no element {name} (its elements: {list})",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_target_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_target_args[] = {
    Arg{"E", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --target takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_target_checks[] = {
    "PC_OBSERVABLE cohesive_energy --target {tok}: {why}",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_weight_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_weight_args[] = {
    Arg{"w", ArgKind::Number, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{}, false, false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --weight takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_weight_checks[] = {
    "PC_OBSERVABLE cohesive_energy: --weight must be >= 0 (got {tok})",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_hotvar_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_hotvar_arg0_words[] = {
    "a0",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_hotvar_args[] = {
    Arg{"name", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_hotvar_arg0_words, 1},
        Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --hotvar names a0, the lattice constant, the one variable of the term (got '{tok}')",
        "", false, ""},
    Arg{"lo", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --hotvar a0 <lo> takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
    Arg{"hi", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --hotvar a0 <hi> takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
    Arg{"init", ArgKind::Number, true, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --hotvar a0 <init> takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_hotvar_checks[] = {
    "PC_OBSERVABLE cohesive_energy: --hotvar a0 <hi> must be greater than <lo> (got <lo> {low}, <hi> {high})",
    "PC_OBSERVABLE cohesive_energy: --hotvar a0 <init> {tok} is outside its window [{low}, {high}]",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_lattice_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_lattice_arg0_words[] = {
    "bcc",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_lattice_args[] = {
    Arg{"lattice", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_lattice_arg0_words, 1},
        Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --lattice takes bcc, the one lattice of the term (got '{tok}')",
        "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_supercell_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_supercell_args[] = {
    Arg{"nx", ArgKind::Integer, false, false,
        Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0},
        "PC_OBSERVABLE cohesive_energy: --supercell <nx>", "", "", false, ""},
    Arg{"ny", ArgKind::Integer, false, false,
        Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0},
        "PC_OBSERVABLE cohesive_energy: --supercell <ny>", "", "", false, ""},
    Arg{"nz", ArgKind::Integer, false, false,
        Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0},
        "PC_OBSERVABLE cohesive_energy: --supercell <nz>", "", "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_loss_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_loss_args[] = {
    Arg{"function", ArgKind::Loss, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_maxiter_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_maxiter_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0},
        "PC_OBSERVABLE cohesive_energy: --maxiter", "", "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_maxeval_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_maxeval_args[] = {
    Arg{"n", ArgKind::Integer, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0},
        "PC_OBSERVABLE cohesive_energy: --maxeval", "", "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_maxwalltime_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_maxwalltime_args[] = {
    Arg{"seconds", ArgKind::Number, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --max-wall-time takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_etol_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_etol_args[] = {
    Arg{"tol", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --etol takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_fmaxconverged_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_fmaxconverged_args[] = {
    Arg{"fmax", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --fmax-converged takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_fmaxconverged_checks[] = {
    "PC_OBSERVABLE cohesive_energy: --fmax-converged must be > 0 (got {tok})",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_unconverged_lines[] = {
    "PC_OBSERVABLE.cohesive_energy",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_unconverged_arg0_words[] = {
    "const",
    "force-scaled",
};
inline constexpr Arg PC_OBSERVABLE_cohesive_energy_unconverged_args[] = {
    Arg{"kind", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_unconverged_arg0_words, 2},
        Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --unconverged kind must be 'const' or 'force-scaled' (got '{tok}')",
        "", false, ""},
    Arg{"value", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "PC_OBSERVABLE cohesive_energy: --unconverged {kind} takes a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view OPTIM_AUTO_RATIO_lines[] = {
    "OPTIM.AUTO",
};
inline constexpr Arg OPTIM_AUTO_RATIO_args[] = {
    Arg{"f", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view OPTIM_AUTO_RATIO_checks[] = {
    "OPTIM {key}: AUTO RATIO must be > 1 (got {tok})",
    "OPTIM {key} ({index}): AUTO RATIO {value} gives the box [seed/{value}, seed*{value}], which needs a seed other than 0, and the seed is 0{hint}",
    "OPTIM {key} ({index}): AUTO RATIO under {what} needs every seed on the line > 0 (got {value}); FRAC, the default, gives a box clamped at 0",
};
inline constexpr std::string_view OPTIM_AUTO_FRAC_lines[] = {
    "OPTIM.AUTO",
};
inline constexpr Arg OPTIM_AUTO_FRAC_args[] = {
    Arg{"p", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view OPTIM_AUTO_FRAC_checks[] = {
    "OPTIM {key}: AUTO FRAC must be >= 0 (got {tok}); FRAC 0 with FLOOR or GROUP_FLOOR gives a constant half-width",
    "OPTIM {key} ({index}): AUTO FRAC gives a zero-width box (the seed {seed} times FRAC {value} is 0){hint}",
    "OPTIM {key} ({index}): every seed on this line is 0, so AUTO FRAC gives a zero-width box, and GROUP_FLOOR {value}, a fraction of the largest seed on the line, adds nothing{hint}",
    "OPTIM {key} ({index}): every seed on this line is 0, so the AUTO box has zero width (FRAC and the GROUP_FLOOR 1e-3 that {what} implies for {key} are relative to the seeds); AUTO FLOOR <a> with a > 0 gives the box [0, a], and the explicit bounds `OPTIM {key} ({index}) 0 <high>` the box [0, <high>]",
    "OPTIM {key} ({index}): every seed on this line is 0, so AUTO FRAC gives a zero-width box{hint}",
    "OPTIM {key} ({index}): the AUTO box around the seed {seed} has zero width in double precision (half-width {width}); a larger FRAC or FLOOR gives it a width",
};
inline constexpr std::string_view OPTIM_AUTO_FLOOR_lines[] = {
    "OPTIM.AUTO",
};
inline constexpr Arg OPTIM_AUTO_FLOOR_args[] = {
    Arg{"a", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view OPTIM_AUTO_FLOOR_checks[] = {
    "OPTIM {key}: AUTO FLOOR must be >= 0 (got {tok})",
    "OPTIM {key}: FLOOR and GROUP_FLOOR set the smallest half-width of an AUTO FRAC box; AUTO RATIO {value} makes the geometric box [seed/{value}, seed*{value}], which they do not change, so they are given with FRAC <p> only",
    "OPTIM {key}: FLOOR and GROUP_FLOOR set the smallest half-width of an AUTO FRAC box; the AUTO box of {key} is geometric by default (RATIO 2), which they do not change, so they are given with FRAC <p> only",
};
inline constexpr std::string_view OPTIM_AUTO_GROUP_FLOOR_lines[] = {
    "OPTIM.AUTO",
};
inline constexpr Arg OPTIM_AUTO_GROUP_FLOOR_args[] = {
    Arg{"r", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view OPTIM_AUTO_GROUP_FLOOR_checks[] = {
    "OPTIM {key}: AUTO GROUP_FLOOR must be >= 0 (got {tok})",
    "OPTIM {key}: FLOOR and GROUP_FLOOR set the smallest half-width of an AUTO FRAC box; AUTO RATIO {value} makes the geometric box [seed/{value}, seed*{value}], which they do not change, so they are given with FRAC <p> only",
    "OPTIM {key}: FLOOR and GROUP_FLOOR set the smallest half-width of an AUTO FRAC box; the AUTO box of {key} is geometric by default (RATIO 2), which they do not change, so they are given with FRAC <p> only",
};
inline constexpr std::string_view OPTIM_LIN_lines[] = {
    "OPTIM",
};
inline constexpr std::string_view OPTIM_LOG_lines[] = {
    "OPTIM",
};
inline constexpr std::string_view OPTIM_LOG_checks[] = {
    "OPTIM {key}: the line is tagged LOG, and <low> {low} is not > 0, and a box tagged LOG lies above 0{hint}",
};
inline constexpr std::string_view OPTIM_ASINH_lines[] = {
    "OPTIM",
};
inline constexpr Arg OPTIM_ASINH_args[] = {
    Arg{"c", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "OPTIM {key}: ASINH <c> must be a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view OPTIM_ASINH_checks[] = {
    "OPTIM {key}: ASINH needs one argument <c> (the linear scale)",
    "OPTIM {key}: ASINH requires a positive linear scale c (got {tok})",
};
inline constexpr std::string_view PC_OBSERVABLE_cohesive_energy_checks[] = {
    "PC_OBSERVABLE cohesive_energy: --species is required",
    "PC_OBSERVABLE cohesive_energy: --target is required",
    "PC_OBSERVABLE cohesive_energy: --weight is required",
    "PC_OBSERVABLE cohesive_energy: --hotvar a0 <lo> <hi> [<init>] is required",
    "PC_OBSERVABLE cohesive_energy: unknown option '{tok}'",
    "PC_OBSERVABLE cohesive_energy: '{option}' requires a value",
    "PC_OBSERVABLE cohesive_energy: {option} takes a finite number (got '{tok}'){underflow}",
};
inline constexpr Arg OPTIM_range_args[] = {
    Arg{"indices", ArgKind::Indices, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
    Arg{"low", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "OPTIM {key}: <low> must be a finite number (got '{tok}'){underflow}",
        "", false, ""},
    Arg{"high", ArgKind::Number, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "OPTIM {key}: <high> must be a finite number (got '{tok}'){underflow}",
        "", false, ""},
};
inline constexpr std::string_view OPTIM_range_checks[] = {
    "OPTIM {key}: unexpected token(s) '{text}' -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>] or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]",
    "OPTIM {key}: <high> {high} is not greater than <low> {low}",
    "OPTIM {key}: under ASINH {value} the box [{low}, {high}] maps to [{min}, {max}], which has no finite width; a c of the size of the values of the box gives one",
    "OPTIM {key}: the line is tagged LOG, and <low> {low} is not > 0, and a box tagged LOG lies above 0{hint}",
    "the starting point {index} of the INIT block of {phase}: {key} = {value} lies outside its OPTIM box [{low}, {high}]",
};
inline constexpr Arg OPTIM_AUTO_args[] = {
    Arg{"indices", ArgKind::Indices, false, false, Limit{}, Limit{}, false,
        false, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view OPTIM_AUTO_checks[] = {
    "OPTIM {key}: AUTO option '{tok}' needs a value -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>] or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]",
    "OPTIM {key}: AUTO {option} must be a finite number (got '{tok}'){underflow}",
    "OPTIM {key}: '{tok}' is not an AUTO option (RATIO, FRAC, FLOOR or GROUP_FLOOR) -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>] or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]",
    "OPTIM {key}: a cutoff key takes explicit bounds, OPTIM {key} (indices) <low> <high>, and no AUTO box: the neighbour list is sized from the explicit high bound",
    "OPTIM {key} ({index}): the seed {value} is negative, but {what} requires every AMPGRIDMB amplitude >= 0",
    "OPTIM {key} ({index}): the seed {value} is negative, but {why}",
    "OPTIM {key} ({index}): the line is tagged LOG, and the AUTO box [{low}, {high}] reaches 0 or below, and a box tagged LOG lies above 0{hint}",
    "OPTIM {key}: AUTO takes RATIO or FRAC, not both",
    "OPTIM {key} ({index}): AUTO without RATIO or FRAC gives {name} the box [seed/2, seed*2] (RATIO 2), which needs a seed other than 0, and the seed is 0{hint}",
    "OPTIM {key} ({index}): the AUTO box [{low}, {high}] around the seed {seed} is not finite; a smaller RATIO or FRAC gives a finite box",
};
inline constexpr std::string_view ENV_TADAH_OPTIM_SCALE_values[] = {
    "AUTO",
    "NONE",
};
inline constexpr std::string_view ENV_TADAH_FAILSCORE_GRADING_values[] = {
    "ON",
    "OFF",
};
inline constexpr std::string_view ENV_TADAH_FAILSCORE_GRADING_checks[] = {
    "{phase}: ASA:ASA: under FAILSCORE_GRADING ON, set by TADAH_FAILSCORE_GRADING, a failed evaluation scores from 0.999 FAILSCORE to below FAILSCORE ({value}), and ASA:ASA takes a cost of about 1e+18 or more for an invalid state and draws another, so it would not see the graded failures at or above that cost; a FAILSCORE of 1e+18 or less keeps every graded failure below it",
};
inline constexpr std::string_view ENV_TADAH_BUMPGATE_values[] = {
    "CANDIDATE",
    "STRICT",
};
inline constexpr std::string_view ENV_TADAH_PC_ISOLATION_values[] = {
    "THREAD",
    "PROCESS",
};
inline constexpr std::string_view ENV_TADAH_PC_WORKER_TIMEOUT_checks[] = {
    "TADAH_PC_WORKER_TIMEOUT is too large for the worker deadline, got '{tok}' (it is kept in 64-bit milliseconds: at most about 9.2e15 s)",
    "TADAH_PC_WORKER_TIMEOUT (got '{tok}') sets the deadline of a script task in a worker process (PC_ISOLATION PROCESS); under PC_ISOLATION THREAD, {why}, the scripts run inside the tadah process without a worker deadline, so it would have no effect",
};
inline constexpr LossParam LOSS_HUBER_params[] = {
    LossParam{"delta", 50.0, true, 0.0},
};
inline constexpr LossParam LOSS_TUKEY_params[] = {
    LossParam{"c", 4.685, true, 0.0},
};
inline constexpr KeyId STRATEGY_TYPE_MULTISTART_keys[] = {
    KeyId::STRATEGY_N_STARTS,
    KeyId::STRATEGY_BUDGET_PER_START,
};
inline constexpr std::string_view STRATEGY_TYPE_MULTISTART_checks[] = {
    "STRATEGY TYPE MULTISTART cannot run {name}: it takes every INIT point in one run, so MULTISTART would run it once, whatever N_STARTS; STRATEGY TYPE PLAIN runs {name} once from every INIT point.",
    "STRATEGY TYPE MULTISTART cannot run {name}: it is a global optimiser that takes one starting point, so restarting it from each INIT point is rejected for any N_STARTS. A local optimiser ({list}) runs from each start, and STRATEGY TYPE PLAIN runs {name} once.",
};
inline constexpr KeyId STRATEGY_TYPE_PRESTAGE_keys[] = {
    KeyId::STRATEGY_TOPK,
};
inline constexpr std::string_view STRATEGY_TYPE_PRESTAGE_checks[] = {
    "STRATEGY TYPE PRESTAGE requires an EXPLORE ... ENDEXPLORE block inside the STRATEGY block (it defines the exploration phase); the refinement optimiser is the STRATEGY-level LIB/ALGO or an optional REFINE ... ENDREFINE block.",
    "STRATEGY TYPE PRESTAGE cannot refine with {name}: it does not start at the point it is given ({note}), so the refinement would never use the exploration's candidates, for any TOPK. A local optimiser starts at each candidate ({list}); {name} runs as the exploration (EXPLORE block) or once, as STRATEGY TYPE PLAIN.",
    "STRATEGY TYPE PRESTAGE with TOPK {value} cannot refine with {name}: the refinement runs its optimiser once from each of the TOPK candidates the exploration forwards, and {name} is a global optimiser that takes one starting point. A local or multi-seed refinement optimiser ({list}) refines each candidate, and TOPK 1 refines the best candidate once.",
};
inline constexpr std::string_view METHOD_CONFIG_checks[] = {
    "{key} CONFIG takes no value",
};
inline constexpr Arg METHOD_RANDOM_args[] = {
    Arg{"K", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "1000000", 1000000.0, true, 1000000ull}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} RANDOM: K", "", "", false, ""},
};
inline constexpr std::string_view METHOD_RANDOM_checks[] = {
    "{key} RANDOM takes one value, K, the number of starting points it draws",
};
inline constexpr Arg METHOD_LHS_args[] = {
    Arg{"K", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "1000000", 1000000.0, true, 1000000ull}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} LHS: K", "", "", false, ""},
};
inline constexpr std::string_view METHOD_LHS_checks[] = {
    "{key} LHS takes one value, K, the number of starting points it draws",
};
inline constexpr Arg METHOD_SOBOL_args[] = {
    Arg{"K", ArgKind::Integer, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "1000000", 1000000.0, true, 1000000ull}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0},
        "{key} SOBOL: K", "", "", false, ""},
};
inline constexpr std::string_view METHOD_SOBOL_checks[] = {
    "{key} SOBOL takes one value, K, the number of starting points it draws",
};
inline constexpr Arg METHOD_WARM_args[] = {
    Arg{"file", ArgKind::Name, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view METHOD_WARM_checks[] = {
    "{key} WARM takes one value, the potential file the starting point is read from",
    "WARM: cannot open seed file '{file}'. Provide the file (production submit scripts copy it into the run dir) or drop the WARM line to start from the in-config values.",
    "WARM: parse failure in seed file '{file}': {why}",
    "WARM: {count} parameter(s) not found in {file}: {list}",
    "WARM: {count} parameter(s) from {file} clamped into OPTIM bounds -- the seed point is NOT the one on disk: {list}",
    "WARM seed {file} {text}, but this run evaluates RHOFORM {type} (from -c) and takes the OPTIM density parameters ({keys}) from the seed: density parameters are meaningful only in the form they were fitted in. {hint}",
};
inline constexpr std::string_view optim_positive_only[] = {
    "SGRID2B",
    "SGRIDMB",
    "SEMBFUNC",
};
inline constexpr std::string_view optim_checks[] = {
    "Failed to read key from input.",
    "{what}: the key is followed by its indices in parentheses, such as {what} (1)",
    "{what}: the indices have no closing parenthesis",
    "{what}: the indices hold another opening parenthesis",
    "{what}: the parentheses hold no index",
    "{what}: a space follows the closing parenthesis of the indices",
    "{what}: the line takes one list of indices in parentheses (unexpected '{tok}')",
    "OPTIM {key}: expected `(indices) <low> <high>` or `(indices) AUTO ...` (got {count} token(s))",
    "OPTIM {key}: bad index list ({indices}): {why}",
    "OPTIM {key}: {key} is not a key of the training configuration (closest key: {name})",
    "OPTIM {key}: the training configuration has no {key} line, and OPTIM {key} searches the values the training configuration gives",
    "DM_Embed: unknown, inactive or discrete parameter '{key}'",
    "OPTIM {key} is not a valid search target: {key} is an internal key, a value tadah computes itself (no configuration sets it, so no evaluation can take a candidate value of it).",
    "OPTIM {key} ({index}) is not a valid search target: value {index} of {key} is an integer ({key} takes {list}), a discrete value rather than a continuous parameter; the training configuration sets it.",
    "OPTIM {key} ({index}): {key} has {count} values in the training configuration, so the indices of OPTIM {key} run from 1 to {max}",
    "OPTIM {key} ({index}): {key} has 1 value in the training configuration, so the index of OPTIM {key} is 1{note}",
    "OPTIM {key} ({index}): {key} has {count} that the descriptors read, so {what}{note}",
    "OPTIM {key} ({index}): {key} has no numeric value in the training configuration, so no index of OPTIM {key} selects one",
    "OPTIM {key} ({index}): the OPTIM {key} line on line {line} selects this index as well; one OPTIM line selects each index (OPTIM keys match in any case)",
    "OPTIM {key} ({index}): the low bound {low} is not > 0, but {what} requires {value}, so the low bound of its box is > 0",
    "OPTIM {key} ({index}): the low bound {low} is negative, but {what} requires {value}, so the low bound of its box is >= 0 (its AUTO FRAC box, the default, is clamped at 0)",
    "OPTIM {key} ({index}): the AUTO box [{low}, {high}] has the low bound {low}, which is not > 0, but {what} requires {value}, so the low bound of its box is > 0",
    "OPTIM {key} ({index}): the AUTO box [{low}, {high}] has the low bound {low}, which is negative, but {what} requires {value}, so the low bound of its box is >= 0",
    "OPTIM FIXWEIGHT requires FIXINDEX in the training configuration (FIXINDEX names the coefficients the FIXWEIGHT values hold).",
    "OPTIM FIXWEIGHT requires FIXWEIGHT in the training configuration (the FIXWEIGHT values hold the coefficients FIXINDEX names, and OPTIM FIXWEIGHT searches those values).",
    "OPTIM FIXWEIGHT requires FIXINDEX and FIXWEIGHT in the training configuration (FIXINDEX names the coefficients the FIXWEIGHT values hold, and OPTIM FIXWEIGHT searches those values).",
};
inline constexpr int OPTIM_TARGET_OALGO_indices[] = {
    1,
};

}  // namespace detail

namespace lexing {
inline constexpr std::string_view comment = "#";
inline constexpr std::string_view continuation = "\\";
inline constexpr std::string_view key_case = "exact";
inline constexpr std::string_view error_frame = "Error on line {line}: {what}\nLine: {text}";
inline constexpr std::string_view unfinished_msg = "the line ends with \\, which continues it on the next line, and the file has no next line";
inline constexpr std::string_view missing_file_msg = "the target file {file} (--hpotarget) cannot be opened";
inline constexpr Span<std::string_view> bool_true = Span<std::string_view>{detail::lexing_bool_true, 4};
inline constexpr Span<std::string_view> bool_false = Span<std::string_view>{detail::lexing_bool_false, 4};
inline constexpr Span<std::string_view> timespec_words = Span<std::string_view>{detail::lexing_timespec_words, 2};
}  // namespace lexing

namespace reading {
inline constexpr std::string_view integer_msg = "{what} must be an integer >= {min}, got '{tok}'";
inline constexpr std::string_view integer_range_msg = "{what} must be an integer from {min} to {max}, got '{tok}'";
inline constexpr std::string_view note_msg = " ({note})";
inline constexpr std::string_view finite_msg = "{what} must be a finite number, got '{tok}'{underflow}";
inline constexpr std::string_view positive_msg = "{what} must be a positive {noun}, got '{tok}'{underflow}";
inline constexpr std::string_view underflow_note = " (too small: its magnitude is below the smallest normal double, 2.2250738585072014e-308)";
inline constexpr std::string_view requires_value_msg = "{key} requires a value";
inline constexpr std::string_view single_value_msg = "{key} takes one value (unexpected '{tok}')";
inline constexpr std::string_view bool_msg = "{what}: expected boolean (true/false/1/0/yes/no/on/off), got '{tok}'";
inline constexpr std::string_view timespec_msg = "{what} must be 'off', 'unlimited' or a time [[HH:]MM:]SS, got '{tok}'{underflow}";
inline constexpr std::string_view script_timeout_msg = "{what} must be a positive time, 'off' or 'unlimited' (a limit <= 0 stops every script), got '{tok}'";
inline constexpr std::string_view script_timeout_large_msg = "{what} is too large for the worker deadline, got '{tok}' (the PC_ISOLATION PROCESS deadline max(3t, t + 120 s) is kept in 64-bit milliseconds: t must stay below about 3.07e15 s)";
}  // namespace reading

namespace structure {
inline constexpr Span<CheckStage> check_order = Span<CheckStage>{detail::structure_check_order, 10};
inline constexpr std::string_view end_without_open_msg = "{end} without matching {name}";
inline constexpr std::string_view end_mismatch_msg = "{end} does not close the innermost open block, the {open} block opened on line {open_line} (END{open} closes it)";
inline constexpr std::string_view end_takes_no_value_msg = "{end} closes the {name} block and takes no value (unexpected '{tok}')";
inline constexpr std::string_view open_takes_no_value_msg = "{name} opens a block and takes no value (unexpected '{tok}'); each key stands on its own line inside the block";
inline constexpr std::string_view unclosed_msg = "{name} block not closed with {end}";
inline constexpr std::string_view unknown_root_msg = "unknown key '{key}'";
inline constexpr std::string_view root_key_in_block_msg = "{key} is a top-level key, and the {name} block opened on line {open_line} is still open ({end} closes it)";
inline constexpr std::string_view removed_msg = "'{key}' was removed from the grammar: {hint}";
}  // namespace structure

/* The placeholders a message may hold: those of the messages
 * whose set the loader fixes (FIXED_PLACEHOLDERS), of the
 * readers' messages, and of the messages of the entries. */
namespace placeholders {
inline constexpr Span<std::string_view> block_misplaced_msg = Span<std::string_view>{detail::placeholders_block_misplaced_msg, 2};
inline constexpr Span<std::string_view> block_refuse_msg = Span<std::string_view>{detail::placeholders_block_refuse_msg, 3};
inline constexpr Span<std::string_view> block_repeat_msg = Span<std::string_view>{detail::placeholders_block_repeat_msg, 1};
inline constexpr Span<std::string_view> block_unknown_msg = Span<std::string_view>{detail::placeholders_block_unknown_msg, 3};
inline constexpr Span<std::string_view> group_leftover_msg = Span<std::string_view>{detail::placeholders_group_leftover_msg, 2};
inline constexpr Span<std::string_view> lexing_error_frame = Span<std::string_view>{detail::placeholders_lexing_error_frame, 3};
inline constexpr Span<std::string_view> lexing_missing_file_msg = Span<std::string_view>{detail::placeholders_lexing_missing_file_msg, 1};
inline constexpr Span<std::string_view> lexing_unfinished_msg = Span<std::string_view>{nullptr, 0};
inline constexpr Span<std::string_view> structure_end_mismatch_msg = Span<std::string_view>{detail::placeholders_structure_end_mismatch_msg, 3};
inline constexpr Span<std::string_view> structure_end_takes_no_value_msg = Span<std::string_view>{detail::placeholders_structure_end_takes_no_value_msg, 3};
inline constexpr Span<std::string_view> structure_end_without_open_msg = Span<std::string_view>{detail::placeholders_structure_end_without_open_msg, 2};
inline constexpr Span<std::string_view> structure_open_takes_no_value_msg = Span<std::string_view>{detail::placeholders_structure_open_takes_no_value_msg, 2};
inline constexpr Span<std::string_view> structure_removed_msg = Span<std::string_view>{detail::placeholders_structure_removed_msg, 2};
inline constexpr Span<std::string_view> structure_root_key_in_block_msg = Span<std::string_view>{detail::placeholders_structure_root_key_in_block_msg, 4};
inline constexpr Span<std::string_view> structure_unclosed_msg = Span<std::string_view>{detail::placeholders_structure_unclosed_msg, 2};
inline constexpr Span<std::string_view> structure_unknown_root_msg = Span<std::string_view>{detail::placeholders_structure_unknown_root_msg, 1};
inline constexpr Span<std::string_view> reading = Span<std::string_view>{detail::placeholders_reading, 8};
inline constexpr Span<std::string_view> entry = Span<std::string_view>{detail::placeholders_entry, 47};
}  // namespace placeholders

inline constexpr Group kGroups[] = {
  Group{"OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      "{key} takes no further value here (unexpected '{tok}')"},
};

inline constexpr Block kBlocks[] = {
  Block{BlockId::OPTIMIZER, "OPTIMIZER", "ENDOPTIMIZER",
      block_bit(BlockId::ROOT), BlockRepeat::Error,
      "OPTIMIZER block given more than once (the first opens on line {first_line}); one OPTIMIZER block holds every optimiser setting",
      "OPTIMIZER block must appear at the top level of the file (it is inside the {parent} block opened on line {parent_line})",
      "unknown key '{key}' inside the OPTIMIZER block opened on line {open_line} (ENDOPTIMIZER closes it)",
      false, Span<Alias>{nullptr, 0}, Span<Refusal>{nullptr, 0}, "",
      Span<std::string_view>{nullptr, 0}, Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{detail::BLOCK_OPTIMIZER_close_checks, 5},
      Span<KeyId>{detail::BLOCK_OPTIMIZER_keys, 17}},
  Block{BlockId::STRATEGY, "STRATEGY", "ENDSTRATEGY",
      block_bit(BlockId::OPTIMIZER), BlockRepeat::Error,
      "STRATEGY block given more than once (the first opens on line {first_line}); one STRATEGY block holds every STRATEGY key",
      "STRATEGY block must appear directly inside OPTIMIZER",
      "unknown key '{key}' inside the STRATEGY block opened on line {open_line} (ENDSTRATEGY closes it)",
      false, Span<Alias>{nullptr, 0}, Span<Refusal>{nullptr, 0}, "",
      Span<std::string_view>{nullptr, 0}, Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{detail::BLOCK_STRATEGY_close_checks, 1},
      Span<KeyId>{detail::BLOCK_STRATEGY_keys, 21}},
  Block{BlockId::INIT, "INIT", "ENDINIT",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE),
      BlockRepeat::Merge, "",
      "INIT block must appear inside OPTIMIZER, STRATEGY, EXPLORE, or REFINE",
      "unknown key '{key}' inside the INIT block opened on line {open_line} (ENDINIT closes it)",
      true, Span<Alias>{detail::BLOCK_INIT_aliases, 1},
      Span<Refusal>{nullptr, 0}, "",
      Span<std::string_view>{detail::BLOCK_INIT_unreachable, 1},
      Span<std::string_view>{nullptr, 0}, Span<std::string_view>{nullptr, 0},
      Span<KeyId>{detail::BLOCK_INIT_keys, 4}},
  Block{BlockId::NOISE, "NOISE", "ENDNOISE", block_bit(BlockId::OPTIMIZER),
      BlockRepeat::Merge, "",
      "NOISE block must appear directly inside OPTIMIZER",
      "unknown key '{key}' inside the NOISE block opened on line {open_line} (only AUTO and CALIBRATION are valid; ENDNOISE closes it)",
      true, Span<Alias>{nullptr, 0}, Span<Refusal>{nullptr, 0}, "",
      Span<std::string_view>{nullptr, 0}, Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{nullptr, 0},
      Span<KeyId>{detail::BLOCK_NOISE_keys, 2}},
  Block{BlockId::EXPLORE, "EXPLORE", "ENDEXPLORE",
      block_bit(BlockId::STRATEGY), BlockRepeat::Merge, "",
      "EXPLORE block must appear inside STRATEGY",
      "unknown key '{key}' inside the EXPLORE block opened on line {open_line} (ENDEXPLORE closes it)",
      false, Span<Alias>{nullptr, 0}, Span<Refusal>{nullptr, 0}, "",
      Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{detail::BLOCK_EXPLORE_requires, 1},
      Span<std::string_view>{detail::BLOCK_EXPLORE_close_checks, 3},
      Span<KeyId>{detail::BLOCK_EXPLORE_keys, 16}},
  Block{BlockId::REFINE, "REFINE", "ENDREFINE", block_bit(BlockId::STRATEGY),
      BlockRepeat::Merge, "", "REFINE block must appear inside STRATEGY",
      "unknown key '{key}' inside the REFINE block opened on line {open_line} (ENDREFINE closes it)",
      false, Span<Alias>{nullptr, 0}, Span<Refusal>{nullptr, 0}, "",
      Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{detail::BLOCK_REFINE_requires, 1},
      Span<std::string_view>{nullptr, 0},
      Span<KeyId>{detail::BLOCK_REFINE_keys, 16}},
  Block{BlockId::INNER, "INNER", "ENDINNER",
      block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE),
      BlockRepeat::ErrorPerParent,
      "INNER block given twice for one optimiser (the first on line {first_line}); STRATEGY and REFINE configure the same optimiser, and so do two EXPLORE blocks",
      "INNER block must appear inside STRATEGY, EXPLORE or REFINE: it holds the local search that an NLopt MLSL or AUGLAG algorithm runs",
      "unknown key '{key}' inside the INNER block opened on line {open_line} (only {keys} are valid; ENDINNER closes it)",
      false, Span<Alias>{nullptr, 0},
      Span<Refusal>{detail::BLOCK_INNER_refuse, 3},
      "{key} is not accepted inside an INNER block: {why}; the enclosing {enclosing} block takes {key}",
      Span<std::string_view>{detail::BLOCK_INNER_unreachable, 1},
      Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{detail::BLOCK_INNER_close_checks, 3},
      Span<KeyId>{detail::BLOCK_INNER_keys, 13}},
  Block{BlockId::RESTART, "RESTART", "ENDRESTART",
      block_bit(BlockId::STRATEGY), BlockRepeat::Merge, "",
      "RESTART block must appear inside STRATEGY",
      "unknown key '{key}' inside the RESTART block opened on line {open_line} (ENDRESTART closes it)",
      true, Span<Alias>{detail::BLOCK_RESTART_aliases, 1},
      Span<Refusal>{nullptr, 0}, "", Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{nullptr, 0},
      Span<std::string_view>{detail::BLOCK_RESTART_close_checks, 1},
      Span<KeyId>{detail::BLOCK_RESTART_keys, 6}},
};

inline constexpr Key kKeys[] = {
  Key{KeyId::ROOT_FAILSCORE, "ROOT.FAILSCORE", "FAILSCORE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_FAILSCORE_args, 1},
      Repeat::Error,
      "FAILSCORE is set twice (first on line {first_line}) and no line between reads the first (a PC_LAMMPS or PC_FAILPROBE line without --failscore); each key is given once",
      "", false, "", false, KeyId::ROOT_FAILSCORE, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::ROOT_PC_SCRIPT_TIMEOUT, "ROOT.PC_SCRIPT_TIMEOUT",
      "PC_SCRIPT_TIMEOUT", "", block_bit(BlockId::ROOT),
      Span<Arg>{detail::ROOT_PC_SCRIPT_TIMEOUT_args, 1}, Repeat::Error,
      "PC_SCRIPT_TIMEOUT is set twice (first on line {first_line}) and no line between reads the first (a PC_LAMMPS or PC_FAILPROBE line without --maxtime); each key is given once",
      "", true, "off", false, KeyId::ROOT_PC_SCRIPT_TIMEOUT, false, "",
      Span<std::string_view>{detail::ROOT_PC_SCRIPT_TIMEOUT_checks, 1}},
  Key{KeyId::ROOT_MAXTIME, "ROOT.MAXTIME", "MAXTIME", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_MAXTIME_args, 1},
      Repeat::Error,
      "MAXTIME is set twice (first on line {first_line}) and no line between reads the first (a PC_LAMMPS or PC_FAILPROBE line without --maxtime); each key is given once",
      "", true, "off", true, KeyId::ROOT_PC_SCRIPT_TIMEOUT, false, "",
      Span<std::string_view>{detail::ROOT_MAXTIME_checks, 1}},
  Key{KeyId::ROOT_PC_ISOLATION, "ROOT.PC_ISOLATION", "PC_ISOLATION", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_ISOLATION_args, 1},
      Repeat::Error,
      "PC_ISOLATION is set twice (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::ROOT_PC_ISOLATION, false,
      "TADAH_PC_ISOLATION",
      Span<std::string_view>{detail::ROOT_PC_ISOLATION_checks, 1}},
  Key{KeyId::ROOT_PC_WORKER_TIMEOUT, "ROOT.PC_WORKER_TIMEOUT",
      "PC_WORKER_TIMEOUT", "", block_bit(BlockId::ROOT),
      Span<Arg>{detail::ROOT_PC_WORKER_TIMEOUT_args, 1}, Repeat::Error,
      "PC_WORKER_TIMEOUT is set twice (first on line {first_line}); each key is given once",
      "", true, "0", false, KeyId::ROOT_PC_WORKER_TIMEOUT, false,
      "TADAH_PC_WORKER_TIMEOUT",
      Span<std::string_view>{detail::ROOT_PC_WORKER_TIMEOUT_checks, 2}},
  Key{KeyId::ROOT_FAILSCORE_GRADING, "ROOT.FAILSCORE_GRADING",
      "FAILSCORE_GRADING", "", block_bit(BlockId::ROOT),
      Span<Arg>{detail::ROOT_FAILSCORE_GRADING_args, 1}, Repeat::Error,
      "FAILSCORE_GRADING is set twice (first on line {first_line}); each key is given once",
      "", true, "OFF", false, KeyId::ROOT_FAILSCORE_GRADING, false,
      "TADAH_FAILSCORE_GRADING",
      Span<std::string_view>{detail::ROOT_FAILSCORE_GRADING_checks, 1}},
  Key{KeyId::ROOT_BUMPGATE, "ROOT.BUMPGATE", "BUMPGATE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_BUMPGATE_args, 1},
      Repeat::Error,
      "BUMPGATE is set twice (first on line {first_line}); each key is given once",
      "", true, "CANDIDATE", false, KeyId::ROOT_BUMPGATE, false,
      "TADAH_BUMPGATE",
      Span<std::string_view>{detail::ROOT_BUMPGATE_checks, 1}},
  Key{KeyId::ROOT_RHOGATE, "ROOT.RHOGATE", "RHOGATE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_RHOGATE_args, 1},
      Repeat::Error,
      "RHOGATE is set twice (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::ROOT_RHOGATE, false, "",
      Span<std::string_view>{detail::ROOT_RHOGATE_checks, 7}},
  Key{KeyId::ROOT_RHOBARGATE, "ROOT.RHOBARGATE", "RHOBARGATE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_RHOBARGATE_args, 4},
      Repeat::Error,
      "RHOBARGATE is set twice (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::ROOT_RHOBARGATE, false, "",
      Span<std::string_view>{detail::ROOT_RHOBARGATE_checks, 8}},
  Key{KeyId::ROOT_QFLUSH, "ROOT.QFLUSH", "QFLUSH", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_QFLUSH_args, 1},
      Repeat::Error,
      "QFLUSH is set twice (first on line {first_line}); each key is given once",
      "", true, "10", false, KeyId::ROOT_QFLUSH, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::ROOT_OUTPUT, "ROOT.OUTPUT", "OUTPUT", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_OUTPUT_args, 1},
      Repeat::Error,
      "OUTPUT is set twice (first on line {first_line}); each key is given once",
      "", true, "1", false, KeyId::ROOT_OUTPUT, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::ROOT_BOUTPUT, "ROOT.BOUTPUT", "BOUTPUT", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_BOUTPUT_args, 1},
      Repeat::Error,
      "BOUTPUT is set twice (first on line {first_line}); each key is given once",
      "", true, "1", false, KeyId::ROOT_BOUTPUT, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::ROOT_DIGITS, "ROOT.DIGITS", "DIGITS", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_DIGITS_args, 1},
      Repeat::Error,
      "DIGITS is set twice (first on line {first_line}); each key is given once",
      "", true, "6", false, KeyId::ROOT_DIGITS, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::ROOT_DUMP, "ROOT.DUMP", "DUMP", "", block_bit(BlockId::ROOT),
      Span<Arg>{detail::ROOT_DUMP_args, 2}, Repeat::Error,
      "DUMP is set twice (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::ROOT_DUMP, false, "",
      Span<std::string_view>{detail::ROOT_DUMP_checks, 10}},
  Key{KeyId::ROOT_BDUMP, "ROOT.BDUMP", "BDUMP", "", block_bit(BlockId::ROOT),
      Span<Arg>{detail::ROOT_BDUMP_args, 2}, Repeat::Error,
      "BDUMP is set twice (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::ROOT_BDUMP, false, "",
      Span<std::string_view>{detail::ROOT_BDUMP_checks, 10}},
  Key{KeyId::ROOT_OPTIM_SCALE, "ROOT.OPTIM_SCALE", "OPTIM_SCALE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_OPTIM_SCALE_args, 1},
      Repeat::Error,
      "OPTIM_SCALE is set twice (first on line {first_line}); each key is given once",
      "", true, "AUTO", false, KeyId::ROOT_OPTIM_SCALE, false,
      "TADAH_OPTIM_SCALE", Span<std::string_view>{nullptr, 0}},
  Key{KeyId::OPTIMIZER_LIB, "OPTIMIZER.LIB", "LIB", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_LIB_args, 1}, Repeat::Error,
      "LIB is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_LIB, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::OPTIMIZER_ALGO, "OPTIMIZER.ALGO", "ALGO", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_ALGO_args, 1}, Repeat::Error,
      "ALGO is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_ALGO, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::OPTIMIZER_MAXEVAL, "OPTIMIZER.MAXEVAL", "MAXEVAL", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_MAXEVAL_args, 1}, Repeat::Error,
      "MAXEVAL is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_MAXEVAL, false, "",
      Span<std::string_view>{detail::OPTIMIZER_MAXEVAL_checks, 3}},
  Key{KeyId::OPTIMIZER_MAXTIME, "OPTIMIZER.MAXTIME", "MAXTIME", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_MAXTIME_args, 1}, Repeat::Error,
      "MAXTIME is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_MAXTIME, false, "",
      Span<std::string_view>{detail::OPTIMIZER_MAXTIME_checks, 1}},
  Key{KeyId::OPTIMIZER_STOPVAL, "OPTIMIZER.STOPVAL", "STOPVAL", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_STOPVAL_args, 1}, Repeat::Error,
      "STOPVAL is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_STOPVAL, false, "",
      Span<std::string_view>{detail::OPTIMIZER_STOPVAL_checks, 1}},
  Key{KeyId::OPTIMIZER_FTOL_REL, "OPTIMIZER.FTOL_REL", "FTOL_REL", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_FTOL_REL_args, 1}, Repeat::Error,
      "FTOL_REL is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_FTOL_REL, false, "",
      Span<std::string_view>{detail::OPTIMIZER_FTOL_REL_checks, 6}},
  Key{KeyId::OPTIMIZER_FTOL_ABS, "OPTIMIZER.FTOL_ABS", "FTOL_ABS", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_FTOL_ABS_args, 1}, Repeat::Error,
      "FTOL_ABS is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_FTOL_ABS, false, "",
      Span<std::string_view>{detail::OPTIMIZER_FTOL_ABS_checks, 6}},
  Key{KeyId::OPTIMIZER_XTOL_REL, "OPTIMIZER.XTOL_REL", "XTOL_REL", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_XTOL_REL_args, 1}, Repeat::Error,
      "XTOL_REL is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_XTOL_REL, false, "",
      Span<std::string_view>{detail::OPTIMIZER_XTOL_REL_checks, 6}},
  Key{KeyId::OPTIMIZER_XTOL_ABS, "OPTIMIZER.XTOL_ABS", "XTOL_ABS", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_XTOL_ABS_args, 1}, Repeat::Error,
      "XTOL_ABS is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_XTOL_ABS, false, "",
      Span<std::string_view>{detail::OPTIMIZER_XTOL_ABS_checks, 6}},
  Key{KeyId::OPTIMIZER_POPULATION, "OPTIMIZER.POPULATION", "POPULATION",
      "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_POPULATION_args, 1}, Repeat::Error,
      "POPULATION is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_POPULATION, false, "",
      Span<std::string_view>{detail::OPTIMIZER_POPULATION_checks, 6}},
  Key{KeyId::OPTIMIZER_STEP, "OPTIMIZER.STEP", "STEP", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_STEP_args, 3}, Repeat::Error,
      "STEP is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_STEP, false, "",
      Span<std::string_view>{detail::OPTIMIZER_STEP_checks, 12}},
  Key{KeyId::OPTIMIZER_SEED, "OPTIMIZER.SEED", "SEED", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_SEED_args, 1}, Repeat::Error,
      "SEED is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_SEED, false, "",
      Span<std::string_view>{detail::OPTIMIZER_SEED_checks, 4}},
  Key{KeyId::OPTIMIZER_VECTOR_ARR, "OPTIMIZER.VECTOR_ARR", "VECTOR_ARR",
      "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_VECTOR_ARR_args, 1}, Repeat::Error,
      "VECTOR_ARR is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::OPTIMIZER_VECTOR_ARR, false, "",
      Span<std::string_view>{detail::OPTIMIZER_VECTOR_ARR_checks, 5}},
  Key{KeyId::OPTIMIZER_THREADS, "OPTIMIZER.THREADS", "THREADS", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_THREADS_args, 1}, Repeat::Error,
      "THREADS is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", true, "1", false, KeyId::OPTIMIZER_THREADS, false, "",
      Span<std::string_view>{detail::OPTIMIZER_THREADS_checks, 2}},
  Key{KeyId::OPTIMIZER_GRAD_STRATEGY, "OPTIMIZER.GRAD_STRATEGY",
      "GRAD_STRATEGY", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_GRAD_STRATEGY_args, 1}, Repeat::Error,
      "GRAD_STRATEGY is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", true, "NUMERICAL", false, KeyId::OPTIMIZER_GRAD_STRATEGY, false, "",
      Span<std::string_view>{detail::OPTIMIZER_GRAD_STRATEGY_checks, 6}},
  Key{KeyId::OPTIMIZER_PARAM, "OPTIMIZER.PARAM", "PARAM", "OPTIMIZER",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY) | block_bit(BlockId::EXPLORE) | block_bit(BlockId::REFINE) | block_bit(BlockId::INNER),
      Span<Arg>{detail::OPTIMIZER_PARAM_args, 2}, Repeat::Add, "", "", false,
      "", false, KeyId::OPTIMIZER_PARAM, false, "",
      Span<std::string_view>{detail::OPTIMIZER_PARAM_checks, 2}},
  Key{KeyId::OPTIMIZER_LOG_HP, "OPTIMIZER.LOG_HP", "LOG_HP", "",
      block_bit(BlockId::OPTIMIZER) | block_bit(BlockId::STRATEGY),
      Span<Arg>{detail::OPTIMIZER_LOG_HP_args, 1}, Repeat::Error,
      "LOG_HP is set twice in the {block} block (first on line {first_line}); each key is given once",
      "LOG_HP is written once, inside OPTIMIZER or STRATEGY: it holds for the whole search space{note}",
      false, "", false, KeyId::OPTIMIZER_LOG_HP, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::STRATEGY_TYPE, "STRATEGY.TYPE", "TYPE", "",
      block_bit(BlockId::STRATEGY), Span<Arg>{detail::STRATEGY_TYPE_args, 1},
      Repeat::Error,
      "TYPE is set twice in the STRATEGY block (first on line {first_line}); each key is given once",
      "", true, "PLAIN", false, KeyId::STRATEGY_TYPE, false, "",
      Span<std::string_view>{detail::STRATEGY_TYPE_checks, 1}},
  Key{KeyId::STRATEGY_N_STARTS, "STRATEGY.N_STARTS", "N_STARTS", "",
      block_bit(BlockId::STRATEGY),
      Span<Arg>{detail::STRATEGY_N_STARTS_args, 1}, Repeat::Error,
      "N_STARTS is set twice in the STRATEGY block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::STRATEGY_N_STARTS, false, "",
      Span<std::string_view>{detail::STRATEGY_N_STARTS_checks, 3}},
  Key{KeyId::STRATEGY_BUDGET_PER_START, "STRATEGY.BUDGET_PER_START",
      "BUDGET_PER_START", "", block_bit(BlockId::STRATEGY),
      Span<Arg>{detail::STRATEGY_BUDGET_PER_START_args, 1}, Repeat::Error,
      "BUDGET_PER_START is set twice in the STRATEGY block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::STRATEGY_BUDGET_PER_START, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::STRATEGY_TOPK, "STRATEGY.TOPK", "TOPK", "",
      block_bit(BlockId::STRATEGY), Span<Arg>{detail::STRATEGY_TOPK_args, 1},
      Repeat::Error,
      "TOPK is set twice in the STRATEGY block (first on line {first_line}); each key is given once",
      "", true, "1", false, KeyId::STRATEGY_TOPK, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::INIT_METHOD, "INIT.METHOD", "METHOD", "",
      block_bit(BlockId::INIT) | block_bit(BlockId::RESTART),
      Span<Arg>{detail::INIT_METHOD_args, 2}, Repeat::Add, "", "", false, "",
      false, KeyId::INIT_METHOD, false, "",
      Span<std::string_view>{detail::INIT_METHOD_checks, 1}},
  Key{KeyId::INIT_K_TOTAL, "INIT.K_TOTAL", "K_TOTAL", "",
      block_bit(BlockId::INIT) | block_bit(BlockId::RESTART),
      Span<Arg>{detail::INIT_K_TOTAL_args, 1}, Repeat::Error,
      "K_TOTAL is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::INIT_K_TOTAL, false, "",
      Span<std::string_view>{detail::INIT_K_TOTAL_checks, 4}},
  Key{KeyId::INIT_SEED, "INIT.SEED", "SEED", "",
      block_bit(BlockId::INIT) | block_bit(BlockId::RESTART),
      Span<Arg>{detail::INIT_SEED_args, 1}, Repeat::Error,
      "SEED is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::INIT_SEED, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::INIT_WARM_ON_MISMATCH, "INIT.WARM_ON_MISMATCH",
      "WARM_ON_MISMATCH", "",
      block_bit(BlockId::INIT) | block_bit(BlockId::RESTART),
      Span<Arg>{detail::INIT_WARM_ON_MISMATCH_args, 1}, Repeat::Error,
      "WARM_ON_MISMATCH is set twice in the {block} block (first on line {first_line}); each key is given once",
      "", true, "warn", false, KeyId::INIT_WARM_ON_MISMATCH, false, "",
      Span<std::string_view>{nullptr, 0}},
  Key{KeyId::NOISE_AUTO, "NOISE.AUTO", "AUTO", "", block_bit(BlockId::NOISE),
      Span<Arg>{detail::NOISE_AUTO_args, 1}, Repeat::Error,
      "AUTO is set twice in the NOISE block (first on line {first_line}); each key is given once",
      "", true, "false", false, KeyId::NOISE_AUTO, false, "",
      Span<std::string_view>{detail::NOISE_AUTO_checks, 1}},
  Key{KeyId::NOISE_CALIBRATION, "NOISE.CALIBRATION", "CALIBRATION", "",
      block_bit(BlockId::NOISE), Span<Arg>{detail::NOISE_CALIBRATION_args, 1},
      Repeat::Error,
      "CALIBRATION is set twice in the NOISE block (first on line {first_line}); each key is given once",
      "", true, "0", false, KeyId::NOISE_CALIBRATION, false, "",
      Span<std::string_view>{detail::NOISE_CALIBRATION_checks, 1}},
  Key{KeyId::RESTART_AFTER_STAGNATION, "RESTART.AFTER_STAGNATION",
      "AFTER_STAGNATION", "", block_bit(BlockId::RESTART),
      Span<Arg>{detail::RESTART_AFTER_STAGNATION_args, 1}, Repeat::Error,
      "AFTER_STAGNATION is set twice in the RESTART block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::RESTART_AFTER_STAGNATION, false, "",
      Span<std::string_view>{detail::RESTART_AFTER_STAGNATION_checks, 1}},
  Key{KeyId::RESTART_MAX, "RESTART.MAX", "MAX", "",
      block_bit(BlockId::RESTART), Span<Arg>{detail::RESTART_MAX_args, 1},
      Repeat::Error,
      "MAX is set twice in the RESTART block (first on line {first_line}); each key is given once",
      "", false, "", false, KeyId::RESTART_MAX, false, "",
      Span<std::string_view>{detail::RESTART_MAX_checks, 1}},
  Key{KeyId::ROOT_PC_EMAE, "ROOT.PC_EMAE", "PC_EMAE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_EMAE_args, 2},
      Repeat::Error,
      "PC_EMAE: repeated declaration (first declared on line {first_line})",
      "", false, "", false, KeyId::ROOT_PC_EMAE, false, "",
      Span<std::string_view>{detail::ROOT_PC_EMAE_checks, 4}},
  Key{KeyId::ROOT_PC_ERMSE, "ROOT.PC_ERMSE", "PC_ERMSE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_ERMSE_args, 3},
      Repeat::Add, "", "", false, "", false, KeyId::ROOT_PC_ERMSE, false, "",
      Span<std::string_view>{detail::ROOT_PC_ERMSE_checks, 13}},
  Key{KeyId::ROOT_PC_ErRMSE, "ROOT.PC_ErRMSE", "PC_ErRMSE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_ErRMSE_args, 2},
      Repeat::Error,
      "PC_ErRMSE: repeated declaration (first declared on line {first_line})",
      "", false, "", false, KeyId::ROOT_PC_ErRMSE, false, "",
      Span<std::string_view>{detail::ROOT_PC_ErRMSE_checks, 4}},
  Key{KeyId::ROOT_PC_ERSQ, "ROOT.PC_ERSQ", "PC_ERSQ", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_ERSQ_args, 2},
      Repeat::Error,
      "PC_ERSQ: repeated declaration (first declared on line {first_line})",
      "", false, "", false, KeyId::ROOT_PC_ERSQ, false, "",
      Span<std::string_view>{detail::ROOT_PC_ERSQ_checks, 6}},
  Key{KeyId::ROOT_PC_SRMSE_GPA, "ROOT.PC_SRMSE_GPA", "PC_SRMSE_GPA", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_SRMSE_GPA_args, 3},
      Repeat::Add, "", "", false, "", false, KeyId::ROOT_PC_SRMSE_GPA, false,
      "", Span<std::string_view>{detail::ROOT_PC_SRMSE_GPA_checks, 11}},
  Key{KeyId::ROOT_PC_FRMSE, "ROOT.PC_FRMSE", "PC_FRMSE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_FRMSE_args, 3},
      Repeat::Add, "", "", false, "", false, KeyId::ROOT_PC_FRMSE, false, "",
      Span<std::string_view>{detail::ROOT_PC_FRMSE_checks, 13}},
  Key{KeyId::ROOT_PC_LAMMPS, "ROOT.PC_LAMMPS", "PC_LAMMPS", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_LAMMPS_args, 2},
      Repeat::Add, "", "", false, "", false, KeyId::ROOT_PC_LAMMPS, false, "",
      Span<std::string_view>{detail::ROOT_PC_LAMMPS_checks, 13}},
  Key{KeyId::ROOT_PC_FAILPROBE, "ROOT.PC_FAILPROBE", "PC_FAILPROBE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_FAILPROBE_args, 2},
      Repeat::Error, "only one PC_FAILPROBE line is allowed", "", false, "",
      false, KeyId::ROOT_PC_FAILPROBE, false, "",
      Span<std::string_view>{detail::ROOT_PC_FAILPROBE_checks, 6}},
  Key{KeyId::ROOT_PC_OBSERVABLE, "ROOT.PC_OBSERVABLE", "PC_OBSERVABLE", "",
      block_bit(BlockId::ROOT), Span<Arg>{detail::ROOT_PC_OBSERVABLE_args, 2},
      Repeat::Add, "", "", false, "", false, KeyId::ROOT_PC_OBSERVABLE, false,
      "", Span<std::string_view>{detail::ROOT_PC_OBSERVABLE_checks, 1}},
  Key{KeyId::ROOT_LOSS, "ROOT.LOSS", "LOSS", "", block_bit(BlockId::ROOT),
      Span<Arg>{detail::ROOT_LOSS_args, 1}, Repeat::Error,
      "LOSS is set twice (first on line {first_line}) and no line between reads the first (a --varloss or PC_OBSERVABLE term that names no loss function); each key is given once",
      "", true, "L2", false, KeyId::ROOT_LOSS, false, "",
      Span<std::string_view>{detail::ROOT_LOSS_checks, 8}},
  Key{KeyId::ROOT_OPTIM, "ROOT.OPTIM", "OPTIM", "", block_bit(BlockId::ROOT),
      Span<Arg>{detail::ROOT_OPTIM_args, 4}, Repeat::Add, "", "", false, "",
      false, KeyId::ROOT_OPTIM, false, "", Span<std::string_view>{nullptr, 0}},
};

/* The top-level keys a block refuses as such: the keys valid at the top
 * level, less the names of the keys of a group. */
inline constexpr Span<std::string_view> kTopLevelKeys = Span<std::string_view>{detail::top_level_keys, 26};

inline constexpr LineOption kLineOptions[] = {
  LineOption{LineOptionId::RHOGATE_R_LO, "RHOGATE.R_LO", "R_LO",
      Span<std::string_view>{detail::RHOGATE_R_LO_lines, 1},
      Span<Arg>{detail::RHOGATE_R_LO_args, 1}, Repeat::Error,
      "RHOGATE: R_LO is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::RHOGATE_R_LO_checks, 2}},
  LineOption{LineOptionId::RHOGATE_NPTS, "RHOGATE.NPTS", "NPTS",
      Span<std::string_view>{detail::RHOGATE_NPTS_lines, 1},
      Span<Arg>{detail::RHOGATE_NPTS_args, 1}, Repeat::Error,
      "RHOGATE: NPTS is given twice on this line; each option is given once",
      true, "20001", false,
      Span<std::string_view>{detail::RHOGATE_NPTS_checks, 1}},
  LineOption{LineOptionId::RHOGATE_SEED, "RHOGATE.SEED", "SEED",
      Span<std::string_view>{detail::RHOGATE_SEED_lines, 1},
      Span<Arg>{detail::RHOGATE_SEED_args, 1}, Repeat::Error,
      "RHOGATE: SEED is given twice on this line; each option is given once",
      true, "ERROR", false,
      Span<std::string_view>{detail::RHOGATE_SEED_checks, 1}},
  LineOption{LineOptionId::RHOBARGATE_NPTS, "RHOBARGATE.NPTS", "NPTS",
      Span<std::string_view>{detail::RHOBARGATE_NPTS_lines, 1},
      Span<Arg>{detail::RHOBARGATE_NPTS_args, 1}, Repeat::Error,
      "RHOBARGATE: NPTS is given twice on this line; each option is given once",
      true, "1001", false,
      Span<std::string_view>{detail::RHOBARGATE_NPTS_checks, 1}},
  LineOption{LineOptionId::RHOBARGATE_SEED, "RHOBARGATE.SEED", "SEED",
      Span<std::string_view>{detail::RHOBARGATE_SEED_lines, 1},
      Span<Arg>{detail::RHOBARGATE_SEED_args, 1}, Repeat::Error,
      "RHOBARGATE: SEED is given twice on this line; each option is given once",
      true, "ERROR", false,
      Span<std::string_view>{detail::RHOBARGATE_SEED_checks, 1}},
  LineOption{LineOptionId::PC_EMAE_skipabove, "PC_EMAE.--skip-above",
      "--skip-above",
      Span<std::string_view>{detail::PC_EMAE_skipabove_lines, 5},
      Span<Arg>{detail::PC_EMAE_skipabove_args, 1}, Repeat::Error,
      "{key}: repeated --skip-above", false, "", false,
      Span<std::string_view>{detail::PC_EMAE_skipabove_checks, 1}},
  LineOption{LineOptionId::PC_EMAE_skiprel, "PC_EMAE.--skip-rel", "--skip-rel",
      Span<std::string_view>{detail::PC_EMAE_skiprel_lines, 5},
      Span<Arg>{detail::PC_EMAE_skiprel_args, 1}, Repeat::Error,
      "{key}: repeated --skip-rel", false, "", false,
      Span<std::string_view>{detail::PC_EMAE_skiprel_checks, 1}},
  LineOption{LineOptionId::PC_LAMMPS_script, "PC_LAMMPS.--script", "--script",
      Span<std::string_view>{detail::PC_LAMMPS_script_lines, 2},
      Span<Arg>{detail::PC_LAMMPS_script_args, 1}, Repeat::Error,
      "{key} --script is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_LAMMPS_script_checks, 1}},
  LineOption{LineOptionId::PC_LAMMPS_varloss, "PC_LAMMPS.--varloss",
      "--varloss", Span<std::string_view>{detail::PC_LAMMPS_varloss_lines, 1},
      Span<Arg>{detail::PC_LAMMPS_varloss_args, 4}, Repeat::Add, "", false, "",
      false, Span<std::string_view>{detail::PC_LAMMPS_varloss_checks, 5}},
  LineOption{LineOptionId::PC_LAMMPS_invar, "PC_LAMMPS.--invar", "--invar",
      Span<std::string_view>{detail::PC_LAMMPS_invar_lines, 2},
      Span<Arg>{detail::PC_LAMMPS_invar_args, 2}, Repeat::Add, "", false, "",
      false, Span<std::string_view>{detail::PC_LAMMPS_invar_checks, 4}},
  LineOption{LineOptionId::PC_LAMMPS_outvar, "PC_LAMMPS.--outvar", "--outvar",
      Span<std::string_view>{detail::PC_LAMMPS_outvar_lines, 2},
      Span<Arg>{detail::PC_LAMMPS_outvar_args, 1}, Repeat::Add, "", false, "",
      false, Span<std::string_view>{detail::PC_LAMMPS_outvar_checks, 1}},
  LineOption{LineOptionId::PC_LAMMPS_failscore, "PC_LAMMPS.--failscore",
      "--failscore",
      Span<std::string_view>{detail::PC_LAMMPS_failscore_lines, 2},
      Span<Arg>{detail::PC_LAMMPS_failscore_args, 1}, Repeat::Error,
      "{key} --failscore is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_LAMMPS_failscore_checks, 2}},
  LineOption{LineOptionId::PC_LAMMPS_maxtime, "PC_LAMMPS.--maxtime",
      "--maxtime", Span<std::string_view>{detail::PC_LAMMPS_maxtime_lines, 2},
      Span<Arg>{detail::PC_LAMMPS_maxtime_args, 1}, Repeat::Error,
      "{key} --maxtime is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_LAMMPS_maxtime_checks, 1}},
  LineOption{LineOptionId::PC_LAMMPS_hotvar, "PC_LAMMPS.--hotvar", "--hotvar",
      Span<std::string_view>{detail::PC_LAMMPS_hotvar_lines, 1},
      Span<Arg>{detail::PC_LAMMPS_hotvar_args, 4}, Repeat::Add, "", false, "",
      false, Span<std::string_view>{detail::PC_LAMMPS_hotvar_checks, 6}},
  LineOption{LineOptionId::PC_LAMMPS_failvar, "PC_LAMMPS.--failvar",
      "--failvar", Span<std::string_view>{detail::PC_LAMMPS_failvar_lines, 1},
      Span<Arg>{detail::PC_LAMMPS_failvar_args, 2}, Repeat::Add, "", false, "",
      false, Span<std::string_view>{detail::PC_LAMMPS_failvar_checks, 5}},
  LineOption{LineOptionId::PC_LAMMPS_ncpu, "PC_LAMMPS.--ncpu", "--ncpu",
      Span<std::string_view>{detail::PC_LAMMPS_ncpu_lines, 2},
      Span<Arg>{nullptr, 0}, Repeat::LastWins, "", false, "", true,
      Span<std::string_view>{detail::PC_LAMMPS_ncpu_checks, 1}},
  LineOption{LineOptionId::PC_FAILPROBE_gate, "PC_FAILPROBE.--gate", "--gate",
      Span<std::string_view>{detail::PC_FAILPROBE_gate_lines, 1},
      Span<Arg>{detail::PC_FAILPROBE_gate_args, 1}, Repeat::Error,
      "PC_FAILPROBE --gate is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_FAILPROBE_gate_checks, 1}},
  LineOption{LineOptionId::PC_FAILPROBE_s0, "PC_FAILPROBE.--s0", "--s0",
      Span<std::string_view>{detail::PC_FAILPROBE_s0_lines, 1},
      Span<Arg>{detail::PC_FAILPROBE_s0_args, 1}, Repeat::Error,
      "PC_FAILPROBE --s0 is given twice on this line; each option is given once",
      true, "1", false,
      Span<std::string_view>{detail::PC_FAILPROBE_s0_checks, 2}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_species,
      "PC_OBSERVABLE.cohesive_energy.--species", "--species",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_species_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_species_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --species is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_species_checks, 2}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_target,
      "PC_OBSERVABLE.cohesive_energy.--target", "--target",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_target_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_target_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --target is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_target_checks, 1}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_weight,
      "PC_OBSERVABLE.cohesive_energy.--weight", "--weight",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_weight_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_weight_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --weight is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_weight_checks, 1}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_hotvar,
      "PC_OBSERVABLE.cohesive_energy.--hotvar", "--hotvar",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_hotvar_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_hotvar_args, 4},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --hotvar is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_hotvar_checks, 2}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_lattice,
      "PC_OBSERVABLE.cohesive_energy.--lattice", "--lattice",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_lattice_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_lattice_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --lattice is given twice on this line; each option is given once",
      false, "", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_supercell,
      "PC_OBSERVABLE.cohesive_energy.--supercell", "--supercell",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_supercell_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_supercell_args, 3},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --supercell is given twice on this line; each option is given once",
      true, "4 4 4", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_loss,
      "PC_OBSERVABLE.cohesive_energy.--loss", "--loss",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_loss_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_loss_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --loss is given twice on this line; each option is given once",
      false, "", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_maxiter,
      "PC_OBSERVABLE.cohesive_energy.--maxiter", "--maxiter",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_maxiter_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_maxiter_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --maxiter is given twice on this line; each option is given once",
      true, "5000", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_maxeval,
      "PC_OBSERVABLE.cohesive_energy.--maxeval", "--maxeval",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_maxeval_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_maxeval_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --maxeval is given twice on this line; each option is given once",
      true, "5000", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_maxwalltime,
      "PC_OBSERVABLE.cohesive_energy.--max-wall-time", "--max-wall-time",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_maxwalltime_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_maxwalltime_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --max-wall-time is given twice on this line; each option is given once",
      true, "0", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_etol,
      "PC_OBSERVABLE.cohesive_energy.--etol", "--etol",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_etol_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_etol_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --etol is given twice on this line; each option is given once",
      true, "0", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_fmaxconverged,
      "PC_OBSERVABLE.cohesive_energy.--fmax-converged", "--fmax-converged",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_fmaxconverged_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_fmaxconverged_args, 1},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --fmax-converged is given twice on this line; each option is given once",
      true, "0.1", false,
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_fmaxconverged_checks, 1}},
  LineOption{LineOptionId::PC_OBSERVABLE_cohesive_energy_unconverged,
      "PC_OBSERVABLE.cohesive_energy.--unconverged", "--unconverged",
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_unconverged_lines, 1},
      Span<Arg>{detail::PC_OBSERVABLE_cohesive_energy_unconverged_args, 2},
      Repeat::Error,
      "PC_OBSERVABLE cohesive_energy: --unconverged is given twice on this line; each option is given once",
      true, "const 100", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::OPTIM_AUTO_RATIO, "OPTIM.AUTO.RATIO", "RATIO",
      Span<std::string_view>{detail::OPTIM_AUTO_RATIO_lines, 1},
      Span<Arg>{detail::OPTIM_AUTO_RATIO_args, 1}, Repeat::Error,
      "OPTIM {key}: AUTO RATIO is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::OPTIM_AUTO_RATIO_checks, 3}},
  LineOption{LineOptionId::OPTIM_AUTO_FRAC, "OPTIM.AUTO.FRAC", "FRAC",
      Span<std::string_view>{detail::OPTIM_AUTO_FRAC_lines, 1},
      Span<Arg>{detail::OPTIM_AUTO_FRAC_args, 1}, Repeat::Error,
      "OPTIM {key}: AUTO FRAC is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::OPTIM_AUTO_FRAC_checks, 6}},
  LineOption{LineOptionId::OPTIM_AUTO_FLOOR, "OPTIM.AUTO.FLOOR", "FLOOR",
      Span<std::string_view>{detail::OPTIM_AUTO_FLOOR_lines, 1},
      Span<Arg>{detail::OPTIM_AUTO_FLOOR_args, 1}, Repeat::Error,
      "OPTIM {key}: AUTO FLOOR is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::OPTIM_AUTO_FLOOR_checks, 3}},
  LineOption{LineOptionId::OPTIM_AUTO_GROUP_FLOOR, "OPTIM.AUTO.GROUP_FLOOR",
      "GROUP_FLOOR",
      Span<std::string_view>{detail::OPTIM_AUTO_GROUP_FLOOR_lines, 1},
      Span<Arg>{detail::OPTIM_AUTO_GROUP_FLOOR_args, 1}, Repeat::Error,
      "OPTIM {key}: AUTO GROUP_FLOOR is given twice on this line; each option is given once",
      false, "", false,
      Span<std::string_view>{detail::OPTIM_AUTO_GROUP_FLOOR_checks, 3}},
  LineOption{LineOptionId::OPTIM_LIN, "OPTIM.LIN", "LIN",
      Span<std::string_view>{detail::OPTIM_LIN_lines, 1},
      Span<Arg>{nullptr, 0}, Repeat::Error,
      "OPTIM {key}: unexpected token(s) '{text}' -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>] or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]",
      false, "", false, Span<std::string_view>{nullptr, 0}},
  LineOption{LineOptionId::OPTIM_LOG, "OPTIM.LOG", "LOG",
      Span<std::string_view>{detail::OPTIM_LOG_lines, 1},
      Span<Arg>{nullptr, 0}, Repeat::Error,
      "OPTIM {key}: unexpected token(s) '{text}' -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>] or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]",
      false, "", false, Span<std::string_view>{detail::OPTIM_LOG_checks, 1}},
  LineOption{LineOptionId::OPTIM_ASINH, "OPTIM.ASINH", "ASINH",
      Span<std::string_view>{detail::OPTIM_ASINH_lines, 1},
      Span<Arg>{detail::OPTIM_ASINH_args, 1}, Repeat::Error,
      "OPTIM {key}: unexpected token(s) '{text}' -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>] or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]",
      false, "", false, Span<std::string_view>{detail::OPTIM_ASINH_checks, 2}},
};

inline constexpr Form kForms[] = {
  Form{FormId::PC_OBSERVABLE_cohesive_energy, "PC_OBSERVABLE.cohesive_energy",
      "PC_OBSERVABLE", "cohesive_energy", Span<Arg>{nullptr, 0}, false,
      Span<std::string_view>{detail::PC_OBSERVABLE_cohesive_energy_checks, 7}},
  Form{FormId::OPTIM_range, "OPTIM.range", "OPTIM", "",
      Span<Arg>{detail::OPTIM_range_args, 3}, false,
      Span<std::string_view>{detail::OPTIM_range_checks, 5}},
  Form{FormId::OPTIM_AUTO, "OPTIM.AUTO", "OPTIM", "AUTO",
      Span<Arg>{detail::OPTIM_AUTO_args, 1}, false,
      Span<std::string_view>{detail::OPTIM_AUTO_checks, 10}},
};

inline constexpr Removed kRemoved[] = {
  Removed{"LOCAL",
      "renamed to INNER, the block of the local search that an NLopt MLSL or AUGLAG algorithm runs."},
  Removed{"ENDLOCAL", "renamed to ENDINNER."},
  Removed{"PRESTAGE",
      "replaced by `STRATEGY { TYPE PRESTAGE EXPLORE { ... } [REFINE { ... }] }` (REFINE is optional; without it the STRATEGY-level LIB/ALGO refines)."},
  Removed{"ENDPRESTAGE",
      "replaced by ENDEXPLORE / ENDREFINE / ENDSTRATEGY (see STRATEGY block)."},
  Removed{"MULTISTART",
      "moved into `STRATEGY { TYPE MULTISTART N_STARTS N }`."},
  Removed{"MULTISTART_PARALLEL",
      "the starts of STRATEGY TYPE MULTISTART run one after another; THREADS sets the number of workers that run the work of the evaluations (the LAMMPS scripts and the training and prediction of the potential)."},
  Removed{"PARALLEL",
      "the starts of STRATEGY TYPE MULTISTART run one after another; THREADS sets the number of workers that run the work of the evaluations (the LAMMPS scripts and the training and prediction of the potential)."},
  Removed{"MULTISTART_BUDGET_PER_START",
      "moved into `STRATEGY { TYPE MULTISTART BUDGET_PER_START N }`."},
  Removed{"STAGE_GLOBAL",
      "STRATEGY TYPE PRESTAGE runs an exploration and then a refinement."},
  Removed{"RESTART_AFTER_STAGNATION",
      "moved into `STRATEGY { RESTART { AFTER_STAGNATION N } ENDRESTART }`."},
  Removed{"RESTART_MAX",
      "moved into `STRATEGY { RESTART { MAX N } ENDRESTART }`."},
  Removed{"RESTART_INIT_STRATEGY",
      "moved into `RESTART { METHOD <method> ... }` (re-uses INIT-block grammar)."},
  Removed{"INIT_STRATEGY",
      "moved into `INIT { METHOD <method> ... } ENDINIT` block."},
  Removed{"INIT_K", "renamed to K_TOTAL inside an INIT block."},
  Removed{"INIT_SEED", "renamed to SEED inside an INIT block."},
  Removed{"INIT_INCLUDE_CONFIG",
      "the values of the training configuration for the search parameters are a starting point only when the INIT block has a `METHOD CONFIG` line."},
  Removed{"INCLUDE_CONFIG",
      "the values of the training configuration for the search parameters are a starting point only when the INIT block has a `METHOD CONFIG` line."},
  Removed{"NOISE_AUTO",
      "moved into `NOISE { AUTO true|false } ENDNOISE` block."},
  Removed{"NOISE_CALIBRATION",
      "moved into `NOISE { CALIBRATION N } ENDNOISE` block."},
  Removed{"PC_SRMSE",
      "replaced by PC_SRMSE_GPA, the root mean square of the nine entries of each residual stress tensor in GPa (PC_SRMSE was that of the six entries x <= y in eV/A^3). PC_SRMSE_GPA of a residual is 160.2 times its PC_SRMSE when the six entries have equal mean squares, 130.8 times for a residual on the diagonal only (a pressure) and 185.0 times for a shear residual only: a target or --skip-above cap of PC_SRMSE times 160.2 is about that of PC_SRMSE_GPA, and a weight under LOSS L2 divided by about 25 670 (160.2 squared)."},
};

inline constexpr Rule kRules[] = {
  Rule{"RULE.density_key_arming", RuleCheck::Parse,
      "{key} cannot act: no OPTIM line searches a key that changes the density or its window ({keys} of a DM_REAM many-body term, or a DENSITY parameter <ID>.DENSITY.<parameter> of a DM_Embed component), so every candidate has the density of the training configuration and its one verdict (OPTIM WATOM only scales the density; the verdict does not change). The {key} line takes effect with an OPTIM line on one of these keys, anywhere in the file."},
  Rule{"RULE.inner_block_required", RuleCheck::Startup,
      "{name} needs an INNER block ({phase}): NLopt has no default local optimiser for it; an INNER block with LIB NLOPT and a local ALGO, such as LN_BOBYQA, names one, or {list} run {default} without one"},
  Rule{"RULE.inner_block_not_accepted", RuleCheck::Startup,
      "{name} takes no INNER block (the INNER block of {phase}): only NLopt's MLSL and AUGLAG algorithms run a local search inside their run, and an INNER block is taken by them only ({list})"},
  Rule{"RULE.inner_not_nlopt", RuleCheck::Startup,
      "the INNER block of {phase} names {lib}:{algo}, but {name} runs its INNER search with NLopt: the INNER block takes LIB NLOPT and an NLopt ALGO, such as LN_BOBYQA"},
  Rule{"RULE.inner_needs_own_inner", RuleCheck::Startup,
      "the INNER block of {phase} names {lib}:{algo}, which needs an INNER block of its own; INNER blocks do not nest, and the INNER block takes a local algorithm, such as LN_BOBYQA"},
  Rule{"RULE.inner_global_search", RuleCheck::Startup,
      "the INNER block of {phase}: {name} cannot run {lib}:{algo} as its INNER search: an INNER search is a local search, which {name} starts at {text}, and {what}. A local NLopt algorithm in the INNER block starts at the point it is given ({list}); {lib}:{algo} runs as the exploration (EXPLORE block) or once, as STRATEGY TYPE PLAIN."},
  Rule{"RULE.local_search_needs_stop", RuleCheck::Startup,
      "{phase}: the local search of {name} (NLopt's default {lib}:{algo}, as the block has no INNER block) needs a stop: {why}; set {list} in the {name} block, which NLopt hands on to it{note}"},
  Rule{"RULE.inner_search_needs_stop", RuleCheck::Startup,
      "{phase}: the INNER search of {name} ({lib}:{algo}) needs a stop: {why}; set {list} in the INNER block, or {keys} in the {name} block, which NLopt hands on to it{note}"},
  Rule{"RULE.search_dimensions_required", RuleCheck::Startup,
      "{phase}: {lib}:{algo} cannot optimise a zero-dimensional search space (there is no OPTIM line; the failure would only surface once the optimiser starts). Add at least one OPTIM line, or use a TADAH-native optimiser, which evaluates the INIT start point once."},
  Rule{"RULE.asa_search_dimensions", RuleCheck::Startup,
      "{phase}: ASA:ASA requires at least one OPTIM dimension"},
  Rule{"RULE.multi_rank_optimiser", RuleCheck::Startup,
      "a run on {count} MPI ranks takes STRATEGY TYPE PLAIN with TADAH:ANNEAL_PT or TADAH:ANNEAL_PA only, and the block has STRATEGY TYPE {type} with {lib}:{algo}"},
  Rule{"RULE.multi_rank_restart", RuleCheck::Startup,
      "a run on {count} MPI ranks takes no RESTART block (AFTER_STAGNATION {value}, MAX {max}): each rank counts its own evaluations without a new best, so one rank would restart without the others; a run with a RESTART block runs on one rank"},
  Rule{"RULE.multi_rank_population", RuleCheck::Startup,
      "TADAH:{algo}: {key} ({value}) must be >= the MPI world size ({count})"},
  Rule{"RULE.set_once_for_optimiser", RuleCheck::Parse,
      "{key} is set on line {first_line} in the {open} block and again here in the {block} block; both blocks configure the same {what}, so each key is given once"},
  Rule{"RULE.set_once_for_starting_points", RuleCheck::Parse,
      "{key} is set on line {first_line} in the INIT block inside {parent} and again here in the INIT block inside {enclosing}; both blocks configure the same starting points, so each key is given once"},
  Rule{"RULE.restart_grid", RuleCheck::Startup,
      "a RESTART round would run the grid search of {phase} (TADAH:GRID) again from its first node: a grid search evaluates its starting point and then the nodes of its grid in one fixed order, so each round would evaluate again the nodes the rounds before it evaluated and never reach the rest of the grid; a grid search has no RESTART block (MAXEVAL bounds it)"},
  Rule{"RULE.OPTIM_MPARAMS", RuleCheck::Startup,
      "OPTIM MPARAMS ({index}) is not a valid search target: {why}"},
  Rule{"RULE.OPTIM_MPARAMS_positive", RuleCheck::Startup,
      "OPTIM MPARAMS ({index}): the box [{low}, {high}] of {name}'s {key} must be positive ({key} > 0)."},
  Rule{"RULE.OPTIM_file_cutoff", RuleCheck::Startup,
      "OPTIM {key} ({index}) is not a valid search target: {why} and refuses another value, so every candidate but the value of the training configuration would fail."},
  Rule{"RULE.OPTIM_unread_watom", RuleCheck::Startup,
      "OPTIM WATOM (line {line} of the target file) is not a valid search target: no descriptor of the training configuration reads WATOM (D2_EAM, DM_EAM, D2_Dummy and DM_Dummy read none), and a fit refuses a WATOM other than 1 that no descriptor reads, so every candidate other than 1 would fail its evaluation."},
  Rule{"RULE.settings_taken_by_lines", RuleCheck::Parse,
      "{key} is set twice (first on line {first_line}{note}) and no line between reads the first ({list}); each key is given once"},
  Rule{"RULE.PC_OBSERVABLE_excludes_PC_LAMMPS", RuleCheck::Parse,
      "{key}: this target file also has {list}; PC_OBSERVABLE lines do not share a target file with PC_LAMMPS or PC_FAILPROBE lines"},
  Rule{"RULE.PC_LAMMPS_hotvar_unique", RuleCheck::Parse,
      "PC_LAMMPS --hotvar {name}: line {line} declares --hotvar {name} as well (script {script}); one line declares each --hotvar name, as an evaluation keeps one value of each name for the scripts that read it, and the line that finished last would set it"},
  Rule{"RULE.PC_OBSERVABLE_GRAD_STRATEGY", RuleCheck::Parse,
      "--hpotarget defines PC_OBSERVABLE but {block} GRAD_STRATEGY=NUMERICAL is set. PC_OBSERVABLE is built for the analytical gradient: use GRAD_STRATEGY ANALYTICAL (an unset GRAD_STRATEGY resolves to it in a block that obtains gradients; it needs the settings of the analytical gradient, such as OALGO 4 with a fixed LAMBDA >= 0, and startup names a missing one) or GRAD_STRATEGY HYBRID, which uses finite differences where the analytical gradient is unavailable."},
  Rule{"RULE.PC_LAMMPS_GRAD_STRATEGY", RuleCheck::Parse,
      "--hpotarget defines {key} on line {line} but {block} GRAD_STRATEGY is set. A PC_LAMMPS or PC_FAILPROBE line does not support analytical/numerical/hybrid gradients. Drop the GRAD_STRATEGY line."},
  Rule{"RULE.PC_LAMMPS_gradient_free", RuleCheck::Parse,
      "--hpotarget defines {key} on line {line} but {block} optimiser {lib}:{algo} is gradient-based. A PC_LAMMPS or PC_FAILPROBE line requires a gradient-free optimiser ({list}); an NLopt MLSL or AUGLAG algorithm is accepted with a derivative-free INNER block."},
  Rule{"RULE.PC_LAMMPS_local_optimiser", RuleCheck::Parse,
      "--hpotarget defines {key} on line {line} but {block} optimiser {lib}:{algo} takes a gradient: NLopt's MLSL and AUGLAG take one for their local search, and NLopt's default local optimiser, {name}, is gradient-based; an INNER block with a derivative-free one takes none. A PC_LAMMPS or PC_FAILPROBE line requires a gradient-free optimiser ({list}); an NLopt MLSL or AUGLAG algorithm is accepted with a derivative-free INNER block."},
  Rule{"RULE.PC_LAMMPS_INNER_local_optimiser", RuleCheck::Parse,
      "--hpotarget defines {key} on line {line} but {block} optimiser {lib}:{algo} takes a gradient: NLopt's MLSL and AUGLAG take one for their local search, and NLopt's default local optimiser, {name}, is gradient-based. A PC_LAMMPS or PC_FAILPROBE line requires a gradient-free optimiser, and an INNER search is a local NLopt algorithm; the INNER block takes a derivative-free one ({list})."},
  Rule{"RULE.PC_LAMMPS_INNER_gradient_free", RuleCheck::Parse,
      "--hpotarget defines {key} on line {line} but {block} optimiser {lib}:{algo} is gradient-based. A PC_LAMMPS or PC_FAILPROBE line requires a gradient-free optimiser, and an INNER search is a local NLopt algorithm; the INNER block takes a derivative-free one ({list})."},
  Rule{"RULE.loss_source_required", RuleCheck::Startup,
      "No loss source: a loss term comes from a PC_EMAE, PC_ERMSE, PC_ErRMSE, PC_ERSQ, PC_FRMSE or PC_SRMSE_GPA line (with --validation), a PC_OBSERVABLE line or a PC_LAMMPS line, and the target file has none (a PC_FAILPROBE line gives no loss term)"},
  Rule{"RULE.metric_needs_validation", RuleCheck::Startup,
      "PC_EMAE/PC_ERMSE/PC_ErRMSE/PC_ERSQ/PC_FRMSE/PC_SRMSE_GPA target validation-set errors but no --validation file was given."},
  Rule{"RULE.OPTIM_LAMBDA_evidence", RuleCheck::Parse,
      "OPTIM LAMBDA: the training configuration has LAMBDA -1, which requests the evidence approximation: each fit then finds the penalty itself, and OPTIM LAMBDA would replace it with the values of the box. LAMBDA >= 0 in the training configuration is the starting point of a search of LAMBDA; without OPTIM LAMBDA the evidence approximation finds it in every evaluation."},
  Rule{"RULE.fit_LAMBDA_negative", RuleCheck::Startup,
      "{text}: only LAMBDA -1 requests the evidence approximation, which OALGO 3 runs; another negative value requests nothing. LAMBDA 0 fits without a penalty, and a positive LAMBDA is the ridge penalty."},
  Rule{"RULE.fit_LAMBDA_one_value", RuleCheck::Startup,
      "{text}: LAMBDA takes one value, the ridge penalty (0 for none, -1 for the evidence approximation of OALGO 3); a second value is not read. The rank threshold of OALGO 1 and 3 is OALGO's second value (OALGO 1 {value}, OALGO 3 {value})."},
  Rule{"RULE.fit_LAMBDA_evidence_OALGO", RuleCheck::Startup,
      "LAMBDA -1 requests the evidence approximation, which only OALGO 3 (the SVD solve) runs; the configuration has OALGO {value} (LAMBDA >= 0 fits with any OALGO)."},
  Rule{"RULE.fit_OALGO1_threshold", RuleCheck::Startup,
      "{text}: the rank threshold M of OALGO 1 (DGELSD), its second value, is a number 0 < M < 1 (a singular value at most M times the largest is left out of the coefficients) or below 0 for the machine precision of LAPACK, 2^-53 (OALGO 1 -1); DGELSD takes the machine precision for an M of 0 or of 1 or more."},
  Rule{"RULE.fit_OALGO3_threshold", RuleCheck::Startup,
      "{text}: the rank threshold M of OALGO 3, its second value, is a number 0 < M < 1 (a singular value at most M times the largest is left out of the coefficients) or 0 or below (only a singular value of 0 is left out); an M of 1 or more leaves out every singular direction, every coefficient 0."},
  Rule{"RULE.fit_OALGO_one_value", RuleCheck::Startup,
      "{text}: {name} takes one value: it keeps every direction and has no rank threshold, the second value of OALGO 1 and 3."},
  Rule{"RULE.fit_OALGO_option", RuleCheck::Startup,
      "OALGO {value}: OALGO is 1 (DGELSD), 2 (DGELS, QR), 3 (the SVD solve) or 4 (normal equations, Cholesky)."},
  Rule{"RULE.OPTIM_OALGO_threshold_box", RuleCheck::Parse,
      "OPTIM OALGO (2): the box [{low}, {high}] is not inside (0, 1), but the second value of OALGO is the rank threshold M of OALGO 1 and 3, which only a value 0 < M < 1 sets: every M below 0 is the machine precision under OALGO 1, every M of 0 or below leaves out only a singular value of 0 under OALGO 3, and a fit refuses an M of 1 or more, so the box of a search is inside (0, 1)"},
  Rule{"RULE.fit_OALGO3_intercepts", RuleCheck::Startup,
      "OALGO=3 (SVD with uniform ridge) does not support intercept exclusion from regularisation. The model has {count} intercept column(s) (BIAS=true with ATOM) and LAMBDA != 0; OALGO 1, 2 or 4 fits such a model with LAMBDA != 0."},
  Rule{"RULE.fit_OALGO3_intercepts_evidence", RuleCheck::Startup,
      "LAMBDA -1 requests the evidence approximation, which only OALGO 3 runs; OALGO 3 penalises every weight alike, and the model has {count} intercept column(s) (BIAS true with ATOM), which a ridge penalty leaves out. The evidence approximation needs BIAS false, or every intercept weight pinned with FIXINDEX and FIXWEIGHT. With BIAS true, LAMBDA 0 fits with any OALGO, and LAMBDA > 0 with OALGO 1, 2 or 4."},
  Rule{"RULE.optim_LAMBDA_OALGO3_intercepts", RuleCheck::Startup,
      "OPTIM LAMBDA: OALGO 3 penalises every weight alike, and the fit has {count} intercept column(s) (BIAS true with ATOM), which a ridge penalty leaves out, so every fit refuses a LAMBDA other than 0: each candidate of the box but 0 would fail its evaluation. OALGO 1, 2 or 4 fits a positive LAMBDA with the intercepts unpenalised."},
  Rule{"RULE.fit_krr_implementation", RuleCheck::Startup,
      "{text}: M_KRR has implementation 1, which fits the energies, forces and stresses of the structures, and implementation 2, which fits the energies per atom of whole structures (the third value of MODEL, 1 when it is left out); {value} is neither."},
  Rule{"RULE.fit_krr_implementation2_LAMBDA", RuleCheck::Startup,
      "MODEL M_KRR {name} 2 cannot use LAMBDA -1: implementation 2 solves (K + LAMBDA sigma^2 I) w = t, sigma = ESIGMA in eV, with a fixed LAMBDA and does not run the evidence approximation. Implementation 1 (MODEL M_KRR {name} 1) with OALGO 3 runs it; LAMBDA >= 0 fits with either."},
  Rule{"RULE.optim_relative_error_scale", RuleCheck::Parse,
      "OPTIM {key} ({index}): the low bound {low} is negative, but the relative part b of {key} is a number >= 0, and every fit refuses a negative one, so the low bound of its box is >= 0"},
  Rule{"RULE.fit_krr_implementation2_ignores", RuleCheck::Startup,
      "MODEL M_KRR {name} 2 fits the energies per atom of whole structures over ESIGMA, without forces, stresses or priorities: {what} would be ignored. Implementation 1 (MODEL M_KRR {name} 1) {why}; implementation 2 fits {text}."},
  Rule{"RULE.fit_krr_implementation2_OALGO", RuleCheck::Startup,
      "MODEL M_KRR {name} 2 solves (K + LAMBDA sigma^2 I) w = t by Cholesky and reads no OALGO: {what} would be ignored. Implementation 1 (MODEL M_KRR {name} 1) solves with OALGO; implementation 2 fits {text}."},
  Rule{"RULE.fit_error_scales", RuleCheck::Startup,
      "tadah hpo: ESIGMA {value}: the energy error scale is a number > 0 ({what} that costs as much as a force off by FSIGMA); no coefficients were fitted."},
  Rule{"RULE.optim_error_scale_quantity_off", RuleCheck::Startup,
      "OPTIM {key}: {name} is false: the fit has no {what}, so {key} would change nothing and every candidate would give the same fit. {name} true fits the {what}."},
  Rule{"RULE.optim_error_scale_only_quantity", RuleCheck::Startup,
      "OPTIM {key}: the {what} are the only quantity of the fit and LAMBDA is {value}: the error scale of the only quantity scales every row of the fit alike, which changes no coefficient, so every candidate would give the same fit. With a LAMBDA > 0 the error scale weighs the data against the penalty, and with {text} in the fit, the {what} against them."},
  Rule{"RULE.optim_wdbfile_one_dataset", RuleCheck::Startup,
      "OPTIM WDBFILE: the run has one training dataset and LAMBDA is {value}: its WDBFILE value multiplies the priority of every row of the fit alike, which changes no coefficient, so every candidate would give the same fit. With a LAMBDA > 0 the priority weighs the data against the penalty, and with more training datasets, the dataset against the others."},
  Rule{"RULE.optim_error_scale_no_fit", RuleCheck::Startup,
      "OPTIM {key}: the evaluations of this run fit no coefficient ({why}), so {key} would change nothing and every candidate would give the same potential."},
};

inline constexpr Env kEnvs[] = {
  Env{"TADAH_OPTIM_SCALE", KeyId::ROOT_OPTIM_SCALE,
      Span<std::string_view>{detail::ENV_TADAH_OPTIM_SCALE_values, 2},
      "TADAH_OPTIM_SCALE must be AUTO or NONE (got '{tok}')",
      Span<std::string_view>{nullptr, 0}},
  Env{"TADAH_FAILSCORE_GRADING", KeyId::ROOT_FAILSCORE_GRADING,
      Span<std::string_view>{detail::ENV_TADAH_FAILSCORE_GRADING_values, 2},
      "TADAH_FAILSCORE_GRADING must be ON or OFF (got '{tok}')",
      Span<std::string_view>{detail::ENV_TADAH_FAILSCORE_GRADING_checks, 1}},
  Env{"TADAH_BUMPGATE", KeyId::ROOT_BUMPGATE,
      Span<std::string_view>{detail::ENV_TADAH_BUMPGATE_values, 2},
      "TADAH_BUMPGATE must be CANDIDATE or STRICT (got '{tok}')",
      Span<std::string_view>{nullptr, 0}},
  Env{"TADAH_PC_ISOLATION", KeyId::ROOT_PC_ISOLATION,
      Span<std::string_view>{detail::ENV_TADAH_PC_ISOLATION_values, 2},
      "TADAH_PC_ISOLATION must be THREAD or PROCESS (got '{tok}')",
      Span<std::string_view>{nullptr, 0}},
  Env{"TADAH_PC_WORKER_TIMEOUT", KeyId::ROOT_PC_WORKER_TIMEOUT,
      Span<std::string_view>{nullptr, 0},
      "TADAH_PC_WORKER_TIMEOUT must be a finite number of seconds >= 0 (0 derives the deadline from each script's timeout), got '{tok}'{underflow}",
      Span<std::string_view>{detail::ENV_TADAH_PC_WORKER_TIMEOUT_checks, 2}},
};

inline constexpr LossFunction kLossFunctions[] = {
  LossFunction{"L1", Span<LossParam>{nullptr, 0}},
  LossFunction{"L2", Span<LossParam>{nullptr, 0}},
  LossFunction{"LOG_COSH", Span<LossParam>{nullptr, 0}},
  LossFunction{"MAPE", Span<LossParam>{nullptr, 0}},
  LossFunction{"HUBER", Span<LossParam>{detail::LOSS_HUBER_params, 1}},
  LossFunction{"RMSLE", Span<LossParam>{nullptr, 0}},
  LossFunction{"TUKEY", Span<LossParam>{detail::LOSS_TUKEY_params, 1}},
};

inline constexpr StrategyType kStrategyTypes[] = {
  StrategyType{"PLAIN", Span<KeyId>{nullptr, 0},
      Span<std::string_view>{nullptr, 0}},
  StrategyType{"MULTISTART",
      Span<KeyId>{detail::STRATEGY_TYPE_MULTISTART_keys, 2},
      Span<std::string_view>{detail::STRATEGY_TYPE_MULTISTART_checks, 2}},
  StrategyType{"PRESTAGE", Span<KeyId>{detail::STRATEGY_TYPE_PRESTAGE_keys, 1},
      Span<std::string_view>{detail::STRATEGY_TYPE_PRESTAGE_checks, 3}},
};

inline constexpr InitMethod kInitMethods[] = {
  InitMethod{"CONFIG", Span<Arg>{nullptr, 0}, false,
      Span<std::string_view>{detail::METHOD_CONFIG_checks, 1}},
  InitMethod{"RANDOM", Span<Arg>{detail::METHOD_RANDOM_args, 1}, false,
      Span<std::string_view>{detail::METHOD_RANDOM_checks, 1}},
  InitMethod{"LHS", Span<Arg>{detail::METHOD_LHS_args, 1}, false,
      Span<std::string_view>{detail::METHOD_LHS_checks, 1}},
  InitMethod{"SOBOL", Span<Arg>{detail::METHOD_SOBOL_args, 1}, false,
      Span<std::string_view>{detail::METHOD_SOBOL_checks, 1}},
  InitMethod{"WARM", Span<Arg>{detail::METHOD_WARM_args, 1}, false,
      Span<std::string_view>{detail::METHOD_WARM_checks, 6}},
};

namespace optim {
inline constexpr Span<std::string_view> positive_only = Span<std::string_view>{detail::optim_positive_only, 3};
inline constexpr std::string_view auto_excluded_prefix = "RCUT";
inline constexpr Span<std::string_view> checks = Span<std::string_view>{detail::optim_checks, 26};
}  // namespace optim

inline constexpr OptimTarget kOptimTargets[] = {
  OptimTarget{"WEIGHTS", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM WEIGHTS: WEIGHTS holds the coefficients each evaluation fits, so a searched value would be replaced by the fit; FIXINDEX and FIXWEIGHT in the training configuration hold coefficients fixed, and OPTIM FIXWEIGHT (<coefficient>) <low> <high> searches them"},
  OptimTarget{"EMBF0", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM EMBF0: EMBF0 (ZERO or FREE) chooses the embedding reference of a DM_REAM many-body term, a choice and not a number to search; the training configuration sets it"},
  OptimTarget{"FIXWEIGHT", OptimTargetStatus::Special, Span<int>{nullptr, 0},
      "OPTIM FIXWEIGHT ({index}): coefficient {index} is not in FIXINDEX ({list}); the indices of OPTIM FIXWEIGHT name coefficients, those FIXINDEX lists, not positions among the FIXWEIGHT values"},
  OptimTarget{"OALGO", OptimTargetStatus::Refused,
      Span<int>{detail::OPTIM_TARGET_OALGO_indices, 1},
      "OPTIM OALGO (1) is not a valid search target: the first OALGO value selects the training algorithm (1 GELSD, 2 GELS, 3 SVD, 4 Cholesky), a discrete choice rather than a continuous parameter; the training configuration sets it. The second value, the rcond M of options 1 and 3, is continuous: OPTIM OALGO (2) <low> <high>."},
  OptimTarget{"RHOFORM", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM RHOFORM is not a valid search target: RHOFORM (SIGNED|NONNEG|MONOTONE) selects the DM_REAM density form, a discrete property of the potential (its amplitudes were fitted in that form), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"TYPE2B", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM TYPE2B is not a valid search target: TYPE2B names the two-body descriptor and its structure tokens (basis code, sizes, species), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"TYPEMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM TYPEMB is not a valid search target: TYPEMB names the many-body descriptor and its structure tokens (basis code, sizes, species), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"RCTYPE2B", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM RCTYPE2B is not a valid search target: RCTYPE2B names the two-body cutoff function, structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"RCTYPEMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM RCTYPEMB is not a valid search target: RCTYPEMB names the many-body cutoff function, structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"KDEG2B", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM KDEG2B is not a valid search target: KDEG2B lists the two-body knot degrees (3|5), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"KDEGMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM KDEGMB is not a valid search target: KDEGMB lists the density knot degrees (3|5), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"KDEGEMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM KDEGEMB is not a valid search target: KDEGEMB lists the embedding knot degrees (3|5), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"DIMER", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM DIMER is not a valid search target: DIMER (on/off, the bond length, whether the intramolecular distance enters the descriptors) declares the rigid-dimer layout of the training data (its bond length is that of the data's molecules), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"FIXINDEX", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM FIXINDEX is not a valid search target: FIXINDEX names the coefficients FIXWEIGHT holds fixed (OPTIM FIXWEIGHT searches their values), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"MODEL", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM MODEL is not a valid search target: MODEL names the model and its basis function or kernel (and an integer option), structure data rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"CGRIDMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM CGRIDMB: the potential has no CGRIDMB key: its many-body term is DM_Embed component(s) (DESCRIPTOR blocks), whose density parameter is the named path {list}; an OPTIM line names the path, OPTIM <path> (indices) <low> <high> (the starter target file of tadah refit names the paths; a target file written for a DM_REAM many-body term names CGRIDMB)"},
  OptimTarget{"SGRIDMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM SGRIDMB: the potential has no SGRIDMB key: its many-body term is DM_Embed component(s) (DESCRIPTOR blocks), whose density parameter is the named path {list}; an OPTIM line names the path, OPTIM <path> (indices) <low> <high> (the starter target file of tadah refit names the paths; a target file written for a DM_REAM many-body term names SGRIDMB)"},
  OptimTarget{"AMPGRIDMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM AMPGRIDMB: the potential has no AMPGRIDMB key: its many-body term is DM_Embed component(s) (DESCRIPTOR blocks), whose density parameter is the named path {list}; an OPTIM line names the path, OPTIM <path> (indices) <low> <high> (the starter target file of tadah refit names the paths; a target file written for a DM_REAM many-body term names AMPGRIDMB)"},
  OptimTarget{"CEMBFUNC", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM CEMBFUNC: the potential has no CEMBFUNC key: its many-body term is DM_Embed component(s) (DESCRIPTOR blocks), whose embedding parameter is the named path {list}; an OPTIM line names the path, OPTIM <path> (indices) <low> <high> (the starter target file of tadah refit names the paths; a target file written for a DM_REAM many-body term names CEMBFUNC)"},
  OptimTarget{"SEMBFUNC", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM SEMBFUNC: the potential has no SEMBFUNC key: its many-body term is DM_Embed component(s) (DESCRIPTOR blocks), whose embedding parameter is the named path {list}; an OPTIM line names the path, OPTIM <path> (indices) <low> <high> (the starter target file of tadah refit names the paths; a target file written for a DM_REAM many-body term names SEMBFUNC)"},
  OptimTarget{"RCUTMB", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM RCUTMB: the potential has no RCUTMB key: its many-body term is DM_Embed component(s) (DESCRIPTOR blocks), whose density parameter is the named path {list}; an OPTIM line names the path, OPTIM <path> (indices) <low> <high> (the starter target file of tadah refit names the paths; a target file written for a DM_REAM many-body term names RCUTMB)"},
  OptimTarget{"LSCALE", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM LSCALE is not a valid search target: LSCALE prepares the data and is applied once, when the run starts and reads them, so no evaluation would take a candidate value of it: its value is the training configuration's."},
  OptimTarget{"ESHIFT", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM ESHIFT is not a valid search target: ESHIFT prepares the data and is applied once, when the run starts and reads them, so no evaluation would take a candidate value of it: its value is the training configuration's."},
  OptimTarget{"ESHIFT_ATOM", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM ESHIFT_ATOM is not a valid search target: ESHIFT_ATOM prepares the data and is applied once, when the run starts and reads them, so no evaluation would take a candidate value of it: its value is the training configuration's."},
  OptimTarget{"ESHIFT_DBATOM", OptimTargetStatus::Refused,
      Span<int>{nullptr, 0},
      "OPTIM ESHIFT_DBATOM is not a valid search target: ESHIFT_DBATOM prepares the data and is applied once, when the run starts and reads them, so no evaluation would take a candidate value of it: its value is the training configuration's."},
  OptimTarget{"EFILTER", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM EFILTER is not a valid search target: EFILTER prepares the data and is applied once, when the run starts and reads them, so no evaluation would take a candidate value of it: its value is the training configuration's."},
  OptimTarget{"FFILTER", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM FFILTER is not a valid search target: FFILTER prepares the data and is applied once, when the run starts and reads them, so no evaluation would take a candidate value of it: its value is the training configuration's."},
  OptimTarget{"ZERO_COM_FORCE", OptimTargetStatus::Refused,
      Span<int>{nullptr, 0},
      "OPTIM ZERO_COM_FORCE is not a valid search target: ZERO_COM_FORCE prepares the data and is applied once, when the run starts and reads them, so no evaluation would take a candidate value of it: its value is the training configuration's."},
  OptimTarget{"WDBFILE", OptimTargetStatus::Special, Span<int>{nullptr, 0},
      "OPTIM WDBFILE ({index}): the low bound {low} is not > 0, but a searched WDBFILE value is the priority of a dataset in the fit, a number > 0 (WDBFILE 0 in the training configuration leaves a dataset out of the fit), so the low bound of its box is > 0"},
  OptimTarget{"ESIGMA", OptimTargetStatus::Special, Span<int>{nullptr, 0},
      "OPTIM ESIGMA ({index}): the low bound {low} is not > 0, but ESIGMA is the energy error scale, a number > 0 (meV/atom, or meV per sqrt(atom) under ENORM SQRT), and every fit refuses another, so the low bound of its box is > 0"},
  OptimTarget{"FSIGMA", OptimTargetStatus::Special, Span<int>{nullptr, 0},
      "OPTIM FSIGMA ({index}): the low bound {low} is not > 0, but the absolute part a of FSIGMA is the force error scale of a structure whose forces are zero, a number > 0 (eV/A), and every fit refuses another, so the low bound of its box is > 0"},
  OptimTarget{"SSIGMA", OptimTargetStatus::Special, Span<int>{nullptr, 0},
      "OPTIM SSIGMA ({index}): the low bound {low} is not > 0, but the absolute part a of SSIGMA is the stress error scale of a structure whose stresses are zero, a number > 0 (GPa), and every fit refuses another, so the low bound of its box is > 0"},
  OptimTarget{"ENORM", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM ENORM is not a valid search target: ENORM (ATOM or SQRT) chooses how the residual of an energy is normalised, a choice rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"ENERGY", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM ENERGY is not a valid search target: ENERGY (true or false) says whether the fit has the energies, a switch rather than a continuous parameter; the training configuration sets it."},
  OptimTarget{"EWEIGHT", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM EWEIGHT is not a valid search target: EWEIGHT is a retired key: EWEIGHT multiplied the squared residual of the total energy of each structure, so one value meant a different accuracy for cells of different sizes. Replaced by: ESIGMA, the energy error scale per atom in meV/atom (ENERGY false for EWEIGHT 0). For cells of N atoms, ESIGMA = 1000/(N sqrt(EWEIGHT)) meV/atom gives the same fit. For cells of different sizes, ESIGMA = 1000/sqrt(EWEIGHT) with the eweight of each structure in the data file multiplied by N^2 reproduces the objective of Tadah! 1.5.0. EWEIGHT 0 (no energies in the fit) is ENERGY false. Under M_KRR implementation 2 (MODEL M_KRR <kernel> 2), whose solve did not read EWEIGHT, deleting the line converts it, and ESIGMA 1000, or LAMBDA times 10^6 with ESIGMA 1, gives the solve of Tadah! 1.5.0, (K + LAMBDA I) w = t."},
  OptimTarget{"FWEIGHT", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM FWEIGHT is not a valid search target: FWEIGHT is a retired key: FWEIGHT multiplied the squared residual of each force component, a number without a unit an accuracy could be read in. Replaced by: FSIGMA, the force error scale in eV/Å (FORCE false for FWEIGHT 0). FSIGMA = 1/sqrt(FWEIGHT) eV/Å gives the same fit. With FORCE false, deleting the line converts it (FSIGMA keeps its default). FWEIGHT 0 (no forces in the fit) is FORCE false."},
  OptimTarget{"SWEIGHT", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM SWEIGHT is not a valid search target: SWEIGHT is a retired key: SWEIGHT multiplied the squared residual of each of the six stress entries xx, xy, xz, yy, yz and zz, in eV/Å^3, so a shear cost depended on the orientation of the cell. Replaced by: SSIGMA, the stress error scale in GPa (STRESS false for SWEIGHT 0). SSIGMA = 160.2/sqrt(SWEIGHT) GPa gives the diagonal entries (xx, yy, zz) the same weight; a fit counts each off-diagonal entry twice (the nine entries of the tensor), so the conversion is exact for the diagonal only. With STRESS false, deleting the line converts it (SSIGMA keeps its default). SWEIGHT 0 (no stresses in the fit) is STRESS false."},
  OptimTarget{"WDBFILE_AUTO", OptimTargetStatus::Refused,
      Span<int>{nullptr, 0},
      "OPTIM WDBFILE_AUTO is not a valid search target: WDBFILE_AUTO is a retired key: WDBFILE_AUTO a multiplied the priorities of each dataset by n_d^-a, n_d its number of structures, so a priority in the fit depended on how many structures each dataset had. Replaced by: WDBFILE, a priority for each dataset. Each WDBFILE value (1 for every dataset without WDBFILE) multiplied by n_d^-a, n_d the number of structures of dataset d after EFILTER and FFILTER, gives the same fit. For WDBFILE_AUTO 0, deleting the line converts it (WDBFILE as it is). A retrain does not read a potential's WDBFILE line: the WDBFILE values of a potential (those converted, or those of its WDBFILE line) go on the command line (--wdbfile) or into a file that includes the potential with CONFIG. Under M_KRR implementation 2 (MODEL M_KRR <kernel> 2), whose solve read no priority, deleting the line converts it, and ESIGMA 1000, or LAMBDA times 10^6 with ESIGMA 1, gives the solve of Tadah! 1.5.0, (K + LAMBDA I) w = t."},
  OptimTarget{"EWEIGHT_TEMP", OptimTargetStatus::Refused,
      Span<int>{nullptr, 0},
      "OPTIM EWEIGHT_TEMP is not a valid search target: EWEIGHT_TEMP is a retired key: EWEIGHT_TEMP T multiplied the energy priority of each structure by exp(-(E/N - E_min)/(kB T)), which reads energy above the lowest as heat: compressed or strained structures, high in energy but cold, got almost no priority. Replaced by: the priorities of the data: WDBFILE for each dataset, and the eweight of each structure in its data file. No key converts it. WDBFILE gives each dataset its priority, and the eweight of a structure in its data file the priority of its energy."},
  OptimTarget{"UNCERTAINTY", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM UNCERTAINTY is not a valid search target: UNCERTAINTY is a retired key: tadah train --uncertainty wrote weights.tadah, each coefficient of the fit beside a diagonal entry of SIGMA. Uncertainty output is removed: the errors of Tadah! 1.0.0 to 1.5.0 were not standard deviations, and no calibrated estimate replaces them. Replaced by: nothing. Deleting the option, or the UNCERTAINTY line, converts it: the fit and its coefficients are unaffected."},
  OptimTarget{"WCOV", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM WCOV is not a valid search target: WCOV is a retired key: No release of Tadah! wrote or read WCOV. Uncertainty output is removed: the errors of Tadah! 1.0.0 to 1.5.0 were not standard deviations, and no calibrated estimate replaces them. Replaced by: nothing. Deleting the line converts it: no prediction reads it, and the fit is unaffected."},
  OptimTarget{"WNOISE", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM WNOISE is not a valid search target: WNOISE is a retired key: No release of Tadah! wrote or read WNOISE. Uncertainty output is removed: the errors of Tadah! 1.0.0 to 1.5.0 were not standard deviations, and no calibrated estimate replaces them. Replaced by: nothing. Deleting the line converts it: no prediction reads it, and the fit is unaffected."},
  OptimTarget{"ERROR", OptimTargetStatus::Refused, Span<int>{nullptr, 0},
      "OPTIM ERROR is not a valid search target: ERROR is a retired key: ERROR, the key of --error, asked tadah predict and tadah properties pairwise for errors computed from SIGMA. Uncertainty output is removed: the errors of Tadah! 1.0.0 to 1.5.0 were not standard deviations, and no calibrated estimate replaces them. Replaced by: nothing. Deleting the line converts it: no command reads it, and the fit is unaffected."},
  OptimTarget{"LAMBDA", OptimTargetStatus::Special, Span<int>{nullptr, 0},
      "OPTIM LAMBDA ({index}): the low bound {low} is negative, but LAMBDA is the ridge penalty, a number >= 0: every fit refuses a negative value other than -1, and -1, which requests the evidence approximation, is no value of a box, so the low bound of its box is >= 0"},
};

/* The block of the name `name`; nullptr for none. */
constexpr const Block* find_block(std::string_view name)
{
  for (const Block& b : kBlocks)
    if (b.name == name) return &b;
  return nullptr;
}

/* The key of the name `name` valid at `place`; nullptr for none. */
constexpr const Key* find_key(std::string_view name, BlockId place)
{
  for (const Key& k : kKeys)
    if (k.name == name && (k.blocks & block_bit(place))) return &k;
  return nullptr;
}

/* The option `option` of the line of the key or form `line`; nullptr for
 * none. */
constexpr const LineOption* find_line_option(std::string_view line,
                                             std::string_view option)
{
  for (const LineOption& o : kLineOptions) {
    if (o.option != option) continue;
    for (std::string_view l : o.lines)
      if (l == line) return &o;
  }
  return nullptr;
}

}  // namespace hpotarget_registry
}  // namespace hpo
}  // namespace tadah
