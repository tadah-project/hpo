// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: HPO/blueprint/anneal_keys.toml
// Generator:       HPO/blueprint/gen_anneal_keys.py --header
// Drift gate:      ctest hpo_anneal_registry_sync (gen_anneal_keys.py --check)
#pragma once

#include <cstddef>
#include <string_view>

namespace tadah {
namespace hpo {
namespace anneal_registry {

inline constexpr unsigned KIND_SERIAL = 1u;
inline constexpr unsigned KIND_PT     = 2u;
inline constexpr unsigned KIND_PA     = 4u;

enum class AnnealKeyType : unsigned char { Bool, Count, Float, Enum };

enum class AnnealRuleType : unsigned char {
  RequiresFlag,      // key valid only when bool `arg` parses to 1
  ConflictsFlag,     // key invalid when bool `arg` parses to 1
  UnusedWhenSet,     // key invalid when `arg` is explicitly set
  UnusedWhenFlagOff, // key invalid when bool `arg` parses to 0
  RequiresEnum,      // key valid only when enum `arg` == `arg2`
  KindReject,        // key invalid for `kind_mask`, custom message
};

struct AnnealKeyInfo {
  const char* name;
  AnnealKeyType type;
  unsigned kinds;          // KIND_* bitmask
  const char* group;
  int rank;                // importance order within the group
  const char* doc;
  // Per-kind defaults (absent = presence-tested or run-time AUTO).
  bool has_def_serial, has_def_pt, has_def_pa;
  double def_serial, def_pt, def_pa;   // enum defaults = value index
  bool has_min, min_open, has_max, max_open;
  double min, max;
  const char* range_msg;   // "must be ..."; nullptr = no table range check
  bool report_value;       // append " (got <v>)" to the range message
  const char* const* enum_values;  // Enum type only, else nullptr
  std::size_t enum_count;
  // 1.5.0 value vocabulary: a numeric key may also accept the token
  // AUTO ("derive at run time") and/or OFF ("feature disabled"),
  // each resolving to the internal value below. Keys where 0 is a
  // literal value (V_FLOOR_FRAC, LADDER_MIN_GAP) declare neither.
  bool accepts_auto, accepts_off;
  double auto_value, off_value;
};

struct AnnealKeyRule {
  AnnealRuleType type;
  const char* key;
  const char* arg;         // flag / other key / enum key
  const char* arg2;        // RequiresEnum: required enum token
  unsigned kind_mask;      // KindReject only
  const char* msg;         // full message suffix after "TADAH:<ALGO>: "
};

struct AnnealKeyRename {
  const char* old_name;
  unsigned kinds;          // where the hint applies
  const char* msg;         // full message suffix after "TADAH:<ALGO>: "
};

inline constexpr const char* kBOUNDARY_MODE_values[] = {"REFLECT", "REDRAW"};
inline constexpr const char* kSTEP_MODE_values[] = {"FIXED", "FROZEN", "ADAPTIVE"};
inline constexpr const char* kINIT_MODE_values[] = {"JITTER", "UNIFORM"};

inline constexpr AnnealKeyInfo kKeys[] = {
  {"CORANA", AnnealKeyType::Bool, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 1, "Select the Corana per-dimension adaptive-step kernel instead of the whole-vector CLASSIC kernel.",
   true, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"STEP_FRAC", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 2, "Initial proposal scale per dimension as a fraction of the OPTIM interval width.",
   true, true, true, 0.1, 0.1, 0.1,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"CHAIN_LENGTH", AnnealKeyType::Count, KIND_SERIAL | KIND_PA,
   "steps", 3, "CLASSIC kernel: whole-vector proposals per temperature rung. Serial default AUTO: the MAXEVAL budget spread evenly over the cooling schedule (journaled).",
   false, false, true, 0.0, 0.0, 50.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   true, false, 0.0, 0.0},
  {"NS", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 4, "Corana kernel: step-variation cycles per step adjustment (Corana 1987 NS = 20). ANNEAL_PT: sweeps between per-slot step adjustments.",
   true, true, true, 20.0, 20.0, 20.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (Corana variant)", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"NT", AnnealKeyType::Count, KIND_SERIAL | KIND_PA,
   "steps", 5, "Corana kernel: step-adjustment cycles per temperature rung. Serial default AUTO: budget-derived, capped at the Corana 1987 value max(100, 5n).",
   false, false, true, 0.0, 0.0, 5.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (Corana variant)", false, nullptr, 0,
   true, false, 0.0, 0.0},
  {"CORANA_C", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 6, "Corana kernel: step-adjustment gain in the 0.4/0.6 acceptance-ratio rule (Corana 1987 c = 2).",
   true, true, true, 2.0, 2.0, 2.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0 (Corana step-adjust gain)", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"V_FLOOR_FRAC", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 7, "Corana kernel: lower clamp on the per-dimension step as a fraction of the OPTIM interval width.",
   true, true, true, 0.001, 0.001, 0.001,
   true, false, true, false, 0.0, 1.0,
   "must be in [0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"BOUNDARY_MODE", AnnealKeyType::Enum, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 8, "Out-of-box proposal handling: REFLECT = exact mirror fold; REDRAW = resample the offending coordinate.",
   true, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, kBOUNDARY_MODE_values, 2,
   false, false, 0.0, 0.0},
  {"RESET_TO_BEST", AnnealKeyType::Bool, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 9, "Snap the chain back to the best-so-far point after each cooling step (PT: coldest slot only, after each epoch).",
   false, false, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"T_INIT", AnnealKeyType::Float, KIND_SERIAL,
   "temperature", 1, "Starting temperature of the cooling schedule; when set, the T0 calibration probes are skipped.",
   false, false, false, 0.0, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"T_HOT", AnnealKeyType::Float, KIND_PT | KIND_PA,
   "temperature", 1, "Hottest ladder temperature (PT) / starting temperature (PA); when set, the T0 calibration probes are skipped.",
   false, false, false, 0.0, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"T_MIN", AnnealKeyType::Float, KIND_SERIAL | KIND_PA,
   "temperature", 2, "Stop temperature of the cooling schedule; unset derives T0 * 1e-6.",
   false, false, false, 0.0, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"RT", AnnealKeyType::Float, KIND_SERIAL | KIND_PA,
   "temperature", 3, "Geometric cooling ratio per temperature rung (Corana 1987 r_T = 0.85). ANNEAL_PT holds a static (optionally self-adapting) ladder and takes no RT.",
   true, false, true, 0.85, 0.0, 0.9,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"T0_PROBES", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 4, "Local probes used by the automatic T0 calibration.",
   true, true, true, 20.0, 20.0, 20.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"T0_ACCEPT", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 5, "Target initial uphill-acceptance probability: T0 = median(uphill dE) / ln(1/T0_ACCEPT).",
   true, true, true, 0.05, 0.05, 0.05,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1)", true, nullptr, 0,
   false, false, 0.0, 0.0},
  {"NEPS", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "stopping", 1, "Plateau ring size: consecutive rung-end values that must agree within FTOL_* to stop (Corana 1987 N_eps = 4).",
   true, true, true, 4.0, 4.0, 4.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"REANNEAL", AnnealKeyType::Bool, KIND_SERIAL,
   "temperature", 6, "When the schedule cools below T_MIN with budget left, restart from the best point at a raised temperature.",
   true, false, false, 1.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"REANNEAL_SCALE", AnnealKeyType::Float, KIND_SERIAL,
   "temperature", 7, "Start temperature of re-anneal cycle n, as a fraction: T_start = T0 * REANNEAL_SCALE^n.",
   true, false, false, 0.25, 0.0, 0.0,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1)", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"REPLICAS", AnnealKeyType::Count, KIND_PT,
   "ladder", 1, "Number of ladder slots K. AUTO resolves to max(THREADS, 2).",
   false, true, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   true, false, 0.0, 0.0},
  {"T_COLD", AnnealKeyType::Float, KIND_PT,
   "temperature", 8, "Coldest ladder slot temperature; unset derives T_HOT / LADDER_SPAN.",
   false, false, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_SPAN", AnnealKeyType::Float, KIND_PT,
   "ladder", 2, "Ratio T_HOT / T_COLD used when T_COLD is not set.",
   false, true, false, 0.0, 1000.0, 0.0,
   true, true, false, false, 1.0, 0.0,
   "must be > 1", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_POWER", AnnealKeyType::Float, KIND_PT,
   "ladder", 3, "Shape of the initial log-temperature ladder: position_k = (k/(K-1))^power; 1.0 = geometric (equal pair acceptance for scale-family energies).",
   false, true, false, 0.0, 1.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_ADAPT", AnnealKeyType::Bool, KIND_PT,
   "ladder", 4, "Adapt interior ladder temperatures toward equal pair swap acceptance (endpoints fixed; Kone-Kofke ~23% motivates equalization).",
   false, true, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_ADAPT_EVERY", AnnealKeyType::Count, KIND_PT,
   "ladder", 5, "Swap attempts per pair required before each ladder update.",
   false, true, false, 0.0, 20.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_ADAPT_GAIN", AnnealKeyType::Float, KIND_PT,
   "ladder", 6, "Initial stochastic-approximation gain of the ladder update.",
   false, true, false, 0.0, 1.0, 0.0,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_ADAPT_DECAY", AnnealKeyType::Float, KIND_PT,
   "ladder", 7, "Diminishing-gain exponent: effective gain = GAIN * update^(-DECAY) (valid stochastic-approximation range (0.5, 1]).",
   false, true, false, 0.0, 0.6, 0.0,
   true, true, true, false, 0.5, 1.0,
   "must be in (0.5,1]", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_MIN_GAP", AnnealKeyType::Float, KIND_PT,
   "ladder", 8, "Floor on every interior log-gap as a fraction of the mean gap; 0 disables the floor.",
   false, true, false, 0.0, 0.1, 0.0,
   true, false, true, true, 0.0, 1.0,
   "must be in [0,1) (0 disables the floor; 1 would leave no adaptable span)", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"LADDER_FREEZE", AnnealKeyType::Bool, KIND_PT,
   "ladder", 9, "Freeze ladder adaptation once per-pair acceptances are statistically indistinguishable from equalized.",
   false, true, false, 0.0, 1.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"SWAP_EVERY", AnnealKeyType::Count, KIND_PT,
   "swaps", 1, "Epochs between exchange rounds (1 = every epoch; frequent DEO swapping is optimal when swaps are eval-free, Syed et al. 2022).",
   false, true, false, 0.0, 1.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"SWAP_WINDOW", AnnealKeyType::Count, KIND_PT,
   "swaps", 2, "Epochs per swap-diagnostics window (windowed tables, step reports, epoch summary cadence); OFF = cumulative tables only.",
   false, true, false, 0.0, 20.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (or OFF for cumulative tables only)", false, nullptr, 0,
   false, true, 0.0, 0.0},
  {"STEP_MODE", AnnealKeyType::Enum, KIND_PT,
   "steps", 10, "CLASSIC step policy: FIXED = STEP_FRAC profile; FROZEN = calibrate per-slot scales, then freeze; ADAPTIVE = calibrate, then keep adapting (diagonal Robust Adaptive Metropolis).",
   false, true, false, 0.0, 1.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, kSTEP_MODE_values, 3,
   false, false, 0.0, 0.0},
  {"STEP_WARMUP_SWEEPS", AnnealKeyType::Count, KIND_PT,
   "steps", 11, "Minimum calibration attempts per coordinate during the step warm-up (the maximum is twice this).",
   false, true, false, 0.0, 12.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"STEP_TRUST_FACTOR", AnnealKeyType::Float, KIND_PT,
   "steps", 12, "Clamp adapted step scales to [baseline/f, baseline*f] around the calibrated baseline; OFF disables the clamp (STEP_MODE ADAPTIVE).",
   false, true, false, 0.0, 64.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, true, 0.0, 0.0},
  {"INIT_MODE", AnnealKeyType::Enum, KIND_PT | KIND_PA,
   "init", 1, "Replica initialisation. JITTER (default): slot 1 sits exactly at the seed and the others are displaced around it -- `JITTER` alone uses a Gaussian of sigma = STEP_FRAC*width, `JITTER <fraction>` a uniform draw on +/-fraction*width, `JITTER AUTO <p>` a uniform draw on +/-sigma_d with sigma_d MEASURED per dimension as the largest displacement holding the loss within p of the seed. UNIFORM: the others are drawn uniformly in the OPTIM box. Under ANNEAL_PT the displacement is additionally ladder-scaled -- see INIT_JITTER_LADDER.",
   false, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, kINIT_MODE_values, 2,
   false, false, 0.0, 0.0},
  {"INIT_JITTER_MAX", AnnealKeyType::Float, KIND_PT | KIND_PA,
   "init", 2, "INIT_MODE JITTER AUTO: cap on the measured sigma as a fraction of the OPTIM interval; also the screen probe point.",
   false, true, true, 0.0, 0.5, 0.5,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"INIT_JITTER_PROBES", AnnealKeyType::Count, KIND_PT | KIND_PA,
   "init", 3, "INIT_MODE JITTER AUTO: maximum objective probes per dimension, split between the two sides.",
   false, true, true, 0.0, 8.0, 8.0,
   true, false, false, false, 2.0, 0.0,
   "must be >= 2 (one screen probe per side)", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"INIT_JITTER_MAXEVAL", AnnealKeyType::Count, KIND_PT | KIND_PA,
   "init", 4, "INIT_MODE JITTER AUTO: hard cap on calibration probes. AUTO resolves to the protocol's own worst case (dimensions x INIT_JITTER_PROBES). Never charged against MAXEVAL.",
   false, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   true, false, 0.0, 0.0},
  {"INIT_JITTER_SAMPLE", AnnealKeyType::Count, KIND_PT | KIND_PA,
   "init", 6, "INIT_MODE JITTER AUTO: measure only this many dimensions, chosen round-robin across parameter families; the rest inherit their family's median sigma/width and are flagged `inferred`. 0 = measure every dimension.",
   false, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"INIT_JITTER_AUDIT", AnnealKeyType::Count, KIND_PT | KIND_PA,
   "init", 7, "INIT_MODE JITTER AUTO: re-probe this many measured dimensions at sigma/2 to detect a loss that is non-monotone in displacement (sigma is a grid certificate, not a proof). 0 = skip the audit.",
   false, true, true, 0.0, 32.0, 32.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"INIT_JITTER_LADDER", AnnealKeyType::Float, KIND_PT,
   "init", 8, "Scale each slot's initial jitter by (T_k/T_HOT)^gamma, so cold slots start near the seed and hot ones explore. AUTO = gamma 0.5, the equipartition law (the equilibrium coordinate spread about a locally quadratic minimum goes as sqrt(T)), floored at the resolvable coordinate scale so cold slots cannot collapse onto the seed. OFF = every slot uses the same sigma (pre-1.5.0 behaviour). A number in (0,4] sets gamma directly. Applies to every INIT_MODE JITTER variant, not only AUTO.",
   false, true, false, 0.0, -1.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   true, true, -1.0, 0.0},
  {"INIT_JITTER_MAXTIME_FRAC", AnnealKeyType::Float, KIND_PT | KIND_PA,
   "init", 5, "INIT_MODE JITTER AUTO: share of OPTIMIZER MAXTIME the calibration may consume. MAXEVAL is not charged, so time is the axis that still bounds the phase.",
   false, true, true, 0.0, 0.25, 0.25,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"SWEEPS", AnnealKeyType::Count, KIND_PA,
   "steps", 14, "Rungs per walker per temperature step.",
   false, false, true, 0.0, 0.0, 1.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   false, false, 0.0, 0.0},
  {"ESS_TARGET", AnnealKeyType::Float, KIND_PA,
   "temperature", 9, "Adaptive temperature-step target for the effective sample size; OFF selects the geometric RT schedule.",
   false, false, true, 0.0, 0.0, 0.5,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1) (or OFF for the geometric RT schedule)", false, nullptr, 0,
   false, true, 0.0, 0.0},
};

inline constexpr AnnealKeyRule kRules[] = {
  {AnnealRuleType::RequiresFlag, "NS", "CORANA", "", 0u, "PARAM NS requires PARAM CORANA ON"},
  {AnnealRuleType::RequiresFlag, "NT", "CORANA", "", 0u, "PARAM NT requires PARAM CORANA ON"},
  {AnnealRuleType::RequiresFlag, "CORANA_C", "CORANA", "", 0u, "PARAM CORANA_C requires PARAM CORANA ON"},
  {AnnealRuleType::RequiresFlag, "V_FLOOR_FRAC", "CORANA", "", 0u, "PARAM V_FLOOR_FRAC requires PARAM CORANA ON"},
  {AnnealRuleType::ConflictsFlag, "CHAIN_LENGTH", "CORANA", "", 0u, "PARAM CHAIN_LENGTH belongs to the CLASSIC variant"},
  {AnnealRuleType::KindReject, "RESET_TO_BEST", "", "", KIND_PA, "PARAM RESET_TO_BEST is not defined for population annealing"},
  {AnnealRuleType::UnusedWhenSet, "T0_PROBES", "T_INIT", "", 0u, "PARAM T0_PROBES is unused when T_INIT is set"},
  {AnnealRuleType::UnusedWhenSet, "T0_ACCEPT", "T_INIT", "", 0u, "PARAM T0_ACCEPT is unused when T_INIT is set"},
  {AnnealRuleType::UnusedWhenSet, "T0_PROBES", "T_HOT", "", 0u, "PARAM T0_PROBES is unused when T_HOT is set"},
  {AnnealRuleType::UnusedWhenSet, "T0_ACCEPT", "T_HOT", "", 0u, "PARAM T0_ACCEPT is unused when T_HOT is set"},
  {AnnealRuleType::UnusedWhenFlagOff, "REANNEAL_SCALE", "REANNEAL", "", 0u, "PARAM REANNEAL_SCALE is unused when REANNEAL is OFF"},
  {AnnealRuleType::UnusedWhenSet, "LADDER_SPAN", "T_COLD", "", 0u, "PARAM LADDER_SPAN is unused when T_COLD is set"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_ADAPT_EVERY", "LADDER_ADAPT", "", 0u, "PARAM LADDER_ADAPT_EVERY is unused when LADDER_ADAPT is OFF"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_ADAPT_GAIN", "LADDER_ADAPT", "", 0u, "PARAM LADDER_ADAPT_GAIN is unused when LADDER_ADAPT is OFF"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_ADAPT_DECAY", "LADDER_ADAPT", "", 0u, "PARAM LADDER_ADAPT_DECAY is unused when LADDER_ADAPT is OFF"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_MIN_GAP", "LADDER_ADAPT", "", 0u, "PARAM LADDER_MIN_GAP is unused when LADDER_ADAPT is OFF"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_FREEZE", "LADDER_ADAPT", "", 0u, "PARAM LADDER_FREEZE is unused when LADDER_ADAPT is OFF"},
  {AnnealRuleType::ConflictsFlag, "STEP_MODE", "CORANA", "", 0u, "PARAM STEP_MODE belongs to the CLASSIC kernel (CORANA adapts its own per-dimension steps)"},
  {AnnealRuleType::ConflictsFlag, "STEP_WARMUP_SWEEPS", "CORANA", "", 0u, "PARAM STEP_WARMUP_SWEEPS belongs to the CLASSIC kernel (CORANA adapts its own per-dimension steps)"},
  {AnnealRuleType::ConflictsFlag, "STEP_TRUST_FACTOR", "CORANA", "", 0u, "PARAM STEP_TRUST_FACTOR belongs to the CLASSIC kernel (CORANA adapts its own per-dimension steps)"},
  {AnnealRuleType::RequiresEnum, "STEP_TRUST_FACTOR", "STEP_MODE", "ADAPTIVE", 0u, "PARAM STEP_TRUST_FACTOR is unused unless STEP_MODE is ADAPTIVE"},
};

inline constexpr AnnealKeyRename kRenamed[] = {
  {"T_INIT", KIND_PT | KIND_PA, "PARAM T_INIT is the serial TADAH:ANNEAL starting temperature; use T_HOT here (the hottest ladder/start temperature)"},
  {"T_HOT", KIND_SERIAL, "PARAM T_HOT is the ANNEAL_PT/ANNEAL_PA hot-end temperature; the serial starting temperature is PARAM T_INIT"},
  {"L", KIND_SERIAL | KIND_PT | KIND_PA, "PARAM L was renamed to CHAIN_LENGTH (same meaning: CLASSIC proposals per temperature rung); update the config"},
  {"C", KIND_SERIAL | KIND_PT | KIND_PA, "PARAM C was renamed to CORANA_C (same meaning: the Corana step-adjust gain); update the config"},
  {"T0_PACCEPT", KIND_SERIAL | KIND_PT | KIND_PA, "PARAM T0_PACCEPT was renamed to T0_ACCEPT (same meaning: target initial acceptance probability); update the config"},
  {"STEP_AUTO", KIND_PT, "PARAM STEP_AUTO was replaced by PARAM STEP_MODE (FIXED / FROZEN / ADAPTIVE); STEP_AUTO 1 corresponds to STEP_MODE FROZEN, STEP_AUTO 0 to STEP_MODE FIXED"},
  {"STEP_AUTO_SWEEPS", KIND_PT, "PARAM STEP_AUTO_SWEEPS was renamed to STEP_WARMUP_SWEEPS (same meaning: minimum calibration attempts per coordinate); update the config"},
  {"STEP_ADAPT_EVERY", KIND_PT, "PARAM STEP_ADAPT_EVERY was removed: STEP_MODE ADAPTIVE applies the robust-adaptive-Metropolis update after every proposal (a batched cadence multiplied the feedback gain by the batch size)"},
  {"LADDER_ADAPT_UPDATES", KIND_PT, "PARAM LADDER_ADAPT_UPDATES was removed; the pooled-evidence equalization freeze (LADDER_FREEZE) governs the adaptation lifetime"},
  {"RT", KIND_PT, "PARAM RT is not used by ANNEAL_PT: parallel tempering runs a static (optionally self-adapting) temperature ladder; cooling schedules belong to the serial TADAH:ANNEAL"},
  {"T_MIN", KIND_PT, "PARAM T_MIN is not used by ANNEAL_PT: parallel tempering runs a static (optionally self-adapting) temperature ladder; cooling schedules belong to the serial TADAH:ANNEAL"},
  {"NT", KIND_PT, "PARAM NT is not used by ANNEAL_PT: slot temperatures are fixed, so there is no per-temperature step-adjust cycle count; the per-slot Corana step adjusts every NS sweeps"},
  {"CHAIN_LENGTH", KIND_PT, "PARAM CHAIN_LENGTH is not used by ANNEAL_PT: a replica runs one sweep per epoch so exchange stays frequent; the diagnostics cadence is SWAP_WINDOW"},
};

inline constexpr const AnnealKeyInfo* find_key(std::string_view name)
{
  for (const auto& k : kKeys)
    if (name == k.name) return &k;
  return nullptr;
}

}  // namespace anneal_registry

/* Default values, one constexpr per key kind-variant with a static
 * default (enum defaults are value indices).  The anneal::Knobs
 * member initialisers consume these; nothing else may restate a
 * default. */
namespace anneal_defaults {
inline constexpr double CORANA = 0.0;
inline constexpr double STEP_FRAC = 0.1;
inline constexpr double CHAIN_LENGTH_PA = 50.0;
inline constexpr double NS = 20.0;
inline constexpr double NT_PA = 5.0;
inline constexpr double CORANA_C = 2.0;
inline constexpr double V_FLOOR_FRAC = 0.001;
inline constexpr double BOUNDARY_MODE = 0.0;
inline constexpr double RT_SERIAL = 0.85;
inline constexpr double RT_PA = 0.9;
inline constexpr double T0_PROBES = 20.0;
inline constexpr double T0_ACCEPT = 0.05;
inline constexpr double NEPS = 4.0;
inline constexpr double REANNEAL = 1.0;
inline constexpr double REANNEAL_SCALE = 0.25;
inline constexpr double REPLICAS = 0.0;
inline constexpr double LADDER_SPAN = 1000.0;
inline constexpr double LADDER_POWER = 1.0;
inline constexpr double LADDER_ADAPT = 0.0;
inline constexpr double LADDER_ADAPT_EVERY = 20.0;
inline constexpr double LADDER_ADAPT_GAIN = 1.0;
inline constexpr double LADDER_ADAPT_DECAY = 0.6;
inline constexpr double LADDER_MIN_GAP = 0.1;
inline constexpr double LADDER_FREEZE = 1.0;
inline constexpr double SWAP_EVERY = 1.0;
inline constexpr double SWAP_WINDOW = 20.0;
inline constexpr double STEP_MODE = 1.0;
inline constexpr double STEP_WARMUP_SWEEPS = 12.0;
inline constexpr double STEP_TRUST_FACTOR = 64.0;
inline constexpr double INIT_MODE = 0.0;
inline constexpr double INIT_JITTER_MAX = 0.5;
inline constexpr double INIT_JITTER_PROBES = 8.0;
inline constexpr double INIT_JITTER_MAXEVAL = 0.0;
inline constexpr double INIT_JITTER_SAMPLE = 0.0;
inline constexpr double INIT_JITTER_AUDIT = 32.0;
inline constexpr double INIT_JITTER_LADDER = -1.0;
inline constexpr double INIT_JITTER_MAXTIME_FRAC = 0.25;
inline constexpr double SWEEPS = 1.0;
inline constexpr double ESS_TARGET = 0.5;
}  // namespace anneal_defaults

/* Named enum value indices for readable driver code, e.g.
 *   kb.step_mode == anneal_enums::STEP_MODE::FROZEN */
namespace anneal_enums {
namespace BOUNDARY_MODE {
inline constexpr int REFLECT = 0;
inline constexpr int REDRAW = 1;
}
namespace STEP_MODE {
inline constexpr int FIXED = 0;
inline constexpr int FROZEN = 1;
inline constexpr int ADAPTIVE = 2;
}
namespace INIT_MODE {
inline constexpr int JITTER = 0;
inline constexpr int UNIFORM = 1;
}
}  // namespace anneal_enums

}  // namespace hpo
}  // namespace tadah
