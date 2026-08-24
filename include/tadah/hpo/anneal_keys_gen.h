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

enum class AnnealKeyType : unsigned char { Bool, Count, Float };

enum class AnnealRuleType : unsigned char {
  RequiresFlag,      // key valid only when `flag` parses to 1
  ConflictsFlag,     // key invalid when `flag` parses to 1
  UnusedWhenSet,     // key invalid when `other` is explicitly set
  UnusedWhenFlagOff, // key invalid when `flag` parses to 0
  KindReject,        // key invalid for `kind`, custom message
};

struct AnnealKeyInfo {
  const char* name;
  AnnealKeyType type;
  unsigned kinds;          // KIND_* bitmask
  const char* group;
  int rank;                // importance order within the group
  const char* doc;
  bool has_default;        // false = presence-tested optional key
  double def;              // base default (bool: 0/1)
  bool has_def_pt, has_def_pa;
  double def_pt, def_pa;   // per-kind default overrides
  bool has_min, min_open, has_max, max_open;
  double min, max;
  const char* range_msg;   // "must be ..."; nullptr = no table range check
  bool report_value;       // append " (got <v>)" to the range message
};

struct AnnealKeyRule {
  AnnealRuleType type;
  const char* key;
  const char* arg;         // flag / other key; kind name for KindReject
  unsigned kind_mask;      // KindReject only
  const char* msg;         // full message suffix after "TADAH:<ALGO>: "
};

inline constexpr AnnealKeyInfo kKeys[] = {
  {"CORANA", AnnealKeyType::Bool, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 1, "Select the Corana per-dimension adaptive-step kernel instead of the whole-vector CLASSIC kernel.",
   true, 0.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"STEP_FRAC", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 2, "Initial proposal scale per dimension as a fraction of the OPTIM interval width.",
   true, 0.1, false, false, 0.0, 0.0,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false},
  {"L", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 3, "CLASSIC kernel: whole-vector proposals per temperature rung.",
   true, 50.0, false, false, 0.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false},
  {"NS", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 4, "Corana kernel: step-variation cycles per step-adjustment cycle.",
   true, 5.0, false, false, 0.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (Corana variant)", false},
  {"NT", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 5, "Corana kernel: step-adjustment cycles per temperature rung.",
   true, 5.0, false, false, 0.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (Corana variant)", false},
  {"C", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 6, "Corana kernel: step-adjustment gain in the 0.4/0.6 acceptance-ratio rule.",
   true, 2.0, false, false, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0 (Corana step-adjust gain)", false},
  {"V_FLOOR_FRAC", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 7, "Corana kernel: lower clamp on the per-dimension step as a fraction of the OPTIM interval width.",
   true, 0.001, false, false, 0.0, 0.0,
   true, false, true, false, 0.0, 1.0,
   "must be in [0,1]", false},
  {"BOUNDARY_MODE", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 8, "Out-of-box proposal handling: 0 = REFLECT (exact mirror fold), 1 = REDRAW the offending coordinate.",
   true, 0.0, false, false, 0.0, 0.0,
   false, false, true, false, 0.0, 1.0,
   "must be 0 (REFLECT) or 1 (REDRAW)", false},
  {"RESET_TO_BEST", AnnealKeyType::Bool, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 9, "Snap the chain back to the best-so-far point after each cooling step (PT: coldest slot only, after each epoch).",
   false, 0.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"T_HOT", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 1, "Starting (hottest) temperature; when set, the T0 calibration probes are skipped.",
   false, 0.0, false, false, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false},
  {"T_MIN", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 2, "Stop temperature; unset derives T0 * 1e-6. PT: only meaningful with RT < 1 (hot-end stop).",
   false, 0.0, false, false, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false},
  {"RT", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 3, "Geometric cooling ratio per rung (PT: applied to every ladder slot per epoch; 1.0 = static ladder).",
   true, 0.9, true, false, 1.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"T0_PROBES", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 4, "Local probes used by the automatic T0 calibration.",
   true, 20.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"T0_PACCEPT", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 5, "Target initial uphill-acceptance probability: T0 = median(uphill dE) / ln(1/T0_PACCEPT).",
   true, 0.05, false, false, 0.0, 0.0,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1)", true},
  {"NEPS", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "stopping", 1, "Plateau ring size: consecutive rung-end values that must agree within FTOL_* to stop.",
   true, 4.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"REANNEAL", AnnealKeyType::Bool, KIND_SERIAL,
   "temperature", 6, "When the schedule cools below T_MIN with budget left, restart from the best point at a raised temperature.",
   true, 1.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"REANNEAL_SCALE", AnnealKeyType::Float, KIND_SERIAL,
   "temperature", 7, "Start temperature of re-anneal cycle n, as a fraction: T_start = T0 * REANNEAL_SCALE^n.",
   true, 0.25, false, false, 0.0, 0.0,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1)", false},
  {"REPLICAS", AnnealKeyType::Count, KIND_PT,
   "ladder", 1, "Number of ladder slots K; 0 resolves to max(THREADS, 2).",
   true, 0.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"T_COLD", AnnealKeyType::Float, KIND_PT,
   "temperature", 8, "Coldest ladder slot temperature; unset derives T_HOT / LADDER_SPAN.",
   false, 0.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"LADDER_SPAN", AnnealKeyType::Float, KIND_PT,
   "ladder", 2, "Ratio T_HOT / T_COLD used when T_COLD is not set.",
   true, 1000.0, false, false, 0.0, 0.0,
   true, true, false, false, 1.0, 0.0,
   "must be > 1", false},
  {"LADDER_POWER", AnnealKeyType::Float, KIND_PT,
   "ladder", 3, "Shape of the initial log-temperature ladder: position_k = (k/(K-1))^power; 1.0 = geometric.",
   true, 1.0, false, false, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false},
  {"LADDER_ADAPT", AnnealKeyType::Bool, KIND_PT,
   "ladder", 4, "Adapt interior ladder temperatures toward equal pair swap acceptance (endpoints fixed).",
   true, 0.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"LADDER_ADAPT_EVERY", AnnealKeyType::Count, KIND_PT,
   "ladder", 5, "Swap attempts per pair required before each ladder update.",
   true, 20.0, false, false, 0.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false},
  {"LADDER_ADAPT_GAIN", AnnealKeyType::Float, KIND_PT,
   "ladder", 6, "Initial stochastic-approximation gain of the ladder update.",
   true, 1.0, false, false, 0.0, 0.0,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false},
  {"LADDER_ADAPT_DECAY", AnnealKeyType::Float, KIND_PT,
   "ladder", 7, "Diminishing-gain exponent: effective gain = GAIN * update^(-DECAY).",
   true, 0.6, false, false, 0.0, 0.0,
   true, true, true, false, 0.5, 1.0,
   "must be in (0.5,1]", false},
  {"LADDER_ADAPT_UPDATES", AnnealKeyType::Count, KIND_PT,
   "ladder", 8, "Hard cap on ladder updates; 0 = no cap (the equalization freeze may still stop adaptation).",
   true, 0.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"LADDER_MIN_GAP", AnnealKeyType::Float, KIND_PT,
   "ladder", 9, "Floor on every interior log-gap as a fraction of the mean gap; 0 disables the floor.",
   true, 0.1, false, false, 0.0, 0.0,
   true, false, true, true, 0.0, 1.0,
   "must be in [0,1) (0 disables the floor; 1 would leave no adaptable span)", false},
  {"LADDER_FREEZE", AnnealKeyType::Bool, KIND_PT,
   "ladder", 10, "Freeze ladder adaptation once per-pair acceptances are statistically indistinguishable from equalized.",
   true, 1.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"SWAP_EVERY", AnnealKeyType::Count, KIND_PT,
   "swaps", 1, "Epochs between exchange rounds.",
   true, 1.0, false, false, 0.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false},
  {"SWAP_WINDOW", AnnealKeyType::Count, KIND_PT,
   "swaps", 2, "Epochs per windowed swap-diagnostics report; 0 = cumulative tables only.",
   true, 20.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"STEP_AUTO", AnnealKeyType::Bool, KIND_PT,
   "steps", 10, "CLASSIC kernel: two-phase automatic step calibration (per-coordinate warm-up, then full-vector pilot).",
   true, 1.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"STEP_AUTO_SWEEPS", AnnealKeyType::Count, KIND_PT,
   "steps", 11, "Minimum calibration attempts per coordinate (the maximum is twice this).",
   true, 12.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"STEP_ADAPT_EVERY", AnnealKeyType::Count, KIND_PT,
   "steps", 12, "Proposals between adaptive step updates after warm-up; 0 = freeze the calibrated step.",
   true, 0.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"STEP_TRUST_FACTOR", AnnealKeyType::Float, KIND_PT,
   "steps", 13, "Clamp adapted step scales to [baseline/f, baseline*f] around the calibrated baseline; 0 = off.",
   true, 64.0, false, false, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false},
  {"INIT_MODE", AnnealKeyType::Count, KIND_PT | KIND_PA,
   "init", 1, "Replica initialisation: 0 = seed point (+ jitter where applicable), 1 = uniform draw in the OPTIM box.",
   true, 0.0, false, false, 0.0, 0.0,
   false, false, true, false, 0.0, 1.0,
   "must be 0 (seed-jitter) or 1 (uniform box)", false},
  {"SWEEPS", AnnealKeyType::Count, KIND_PA,
   "steps", 14, "Rungs per walker per temperature step.",
   true, 1.0, false, false, 0.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false},
  {"ESS_TARGET", AnnealKeyType::Float, KIND_PA,
   "temperature", 9, "Adaptive temperature-step target for the effective sample size; 0 = geometric RT schedule.",
   true, 0.5, false, false, 0.0, 0.0,
   true, false, true, true, 0.0, 1.0,
   "must be in [0,1) (0 = geometric RT schedule)", false},
};

inline constexpr AnnealKeyRule kRules[] = {
  {AnnealRuleType::RequiresFlag, "NS", "CORANA", 0u, "PARAM NS requires PARAM CORANA 1"},
  {AnnealRuleType::RequiresFlag, "NT", "CORANA", 0u, "PARAM NT requires PARAM CORANA 1"},
  {AnnealRuleType::RequiresFlag, "C", "CORANA", 0u, "PARAM C requires PARAM CORANA 1"},
  {AnnealRuleType::RequiresFlag, "V_FLOOR_FRAC", "CORANA", 0u, "PARAM V_FLOOR_FRAC requires PARAM CORANA 1"},
  {AnnealRuleType::ConflictsFlag, "L", "CORANA", 0u, "PARAM L belongs to the CLASSIC variant"},
  {AnnealRuleType::KindReject, "RESET_TO_BEST", "pa", KIND_PA, "PARAM RESET_TO_BEST is not defined for population annealing"},
  {AnnealRuleType::UnusedWhenSet, "T0_PROBES", "T_HOT", 0u, "PARAM T0_PROBES is unused when T_HOT is set"},
  {AnnealRuleType::UnusedWhenSet, "T0_PACCEPT", "T_HOT", 0u, "PARAM T0_PACCEPT is unused when T_HOT is set"},
  {AnnealRuleType::UnusedWhenFlagOff, "REANNEAL_SCALE", "REANNEAL", 0u, "PARAM REANNEAL_SCALE is unused when REANNEAL is 0"},
  {AnnealRuleType::UnusedWhenSet, "LADDER_SPAN", "T_COLD", 0u, "PARAM LADDER_SPAN is unused when T_COLD is set"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_ADAPT_EVERY", "LADDER_ADAPT", 0u, "PARAM LADDER_ADAPT_EVERY is unused when LADDER_ADAPT is 0"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_ADAPT_GAIN", "LADDER_ADAPT", 0u, "PARAM LADDER_ADAPT_GAIN is unused when LADDER_ADAPT is 0"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_ADAPT_DECAY", "LADDER_ADAPT", 0u, "PARAM LADDER_ADAPT_DECAY is unused when LADDER_ADAPT is 0"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_ADAPT_UPDATES", "LADDER_ADAPT", 0u, "PARAM LADDER_ADAPT_UPDATES is unused when LADDER_ADAPT is 0"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_MIN_GAP", "LADDER_ADAPT", 0u, "PARAM LADDER_MIN_GAP is unused when LADDER_ADAPT is 0"},
  {AnnealRuleType::UnusedWhenFlagOff, "LADDER_FREEZE", "LADDER_ADAPT", 0u, "PARAM LADDER_FREEZE is unused when LADDER_ADAPT is 0"},
};

inline constexpr const AnnealKeyInfo* find_key(std::string_view name)
{
  for (const auto& k : kKeys)
    if (name == k.name) return &k;
  return nullptr;
}

}  // namespace anneal_registry

/* Default values, one constexpr per key with a static default.  The
 * anneal::Knobs member initialisers consume these; nothing else may
 * restate a default. */
namespace anneal_defaults {
inline constexpr double CORANA = 0.0;
inline constexpr double STEP_FRAC = 0.1;
inline constexpr double L = 50.0;
inline constexpr double NS = 5.0;
inline constexpr double NT = 5.0;
inline constexpr double C = 2.0;
inline constexpr double V_FLOOR_FRAC = 0.001;
inline constexpr double BOUNDARY_MODE = 0.0;
inline constexpr double RT = 0.9;
inline constexpr double T0_PROBES = 20.0;
inline constexpr double T0_PACCEPT = 0.05;
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
inline constexpr double LADDER_ADAPT_UPDATES = 0.0;
inline constexpr double LADDER_MIN_GAP = 0.1;
inline constexpr double LADDER_FREEZE = 1.0;
inline constexpr double SWAP_EVERY = 1.0;
inline constexpr double SWAP_WINDOW = 20.0;
inline constexpr double STEP_AUTO = 1.0;
inline constexpr double STEP_AUTO_SWEEPS = 12.0;
inline constexpr double STEP_ADAPT_EVERY = 0.0;
inline constexpr double STEP_TRUST_FACTOR = 64.0;
inline constexpr double INIT_MODE = 0.0;
inline constexpr double SWEEPS = 1.0;
inline constexpr double ESS_TARGET = 0.5;
inline constexpr double RT_PT = 1.0;
}  // namespace anneal_defaults

}  // namespace hpo
}  // namespace tadah
