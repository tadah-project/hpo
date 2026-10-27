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
  const char* long_doc;     // detail after doc; nullptr = none
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
  unsigned auto_kinds;     // KIND_* bitmask of the kinds that accept AUTO
  // The default in words of each kind (default_doc): a rule rather
  // than a value, or nullptr when the kind has a value default.
  const char* def_doc_serial;
  const char* def_doc_pt;
  const char* def_doc_pa;
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
   "steps", 1, "Selects the proposal kernel: ON takes Corana's adaptive steps, one coordinate at a time with a step for each dimension; OFF takes the CLASSIC kernel, which moves every coordinate at once.",
   nullptr,
   true, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"STEP_FRAC", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 2, "The first proposal scale of each dimension, as a fraction of the width of its OPTIM box.",
   nullptr,
   true, true, true, 0.1, 0.1, 0.1,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"CHAIN_LENGTH", AnnealKeyType::Count, KIND_SERIAL | KIND_PA,
   "steps", 3, "The number of proposals of the CLASSIC kernel at each temperature step.",
   "The default of TADAH:ANNEAL is AUTO: MAXEVAL spread evenly over the cooling schedule, which hpo_run.log reports.",
   false, false, true, 0.0, 0.0, 50.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   true, false, 0.0, 0.0,
   KIND_SERIAL,
   "AUTO, from MAXEVAL", nullptr, nullptr},
  {"NS", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 4, "Corana kernel: the number of cycles through the dimensions between two adjustments of the steps. Under TADAH:ANNEAL_PT, the number of sweeps between two adjustments of the steps of a slot.",
   nullptr,
   true, true, true, 20.0, 20.0, 20.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (Corana variant)", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"NT", AnnealKeyType::Count, KIND_SERIAL | KIND_PA,
   "steps", 5, "Corana kernel: the number of step adjustments at each temperature step.",
   "The default of TADAH:ANNEAL is AUTO: a number that follows from MAXEVAL, at most the larger of 100 and 5n for n dimensions.",
   false, false, true, 0.0, 0.0, 5.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (Corana variant)", false, nullptr, 0,
   true, false, 0.0, 0.0,
   KIND_SERIAL,
   "AUTO, from MAXEVAL", nullptr, nullptr},
  {"CORANA_C", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 6, "Corana kernel: the gain of the step adjustment, which lengthens the step of a dimension whose acceptance ratio is above 0.6 and shortens one below 0.4.",
   nullptr,
   true, true, true, 2.0, 2.0, 2.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0 (Corana step-adjust gain)", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"V_FLOOR_FRAC", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 7, "Corana kernel: the smallest step of each dimension, as a fraction of the width of its OPTIM box.",
   nullptr,
   true, true, true, 0.001, 0.001, 0.001,
   true, false, true, false, 0.0, 1.0,
   "must be in [0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"BOUNDARY_MODE", AnnealKeyType::Enum, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 8, "What a proposal outside the OPTIM box becomes: REFLECT folds it back into the box as a mirror does; REDRAW draws each coordinate outside the box again.",
   nullptr,
   true, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, kBOUNDARY_MODE_values, 2,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"RESET_TO_BEST", AnnealKeyType::Bool, KIND_SERIAL | KIND_PT | KIND_PA,
   "steps", 9, "ON moves the chain back to the best point so far after each cooling step; under TADAH:ANNEAL_PT it moves the coldest slot after each epoch.",
   nullptr,
   false, true, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   "OFF; ON with CORANA ON", nullptr, nullptr},
  {"T_INIT", AnnealKeyType::Float, KIND_SERIAL,
   "temperature", 1, "The starting temperature of the cooling schedule. Without it, the search calibrates the starting temperature with probes (T0_PROBES, T0_ACCEPT).",
   nullptr,
   false, false, false, 0.0, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   "calibrated with probes", nullptr, nullptr},
  {"T_HOT", AnnealKeyType::Float, KIND_PT | KIND_PA,
   "temperature", 1, "The hottest temperature of the ladder of TADAH:ANNEAL_PT, and the starting temperature of TADAH:ANNEAL_PA. Without it, the search calibrates it with probes (T0_PROBES, T0_ACCEPT).",
   nullptr,
   false, false, false, 0.0, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, "calibrated with probes", "calibrated with probes"},
  {"T_MIN", AnnealKeyType::Float, KIND_SERIAL | KIND_PA,
   "temperature", 2, "The temperature at which the cooling schedule ends; without it, 1e-6 times the starting temperature.",
   nullptr,
   false, false, false, 0.0, 0.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   "1e-6 times the starting temperature", nullptr, "1e-6 times the starting temperature"},
  {"RT", AnnealKeyType::Float, KIND_SERIAL | KIND_PA,
   "temperature", 3, "The ratio of two successive temperatures of the geometric cooling schedule, above 0 and below 1.",
   nullptr,
   true, false, true, 0.85, 0.0, 0.9,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"T0_PROBES", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 4, "The number of probes near the starting point with which the search calibrates its starting temperature.",
   nullptr,
   true, true, true, 20.0, 20.0, 20.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"T0_ACCEPT", AnnealKeyType::Float, KIND_SERIAL | KIND_PT | KIND_PA,
   "temperature", 5, "The probability with which the calibrated starting temperature accepts a step up in loss: the temperature is the median rise of the loss over the probes divided by ln(1/T0_ACCEPT).",
   nullptr,
   true, true, true, 0.05, 0.05, 0.05,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1)", true, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"NEPS", AnnealKeyType::Count, KIND_SERIAL | KIND_PT | KIND_PA,
   "stopping", 1, "The length of the history of the FTOL_REL and FTOL_ABS stop, which has an effect only with one of them.",
   "TADAH:ANNEAL compares the loss at the end of a temperature step with the losses at the end of this many earlier steps and with the best loss; TADAH:ANNEAL_PA applies the same rule to the lowest loss of the population after each temperature step; TADAH:ANNEAL_PT counts the successive diagnostics windows without a significant improvement of the best loss.",
   true, true, true, 4.0, 4.0, 4.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"REANNEAL", AnnealKeyType::Bool, KIND_SERIAL,
   "temperature", 6, "ON restarts from the best point at a higher temperature when the schedule cools below T_MIN with evaluations left.",
   nullptr,
   true, false, false, 1.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"REANNEAL_SCALE", AnnealKeyType::Float, KIND_SERIAL,
   "temperature", 7, "The starting temperature of the n-th reannealing, as a fraction: the starting temperature times REANNEAL_SCALE to the power n.",
   nullptr,
   true, false, false, 0.25, 0.0, 0.0,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1)", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"REPLICAS", AnnealKeyType::Count, KIND_PT,
   "ladder", 1, "The number of replicas, one for each temperature of the ladder; AUTO takes the larger of THREADS and 2.",
   nullptr,
   false, true, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   true, false, 0.0, 0.0,
   KIND_PT,
   nullptr, "AUTO, the larger of THREADS and 2", nullptr},
  {"T_COLD", AnnealKeyType::Float, KIND_PT,
   "temperature", 8, "The coldest temperature of the ladder; without it, T_HOT divided by LADDER_SPAN.",
   nullptr,
   false, false, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, "T_HOT divided by LADDER_SPAN", nullptr},
  {"LADDER_SPAN", AnnealKeyType::Float, KIND_PT,
   "ladder", 2, "The ratio of T_HOT to T_COLD from which an omitted T_COLD follows.",
   nullptr,
   false, true, false, 0.0, 1000.0, 0.0,
   true, true, false, false, 1.0, 0.0,
   "must be > 1", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"LADDER_POWER", AnnealKeyType::Float, KIND_PT,
   "ladder", 3, "The shape of the first ladder: of K slots, slot k lies at the fraction (k/(K-1))^LADDER_POWER of the span of the logarithm of the temperature. 1 gives a geometric ladder.",
   nullptr,
   false, true, false, 0.0, 1.0, 0.0,
   true, true, false, false, 0.0, 0.0,
   "must be > 0", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"LADDER_ADAPT", AnnealKeyType::Bool, KIND_PT,
   "ladder", 4, "ON adapts the inner temperatures of the ladder towards the same swap acceptance for every pair of neighbours; the hottest and the coldest temperature stay. It needs 3 replicas or more.",
   nullptr,
   false, true, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"LADDER_ADAPT_EVERY", AnnealKeyType::Count, KIND_PT,
   "ladder", 5, "The number of swap attempts of each pair before each update of the ladder.",
   nullptr,
   false, true, false, 0.0, 20.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"LADDER_ADAPT_GAIN", AnnealKeyType::Float, KIND_PT,
   "ladder", 6, "The first gain of the update of the ladder, a stochastic approximation.",
   nullptr,
   false, true, false, 0.0, 1.0, 0.0,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"LADDER_ADAPT_DECAY", AnnealKeyType::Float, KIND_PT,
   "ladder", 7, "The exponent with which the gain of the ladder update falls: the gain of update u is LADDER_ADAPT_GAIN times u to the power -LADDER_ADAPT_DECAY.",
   nullptr,
   false, true, false, 0.0, 0.6, 0.0,
   true, true, true, false, 0.5, 1.0,
   "must be in (0.5,1]", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"LADDER_MIN_GAP", AnnealKeyType::Float, KIND_PT,
   "ladder", 8, "The smallest gap between neighbouring temperatures of the ladder, in their logarithm, as a fraction of the mean gap; 0 sets no smallest gap.",
   nullptr,
   false, true, false, 0.0, 0.1, 0.0,
   true, false, true, true, 0.0, 1.0,
   "must be in [0,1) (0 disables the floor; 1 would leave no adaptable span)", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"LADDER_FREEZE", AnnealKeyType::Bool, KIND_PT,
   "ladder", 9, "ON stops the adaptation of the ladder once the swap acceptances of the pairs cannot be told apart statistically from equal ones.",
   nullptr,
   false, true, false, 0.0, 1.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"SWAP_EVERY", AnnealKeyType::Count, KIND_PT,
   "swaps", 1, "The number of epochs between two rounds of swap attempts; 1 attempts swaps in every epoch.",
   nullptr,
   false, true, false, 0.0, 1.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"SWAP_WINDOW", AnnealKeyType::Count, KIND_PT,
   "swaps", 2, "The number of epochs of a window of the swap diagnostics in hpo_run.log: the tables of a window, the reports of the steps and the summaries of the epochs. OFF writes cumulative tables only.",
   nullptr,
   false, true, false, 0.0, 20.0, 0.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1 (or OFF for cumulative tables only)", false, nullptr, 0,
   false, true, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"STEP_MODE", AnnealKeyType::Enum, KIND_PT,
   "steps", 10, "How the CLASSIC kernel sets its steps: FIXED keeps the scales STEP_FRAC gives; FROZEN calibrates the scales of each slot, then keeps them; ADAPTIVE calibrates them and keeps adapting them, a diagonal form of robust adaptive Metropolis. The calibration spends at most a quarter of MAXEVAL.",
   nullptr,
   false, true, false, 0.0, 1.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, kSTEP_MODE_values, 3,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"STEP_WARMUP_SWEEPS", AnnealKeyType::Count, KIND_PT,
   "steps", 11, "The smallest number of calibration attempts of each coordinate in the warm-up of the steps; the largest is twice this number.",
   nullptr,
   false, true, false, 0.0, 12.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"STEP_TRUST_FACTOR", AnnealKeyType::Float, KIND_PT,
   "steps", 12, "Under STEP_MODE ADAPTIVE, keeps each adapted scale between the calibrated scale divided by this factor and multiplied by it; OFF sets no bound.",
   nullptr,
   false, true, false, 0.0, 64.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, true, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"INIT_MODE", AnnealKeyType::Enum, KIND_PT | KIND_PA,
   "init", 1, "How the replicas or walkers start: the first at the starting point of the search; JITTER moves the others away from it, UNIFORM draws them from the OPTIM box.",
   "JITTER alone draws Gaussian displacements with the first proposal scales, and JITTER <fraction> uniform displacements of up to the fraction times the width of each box. Under TADAH:ANNEAL_PT, JITTER AUTO <p> measures the bounds of the uniform displacements of each dimension against a band of the loss relative to that of the starting point, and scales them by INIT_JITTER_LADDER, which scales the other JITTER forms too. Under TADAH:ANNEAL_PA, JITTER AUTO takes no tolerance: the bounds are the first proposal scales.",
   false, true, true, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, kINIT_MODE_values, 2,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"INIT_JITTER_MAX", AnnealKeyType::Float, KIND_PT,
   "init", 2, "Under INIT_MODE JITTER AUTO of TADAH:ANNEAL_PT, the largest measured displacement, as a fraction of the width of the OPTIM box, where the first probe lies too.",
   nullptr,
   false, true, false, 0.0, 0.5, 0.0,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"INIT_JITTER_PROBES", AnnealKeyType::Count, KIND_PT,
   "init", 3, "Under INIT_MODE JITTER AUTO of TADAH:ANNEAL_PT, the largest number of probes of the loss in each dimension, split between its two sides.",
   nullptr,
   false, true, false, 0.0, 8.0, 0.0,
   true, false, false, false, 2.0, 0.0,
   "must be >= 2 (one screen probe per side)", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"INIT_JITTER_MAXEVAL", AnnealKeyType::Count, KIND_PT,
   "init", 4, "Under INIT_MODE JITTER AUTO of TADAH:ANNEAL_PT, the largest number of calibration probes, which MAXEVAL does not count.",
   "AUTO takes the number of measured dimensions times INIT_JITTER_PROBES, plus the smaller of INIT_JITTER_AUDIT and the number of measured dimensions; INIT_JITTER_SAMPLE limits the measured dimensions.",
   false, true, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   true, false, 0.0, 0.0,
   KIND_PT,
   nullptr, nullptr, nullptr},
  {"INIT_JITTER_SAMPLE", AnnealKeyType::Count, KIND_PT,
   "init", 6, "Under INIT_MODE JITTER AUTO of TADAH:ANNEAL_PT, the largest number of dimensions measured, taken in turn from the families of parameters; 0 measures every dimension.",
   "A dimension that is not measured takes the median displacement of its family, as a fraction of the width, and hpo_run.log marks it inferred.",
   false, true, false, 0.0, 0.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"INIT_JITTER_AUDIT", AnnealKeyType::Count, KIND_PT,
   "init", 7, "Under INIT_MODE JITTER AUTO of TADAH:ANNEAL_PT, the largest number of measured dimensions probed again at half their displacement, to check the measured displacements for a loss that does not grow with the displacement; 0 checks none.",
   nullptr,
   false, true, false, 0.0, 32.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"INIT_JITTER_LADDER", AnnealKeyType::Float, KIND_PT,
   "init", 8, "Scales the first displacement of each replica of TADAH:ANNEAL_PT by (T_k/T_HOT)^gamma, so that cold replicas start near the starting point and hot ones farther away. AUTO takes gamma 0.5, OFF gives every replica the same displacement, and a number above 0 and up to 4 is gamma.",
   "AUTO follows the equipartition law: about a quadratic minimum, the spread of a coordinate at equilibrium grows as the square root of the temperature; its displacements are at least the smallest change of a coordinate that the number format resolves. The scaling applies to every JITTER form of INIT_MODE, AUTO included.",
   false, true, false, 0.0, -1.0, 0.0,
   false, false, false, false, 0.0, 0.0,
   nullptr, false, nullptr, 0,
   true, true, -1.0, 0.0,
   KIND_PT,
   nullptr, nullptr, nullptr},
  {"INIT_JITTER_MAXTIME_FRAC", AnnealKeyType::Float, KIND_PT,
   "init", 5, "Under INIT_MODE JITTER AUTO of TADAH:ANNEAL_PT, the share of the MAXTIME of the search that the calibration may take, when MAXTIME is set; INIT_JITTER_MAXEVAL caps its probes, which have a budget of their own.",
   nullptr,
   false, true, false, 0.0, 0.25, 0.0,
   true, true, true, false, 0.0, 1.0,
   "must be in (0,1]", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"SWEEPS", AnnealKeyType::Count, KIND_PA,
   "steps", 14, "The number of rounds each walker of TADAH:ANNEAL_PA runs at each temperature: a round is CHAIN_LENGTH proposals of the CLASSIC kernel, or NT step adjustments of the Corana kernel.",
   nullptr,
   false, false, true, 0.0, 0.0, 1.0,
   true, false, false, false, 1.0, 0.0,
   "must be >= 1", false, nullptr, 0,
   false, false, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
  {"ESS_TARGET", AnnealKeyType::Float, KIND_PA,
   "temperature", 9, "The effective sample size that each temperature step of TADAH:ANNEAL_PA keeps, as a fraction of the population: it sets the size of the step. OFF cools on the geometric schedule of RT.",
   nullptr,
   false, false, true, 0.0, 0.0, 0.5,
   true, true, true, true, 0.0, 1.0,
   "must be in (0,1) (or OFF for the geometric RT schedule)", false, nullptr, 0,
   false, true, 0.0, 0.0,
   0u,
   nullptr, nullptr, nullptr},
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
  {"STEP_ADAPT_EVERY", KIND_PT, "PARAM STEP_ADAPT_EVERY was removed: STEP_MODE ADAPTIVE applies the robust-adaptive-Metropolis update after every proposal (an update every n proposals multiplied the feedback gain by n)"},
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
inline constexpr double RESET_TO_BEST_PT = 0.0;
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
