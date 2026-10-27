// GENERATED FILE -- DO NOT EDIT.
//
// Source of truth: HPO/blueprint/hpotarget_params.toml (a part file of HPO/blueprint/hpotarget.toml)
// Generator:       HPO/blueprint/gen_hpotarget.py --header
//                  (build target hpo_regenerate_registries)
// Drift gate:      ctest hpo_hpotarget_registry_sync (gen_hpotarget.py --check)
//
// The PARAM keys of the target-file registry: [[param]],
// [[param_excluded]] and [[param_policy]] (blueprint/README.md). The
// PARAM keys of the annealers are anneal_keys_gen.h's.
#pragma once

#include <tadah/hpo/hpotarget_registry_gen.h>

#include <cstddef>
#include <string_view>

namespace tadah {
namespace hpo {
namespace param_registry {

using hpotarget_registry::Arg;
using hpotarget_registry::ArgKind;
using hpotarget_registry::Limit;
using hpotarget_registry::RefusedWord;
using hpotarget_registry::Span;

/* The PARAM keys, in registry order: kParams[i].id is ParamId(i). */
enum class ParamId : unsigned short {
  PARAM_TADAH_GRID_GRID_POINTS,
  PARAM_DLIB_BFGS_tol_gradinf,
  PARAM_CERES_tol_gradinf,
  PARAM_CERES_max_iter,
  PARAM_CERES_max_ls_iter,
  PARAM_FD_fd_step,
  PARAM_FD_fd_kernel,
  PARAM_FD_diff_method,
  PARAM_ASA_ASA_Limit_Acceptances,
  PARAM_ASA_ASA_Limit_Generated,
  PARAM_ASA_ASA_Limit_Invalid_Generated_States,
  PARAM_ASA_ASA_Accepted_To_Generated_Ratio,
  PARAM_ASA_ASA_Cost_Precision,
  PARAM_ASA_ASA_Maximum_Cost_Repeat,
  PARAM_ASA_ASA_Number_Cost_Samples,
  PARAM_ASA_ASA_Temperature_Ratio_Scale,
  PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio,
  PARAM_ASA_ASA_Temperature_Anneal_Scale,
  PARAM_ASA_ASA_Sequential_Parameters,
  PARAM_ASA_ASA_Initial_Parameter_Temperature,
  PARAM_ASA_ASA_Acceptance_Frequency_Modulus,
  PARAM_ASA_ASA_Generated_Frequency_Modulus,
  PARAM_ASA_ASA_Reanneal_Cost,
  PARAM_ASA_ASA_Reanneal_Parameters,
  PARAM_ASA_ASA_Delta_X,
  PARAM_ASA_ASA_User_Tangents,
  PARAM_ASA_ASA_Curvature_0,
  PARAM_ASA_ASA_User_Quench_Cost_Scale,
  PARAM_ASA_ASA_User_Quench_Param_Scale,
  PARAM_ASA_ASA_User_Temperature_Ratio,
  PARAM_ASA_ASA_User_Delta_Parameter,
  PARAM_ASA_ASA_User_Parameter_Temperature,
  PARAM_ASA_ASA_Multi_Number,
  PARAM_ASA_ASA_Multi_Specify,
  PARAM_ASA_ASA_Multi_Grid,
  PARAM_ASA_ASA_Queue_Size,
  PARAM_ASA_ASA_Queue_Resolution,
  PARAM_ASA_ASA_Gener_Block,
  PARAM_ASA_ASA_Gener_Block_Max,
  PARAM_ASA_ASA_Gener_Mov_Avr,
};

struct Param {
  ParamId id;
  std::string_view id_text;
  std::string_view name;
  Span<std::string_view> optimizers;   // LIB:ALGO, LIB, or * for every one
  Span<Arg> args;
  bool has_default_literal;
  std::string_view default_literal;
  bool indexed;                        // the name takes [<i>]
  Span<std::string_view> checks;
};

/* A PARAM key an optimiser names but does not take, and why. */
struct ParamExcluded {
  std::string_view name;
  Span<std::string_view> optimizers;
  std::string_view why;
};

/* What an optimiser does with a PARAM key no [[param]] entry names. */
struct ParamPolicy {
  std::string_view id_text;
  Span<std::string_view> optimizers;
  std::string_view msg;
  Span<std::string_view> checks;
};

/* The PARAM keys of one optimiser as the entries name it (LIB:ALGO, LIB or
 * *), in registry order. */
struct OptimiserParams {
  std::string_view optimizer;
  Span<ParamId> params;
};

namespace detail {

inline constexpr std::string_view PARAM_TADAH_GRID_GRID_POINTS_optimizers[] = {
    "TADAH:GRID",
};
inline constexpr Arg PARAM_TADAH_GRID_GRID_POINTS_args[] = {
    Arg{"n", ArgKind::Number, false, false, Limit{true, "2", 2.0, true, 2ull},
        Limit{true, "UINT_MAX", 4294967295.0, true, 4294967295ull}, false,
        true, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view PARAM_TADAH_GRID_GRID_POINTS_checks[] = {
    "{phase}: TADAH:GRID: PARAM GRID_POINTS must be an integer from 2 to 4294967295, got '{value}' (the points of each dimension include both of its bounds)",
};
inline constexpr std::string_view PARAM_DLIB_BFGS_tol_gradinf_optimizers[] = {
    "DLIB:BFGS",
};
inline constexpr Arg PARAM_DLIB_BFGS_tol_gradinf_args[] = {
    Arg{"tolerance", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_DLIB_BFGS_tol_gradinf_checks[] = {
    "{phase}: DLIB:BFGS requires PARAM tol_gradinf > 0",
    "{phase}: DLIB:BFGS accepts either FTOL_ABS or PARAM tol_gradinf, but not both",
};
inline constexpr std::string_view PARAM_CERES_tol_gradinf_optimizers[] = {
    "CERES:STEEPEST_DESCENT",
};
inline constexpr Arg PARAM_CERES_tol_gradinf_args[] = {
    Arg{"tolerance", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_CERES_tol_gradinf_checks[] = {
    "{phase}: CERES:{algo} requires PARAM tol_gradinf > 0",
};
inline constexpr std::string_view PARAM_CERES_max_iter_optimizers[] = {
    "CERES:STEEPEST_DESCENT",
};
inline constexpr Arg PARAM_CERES_max_iter_args[] = {
    Arg{"n", ArgKind::Number, false, false, Limit{true, "0", 0.0, true, 0ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_CERES_max_iter_checks[] = {
    "{phase}: CERES:{algo} requires PARAM max_iter to be an integer from 0 to 2147483647 (Ceres' max_num_iterations), got {value}",
};
inline constexpr std::string_view PARAM_CERES_max_ls_iter_optimizers[] = {
    "CERES:STEEPEST_DESCENT",
};
inline constexpr Arg PARAM_CERES_max_ls_iter_args[] = {
    Arg{"n", ArgKind::Number, false, false, Limit{true, "1", 1.0, true, 1ull},
        Limit{true, "INT_MAX", 2147483647.0, true, 2147483647ull}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_CERES_max_ls_iter_checks[] = {
    "{phase}: CERES:{algo} requires PARAM max_ls_iter to be an integer from 1 to 2147483647 (Ceres' max_num_line_search_step_size_iterations), got {value}",
};
inline constexpr std::string_view PARAM_FD_fd_step_optimizers[] = {
    "*",
};
inline constexpr Arg PARAM_FD_fd_step_args[] = {
    Arg{"step", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_FD_fd_step_checks[] = {
    "{phase} {lib}:{algo}: PARAM fd_step must be a positive finite number (got {value})",
    "{phase} {lib}:{algo}: PARAM fd_step never takes effect: {why}; remove it.",
    "{phase} {lib}:{algo}: PARAM fd_step never takes effect: an INNER search takes the gradient of the block it runs in, with that block's finite-difference settings; set it in the enclosing block.",
    "{phase} {lib}:{algo}: NOISE AUTO has no effect: PARAM fd_step {value} is set; remove one of them.",
    "{phase} {lib}:{algo}: {value} does not fit these OPTIM boxes: no finite-difference point is evaluated outside the box, and next to a bound the {why} inside it; the step is not shrunk:{list}\n  {note}.",
    "{phase} {lib}:{algo}: PARAM {list} never take effect: {why}; remove them.",
};
inline constexpr std::string_view PARAM_FD_fd_kernel_optimizers[] = {
    "*",
};
inline constexpr std::string_view PARAM_FD_fd_kernel_arg0_words[] = {
    "inhouse",
    "INHOUSE",
    "ceres",
    "CERES",
};
inline constexpr Arg PARAM_FD_fd_kernel_args[] = {
    Arg{"rule", ArgKind::Word, false, false, Limit{}, Limit{}, false, false,
        Span<std::string_view>{detail::PARAM_FD_fd_kernel_arg0_words, 4},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view PARAM_FD_fd_kernel_checks[] = {
    "{phase} {lib}:{algo}: PARAM fd_kernel must be inhouse or ceres (got '{value}')",
    "{phase} {lib}:{algo}: PARAM fd_kernel must be inhouse or ceres (got {value})",
    "{phase} {lib}:{algo}: PARAM fd_kernel never takes effect: {why}; remove it.",
    "{phase} {lib}:{algo}: PARAM fd_kernel never takes effect: an INNER search takes the gradient of the block it runs in, with that block's finite-difference settings; set it in the enclosing block.",
};
inline constexpr std::string_view PARAM_FD_diff_method_optimizers[] = {
    "*",
};
inline constexpr Arg PARAM_FD_diff_method_args[] = {
    Arg{"scheme", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{true, "1", 1.0, true, 1ull},
        false, true, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view PARAM_FD_diff_method_checks[] = {
    "{phase} {lib}:{algo}: PARAM diff_method must be 0 (CENTRAL, the default) or 1 (FORWARD) (got {value})",
    "{phase} {lib}:{algo}: PARAM diff_method 2 (RIDDERS) is refused: Ridders' extrapolation ignored PARAM fd_step, started from a step of 0.32 x max(|u|, 1) -- outside most OPTIM boxes -- and took up to 20 evaluations per parameter; diff_method 0 (CENTRAL, the default) and 1 (FORWARD) take PARAM fd_step",
    "{phase} {lib}:{algo}: PARAM diff_method never takes effect: {why}; remove it.",
    "{phase} {lib}:{algo}: PARAM diff_method never takes effect: an INNER search takes the gradient of the block it runs in, with that block's finite-difference settings; set it in the enclosing block.",
};
inline constexpr std::string_view PARAM_ASA_ASA_Limit_Acceptances_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Limit_Acceptances_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Limit_Acceptances_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Limit_Generated_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Limit_Generated_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Limit_Generated_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Limit_Invalid_Generated_States_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Limit_Invalid_Generated_States_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "-1", -1.0, false, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Limit_Invalid_Generated_States_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Accepted_To_Generated_Ratio_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Accepted_To_Generated_Ratio_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Accepted_To_Generated_Ratio_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM Accepted_To_Generated_Ratio must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Cost_Precision_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Cost_Precision_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Cost_Precision_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM Cost_Precision must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Maximum_Cost_Repeat_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Maximum_Cost_Repeat_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Maximum_Cost_Repeat_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Number_Cost_Samples_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Number_Cost_Samples_args[] = {
    Arg{"value", ArgKind::Number, false, false, Limit{}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Number_Cost_Samples_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM Number_Cost_Samples must not be 0 or -1 (>0 = mean |cost| sample, <-1 = deviation sample)",
};
inline constexpr std::string_view PARAM_ASA_ASA_Temperature_Ratio_Scale_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Temperature_Ratio_Scale_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Temperature_Ratio_Scale_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM Temperature_Ratio_Scale must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM Cost_Parameter_Scale_Ratio must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Temperature_Anneal_Scale_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Temperature_Anneal_Scale_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Temperature_Anneal_Scale_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM Temperature_Anneal_Scale must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Sequential_Parameters_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Sequential_Parameters_args[] = {
    Arg{"value", ArgKind::Number, false, false, Limit{}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Sequential_Parameters_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM Sequential_Parameters must be < the number of OPTIM dimensions ({count})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Initial_Parameter_Temperature_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Initial_Parameter_Temperature_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Initial_Parameter_Temperature_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM Initial_Parameter_Temperature must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Acceptance_Frequency_Modulus_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Acceptance_Frequency_Modulus_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Acceptance_Frequency_Modulus_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Generated_Frequency_Modulus_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Generated_Frequency_Modulus_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Generated_Frequency_Modulus_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Reanneal_Cost_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Reanneal_Cost_args[] = {
    Arg{"value", ArgKind::Number, false, false, Limit{}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Reanneal_Cost_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM Reanneal_Cost == -1 is invalid upstream (use 0 = off, 1 = default, >1 or <-1 = sample variants)",
};
inline constexpr std::string_view PARAM_ASA_ASA_Reanneal_Parameters_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Reanneal_Parameters_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{true, "1", 1.0, true, 1ull},
        false, true, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Reanneal_Parameters_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be <= {max} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Delta_X_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Delta_X_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Delta_X_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase} ASA:ASA: ASA's finite-difference probes -- the reannealing and exit tangents at (1 + Delta) x, or (1 - Delta) x when that lies outside the box{note} -- would be evaluated outside the OPTIM box of these dimensions:{list}\n  Lower the Delta{why}.",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Tangents_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_User_Tangents_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{true, "0", 0.0, true, 0ull},
        false, true, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Tangents_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be <= {max} (got {value})",
    "{phase}: ASA:ASA: PARAM User_Tangents takes 0 only: tadah hpo gives ASA no analytic tangents, and the reannealing takes them by finite differences (PARAM Delta_X)",
};
inline constexpr std::string_view PARAM_ASA_ASA_Curvature_0_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Curvature_0_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "-1", -1.0, false, 0ull},
        Limit{true, "1", 1.0, true, 1ull}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Curvature_0_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be <= {max} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Quench_Cost_Scale_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_User_Quench_Cost_Scale_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Quench_Cost_Scale_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM User_Quench_Cost_Scale must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Quench_Param_Scale_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_User_Quench_Param_Scale_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Quench_Param_Scale_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Temperature_Ratio_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_User_Temperature_Ratio_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Temperature_Ratio_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Delta_Parameter_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_User_Delta_Parameter_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Delta_Parameter_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be > 0 (got {value})",
    "{phase} ASA:ASA: ASA's finite-difference probes -- the reannealing and exit tangents at (1 + Delta) x, or (1 - Delta) x when that lies outside the box{note} -- would be evaluated outside the OPTIM box of these dimensions:{list}\n  Lower the Delta{why}.",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Parameter_Temperature_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_User_Parameter_Temperature_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, true, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Parameter_Temperature_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be > 0 (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Multi_Number_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Multi_Number_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "1", 1.0, true, 1ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Multi_Number_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Multi_Specify_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Multi_Specify_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{true, "1", 1.0, true, 1ull},
        false, true, Span<std::string_view>{nullptr, 0},
        Span<RefusedWord>{nullptr, 0}, "", "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Multi_Specify_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be <= {max} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Multi_Grid_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Multi_Grid_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Multi_Grid_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Queue_Size_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Queue_Size_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Queue_Size_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Queue_Resolution_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Queue_Resolution_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, false,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Queue_Resolution_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Gener_Block_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Gener_Block_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "1", 1.0, true, 1ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Gener_Block_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr std::string_view PARAM_ASA_ASA_Gener_Block_Max_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Gener_Block_Max_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "1", 1.0, true, 1ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Gener_Block_Max_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
    "{phase}: ASA:ASA: PARAM Gener_Block_Max must be >= Gener_Block (buffers are sized once by the cap)",
};
inline constexpr std::string_view PARAM_ASA_ASA_Gener_Mov_Avr_optimizers[] = {
    "ASA:ASA",
};
inline constexpr Arg PARAM_ASA_ASA_Gener_Mov_Avr_args[] = {
    Arg{"value", ArgKind::Number, false, false,
        Limit{true, "0", 0.0, true, 0ull}, Limit{}, false, true,
        Span<std::string_view>{nullptr, 0}, Span<RefusedWord>{nullptr, 0}, "",
        "", "", false, ""},
};
inline constexpr std::string_view PARAM_ASA_ASA_Gener_Mov_Avr_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} must be an integer (got {value})",
    "{phase}: ASA:ASA: PARAM {key} must be >= {min} (got {value})",
};
inline constexpr ParamId optimiser0_params[] = {
    ParamId::PARAM_TADAH_GRID_GRID_POINTS,
};
inline constexpr ParamId optimiser1_params[] = {
    ParamId::PARAM_DLIB_BFGS_tol_gradinf,
};
inline constexpr ParamId optimiser2_params[] = {
    ParamId::PARAM_CERES_tol_gradinf,
    ParamId::PARAM_CERES_max_iter,
    ParamId::PARAM_CERES_max_ls_iter,
};
inline constexpr ParamId optimiser3_params[] = {
    ParamId::PARAM_FD_fd_step,
    ParamId::PARAM_FD_fd_kernel,
    ParamId::PARAM_FD_diff_method,
};
inline constexpr ParamId optimiser4_params[] = {
    ParamId::PARAM_ASA_ASA_Limit_Acceptances,
    ParamId::PARAM_ASA_ASA_Limit_Generated,
    ParamId::PARAM_ASA_ASA_Limit_Invalid_Generated_States,
    ParamId::PARAM_ASA_ASA_Accepted_To_Generated_Ratio,
    ParamId::PARAM_ASA_ASA_Cost_Precision,
    ParamId::PARAM_ASA_ASA_Maximum_Cost_Repeat,
    ParamId::PARAM_ASA_ASA_Number_Cost_Samples,
    ParamId::PARAM_ASA_ASA_Temperature_Ratio_Scale,
    ParamId::PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio,
    ParamId::PARAM_ASA_ASA_Temperature_Anneal_Scale,
    ParamId::PARAM_ASA_ASA_Sequential_Parameters,
    ParamId::PARAM_ASA_ASA_Initial_Parameter_Temperature,
    ParamId::PARAM_ASA_ASA_Acceptance_Frequency_Modulus,
    ParamId::PARAM_ASA_ASA_Generated_Frequency_Modulus,
    ParamId::PARAM_ASA_ASA_Reanneal_Cost,
    ParamId::PARAM_ASA_ASA_Reanneal_Parameters,
    ParamId::PARAM_ASA_ASA_Delta_X,
    ParamId::PARAM_ASA_ASA_User_Tangents,
    ParamId::PARAM_ASA_ASA_Curvature_0,
    ParamId::PARAM_ASA_ASA_User_Quench_Cost_Scale,
    ParamId::PARAM_ASA_ASA_User_Quench_Param_Scale,
    ParamId::PARAM_ASA_ASA_User_Temperature_Ratio,
    ParamId::PARAM_ASA_ASA_User_Delta_Parameter,
    ParamId::PARAM_ASA_ASA_User_Parameter_Temperature,
    ParamId::PARAM_ASA_ASA_Multi_Number,
    ParamId::PARAM_ASA_ASA_Multi_Specify,
    ParamId::PARAM_ASA_ASA_Multi_Grid,
    ParamId::PARAM_ASA_ASA_Queue_Size,
    ParamId::PARAM_ASA_ASA_Queue_Resolution,
    ParamId::PARAM_ASA_ASA_Gener_Block,
    ParamId::PARAM_ASA_ASA_Gener_Block_Max,
    ParamId::PARAM_ASA_ASA_Gener_Mov_Avr,
};
inline constexpr std::string_view PARAM_ASA_ASA_Include_Integer_Parameters_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Initial_Parameters_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Coarse_Resolution_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_User_Cost_Temperature_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Asa_Recursive_Level_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Asymp_Exp_Param_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Fit_Local_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Iter_Max_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Penalty_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_NoOfSamples_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Random_Array_Dim_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Bias_Acceptance_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_Limit_Weights_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_NLOPT_optimizers[] = {
    "NLOPT",
};
inline constexpr std::string_view PARAM_NLOPT_checks[] = {
    "{phase}: NLOPT:{algo}: PARAM {key} takes a number (got '{value}')",
};
inline constexpr std::string_view PARAM_TADAH_RANDOM_optimizers[] = {
    "TADAH:RANDOM",
};
inline constexpr std::string_view PARAM_DLIB_GFS_optimizers[] = {
    "DLIB:GFS",
};
inline constexpr std::string_view PARAM_TADAH_GRID_optimizers[] = {
    "TADAH:GRID",
};
inline constexpr std::string_view PARAM_DLIB_BFGS_optimizers[] = {
    "DLIB:BFGS",
};
inline constexpr std::string_view PARAM_DLIB_BFGS_checks[] = {
    "{phase}: DLIB:BFGS: PARAM {key} takes a number (got '{value}')",
};
inline constexpr std::string_view PARAM_CERES_optimizers[] = {
    "CERES:STEEPEST_DESCENT",
};
inline constexpr std::string_view PARAM_CERES_checks[] = {
    "{phase}: CERES:{algo}: PARAM {key} takes a number (got '{value}')",
};
inline constexpr std::string_view PARAM_ASA_ASA_optimizers[] = {
    "ASA:ASA",
};
inline constexpr std::string_view PARAM_ASA_ASA_checks[] = {
    "{phase}: ASA:ASA: PARAM {key} takes a number (got '{value}')",
    "{phase}: ASA:ASA does not take the ASA option PARAM {name}: {why}",
    "{phase}: ASA:ASA: unknown PARAM '{key}': ASA option names are case-sensitive, and ASA:ASA does not take the ASA option {name}: {why}",
    "{phase}: ASA:ASA: unknown PARAM '{key}': ASA option names are case-sensitive, and the option of that name is {name}",
    "{phase}: ASA:ASA: PARAM {key} is a scalar option (indexed form not accepted)",
    "{phase}: ASA:ASA: PARAM {key}: index 0 is no dimension: the indices count the dimensions of the search space from 1, as the dim column of hpo_search_space.tsv does",
    "{phase}: ASA:ASA: PARAM {key}: index {index} is above the number of dimensions of the search space, {count} (indices count from 1)",
    "{phase}: ASA:ASA: malformed indexed PARAM '{name}' (expected Name[<1-based index>])",
    "{phase}: ASA:ASA: malformed index in PARAM '{name}' (expected a positive integer)",
};
inline constexpr std::string_view PARAM_TADAH_ANNEAL_optimizers[] = {
    "TADAH:ANNEAL",
    "TADAH:ANNEAL_PT",
    "TADAH:ANNEAL_PA",
};
inline constexpr std::string_view PARAM_TADAH_ANNEAL_checks[] = {
    "{phase}: TADAH:ANNEAL_PA: PARAM INIT_MODE JITTER AUTO takes no tolerance (got '{tok}'): TADAH:ANNEAL_PA measures no displacement and takes the first proposal scales as the bounds; TADAH:ANNEAL_PT measures them against the tolerance",
    "{phase}: TADAH:{algo}: PARAM {key} is a key of {name}",
    "{phase}: TADAH:{algo}: PARAM {key} must be {what} (got '{tok}'{note})",
    "{phase}: TADAH:{algo}: PARAM {key} must be one of: {list} (got '{tok}')",
};

}  // namespace detail

inline constexpr Param kParams[] = {
  Param{ParamId::PARAM_TADAH_GRID_GRID_POINTS, "PARAM.TADAH:GRID.GRID_POINTS",
      "GRID_POINTS",
      Span<std::string_view>{detail::PARAM_TADAH_GRID_GRID_POINTS_optimizers, 1},
      Span<Arg>{detail::PARAM_TADAH_GRID_GRID_POINTS_args, 1}, true, "20",
      false,
      Span<std::string_view>{detail::PARAM_TADAH_GRID_GRID_POINTS_checks, 1}},
  Param{ParamId::PARAM_DLIB_BFGS_tol_gradinf, "PARAM.DLIB:BFGS.tol_gradinf",
      "tol_gradinf",
      Span<std::string_view>{detail::PARAM_DLIB_BFGS_tol_gradinf_optimizers, 1},
      Span<Arg>{detail::PARAM_DLIB_BFGS_tol_gradinf_args, 1}, false, "", false,
      Span<std::string_view>{detail::PARAM_DLIB_BFGS_tol_gradinf_checks, 2}},
  Param{ParamId::PARAM_CERES_tol_gradinf, "PARAM.CERES.tol_gradinf",
      "tol_gradinf",
      Span<std::string_view>{detail::PARAM_CERES_tol_gradinf_optimizers, 1},
      Span<Arg>{detail::PARAM_CERES_tol_gradinf_args, 1}, true, "1e-10", false,
      Span<std::string_view>{detail::PARAM_CERES_tol_gradinf_checks, 1}},
  Param{ParamId::PARAM_CERES_max_iter, "PARAM.CERES.max_iter", "max_iter",
      Span<std::string_view>{detail::PARAM_CERES_max_iter_optimizers, 1},
      Span<Arg>{detail::PARAM_CERES_max_iter_args, 1}, false, "", false,
      Span<std::string_view>{detail::PARAM_CERES_max_iter_checks, 1}},
  Param{ParamId::PARAM_CERES_max_ls_iter, "PARAM.CERES.max_ls_iter",
      "max_ls_iter",
      Span<std::string_view>{detail::PARAM_CERES_max_ls_iter_optimizers, 1},
      Span<Arg>{detail::PARAM_CERES_max_ls_iter_args, 1}, false, "", false,
      Span<std::string_view>{detail::PARAM_CERES_max_ls_iter_checks, 1}},
  Param{ParamId::PARAM_FD_fd_step, "PARAM.FD.fd_step", "fd_step",
      Span<std::string_view>{detail::PARAM_FD_fd_step_optimizers, 1},
      Span<Arg>{detail::PARAM_FD_fd_step_args, 1}, true, "1e-6", false,
      Span<std::string_view>{detail::PARAM_FD_fd_step_checks, 6}},
  Param{ParamId::PARAM_FD_fd_kernel, "PARAM.FD.fd_kernel", "fd_kernel",
      Span<std::string_view>{detail::PARAM_FD_fd_kernel_optimizers, 1},
      Span<Arg>{detail::PARAM_FD_fd_kernel_args, 1}, false, "", false,
      Span<std::string_view>{detail::PARAM_FD_fd_kernel_checks, 4}},
  Param{ParamId::PARAM_FD_diff_method, "PARAM.FD.diff_method", "diff_method",
      Span<std::string_view>{detail::PARAM_FD_diff_method_optimizers, 1},
      Span<Arg>{detail::PARAM_FD_diff_method_args, 1}, true, "0", false,
      Span<std::string_view>{detail::PARAM_FD_diff_method_checks, 4}},
  Param{ParamId::PARAM_ASA_ASA_Limit_Acceptances,
      "PARAM.ASA:ASA.Limit_Acceptances", "Limit_Acceptances",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Limit_Acceptances_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Limit_Acceptances_args, 1}, true,
      "10000", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Limit_Acceptances_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Limit_Generated,
      "PARAM.ASA:ASA.Limit_Generated", "Limit_Generated",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Limit_Generated_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Limit_Generated_args, 1}, true, "99999",
      false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Limit_Generated_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Limit_Invalid_Generated_States,
      "PARAM.ASA:ASA.Limit_Invalid_Generated_States",
      "Limit_Invalid_Generated_States",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Limit_Invalid_Generated_States_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Limit_Invalid_Generated_States_args, 1},
      true, "1000", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Limit_Invalid_Generated_States_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Accepted_To_Generated_Ratio,
      "PARAM.ASA:ASA.Accepted_To_Generated_Ratio",
      "Accepted_To_Generated_Ratio",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Accepted_To_Generated_Ratio_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Accepted_To_Generated_Ratio_args, 1},
      true, "1e-6", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Accepted_To_Generated_Ratio_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Cost_Precision, "PARAM.ASA:ASA.Cost_Precision",
      "Cost_Precision",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Cost_Precision_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Cost_Precision_args, 1}, true, "1e-18",
      false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Cost_Precision_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Maximum_Cost_Repeat,
      "PARAM.ASA:ASA.Maximum_Cost_Repeat", "Maximum_Cost_Repeat",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Maximum_Cost_Repeat_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Maximum_Cost_Repeat_args, 1}, true, "5",
      false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Maximum_Cost_Repeat_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Number_Cost_Samples,
      "PARAM.ASA:ASA.Number_Cost_Samples", "Number_Cost_Samples",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Number_Cost_Samples_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Number_Cost_Samples_args, 1}, true, "5",
      false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Number_Cost_Samples_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Temperature_Ratio_Scale,
      "PARAM.ASA:ASA.Temperature_Ratio_Scale", "Temperature_Ratio_Scale",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Temperature_Ratio_Scale_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Temperature_Ratio_Scale_args, 1}, true,
      "1e-5", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Temperature_Ratio_Scale_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio,
      "PARAM.ASA:ASA.Cost_Parameter_Scale_Ratio", "Cost_Parameter_Scale_Ratio",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio_args, 1},
      true, "1.0", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Cost_Parameter_Scale_Ratio_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Temperature_Anneal_Scale,
      "PARAM.ASA:ASA.Temperature_Anneal_Scale", "Temperature_Anneal_Scale",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Temperature_Anneal_Scale_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Temperature_Anneal_Scale_args, 1}, true,
      "100.0", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Temperature_Anneal_Scale_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Sequential_Parameters,
      "PARAM.ASA:ASA.Sequential_Parameters", "Sequential_Parameters",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Sequential_Parameters_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Sequential_Parameters_args, 1}, true,
      "-1", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Sequential_Parameters_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Initial_Parameter_Temperature,
      "PARAM.ASA:ASA.Initial_Parameter_Temperature",
      "Initial_Parameter_Temperature",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Initial_Parameter_Temperature_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Initial_Parameter_Temperature_args, 1},
      true, "1.0", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Initial_Parameter_Temperature_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Acceptance_Frequency_Modulus,
      "PARAM.ASA:ASA.Acceptance_Frequency_Modulus",
      "Acceptance_Frequency_Modulus",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Acceptance_Frequency_Modulus_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Acceptance_Frequency_Modulus_args, 1},
      true, "100", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Acceptance_Frequency_Modulus_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Generated_Frequency_Modulus,
      "PARAM.ASA:ASA.Generated_Frequency_Modulus",
      "Generated_Frequency_Modulus",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Generated_Frequency_Modulus_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Generated_Frequency_Modulus_args, 1},
      true, "10000", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Generated_Frequency_Modulus_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Reanneal_Cost, "PARAM.ASA:ASA.Reanneal_Cost",
      "Reanneal_Cost",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Reanneal_Cost_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Reanneal_Cost_args, 1}, true, "1", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Reanneal_Cost_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Reanneal_Parameters,
      "PARAM.ASA:ASA.Reanneal_Parameters", "Reanneal_Parameters",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Reanneal_Parameters_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Reanneal_Parameters_args, 1}, true, "1",
      false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Reanneal_Parameters_checks, 3}},
  Param{ParamId::PARAM_ASA_ASA_Delta_X, "PARAM.ASA:ASA.Delta_X", "Delta_X",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Delta_X_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Delta_X_args, 1}, true, "0.001", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Delta_X_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_User_Tangents, "PARAM.ASA:ASA.User_Tangents",
      "User_Tangents",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Tangents_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_User_Tangents_args, 1}, true, "0", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Tangents_checks, 4}},
  Param{ParamId::PARAM_ASA_ASA_Curvature_0, "PARAM.ASA:ASA.Curvature_0",
      "Curvature_0",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Curvature_0_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Curvature_0_args, 1}, true, "1", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Curvature_0_checks, 3}},
  Param{ParamId::PARAM_ASA_ASA_User_Quench_Cost_Scale,
      "PARAM.ASA:ASA.User_Quench_Cost_Scale", "User_Quench_Cost_Scale",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Quench_Cost_Scale_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_User_Quench_Cost_Scale_args, 1}, true,
      "1.0", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Quench_Cost_Scale_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_User_Quench_Param_Scale,
      "PARAM.ASA:ASA.User_Quench_Param_Scale", "User_Quench_Param_Scale",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Quench_Param_Scale_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_User_Quench_Param_Scale_args, 1}, true,
      "1.0", true,
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Quench_Param_Scale_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_User_Temperature_Ratio,
      "PARAM.ASA:ASA.User_Temperature_Ratio", "User_Temperature_Ratio",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Temperature_Ratio_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_User_Temperature_Ratio_args, 1}, true,
      "1.0", true,
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Temperature_Ratio_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_User_Delta_Parameter,
      "PARAM.ASA:ASA.User_Delta_Parameter", "User_Delta_Parameter",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Delta_Parameter_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_User_Delta_Parameter_args, 1}, false, "",
      true,
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Delta_Parameter_checks, 3}},
  Param{ParamId::PARAM_ASA_ASA_User_Parameter_Temperature,
      "PARAM.ASA:ASA.User_Parameter_Temperature", "User_Parameter_Temperature",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Parameter_Temperature_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_User_Parameter_Temperature_args, 1},
      false, "", true,
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Parameter_Temperature_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Multi_Number, "PARAM.ASA:ASA.Multi_Number",
      "Multi_Number",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Multi_Number_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Multi_Number_args, 1}, true, "1", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Multi_Number_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Multi_Specify, "PARAM.ASA:ASA.Multi_Specify",
      "Multi_Specify",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Multi_Specify_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Multi_Specify_args, 1}, true, "1", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Multi_Specify_checks, 3}},
  Param{ParamId::PARAM_ASA_ASA_Multi_Grid, "PARAM.ASA:ASA.Multi_Grid",
      "Multi_Grid",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Multi_Grid_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Multi_Grid_args, 1}, true, "0.0", true,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Multi_Grid_checks, 1}},
  Param{ParamId::PARAM_ASA_ASA_Queue_Size, "PARAM.ASA:ASA.Queue_Size",
      "Queue_Size",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Queue_Size_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Queue_Size_args, 1}, true, "0", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Queue_Size_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Queue_Resolution,
      "PARAM.ASA:ASA.Queue_Resolution", "Queue_Resolution",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Queue_Resolution_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Queue_Resolution_args, 1}, true, "0.0",
      true,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Queue_Resolution_checks, 1}},
  Param{ParamId::PARAM_ASA_ASA_Gener_Block, "PARAM.ASA:ASA.Gener_Block",
      "Gener_Block",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Gener_Block_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Gener_Block_args, 1}, true, "1", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Gener_Block_checks, 2}},
  Param{ParamId::PARAM_ASA_ASA_Gener_Block_Max,
      "PARAM.ASA:ASA.Gener_Block_Max", "Gener_Block_Max",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Gener_Block_Max_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Gener_Block_Max_args, 1}, false, "",
      false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Gener_Block_Max_checks, 3}},
  Param{ParamId::PARAM_ASA_ASA_Gener_Mov_Avr, "PARAM.ASA:ASA.Gener_Mov_Avr",
      "Gener_Mov_Avr",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Gener_Mov_Avr_optimizers, 1},
      Span<Arg>{detail::PARAM_ASA_ASA_Gener_Mov_Avr_args, 1}, true, "3", false,
      Span<std::string_view>{detail::PARAM_ASA_ASA_Gener_Mov_Avr_checks, 2}},
};

inline constexpr OptimiserParams kParamsByOptimiser[] = {
  OptimiserParams{"TADAH:GRID", Span<ParamId>{detail::optimiser0_params, 1}},
  OptimiserParams{"DLIB:BFGS", Span<ParamId>{detail::optimiser1_params, 1}},
  OptimiserParams{"CERES:STEEPEST_DESCENT",
      Span<ParamId>{detail::optimiser2_params, 3}},
  OptimiserParams{"*", Span<ParamId>{detail::optimiser3_params, 3}},
  OptimiserParams{"ASA:ASA", Span<ParamId>{detail::optimiser4_params, 32}},
};

inline constexpr ParamExcluded kParamExcluded[] = {
  ParamExcluded{"Include_Integer_Parameters",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Include_Integer_Parameters_optimizers, 1},
      "every OPTIM dimension of tadah hpo takes real values, so ASA has no integer dimension to include."},
  ParamExcluded{"User_Initial_Parameters",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Initial_Parameters_optimizers, 1},
      "ASA starts at the first starting point of the INIT block (the values of the training configuration without one); METHOD RANDOM 1 in the INIT block draws a random start from its SEED."},
  ParamExcluded{"Coarse_Resolution",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Coarse_Resolution_optimizers, 1},
      "it rounds each generated value to a grid of that width and makes ASA's queue compare points within it in place of Queue_Resolution; the ASA of tadah hpo leaves it out. PARAM Queue_Size with Queue_Resolution reuses the loss of a point within Queue_Resolution of one of the last Queue_Size points."},
  ParamExcluded{"User_Cost_Temperature",
      Span<std::string_view>{detail::PARAM_ASA_ASA_User_Cost_Temperature_optimizers, 1},
      "it gives the initial cost temperature in place of ASA's sampling of it over Number_Cost_Samples states, which tadah hpo runs (PARAM Number_Cost_Samples)."},
  ParamExcluded{"Asa_Recursive_Level",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Asa_Recursive_Level_optimizers, 1},
      "it belongs to ASA's recursive optimisation of its own options, which the ASA of tadah hpo leaves out."},
  ParamExcluded{"Asymp_Exp_Param",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Asymp_Exp_Param_optimizers, 1},
      "it is the parameter of an asymptotic form of the acceptance test (1 gives the Boltzmann test), which the ASA of tadah hpo leaves out: it accepts by the Boltzmann test."},
  ParamExcluded{"Fit_Local",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Fit_Local_optimizers, 1},
      "it belongs to the local simplex fit that ASA's user module runs after the anneal, and tadah hpo does not include that module."},
  ParamExcluded{"Iter_Max",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Iter_Max_optimizers, 1},
      "it belongs to the local simplex fit that ASA's user module runs after the anneal, and tadah hpo does not include that module."},
  ParamExcluded{"Penalty",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Penalty_optimizers, 1},
      "it belongs to the local simplex fit that ASA's user module runs after the anneal, and tadah hpo does not include that module."},
  ParamExcluded{"NoOfSamples",
      Span<std::string_view>{detail::PARAM_ASA_ASA_NoOfSamples_optimizers, 1},
      "it belongs to the fuzzy-logic tuning of ASA's options in ASA's user module, which tadah hpo does not include."},
  ParamExcluded{"Random_Array_Dim",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Random_Array_Dim_optimizers, 1},
      "it belongs to ASA's saving of its state to resume a run, which the ASA of tadah hpo leaves out: tadah hpo does not resume a run."},
  ParamExcluded{"Bias_Acceptance",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Bias_Acceptance_optimizers, 1},
      "it belongs to ASA's importance sampling, which the ASA of tadah hpo leaves out."},
  ParamExcluded{"Limit_Weights",
      Span<std::string_view>{detail::PARAM_ASA_ASA_Limit_Weights_optimizers, 1},
      "it belongs to ASA's importance sampling, which the ASA of tadah hpo leaves out."},
};

inline constexpr ParamPolicy kParamPolicies[] = {
  ParamPolicy{"PARAM.NLOPT",
      Span<std::string_view>{detail::PARAM_NLOPT_optimizers, 1},
      "{phase}: NLOPT:{algo}: unknown PARAM '{key}' (it takes {keys})",
      Span<std::string_view>{detail::PARAM_NLOPT_checks, 1}},
  ParamPolicy{"PARAM.TADAH:RANDOM",
      Span<std::string_view>{detail::PARAM_TADAH_RANDOM_optimizers, 1},
      "{phase}: TADAH:RANDOM: unknown PARAM '{key}' (it takes none)",
      Span<std::string_view>{nullptr, 0}},
  ParamPolicy{"PARAM.DLIB:GFS",
      Span<std::string_view>{detail::PARAM_DLIB_GFS_optimizers, 1},
      "{phase}: DLIB:GFS: unknown PARAM '{key}' (it takes none)",
      Span<std::string_view>{nullptr, 0}},
  ParamPolicy{"PARAM.TADAH:GRID",
      Span<std::string_view>{detail::PARAM_TADAH_GRID_optimizers, 1},
      "{phase}: TADAH:GRID: unknown PARAM '{key}' (it takes GRID_POINTS)",
      Span<std::string_view>{nullptr, 0}},
  ParamPolicy{"PARAM.DLIB:BFGS",
      Span<std::string_view>{detail::PARAM_DLIB_BFGS_optimizers, 1},
      "{phase}: DLIB:BFGS does not support PARAM {key}; supported PARAM keys are fd_step, fd_kernel, diff_method and tol_gradinf",
      Span<std::string_view>{detail::PARAM_DLIB_BFGS_checks, 1}},
  ParamPolicy{"PARAM.CERES",
      Span<std::string_view>{detail::PARAM_CERES_optimizers, 1},
      "{phase}: CERES:{algo} does not support PARAM {key}; supported: fd_step, fd_kernel, diff_method, tol_gradinf, max_iter, max_ls_iter",
      Span<std::string_view>{detail::PARAM_CERES_checks, 1}},
  ParamPolicy{"PARAM.ASA:ASA",
      Span<std::string_view>{detail::PARAM_ASA_ASA_optimizers, 1},
      "{phase}: ASA:ASA: unknown PARAM '{key}'; accepted keys: {keys}",
      Span<std::string_view>{detail::PARAM_ASA_ASA_checks, 9}},
  ParamPolicy{"PARAM.TADAH:ANNEAL",
      Span<std::string_view>{detail::PARAM_TADAH_ANNEAL_optimizers, 3},
      "{phase}: TADAH:{algo}: unknown PARAM '{key}'; accepted keys: {keys}",
      Span<std::string_view>{detail::PARAM_TADAH_ANNEAL_checks, 4}},
};

}  // namespace param_registry
}  // namespace hpo
}  // namespace tadah
