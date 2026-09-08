#pragma once
/* asa_options.h -- option surface of the LIB ASA / ALGO ASA optimizer
 * (vendored Ingber ASA-30.53; see external/asa/TADAH-PATCHES.md).
 *
 * PARAM keys use the EXACT upstream spellings (case-sensitive), e.g.
 *     PARAM Limit_Acceptances 10000
 * Per-dimension options accept an optional 1-based index suffix, using
 * the same dimension numbering as the `dim` column of
 * hpo_search_space.tsv:
 *     PARAM User_Quench_Param_Scale 0.9      (fills every dimension)
 *     PARAM User_Quench_Param_Scale[3] 0.7   (overrides dimension 3)
 * Unknown keys are hard errors; options excluded from the vendored
 * build get tailored messages naming the reason.
 *
 * This header is installed and must not include any ASA header. */

#include <cstdint>
#include <iosfwd>
#include <string>
#include <vector>

namespace tadah::hpo {

struct OptimizerSpec;

/* Parsed + validated ASA options in ASA's own units.  Defaults are the
 * canonical ASA-README defaults except where noted (Curvature_0 -- see
 * the docs "deviations from upstream defaults" table). */
struct AsaOptions {
    /* ---- the 21 canonical asa_opt options ------------------------- */
    long   limit_acceptances              = 10000;
    long   limit_generated                = 99999;
    int    limit_invalid_generated_states = 1000;
    double accepted_to_generated_ratio    = 1.0e-6;
    double cost_precision                 = 1.0e-18;
    int    maximum_cost_repeat            = 5;
    int    number_cost_samples            = 5;
    double temperature_ratio_scale        = 1.0e-5;
    double cost_parameter_scale_ratio     = 1.0;
    double temperature_anneal_scale       = 100.0;
    int    include_integer_parameters     = 0;
    int    user_initial_parameters        = 0;    /* auto: 1 iff INIT seed */
    long   sequential_parameters          = -1;
    double initial_parameter_temperature  = 1.0;
    int    acceptance_frequency_modulus   = 100;
    int    generated_frequency_modulus    = 10000;
    int    reanneal_cost                  = 1;
    int    reanneal_parameters            = 1;
    double delta_x                        = 0.001;
    int    user_tangents                  = 0;    /* 1 is a hard error */
    int    curvature_0                    = 1;    /* TADAH default (upstream 0):
                                                     0 costs O(D^2) exit evals */
    /* ---- QUENCH_COST / QUENCH_PARAMETERS -------------------------- */
    double user_quench_cost_scale = 1.0;
    std::vector<double> user_quench_param_scale;    /* size D, 1.0     */
    /* ---- RATIO_TEMPERATURE_SCALES ---------------------------------- */
    std::vector<double> user_temperature_ratio;     /* size D, 1.0     */
    /* ---- DELTA_PARAMETERS (mirrors Delta_X unless set) ------------- */
    std::vector<double> user_delta_parameter;       /* size D, Delta_X */
    /* ---- USER_INITIAL_PARAMETERS_TEMPS (mirrors the scalar) -------- */
    std::vector<double> user_parameter_temperature; /* size D, IPT     */
    /* ---- MULTI_MIN -------------------------------------------------- */
    int multi_number  = 1;
    int multi_specify = 1;      /* TADAH default (template 0): strict < */
    std::vector<double> multi_grid;                 /* size D, 0.0     */
    /* ---- ASA_QUEUE (Queue_Size 0 = cache off) ----------------------- */
    long queue_size = 0;
    std::vector<double> queue_resolution;           /* size D, 0.0     */
    /* ---- ASA_PARALLEL (Gener_Block 1 = canonical serial sampling) --- */
    long gener_block     = 1;
    long gener_block_max = 1;   /* resolved to >= gener_block           */
    int  gener_mov_avr   = 3;
};

/* Parse sp.param into AsaOptions for a @p dim -dimensional problem.
 * Applies the auto-defaults (User_Initial_Parameters from sp.optim_init,
 * per-dim mirrors), validates every value against the ranges enforced by
 * asa_test_asa_options(), and hard-errors on unknown or excluded keys.
 * Also rejects unsupported generic spec knobs (FTOL/XTOL/POPULATION/
 * STEP/GRAD_STRATEGY/VECTOR_ARR) with tailored messages. */
AsaOptions parse_asa_options(const OptimizerSpec& sp, std::size_t dim);

/* Human-readable name of an asa() exit_status code. */
const char* asa_exit_name(int exit_status);

/* One-line-per-key PARAM table (used by OptimizerSpec::print_caps). */
void print_asa_param_table(std::ostream& os);

}  // namespace tadah::hpo
