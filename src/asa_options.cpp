/* asa_options.cpp -- PARAM parsing/validation for LIB ASA / ALGO ASA.
 * One descriptor table drives the parser, print_caps, and the tests.
 * Value ranges mirror asa_test_asa_options() in external/asa/asa.c so
 * an INVALID_USER_INPUT exit can only indicate a glue bug. */

#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/optimizer_spec.h>
#include <tadah/hpo/strict_parse.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>

namespace tadah::hpo {

namespace {

enum class Kind { Real, Count, Bool };  /* Count = integral, range-checked */

struct Desc {
    const char* name;      /* exact upstream spelling                   */
    Kind        kind;
    bool        per_dim;   /* accepts the [i] indexed form              */
    double      min;       /* inclusive; NaN = unbounded                */
    double      max;       /* inclusive; NaN = unbounded                */
    const char* def_str;   /* default, for the help table               */
    const char* doc;
};

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/* Order = asa_opt file order, then gated groups; the help table prints
 * in this order. */
const Desc kDescs[] = {
  {"Limit_Acceptances", Kind::Count, false, 0, kNaN, "10000",
   "max accepted states; 0 = no limit"},
  {"Limit_Generated", Kind::Count, false, 0, kNaN, "99999",
   "max generated states; 0 = no limit"},
  {"Limit_Invalid_Generated_States", Kind::Count, false, -1, kNaN, "1000",
   "invalid-state budget; 0 = exit ASAP, -1 = bypass test"},
  {"Accepted_To_Generated_Ratio", Kind::Real, false, 0, kNaN, "1e-6",
   "min accepted/generated ratio before reanneal test (must be > 0)"},
  {"Cost_Precision", Kind::Real, false, 0, kNaN, "1e-18",
   "|last-best| threshold of the cost-repeat plateau test (must be > 0)"},
  {"Maximum_Cost_Repeat", Kind::Count, false, 0, kNaN, "5",
   "consecutive plateau cycles before COST_REPEATING exit; 0 = bypass"},
  {"Number_Cost_Samples", Kind::Count, false, kNaN, kNaN, "5",
   ">0: initial cost temp from mean |cost|; <-1: from a deviation sample"},
  {"Temperature_Ratio_Scale", Kind::Real, false, 0, kNaN, "1e-5",
   "the main control of the annealing rate: m = -log(TRS)"},
  {"Cost_Parameter_Scale_Ratio", Kind::Real, false, 0, kNaN, "1.0",
   "cost temperature schedule scale relative to parameters"},
  {"Temperature_Anneal_Scale", Kind::Real, false, 0, kNaN, "100.0",
   "n = log(TAS); above the number of dimensions for large problems"},
  {"Sequential_Parameters", Kind::Count, false, kNaN, kNaN, "-1",
   "start the cyclic generation of one dimension at a time at this index; any negative = off"},
  {"Initial_Parameter_Temperature", Kind::Real, false, 0, kNaN, "1.0",
   "initial parameter temperature (must be > 0)"},
  {"Acceptance_Frequency_Modulus", Kind::Count, false, 0, kNaN, "100",
   "a periodic test follows each state, or each block of states, that "
   "brings the count of accepted states to a multiple of this number; "
   "0 = off"},
  {"Generated_Frequency_Modulus", Kind::Count, false, 0, kNaN, "10000",
   "a periodic test follows each state, or each block of states, that "
   "brings the count of generated states to a multiple of this number; "
   "0 = off"},
  {"Reanneal_Cost", Kind::Count, false, kNaN, kNaN, "1",
   "0 = off, 1 = default, >1 / <-1 = deviation-sample variants (-1 invalid)"},
  {"Reanneal_Parameters", Kind::Bool, false, 0, 1, "1",
   "adapt the temperature of each dimension to its tangent sensitivity"},
  {"Delta_X", Kind::Real, false, 0, kNaN, "0.001",
   "finite-difference step for tangents; 0 disables tangents"},
  {"User_Tangents", Kind::Bool, false, 0, 1, "0",
   "unsupported: tadah hpo has no analytic tangents (1 = error)"},
  {"Curvature_0", Kind::Count, false, -1, 1, "1",
   "1 = skip curvature array (Tadah!'s default; ASA's own default 0 costs "
   "about n^2 evaluations at the exit, n the dimensions), 0 = full curvature "
   "at exit, -1 = periodic"},
  /* gated groups */
  {"User_Quench_Cost_Scale", Kind::Real, false, 0, kNaN, "1.0",
   "cost-temperature quench exponent Q (>1 quenches; proof-safe)"},
  {"User_Quench_Param_Scale", Kind::Real, true, 0, kNaN, "1.0",
   "quench exponent Q_i of each dimension (>1 voids the sampling proof)"},
  {"User_Temperature_Ratio", Kind::Real, true, 0, kNaN, "1.0",
   "multiplier of -log(Temperature_Ratio_Scale) of each dimension"},
  {"User_Delta_Parameter", Kind::Real, true, 0, kNaN, "Delta_X",
   "finite-difference step of each dimension (defaults to Delta_X)"},
  {"User_Parameter_Temperature", Kind::Real, true, 0, kNaN, "IPT",
   "initial temperature of each dimension (defaults to Initial_Parameter_Temperature)"},
  {"Multi_Number", Kind::Count, false, 1, kNaN, "1",
   "track the best N distinct minima (reported in the journal)"},
  {"Multi_Specify", Kind::Bool, false, 0, 1, "1",
   "1 = strict < for new minima (Tadah!'s default); 0 = <=, as in ASA's "
   "example options"},
  {"Multi_Grid", Kind::Real, true, 0, kNaN, "0.0",
   "distinctness grid of each dimension for MULTI_MIN (0 -> EPS_DOUBLE)"},
  {"Queue_Size", Kind::Count, false, 0, kNaN, "0",
   "FIFO eval-cache size; 0 = cache off"},
  {"Queue_Resolution", Kind::Real, true, 0, kNaN, "0.0",
   "cache-hit resolution of each dimension"},
  {"Gener_Block", Kind::Count, false, 1, kNaN, "1",
   "initial generated-state block size; 1 (with Gener_Block_Max 1) = "
   "canonical serial sampling, >1 evaluates the block concurrently "
   "(THREADS-wide) and changes sampling"},
  {"Gener_Block_Max", Kind::Count, false, 1, kNaN, "Gener_Block",
   "block-size cap (buffers sized once); the adaptive resizing keeps the "
   "block between 1 and this cap, so it can shrink below Gener_Block"},
  {"Gener_Mov_Avr", Kind::Count, false, 0, kNaN, "3",
   "moving-average window of the adaptive block resizing; 0 = no "
   "resizing: every block holds Gener_Block states"},
};

/* Upstream option names deliberately not exposed: tailored errors. Two of
 * them are set by tadah hpo: Include_Integer_Parameters 0 (no OPTIM
 * dimension is an integer one) and User_Initial_Parameters from the
 * starting points (parse_asa_options). */
const std::map<std::string, const char*> kExcluded = {
  {"Include_Integer_Parameters",
   "every OPTIM dimension of tadah hpo takes real values, so ASA has no "
   "integer dimension to include."},
  {"User_Initial_Parameters",
   "ASA starts at the first starting point of the INIT block (the values of "
   "the training configuration without one); METHOD RANDOM 1 in the INIT "
   "block draws a random start from its SEED."},
  {"Coarse_Resolution",
   "ASA_RESOLUTION is compiled out: the grid snap has no trajectory-"
   "neutral default. Use Queue_Resolution (ASA_QUEUE) for dedup."},
  {"User_Cost_Temperature",
   "USER_INITIAL_COST_TEMP is compiled out: it replaces the "
   "Number_Cost_Samples initial cost-temperature sampling (not "
   "trajectory-neutral)."},
  {"Asa_Recursive_Level",
   "recursive/self-optimizing ASA (SELF_OPTIMIZE) is not built."},
  {"Asymp_Exp_Param",
   "USER_ACCEPT_ASYMP_EXP is compiled out: it changes the acceptance "
   "law globally."},
  {"Fit_Local",
   "FITLOC lives in ASA's un-vendored user module (asa_usr.c)."},
  {"Iter_Max",
   "FITLOC lives in ASA's un-vendored user module (asa_usr.c)."},
  {"Penalty",
   "FITLOC lives in ASA's un-vendored user module (asa_usr.c)."},
  {"NoOfSamples",
   "ASA_FUZZY lives in ASA's un-vendored user module (asa_usr.c)."},
  {"Random_Array_Dim",
   "ASA_SAVE checkpointing is compiled out (HPO runs are not resumable)."},
  {"Bias_Acceptance",
   "ASA_SAMPLE (importance sampling) is not built."},
  {"Limit_Weights",
   "ASA_SAMPLE (importance sampling) is not built."},
};

const Desc* find_desc(const std::string& name)
{
    for (const auto& d : kDescs)
        if (name == d.name) return &d;
    return nullptr;
}

std::string lower(std::string s)
{
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    return s;
}

[[noreturn]] void unknown_key_error(const std::string& key)
{
    std::ostringstream oss;
    oss << "ASA:ASA: unknown PARAM '" << key << "'.";
    const std::string kl = lower(key);
    for (const auto& d : kDescs)
        if (kl == lower(d.name)) {
            oss << " Did you mean '" << d.name
                << "'? ASA option names are case-sensitive.";
            throw std::runtime_error(oss.str());
        }
    for (const auto& [name, why] : kExcluded)
        if (kl == lower(name)) {
            oss << " '" << name << "' is not available: " << why;
            throw std::runtime_error(oss.str());
        }
    oss << " Accepted keys:";
    for (const auto& d : kDescs) oss << " " << d.name;
    throw std::runtime_error(oss.str());
}

/* Split "Name[3]" -> {"Name", 3}; plain "Name" -> {"Name", -1}.
 * The index is 1-based and means the same dimension as the `dim` column
 * of hpo_search_space.tsv and the `#` column of the journal's SEARCH
 * SPACE table, so a dimension number a user reads is the one they type.
 * Index 0 is rejected by the caller (it was the first dimension in
 * Tadah 1.4 and earlier). */
std::pair<std::string, long> split_indexed(const std::string& key)
{
    const auto lb = key.find('[');
    if (lb == std::string::npos) return {key, -1};
    if (key.back() != ']' || lb + 1 >= key.size() - 1u + 1u)
        throw std::runtime_error(
            "ASA:ASA: malformed indexed PARAM '" + key +
            "' (expected Name[<1-based index>])");
    const std::string base = key.substr(0, lb);
    const std::string idx  = key.substr(lb + 1, key.size() - lb - 2);
    if (idx.empty() ||
        !std::all_of(idx.begin(), idx.end(),
                     [](unsigned char c) { return std::isdigit(c); }))
        throw std::runtime_error(
            "ASA:ASA: malformed index in PARAM '" + key +
            "' (expected a positive integer)");
    /* Digits only, so the one failure left is an index too large for a
     * long (std::stol threw a bare "stol"). */
    return {base, static_cast<long>(parse_uint(
                      idx, "ASA:ASA: the index in PARAM '" + key + "'", 0,
                      static_cast<std::uint64_t>(
                          std::numeric_limits<long>::max())))};
}

void check_value(const Desc& d, const std::string& key, double v)
{
    if (!std::isfinite(v))
        throw std::runtime_error("ASA:ASA: PARAM " + key +
                                 " must be finite");
    if (d.kind != Kind::Real && v != std::floor(v))
        throw std::runtime_error("ASA:ASA: PARAM " + key +
                                 " must be an integer value");
    if (!std::isnan(d.min) && v < d.min)
        throw std::runtime_error("ASA:ASA: PARAM " + key + " must be >= " +
                                 std::to_string(d.min));
    if (!std::isnan(d.max) && v > d.max)
        throw std::runtime_error("ASA:ASA: PARAM " + key + " must be <= " +
                                 std::to_string(d.max));
}

}  // namespace

AsaOptions parse_asa_options(const OptimizerSpec& sp, std::size_t dim)
{
    /* ---- generic spec knobs ASA does not implement ----------------- */
    if (sp.stop.ftol_rel || sp.stop.ftol_abs)
        throw std::runtime_error(
            "ASA:ASA does not implement FTOL_REL/FTOL_ABS; use ASA's own "
            "stops: PARAM Cost_Precision + Maximum_Cost_Repeat (plateau) "
            "or Accepted_To_Generated_Ratio");
    if (sp.stop.xtol_rel || sp.stop.xtol_abs)
        throw std::runtime_error(
            "ASA:ASA does not implement XTOL_REL/XTOL_ABS");
    if (sp.pop)
        throw std::runtime_error(
            "ASA:ASA does not accept POPULATION (single-chain optimizer; "
            "block parallelism is PARAM Gener_Block)");
    if (sp.step)
        throw std::runtime_error(
            "ASA:ASA does not accept STEP: the width of the generating "
            "distribution follows the parameter temperature of each "
            "dimension (Initial_Parameter_Temperature, "
            "Temperature_Ratio_Scale)");
    if (sp.grad_strategy)
        throw std::runtime_error(
            "ASA:ASA does not accept GRAD_STRATEGY (derivative-free; "
            "reanneal tangents use PARAM Delta_X finite differences)");
    if (sp.vector_storage)
        throw std::runtime_error("ASA:ASA does not accept VECTOR_ARR");
    if (sp.num_threads && *sp.num_threads == 0)
        throw std::runtime_error("ASA:ASA: THREADS must be >= 1");
    if (sp.stop.maxtime && !(*sp.stop.maxtime > 0.0))
        throw std::runtime_error("ASA:ASA: MAXTIME must be > 0 seconds");

    AsaOptions o;
    /* ASA starts at the first starting point; with none (a WARM seed the
     * INIT block dropped) it draws its own. */
    o.user_initial_parameters = sp.optim_init.empty() ? 0 : 1;
    o.user_quench_param_scale.assign(dim, 1.0);
    o.user_temperature_ratio.assign(dim, 1.0);
    o.multi_grid.assign(dim, 0.0);
    o.queue_resolution.assign(dim, 0.0);

    /* ---- two passes: scalars/unindexed fills, then [i] overrides --- */
    bool have_gener_block_max = false;
    std::vector<std::pair<std::string, double>> indexed;

    auto set_scalar = [&](const Desc& d, const std::string& key, double v) {
        check_value(d, key, v);
        const std::string n = d.name;
        if      (n == "Limit_Acceptances")       o.limit_acceptances = (long)v;
        else if (n == "Limit_Generated")         o.limit_generated = (long)v;
        else if (n == "Limit_Invalid_Generated_States")
            o.limit_invalid_generated_states = (int)v;
        else if (n == "Accepted_To_Generated_Ratio") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM Accepted_To_Generated_Ratio must be > 0");
            o.accepted_to_generated_ratio = v;
        }
        else if (n == "Cost_Precision") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM Cost_Precision must be > 0");
            o.cost_precision = v;
        }
        else if (n == "Maximum_Cost_Repeat")     o.maximum_cost_repeat = (int)v;
        else if (n == "Number_Cost_Samples") {
            if (v == 0.0 || v == -1.0) throw std::runtime_error(
                "ASA:ASA: PARAM Number_Cost_Samples must not be 0 or -1 "
                "(>0 = mean |cost| sample, <-1 = deviation sample)");
            o.number_cost_samples = (int)v;
        }
        else if (n == "Temperature_Ratio_Scale") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM Temperature_Ratio_Scale must be > 0");
            o.temperature_ratio_scale = v;
        }
        else if (n == "Cost_Parameter_Scale_Ratio") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM Cost_Parameter_Scale_Ratio must be > 0");
            o.cost_parameter_scale_ratio = v;
        }
        else if (n == "Temperature_Anneal_Scale") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM Temperature_Anneal_Scale must be > 0");
            o.temperature_anneal_scale = v;
        }
        else if (n == "Sequential_Parameters") {
            if (v >= (double)dim) throw std::runtime_error(
                "ASA:ASA: PARAM Sequential_Parameters must be < the "
                "number of OPTIM dimensions (" + std::to_string(dim) + ")");
            o.sequential_parameters = (long)v;
        }
        else if (n == "Initial_Parameter_Temperature") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM Initial_Parameter_Temperature must be > 0");
            o.initial_parameter_temperature = v;
        }
        else if (n == "Acceptance_Frequency_Modulus")
            o.acceptance_frequency_modulus = (int)v;
        else if (n == "Generated_Frequency_Modulus")
            o.generated_frequency_modulus = (int)v;
        else if (n == "Reanneal_Cost") {
            if (v == -1.0) throw std::runtime_error(
                "ASA:ASA: PARAM Reanneal_Cost == -1 is invalid upstream "
                "(use 0 = off, 1 = default, >1 or <-1 = sample variants)");
            o.reanneal_cost = (int)v;
        }
        else if (n == "Reanneal_Parameters")     o.reanneal_parameters = (int)v;
        else if (n == "Delta_X")                 o.delta_x = v;  // >= 0: kDescs
        else if (n == "User_Tangents") {
            if (v != 0.0) throw std::runtime_error(
                "ASA:ASA: PARAM User_Tangents 1 is unsupported: the HPO "
                "host cannot supply analytic tangents (reanneal uses "
                "Delta_X finite differences)");
        }
        else if (n == "Curvature_0")             o.curvature_0 = (int)v;
        else if (n == "User_Quench_Cost_Scale") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM User_Quench_Cost_Scale must be > 0");
            o.user_quench_cost_scale = v;
        }
        else if (n == "Multi_Number")            o.multi_number = (int)v;
        else if (n == "Multi_Specify")           o.multi_specify = (int)v;
        else if (n == "Queue_Size")              o.queue_size = (long)v;
        else if (n == "Gener_Block")             o.gener_block = (long)v;
        else if (n == "Gener_Block_Max") {
            have_gener_block_max = true;
            o.gener_block_max = (long)v;
        }
        else if (n == "Gener_Mov_Avr")           o.gener_mov_avr = (int)v;
        /* per-dim uniform fills */
        else if (n == "User_Quench_Param_Scale") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM User_Quench_Param_Scale must be > 0");
            o.user_quench_param_scale.assign(dim, v);
        }
        else if (n == "User_Temperature_Ratio") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM User_Temperature_Ratio must be > 0");
            o.user_temperature_ratio.assign(dim, v);
        }
        else if (n == "User_Delta_Parameter") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM User_Delta_Parameter must be > 0");
            o.user_delta_parameter.assign(dim, v);
        }
        else if (n == "User_Parameter_Temperature") {
            if (!(v > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM User_Parameter_Temperature must be > 0");
            o.user_parameter_temperature.assign(dim, v);
        }
        else if (n == "Multi_Grid")        o.multi_grid.assign(dim, v);
        else if (n == "Queue_Resolution")  o.queue_resolution.assign(dim, v);
    };

    /* A word (OptimizerSpec::param_raw): refused under a name ASA does not
     * take as a number is, and under one of its options, which take
     * numbers. */
    for (const auto& [key, word] : sp.param_raw) {
        const auto [base, idx] = split_indexed(key);
        (void)idx;
        if (!find_desc(base)) unknown_key_error(key);
        throw std::runtime_error("ASA:ASA: PARAM " + key +
                                 " takes a number (got '" + word + "')");
    }
    for (const auto& [key, val] : sp.param) {
        const auto [base, idx] = split_indexed(key);
        const Desc* d = find_desc(base);
        if (!d) unknown_key_error(key);
        if (idx >= 0) {
            if (!d->per_dim)
                throw std::runtime_error(
                    "ASA:ASA: PARAM " + base +
                    " is a scalar option (indexed form not accepted)");
            if (idx == 0)
                throw std::runtime_error(
                    "ASA:ASA: PARAM " + key + " index 0 is not a dimension: "
                    "per-dimension indices are 1-based and match the dim "
                    "column of hpo_search_space.tsv. They were 0-based in "
                    "Tadah 1.4 and earlier, so every old index shifts up by "
                    "one ([0] is now [1])");
            if ((std::size_t)idx > dim)
                throw std::runtime_error(
                    "ASA:ASA: PARAM " + key + " index out of range (dim=" +
                    std::to_string(dim) + ", 1-based)");
            check_value(*d, key, val);
            indexed.emplace_back(key, val);
        } else {
            set_scalar(*d, key, val);
        }
    }

    /* mirrors: fill from scalars unless the uniform form set them */
    if (o.user_delta_parameter.empty())
        o.user_delta_parameter.assign(dim, o.delta_x);
    if (o.user_parameter_temperature.empty())
        o.user_parameter_temperature.assign(
            dim, o.initial_parameter_temperature);

    for (const auto& [key, val] : indexed) {
        const auto [base, idx] = split_indexed(key);
        auto put = [&](std::vector<double>& v) {
            v[(std::size_t)(idx - 1)] = val;   /* [i] is 1-based */
        };
        if      (base == "User_Quench_Param_Scale") {
            if (!(val > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM " + key + " must be > 0");
            put(o.user_quench_param_scale);
        }
        else if (base == "User_Temperature_Ratio") {
            if (!(val > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM " + key + " must be > 0");
            put(o.user_temperature_ratio);
        }
        else if (base == "User_Delta_Parameter") {
            if (!(val > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM " + key + " must be > 0");
            put(o.user_delta_parameter);
        }
        else if (base == "User_Parameter_Temperature") {
            if (!(val > 0.0)) throw std::runtime_error(
                "ASA:ASA: PARAM " + key + " must be > 0");
            put(o.user_parameter_temperature);
        }
        else if (base == "Multi_Grid")       put(o.multi_grid);
        else if (base == "Queue_Resolution") put(o.queue_resolution);
    }

    /* ---- cross-checks ---------------------------------------------- */
    if (!have_gener_block_max)
        o.gener_block_max = o.gener_block;
    if (o.gener_block_max < o.gener_block)
        throw std::runtime_error(
            "ASA:ASA: PARAM Gener_Block_Max must be >= Gener_Block "
            "(buffers are sized once by the cap)");
    /* The Queue_Resolution hint is the run's (announce_asa_options): this
     * parse is also the startup check of the spec (backend_rules.h),
     * which journals nothing. */
    return o;
}

void announce_asa_options(const AsaOptions& o)
{
    if (o.queue_size > 0 &&
        std::all_of(o.queue_resolution.begin(), o.queue_resolution.end(),
                    [](double r) { return r == 0.0; }))
        HPOLog::instance().hint(
            "ASA: Queue_Size > 0 with Queue_Resolution 0 caches only "
            "near-exact revisits; set PARAM Queue_Resolution to the "
            "scale of each dimension below which two points are 'the same'");
}

const char* asa_exit_name(int exit_status)
{
    switch (exit_status) {
        case 0:  return "NORMAL_EXIT";
        case 1:  return "P_TEMP_TOO_SMALL";
        case 2:  return "C_TEMP_TOO_SMALL";
        case 3:  return "COST_REPEATING";
        case 4:  return "TOO_MANY_INVALID_STATES";
        case 5:  return "IMMEDIATE_EXIT";
        case 7:  return "INVALID_USER_INPUT";
        case 8:  return "INVALID_COST_FUNCTION";
        case 9:  return "INVALID_COST_FUNCTION_DERIV";
        case -1: return "CALLOC_FAILED";
        default: return "UNKNOWN_EXIT_CODE";
    }
}

void print_asa_param_table(std::ostream& os)
{
    for (const auto& d : kDescs) {
        os << "  PARAM " << d.name;
        if (d.per_dim) os << "[/i]";
        os << "  (default " << d.def_str << ")  " << d.doc << "\n";
    }
}

}  // namespace tadah::hpo
