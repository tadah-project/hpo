#include <tadah/hpo/backend_rules.h>

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace tadah::hpo {

namespace {

std::string backend_name(const OptimizerSpec& s)
{
    return s.lib + ":" + s.algo;
}

/* NLopt counts evaluations in an int (nlopt::opt::set_maxeval): a larger
 * MAXEVAL wrapped (4294967299 ran 3 evaluations, 2147483648 became a
 * negative budget, which NLopt reads as no limit). HPOTargets bounds
 * MAXEVAL at its line; a spec built another way is refused here. */
void require_nlopt_maxeval(const OptimizerSpec& sp)
{
    constexpr auto kIntMax =
        static_cast<std::uint64_t>(std::numeric_limits<int>::max());
    if (sp.stop.maxeval && *sp.stop.maxeval > kIntMax)
        throw std::invalid_argument(
            "NLopt spec with MAXEVAL " + std::to_string(*sp.stop.maxeval) +
            ": NLopt counts evaluations in an int (at most 2147483647)");
}

/* "A, B or C" ("A" for one item). */
std::string or_list(const std::vector<std::string>& items)
{
    std::string out;
    for (std::size_t i = 0; i < items.size(); ++i) {
        if (i) out += (i + 1 == items.size()) ? " or " : ", ";
        out += items[i];
    }
    return out;
}

/* The tolerances that end a run of `caps`. */
std::vector<std::string> honoured_tolerances(const OptimizerCaps& caps)
{
    std::vector<std::string> t;
    if (caps.supports_ftol_rel) t.push_back("FTOL_REL");
    if (caps.supports_ftol_abs) t.push_back("FTOL_ABS");
    if (caps.supports_xtol_rel) t.push_back("XTOL_REL");
    if (caps.supports_xtol_abs) t.push_back("XTOL_ABS");
    return t;
}

/* The stops to set, as a message lists them: `keys` (MAXEVAL, MAXTIME,
 * STOPVAL, as they apply) and the tolerances of `caps`. */
std::string stop_list(std::vector<std::string> keys, const OptimizerCaps& caps,
                      bool say_none)
{
    const std::vector<std::string> tol = honoured_tolerances(caps);
    if (tol.empty())
        return or_list(keys) + (say_none ? " (it honours no tolerance)" : "");
    std::string joined;
    for (const auto& k : tol) joined += (joined.empty() ? "" : ", ") + k;
    keys.push_back("a positive tolerance it honours (" + joined + ")");
    return or_list(keys);
}

/* The caps' note on how its tolerances end it, as a message's last
 * clause ("" when it has none). */
std::string tolerance_note(const OptimizerCaps& caps)
{
    if (!caps.tolerance_note[0] || honoured_tolerances(caps).empty())
        return "";
    return std::string("; ") + caps.tolerance_note;
}

/* MAXEVAL, MAXTIME and STOPVAL, as `caps` honours them. */
std::vector<std::string> budget_keys(const OptimizerCaps& caps, bool stopval)
{
    std::vector<std::string> k;
    if (caps.supports_maxeval) k.push_back("MAXEVAL");
    if (caps.supports_maxtime) k.push_back("MAXTIME");
    if (stopval && caps.supports_stopval) k.push_back("STOPVAL");
    return k;
}

/* `spec` sets a stop that ends a run of `caps`, or the RESTART stagnation
 * stop bounds the run. A tolerance counts only above 0: NLopt switches a
 * tolerance of 0 or below off (its stop tests compare a change with it),
 * and ANNEAL_PT's best-stall rule arms only above 0, so FTOL_REL 0 -- a
 * common way to write "no tolerance" -- ends nothing. MAXEVAL and MAXTIME
 * are positive by their grammar, and any STOPVAL is a target. */
bool has_stop(const OptimizerSpec& spec, const OptimizerCaps& caps,
              RestartStop restart)
{
    if (restart == RestartStop::On) return true;
    const StopCriteria& st = spec.stop;
    const auto positive = [](const std::optional<double>& v) {
        return v && *v > 0.0;
    };
    return (st.maxeval && caps.supports_maxeval) ||
           (st.maxtime && caps.supports_maxtime) ||
           (st.stopval && caps.supports_stopval) ||
           (positive(st.ftol_rel) && caps.supports_ftol_rel) ||
           (positive(st.ftol_abs) && caps.supports_ftol_abs) ||
           (positive(st.xtol_rel) && caps.supports_xtol_rel) ||
           (positive(st.xtol_abs) && caps.supports_xtol_abs);
}

/* TADAH:RANDOM and DLIB:GFS read no PARAM, TADAH:GRID only GRID_POINTS
 * (validate_grid_spec): a PARAM they do not read, numeric or a word, is
 * refused with the list of those they take. */
void reject_unknown_params(const OptimizerSpec& spec,
                           const std::vector<std::string>& takes)
{
    const auto known = [&](const std::string& k) {
        return std::find(takes.begin(), takes.end(), k) != takes.end();
    };
    std::string takes_list;
    for (const auto& k : takes) takes_list += (takes_list.empty() ? "" : ", ") + k;
    if (takes_list.empty()) takes_list = "none";
    const auto refuse = [&](const std::string& key) {
        throw std::runtime_error(backend_name(spec) + ": unknown PARAM '" +
                                 key + "' (it takes " + takes_list + ")");
    };
    for (const auto& kv : spec.param)
        if (!known(kv.first)) refuse(kv.first);
    for (const auto& kv : spec.param_raw)
        if (!known(kv.first)) refuse(kv.first);
}

/* Whether a run of `spec` tests a tolerance its caps mark with `supports`.
 * NLopt's MLSL and AUGLAG (accepts_inner) hand the block's tolerances only
 * to the default local optimiser NLopt creates without an INNER block
 * (optimize.c), which tests them; with an INNER block they reach no search
 * (mlsl.c tests none, auglag.c only names the ending its INNER search
 * decides). */
bool tolerance_tested(bool supports, const OptimizerSpec& spec,
                      const OptimizerCaps& caps)
{
    if (!caps.accepts_inner) return supports;
    if (spec.inner) return false;
    return supports || caps.default_inner != nullptr;
}

/* The backends whose caps mark `reads`, as a refusal lists them: all of
 * them, sorted. */
std::string readers(CapsPredicate reads)
{
    const auto caps = filter_optimizer_caps(reads);
    return format_caps_list(caps, caps.size());
}

/* Refuses an attribute the run of `spec` never reads (backend_rules.h):
 * a tolerance it does not test, then POPULATION, VECTOR_ARR and SEED,
 * each from the caps. */
void reject_unread_attributes(const OptimizerSpec& spec,
                              const OptimizerCaps& caps, SpecRole role)
{
    const StopCriteria& st = spec.stop;
    for (const auto& [set, supports, key] :
         {std::tuple<bool, bool, const char*>{st.ftol_rel.has_value(),
                                              caps.supports_ftol_rel,
                                              "FTOL_REL"},
          {st.ftol_abs.has_value(),
           caps.supports_ftol_abs || caps.reads_ftol_abs_as_setting,
           "FTOL_ABS"},
          {st.xtol_rel.has_value(), caps.supports_xtol_rel, "XTOL_REL"},
          {st.xtol_abs.has_value(), caps.supports_xtol_abs, "XTOL_ABS"}}) {
        if (!set || tolerance_tested(supports, spec, caps)) continue;
        std::string msg = backend_name(spec) + " does not support " + key;
        if (caps.accepts_inner && spec.inner) msg += " with an INNER block";
        if (caps.untested_tolerance_note[0])
            msg += std::string(": ") + caps.untested_tolerance_note;
        if (caps.accepts_inner)
            msg += "; set it in the INNER block, whose local search tests it";
        else if (role == SpecRole::Inner)
            msg += "; stop it with " +
                   stop_list(budget_keys(caps, false), caps, false) +
                   " in the INNER block" +
                   (caps.supports_stopval
                        ? ", or STOPVAL in the enclosing block" : "");
        else
            msg += "; stop it with " +
                   stop_list(budget_keys(caps, true), caps, false);
        throw std::runtime_error(msg);
    }
    if (spec.pop && !caps.reads_population)
        throw std::runtime_error(
            backend_name(spec) + " does not use POPULATION; remove it (it is "
            "read by " +
            readers([](const OptimizerCaps& c) { return c.reads_population; }) +
            ")");
    if (spec.vector_storage && !caps.reads_vector_storage)
        throw std::runtime_error(
            backend_name(spec) + " does not use VECTOR_ARR; remove it (it is "
            "read by " +
            readers([](const OptimizerCaps& c) {
                return c.reads_vector_storage;
            }) +
            ")");
    if (spec.seed && !caps.reads_seed && !caps.warns_unread_seed)
        throw std::runtime_error(backend_name(spec) +
                                 " does not use SEED: it draws no random "
                                 "numbers; remove it");
}

}  // namespace

RestartStop restart_stop_of(const OptimizerSpec& root)
{
    return root.restart.active() ? RestartStop::On : RestartStop::Off;
}

bool sets_own_stop(const OptimizerSpec& spec)
{
    return has_stop(spec, optimizer_caps(spec), RestartStop::Off);
}

void require_stop(const OptimizerSpec& spec, RestartStop restart)
{
    const OptimizerCaps& caps = optimizer_caps(spec);
    if (!caps.needs_stop || has_stop(spec, caps, restart)) return;
    throw std::runtime_error(
        backend_name(spec) + " needs a stop: " + caps.stop_note + "; set " +
        stop_list(budget_keys(caps, true), caps, true) +
        tolerance_note(caps));
}

void require_inner_run_stop(const OptimizerSpec& parent, RestartStop restart)
{
    const std::optional<OptimizerSpec> run = inner_run_spec(parent);
    if (!run) return;
    const OptimizerCaps& caps = optimizer_caps(*run);
    if (!caps.needs_stop || has_stop(*run, caps, restart)) return;
    const std::string outer = backend_name(parent);
    const std::string inner = backend_name(*run);
    const OptimizerCaps& outer_caps = optimizer_caps(parent);
    const std::string handed_on =
        or_list(budget_keys(outer_caps, true)) + " in the " + outer +
        " block, which NLopt hands on to it";
    if (parent.inner)
        throw std::runtime_error(
            "the INNER search of " + outer + " (" + inner + ") needs a stop: " +
            caps.stop_note + "; set " +
            stop_list(budget_keys(caps, false), caps, false) +
            " in the INNER block, or " + handed_on + tolerance_note(caps));
    /* NLopt's default local optimiser: it takes the block's tolerances
     * too. */
    throw std::runtime_error(
        "the local search of " + outer + " (NLopt's default " + inner +
        ", as the block has no INNER block) needs a stop: " + caps.stop_note +
        "; set " + stop_list(budget_keys(caps, true), caps, false) +
        " in the " + outer + " block, which NLopt hands on to it" +
        tolerance_note(caps));
}

void validate_random_spec(const OptimizerSpec& spec, RestartStop restart)
{
    reject_unknown_params(spec, {});
    reject_unread_attributes(spec, optimizer_caps(spec), SpecRole::Phase);
    require_stop(spec, restart);
}

void validate_dlib_gfs_spec(const OptimizerSpec& spec, RestartStop restart)
{
    reject_unknown_params(spec, {});
    reject_unread_attributes(spec, optimizer_caps(spec), SpecRole::Phase);
    require_stop(spec, restart);
}

unsigned validate_grid_spec(const OptimizerSpec& sp)
{
    /* The historic `STEP <n>` overload is rejected (the scale validator
     * says so first, StepUse::GridCount): STEP < 1 used to truncate to 0
     * points and loop forever at the low corner, STEP 1 divided by zero. */
    if (sp.step)
        throw std::runtime_error(
            "TADAH:GRID: STEP is not the grid resolution; use PARAM GRID_POINTS "
            "<n> (integer >= 2, default 20 points per dimension)");
    unsigned steps = 20;
    if (auto it = sp.param.find("GRID_POINTS"); it != sp.param.end()) {
        const double v = it->second;
        if (!std::isfinite(v) || v < 2.0 || v != std::floor(v) ||
            v > static_cast<double>(std::numeric_limits<unsigned>::max()))
            throw std::runtime_error(
                "TADAH:GRID PARAM GRID_POINTS must be an integer >= 2 (got " +
                std::to_string(v) + "; a 1-point grid divides by zero)");
        steps = static_cast<unsigned>(v);
    }
    reject_unknown_params(sp, {"GRID_POINTS"});
    /* A word where a number belongs (PARAM GRID_POINTS AUTO) was stored as
     * a word and ignored. */
    if (auto it = sp.param_raw.find("GRID_POINTS"); it != sp.param_raw.end())
        throw std::runtime_error(
            "TADAH:GRID PARAM GRID_POINTS must be an integer >= 2 (got '" +
            it->second + "')");
    reject_unread_attributes(sp, optimizer_caps(sp), SpecRole::Phase);
    return steps;
}

void validate_nlopt_spec(const OptimizerSpec& sp, RestartStop restart,
                         SpecRole role)
{
    if (sp.inner && (sp.inner->seed || sp.inner->init_strategy_spec.seed))
        throw std::invalid_argument(
            "NLopt INNER spec with a seed: NLopt draws the INNER search from "
            "the enclosing run's random stream; set SEED on the enclosing "
            "block");
    /* The parser rejects GRAD_STRATEGY inside INNER; a spec built outside
     * it must not carry one either, since it would not be used. */
    if (sp.inner && sp.inner->grad_strategy)
        throw std::invalid_argument(
            "NLopt INNER spec with a GRAD_STRATEGY: the enclosing block's "
            "GRAD_STRATEGY applies to its local search; set GRAD_STRATEGY on "
            "the enclosing block");
    /* The parser rejects STOPVAL inside INNER: mlsl.c and auglag.c set the
     * local search's stopval to the enclosing run's (inner_run_spec()). */
    if (sp.inner && sp.inner->stop.stopval)
        throw std::invalid_argument(
            "NLopt INNER spec with a STOPVAL: NLopt's MLSL and AUGLAG replace "
            "the INNER search's STOPVAL with the enclosing block's; set "
            "STOPVAL on the enclosing block");
    require_nlopt_maxeval(sp);
    if (sp.inner) require_nlopt_maxeval(*sp.inner);
    reject_unread_attributes(sp, optimizer_caps(sp), role);
    require_stop(sp, restart);
    require_inner_run_stop(sp, restart);
}

void validate_asa_spec(const OptimizerSpec& sp)
{
    const std::size_t dim = sp.low.size();
    if (dim == 0)
        throw std::runtime_error(
            "ASA:ASA requires at least one OPTIM dimension");
    (void)parse_asa_options(sp, dim);
}

void validate_backend(const OptimizerSpec& spec, RestartStop restart,
                      SpecRole role)
{
    if (spec.lib == "NOT SET" || spec.algo == "NOT SET") return;
    /* The same order as run_inner(): the deferred CERES least-squares
     * algorithms get their tailored message before the caps lookup
     * rejects an unknown LIB:ALGO. */
    reject_deferred_ceres_nlls(spec);
    (void)optimizer_caps(spec);

    const std::string& lib  = spec.lib;
    const std::string& algo = spec.algo;
    if (lib == "TADAH") {
        if (algo == "RANDOM") {
            validate_random_spec(spec, restart);
        } else if (algo == "GRID") {
            (void)validate_grid_spec(spec);
        } else if (algo == "ANNEAL") {
            (void)anneal::Knobs::parse(spec, anneal::AlgoKind::Serial);
        } else if (algo == "ANNEAL_PT") {
            const auto kb = anneal::Knobs::parse(spec, anneal::AlgoKind::PT);
            (void)anneal::pt_replica_count(spec, kb);
            require_stop(spec, restart);
        } else if (algo == "ANNEAL_PA") {
            (void)anneal::Knobs::parse(spec, anneal::AlgoKind::PA);
        }
    } else if (lib == "DLIB") {
        if (algo == "GFS")       validate_dlib_gfs_spec(spec, restart);
        else if (algo == "BFGS") detail::validate_dlib_bfgs_spec(spec);
    } else if (lib == "NLOPT") {
        validate_nlopt_spec(spec, restart, role);
    } else if (lib == "ASA") {
        validate_asa_spec(spec);
    } else if (lib == "CERES") {
#ifndef TADAH_USE_CERES
        throw std::runtime_error(
            "CERES support is not built; reconfigure with "
            "-DTADAH_ENABLE_CERES=ON");
#else
        if (algo == "LBFGS")                 detail::validate_ceres_lbfgs_spec(spec);
        else if (algo == "WOLFE")            detail::validate_ceres_wolfe_spec(spec);
        else if (algo == "NLCG")             detail::validate_ceres_nlcg_spec(spec);
        else if (algo == "STEEPEST_DESCENT") detail::validate_ceres_steepest_spec(spec);
#endif
    }
    /* Every backend that needs a stop has checked it above; this covers
     * one registered later without a validator of its own. */
    require_stop(spec, restart);
}

void validate_phase_backends(const OptimizerSpec& root)
{
    /* RESTART wraps every phase (RestartPolicy, run_policy.h). */
    const RestartStop restart = restart_stop_of(root);
    for (const PhaseSpec& phase : strategy_phase_specs(root)) {
        try {
            validate_backend(phase.spec, restart,
                             phase.inner ? SpecRole::Inner : SpecRole::Phase);
        } catch (const std::exception& e) {
            throw std::runtime_error(phase.name + ": " + e.what());
        }
    }
}

}  // namespace tadah::hpo
