#include <tadah/hpo/backend_rules.h>

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/strategy_rules.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tadah::hpo {

namespace {

std::string backend_name(const OptimizerSpec& s)
{
    return s.lib + ":" + s.algo;
}

/* NLOPT, DLIB and CERES cannot search a space without a dimension: NLopt
 * refuses the call ("NULL args to nlopt_optimize_"), dlib's GFS asserts
 * ("The vectors can't be empty"), DLIB:BFGS and the CERES drivers check
 * their bounds when they start. The TADAH optimisers evaluate the INIT
 * point; ASA refuses it in its own validator (validate_asa_spec). */
void require_search_dimensions(const OptimizerSpec& spec)
{
    if (!spec.low.empty()) return;
    if (spec.lib != "NLOPT" && spec.lib != "DLIB" && spec.lib != "CERES")
        return;
    throw std::runtime_error(
        backend_name(spec) + " cannot optimise a zero-dimensional search "
        "space (there is no OPTIM line; the failure would only surface once "
        "the optimiser starts). Add at least one OPTIM line, or use a "
        "TADAH-native optimiser, which evaluates the INIT start point once.");
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

/* MAXEVAL, MAXTIME and STOPVAL, as `caps` (the block they are set in)
 * honours them, that are a stop for a run of `run`: only MAXEVAL when `run`
 * ignores the forced stop (OptimizerCaps::ignores_forced_stop), which only
 * NLopt's own MAXEVAL ends promptly once HPO stops it. */
std::vector<std::string> budget_keys(const OptimizerCaps& caps, bool stopval,
                                     const OptimizerCaps& run)
{
    std::vector<std::string> k;
    if (caps.supports_maxeval) k.push_back("MAXEVAL");
    if (run.ignores_forced_stop) return k;
    if (caps.supports_maxtime) k.push_back("MAXTIME");
    if (stopval && caps.supports_stopval) k.push_back("STOPVAL");
    return k;
}

/* `spec` sets a stop that ends a run of `caps`, or the RESTART stagnation
 * stop bounds the run. A tolerance counts only above 0: NLopt switches a
 * tolerance of 0 or below off (its stop tests compare a change with it),
 * and ANNEAL_PT's best-stall rule arms only above 0, so FTOL_REL 0 -- a
 * common way to write "no tolerance" -- ends nothing. MAXEVAL and MAXTIME
 * are positive by their grammar, and any STOPVAL is a target. A backend
 * that ignores the forced stop (OptimizerCaps::ignores_forced_stop) needs
 * MAXEVAL: HPO ends a search by that stop (a signal, an error, the RESTART
 * stagnation stop, a withheld loss at STOPVAL), and such a backend calls on,
 * each call refused (a few microseconds), until NLopt's own count ends it;
 * its clock would end it only once the rest of MAXTIME had passed. */
bool has_stop(const OptimizerSpec& spec, const OptimizerCaps& caps,
              RestartStop restart)
{
    const StopCriteria& st = spec.stop;
    if (caps.ignores_forced_stop)
        return st.maxeval && caps.supports_maxeval;
    if (restart == RestartStop::On) return true;
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

/* A tolerance key: its value in StopCriteria and the caps flag that says a
 * run tests it. */
struct ToleranceKey {
    const char* key;
    std::optional<double> StopCriteria::*value;
    bool OptimizerCaps::*supports;
};

constexpr ToleranceKey kToleranceKeys[] = {
    {"FTOL_REL", &StopCriteria::ftol_rel, &OptimizerCaps::supports_ftol_rel},
    {"FTOL_ABS", &StopCriteria::ftol_abs, &OptimizerCaps::supports_ftol_abs},
    {"XTOL_REL", &StopCriteria::xtol_rel, &OptimizerCaps::supports_xtol_rel},
    {"XTOL_ABS", &StopCriteria::xtol_abs, &OptimizerCaps::supports_xtol_abs}};

/* Whether a run of `spec` reads the tolerance `t`: one it tests
 * (tolerance_tested), or DLIB:GFS's FTOL_ABS, its solver epsilon. */
bool reads_tolerance(const ToleranceKey& t, const OptimizerSpec& spec,
                     const OptimizerCaps& caps)
{
    const bool supports =
        caps.*t.supports ||
        (t.value == &StopCriteria::ftol_abs && caps.reads_ftol_abs_as_setting);
    return tolerance_tested(supports, spec, caps);
}

/* Where to set the tolerance `t` that an MLSL or AUGLAG `spec` refuses:
 * in the INNER block, when its search tests it; otherwise the stops that
 * search honours, as require_inner_run_stop() names them (an INNER GN_ESCH
 * tests no tolerance either). Without an INNER block only G_MLSL and
 * G_MLSL_LDS refuse one (tolerance_tested: they have no default local
 * optimiser), a spec require_inner_backend() refuses: the block it needs. */
std::string inner_tolerance_advice(const ToleranceKey& t,
                                   const OptimizerSpec& spec,
                                   const OptimizerCaps& caps)
{
    if (!spec.inner)
        return "; " + backend_name(spec) + " needs an INNER block, and only "
               "that block's search can test it";
    const OptimizerSpec& inner = *spec.inner;
    const OptimizerCaps& inner_caps = optimizer_caps(inner);
    if (reads_tolerance(t, inner, inner_caps))
        return "; set it in the INNER block, whose local search tests it";
    return "; the INNER block's " + backend_name(inner) +
           " does not test it either: stop that search with " +
           stop_list(budget_keys(inner_caps, false, inner_caps), inner_caps,
                     false) +
           " in the INNER block, or " +
           or_list(budget_keys(caps, true, inner_caps)) +
           " in the " + backend_name(spec) +
           " block, which NLopt hands on to it";
}

/* Refuses an attribute the run of `spec` never reads (backend_rules.h):
 * a tolerance it does not test, then POPULATION, VECTOR_ARR and SEED,
 * each from the caps. */
void reject_unread_attributes(const OptimizerSpec& spec,
                              const OptimizerCaps& caps, SpecRole role)
{
    for (const ToleranceKey& t : kToleranceKeys) {
        if (!(spec.stop.*t.value) || reads_tolerance(t, spec, caps)) continue;
        const std::string key = t.key;
        std::string msg = backend_name(spec) + " does not support " + key;
        if (caps.accepts_inner && spec.inner) msg += " with an INNER block";
        if (caps.untested_tolerance_note[0])
            msg += std::string(": ") + caps.untested_tolerance_note;
        /* An MLSL or AUGLAG: whether its INNER search would test it. */
        if (caps.accepts_inner)
            msg += inner_tolerance_advice(t, spec, caps);
        else if (role == SpecRole::Inner)
            msg += "; stop it with " +
                   stop_list(budget_keys(caps, false, caps), caps, false) +
                   " in the INNER block" +
                   (caps.supports_stopval && !caps.ignores_forced_stop
                        ? ", or STOPVAL in the enclosing block" : "");
        else
            msg += "; stop it with " +
                   stop_list(budget_keys(caps, true, caps), caps, false);
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
        stop_list(budget_keys(caps, true, caps), caps, true) +
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
    /* The enclosing block's stops that end the local search: only MAXEVAL
     * for one that ignores the forced stop. */
    const std::string handed_on =
        or_list(budget_keys(outer_caps, true, caps)) + " in the " + outer +
        " block, which NLopt hands on to it";
    if (parent.inner)
        throw std::runtime_error(
            "the INNER search of " + outer + " (" + inner + ") needs a stop: " +
            caps.stop_note + "; set " +
            stop_list(budget_keys(caps, false, caps), caps, false) +
            " in the INNER block, or " + handed_on + tolerance_note(caps));
    /* NLopt's default local optimiser: it takes the block's tolerances
     * too. */
    throw std::runtime_error(
        "the local search of " + outer + " (NLopt's default " + inner +
        ", as the block has no INNER block) needs a stop: " + caps.stop_note +
        "; set " + stop_list(budget_keys(caps, true, caps), caps, false) +
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
    /* The parser rejects INIT inside INNER, and nothing resolves an INNER
     * chain: a spec built outside it must not carry one either. */
    if (sp.inner && (!sp.inner->init_strategy_spec.chain.empty() ||
                     sp.inner->init_strategy_spec.k_total))
        throw std::invalid_argument(
            "NLopt INNER spec with an INIT chain: MLSL and AUGLAG choose the "
            "INNER search's starting points; an INIT block belongs to the "
            "enclosing block");
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
    /* A refused backend first, for its own phase, with validate_run_policy()'s
     * text: strategy_phase_specs() looks up the caps of the STRATEGY-level
     * backend to plan a MULTISTART from INIT points and a PRESTAGE
     * refinement of TOPK > 1, and the checks of an MLSL or AUGLAG phase
     * those of its INNER block; either lookup would report the refusal with
     * no phase, or for the enclosing one. */
    reject_refused_backends(root);
    for (const PhaseSpec& phase : strategy_phase_specs(root)) {
        try {
            /* The search space first: without a dimension no other setting
             * of NLOPT, DLIB or CERES matters (every phase, the EXPLORE
             * block's included; it was checked for the STRATEGY-level LIB
             * only, so an EXPLORE NLopt or DLIB:GFS failed in the run). */
            require_search_dimensions(phase.spec);
            validate_backend(phase.spec, restart,
                             phase.inner ? SpecRole::Inner : SpecRole::Phase);
        } catch (const std::exception& e) {
            /* At the line the phase's block closes at, as the parser's
             * refusal of a backend (with_block_line). */
            throw std::runtime_error(
                with_block_line(phase.spec, phase.name + ": " + e.what()));
        }
    }
}

}  // namespace tadah::hpo
