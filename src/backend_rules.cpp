#include <tadah/hpo/backend_rules.h>

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/fd_stencil.h>
#include <tadah/hpo/hpo.h>              // should_run_noise_calibration
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/repeat_points.h>
#include <tadah/hpo/strategy_rules.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
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
    /* A refused backend (Ceres' least-squares solvers among them) and an
     * unknown one: the caps lookup's error, as in run_inner(). */
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

namespace {

/* "1 evaluation", "4 evaluations". */
std::string evaluations(std::uint64_t n)
{
    return std::to_string(n) + (n == 1 ? " evaluation" : " evaluations");
}

/* "2 dimensions", "1 dimension". */
std::string dimensions(std::size_t n)
{
    return std::to_string(n) + (n == 1 ? " dimension" : " dimensions");
}

/* Whether the loss at a point a gradient search asks for, with the
 * gradient there, costs no evaluation: the point is one the run has
 * evaluated (`recorded`), and the run answers the call from its record
 * (repeat_use, repeat_points.h, the rule of the NLopt and DLIB:BFGS
 * objectives): under `reuse` and GRAD_STRATEGY NUMERICAL, whose finite
 * differences take the recorded loss as their centre (RepeatUse::Centre).
 * The record has no gradient there: a search forms its gradients itself,
 * and MLSL evaluates its samples without one. A point whose loss was not
 * usable is not recorded (the run records usable losses only) and is
 * evaluated again; MLSL takes its local starts in order of their loss
 * (mlsl.c), so its first is a failed point only when no usable one
 * qualifies, which the warning, before any evaluation, cannot know. */
bool loss_from_record(const OptimizerSpec& spec, bool reuse, bool recorded)
{
    return recorded &&
           repeat_use(reuse, spec.grad_strategy.value_or(GradStrategy::Numerical),
                      /*with_gradient=*/true, /*has_gradient=*/false) !=
               RepeatUse::Evaluate;
}

/* What the start of a gradient search costs before its first step, said
 * in a warning: "evaluates the loss <evaluated_where> (1 evaluation) and
 * the gradient there (<cost>)", or, with the loss from the record
 * (loss_from_record), "takes the loss <recorded_where> from the run's
 * record (no evaluation) and evaluates the gradient there (<cost>)". */
std::string start_clause(bool from_record, const std::string& evaluated_where,
                         const std::string& recorded_where,
                         const GradientCost& cost)
{
    return from_record
               ? "takes the loss " + recorded_where +
                     " from the run's record (no evaluation) and "
                     "evaluates the gradient there (" + cost.text + ")"
               : "evaluates the loss " + evaluated_where +
                     " (1 evaluation) and the gradient there (" + cost.text +
                     ")";
}

/* The points NLopt's MLSL samples in each round when POPULATION does not
 * set them (mlsl.c: d.N = 4 for Nsamples 0; optimize.c passes POP(0)). */
constexpr unsigned kMlslDefaultSamples = 4;

/* Whether the first call of a search of `spec` is at the point it is given
 * (search_ignoring_start(), phase_specs.h): not for NLopt's StoGO, which
 * samples its own box from its centre, nor for DIRECT or AGS. AUGLAG, with
 * no constraint, goes straight into its local search from its starting
 * point (auglag.c), so its first call is that search's (single_local_
 * search's rule). MLSL evaluates its starting point first (mlsl.c); where
 * its local searches start is local_starts_at_given(). */
bool first_call_at_start(const OptimizerSpec& spec)
{
    return !search_ignoring_start(spec);
}

/* Whether the local search an MLSL or AUGLAG runs (inner_run_spec) starts at
 * the point it is handed -- for MLSL one of its own evaluated points
 * (nlopt_optimize_limited from a point of its sample tree), for AUGLAG the
 * phase's start --, as every local NLopt algorithm does; an INNER GD_STOGO
 * starts at the centre of its box instead. */
bool local_starts_at_given(const OptimizerSpec& spec)
{
    const std::optional<OptimizerSpec> local = inner_run_spec(spec);
    return !local || first_call_at_start(*local);
}

/* The evaluations without a gradient a search makes before its first local
 * search asks for a gradient, and the text that says so (empty for none).
 * MLSL, a global search that runs a local one (mlsl.c), evaluates its
 * starting point and a first round of N samples with grad NULL (N its
 * POPULATION, else kMlslDefaultSamples); its first local search then starts
 * at one of these points (nlopt_optimize_limited from a point of its
 * sample tree), a point the search has evaluated, and asks for the loss
 * and the gradient there (start_clause) -- unless that local search does
 * not start at the point it is handed (local_starts_at_given: an INNER
 * GD_STOGO), when it asks for a point of its own. Its own starting point is
 * the phase's: when that is a point the run has evaluated
 * (`start_recorded`, start_is_recorded), the call there, which asks for no
 * gradient, is answered from the record under `reuse` (RepeatUse::Record).
 * AUGLAG, with no constraint, goes straight into its local search
 * (auglag.c), and a local gradient search takes the gradient at its start:
 * neither spends anything first. */
struct BeforeLocalSearch {
    std::uint64_t evaluations = 0;
    std::string   text;
};
BeforeLocalSearch before_local_search(const OptimizerSpec& spec, bool reuse,
                                      bool start_recorded)
{
    BeforeLocalSearch b;
    const OptimizerCaps& caps = optimizer_caps(spec);
    if (!(caps.is_global && caps.accepts_inner)) return b;
    const bool own = spec.pop && *spec.pop > 0;
    const unsigned n = own ? *spec.pop : kMlslDefaultSamples;
    const bool start_from_record =
        start_recorded &&
        repeat_use(reuse, spec.grad_strategy.value_or(GradStrategy::Numerical),
                   /*with_gradient=*/false, /*has_gradient=*/false) !=
            RepeatUse::Evaluate;
    b.evaluations = (start_from_record ? 0 : 1) + static_cast<std::uint64_t>(n);
    const std::string population = "; POPULATION " + std::to_string(n) +
                                   (own ? "" : ", NLopt's default") + ")";
    b.text = start_from_record
                 ? "MLSL takes the loss at its starting point from the "
                   "run's record and evaluates its first " +
                       std::to_string(n) + " samples without a gradient (" +
                       evaluations(b.evaluations) + population
                 : "MLSL evaluates its starting point and its first " +
                       std::to_string(n) + " samples without a gradient (" +
                       evaluations(b.evaluations) + population;
    b.text += local_starts_at_given(spec)
                  ? ", then its first local search, from one of them, "
                  : ", then its first local search, which does not start at "
                    "the point MLSL gives it, ";
    return b;
}

/* Whether two points are the same doubles, bit for bit (the run's record's
 * key, RunRecord::key). */
bool same_point_bits(const std::vector<double>& a, const std::vector<double>& b)
{
    return a.size() == b.size() &&
           (a.empty() ||
            std::memcmp(a.data(), b.data(), a.size() * sizeof(double)) == 0);
}

/* Whether the search of phase `s` (strategy_phase_specs of `root`) asks
 * first for a point the run has evaluated with a usable loss, on a backend
 * that answers a call from the run's record (repeat_points.h: NLopt's
 * objective and DLIB:BFGS; CERES, DLIB:GFS and the TADAH drivers evaluate
 * every point they ask for) and whose first call is at the point it is
 * given (first_call_at_start: not NLopt's StoGO, which samples its own box
 * from its centre, nor an AUGLAG whose INNER search is StoGO):
 *   - a PRESTAGE refinement starts at an exploration candidate, a usable
 *     evaluation of the run (the refinement takes usable candidates only);
 *   - under NOISE_CALIBRATION (should_run_noise_calibration) the run's
 *     first search -- PLAIN, the first MULTISTART start (for a MULTISTART
 *     that is a single run, that run), a PRESTAGE exploration -- starts at
 *     the seed point the calibration evaluated (run_noise_calibration: the
 *     spec's first INIT point) when its own first INIT point is that one;
 *     a seed anchor (search_space.h) makes the search ask for it bit for
 *     bit. A MULTISTART's later starts begin at INIT points the run has not
 *     evaluated: gradient_budget_warnings names the first start apart.
 * The warning, before any evaluation, cannot know a start whose evaluation
 * was not usable (it is evaluated again), nor the refinement from INIT
 * points that a PRESTAGE whose exploration has no usable candidate falls
 * back on, nor whether the point of its own a search that does not start
 * at its given point asks for first (StoGO's box centre) is one the run has
 * evaluated (it is, for example, when the configuration's point that a
 * NOISE CALIBRATION or an exploration evaluated lies at that centre, bit
 * for bit): it counts that point as evaluated. A RESTART round's start
 * changes no phase's budget: its warning is round 0's. */
bool start_is_recorded(const OptimizerSpec& root, const OptimizerSpec& s)
{
    const bool consults_record =
        s.lib == "NLOPT" || (s.lib == "DLIB" && s.algo == "BFGS");
    if (!consults_record || !first_call_at_start(s)) return false;
    if (root.strategy_type == StrategyType::PRESTAGE &&
        s.stream_phase == StreamPhase::Refinement)
        return true;
    if (!should_run_noise_calibration(root.noise.calibration_n) ||
        root.optim_init.empty() || s.optim_init.empty())
        return false;
    bool first_search = false;
    switch (root.strategy_type) {
        case StrategyType::PLAIN:
        case StrategyType::MULTISTART:
            first_search = true;
            break;
        case StrategyType::PRESTAGE:
            first_search = s.stream_phase == StreamPhase::Search;
            break;
    }
    return first_search &&
           same_point_bits(s.optim_init.front(), root.optim_init.front());
}

/* What the start of the gradient search of phase `s` costs before it
 * evaluates a point it steps to: an MLSL's evaluations before its first
 * local search (before_local_search), the loss at that search's start (none
 * from the record, loss_from_record), its gradient, and one point stepped
 * to. `start_recorded` as start_is_recorded() says, or false for the later
 * starts of a MULTISTART. */
struct StartCost {
    BeforeLocalSearch first;
    bool              mlsl = false;
    bool              local_at_given = true;  // the local search starts
                                              // at the point it is given
    bool              start_from_record = false;
    GradientCost      cost;
    std::uint64_t     need = 0;
};
StartCost start_cost(const OptimizerSpec& s, bool reuse, bool start_recorded)
{
    StartCost c;
    c.cost = gradient_cost(s);
    c.first = before_local_search(s, reuse, start_recorded);
    c.mlsl = !c.first.text.empty();
    /* The point the first gradient is taken at: for an MLSL one of the
     * points it evaluated, if its local search starts there; otherwise the
     * phase's start, if the search's first call is there (start_recorded
     * is false otherwise, start_is_recorded). */
    c.local_at_given = c.mlsl ? local_starts_at_given(s)
                              : first_call_at_start(s);
    c.start_from_record = loss_from_record(
        s, reuse, c.mlsl ? c.local_at_given : start_recorded);
    c.need = c.first.evaluations + (c.start_from_record ? 0 : 1) +
             c.cost.evaluations + 1;
    return c;
}

/* The warnings of phase `s` named `name` ("the STRATEGY-level optimiser",
 * "each MULTISTART start", ...) whose gradient search starts as
 * `start_recorded` says (start_cost), appended to `out`. */
void phase_budget_warnings(const OptimizerSpec& root, const std::string& name,
                           const OptimizerSpec& s, bool reuse,
                           bool start_recorded, std::vector<std::string>& out)
{
    const StartCost c = start_cost(s, reuse, start_recorded);
    const GradientCost& cost = c.cost;
    const std::string who = name + " " + backend_name(s);
    const std::string own_start =
        ", which does not start at the point it is given, ";
    std::string tail;
    if (c.mlsl)
        tail = " takes no step of a local search: " + c.first.text +
               start_clause(c.start_from_record,
                            c.local_at_given ? "there again"
                                             : "at a point of its own",
                            "there", cost);
    else if (!c.local_at_given)
        tail = " takes no step: its gradient search" + own_start +
               start_clause(false, "at a point of its own", "", cost);
    else
        tail = " takes no step: its gradient search " +
               start_clause(c.start_from_record,
                            start_recorded ? "at its starting point again"
                                           : "at a starting point",
                            "at its starting point", cost);
    tail += " before it evaluates a point it steps to, which needs ";
    const std::string more =
        cost.at_least ? ", and more for each parameter it differences" : "";
    if (s.stop.maxeval && *s.stop.maxeval < c.need) {
        const std::string have = std::to_string(*s.stop.maxeval);
        const std::string needs = std::to_string(c.need);
        /* The budget as the configuration writes it (plan_multistart):
         * each MULTISTART start's BUDGET_PER_START or share of MAXEVAL,
         * the PRESTAGE refinement's share of MAXEVAL over its TOPK
         * candidates, otherwise the block's MAXEVAL (the exploration's
         * is its EXPLORE block's). */
        const bool starts =
            root.strategy_type == StrategyType::MULTISTART &&
            !plan_multistart(root, root.multistart.n_starts,
                             root.multistart.budget_per_start)
                 .single_run;
        const bool candidates =
            root.strategy_type == StrategyType::PRESTAGE &&
            s.stream_phase == StreamPhase::Refinement;
        const bool split = (starts || candidates) &&
                           root.stop.maxeval.has_value() &&
                           s.stop.maxeval != root.stop.maxeval;
        std::string budget = "MAXEVAL " + have;
        std::string remedy = "MAXEVAL " + needs + " or more";
        if (starts && root.multistart.budget_per_start) {
            budget = "BUDGET_PER_START " + have;
            remedy = "BUDGET_PER_START " + needs + " or more";
        } else if (split) {
            const std::size_t k =
                starts ? plan_multistart(root, root.multistart.n_starts,
                                         root.multistart.budget_per_start)
                             .starts
                       : std::max<std::size_t>(1, root.stage_topk);
            budget = "MAXEVAL " + std::to_string(*root.stop.maxeval) +
                     " over " + std::to_string(k) +
                     (starts ? " starts" : " TOPK candidates") + ", " +
                     have + " each,";
            remedy = needs + " evaluations or more per " +
                     (starts ? "start" : "candidate");
        }
        out.push_back(who + ": " + budget + tail + remedy + more);
    }
    /* An INNER block's local search: NLopt hands it its own MAXEVAL in
     * NLopt calls (nlopt_optimize_limited), each the loss and the
     * gradient at a point, so one call is its starting point -- for an
     * MLSL a point it evaluated, every local search's (mlsl.c starts
     * them at points of its sample tree), whose loss the record can
     * answer; for AUGLAG the phase's start; for an INNER GD_STOGO a point
     * of its own (local_starts_at_given), which the record does not
     * answer. NLopt counts a call answered from the record too, so the
     * budget is the same. */
    if (!s.inner || !s.inner->stop.maxeval || *s.inner->stop.maxeval >= 2)
        return;
    const std::optional<OptimizerSpec> local = inner_run_spec(s);
    if (!local || !obtains_gradients(*local)) return;
    const std::string call =
        !local_starts_at_given(s)
            ? "(1 + " + cost.text + "), so each local search" + own_start +
                  "evaluates a point of its own"
        : c.start_from_record
            ? "(" + cost.text + "; " +
                  (c.mlsl ? std::string("MLSL's local searches start at "
                                        "points it evaluated")
                          : std::string("its local search starts at a "
                                        "point the run has evaluated")) +
                  ", whose loss is taken from the run's record), so "
                  "each local search evaluates the gradient at its "
                  "starting point"
            : "(1 + " + cost.text + "), so each local search evaluates "
              "its starting point";
    out.push_back("the INNER block of " + name + " " +
                  backend_name(*s.inner) + ": MAXEVAL " +
                  std::to_string(*s.inner->stop.maxeval) +
                  " takes no step: NLopt counts an INNER block's MAXEVAL "
                  "in its calls, each the loss and the gradient at a "
                  "point " + call + " and ends there, which needs "
                  "MAXEVAL 2 or more");
}

}  // namespace

GradientCost gradient_cost(const OptimizerSpec& spec)
{
    GradientCost c;
    switch (spec.grad_strategy.value_or(GradStrategy::Numerical)) {
        case GradStrategy::Analytical:
            c.evaluations = 1;
            c.text = "1 evaluation: one analytical gradient, GRAD_STRATEGY "
                     "ANALYTICAL";
            return c;
        case GradStrategy::Hybrid:
            c.evaluations = 1;
            c.at_least = true;
            c.text = "at least 1 evaluation: the analytical gradient, and the "
                     "finite differences of each parameter it does not cover, "
                     "GRAD_STRATEGY HYBRID";
            return c;
        case GradStrategy::Numerical:
            break;
    }
    const std::size_t n = spec.low.size();
    const bool central = fd_settings_of(spec).scheme == FdScheme::Central;
    c.evaluations = central ? 2 * static_cast<std::uint64_t>(n)
                            : static_cast<std::uint64_t>(n);
    c.text = evaluations(c.evaluations) + ": " +
             (central ? "central" : "forward") + " differences in " +
             dimensions(n) + ", GRAD_STRATEGY NUMERICAL";
    return c;
}

namespace {

/* "<who> <name><aside> does not start at the point it is given (<note>)",
 * for a phase `phase` whose search `blind` (search_ignoring_start(phase))
 * does not start at its point; for an AUGLAG, whose local search it is,
 * "<who> <name><aside> runs its INNER search <inner> from the point it is
 * given, and <inner> does not start at the point it is given (<note>)". */
std::string ignores_its_start(const std::string& who,
                              const OptimizerSpec& phase,
                              const OptimizerSpec& blind,
                              const std::string& aside)
{
    const std::string head = who + " " + backend_name(phase) + aside;
    const std::string note = optimizer_caps(blind).start_note;
    if (!optimizer_caps(phase).starts_at_x0)
        return head + " does not start at the point it is given (" + note +
               ")";
    const std::string inner = backend_name(blind);
    return head + " runs its INNER search " + inner + " from the point it is "
           "given, and " + inner + " does not start at the point it is given "
           "(" + note + ")";
}

/* Which of round 0's points a RESTART round of the search `blind`
 * (search_ignoring_start) asks for again, whatever points it starts from:
 * all of them (it draws no random numbers, OptimizerCaps::random_draws), the
 * points of its first local search (GD_STOGO_RAND below POPULATION 2n + 1,
 * which runs it from the box centre before any random trial point), or
 * none it need ask for (it draws its numbers from the round's seed). */
enum class RoundRepeat { None, All, FirstLocalSearch };
struct RoundRepeatClause {
    RoundRepeat what = RoundRepeat::None;
    std::string clause;  // " and draws no random numbers", ...
};

/* The RoundRepeat of the search `blind` of a phase of `dims` OPTIM
 * dimensions, with the clause that says why. */
RoundRepeatClause round_repeat(const OptimizerSpec& blind, std::size_t dims)
{
    using RD = OptimizerCaps::RandomDraws;
    const RD draws = optimizer_caps(blind).random_draws;
    if (draws == RD::Never)
        return {RoundRepeat::All, " and draws no random numbers"};
    if (draws == RD::BetweenSides && dims == 1)
        return {RoundRepeat::All,
                " and, in one OPTIM dimension, draws no random numbers (it "
                "draws them only to choose between equally long sides of a "
                "box)"};
    if (draws == RD::TrialPoints) {
        /* optimize.c: POP(2n) random trial points, NLopt's default 2n for
         * an unset or zero POPULATION; stogo.cc det_pnts = 2n + 1 - it. */
        const std::size_t full = 2 * dims + 1;
        const bool own = blind.pop && *blind.pop > 0;
        const std::size_t pop = own ? *blind.pop : 2 * dims;
        if (pop < full) {
            const std::string n =
                "n = " + std::to_string(dims) +
                (dims == 1 ? " OPTIM dimension" : " OPTIM dimensions");
            std::string c = " and, with POPULATION " + std::to_string(pop);
            c += own ? ", less than 2n + 1 = " + std::to_string(full) + " (" +
                           n + ")"
                     : " (NLopt's default, 2n for " + n +
                           "), less than 2n + 1 = " + std::to_string(full);
            c += ", takes the centre of the box as its first trial point and "
                 "runs a local search from it before any of its random trial "
                 "points";
            return {RoundRepeat::FirstLocalSearch, c};
        }
    }
    return {};
}

/* What a RESTART round gets when it asks for round 0's points again (the
 * run's record, repeat_points.h): with a `reuse_blocker` the run evaluates
 * them again; else the record answers their losses, and a search that
 * obtains gradients forms them itself (the record keeps no gradient of an
 * earlier search): under GRAD_STRATEGY NUMERICAL the round evaluates the
 * finite-difference stencils again (RepeatUse::Centre), under ANALYTICAL
 * and HYBRID each point at which it asks for one (RepeatUse::Evaluate). */
std::string round_answer(const OptimizerSpec& phase, const OptimizerSpec& blind,
                         const std::string& reuse_blocker)
{
    if (!reuse_blocker.empty())
        return " as long as their losses are those of round 0 (the run "
               "evaluates them again: " + reuse_blocker + ")";
    if (!obtains_gradients(blind)) return ", which the run's record answers";
    const GradStrategy g = phase.grad_strategy.value_or(GradStrategy::Numerical);
    if (repeat_use(true, g, /*with_gradient=*/true, /*has_gradient=*/false) ==
        RepeatUse::Centre)
        return ", and their gradients: the run's record answers their losses, "
               "but a round forms its gradients itself and evaluates their "
               "finite-difference stencils again (GRAD_STRATEGY " +
               std::string(grad_strategy_name(g)) + ")";
    return ", and their gradients: the run's record answers their losses "
           "where a round asks for no gradient, but each point at which it "
           "asks for one is evaluated again, with its gradient (GRAD_STRATEGY " +
           std::string(grad_strategy_name(g)) +
           " differentiates an evaluation of the point)";
}

std::string block_on_line(const char* block, std::size_t line)
{
    return std::string("the ") + block + " block" +
           (line > 0 ? " on line " + std::to_string(line) : std::string()) +
           ": ";
}

}  // namespace

std::vector<std::string> start_point_warnings(const OptimizerSpec& root,
                                              const std::string& reuse_blocker)
{
    std::vector<std::string> out;
    /* The search each round starts with: PLAIN's STRATEGY-level optimiser,
     * PRESTAGE's exploration (MULTISTART starts and a PRESTAGE refinement
     * on such a search are refused). */
    const bool prestage = root.strategy_type == StrategyType::PRESTAGE;
    if (root.strategy_type == StrategyType::MULTISTART) return out;
    const OptimizerSpec* phase = prestage ? root.prestage.get() : &root;
    if (!phase || phase->lib == "NOT SET" || phase->algo == "NOT SET")
        return out;
    const std::optional<OptimizerSpec> blind = search_ignoring_start(*phase);
    if (!blind) return out;
    const std::string who = prestage ? "the PRESTAGE exploration"
                                     : "the STRATEGY-level optimiser";
    const std::string calibration =
        should_run_noise_calibration(root.noise.calibration_n)
            ? " (NOISE CALIBRATION evaluates the first of them " +
                  std::to_string(root.noise.calibration_n) + " times)"
            : std::string();
    const std::string to_start =
        prestage ? " To start the exploration at them, use an exploration "
                   "optimiser that starts at the point it is given."
                 : " To start the search at them, use an optimiser that "
                   "starts at the point it is given.";

    /* An INIT block with a METHOD: the phase's own (PLAIN: the
     * OPTIMIZER/STRATEGY-level block; PRESTAGE: the EXPLORE block's), and
     * under PRESTAGE the OPTIMIZER-level block, which the exploration takes
     * when its EXPLORE block has none (share_init_with_nested_phases) and
     * the refinement only when the exploration has no usable candidate. */
    const InitStrategySpec& own = phase->init_strategy_spec;
    if (!own.chain.empty())
        out.push_back(
            block_on_line("INIT", own.line) +
            ignores_its_start(who, *phase, *blind, "") + ", so the " +
            (prestage ? "exploration" : "search") +
            " never evaluates the block's INIT points" +
            (prestage ? std::string() : calibration) + "." + to_start);
    const InitStrategySpec& outer = root.init_strategy_spec;
    if (prestage && own.chain.empty() && !outer.chain.empty())
        out.push_back(
            block_on_line("INIT", outer.line) +
            ignores_its_start(who, *phase, *blind,
                              ", which is given the block's INIT points (its "
                              "EXPLORE block has no INIT block),") +
            ", so the exploration never evaluates them" + calibration +
            "; the refinement starts from them only when no evaluation of "
            "the exploration has a usable loss." + to_start);

    /* An active RESTART block (RestartPolicy changes the start points and
     * the seeds only): a search that draws no random numbers repeats round
     * 0's points in every round (GD_STOGO_RAND below POPULATION 2n + 1 the
     * points of its first local search), and one that draws them never
     * evaluates the points of the block's METHOD lines either. */
    if (root.restart.active()) {
        const std::size_t dims = phase->low.size();
        const RoundRepeatClause rep = round_repeat(*blind, dims);
        const std::string window =
            "AFTER_STAGNATION " + std::to_string(root.restart.after_stagnation);
        if (rep.what != RoundRepeat::None) {
            const bool first = rep.what == RoundRepeat::FirstLocalSearch;
            const std::string asks =
                first ? "asks first for the points of round 0's first local "
                        "search again"
                      : prestage ? "asks for round 0's exploration points again"
                                 : "asks for round 0's points again";
            std::string w = block_on_line("RESTART", root.restart.line) +
                            ignores_its_start(who, *phase, *blind, "") +
                            rep.clause + ", so " +
                            (prestage ? "the exploration of every RESTART "
                                        "round, whatever points it starts "
                                        "from, "
                                      : "every RESTART round, whatever points "
                                        "it starts from, ") +
                            asks + round_answer(*phase, *blind, reuse_blocker);
            if (reuse_blocker.empty()) {
                if (first)
                    w += prestage
                             ? ": until that search ends a round's "
                               "exploration evaluates no new point, and it "
                               "reaches its random trial points only if the "
                               "search ends before the exploration has had " +
                                   window + " evaluations without a new best"
                             : ": until that search ends a round evaluates no "
                               "new point, and it reaches its random trial "
                               "points only if the search ends before the "
                               "round has had " + window +
                                   " evaluations without a new best";
                else if (prestage)
                    w += ": a round's refinement starts from candidates among "
                         "round 0's exploration points";
                else if (obtains_gradients(*blind))
                    w += ": no round after round 0 evaluates a new point, and "
                         "each round evaluates up to " + window +
                         " of round 0's points again before it ends without a "
                         "new best";
                else
                    w += ": no round after round 0 evaluates a new point, and "
                         "the RESTART block's only effect is to end the run "
                         "once round 0 has had no new best in " + window +
                         " evaluations";
            }
            const std::string population =
                first ? ", or POPULATION " + std::to_string(2 * dims + 1) +
                            " (2n + 1) or more, whose trial points are all "
                            "random"
                      : std::string();
            w += prestage ? ". For explorations that search anew, use an "
                            "exploration optimiser that starts at the point "
                            "it is given" + population + "."
                          : ". For rounds that search anew, use an optimiser "
                            "that starts at the point it is given" +
                                population + ".";
            out.push_back(w);
        } else if (!root.restart.init_strategy.chain.empty() &&
                   (!prestage || phase->init_strategy_spec.chain.empty())) {
            /* The round's points go to the search (to a PRESTAGE
             * exploration without an INIT block of its own,
             * share_init_with_nested_phases), which never evaluates them;
             * only its random numbers differ from round 0's. */
            out.push_back(
                block_on_line("RESTART", root.restart.line) +
                (prestage
                     ? ignores_its_start(who, *phase, *blind,
                                         ", which is given each round's "
                                         "points (its EXPLORE block has no "
                                         "INIT block),") +
                           ", so no round's exploration evaluates the points "
                           "of the block's METHOD lines, and a round's "
                           "refinement starts from them only when no "
                           "evaluation of its exploration has a usable loss: "
                           "a round's exploration differs from round 0's only "
                           "in the random numbers it draws. To start each "
                           "round's exploration at them, use an exploration "
                           "optimiser that starts at the point it is given."
                     : ignores_its_start(who, *phase, *blind, "") +
                           ", so no RESTART round evaluates the points of the "
                           "block's METHOD lines: a round's search differs "
                           "from round 0's only in the random numbers it "
                           "draws. To start each round at the block's points, "
                           "use an optimiser that starts at the point it is "
                           "given."));
        }
    }
    return out;
}

std::vector<std::string> gradient_budget_warnings(const OptimizerSpec& root,
                                                  bool reuse)
{
    std::vector<std::string> out;
    for (const PhaseSpec& phase : strategy_phase_specs(root)) {
        /* An INNER block's MAXEVAL counts NLopt calls: below, with the
         * phase whose gradient its local search takes. */
        if (phase.inner) continue;
        const OptimizerSpec& s = phase.spec;
        if (s.lib == "NOT SET" || s.algo == "NOT SET") continue;
        if (!obtains_gradients(s)) continue;
        /* Whether the phase starts at a point the run has evaluated
         * (start_is_recorded): the run keeps one record of its usable
         * evaluations (repeat_points.h, HPO_Host::recorded), which NLopt
         * and DLIB:BFGS answer a call from, so a PRESTAGE refinement's
         * start, an exploration candidate, is one, and so is the start of
         * the run's first search under NOISE_CALIBRATION, when the search
         * asks for its starting point first (not NLopt's StoGO). The
         * counts and the texts follow it (start_cost). */
        const bool start_recorded = start_is_recorded(root, s);
        /* Under NOISE_CALIBRATION the first of several MULTISTART starts
         * begins at the calibration's seed point and the later ones at
         * INIT points the run has not evaluated: where the two cost
         * different budgets, each is named apart. */
        bool several_starts = false;
        if (root.strategy_type == StrategyType::MULTISTART) {
            const MultiStartPlan plan =
                plan_multistart(root, root.multistart.n_starts,
                                root.multistart.budget_per_start);
            several_starts = !plan.single_run && plan.starts > 1;
        }
        if (start_recorded && several_starts &&
            start_cost(s, reuse, true).need !=
                start_cost(s, reuse, false).need) {
            phase_budget_warnings(root, "the first MULTISTART start", s,
                                  reuse, true, out);
            phase_budget_warnings(root, "each MULTISTART start after the first",
                                  s, reuse, false, out);
            continue;
        }
        phase_budget_warnings(root, phase.name, s, reuse,
                              start_recorded && !several_starts, out);
    }
    return out;
}

}  // namespace tadah::hpo
