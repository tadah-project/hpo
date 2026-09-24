#include <tadah/hpo/backend_rules.h>

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/asa_options.h>
#include <tadah/hpo/ceres_adapter.h>
#include <tadah/hpo/dlib_adapter.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>

#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

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

}  // namespace

void require_maxeval_budget(const OptimizerSpec& spec)
{
    if (spec.stop.maxeval) return;
    const OptimizerCaps& caps = optimizer_caps(spec);
    if (!caps.needs_maxeval) return;
    throw std::runtime_error(
        backend_name(spec) + " needs MAXEVAL: " + caps.budget_note +
        "; set MAXEVAL (an integer from 1 to 2147483647)");
}

void validate_random_spec(const OptimizerSpec& spec)
{
    require_maxeval_budget(spec);
}

void validate_dlib_gfs_spec(const OptimizerSpec& spec)
{
    require_maxeval_budget(spec);
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
    for (const auto& kv : sp.param)
        if (kv.first != "GRID_POINTS")
            throw std::runtime_error("TADAH:GRID: unknown PARAM '" + kv.first +
                                     "' (the only GRID parameter is "
                                     "GRID_POINTS)");
    /* A word where a number belongs (PARAM GRID_POINTS AUTO) was stored as
     * a word and ignored. */
    for (const auto& kv : sp.param_raw)
        throw std::runtime_error(
            kv.first == "GRID_POINTS"
                ? "TADAH:GRID PARAM GRID_POINTS must be an integer >= 2 (got '" +
                      kv.second + "')"
                : "TADAH:GRID: unknown PARAM '" + kv.first +
                      "' (the only GRID parameter is GRID_POINTS)");
    return steps;
}

void validate_nlopt_spec(const OptimizerSpec& sp)
{
    if (sp.inner && (sp.inner->seed || sp.inner->init_strategy_spec.seed))
        throw std::invalid_argument(
            "NLopt INNER spec with a seed: NLopt draws the INNER search from "
            "the enclosing run's random stream; set SEED on the enclosing "
            "block");
    require_nlopt_maxeval(sp);
    if (sp.inner) require_nlopt_maxeval(*sp.inner);
}

void validate_asa_spec(const OptimizerSpec& sp)
{
    const std::size_t dim = sp.low.size();
    if (dim == 0)
        throw std::runtime_error(
            "ASA:ASA requires at least one OPTIM dimension");
    (void)parse_asa_options(sp, dim);
}

void validate_backend(const OptimizerSpec& spec)
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
            validate_random_spec(spec);
        } else if (algo == "GRID") {
            (void)validate_grid_spec(spec);
        } else if (algo == "ANNEAL") {
            (void)anneal::Knobs::parse(spec, anneal::AlgoKind::Serial);
        } else if (algo == "ANNEAL_PT") {
            const auto kb = anneal::Knobs::parse(spec, anneal::AlgoKind::PT);
            (void)anneal::pt_replica_count(spec, kb);
        } else if (algo == "ANNEAL_PA") {
            (void)anneal::Knobs::parse(spec, anneal::AlgoKind::PA);
        }
    } else if (lib == "DLIB") {
        if (algo == "GFS")       validate_dlib_gfs_spec(spec);
        else if (algo == "BFGS") detail::validate_dlib_bfgs_spec(spec);
    } else if (lib == "NLOPT") {
        validate_nlopt_spec(spec);
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
}

void validate_phase_backends(const OptimizerSpec& root)
{
    for (const PhaseSpec& phase : strategy_phase_specs(root)) {
        try {
            validate_backend(phase.spec);
        } catch (const std::exception& e) {
            throw std::runtime_error(phase.name + ": " + e.what());
        }
    }
}

}  // namespace tadah::hpo
