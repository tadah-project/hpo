#include <tadah/hpo/observables/observable_parser.h>

#include <tadah/core/number_format.h>

#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/strict_parse.h>         // parse_number, parse_uint

#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

namespace {

/* Tokenise on whitespace (no shell-style quoting; PC_OBSERVABLE values
 * are simple identifiers / numbers). */
std::vector<std::string> ws_tokens(const std::string& s) {
    std::vector<std::string> out;
    std::istringstream iss(s);
    std::string t;
    while (iss >> t) out.push_back(t);
    return out;
}

/* A finite number of the number grammar every key reads (strict_parse.h,
 * parse_number). */
double parse_double(const std::string& s, const std::string& flag) {
    double v = 0.0;
    if (!parse_number(s, v))
        throw std::runtime_error(
            "PC_OBSERVABLE: flag '" + flag + "' expected a finite number, got '"
            + s + "'" + number_underflow_note(s));
    return v;
}

/* An integer flag: decimal digits only, from `min` to INT_MAX, the
 * integer grammar of every key (strict_parse.h, parse_uint). It used to be
 * narrowed from long long without a check (4294967300 became 4), and a
 * sign was read: `--maxiter -0` was 0, and a negative value got a range
 * error that did not name it. */
int parse_int(const std::string& s, const std::string& flag, int min) {
    return static_cast<int>(parse_uint(
        s, "PC_OBSERVABLE: flag '" + flag + "'",
        static_cast<std::uint64_t>(min),
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
}

std::unique_ptr<Observable>
parse_cohesive_energy(const std::vector<std::string>& tokens, std::size_t start)
{
    CohesiveEnergyObservable::Config cfg;
    /* Defaults. */
    cfg.nx = cfg.ny = cfg.nz = 4;
    /* Loss defaults to L2 (LossFunction's own default). */

    bool seen_species = false, seen_target = false, seen_weight = false;
    bool seen_hotvar  = false;
    std::string lattice = "bcc";

    auto need_value = [&](std::size_t& i, const std::string& flag) -> const std::string& {
        if (i + 1 >= tokens.size())
            throw std::runtime_error("PC_OBSERVABLE cohesive_energy: '" + flag
                                     + "' requires a value");
        return tokens[++i];
    };

    for (std::size_t i = start; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];
        if (t == "--species") {
            cfg.species = need_value(i, t);
            seen_species = true;
        } else if (t == "--target") {
            cfg.target = parse_double(need_value(i, t), t);
            seen_target = true;
        } else if (t == "--weight") {
            cfg.weight = parse_double(need_value(i, t), t);
            seen_weight = true;
        } else if (t == "--lattice") {
            lattice = need_value(i, t);
            if (lattice != "bcc")
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: lattice='" + lattice
                    + "' not supported in MVP (only bcc).");
        } else if (t == "--supercell") {
            /* At least one cell along each axis. */
            cfg.nx = parse_int(need_value(i, t), "--supercell nx", 1);
            cfg.ny = parse_int(need_value(i, t), "--supercell ny", 1);
            cfg.nz = parse_int(need_value(i, t), "--supercell nz", 1);
        } else if (t == "--hotvar") {
            /* --hotvar a0 <lo> <hi> [<init>] */
            const std::string& name = need_value(i, t);
            if (name != "a0")
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: only --hotvar a0 is "
                    "supported in MVP (got '" + name + "').");
            cfg.a0_min = parse_double(need_value(i, "--hotvar a0 <lo>"),
                                       "--hotvar a0 lo");
            cfg.a0_max = parse_double(need_value(i, "--hotvar a0 <hi>"),
                                       "--hotvar a0 hi");
            /* Optional initial; peek next token. */
            if (i + 1 < tokens.size() && tokens[i + 1].substr(0, 2) != "--") {
                cfg.a0_initial = parse_double(tokens[++i], "--hotvar a0 init");
            }
            seen_hotvar = true;
        } else if (t == "--loss") {
            const std::string& kind = need_value(i, t);
            cfg.lossfn = tadah::LossFunction(
                tadah::LossFunction::from_string(kind));
        } else if (t == "--maxiter") {
            /* Caps NativeRelaxer iterations. Default is 5000 (deep
             * convergence). On randomly-initialised potentials, the
             * energy surface is often non-physical and the relaxer
             * wanders without converging; cap to a small budget
             * (50-200) for HPO use where the optimiser sweeps theta
             * before any pot is good enough for a true ground state. */
            cfg.relax_maxiter = static_cast<std::size_t>(
                parse_int(need_value(i, t), "--maxiter", 0));
        } else if (t == "--max-wall-time") {
            /* Per-relax wall-time cap (seconds). PC_OBSERVABLE's
             * analogue of LAMMPS's MAXTIME (per-eval subprocess kill).
             * With NN caching, a pathological theta can burn maxeval=5000
             * in ~5 s; 1.0 s matches the LAMMPS reference's MAXTIME=1. */
            cfg.relax_max_wall_time_s =
                parse_double(need_value(i, t), "--max-wall-time");
        } else if (t == "--etol") {
            /* LAMMPS-style energy convergence tolerance, mirrors 1st
             * positional arg of `minimize etol ftol maxiter maxeval`.
             * Production runs use 1e-15 to match Ecoh_fast.in. Default
             * 0 = disabled (force-only convergence). */
            cfg.relax_etol =
                parse_double(need_value(i, t), "--etol");
        } else if (t == "--maxeval") {
            /* LAMMPS-style total energy/force-eval cap. Mirrors the
             * 4th positional arg of LAMMPS `minimize etol ftol maxiter
             * maxeval`. Critical for HPO: without it, a pathological theta
             * that never converges does maxiter x maxiter_ls evals and
             * burns minutes per HPO iter. */
            cfg.relax_maxeval = static_cast<std::size_t>(
                parse_int(need_value(i, t), "--maxeval", 0));
        } else if (t == "--fmax-converged") {
            /* eq_ok threshold (matches Ecoh_fast.in's
             * `ternary($(fmax) < 0.1, 1, 0)`). If post-relax fmax
             * exceeds this, return the --unconverged value as Ecoh. */
            cfg.eq_fmax_threshold =
                parse_double(need_value(i, t), "--fmax-converged");
        } else if (t == "--unconverged") {
            /* Strategy + value for unconverged relax outcomes. Syntax:
             *   --unconverged const        <V>     # res.value = V
             *   --unconverged force-scaled <slope> # res.value = target + slope.fmax
             *
             * `const 100.0` mirrors Ecoh_fast.in's eq_ok=false branch
             * (the cliff). `force-scaled <slope>` gives local optimisers
             * a continuous gradient toward fmax -> 0; slope tunes how
             * steeply unconverged theta are penalised. See
             * CohesiveEnergyObservable::Config::UnconvergedStrategy
             * docs for the full rationale. */
            using Strat = CohesiveEnergyObservable::Config::UnconvergedStrategy;
            const std::string& kind = need_value(i, t);
            if (kind == "const") {
                cfg.unconverged_strategy = Strat::Const;
            } else if (kind == "force-scaled") {
                cfg.unconverged_strategy = Strat::ForceScaled;
            } else {
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --unconverged kind must be "
                    "'const' or 'force-scaled' (got '" + kind + "')");
            }
            cfg.unconverged_value = parse_double(
                need_value(i, "--unconverged " + kind + " <value>"),
                "--unconverged " + kind);
        } else {
            throw std::runtime_error(
                "PC_OBSERVABLE cohesive_energy: unknown flag '" + t + "'");
        }
    }

    if (!seen_species)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --species is required");
    if (!seen_target)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --target is required");
    if (!seen_weight)
        throw std::runtime_error("PC_OBSERVABLE cohesive_energy: --weight is required");
    if (!seen_hotvar)
        throw std::runtime_error(
            "PC_OBSERVABLE cohesive_energy: --hotvar a0 <lo> <hi> [<init>] "
            "is required");

    return std::make_unique<CohesiveEnergyObservable>(std::move(cfg));
}

}  // anonymous namespace

std::unique_ptr<Observable> parse_pc_observable_line(const std::string& cmd)
{
    auto tokens = ws_tokens(cmd);
    if (tokens.empty())
        throw std::runtime_error("PC_OBSERVABLE: empty line");

    const std::string& kind = tokens[0];
    if (kind == "cohesive_energy") {
        return parse_cohesive_energy(tokens, /*start=*/1);
    }
    /* Future observable kinds dispatch here:
     *   if (kind == "lattice_param")        return parse_lattice_param(...);
     *   if (kind == "vacancy_formation")    return parse_vacancy_formation(...);
     *   if (kind == "elastic_constants")    return parse_elastic_constants(...);
     *   if (kind == "surface_energy")       return parse_surface_energy(...);
     *   if (kind == "gamma_surface")        return parse_gamma_surface(...);
     *   if (kind == "phase_transition_energy") return parse_phase_transition(...);
     *   if (kind == "relax_energy")         return parse_relax_energy(...);   // file-loaded Tier-2
     */
    throw std::runtime_error("PC_OBSERVABLE: unknown observable kind '" + kind + "'");
}

}  // namespace hpo
}  // namespace tadah
