#include <tadah/hpo/observables/observable_parser.h>

#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/cohesive_energy.h>

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

double parse_double(const std::string& s, const std::string& flag) {
    try {
        std::size_t consumed = 0;
        double v = std::stod(s, &consumed);
        if (consumed != s.size())
            throw std::invalid_argument("trailing chars");
        return v;
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "PC_OBSERVABLE: flag '" + flag + "' expected a number, got '"
            + s + "'");
    }
}

int parse_int(const std::string& s, const std::string& flag) {
    try {
        std::size_t consumed = 0;
        long long v = std::stoll(s, &consumed);
        if (consumed != s.size())
            throw std::invalid_argument("trailing chars");
        return static_cast<int>(v);
    } catch (const std::exception& e) {
        throw std::runtime_error(
            "PC_OBSERVABLE: flag '" + flag + "' expected an integer, got '"
            + s + "'");
    }
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
            cfg.nx = parse_int(need_value(i, t), "--supercell nx");
            cfg.ny = parse_int(need_value(i, t), "--supercell ny");
            cfg.nz = parse_int(need_value(i, t), "--supercell nz");
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
             * (50-200) for HPO use where the optimiser sweeps θ
             * before any pot is good enough for a true ground state. */
            const int v = parse_int(need_value(i, t), "--maxiter");
            if (v < 0)
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --maxiter must be >= 0");
            cfg.relax_maxiter = static_cast<std::size_t>(v);
        } else if (t == "--maxeval") {
            /* LAMMPS-style total energy/force-eval cap. Mirrors the
             * 4th positional arg of LAMMPS `minimize etol ftol maxiter
             * maxeval`. Critical for HPO: without it, a pathological θ
             * that never converges does maxiter × maxiter_ls evals and
             * burns minutes per HPO iter. */
            const int v = parse_int(need_value(i, t), "--maxeval");
            if (v < 0)
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --maxeval must be >= 0");
            cfg.relax_maxeval = static_cast<std::size_t>(v);
        } else if (t == "--fmax-converged") {
            /* eq_ok threshold (matches Ecoh_fast.in's
             * `ternary($(fmax) < 0.1, 1, 0)`). If post-relax fmax
             * exceeds this, return `--unconverged-penalty` as Ecoh. */
            cfg.eq_fmax_threshold =
                parse_double(need_value(i, t), "--fmax-converged");
        } else if (t == "--unconverged-penalty") {
            /* Penalty Ecoh value when the relax didn't converge.
             * Mirrors Ecoh_fast.in's `Ecoh = 100.0` fallback. Pass
             * "nan" to fall back to hard NaN reject. */
            const std::string& v = need_value(i, t);
            if (v == "nan" || v == "NaN" || v == "NAN") {
                cfg.unconverged_penalty =
                    std::numeric_limits<double>::quiet_NaN();
            } else {
                cfg.unconverged_penalty =
                    parse_double(v, "--unconverged-penalty");
            }
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
