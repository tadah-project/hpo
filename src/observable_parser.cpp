#include <tadah/hpo/observables/observable_parser.h>

#include <tadah/core/number_format.h>
#include <tadah/core/periodic_table.h>

#include <tadah/hpo/loss_function.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/hpo/strict_parse.h>         // parse_number, parse_uint

#include <algorithm>
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
 * parse_number), the value of @p option of a PC_OBSERVABLE line of the
 * observable @p kind. */
double parse_double(const std::string& s, const std::string& kind,
                    const std::string& option) {
    double v = 0.0;
    if (!parse_number(s, v))
        throw std::runtime_error(
            "PC_OBSERVABLE " + kind + ": " + option + " takes a finite number "
            "(got '" + s + "')" + number_underflow_note(s));
    return v;
}

/* An integer flag: decimal digits only, from `min` to INT_MAX, the
 * integer grammar of every key (strict_parse.h, parse_uint). It used to be
 * narrowed from long long without a check (4294967300 became 4), and a
 * sign was read: `--maxiter -0` was 0, and a negative value got a range
 * error that did not name it. */
int parse_int(const std::string& s, const std::string& kind,
              const std::string& option, int min) {
    return static_cast<int>(parse_uint(
        s, "PC_OBSERVABLE " + kind + ": " + option,
        static_cast<std::uint64_t>(min),
        static_cast<std::uint64_t>(std::numeric_limits<int>::max())));
}

std::unique_ptr<Observable>
parse_cohesive_energy(const std::vector<std::string>& tokens, std::size_t start,
                      const tadah::LossFunction& default_loss)
{
    const std::string observable = "cohesive_energy";
    auto number = [&](const std::string& s, const std::string& option) {
        return parse_double(s, observable, option);
    };
    auto integer = [&](const std::string& s, const std::string& option,
                       int min) { return parse_int(s, observable, option, min); };
    CohesiveEnergyObservable::Config cfg;
    /* Defaults. */
    cfg.nx = cfg.ny = cfg.nz = 4;
    /* The loss function of the LOSS line read above the line, as for a
     * --varloss term (it was L2 whatever LOSS said); --loss replaces it. */
    cfg.lossfn = default_loss;

    bool seen_species = false, seen_target = false, seen_weight = false;
    bool seen_hotvar  = false;
    std::string lattice = "bcc";
    std::string target_tok, hotvar_lo_tok, hotvar_hi_tok, hotvar_init_tok;

    auto need_value = [&](std::size_t& i, const std::string& flag) -> const std::string& {
        if (i + 1 >= tokens.size())
            throw std::runtime_error("PC_OBSERVABLE cohesive_energy: '" + flag
                                     + "' requires a value");
        return tokens[++i];
    };
    /* Each option is given once: a second would replace the first's
     * values. */
    std::vector<std::string> seen;
    auto once = [&](const std::string& flag) {
        if (std::find(seen.begin(), seen.end(), flag) != seen.end())
            throw std::runtime_error(
                "PC_OBSERVABLE cohesive_energy: " + flag + " is given twice "
                "on this line; each option is given once");
        seen.push_back(flag);
    };

    for (std::size_t i = start; i < tokens.size(); ++i) {
        const std::string& t = tokens[i];
        if (t == "--species") {
            once(t);
            cfg.species = need_value(i, t);
            /* An element symbol, in its case ("Ta"): every evaluation
             * failed on another word ("Unknown symbol: Xx"). Whether the
             * potential has the element is a startup check
             * (validate_hpo_setup). */
            tadah::core::PeriodicTable::initialize();
            try {
                (void)tadah::core::PeriodicTable::find_by_symbol(cfg.species);
            } catch (const std::exception&) {
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --species " + cfg.species +
                    " is not the symbol of an element");
            }
            seen_species = true;
        } else if (t == "--target") {
            once(t);
            target_tok = need_value(i, t);
            cfg.target = number(target_tok, t);
            seen_target = true;
        } else if (t == "--weight") {
            once(t);
            const std::string& v = need_value(i, t);
            cfg.weight = number(v, t);
            /* A weight below 0 would reward a worse cohesive energy. */
            if (cfg.weight < 0.0)
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --weight must be >= 0 "
                    "(got " + v + ")");
            seen_weight = true;
        } else if (t == "--lattice") {
            once(t);
            lattice = need_value(i, t);
            if (lattice != "bcc")
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --lattice takes bcc, the "
                    "one lattice of the term (got '" + lattice + "')");
        } else if (t == "--supercell") {
            once(t);
            /* At least one cell along each axis. */
            cfg.nx = integer(need_value(i, t), "--supercell <nx>", 1);
            cfg.ny = integer(need_value(i, t), "--supercell <ny>", 1);
            cfg.nz = integer(need_value(i, t), "--supercell <nz>", 1);
        } else if (t == "--hotvar") {
            once(t);
            /* --hotvar a0 <lo> <hi> [<init>] */
            const std::string& name = need_value(i, t);
            if (name != "a0")
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --hotvar names a0, the "
                    "lattice constant, the one variable of the term (got '" +
                    name + "')");
            hotvar_lo_tok = need_value(i, "--hotvar a0 <lo>");
            cfg.a0_min = number(hotvar_lo_tok, "--hotvar a0 <lo>");
            hotvar_hi_tok = need_value(i, "--hotvar a0 <hi>");
            cfg.a0_max = number(hotvar_hi_tok, "--hotvar a0 <hi>");
            /* The window of the relaxed a0 (the constructor keeps the
             * check as an invariant). */
            if (!(cfg.a0_max > cfg.a0_min))
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --hotvar a0 <hi> must be "
                    "greater than <lo> (got <lo> " + hotvar_lo_tok + ", <hi> " +
                    hotvar_hi_tok + ")");
            /* Optional initial; peek next token. */
            if (i + 1 < tokens.size() && tokens[i + 1].substr(0, 2) != "--") {
                hotvar_init_tok = tokens[++i];
                cfg.a0_initial = number(hotvar_init_tok, "--hotvar a0 <init>");
            }
            seen_hotvar = true;
        } else if (t == "--loss") {
            once(t);
            /* --loss <name> [<parameter>]: the parameter of HUBER (delta)
             * and TUKEY (c), as LOSS and --varloss take it
             * (LossFunction::resolve_params). */
            const std::string& kind = need_value(i, t);
            const auto type = tadah::LossFunction::from_name(
                kind, "PC_OBSERVABLE cohesive_energy: --loss");
            std::vector<double> params;
            while (i + 1 < tokens.size() && tokens[i + 1].substr(0, 2) != "--")
                params.push_back(
                    number(tokens[++i], "--loss " + kind + " <param>"));
            cfg.lossfn = tadah::LossFunction(type, params);
            cfg.lossfn_given = true;
        } else if (t == "--maxiter") {
            once(t);
            /* Caps NativeRelaxer iterations. Default is 5000 (deep
             * convergence). On randomly-initialised potentials, the
             * energy surface is often non-physical and the relaxer
             * wanders without converging; cap to a small budget
             * (50-200) for HPO use where the optimiser sweeps theta
             * before any pot is good enough for a true ground state. */
            cfg.relax_maxiter = static_cast<std::size_t>(
                integer(need_value(i, t), "--maxiter", 0));
        } else if (t == "--max-wall-time") {
            once(t);
            /* Per-relax wall-time cap (seconds). PC_OBSERVABLE's
             * analogue of LAMMPS's MAXTIME (per-eval subprocess kill).
             * With NN caching, a pathological theta can burn maxeval=5000
             * in ~5 s; 1.0 s matches the LAMMPS reference's MAXTIME=1. */
            cfg.relax_max_wall_time_s =
                number(need_value(i, t), "--max-wall-time");
        } else if (t == "--etol") {
            once(t);
            /* LAMMPS-style energy convergence tolerance, mirrors 1st
             * positional arg of `minimize etol ftol maxiter maxeval`.
             * Production runs use 1e-15 to match Ecoh_fast.in. Default
             * 0 = disabled (force-only convergence). */
            cfg.relax_etol =
                number(need_value(i, t), "--etol");
        } else if (t == "--maxeval") {
            once(t);
            /* LAMMPS-style total energy/force-eval cap. Mirrors the
             * 4th positional arg of LAMMPS `minimize etol ftol maxiter
             * maxeval`. Critical for HPO: without it, a pathological theta
             * that never converges does maxiter x maxiter_ls evals and
             * burns minutes per HPO iter. */
            cfg.relax_maxeval = static_cast<std::size_t>(
                integer(need_value(i, t), "--maxeval", 0));
        } else if (t == "--fmax-converged") {
            once(t);
            /* The force tolerance of the atoms in the relaxation and its
             * eq_ok test (matches Ecoh_fast.in's `ternary($(fmax) < 0.1,
             * 1, 0)`): the atoms have converged when the largest atomic
             * force falls below it, and a relaxation that ends above it
             * returns the --unconverged value as Ecoh. Positive: no
             * relaxation reaches a tolerance of 0. */
            const std::string& v = need_value(i, t);
            cfg.eq_fmax_threshold = number(v, "--fmax-converged");
            if (!(cfg.eq_fmax_threshold > 0.0))
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --fmax-converged must be "
                    "> 0 (got " + v + ")");
        } else if (t == "--ptol") {
            once(t);
            /* The pressure tolerance of the cell in the relaxation, in GPa.
             * Positive: no relaxation reaches a pressure of 0. */
            const std::string& v = need_value(i, t);
            cfg.relax_ptol = number(v, "--ptol");
            if (!(cfg.relax_ptol > 0.0))
                throw std::runtime_error(
                    "PC_OBSERVABLE cohesive_energy: --ptol must be > 0 (got " +
                    v + ")");
        } else if (t == "--unconverged") {
            once(t);
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
            cfg.unconverged_value = number(
                need_value(i, "--unconverged " + kind + " <value>"),
                "--unconverged " + kind);
        } else {
            throw std::runtime_error(
                "PC_OBSERVABLE cohesive_energy: unknown option '" + t + "'");
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
    /* The start of a chain's first relaxation lies in the window its
     * relaxed a0 must stay in; it was moved onto the nearer bound without
     * a word. */
    if (cfg.a0_initial && (*cfg.a0_initial < cfg.a0_min ||
                           *cfg.a0_initial > cfg.a0_max) &&
        cfg.a0_max > cfg.a0_min)
        throw std::runtime_error(
            "PC_OBSERVABLE cohesive_energy: --hotvar a0 <init> " +
            hotvar_init_tok + " is outside its window [" + hotvar_lo_tok +
            ", " + hotvar_hi_tok + "]");
    /* A target the loss function cannot measure (MAPE 0, RMSLE at or
     * below -1e-12): every evaluation would score the term the same, or
     * fail it. */
    if (const std::string why = cfg.lossfn.target_refusal(cfg.target);
        !why.empty())
        throw std::runtime_error(
            "PC_OBSERVABLE cohesive_energy --target " + target_tok + ": " + why);

    return std::make_unique<CohesiveEnergyObservable>(std::move(cfg));
}

}  // anonymous namespace

std::unique_ptr<Observable> parse_pc_observable_line(
    const std::string& cmd, const tadah::LossFunction& default_loss)
{
    auto tokens = ws_tokens(cmd);
    if (tokens.empty())
        throw std::runtime_error("PC_OBSERVABLE: empty line");

    const std::string& kind = tokens[0];
    if (kind == "cohesive_energy") {
        return parse_cohesive_energy(tokens, /*start=*/1, default_loss);
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
