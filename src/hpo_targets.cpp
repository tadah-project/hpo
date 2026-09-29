#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/pc_worker_pool.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/scale_validation.h>
#include <tadah/hpo/strategy_rules.h>
#include <tadah/hpo/strict_parse.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <cctype>
#include <cstdlib>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/core/registry.h>
#include <tadah/core/number_format.h>
#include <tadah/hpo/loss_function.h>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tadah {
namespace hpo {

namespace fs = std::filesystem;

namespace {

/* The density-gate lines (RHOGATE, RHOBARGATE): strict grammar. Every
 * number is in the number grammar of every hpotarget key (strict_parse.h):
 * a real one whole finite decimal token (parse_finite), NPTS an integer in
 * decimal digits (parse_uint). Keywords and their values are
 * case-insensitive, and an unknown, repeated or value-less keyword is an
 * error, so a typo cannot pass silently. read() names the line. */
double gate_number(const std::string& tok, const std::string& what) {
  return parse_finite(tok, what);
}

std::string gate_upper(std::string s) {
  for (auto& ch : s) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
  return s;
}

/* The optional keyword pairs of a gate line from tokens[first] on: NPTS
 * (both gates, [npts_min, 1e7]), SEED ERROR|WARN (both), R_LO (RHOGATE). */
void parse_gate_options(const std::string& key, const std::string& grammar,
                        const std::vector<std::string>& toks, std::size_t first,
                        std::size_t npts_min, std::size_t& npts, bool& seed_warn,
                        std::optional<double>* r_lo) {
  std::vector<std::string> seen;
  for (std::size_t t = first; t < toks.size(); t += 2) {
    const std::string kw = gate_upper(toks[t]);
    const bool known = kw == "NPTS" || kw == "SEED" || (r_lo && kw == "R_LO");
    if (!known)
      throw std::runtime_error(key + ": unexpected token '" + toks[t] + "'; " + grammar);
    if (std::find(seen.begin(), seen.end(), kw) != seen.end())
      throw std::runtime_error(key + ": " + kw + " given more than once; " + grammar);
    seen.push_back(kw);
    if (t + 1 >= toks.size())
      throw std::runtime_error(key + ": " + kw + " needs a value; " + grammar);
    const std::string& val = toks[t + 1];
    if (kw == "NPTS") {
      npts = static_cast<std::size_t>(
          parse_uint(val, key + " NPTS", npts_min, 10000000));
    } else if (kw == "SEED") {
      const std::string m = gate_upper(val);
      if (m != "ERROR" && m != "WARN")
        throw std::runtime_error(key + " SEED must be ERROR or WARN, got '" + val + "'");
      seed_warn = m == "WARN";
    } else {
      const double v = gate_number(val, key + " R_LO");
      if (v < 0.0)
        throw std::runtime_error(key + " R_LO must be >= 0, got '" + val + "'");
      *r_lo = v;
    }
  }
}

RhoGateSpec parse_rhogate(std::istringstream& iss) {
  static const std::string grammar =
      "expected RHOGATE <tol> [R_LO <r>] [NPTS <n>] [SEED ERROR|WARN]";
  std::vector<std::string> toks;
  for (std::string t; iss >> t;) toks.push_back(t);
  if (toks.empty())
    throw std::runtime_error("RHOGATE: the tolerance is missing; " + grammar);
  RhoGateSpec s;
  s.tol = gate_number(toks[0], "RHOGATE tol");
  if (s.tol < 0.0)
    throw std::runtime_error("RHOGATE tol must be >= 0, got '" + toks[0] + "'");
  parse_gate_options("RHOGATE", grammar, toks, 1, 2, s.npts, s.seed_warn, &s.r_lo);
  return s;
}

RhoBarGateSpec parse_rhobargate(std::istringstream& iss) {
  static const std::string grammar =
      "expected RHOBARGATE <bcc|fcc|hcp> <a0> <f_lo> <f_hi> [NPTS <n>] "
      "[SEED ERROR|WARN]";
  std::vector<std::string> toks;
  for (std::string t; iss >> t;) toks.push_back(t);
  if (toks.size() < 4)
    throw std::runtime_error("RHOBARGATE: the lattice, a0, f_lo and f_hi are "
                             "required; " + grammar);
  RhoBarGateSpec s;
  s.lattice = toks[0];
  for (auto& ch : s.lattice) ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (s.lattice != "bcc" && s.lattice != "fcc" && s.lattice != "hcp")
    throw std::runtime_error("RHOBARGATE lattice must be bcc, fcc or hcp, got '" +
                             toks[0] + "'");
  s.a0   = gate_number(toks[1], "RHOBARGATE a0");
  s.f_lo = gate_number(toks[2], "RHOBARGATE f_lo");
  s.f_hi = gate_number(toks[3], "RHOBARGATE f_hi");
  if (!(s.a0 > 0.0))
    throw std::runtime_error("RHOBARGATE a0 must be > 0, got '" + toks[1] + "'");
  if (!(s.f_lo > 0.0) || !(s.f_lo < s.f_hi))
    throw std::runtime_error("RHOBARGATE needs 0 < f_lo < f_hi, got f_lo '" +
                             toks[2] + "', f_hi '" + toks[3] + "'");
  parse_gate_options("RHOBARGATE", grammar, toks, 4, 3, s.npts, s.seed_warn, nullptr);
  return s;
}

}  // namespace

size_t HPOTargets::ui_offset(std::vector<std::string> &tokens, size_t ui) {
  size_t offset = 0;
  size_t count = 0;
  for (const auto& t : tokens) {
    if (core::is_number(t)) count++;
    if (count == ui) return offset;
    offset++;
  }
  return offset;
}

namespace {
// OPTIM and indexed PC lines share the same parenthesized selector grammar.
std::string read_index_selector(std::istringstream& iss) {
    iss >> std::ws;
    if (iss.get() != '(') throw std::runtime_error("Expected (indices)");
    std::string indices;
    if (!std::getline(iss, indices, ')') || iss.eof())
        throw std::runtime_error("Unmatched opening parenthesis in index selector");
    if (indices.find('(') != std::string::npos)
        throw std::runtime_error("Nested parentheses in index selector");
    if (core::trim(indices).empty()) throw std::runtime_error("Empty index selector");
    const int next = iss.peek();
    if (next != std::char_traits<char>::eof() && !std::isspace(static_cast<unsigned char>(next)))
        throw std::runtime_error("Expected whitespace after (indices)");
    return indices;
}

/* True when a METHOD of the chain draws from the chain's SEED: RANDOM,
 * LHS and SOBOL. CONFIG and WARM take no random numbers, and an empty
 * chain falls back to CONFIG. */
bool chain_draws_from_seed(const InitStrategySpec& ispec) {
    return std::any_of(ispec.chain.begin(), ispec.chain.end(),
                       [](const InitStrategySpec::Entry& e) {
                           return e.method == "RANDOM" || e.method == "LHS" ||
                                  e.method == "SOBOL";
                       });
}

/* "CONFIG, WARM" -- the chain's METHODs -- or "empty -> CONFIG fallback". */
std::string chain_methods(const InitStrategySpec& ispec) {
    if (ispec.chain.empty()) return "empty -> CONFIG fallback";
    std::string m;
    for (const auto& e : ispec.chain) {
        if (!m.empty()) m += ", ";
        m += e.method;
    }
    return m;
}
}

std::vector<std::string> HPOTargets::parse_optim_line(std::istringstream& iss) {
    const auto indices = read_index_selector(iss);
    std::vector<std::string> elements;
    std::string token;
    while (iss >> token) {
        if (token.find_first_of("()") != std::string::npos)
            throw std::runtime_error("Only one (indices) selector is accepted");
        elements.push_back(token);
    }
    elements.push_back(indices);
    return elements;
}

void HPOTargets::Opti::resize(size_t s) {
  if (opt.size() >= s) return;
  opt.resize(s, false);
  line.resize(s, 0);
  config.resize(s);
  scale.resize(s);
  origin.resize(s, BoxOrigin::Explicit);
  lowbound.resize(s, 0);
  highbound.resize(s, 0);
}

HPOTargets::HPOTargets(core::Context &c, std::string fn) : context(c) {
  /* Potential-file precision. Resolved first, before anything
   * below can default-insert NUMERIC into the context: only a NUMERIC the
   * configuration really carries counts as explicit. */
  {
    const core::PotDigits pd = core::resolve_pot_digits(context);
    pot_digits = pd.digits;
    pot_digits_explicit = pd.explicit_;
    const std::string w = core::pot_digits_warning(pd);
    if (!w.empty()) {
      const std::string msg =
          "HPO potential files (best_pot.tadah, BDUMP/DUMP archives): " + w +
          " Potential files written by tadah 1.5.0 hpo carry an "
          "auto-inserted 'NUMERIC 12' line; if this configuration is (or "
          "was copied from) such a best_pot.tadah, delete that line.";
      std::cerr << "[HPO] WARNING: " << msg << "\n";
      HPOLog::instance().warning(msg);
    }
  }

  models::D_mJoin::expand_grids(context);

  /* RHOFORM (the DM_REAM density form) is mandatory and has no default:
   * read it before the OPTIM lines (non-negative AMPGRIDMB boxes) and the
   * WARM seeds (form check) that depend on it. Stored value only. */
  if (models::context_has_ream(context))
    rho_form_ = models::require_rhoform(context, "tadah hpo");

  /* LS keeps configuration order: the outvar/loss columns, the
   * per-evaluation rows and the potential-file header are emitted in LS
   * order, and failprobe_idx is recorded against it at parse time. */
  read(fn);
  /* ---------- search space ----------------------------------------
   * One SearchSpace from the OPTIM lines (real boxes + per-line shape
   * tags), OPTIM_SCALE and LOG_HP; bound to the outer spec and, below,
   * to every sub-spec. spec.low/high are the INTERNAL bounds. */
  /* Env override TADAH_OPTIM_SCALE=AUTO|NONE (harness A/B of the
   * coordinate system; the reference-suite gate runs every example both
   * ways). Read here because the space is built in this constructor. */
  if (const char* e = std::getenv("TADAH_OPTIM_SCALE")) {
    std::string v = e;
    for (auto& ch : v) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if      (v == "AUTO") scale_policy_.normalise = true;
    else if (v == "NONE") scale_policy_.normalise = false;
    else throw std::runtime_error("TADAH_OPTIM_SCALE must be AUTO or NONE (got '" +
                                  std::string(e) + "')");
    std::ostringstream oss;
    oss << "OPTIM_SCALE " << v << " taken from the TADAH_OPTIM_SCALE environment "
           "override" << (scale_policy_set_ ? " (config value ignored)" : "");
    std::cerr << "[HPO] " << oss.str() << "\n";
    HPOLog::instance().info(oss.str());
    scale_policy_set_ = true;
  }
  space_ = std::make_shared<const SearchSpace>(build_search_space());
  spec.scale = space_->policy();
  spec.bind_space(space_);

  /* Optimisers need to tell a real loss from the failed-evaluation
   * sentinel (e.g. the ANNEAL T0 calibration excludes FAILSCORE
   * probes from its statistics). */
  spec.failscore = FAIL_SCORE;

  /* ---------- NOISE_AUTO: heuristic for finite-difference fd_step ---
   *
   * For DLIB:BFGS the user can ask the engine to pick a sensible
   * fd_step from the search-space geometry. Heuristic:
   *
   *     fd_step = max(1e-3 * box_diagonal / sqrt(dim), 1e-6)
   *
   * where box_diagonal is the L2 norm of (high - low) in the
   * post-transform space. The hand-set PARAM fd_step always wins so
   * users keep an escape hatch. */
  if (spec.noise.autodetect) {
    /* fd_step is only consulted by gradient-based backends that take
     * finite-difference derivatives (today: DLIB:BFGS via
     * dlib::derivative). The caps registry flags those; for everything
     * else the auto-derived fd_step would be a dead variable. read()
     * rejects a configuration without LIB/ALGO. */
    const bool gradient_based = optimizer_caps(spec).is_gradient_based;
    if (!gradient_based) {
      std::cerr << "[HPO] NOISE_AUTO derives PARAM fd_step for gradient-"
                << "based optimisers only (current: " << spec.lib << ":"
                << spec.algo << "); ignored\n";
    } else if (spec.param.count("fd_step")) {
      std::cerr << "[HPO] NOISE_AUTO: PARAM fd_step already set "
                   "explicitly; user value retained\n";
    } else if (spec.low.empty()) {
      std::cerr << "[HPO] NOISE_AUTO: no optimisable dimensions; "
                   "skipping\n";
    } else {
      double sq = 0.0;
      for (std::size_t i = 0; i < spec.low.size(); ++i) {
        const double w = spec.high[i] - spec.low[i];
        sq += w * w;
      }
      const double diag = std::sqrt(sq);
      const double dim = static_cast<double>(spec.low.size());
      double fd = 1e-3 * diag / std::sqrt(dim);
      if (!(fd > 1e-6)) fd = 1e-6;
      spec.param["fd_step"] = fd;
      std::ostringstream oss;
      oss << "NOISE_AUTO: PARAM fd_step set to " << fd
          << " (diag=" << diag << ", dim=" << spec.low.size() << ")";
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
  }

  /* Propagate the search space into every sub-spec FIRST, before
   * resolving any INIT_STRATEGY chain on them: strategies like RANDOM/
   * LHS/SOBOL sample within bounds, so an unbound sub-spec produces
   * 0-dim points. (Bound inheritance must precede init resolution.) */
  bind_space_recursive();

  /* ---------- resolve INIT_STRATEGY chain into optim_init -----------
   *
   * Apply to OUTER, PRESTAGE, and PRESTAGE->LOCAL (LOCAL inside OUTER
   * is the NLopt MLSL local sub-algo and inherits the outer's
   * starting point -- it does not need its own chain). Each spec's
   * INIT block (STRATEGY / SEED / K_TOTAL / WARM_ON_MISMATCH lines)
   * is honoured independently of every other spec.
   *
   * Diagnostics tag the location ("OUTER" / "PRESTAGE" / "PRESTAGE.LOCAL")
   * so the user can see which block fired which strategy. */
  auto resolve_init = [&](OptimizerSpec& s, const char* where) {
    auto strategy = build_init_strategy(s.init_strategy_spec);
    /* A SEED with no seed-consuming METHOD in the chain is a silent
     * no-op for the starting point: an empty chain falls back to
     * CONFIG, and CONFIG/WARM ignore ctx.seed. For an NLopt phase without
     * a SEED of its own the seed still reaches nlopt::srand in make_nlopt,
     * but only stochastic non-LDS NLopt algos (G_MLSL, CRS, ISRES, ESCH,
     * ...) ever read that RNG. Surface it so "I set SEED and nothing
     * changed" is answerable from the log. */
    /* SEED inside the INIT block seeds the init-strategy chain -- and
     * NLopt only as a fallback, when the phase has no SEED of its own
     * (make_nlopt); it does not seed any other optimiser. For the ANNEAL
     * family (whose proposal stream falls back to random_device without
     * an optimiser-level SEED) this is a classic footgun: the user "set a
     * seed" yet gets a non-reproducible run. */
    if (!s.seed && s.init_strategy_spec.seed &&
        ((s.lib == "TADAH" && s.algo.rfind("ANNEAL", 0) == 0) ||
         s.lib == "ASA")) {
      std::ostringstream oss;
      oss << "INIT[" << where << "]: SEED inside the INIT block does not "
             "seed " << s.lib << ":" << s.algo << " -- the annealer's "
             "proposal stream falls back to random_device "
             "(non-reproducible). "
             "Set SEED at the OPTIMIZER/STRATEGY level as well.";
      std::cerr << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().warning(oss.str());
    }
    if (s.init_strategy_spec.seed) {
      if (!chain_draws_from_seed(s.init_strategy_spec)) {
        std::ostringstream oss;
        oss << "INIT[" << where << "]: SEED is set but the strategy chain ("
            << chain_methods(s.init_strategy_spec)
            << ") contains no stochastic method, so the seed does not "
               "affect the starting point. Add METHOD RANDOM/LHS/SOBOL "
               "to randomise it.";
        if (s.lib == "NLOPT" && !s.seed)
          oss << " (Without a SEED of its own, the NLopt optimiser still "
                 "takes its random seed from it; only stochastic non-LDS "
                 "NLopt algorithms read one.)";
        std::cerr << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }
    std::uint64_t seed = s.init_strategy_spec.seed.value_or(
        static_cast<std::uint64_t>(std::random_device{}()));
    InitContext ictx{s, *this, seed, &context};
    std::size_t k = s.init_strategy_spec.k_total.value_or(strategy->default_k());
    /* with the provenance of every point (CONFIG, WARM <file>, RANDOM, LHS,
     * SOBOL; an empty chain is CONFIG) */
    s.optim_init = strategy->generate_tagged(ictx, k, s.optim_init_origin);
    if (s.optim_init.empty()) {
      /* A non-empty chain can still come back empty (e.g. WARM with a
       * missing file under WARM_ON_MISMATCH warn/skip and no further
       * strategy behind it). The optimiser then starts from ITS OWN
       * default -- the ANNEAL family uses the OPTIM box centre -- which
       * is almost never what a WARM user intended. Say so explicitly;
       * WARM_ON_MISMATCH error makes it fatal instead. */
      std::ostringstream oss;
      oss << "InitStrategy[" << where << "] " << strategy->name()
          << " produced 0 starting points -- the optimiser falls back to "
             "its default start (ANNEAL family: the OPTIM box CENTRE, not "
             "your seed). Set WARM_ON_MISMATCH error in the INIT block to "
             "make this fatal.";
      std::cerr << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().warning(oss.str());
    }
    if (!s.optim_init.empty()) {
      std::ostringstream oss;
      oss << "InitStrategy[" << where << "] " << strategy->name()
          << " produced " << s.optim_init.size()
          << " starting point" << (s.optim_init.size()==1 ? "" : "s");
      std::cout << "[HPO] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    return strategy;
  };

  resolve_init(spec, "OUTER");

  if (!spec.optim_init.empty()) {
    /* Warn when the resolved OUTER chain produced more points than
     * the OUTER optimiser will actually consume. Backends declare
     * accepts_multi_seed=true in the caps registry when a single run
     * ingests >1 seed; strategy decorators (MULTISTART/PRESTAGE) consume
     * the chain too. Every other path drops everything past
     * optim_init.front(). */
    const bool strategy_consumes =
        (spec.strategy_type == StrategyType::MULTISTART && spec.multistart.n_starts >= 2) ||
        (spec.strategy_type == StrategyType::PRESTAGE && spec.prestage != nullptr);
    const bool backend_consumes = optimizer_caps(spec).accepts_multi_seed;
    const bool multi_seed_consumer = strategy_consumes || backend_consumes;
    if (spec.optim_init.size() > 1 && !multi_seed_consumer) {
      /* Surface compatible options from the live caps registry rather
       * than hard-coding LIB:ALGO names -- the registry is the single
       * source of truth for who consumes >1 seed in one run. */
      auto multi_seed = filter_optimizer_caps(
          [](const OptimizerCaps& c) { return c.accepts_multi_seed; });
      std::cerr << "[HPO] WARNING: INIT chain produced "
                << spec.optim_init.size() << " seeds but " << spec.lib
                << ":" << spec.algo << " consumes only the first; "
                << (spec.optim_init.size() - 1)
                << " seed(s) will be discarded. "
                << "Use STRATEGY TYPE MULTISTART (N_STARTS >= 2) or TYPE "
                   "PRESTAGE, or pick a multi-seed backend (" << format_caps_list(multi_seed)
                << ") to use them.\n";
    }
  }

  if (spec.prestage) {
    /* If PRESTAGE has no explicit chain and no INIT_STRATEGY entries,
     * it inherits the OUTER's optim_init (legacy behaviour: a CONFIG
     * point flows in). If PRESTAGE has its own chain, that chain wins
     * and the OUTER's optim_init is *not* prepended -- PRESTAGE owns its
     * starting points. The same holds for PRESTAGE's nested INNER. */
    if (!spec.prestage->init_strategy_spec.chain.empty())
      resolve_init(*spec.prestage, "PRESTAGE");
    if (spec.prestage->inner &&
        !spec.prestage->inner->init_strategy_spec.chain.empty())
      resolve_init(*spec.prestage->inner, "PRESTAGE.INNER");
    spec.share_init_with_nested_phases();
  }

  /* The RESTART block's SEED seeds the METHOD chain that draws each
   * restart's points (round r with SEED + r, RestartPolicy). Like an INIT
   * block's SEED, it does nothing without a RANDOM/LHS/SOBOL method; say so,
   * when the RESTART block is active at all. */
  if (spec.restart.active() && spec.restart.init_strategy.seed &&
      !chain_draws_from_seed(spec.restart.init_strategy)) {
    std::ostringstream oss;
    oss << "RESTART: SEED is set but the METHOD chain ("
        << chain_methods(spec.restart.init_strategy)
        << ") contains no stochastic method, so the seed does not affect "
           "the restart points. Add METHOD RANDOM/LHS/SOBOL to randomise "
           "them.";
    std::cerr << "[HPO] " << oss.str() << "\n";
    HPOLog::instance().warning(oss.str());
  }

  /* The RESTART chain's seeds (CONFIG, WARM), resolved once here as a
   * round draws them (RestartPolicy: the outer spec, the run context, the
   * block's K_TOTAL), so a WARM file that is missing or unreadable, or
   * whose RHOFORM the run refuses, is a startup error and not the first
   * stagnation's, the EMBF0 note comes at startup, and the density gates
   * check these points as seeds (collect_gate_seeds). The sampled points
   * of the draw are dropped: each round draws its own. */
  if (spec.restart.active()) {
    const InitStrategySpec& rs = spec.restart.init_strategy;
    auto refresh = build_init_strategy(rs);
    InitContext ictx{spec, *this, rs.seed.value_or(0), &context};
    std::vector<InitOrigin> origins;
    const auto pts = refresh->generate_tagged(
        ictx, rs.k_total.value_or(refresh->default_k()), origins);
    for (std::size_t i = 0; i < pts.size() && i < origins.size(); ++i) {
      if (!origins[i].is_seed()) continue;
      spec.restart.seed_points.push_back(pts[i]);
      spec.restart.seed_origins.push_back(origins[i]);
    }
  }

  /* Fail fast on the initial-step policy (registry-driven). Seeds vs
   * box and the width audit run in validate_hpo_setup(), with the other
   * startup rules. */
  run_scale_validation(spec, /*check_start_points=*/false);
}

// -------------------------------------------------------------------
// HPOTargets::read   - complete, no omissions
// -------------------------------------------------------------------
/* parse_bool() and dispatch_opt_attr() moved to optimizer_block.{h,cpp} —
 * shared with the `tadah refit` stage parser (RefitParser). The calls below
 * resolve to those tadah::hpo:: free functions. */

/* Helper: parse one INIT-block key/value into an InitStrategySpec.
 * Returns true if consumed; false otherwise.
 *
 * METHOD is the canonical chain-entry keyword. STRATEGY is kept as a
 * silent alias for older configs but is a poor fit syntactically: at
 * the OPTIMIZER level STRATEGY/ENDSTRATEGY opens a block, so reusing
 * it as a method-line key inside INIT (`STRATEGY RANDOM 1`) reads as
 * if a nested block follows. New configs should write
 *     INIT
 *       METHOD RANDOM 1
 *     ENDINIT
 * which has no such collision. */
static bool dispatch_init_attr(const std::string& key,
                               std::istringstream& iss,
                               InitStrategySpec& a)
{
    if (key=="METHOD" || key=="STRATEGY") {
        InitStrategySpec::Entry e;
        iss >> e.method;
        if (e.method.empty())
            throw std::runtime_error(key + " requires a method name");
        if (e.method == "NOISE_AUTO")
            throw std::runtime_error(
                "INIT." + key + " NOISE_AUTO is not implemented "
                "(reserved keyword). Use RANDOM/LHS/SOBOL/WARM/CONFIG.");
        std::string tok;
        while (iss >> tok) e.args.push_back(tok);
        /* Arity and K are checked here, at the line, not when the chain
         * is resolved after the whole file has been read. */
        validate_init_entry(e);
        a.chain.push_back(std::move(e));
        return true;
    }
    if (key=="K_TOTAL") {
        a.k_total = parse_uint(single_value(iss, key), key);
        return true;
    }
    if (key=="SEED") {
        a.seed = parse_uint(single_value(iss, key), key);
        return true;
    }
    if (key=="WARM_ON_MISMATCH") {
        const std::string v = single_value(iss, key);
        if (v!="warn" && v!="error" && v!="skip")
            throw std::runtime_error("WARM_ON_MISMATCH must be one of: warn, error, skip");
        a.warm_on_mismatch = v;
        return true;
    }
    return false;
}

/* An INNER block configures the local search that NLopt's MLSL and AUGLAG
 * run inside the enclosing block's NLopt run. It takes every optimiser
 * attribute but two, which belong to the enclosing block:
 *   - SEED: NLopt has one random stream per run, which make_nlopt seeds
 *     from the enclosing block (its SEED, else its INIT block's SEED), and
 *     the INNER search draws from that stream. An INNER SEED could only
 *     reseed it in place of the enclosing block's SEED.
 *   - GRAD_STRATEGY: MLSL and AUGLAG hand their local search the gradient
 *     of their own objective, which the enclosing block's GRAD_STRATEGY
 *     obtains (nlopt_adapter.h). An INNER GRAD_STRATEGY would never be
 *     used.
 * Returns why `key` is not accepted inside INNER, or nullptr when it is. */
static const char* inner_rejects_attr(const std::string& key)
{
    if (key == "SEED")
        return "NLopt draws the INNER search from the enclosing run's random "
               "stream";
    if (key == "GRAD_STRATEGY")
        return "the enclosing block's GRAD_STRATEGY applies to its local "
               "search";
    return nullptr;
}

static bool inner_takes_attr(const std::string& key)
{
    return inner_rejects_attr(key) == nullptr;
}

static std::string inner_attr_keys_joined(const char* sep)
{
    std::string out;
    for (const auto& k : opt_attr_keys()) {
        if (!inner_takes_attr(k)) continue;
        if (!out.empty()) out += sep;
        out += k;
    }
    return out;
}

/* Helper: throw a "renamed to X" error for any old-grammar keyword.
 * Returns true if the key was a known-old keyword (caller should not
 * fall through to an "unknown key" error). */
static bool reject_old_keyword(const std::string& key)
{
    struct Rename { const char* old_; const char* hint; };
    static const Rename map[] = {
        {"LOCAL",                       "renamed to INNER (the MLSL/AUGLAG inner sub-optimizer block)."},
        {"ENDLOCAL",                    "renamed to ENDINNER."},
        {"PRESTAGE",                    "replaced by `STRATEGY { TYPE PRESTAGE EXPLORE { ... } [REFINE { ... }] }` (REFINE is optional; without it the STRATEGY-level LIB/ALGO refines)."},
        {"ENDPRESTAGE",                 "replaced by ENDEXPLORE / ENDREFINE / ENDSTRATEGY (see STRATEGY block)."},
        {"MULTISTART",                  "moved into `STRATEGY { TYPE MULTISTART N_STARTS N }`."},
        {"MULTISTART_PARALLEL",         "moved into `STRATEGY { TYPE MULTISTART PARALLEL N }`."},
        {"MULTISTART_BUDGET_PER_START", "moved into `STRATEGY { TYPE MULTISTART BUDGET_PER_START N }`."},
        {"STAGE_GLOBAL",                "removed. Use `STRATEGY { TYPE PRESTAGE ... }` for staged-global runs."},
        {"RESTART_AFTER_STAGNATION",    "moved into `STRATEGY { RESTART { AFTER_STAGNATION N } ENDRESTART }`."},
        {"RESTART_MAX",                 "moved into `STRATEGY { RESTART { MAX N } ENDRESTART }`."},
        {"RESTART_INIT_STRATEGY",       "moved into `RESTART { METHOD <method> ... }` (re-uses INIT-block grammar)."},
        {"INIT_STRATEGY",               "moved into `INIT { METHOD <method> ... } ENDINIT` block."},
        {"INIT_K",                      "renamed to K_TOTAL inside an INIT block."},
        {"INIT_SEED",                   "renamed to SEED inside an INIT block."},
        {"INIT_INCLUDE_CONFIG",         "removed. The in-config point is now included only when you list `METHOD CONFIG` explicitly inside the INIT block; omit it to leave the in-config point out."},
        {"INCLUDE_CONFIG",              "removed. The in-config point is now included only when you list `METHOD CONFIG` explicitly inside the INIT block; omit it to leave the in-config point out."},
        {"NOISE_AUTO",                  "moved into `NOISE { AUTO true|false } ENDNOISE` block."},
        {"NOISE_CALIBRATION",           "moved into `NOISE { CALIBRATION N } ENDNOISE` block."},
        /* TOPK is now legal inside STRATEGY (TYPE PRESTAGE) -- don't reject it
         * unconditionally; the STRATEGY block dispatch handles it. */
    };
    for (const auto& r : map) {
        if (key == r.old_)
            throw std::runtime_error(
                std::string("'") + key + "' was removed from the grammar: " + r.hint);
    }
    return false;
}

void HPOTargets::read(const std::string& fn)
{
    std::ifstream ifs(fn);
    if (!ifs) throw std::runtime_error("HPO file " + fn + " not found");

    /* Block state machine.
     *
     * Block hierarchy:
     *   ROOT
     *   +-- OUTER         OPTIMIZER ... ENDOPTIMIZER
     *       +-- INIT_BLK    INIT ... ENDINIT          (optional, populates spec.init_strategy_spec)
     *       +-- NOISE_BLK   NOISE ... ENDNOISE        (optional, populates spec.noise)
     *       +-- STRATEGY    STRATEGY ... ENDSTRATEGY  (drives StrategyType; PLAIN when absent)
     *           +-- EXPLORE     EXPLORE ... ENDEXPLORE    (only with TYPE PRESTAGE; spec.prestage)
     *           |   +-- INNER       INNER ... ENDINNER       (MLSL/AUGLAG inner sub-optimiser)
     *           +-- REFINE      REFINE ... ENDREFINE      (only with TYPE PRESTAGE; the OUTER spec)
     *           |   +-- INNER       INNER ... ENDINNER
     *           +-- INNER       INNER ... ENDINNER       (inner sub-optimiser of the STRATEGY-level LIB/ALGO)
     *           +-- RESTART     RESTART ... ENDRESTART   (decorator; populates spec.restart) */
    enum Block { ROOT, OUTER, INIT_BLK, NOISE_BLK, STRATEGY_BLK,
                 EXPLORE_BLK, REFINE_BLK, INNER_BLK, RESTART_BLK };
    Block            block      = ROOT;
    Block            prev_inner = ROOT;       // restored on ENDINNER
    OptimizerSpec*   active_opt    = nullptr;  // recipient of LIB/ALGO/...
    InitStrategySpec* active_init  = nullptr;  // recipient of STRATEGY/K_TOTAL/...
    OptimizerSpec*   inner_parent  = nullptr;  // outer spec of an INNER block

    /* STRATEGY keys that belong to one TYPE (strategy_rules.h), checked
     * when the STRATEGY block ends: TYPE may come after them. So are the
     * EXPLORE and REFINE blocks, which need TYPE PRESTAGE before them and
     * still at the end of the block. */
    struct TypeKeySeen { std::string key; std::size_t line; std::string text; };
    std::vector<TypeKeySeen> type_keys;
    std::vector<TypeKeySeen> prestage_blocks;
    bool strategy_type_given = false;
    /* The line of the STRATEGY block (0 before it): there is one. A second
     * block reopened the first and could change the TYPE its keys had been
     * checked against. */
    std::size_t strategy_line = 0;

    /* The block the open INNER block sits in: an INNER block opens only
     * inside STRATEGY, EXPLORE or REFINE. */
    auto inner_enclosing = [&]() -> std::string {
        return prev_inner == STRATEGY_BLK ? "STRATEGY" :
               prev_inner == EXPLORE_BLK  ? "EXPLORE"  : "REFINE";
    };

    /* Helper: re-derive active_opt for a non-INNER block. */
    auto refresh_active_opt = [&]() {
        switch (block) {
            case OUTER:        active_opt = &spec; break;
            case STRATEGY_BLK: active_opt = &spec; break;
            case EXPLORE_BLK:  /* PRESTAGE's one-pass exploration */
                active_opt = spec.prestage.get();
                break;
            case REFINE_BLK:   /* PRESTAGE -- REFINE is the OUTER spec */
                active_opt = &spec;
                break;
            case INIT_BLK:
            case NOISE_BLK:
            case RESTART_BLK:
            case INNER_BLK:
            case ROOT:
                active_opt = nullptr; break;
        }
    };

    std::string line;
    size_t      nline = 0;
    /* The density gate lines, for the arming rule at the end of the file
     * (an OPTIM line that arms a gate may follow it). */
    struct GateLine { const char* key; std::size_t line; std::string text; };
    std::vector<GateLine> gate_lines;

    /* Every parse error names the line it is about. */
    auto at_line = [](std::size_t n, const std::string& what,
                      const std::string& text) {
        return "Error on line " + core::to_string(n) + ": " + what +
               "\nLine: " + text;
    };
    /* An error about an earlier line than the one being read: a STRATEGY
     * key found at the end of its block to have no effect. */
    struct EarlierLineError : std::runtime_error {
        std::size_t line;
        std::string text;
        EarlierLineError(std::size_t l, std::string t, const std::string& what)
            : std::runtime_error(what), line(l), text(std::move(t)) {}
    };
    /* The keys of one TYPE, and the EXPLORE and REFINE blocks, must match
     * the TYPE the block ends with. */
    auto check_type_keys = [&]() {
        for (const auto& k : type_keys) {
            try {
                require_strategy_key(k.key, spec.strategy_type,
                                     strategy_type_given);
            } catch (const std::exception& e) {
                throw EarlierLineError(k.line, k.text, e.what());
            }
        }
        type_keys.clear();
        for (const auto& b : prestage_blocks) {
            try {
                require_prestage_block(b.key, spec.strategy_type);
            } catch (const std::exception& e) {
                throw EarlierLineError(b.line, b.text, e.what());
            }
        }
        prestage_blocks.clear();
    };

    auto strip_comment = [](std::string& s){
        auto p = s.find('#'); if (p!=std::string::npos) s.resize(p);
    };
    auto rtrim = [](std::string& s){
        s.erase(std::find_if(s.rbegin(), s.rend(),
                 [](int c){return !std::isspace(c);} ).base(), s.end());
    };

    while (std::getline(ifs,line))
    {
        ++nline;
        strip_comment(line);   rtrim(line);
        if (line.empty()) continue;

        while (!line.empty() && line.back()=='\\')
        {
            line.pop_back();
            std::string cont;
            if (!std::getline(ifs,cont))
                throw std::runtime_error("unfinished line "+std::to_string(nline));
            ++nline;  strip_comment(cont);  rtrim(cont);
            line += cont;
        }

        std::istringstream iss(line);
        std::string key;  iss>>key;
        if (key.empty()) continue;

        try {
            /* ============== block delimiters ====================== */

            if (key=="OPTIMIZER")    { block=OUTER; refresh_active_opt(); continue; }
            if (key=="ENDOPTIMIZER") {
                /* The STRATEGY-level optimiser is complete here: a short-form
                 * LIB/ALGO may follow the STRATEGY block. */
                require_driver(spec, DriverPhase::Run);
                block=ROOT;  active_opt=nullptr;  continue;
            }

            if (key=="INIT") {
                if (block != OUTER && block != STRATEGY_BLK &&
                    block != EXPLORE_BLK && block != REFINE_BLK)
                    throw std::runtime_error(
                        "INIT block must appear inside OPTIMIZER, STRATEGY, EXPLORE, or REFINE");
                /* INIT populates the active OptimizerSpec's init_strategy_spec. */
                if (!active_opt)
                    throw std::runtime_error("INIT block requires an active OptimizerSpec");
                active_init = &active_opt->init_strategy_spec;
                /* Suspend active_opt so SEED/STRATEGY don't get gobbled by
                 * dispatch_opt_attr; restored on ENDINIT. */
                inner_parent = active_opt;     // reuse field for stash
                active_opt   = nullptr;
                prev_inner   = block;
                block = INIT_BLK;
                continue;
            }
            if (key=="ENDINIT") {
                if (block != INIT_BLK)
                    throw std::runtime_error("ENDINIT without matching INIT");
                active_init = nullptr;
                block = prev_inner;
                active_opt = inner_parent;
                inner_parent = nullptr;
                continue;
            }

            if (key=="NOISE") {
                if (block != OUTER)
                    throw std::runtime_error("NOISE block must appear inside OPTIMIZER (OUTER level)");
                block = NOISE_BLK;
                continue;
            }
            if (key=="ENDNOISE") {
                if (block != NOISE_BLK)
                    throw std::runtime_error("ENDNOISE without matching NOISE");
                block = OUTER;
                refresh_active_opt();
                continue;
            }

            if (key=="STRATEGY") {
                /* STRATEGY has two roles:
                 *   - Inside INIT/RESTART, it's a deprecated alias for
                 *     METHOD that adds a chain entry. Kept silent so old
                 *     configs keep parsing; new configs should use METHOD
                 *     to avoid the keyword collision with the OUTER block.
                 *   - At OUTER level, it opens the STRATEGY block.
                 * Discriminate by context. */
                if (block == INIT_BLK || block == RESTART_BLK) {
                    if (active_init && dispatch_init_attr(key, iss, *active_init))
                        continue;
                    /* fall through to error */
                }
                if (block != OUTER)
                    throw std::runtime_error("STRATEGY block must appear inside OPTIMIZER (OUTER level)");
                if (strategy_line != 0)
                    throw std::runtime_error(
                        "STRATEGY block given more than once (the first opens "
                        "on line " + std::to_string(strategy_line) + "); write "
                        "every STRATEGY key in one block");
                strategy_line = nline;
                block = STRATEGY_BLK;
                refresh_active_opt();
                continue;
            }
            if (key=="ENDSTRATEGY") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("ENDSTRATEGY without matching STRATEGY");
                check_type_keys();
                block = OUTER;
                refresh_active_opt();
                continue;
            }

            if (key=="EXPLORE") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("EXPLORE block must appear inside STRATEGY");
                if (spec.strategy_type != StrategyType::PRESTAGE)
                    throw std::runtime_error(
                        "EXPLORE block is only valid with TYPE PRESTAGE "
                        "(declare TYPE before EXPLORE)");
                if (!spec.prestage)
                    spec.prestage = std::make_unique<OptimizerSpec>();
                prestage_blocks.push_back({key, nline, line});
                block = EXPLORE_BLK;
                refresh_active_opt();
                continue;
            }
            if (key=="ENDEXPLORE") {
                if (block != EXPLORE_BLK)
                    throw std::runtime_error("ENDEXPLORE without matching EXPLORE");
                require_driver(*spec.prestage, DriverPhase::Explore);
                block = STRATEGY_BLK;
                refresh_active_opt();
                continue;
            }

            if (key=="REFINE") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("REFINE block must appear inside STRATEGY");
                if (spec.strategy_type != StrategyType::PRESTAGE)
                    throw std::runtime_error(
                        "REFINE block is only valid with TYPE PRESTAGE "
                        "(declare TYPE before REFINE)");
                prestage_blocks.push_back({key, nline, line});
                block = REFINE_BLK;
                refresh_active_opt();
                continue;
            }
            if (key=="ENDREFINE") {
                if (block != REFINE_BLK)
                    throw std::runtime_error("ENDREFINE without matching REFINE");
                block = STRATEGY_BLK;
                refresh_active_opt();
                continue;
            }

            if (key=="INNER") {
                if (block != STRATEGY_BLK && block != EXPLORE_BLK && block != REFINE_BLK)
                    throw std::runtime_error(
                        "INNER block must appear inside STRATEGY, EXPLORE, or REFINE "
                        "(it configures the MLSL/AUGLAG inner local sub-optimiser)");
                if (!active_opt)
                    throw std::runtime_error("INNER block has no parent OptimizerSpec");
                if (active_opt->inner)
                    throw std::runtime_error("INNER block specified more than once in this scope");
                active_opt->inner = std::make_unique<OptimizerSpec>();
                inner_parent = active_opt;
                active_opt = active_opt->inner.get();
                prev_inner = block;
                block = INNER_BLK;
                continue;
            }
            if (key=="ENDINNER") {
                if (block != INNER_BLK)
                    throw std::runtime_error("ENDINNER without matching INNER");
                /* The INNER optimiser is complete here, like an EXPLORE
                 * block's at ENDEXPLORE. */
                require_inner_driver(*active_opt, inner_enclosing());
                block = prev_inner;
                inner_parent = nullptr;
                refresh_active_opt();
                continue;
            }

            if (key=="RESTART") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("RESTART block must appear inside STRATEGY");
                active_init = &spec.restart.init_strategy;
                /* Suspend active_opt while inside RESTART so SEED/STRATEGY
                 * go to active_init, not the OUTER spec. A RESTART with no
                 * STRATEGY lines falls back to a single CONFIG point. */
                inner_parent = active_opt;
                active_opt   = nullptr;
                prev_inner   = block;
                block = RESTART_BLK;
                continue;
            }
            if (key=="ENDRESTART") {
                if (block != RESTART_BLK)
                    throw std::runtime_error("ENDRESTART without matching RESTART");
                active_init = nullptr;
                block = prev_inner;
                active_opt = inner_parent;
                inner_parent = nullptr;
                continue;
            }

            /* ============== old-grammar rejection ================= */
            reject_old_keyword(key);

            /* ============== block-specific dispatch =============== */

            /* SEED and GRAD_STRATEGY inside INNER: the INNER search draws
             * from the random stream of the enclosing block's run and
             * takes the gradient its GRAD_STRATEGY obtains
             * (inner_rejects_attr). */
            if (block == INNER_BLK && !inner_takes_attr(key)) {
                throw std::runtime_error(
                    key + " is not accepted inside an INNER block: " +
                    inner_rejects_attr(key) + "; set " + key +
                    " on the enclosing " + inner_enclosing() + " block");
            }

            /* Common opt-attribute keys (LIB/ALGO/MAXEVAL/...) accepted in
             * any block where active_opt is set. */
            if (active_opt && dispatch_opt_attr(key, iss, *active_opt)) {
                /* Strict arity. dispatch_opt_attr() reads exactly the value
                 * tokens its key defines, so anything still on the line is a
                 * typo, a stray word, or a value for a key that does not take
                 * one -- e.g. `PARAM STEP_FRAC 0.15 0.3`, or the pre-1.5.0
                 * `PARAM INIT_MODE JITTER AUTO 0.05` on a build that does not
                 * know AUTO. All of those were silently discarded before.
                 * This is safe only here: HPOTargets::read() is line-framed,
                 * whereas read_optimizer_block() sees a flat token stream in
                 * which "leftovers" are simply the next key. */
                std::string extra_tok;
                if (iss >> extra_tok)
                    throw std::runtime_error(
                        key + " takes no further value here (unexpected '" +
                        extra_tok + "')");
                continue;
            }

            /* INIT / RESTART block keys */
            if (block == INIT_BLK || block == RESTART_BLK) {
                if (active_init && dispatch_init_attr(key, iss, *active_init))
                    continue;
                if (block == RESTART_BLK) {
                    if (key=="AFTER_STAGNATION") {
                        spec.restart.after_stagnation =
                            parse_uint(single_value(iss, key), key);
                        continue;
                    }
                    if (key=="MAX") {
                        spec.restart.max = parse_uint(single_value(iss, key), key);
                        continue;
                    }
                }
                throw std::runtime_error("unknown key '" + key + "' inside "
                    + std::string(block==INIT_BLK ? "INIT" : "RESTART") + " block");
            }

            /* NOISE block keys */
            if (block == NOISE_BLK) {
                if (key=="AUTO") {
                    /* AUTO alone means AUTO true. */
                    std::string v;
                    iss >> v;
                    expect_end(iss, "AUTO takes a single value");
                    spec.noise.autodetect = v.empty() ? true : parse_bool(v, "NOISE.AUTO");
                    continue;
                }
                if (key=="CALIBRATION") {
                    const std::size_t v = parse_uint(single_value(iss, key), key);
                    if (v == 1)
                        throw std::runtime_error("NOISE.CALIBRATION must be 0 (off) or >= 2");
                    spec.noise.calibration_n = v;
                    continue;
                }
                throw std::runtime_error("unknown key '" + key + "' inside NOISE block");
            }

            /* LOG_HP is a property of the search space (one transform for
             * the whole run): accepted once at OPTIMIZER (short form) or
             * STRATEGY level -- the docs always promised both, the parser
             * used to accept STRATEGY only and throw 'unknown key' at
             * OUTER -- and rejected inside stage / INNER / INIT blocks. */
            if (key=="LOG_HP") {
                if (block == OUTER || block == STRATEGY_BLK) {
                    const std::string tok = single_value(iss, key);
                    const double v = parse_finite(tok, key);
                    if (!(v >= 1.0))
                        throw std::runtime_error(
                            "LOG_HP must be a finite ratio >= 1 (a dimension is "
                            "log-scaled when high/low >= LOG_HP and low > 0), "
                            "got '" + tok + "'");
                    if (spec.log_threshold) {
                        std::ostringstream oss;
                        oss << "LOG_HP given more than once; the last value ("
                            << v << ") wins";
                        std::cerr << "[HPO] " << oss.str() << "\n";
                        HPOLog::instance().warning(oss.str());
                    }
                    spec.log_threshold = v;
                    continue;
                }
                throw std::runtime_error(
                    "LOG_HP is a search-space property; declare it once at the "
                    "OPTIMIZER or STRATEGY level, not inside a stage, INNER or "
                    "INIT block");
            }

            /* STRATEGY block keys */
            if (block == STRATEGY_BLK) {
                if (key=="TYPE") {
                    const std::string t = single_value(iss, key);
                    if      (t=="PLAIN")      spec.strategy_type = StrategyType::PLAIN;
                    else if (t=="MULTISTART") spec.strategy_type = StrategyType::MULTISTART;
                    else if (t=="PRESTAGE")   spec.strategy_type = StrategyType::PRESTAGE;
                    else throw std::runtime_error("STRATEGY TYPE must be PLAIN, MULTISTART, or PRESTAGE (got '" + t + "')");
                    strategy_type_given = true;
                    continue;
                }
                /* The keys of one TYPE are checked against the block's TYPE
                 * when it ends (TYPE may follow them). */
                if (is_strategy_type_key(key))
                    type_keys.push_back({key, nline, line});
                /* MULTISTART knobs. */
                if (key=="N_STARTS") {
                    spec.multistart.n_starts = parse_uint(single_value(iss, key), key, 1);
                    continue;
                }
                if (key=="PARALLEL") {
                    spec.multistart.parallel = parse_uint(single_value(iss, key), key, 1);
                    continue;
                }
                if (key=="BUDGET_PER_START") {
                    /* Each start's MAXEVAL: the int budget MAXEVAL is
                     * (optimizer_block.cpp). */
                    spec.multistart.budget_per_start =
                        parse_uint(single_value(iss, key), key, 1,
                                   std::numeric_limits<int>::max());
                    continue;
                }
                /* PRESTAGE knob. */
                if (key=="TOPK") {
                    spec.stage_topk = parse_uint(single_value(iss, key), key, 1);
                    continue;
                }
                throw std::runtime_error("unknown key '" + key + "' inside STRATEGY block");
            }

            /* EXPLORE / REFINE block keys: only the optimiser attributes
             * dispatched above (LIB/ALGO/MAXEVAL/...) are valid there. */
            if (block == EXPLORE_BLK || block == REFINE_BLK) {
                throw std::runtime_error("unknown key '" + key + "' inside "
                    + std::string(block == EXPLORE_BLK ? "EXPLORE" : "REFINE") + " block");
            }

            /* INNER block -- only opt attributes, dispatched above. */
            if (block == INNER_BLK) {
                throw std::runtime_error("unknown key '" + key + "' inside INNER block "
                    "(only " + inner_attr_keys_joined("/") + " are valid)");
            }

            /* Anything still unhandled inside an OPTIMIZER-level block is
             * a user mistake (LIB/ALGO/MAXEVAL etc. now live inside
             * STRATEGY). LOSS / PC_LAMMPS / OPTIM / FAILSCORE / OUTPUT /
             * BDUMP and friends are ROOT-level and fall through. */
            if (block != ROOT) {
                throw std::runtime_error("unknown key '" + key + "' in this block");
            }

            /* ==================  global (ROOT) section  =========== */

            /* optimizer-independent settings ----------------------- *
             * Every value is parsed strictly (strict_parse.h): the whole
             * token, one per single-value key. */
            /* A positive finite number: the failure band [0.999 FS, FS)
             * lies below it, and a loss is usable only below the band. With
             * 0 or below there is no band: every finite loss, FAILSCORE
             * itself included, would count as usable, a run whose every
             * evaluation failed would end as a completed one with a failed
             * best_pot.tadah, and the ceiling would clip every positive loss
             * to FAILSCORE. Negative losses (negative-weight --varloss
             * terms) stay usable under any positive FAILSCORE. */
            if (key=="FAILSCORE") {
                FAIL_SCORE = parse_positive(single_value(iss, key), key,
                                            "finite number");
            }
            /* PC_SCRIPT_TIMEOUT: per-pc_script soft wall-clock limit (a LAMMPS
             * time spec: off, unlimited or a positive [[HH:]MM:]SS) injected as
             * LAMMPS `timer timeout` after each clear. MAXTIME is
             * the legacy alias, kept so existing .hpo configs keep parsing.
             * NOTE: the OPTIMIZER-block MAXTIME (whole-run stop) is a DIFFERENT
             * setting parsed in optimizer_block.cpp -- do not conflate. If both
             * spellings appear here, the later line wins (each overwrites
             * MAX_TIME); hpo.cpp warns when both were given. */
            else if (key=="PC_SCRIPT_TIMEOUT" || key=="MAXTIME") {
                /* A LAMMPS time spec that is off, unlimited or positive:
                 * it becomes `timer timeout <spec>`, and its seconds size
                 * the PROCESS worker deadline. */
                MAX_TIME = single_value(iss, key);
                MAX_TIME_seconds = parse_script_timeout(MAX_TIME, key);
                MAX_TIME_key = key;
                if (key=="MAXTIME") MAX_TIME_seen_legacy = true;
                else                MAX_TIME_seen_new    = true;
            }
            else if (key=="PC_ISOLATION") {
                PC_ISOLATION = single_value(iss, key);
                PC_ISOLATION_set = true;
                if (PC_ISOLATION != "THREAD" && PC_ISOLATION != "PROCESS")
                    throw std::runtime_error(
                        "PC_ISOLATION must be THREAD or PROCESS, got '"
                        + PC_ISOLATION + "'");
            }
            else if (key=="PC_WORKER_TIMEOUT") {
                PC_WORKER_TIMEOUT = parse_pc_worker_timeout(
                    single_value(iss, key), "PC_WORKER_TIMEOUT");
            }
            else if (key=="FAILSCORE_GRADING") {
                const std::string v = single_value(iss, key);
                if (v=="ON")       FAILSCORE_GRADING = true;
                else if (v=="OFF") FAILSCORE_GRADING = false;
                else throw std::runtime_error(
                        "FAILSCORE_GRADING must be ON or OFF, got '" + v + "'");
            }
            else if (key=="BUMPGATE") {
                const std::string v = single_value(iss, key);
                if (v=="STRICT")         BUMPGATE_STRICT = true;
                else if (v=="CANDIDATE") BUMPGATE_STRICT = false;
                else throw std::runtime_error(
                        "BUMPGATE must be CANDIDATE or STRICT, got '" + v + "'");
            }
            /* RHOGATE / RHOBARGATE: the DM_REAM density gates (strict
             * grammar above; semantic checks in validate_hpo_setup). */
            else if (key=="RHOGATE") {
                if (RHOGATE)
                    throw std::runtime_error("RHOGATE given more than once");
                RHOGATE = parse_rhogate(iss);
                gate_lines.push_back({"RHOGATE", nline, line});
            }
            else if (key=="RHOBARGATE") {
                if (RHOBARGATE)
                    throw std::runtime_error("RHOBARGATE given more than once");
                RHOBARGATE = parse_rhobargate(iss);
                gate_lines.push_back({"RHOBARGATE", nline, line});
            }

            /* PC_FAILPROBE: minimiser-free potential-sanity probe (graded-
             * FAILSCORE Tier 3). Strips the probe-only flags (--gate, --s0)
             * before handing the rest to the LammpsSim parser, then flags the
             * sim. Loss components on the probe are forbidden -- it must not
             * shape the feasible-region objective. */
            else if (key=="PC_FAILPROBE")
            {
                if (failprobe_idx >= 0)
                    throw std::runtime_error("only one PC_FAILPROBE line is allowed");
                std::string cmd; std::getline(iss,cmd);
                std::istringstream ts(cmd);
                std::vector<std::string> toks, keep;
                { std::string t; while (ts>>t) toks.push_back(t); }
                for (std::size_t i2 = 0; i2 < toks.size(); ++i2) {
                    if (toks[i2]=="--gate" || toks[i2]=="--s0") {
                        if (i2+1 >= toks.size())
                            throw std::runtime_error("PC_FAILPROBE " + toks[i2]
                                                     + " requires a value");
                        const std::string what = "PC_FAILPROBE " + toks[i2];
                        if (toks[i2]=="--gate")
                            failprobe_gate = parse_finite(toks[i2+1], what);
                        else
                            failprobe_s0 = parse_positive(toks[i2+1], what);
                        ++i2;
                    } else keep.push_back(toks[i2]);
                }
                std::string cmd2;
                for (const auto& t : keep) { cmd2 += ' '; cmd2 += t; }
                LS.emplace_back(cmd2, MAX_TIME, FAIL_SCORE, globalLoss,
                                /*failprobe=*/true);
                LammpsSim& probe = LS.back();
                if (!probe.n_varloss.empty() || !probe.i_varloss.empty())
                    throw std::runtime_error(
                        "PC_FAILPROBE must not declare --varloss (it grades "
                        "failures; it must not shape the feasible objective)");
                if (probe.n_outvar.empty())
                    throw std::runtime_error(
                        "PC_FAILPROBE requires --outvar <S> (the sanity scalar)");
                if (!probe.hotvars.empty() || !probe.failvars.empty())
                    throw std::runtime_error(
                        "PC_FAILPROBE must not declare --hotvar/--failvar");
                failprobe_idx  = static_cast<int>(LS.size()) - 1;
                failprobe_svar = probe.n_outvar.front();
            }
            else if (key=="QFLUSH") {
                QFLUSH_THRESHOLD = parse_uint(single_value(iss, key), key);
            }
            /* D14: these are moduli (`iter % every`), so 0 is an integer
             * division by zero -- SIGFPE on the first evaluation, with no
             * message. "0" is also the spelling a user naturally reaches for
             * to mean "do not write these files", which makes it a trap
             * rather than a typo. Reject it at the parse site, where the
             * line number is still known, in the same shape as DIGITS
             * below. */
            else if (key=="OUTPUT") {
                outputEvery = parse_uint(single_value(iss, key), key, 1,
                    std::numeric_limits<std::uint64_t>::max(),
                    "it is a cadence: write every Nth evaluation; to write as "
                    "little as possible, use a large value such as OUTPUT "
                    "1000000");
            }
            else if (key=="BOUTPUT") {
                bestOutputEvery = parse_uint(single_value(iss, key), key, 1,
                    std::numeric_limits<std::uint64_t>::max(),
                    "it is a cadence: write every Nth new best");
            }
            else if (key=="DIGITS") {
                outputDigits = static_cast<int>(
                    parse_uint(single_value(iss, key), key, 1, 15));
            }
            else if (key=="DUMP" || key=="BDUMP") {
                /* <cadence> <directory>: the directory is created only once
                 * the whole line is valid. */
                std::string every, dir;
                if (!(iss >> every >> dir))
                    throw std::runtime_error(
                        key + " requires a cadence and a directory");
                expect_end(iss, key + " takes a cadence and a directory");
                const std::size_t n = parse_uint(every, key + " cadence", 1);
                if (key == "DUMP") { dumpPotEvery = n; dumpPotDir = dir; }
                else { dumpBestPotEvery = n; dumpBestPotDir = dir; }
                handle_directory_creation(dir);
            }

            /* loss -------------------------------------------------- */
            else if (key=="LOSS")
            {
                /* LOSS <type> [<param>]. The optional parameter (HUBER
                 * delta, TUKEY c) is validated and defaulted by
                 * LossFunction::resolve_params -- the same path the
                 * PC_LAMMPS --varloss override takes. */
                std::string type;
                if (!(iss>>type))
                    throw std::runtime_error("LOSS requires a loss type");
                std::vector<double> params;
                std::string tok;
                while (iss>>tok) {
                    double p = 0.0;
                    if (!parse_number(tok, p))
                        throw std::runtime_error(
                            "LOSS " + type + ": parameter '" + tok +
                            "' is not a number" + number_underflow_note(tok));
                    params.push_back(p);
                }
                auto id = tadah::LossFunction::from_string(type);
                globalLoss = tadah::LossFunction(id,params);
            }

            /* optimisation variables --------------------------------*/
            else if (key=="OPTIM")   { process_optim_config(iss, nline); }
            else if (key=="OPTIM_SCALE") {
                std::string v = single_value(iss, key);
                for (auto& ch : v) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
                if (scale_policy_set_)
                    throw std::runtime_error("OPTIM_SCALE given more than once");
                if      (v == "AUTO") scale_policy_.normalise = true;
                else if (v == "NONE") scale_policy_.normalise = false;
                else throw std::runtime_error(
                    "OPTIM_SCALE must be AUTO (every OPTIM dimension is mapped to the unit "
                    "box [0,1]) or NONE (optimisers work in raw OPTIM units; ln for LOG "
                    "dims) -- got '" + v + "'");
                scale_policy_set_ = true;
            }

            /* LAMMPS sims ------------------------------------------ */
            else if (key=="PC_LAMMPS")
            {
                std::string cmd; std::getline(iss,cmd);
                LS.emplace_back(cmd, MAX_TIME, FAIL_SCORE, globalLoss);
            }

            /* Typed observables ------------------------------------
             *   PC_OBSERVABLE <kind> --flag value ...
             *
             * MVP kinds: cohesive_energy. The parser dispatches on
             * <kind> and constructs the appropriate Observable*
             * subclass. Future observables plug into the same dispatch
             * without touching this file. */
            else if (key=="PC_OBSERVABLE")
            {
                std::string cmd; std::getline(iss, cmd);
                observables.push_back(parse_pc_observable_line(cmd));
            }

            /* per-metric weight lines ------------------------------
             *   PC_<METRIC>  <target>  <weight>  [--skip-above N] [--skip-rel R]
             *
             * --skip-above and --skip-rel attach a "bad-potential"
             * pre-LAMMPS gate to the metric: if the eval's <metric>
             * exceeds the cap, the LAMMPS phase is skipped and the
             * iter is failscored. Replaces the legacy env vars
             * TADAH_HPO_BAD_POT_ERMSE_MEV / _REL (which only ever
             * gated on ERMSE; this is general). */
            else if (key=="PC_EMAE"
                  || key=="PC_ERMSE"
                  || key=="PC_ErRMSE"
                  || key=="PC_ERSQ"
                  || key=="PC_SRMSE"
                  || key=="PC_FRMSE")
            {
                const bool dataset_metric = key == "PC_ERMSE" || key == "PC_FRMSE";
                std::vector<std::size_t> indices;
                iss >> std::ws;
                if (iss.peek() == '(') {
                    if (!dataset_metric)
                        throw std::runtime_error(key + ": indexed targets are supported only for PC_ERMSE and PC_FRMSE");
                    indices = core::parse_indices_strict(read_index_selector(iss));
                }
                std::string target_token, weight_token;
                if (!(iss >> target_token >> weight_token))
                    throw std::runtime_error(key + " requires <target> <weight>");
                const double tgt = parse_metric_number(target_token, key + " <target>");
                const double wgt = parse_metric_number(weight_token, key + " <weight>");
                if (wgt < 0)
                    throw std::runtime_error(key + " <weight> must be a number >= 0, got '" +
                                             weight_token + "'");
                HPOTargets::SkipCaps caps;
                std::string flag;
                while (iss >> flag) {
                    if (flag != "--skip-above" && flag != "--skip-rel")
                        throw std::runtime_error(key + ": unknown flag '" + flag + "'");
                    std::string token;
                    if (!(iss >> token)) throw std::runtime_error(key + " " + flag + " requires a positive number");
                    const double value = parse_positive(token, key + " " + flag);
                    auto& cap = flag == "--skip-above" ? caps.abs_above : caps.rel_above;
                    if (cap) throw std::runtime_error(key + ": repeated " + flag);
                    cap = value;
                }
                if (dataset_metric) {
                    auto& targets = key == "PC_ERMSE" ? ermse_targets : frmse_targets;
                    targets.add(indices, {tgt, wgt, caps, nline});
                    if (key == "PC_ERMSE") use_ermse = true;
                    else use_frmse = true;
                }
                else if (key=="PC_EMAE")   { emae=tgt;   w_emae=wgt;   use_emae=true;   skip_emae=caps; }
                else if (key=="PC_ErRMSE") { errmse=tgt; w_errmse=wgt; use_errmse=true; skip_errmse=caps; }
                else if (key=="PC_ERSQ")   { ersq=tgt;   w_ersq=wgt;   use_ersq=true;   skip_ersq=caps; }
                else if (key=="PC_SRMSE")  { srmse=tgt;  w_srmse=wgt;  use_srmse=true;  skip_srmse=caps; }
            }
            else
                throw std::runtime_error("unknown key \""+key+"\"");
        }
        catch (const EarlierLineError& e)
        {
            throw std::runtime_error(at_line(e.line, e.what(), e.text));
        }
        catch(const std::exception& e)
        {
            throw std::runtime_error(at_line(nline, e.what(), line));
        }
    }

    /* A block the file never closes, or no OPTIMIZER block at all: the end
     * of the file is where these are known. */
    try {
        check_type_keys();
    } catch (const EarlierLineError& e) {
        throw std::runtime_error(at_line(e.line, e.what(), e.text));
    }
    /* A density gate that no OPTIM line arms cannot act: no dimension of
     * the search changes rho(r) or its window, so every candidate has the
     * -c density and its one verdict (OPTIM WATOM only scales the density,
     * and the gate verdicts are scale-invariant). Refused at the gate's
     * line, the first in the file (owner decision 2026-09-29). */
    {
        bool armed = false;
        for (const auto& k : density_gate_keys()) {
            const auto it = optimap.find(k);
            if (it == optimap.end()) continue;
            for (bool b : it->second.opt) armed = armed || b;
        }
        if (!armed && !gate_lines.empty()) {
            const GateLine& g = gate_lines.front();
            std::string keys;
            const auto& dk = density_gate_keys();
            for (std::size_t i = 0; i < dk.size(); ++i)
                keys += (i == 0 ? "" : i + 1 == dk.size() ? " or " : ", ") + dk[i];
            throw std::runtime_error(at_line(g.line,
                std::string(g.key) + " cannot act: no OPTIM line searches a key "
                "that changes the DM_REAM density or its window (" + keys +
                "), so every candidate has the -c density and its one verdict "
                "(OPTIM WATOM only scales the density; the verdict does not "
                "change). Add an OPTIM line on " + keys + ", or remove the " +
                g.key + " line.", g.text));
        }
    }

    if (block == INNER_BLK)
        require_inner_driver(*active_opt, inner_enclosing());
    require_driver(spec, DriverPhase::Run);
    if (spec.prestage) require_driver(*spec.prestage, DriverPhase::Explore);

    /* Post-parse validation: PC_OBSERVABLE and PC_LAMMPS are mutually
     * exclusive paths in the experimental MVP. The typed-observable
     * path consumes analytical gradients exclusively; PC_LAMMPS now
     * only supports gradient-free optimisers. Mixing them in one
     * --hpotarget produces a config the orchestrator cannot serve
     * consistently, so reject it early with a clear error rather than
     * letting the user discover the problem at eval-time. */
    if (!observables.empty() && !LS.empty()) {
        throw std::runtime_error(
            "--hpotarget defines both PC_OBSERVABLE (" +
            core::to_string(observables.size()) +
            " entries) and PC_LAMMPS (" +
            core::to_string(LS.size()) +
            " entries); these paths are mutually exclusive in the "
            "current MVP. Remove one set of lines and rerun.");
    }

    /* Reject the same --hotvar NAME declared by two DISTINCT PC_LAMMPS
     * scripts. run_lammps' phase A writes hot_values_for_iter[name] (keyed by
     * name only) and warmstart::set(script,name); if two scripts share a hot
     * name, "last writer wins" makes the injected value order-dependent -- a
     * latent non-determinism that longest-first fan-out reordering could
     * expose. Hot names are meant to be unique per script (lammps_sim.h), so
     * enforce it at parse time with a clear message. */
    {
        std::unordered_map<std::string, std::size_t> hot_owner;
        for (std::size_t s = 0; s < LS.size(); ++s) {
            for (const auto& hv : LS[s].hotvars) {
                auto it = hot_owner.find(hv.name);
                if (it != hot_owner.end() && it->second != s) {
                    throw std::runtime_error(
                        "PC_LAMMPS: --hotvar name '" + hv.name +
                        "' is declared by two different scripts ('" +
                        LS[it->second].script + "' and '" + LS[s].script +
                        "'). Hot-var names must be unique across scripts: the "
                        "per-iter hot value is keyed by name only, so sharing a "
                        "name makes the injected value order-dependent. Rename "
                        "one of them.");
                }
                hot_owner.emplace(hv.name, s);
            }
        }
    }

    /* PC_OBSERVABLE is built for the analytical gradient path: its only
     * observable (cohesive_energy) provides closed-form dO/dW and the
     * descriptors of its structures, and the fit-side walker differentiates
     * the refit. NUMERICAL with PC_OBSERVABLE would run finite differences
     * over the whole HPO objective -- correct but slow, and contrary to the
     * redesign's purpose. Reject it at parse time with a clear message.
     * ANALYTICAL and HYBRID are both accepted; HYBRID differs only where
     * the analytical gradient is unavailable (a parameter without a
     * registered derivative, a model setting the analytical gradient does
     * not cover), which it fills with finite differences. */
    auto reject_numerical_with_observable =
        [&](const OptimizerSpec& sp, const char* tag) {
            if (sp.grad_strategy
                && *sp.grad_strategy == GradStrategy::Numerical) {
                throw std::runtime_error(
                    std::string("--hpotarget defines PC_OBSERVABLE but ")
                    + tag + " GRAD_STRATEGY=NUMERICAL is set. PC_OBSERVABLE "
                    "is built for the analytical gradient: use GRAD_STRATEGY "
                    "ANALYTICAL (an unset GRAD_STRATEGY resolves to it in a "
                    "block that obtains gradients; it needs the settings of "
                    "the analytical gradient, such as OALGO 4 with a fixed "
                    "LAMBDA >= 0, and startup names a missing one) or "
                    "GRAD_STRATEGY HYBRID, which uses finite differences "
                    "where the analytical gradient is unavailable.");
            }
        };
    /* Every block that configures an optimiser: the STRATEGY-level one
     * (under PRESTAGE the refinement) and its INNER block, and the PRESTAGE
     * EXPLORE block and its INNER block. The rules below hold for each:
     * the exploration evaluates the same loss. An INNER block never holds
     * a GRAD_STRATEGY (the parser rejects one where it is written), so of
     * the GRAD_STRATEGY rules only the PC_LAMMPS optimiser rule has
     * something to check there. */
    struct OptimiserBlock { OptimizerSpec* sp; const char* tag; bool inner; };
    std::vector<OptimiserBlock> optimiser_blocks = {{&spec, "OPTIMIZER's", false}};
    if (spec.inner)
        optimiser_blocks.push_back({spec.inner.get(), "INNER's", true});
    if (spec.prestage) {
        optimiser_blocks.push_back({spec.prestage.get(), "EXPLORE's", false});
        if (spec.prestage->inner)
            optimiser_blocks.push_back(
                {spec.prestage->inner.get(), "EXPLORE INNER's", true});
    }
    if (!observables.empty())
        for (const auto& b : optimiser_blocks)
            reject_numerical_with_observable(*b.sp, b.tag);

    /* ...and an unset GRAD_STRATEGY, which means NUMERICAL, resolves to
     * ANALYTICAL in each block that runs an optimiser and obtains gradients
     * (obtains_gradients: a gradient-based backend, or an MLSL/AUGLAG whose
     * local search is one) -- the STRATEGY-level block and EXPLORE.
     * grad_strategy_resolved puts it in the journal. An INNER block runs
     * no optimiser of its own: MLSL and AUGLAG hand their local search the
     * gradient of their own objective, which the enclosing block's
     * GRAD_STRATEGY obtains (nlopt_adapter.h), so an INNER block takes no
     * GRAD_STRATEGY and is not resolved. The derivative-free blocks are
     * left alone too: they never ask for a gradient, and ANNEAL and ASA
     * reject any GRAD_STRATEGY. */
    if (!observables.empty()) {
        for (const auto& b : optimiser_blocks) {
            if (b.inner || b.sp->grad_strategy || !obtains_gradients(*b.sp))
                continue;
            b.sp->grad_strategy = GradStrategy::Analytical;
            b.sp->grad_strategy_resolved = true;
        }
    }

    /* PC_LAMMPS: gradient-support dropped 2026-05-13.
     *
     * The HPO-side analytical-gradient wiring for PC_LAMMPS (Phase 3
     * IFT-adjoint chain) is no longer engaged. PC_LAMMPS configs must
     * use gradient-free optimisers -- NLOPT GN_* / LN_*, TADAH built-ins
     * (RANDOM/GRID/ANNEAL). The single accepted gradient-based exception
     * is DLIB:BFGS, which performs its own per-theta-per-iter finite-
     * difference dtheta inside dlib::find_min and never asks the host
     * for a full gradient vector.
     *
     * Also reject any explicit GRAD_STRATEGY={NUMERICAL,HYBRID,
     * ANALYTICAL} when PC_LAMMPS is present -- these tags only make
     * sense in the PC_OBSERVABLE world post-2026-05-13. */
    auto reject_pc_lammps_grad =
        [&](const OptimizerSpec& sp, const char* tag) {
            if (sp.grad_strategy) {
                throw std::runtime_error(
                    std::string("--hpotarget defines PC_LAMMPS but ") + tag
                    + " GRAD_STRATEGY is set. PC_LAMMPS no longer "
                    "supports analytical/numerical/hybrid gradients "
                    "(2026-05-13). Drop the GRAD_STRATEGY line.");
            }
            /* read() has required LIB and ALGO of every block by now, the
             * EXPLORE and INNER blocks included (require_driver,
             * require_inner_driver). */
            const bool dlib_bfgs_exception =
                (sp.lib == "DLIB" && sp.algo == "BFGS");
            if (dlib_bfgs_exception) return;
            const auto& caps = optimizer_caps(sp);
            if (caps.is_gradient_based) {
                throw std::runtime_error(
                    std::string("--hpotarget defines PC_LAMMPS but ") + tag
                    + " optimiser " + sp.lib + ":" + sp.algo +
                    " is gradient-based. PC_LAMMPS requires a "
                    "gradient-free optimiser (NLOPT GN_*/LN_*, "
                    "TADAH RANDOM/GRID/ANNEAL). The only gradient-based "
                    "exception is DLIB:BFGS (uses per-theta-per-iter "
                    "dtheta, not a full gradient).");
            }
        };
    if (!LS.empty())
        for (const auto& b : optimiser_blocks)
            reject_pc_lammps_grad(*b.sp, b.tag);
}

double HPOTargets::parse_pc_worker_timeout(const std::string& token,
                                           const std::string& origin) {
  double v = 0.0;
  if (!parse_number(token, v) || v < 0.0)
    throw std::runtime_error(
        origin + " must be a finite number of seconds >= 0 (0 derives the "
        "deadline from each script's timeout), got '" + token + "'" +
        number_underflow_note(token));
  if (!pc_worker_deadline_fits(v, std::nullopt))
    throw std::runtime_error(
        origin + " is too large for the worker deadline, got '" + token +
        "' (it is kept in 64-bit milliseconds: at most about 9.2e15 s)");
  return v;
}

void HPOTargets::handle_directory_creation(const std::string& directory) {
  try {
    if (!fs::exists(directory)) {
      if (fs::create_directory(directory)) {
        std::cout << "Directory created: " << directory << std::endl;
      } else {
        std::cerr << "Failed to create directory: " << directory << std::endl;
      }
    } else {
      std::cout << "Directory already exists: " << directory << std::endl;
    }
  } catch (const std::exception& e) {
    throw std::runtime_error("Error in handle_directory_creation: "
                             + std::string(e.what()));
  }
}

/* Positive-only OPTIM keys, from the same table the refit hpotarget writer
 * uses (MLIP hpotarget_writer.h) so HPO and the writer cannot disagree;
 * keys unknown to that table are sign-free. */
bool HPOTargets::optim_key_positive_only(const std::string& key) {
  for (const auto& g : mlip::engines::hpotarget_group_specs())
    if (key == g.key) return g.positive_only;
  return false;
}

std::string HPOTargets::optim_structure_key_error(const std::string& key) {
  static const struct { const char* key; const char* what; } structure[] = {
    {"RHOFORM",  "RHOFORM (SIGNED|NONNEG|MONOTONE) selects the DM_REAM density "
                 "form, a discrete property of the potential (its amplitudes "
                 "were fitted in that form)"},
    {"TYPE2B",   "TYPE2B names the two-body descriptor and its structure "
                 "tokens (basis code, sizes, species)"},
    {"TYPEMB",   "TYPEMB names the many-body descriptor and its structure "
                 "tokens (basis code, sizes, species)"},
    {"RCTYPE2B", "RCTYPE2B names the two-body cutoff function"},
    {"RCTYPEMB", "RCTYPEMB names the many-body cutoff function"},
    {"KDEG2B",   "KDEG2B lists the two-body knot degrees (3|5)"},
    {"KDEGMB",   "KDEGMB lists the density knot degrees (3|5)"},
    {"KDEGEMB",  "KDEGEMB lists the embedding knot degrees (3|5)"},
    {"DIMER",    "DIMER (on/off, the bond length, whether the intramolecular "
                 "distance enters the descriptors) declares the rigid-dimer "
                 "layout of the training data (its bond length is that of "
                 "the data's molecules)"},
    {"FIXINDEX", "FIXINDEX names which model weights FIXWEIGHT pins (search "
                 "their values with OPTIM FIXWEIGHT)"},
    {"MODEL",    "MODEL names the model and its basis function or kernel "
                 "(and an integer option)"},
  };
  const std::string k = core::to_upper(key);
  for (const auto& e : structure)
    if (k == e.key)
      return "OPTIM " + k + " is not a valid search target: " + e.what +
             ", structure data rather than a continuous parameter; set it "
             "in the -c potential/configuration.";
  return "";
}

std::string HPOTargets::optim_discrete_index_error(const std::string& key,
                                                   std::size_t index) {
  if (core::to_upper(key) == "OALGO" && index == 1)
    return "OPTIM OALGO (1) is not a valid search target: the first OALGO "
           "value selects the training algorithm (1 GELSD, 2 GELS, 3 SVD, "
           "4 Cholesky), a discrete choice rather than a continuous "
           "parameter; set it in the -c configuration. The second value, "
           "the rcond M of options 1 and 3, is continuous: OPTIM OALGO (2) "
           "<low> <high>.";
  return "";
}

const std::vector<std::string>& HPOTargets::density_gate_keys() {
  static const std::vector<std::string> keys = {"AMPGRIDMB", "CGRIDMB",
                                                "SGRIDMB", "RCUTMB"};
  return keys;
}

SearchSpace HPOTargets::build_search_space() const {
  std::vector<DimSpec> dims;
  for (const auto& kv : optimap) {
    const Opti& o = kv.second;
    /* user_index is the index the OPTIM line uses -- and the only one the
     * user ever sees (labels, journal, sidecars): the 1-based position among
     * the key's numeric values, or for FIXWEIGHT the MODEL WEIGHT the slot
     * pins (FIXINDEX; process_optim_config maps it to the slot). Never the
     * 0-based token position, which is an implementation detail. */
    std::vector<std::size_t> fixidx;
    if (kv.first == "FIXWEIGHT" && context.contains("FIXINDEX")) {
      std::vector<std::string> fi(context.size("FIXINDEX"));
      context.get("FIXINDEX", fi);
      fixidx = core::parse_indices(fi);
    }
    std::size_t numeric_ordinal = 0;
    for (std::size_t i = 0; i < o.opt.size(); ++i) {
      const bool numeric = i < o.config.size() && core::is_number(o.config[i]);
      if (numeric) ++numeric_ordinal;
      if (!o.opt[i]) continue;
      DimSpec d;
      d.key        = kv.first;
      d.user_index = numeric ? numeric_ordinal : 0;
      if (!fixidx.empty() && d.user_index >= 1 && d.user_index <= fixidx.size())
        d.user_index = fixidx[d.user_index - 1];
      d.label      = kv.first;
      if (o.opt.size() > 1) d.label += "[" + std::to_string(d.user_index) + "]";
      d.lo = o.lowbound[i];
      d.hi = o.highbound[i];
      d.tf = i < o.scale.size() ? o.scale[i] : DimTransform{};
      d.origin = i < o.origin.size() ? o.origin[i] : BoxOrigin::Explicit;
      if (numeric) {
        try { d.x0 = core::to_double(o.config[i]); } catch (const std::exception&) {}
      }
      dims.push_back(std::move(d));
    }
  }
  ScalePolicy pol = scale_policy_;
  pol.log_ratio = spec.log_threshold;
  return SearchSpace::build(std::move(dims), pol);
}

void HPOTargets::write_search_space_files(const std::string& dir) const {
  if (!space_) return;
  const SearchSpace& ss = *space_;
  const std::vector<double>* x0 = spec.optim_init.empty() ? nullptr : &spec.optim_init.front();
  {
    /* Human-first layout: every column space-padded to one width and the
     * columns tab-separated, so the file lines up in any editor AND stays a
     * TSV (strip whitespace when parsing). 10 significant digits -- the
     * exact box is what the run holds; hpo_optim_resolved.hpo is the
     * paste-ready record. */
    auto num = [](double v) {
      std::ostringstream o; o << std::setprecision(10) << v; return o.str();
    };
    const std::vector<std::string> header = {
      "dim", "label", "lo", "hi", "origin", "x0", "scale", "c", "u0",
      "internal_lo", "internal_hi", "version"};
    const std::vector<bool> right_align = {
      true, false, true, true, false, true, false, true, true, true, true, true};
    std::vector<std::vector<std::string>> rows;
    rows.reserve(ss.dim());
    for (std::size_t i = 0; i < ss.dim(); ++i) {
      const DimSpec& d = ss.at(i);
      const double seed = (x0 && i < x0->size()) ? (*x0)[i] : d.x0;
      const bool ok = std::isfinite(seed);
      rows.push_back({std::to_string(i + 1), d.label, num(d.lo), num(d.hi),
                      box_origin_name(d.origin), ok ? num(seed) : "nan",
                      scale_kind_name(d.tf.kind), num(d.tf.c),
                      ok ? num(ss.to_internal(i, seed)) : "nan",
                      num(ss.internal_lo(i)), num(ss.internal_hi(i)),
                      std::to_string(ss.version())});
    }
    std::vector<std::size_t> width(header.size());
    for (std::size_t c = 0; c < header.size(); ++c) {
      width[c] = header[c].size();
      for (const auto& r : rows) width[c] = std::max(width[c], r[c].size());
    }
    std::ofstream tsv(dir + "/hpo_search_space.tsv");
    tsv <<
      "# tadah hpo search space -- one row per optimised dimension, in optimiser order\n"
      "# (the same order as the params.tadah columns and the journal tables).\n"
      "#   dim          1-based position in the parameter vector\n"
      "#   label        KEY[n]: n is the OPTIM index (1-based; for FIXWEIGHT the model weight\n"
      "#                named in FIXINDEX); single-value keys carry no index\n"
      "#   lo, hi       real bounds the optimiser may not leave\n"
      "#   origin       OPTIM = explicit bounds; AUTO FRAC / AUTO RATIO = derived from the seed;\n"
      "#                AUTO FRAC>=0 = AUTO FRAC with lo = max(0, x0 - d) (AMPGRIDMB under\n"
      "#                RHOFORM NONNEG / MONOTONE); lo is 0 only where x0 - d < 0\n"
      "#   x0           starting point (real units)\n"
      "#   scale, c     shape of the internal coordinate: LIN, LOG (ln x) or ASINH (asinh(x/c));\n"
      "#                c is the ASINH linear scale, 1 for the other shapes\n"
      "#   u0           x0 in optimiser units (0..1 under OPTIM_SCALE AUTO, raw or ln under NONE)\n"
      "#   internal_lo, internal_hi   optimiser-side bounds (0 and 1 under AUTO)\n"
      "#   version      revision of the search space. 1 = the boxes parsed from the config; it\n"
      "#                would only increase if a bound-adaptation step rewrote the boxes during\n"
      "#                the run (no such step exists yet), so a journal reference\n"
      "#                to a version can always be matched to one box set\n"
      "# Columns are tab-separated and space-padded to equal width -- strip whitespace when parsing\n"
      "# (pandas: read_csv(path, sep='\\t', comment='#') then .str.strip() on text columns).\n";
    auto emit = [&](const std::vector<std::string>& r) {
      for (std::size_t c = 0; c < r.size(); ++c) {
        tsv << (right_align[c] ? std::right : std::left) << std::setw(static_cast<int>(width[c])) << r[c];
        tsv << (c + 1 < r.size() ? '\t' : '\n');
      }
    };
    emit(header);
    for (const auto& r : rows) emit(r);
  }
  bool any_auto = false;
  for (std::size_t i = 0; i < ss.dim(); ++i)
    if (ss.at(i).origin != BoxOrigin::Explicit) any_auto = true;
  if (!any_auto) return;
  std::ofstream hpo(dir + "/hpo_optim_resolved.hpo");
  hpo << "# Resolved OPTIM block (explicit bounds for every searched dimension).\n"
         "# Written by tadah hpo because at least one box was derived by an AUTO rule;\n"
         "# paste these lines in place of the AUTO lines to pin exactly this search space.\n";
  /* Lossless numbers (shortest round-trip text): pasted back, every bound
   * must parse to the very double this run searched -- 10 digits moved AUTO
   * bounds by up to 5e-11 relative. */
  for (std::size_t i = 0; i < ss.dim(); ++i) {
    const DimSpec& d = ss.at(i);
    hpo << "OPTIM " << d.key << " (" << (d.user_index ? d.user_index : 1) << ") "
        << core::format_roundtrip(d.lo) << " " << core::format_roundtrip(d.hi);
    if (d.tf.kind == ScaleKind::Log)   hpo << " LOG";
    if (d.tf.kind == ScaleKind::Asinh) hpo << " ASINH " << core::format_roundtrip(d.tf.c);
    hpo << "   # " << box_origin_name(d.origin) << '\n';
  }
}

void HPOTargets::bind_space_recursive() {
  auto bind = [&](OptimizerSpec* sub) {
    if (!sub) return;
    sub->bind_space(space_);
    sub->scale     = spec.scale;
    sub->failscore = spec.failscore;
    if (sub->inner) {
      sub->inner->bind_space(space_);
      sub->inner->scale     = spec.scale;
      sub->inner->failscore = spec.failscore;
    }
  };
  bind(spec.inner.get());
  bind(spec.prestage.get());
}

void HPOTargets::process_optim_config(std::istringstream& iss,
                                      std::size_t nline) {
  std::string k;
  if (!(iss >> k)) {
    throw std::runtime_error("Failed to read key from input.");
  }
  /* A configuration key, and configuration keys are case-insensitive
   * (core::Context stores them upper-case): the OPTIM key is upper-cased
   * here, so optimap, the labels and every rule and lookup keyed on it
   * (below, startup_validation, the gates, the neighbour-list cutoff) see
   * one spelling whatever the case the line used. */
  k = core::to_upper(k);

  if (k == "WEIGHTS") {
    throw std::runtime_error(
      "OPTIM WEIGHTS is not a valid search target: the linear WEIGHTS are "
      "re-fitted by least squares on every evaluation, so optimising them "
      "is a silent no-op. To search linear coefficients, pin them with "
      "FIXINDEX + FIXWEIGHT in the training configuration and use "
      "`OPTIM FIXWEIGHT (<weight-index>) <low> <high>`.");
  }
  /* Before context.size(k) below, which would materialise the blueprint
   * default EMBF0 ZERO into the run context. */
  if (k == "EMBF0") {
    throw std::runtime_error(
      "OPTIM EMBF0 is not a valid search target: EMBF0 (ZERO|FREE) selects "
      "the DM_REAM embedding reference, a discrete property of the "
      "potential; set it in the -c potential/configuration.");
  }
  /* Structure keys, also before context.size(k): RHOFORM has no blueprint
   * default, so size() would log an ERROR and throw a generic message. */
  if (const std::string e = optim_structure_key_error(k); !e.empty())
    throw std::runtime_error(e);

  Opti& omap = optimap[k];
  size_t s = context.size(k);
  omap.resize(s);

  std::vector<std::string> tokens = parse_optim_line(iss);

  /* At least the (indices) group plus one token (AUTO, or <low> <high>);
   * the exact grammar is checked below. */
  /* read() names the line of every error ("Error on line N: "). */
  if (tokens.size() < 2) {
    throw std::runtime_error(
      "OPTIM " + k + ": expected `(indices) <low> <high>` or `(indices) AUTO "
      "...` (got " + core::to_string(tokens.size() - (tokens.empty() ? 0 : 1)) +
      " token(s))");
  }

  std::vector<size_t> user_indices;   // FIXWEIGHT: mapped to slots below
  try {
    user_indices = core::parse_indices_strict(tokens.back());
    if (std::find(user_indices.begin(), user_indices.end(), 0) != user_indices.end())
      throw std::runtime_error("OPTIM indices start at 1");
  } catch (const std::exception& e) {
    throw std::runtime_error("OPTIM " + k + ": bad index list (" +
                             tokens.back() + "): " + e.what());
  }
  tokens.pop_back();
  for (const size_t ui : user_indices)
    if (const std::string e = optim_discrete_index_error(k, ui); !e.empty())
      throw std::runtime_error(e);

  /* Strict grammar: OPTIM <KEY> (indices) <low> <high>. Anything else
   * used to be silently ignored (the bounds were popped from the END of
   * the token list), which swallowed typos and stray tokens. */
  /* Two forms after the (indices) group has been extracted:
   *   <low> <high> [LIN | LOG | ASINH <c>]                  explicit box
   *   AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>] [LIN | LOG | ASINH <c>]
   *                                                        seed-relative box
   * AUTO derives the box from the CONFIG seed of each selected index:
   *   geometric  [x0/f, x0*f] (mirrored for x0 < 0)   default for positive-only keys (RATIO 2)
   *   linear     x0 +/- max(p*|x0|, a, r*max|x0| over this OPTIM line)   otherwise (FRAC 0.5)
   *   FRAC 0 + FLOOR a  -> every index gets the same absolute half-width a (grid centres)
   * Cutoff keys (RCUT*) are excluded: the neighbour list is sized from
   * explicit high bounds. */
  DimTransform tf;
  auto upper = [](std::string t) {
    for (auto& ch : t) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    return t;
  };
  /* Every number on an OPTIM line: a finite number of the number grammar
   * (strict_parse.h). */
  auto parse_num = [&](const std::string& tok, const char* what) {
    double v = 0.0;
    if (!parse_number(tok, v))
      throw std::runtime_error("OPTIM " + k + ": " + what +
                               " must be a finite number (got '" + tok + "')" +
                               number_underflow_note(tok));
    return v;
  };
  /* trailing shape tag (both forms) */
  auto take_tag = [&](std::vector<std::string>& toks) {
    if (toks.size() >= 2 && upper(toks[toks.size() - 2]) == "ASINH") {
      tf.kind = ScaleKind::Asinh;
      tf.c = parse_num(toks.back(), "ASINH <c>");
      if (!(tf.c > 0.0))
        throw std::runtime_error("OPTIM " + k + ": ASINH requires a positive "
                                 "linear scale c (got " + toks.back() + ")");
      toks.pop_back(); toks.pop_back();
      return;
    }
    if (!toks.empty()) {
      const std::string tag = upper(toks.back());
      if (tag == "LIN")      { tf.kind = ScaleKind::Linear; toks.pop_back(); }
      else if (tag == "LOG") { tf.kind = ScaleKind::Log;    toks.pop_back(); }
      else if (tag == "ASINH")
        throw std::runtime_error("OPTIM " + k + ": ASINH needs one argument <c> "
                                 "(the linear scale)");
    }
  };
  const std::string grammar =
    " -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>]"
    " or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>]"
    " [LIN | LOG | ASINH <c>]";

  /* The low-bound rule of the box (models::optim_low_bound_rule(k,
   * context)). RHOFORM NONNEG / MONOTONE (rho >= 0 by construction): the
   * AMPGRIDMB boxes are non-negative (models::optim_key_lower_bound: AUTO
   * lo = max(0, x0 - d), an explicit lo < 0 is refused), the WATOM boxes
   * positive and the SGRIDMB boxes positive under MONOTONE and, for the
   * Gaussian basis, under NONNEG. Every form: a radial bump width box
   * (SGRID2B of D2_Blip/D2_Blip5, SGRIDMB of a Blip/Blip5 density) is
   * positive. */
  const bool nonneg_amp = models::optim_key_lower_bound(k, rho_form_) ==
                          models::OptimLowerBound::NonNegative;
  const auto low_rule = models::optim_low_bound_rule(k, context);
  const std::string form_name =
      rho_form_ ? std::string("RHOFORM ") + models::rhoform_name(*rho_form_) : "";

  const bool auto_box = !tokens.empty() && upper(tokens.front()) == "AUTO";
  double low = 0.0, high = 0.0;
  struct AutoRule { bool geometric = false; double factor = 0.0; double floor = 0.0; double group_floor = 0.0; } rule;
  bool floor_set = false;   // FLOOR or GROUP_FLOOR given
  if (auto_box) {
    tokens.erase(tokens.begin());
    take_tag(tokens);
    bool mode_set = false;
    for (std::size_t t = 0; t < tokens.size(); t += 2) {
      const std::string opt = upper(tokens[t]);
      if (t + 1 >= tokens.size())
        throw std::runtime_error("OPTIM " + k + ": AUTO option '" + tokens[t] +
                                 "' needs a value" + grammar);
      const double v = parse_num(tokens[t + 1], ("AUTO " + opt).c_str());
      if (opt == "RATIO") {
        if (mode_set) throw std::runtime_error("OPTIM " + k + ": RATIO and FRAC are mutually exclusive");
        if (!(v > 1.0)) throw std::runtime_error("OPTIM " + k + ": AUTO RATIO must be > 1 (got " + tokens[t + 1] + ")");
        rule.geometric = true; rule.factor = v; mode_set = true;
      } else if (opt == "FRAC") {
        if (mode_set) throw std::runtime_error("OPTIM " + k + ": RATIO and FRAC are mutually exclusive");
        if (!(v >= 0.0)) throw std::runtime_error("OPTIM " + k + ": AUTO FRAC must be >= 0 (got " + tokens[t + 1] + "); FRAC 0 with FLOOR/GROUP_FLOOR gives a constant absolute half-width");
        rule.geometric = false; rule.factor = v; mode_set = true;
      } else if (opt == "FLOOR") {
        if (v < 0.0) throw std::runtime_error("OPTIM " + k + ": AUTO FLOOR must be >= 0");
        rule.floor = v;
        floor_set = true;
      } else if (opt == "GROUP_FLOOR") {
        if (v < 0.0) throw std::runtime_error("OPTIM " + k + ": AUTO GROUP_FLOOR must be >= 0");
        rule.group_floor = v;
        floor_set = true;
      } else {
        throw std::runtime_error("OPTIM " + k + ": unexpected AUTO option '" +
                                 tokens[t] + "'" + grammar);
      }
    }
    if (k.rfind("RCUT", 0) == 0)
      throw std::runtime_error("OPTIM " + k + ": AUTO boxes are not allowed on cutoff keys "
                               "(the neighbour list is sized from an explicit high bound); "
                               "give explicit bounds");
    if (!mode_set) {
      rule.geometric = optim_key_positive_only(k);
      rule.factor    = rule.geometric ? 2.0 : 0.5;
    }
    /* Non-negative amplitudes: NNLS fits put amplitudes at exactly 0, so a
     * linear rule without a floor gets the refit writer's group floor
     * (1e-3 max|x0| of the line) and the box [0, d] for a zero seed. */
    if (nonneg_amp && !rule.geometric && !floor_set) rule.group_floor = 1e-3;
  } else {
    take_tag(tokens);
    if (tokens.size() != 2) {
      throw std::runtime_error(
        "OPTIM " + k + ": unexpected token(s) '" + core::join(tokens) + "'" +
        grammar);
    }
    low  = parse_num(tokens[0], "<low>");
    high = parse_num(tokens[1], "<high>");
  }

  /* The indices as the line wrote them (for FIXWEIGHT, model weights). */
  const std::vector<size_t> typed_indices = user_indices;
  if (k == "FIXWEIGHT") {
    /* OPTIM FIXWEIGHT indices name MODEL WEIGHTS (the weights FIXINDEX
     * pins), not positions in the FIXWEIGHT value list. Map each user
     * index to its FIXWEIGHT slot through FIXINDEX; declaration order
     * defines the pairing. Back-compatible with `tadah refit` targets,
     * which pin FIXINDEX 1-K so weight index == slot. */
    if (!context.contains("FIXINDEX")) {
      throw std::runtime_error(
        "OPTIM FIXWEIGHT requires FIXINDEX in the training configuration "
        "(FIXINDEX names which model weights the FIXWEIGHT values pin).");
    }
    std::vector<std::string> fi_str(context.size("FIXINDEX"));
    context.get("FIXINDEX", fi_str);
    const std::vector<size_t> fixidx = core::parse_indices(fi_str);
    for (auto& ui : user_indices) {
      auto it = std::find(fixidx.begin(), fixidx.end(), ui);
      if (it == fixidx.end()) {
        throw std::runtime_error(
          "OPTIM FIXWEIGHT (" + core::to_string(ui) + "): weight " +
          core::to_string(ui) + " is not pinned by FIXINDEX (" + core::join(fi_str)
          + "). OPTIM FIXWEIGHT indices name MODEL WEIGHTS, not FIXWEIGHT "
          "slots; add the weight to FIXINDEX/FIXWEIGHT or drop it from "
          "this OPTIM line.");
      }
      ui = static_cast<size_t>(it - fixidx.begin()) + 1;  // 1-based slot
    }
  }

  context.get<std::vector<std::string>>(k, omap.config);

  /* AUTO: the group floor is r * max|x0| over the indices of THIS line. */
  double group_max = 0.0;
  if (auto_box) {
    for (const auto& ui : user_indices) {
      const size_t i = ui_offset(omap.config, ui);
      if (i < s && core::is_number(omap.config[i]))
        group_max = std::max(group_max, std::fabs(core::to_double(omap.config[i])));
    }
  }
  /* A geometric box cannot reach 0: non-negative amplitudes take RATIO only
   * when every seed of the line is > 0. */
  if (auto_box && nonneg_amp && rule.geometric) {
    for (const auto& ui : user_indices) {
      const size_t i = ui_offset(omap.config, ui);
      if (i >= s || !core::is_number(omap.config[i])) continue;  // reported below
      if (core::to_double(omap.config[i]) > 0.0) continue;
      throw std::runtime_error(
          "OPTIM " + k + " (" + core::to_string(ui) + "): AUTO RATIO under " +
          form_name + " needs every seed on the line > 0 (got " +
          omap.config[i] + "); use FRAC (the default), whose box is clamped "
          "at 0");
    }
  }

  for (std::size_t j = 0; j < user_indices.size(); ++j) {
    const size_t ui = user_indices[j];
    try {
      size_t i = ui_offset(omap.config, ui);
      if (i>=s) throw std::runtime_error("Out of bound index: " + core::to_string(ui));
      /* One OPTIM line per index: a second box for an index would replace
       * the first without a word. Lines of one key that select other
       * indices add to it (in any case, the key being upper-cased). */
      if (omap.opt[i])
        throw std::runtime_error(
            "the index is already selected by the OPTIM " + k +
            " line on line " + core::to_string(omap.line[i]) +
            " (OPTIM keys are case-insensitive); give each index one OPTIM "
            "line");
      double lo_i = low, hi_i = high;
      if (auto_box) {
        if (!core::is_number(omap.config[i]))
          throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                   "): AUTO needs a numeric CONFIG seed (got '" +
                                   omap.config[i] + "')");
        const double x0 = core::to_double(omap.config[i]);
        if (nonneg_amp && x0 < 0.0)
          throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                   "): the seed " + omap.config[i] + " is negative, but " +
                                   form_name + " requires every AMPGRIDMB amplitude >= 0");
        if (rule.geometric) {
          if (!(x0 != 0.0))
            throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                     "): AUTO RATIO needs a non-zero seed; use FRAC + FLOOR");
          const double a = x0 / rule.factor, b = x0 * rule.factor;
          lo_i = std::min(a, b); hi_i = std::max(a, b);
          omap.origin[i] = BoxOrigin::AutoGeometric;
        } else {
          const double d = std::max({rule.factor * std::fabs(x0), rule.floor,
                                     rule.group_floor * group_max});
          if (!(d > 0.0) && nonneg_amp && !floor_set)
            throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                     "): every seed on this line is 0, so the AUTO box has "
                                     "zero width (FRAC and the GROUP_FLOOR 1e-3 that " +
                                     form_name + " implies for AMPGRIDMB are relative to the "
                                     "seeds); set FLOOR <a> for the box [0, a], or give "
                                     "explicit bounds `OPTIM " + k + " (" +
                                     core::to_string(ui) + ") 0 <high>`");
          if (!(d > 0.0))
            throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                     "): AUTO FRAC gives a zero-width box (seed 0 or FRAC 0); " +
                                     ((rule.group_floor > 0.0 && !(group_max > 0.0))
                                        ? "every seed on this line is 0, so GROUP_FLOOR has "
                                          "nothing to scale; set FLOOR"
                                        : "set FLOOR or GROUP_FLOOR"));
          lo_i = x0 - d; hi_i = x0 + d;
          omap.origin[i] = BoxOrigin::AutoLinear;
          /* non-negative amplitudes: lo = max(0, x0 - d), before the LOG
           * check below (a LOG box cannot start at 0) */
          if (nonneg_amp) {
            if (lo_i < 0.0) lo_i = 0.0;
            omap.origin[i] = BoxOrigin::AutoLinearNonneg;
          }
        }
        if (tf.kind == ScaleKind::Log && lo_i <= 0.0)
          throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                   "): AUTO box [" + core::to_string(lo_i) + ", " +
                                   core::to_string(hi_i) + "] is not positive but the dim is "
                                   "tagged LOG; use RATIO (geometric) or drop the tag");
      }
      if (low_rule && !low_rule->admits(lo_i))
        throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                 "): the low bound " + core::to_string(lo_i) +
                                 (low_rule->strict ? " is not > 0" : " is negative") +
                                 ", but " + low_rule->by + " requires " +
                                 low_rule->requirement + "; use a low bound " +
                                 (low_rule->strict ? "> 0"
                                                   : ">= 0 (0 is allowed) or AUTO"));
      omap.opt[i] = true;
      omap.line[i] = nline;
      omap.highbound[i] = hi_i;
      omap.lowbound[i] = lo_i;
      omap.scale[i] = tf;
    } catch (const std::exception& e) {
      throw std::runtime_error("OPTIM " + k + " (" +
                               core::to_string(typed_indices[j]) + "): " +
                               e.what());
    }
  }
  for (size_t i = 0; i < omap.lowbound.size(); ++i) {
    if (omap.opt[i] && omap.highbound[i] <= omap.lowbound[i]) {
      throw std::runtime_error("Error: Highbound must be greater than lowbound: "
      + core::to_string(omap.lowbound[i])+">="+core::to_string(omap.highbound[i])+ "\n");
    }
  }
}
} // namespace hpo
} // namespace tadah

