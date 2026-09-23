#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/scale_validation.h>
#include <tadah/mlip/engines/hpotarget_writer.h>
#include <cctype>
#include <cstdlib>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/core/registry.h>
#include <tadah/hpo/loss_function.h>

#include <algorithm>
#include <iomanip>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <unordered_map>

namespace tadah {
namespace hpo {

namespace fs = std::filesystem;

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
  config.resize(s);
  scale.resize(s);
  origin.resize(s, BoxOrigin::Explicit);
  lowbound.resize(s, 0);
  highbound.resize(s, 0);
}

HPOTargets::HPOTargets(core::Context &c, std::string fn) : context(c) {
  models::D_mJoin::expand_grids(context);

  read(fn);
  std::stable_sort(LS.begin(), LS.end(), LammpsSim::greater);
  /* read() records failprobe_idx against parse order; the sort above
   * invalidates it. Recover it from the flag carried on the sim itself
   * (exactly one probe -- enforced at parse). */
  if (failprobe_idx >= 0) {
    failprobe_idx = -1;
    for (std::size_t i = 0; i < LS.size(); ++i)
      if (LS[i].is_failprobe) { failprobe_idx = static_cast<int>(i); break; }
    if (failprobe_idx < 0)
      throw std::runtime_error(
          "internal: PC_FAILPROBE sim lost during LS ordering");
  }
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
     * else the auto-derived fd_step would be a dead variable.
     *
     * BASINS leaves the OUTER lib/algo unset (the per-stage EXPLORE/
     * REFINE specs drive the work), so a caps lookup here would throw.
     * Treat unset OUTER as "not gradient-based" -- NOISE_AUTO is a
     * leaf-optimiser knob anyway and would have nothing to write into
     * a stage sub-spec. */
    const bool gradient_based = (spec.lib != "NOT SET") &&
        optimizer_caps(spec).is_gradient_based;
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
     * CONFIG, and CONFIG/WARM ignore ctx.seed. The seed still reaches
     * nlopt::srand in make_nlopt, but only stochastic non-LDS NLopt
     * algos (G_MLSL, CRS, ISRES, ESCH, ...) ever read that RNG.
     * Surface it so "I set SEED and nothing changed" is answerable
     * from the log. */
    /* SEED inside the INIT block seeds the init-strategy chain ONLY --
     * it does not seed the optimiser itself. For the ANNEAL family
     * (whose proposal stream falls back to random_device without an
     * optimiser-level SEED) this is a classic footgun: the user "set a
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
      const auto& chain = s.init_strategy_spec.chain;
      const bool seed_consumed = std::any_of(
          chain.begin(), chain.end(),
          [](const InitStrategySpec::Entry& e) {
            return e.method == "RANDOM" || e.method == "LHS" ||
                   e.method == "SOBOL";
          });
      if (!seed_consumed) {
        std::ostringstream oss;
        oss << "INIT[" << where << "]: SEED is set but the strategy chain ("
            << (chain.empty() ? std::string("empty -> CONFIG fallback")
                              : [&]{ std::string m;
                                     for (const auto& e : chain) {
                                       if (!m.empty()) m += ", ";
                                       m += e.method;
                                     } return m; }())
            << ") contains no stochastic method, so the seed does not "
               "affect the starting point. Add METHOD RANDOM/LHS/SOBOL "
               "to randomise it. (The seed still feeds NLopt's global "
               "RNG, which only stochastic non-LDS algos consume.)";
        std::cerr << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }
    std::uint64_t seed = s.init_strategy_spec.seed.value_or(
        static_cast<std::uint64_t>(std::random_device{}()));
    InitContext ictx{s, *this, seed};
    std::size_t k = s.init_strategy_spec.k_total.value_or(strategy->default_k());
    s.optim_init = strategy->generate(ictx, k);
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
     * ingests >1 seed; strategy decorators (MULTISTART/PRESTAGE/
     * BASINS) consume the chain too. Every other path drops
     * everything past optim_init.front().
     *
     * BASINS leaves OUTER lib/algo unset; the strategy_type check
     * already short-circuits, but we order it first so the caps lookup
     * is only attempted when the OUTER spec has a real (lib, algo). */
    const bool strategy_consumes =
        (spec.strategy_type == StrategyType::MULTISTART && spec.multistart.n_starts >= 2) ||
        (spec.strategy_type == StrategyType::PRESTAGE && spec.prestage != nullptr) ||
        (spec.strategy_type == StrategyType::BASINS);
    const bool backend_consumes =
        (spec.lib != "NOT SET") && optimizer_caps(spec).accepts_multi_seed;
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
     * starting points. */
    if (spec.prestage->init_strategy_spec.chain.empty()) {
      /* Default: inherit from OUTER. */
      if (spec.prestage->optim_init.empty())
        spec.prestage->optim_init = spec.optim_init;
    } else {
      resolve_init(*spec.prestage, "PRESTAGE");
    }
    /* PRESTAGE's nested LOCAL: inherit OUTER's seed by default, or
     * resolve its own chain when one was given. */
    if (spec.prestage->inner) {
      if (spec.prestage->inner->init_strategy_spec.chain.empty()) {
        if (spec.prestage->inner->optim_init.empty())
          spec.prestage->inner->optim_init = spec.optim_init;
      } else {
        resolve_init(*spec.prestage->inner, "PRESTAGE.INNER");
      }
    }
  }

  /* For BASINS: resolve INIT chains on EXPLORE/REFINE if they have one,
   * else inherit OUTER's optim_init so the seed point flows in. */
  if (spec.strategy_type == StrategyType::BASINS) {
    auto resolve_or_inherit = [&](OptimizerSpec& s, const char* tag) {
      if (s.init_strategy_spec.chain.empty()) {
        if (s.optim_init.empty())
          s.optim_init = spec.optim_init;
      } else {
        resolve_init(s, tag);
      }
    };
    if (spec.basins.explore) resolve_or_inherit(*spec.basins.explore, "BASINS.EXPLORE");
    if (spec.basins.refine ) resolve_or_inherit(*spec.basins.refine,  "BASINS.REFINE");
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
        a.chain.push_back(std::move(e));
        return true;
    }
    if (key=="K_TOTAL") { std::size_t v; iss>>v; a.k_total=v; return true; }
    if (key=="SEED")    { std::uint64_t v; iss>>v; a.seed=v; return true; }
    if (key=="WARM_ON_MISMATCH") {
        std::string v; iss>>v;
        if (v!="warn" && v!="error" && v!="skip")
            throw std::runtime_error("WARM_ON_MISMATCH must be one of: warn, error, skip");
        a.warm_on_mismatch = v;
        return true;
    }
    return false;
}

/* Helper: throw a "renamed to X" error for any old-grammar keyword.
 * Returns true if the key was a known-old keyword (caller should not
 * fall through to an "unknown key" error). */
static bool reject_old_keyword(const std::string& key)
{
    struct Rename { const char* old_; const char* hint; };
    static const Rename map[] = {
        {"LOCAL",                       "renamed to INNER (the MLSL/AUGLAG inner sub-optimizer block). For BASINS' per-basin refinement use REFINE/ENDREFINE inside STRATEGY."},
        {"ENDLOCAL",                    "renamed to ENDINNER."},
        {"PRESTAGE",                    "replaced by `STRATEGY { TYPE PRESTAGE EXPLORE { ... } [REFINE { ... }] }` (REFINE is optional; without it the STRATEGY-level LIB/ALGO refines)."},
        {"ENDPRESTAGE",                 "replaced by ENDEXPLORE / ENDREFINE / ENDSTRATEGY (see STRATEGY block)."},
        {"GLOBAL_LIB",                  "moved into `STRATEGY { TYPE BASINS EXPLORE { LIB ... } }`."},
        {"GLOBAL_ALGO",                 "moved into `STRATEGY { TYPE BASINS EXPLORE { ALGO ... } }`."},
        {"LOCAL_LIB",                   "moved into `STRATEGY { TYPE BASINS REFINE { LIB ... } }`."},
        {"LOCAL_ALGO",                  "moved into `STRATEGY { TYPE BASINS REFINE { ALGO ... } }`."},
        {"TOL_GLO_LADDER",              "moved into `EXPLORE { FTOL_LADDER ... }` inside STRATEGY."},
        {"TOL_LOC_LADDER",              "moved into `REFINE { FTOL_LADDER ... }` inside STRATEGY."},
        {"STAGE_RATE_FACTOR",           "split into per-stage `EXPLORE { BUDGET_RATE_FACTOR ... }` and `REFINE { BUDGET_RATE_FACTOR ... }`."},
        {"STAGE_MIN_EVALS",             "split into per-stage `EXPLORE/REFINE { BUDGET_MIN_EVALS ... }`."},
        {"STAGE_MAX_EVALS",             "split into per-stage `EXPLORE/REFINE { BUDGET_MAX_EVALS ... }`."},
        {"EMERGENCY_NO_IMPROVE_ROUNDS", "renamed to STOP_NO_IMPROVE_ROUNDS (inside STRATEGY)."},
        {"GLOBAL_SEED_TOP_N",           "renamed to SEED_TOP_N (inside STRATEGY)."},
        {"CLAMP_LADDER_TO_NOISE_FLOOR", "renamed to CLAMP_TO_NOISE_FLOOR (inside STRATEGY)."},
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
     *       +-- STRATEGY    STRATEGY ... ENDSTRATEGY  (required, drives StrategyType)
     *           +-- EXPLORE     EXPLORE ... ENDEXPLORE    (only with TYPE BASINS or PRESTAGE)
     *           |   +-- INNER       INNER ... ENDINNER       (MLSL inner sub-optimiser)
     *           +-- REFINE      REFINE ... ENDREFINE      (only with TYPE BASINS or PRESTAGE)
     *           |   +-- INNER       INNER ... ENDINNER
     *           +-- INNER       INNER ... ENDINNER       (when LIB/ALGO at STRATEGY level -- TYPE PLAIN/MULTISTART)
     *           +-- RESTART     RESTART ... ENDRESTART   (decorator; populates spec.restart) */
    enum Block { ROOT, OUTER, INIT_BLK, NOISE_BLK, STRATEGY_BLK,
                 EXPLORE_BLK, REFINE_BLK, INNER_BLK, RESTART_BLK };
    Block            block      = ROOT;
    Block            prev_inner = ROOT;       // restored on ENDINNER
    OptimizerSpec*   active_opt    = nullptr;  // recipient of LIB/ALGO/...
    InitStrategySpec* active_init  = nullptr;  // recipient of STRATEGY/K_TOTAL/...
    OptimizerSpec*   inner_parent  = nullptr;  // outer spec of an INNER block

    /* Helper: re-derive active_opt for a non-INNER block. */
    auto refresh_active_opt = [&]() {
        switch (block) {
            case OUTER:        active_opt = &spec; break;
            case STRATEGY_BLK: active_opt = &spec; break;
            case EXPLORE_BLK:
                if (spec.strategy_type == StrategyType::BASINS)
                    active_opt = spec.basins.explore.get();
                else if (spec.strategy_type == StrategyType::BASINSV2)
                    active_opt = spec.basins_v2.explore.get();
                else  /* PRESTAGE */
                    active_opt = spec.prestage.get();
                break;
            case REFINE_BLK:
                if (spec.strategy_type == StrategyType::BASINS)
                    active_opt = spec.basins.refine.get();
                else if (spec.strategy_type == StrategyType::BASINSV2)
                    active_opt = spec.basins_v2.refine.get();
                else  /* PRESTAGE -- REFINE is the OUTER spec */
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
            if (key=="ENDOPTIMIZER") { block=ROOT;  active_opt=nullptr;  continue; }

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
                block = STRATEGY_BLK;
                refresh_active_opt();
                continue;
            }
            if (key=="ENDSTRATEGY") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("ENDSTRATEGY without matching STRATEGY");
                block = OUTER;
                refresh_active_opt();
                continue;
            }

            if (key=="EXPLORE") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("EXPLORE block must appear inside STRATEGY");
                if (spec.strategy_type != StrategyType::BASINS &&
                    spec.strategy_type != StrategyType::BASINSV2 &&
                    spec.strategy_type != StrategyType::PRESTAGE)
                    throw std::runtime_error(
                        "EXPLORE block is only valid with TYPE BASINS, BASINSV2, or PRESTAGE "
                        "(declare TYPE before EXPLORE)");
                if (spec.strategy_type == StrategyType::PRESTAGE) {
                    if (!spec.prestage)
                        spec.prestage = std::make_unique<OptimizerSpec>();
                }
                block = EXPLORE_BLK;
                refresh_active_opt();
                continue;
            }
            if (key=="ENDEXPLORE") {
                if (block != EXPLORE_BLK)
                    throw std::runtime_error("ENDEXPLORE without matching EXPLORE");
                block = STRATEGY_BLK;
                refresh_active_opt();
                continue;
            }

            if (key=="REFINE") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("REFINE block must appear inside STRATEGY");
                if (spec.strategy_type != StrategyType::BASINS &&
                    spec.strategy_type != StrategyType::BASINSV2 &&
                    spec.strategy_type != StrategyType::PRESTAGE)
                    throw std::runtime_error(
                        "REFINE block is only valid with TYPE BASINS, BASINSV2, or PRESTAGE "
                        "(declare TYPE before REFINE)");
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
                        "line " + std::to_string(nline) + ": " + key +
                        " takes no further value here (unexpected '" +
                        extra_tok + "')");
                continue;
            }

            /* INIT / RESTART block keys */
            if (block == INIT_BLK || block == RESTART_BLK) {
                if (active_init && dispatch_init_attr(key, iss, *active_init))
                    continue;
                if (block == RESTART_BLK) {
                    if (key=="AFTER_STAGNATION") {
                        std::size_t v; iss>>v;
                        spec.restart.after_stagnation = v;
                        continue;
                    }
                    if (key=="MAX") {
                        std::size_t v; iss>>v;
                        spec.restart.max = v;
                        continue;
                    }
                }
                throw std::runtime_error("unknown key '" + key + "' inside "
                    + std::string(block==INIT_BLK ? "INIT" : "RESTART") + " block");
            }

            /* NOISE block keys */
            if (block == NOISE_BLK) {
                if (key=="AUTO") {
                    std::string v; iss>>v;
                    spec.noise.autodetect = v.empty() ? true : parse_bool(v, "NOISE.AUTO");
                    continue;
                }
                if (key=="CALIBRATION") {
                    std::size_t v; iss>>v;
                    if (!iss)
                        throw std::runtime_error("NOISE.CALIBRATION expects a non-negative integer");
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
                    double v = 0.0;
                    if (!(iss >> v) || !std::isfinite(v) || !(v >= 1.0))
                        throw std::runtime_error(
                            "LOG_HP must be a finite ratio >= 1 (a dimension is "
                            "log-scaled when high/low >= LOG_HP and low > 0)");
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
                    std::string t; iss>>t;
                    if      (t=="PLAIN")      spec.strategy_type = StrategyType::PLAIN;
                    else if (t=="BASINS")     spec.strategy_type = StrategyType::BASINS;
                    else if (t=="BASINSV2")   spec.strategy_type = StrategyType::BASINSV2;
                    else if (t=="MULTISTART") spec.strategy_type = StrategyType::MULTISTART;
                    else if (t=="PRESTAGE")   spec.strategy_type = StrategyType::PRESTAGE;
                    else throw std::runtime_error("STRATEGY TYPE must be PLAIN, BASINS, BASINSV2, MULTISTART, or PRESTAGE (got '" + t + "')");
                    /* For BASINS / BASINSV2, allocate the EXPLORE/REFINE stage
                     * sub-specs with NLopt defaults so a minimal
                     * `STRATEGY { TYPE ... }` config is functional. (Not
                     * done in the spec's default ctor to avoid recursive
                     * default-construction -- see optimizer_spec.h comment.) */
                    if (spec.strategy_type == StrategyType::BASINS)
                        basins_spec_ensure_defaults(spec.basins);
                    else if (spec.strategy_type == StrategyType::BASINSV2)
                        basins_v2_spec_ensure_defaults(spec.basins_v2);
                    continue;
                }
                /* MULTISTART knobs (only meaningful with TYPE MULTISTART). */
                if (key=="N_STARTS") {
                    std::size_t v; iss>>v;
                    if (v == 0) throw std::runtime_error("N_STARTS must be >= 1");
                    spec.multistart.n_starts = v;
                    continue;
                }
                if (key=="PARALLEL") {
                    std::size_t v; iss>>v;
                    if (v == 0) throw std::runtime_error("PARALLEL must be >= 1");
                    spec.multistart.parallel = v;
                    continue;
                }
                if (key=="BUDGET_PER_START") {
                    std::size_t v; iss>>v;
                    if (v == 0) throw std::runtime_error("BUDGET_PER_START must be >= 1");
                    spec.multistart.budget_per_start = v;
                    continue;
                }
                /* PRESTAGE knob. */
                if (key=="TOPK") {
                    std::size_t v; iss>>v;
                    if (v == 0) throw std::runtime_error("TOPK must be >= 1");
                    spec.stage_topk = v;
                    continue;
                }
                /* BASINS / BASINSV2 shared knobs. The destination depends
                 * on STRATEGY TYPE (set above by the TYPE key).  */
                const bool is_v1 = (spec.strategy_type == StrategyType::BASINS);
                const bool is_v2 = (spec.strategy_type == StrategyType::BASINSV2);

                if (key=="BASIN_POOL_MAX") {
                    std::size_t v; iss>>v;
                    if (v == 0) throw std::runtime_error("BASIN_POOL_MAX must be >= 1");
                    if (is_v1)      spec.basins.basin_pool_max    = v;
                    else if (is_v2) spec.basins_v2.basin_pool_max = v;
                    else throw std::runtime_error("BASIN_POOL_MAX requires TYPE BASINS or BASINSV2");
                    continue;
                }
                if (key=="BASIN_DEDUP_DIST") {
                    double v; iss>>v;
                    if (v < 0.0) throw std::runtime_error("BASIN_DEDUP_DIST must be >= 0");
                    if (is_v1)      spec.basins.basin_dedup_dist    = v;
                    else if (is_v2) spec.basins_v2.basin_dedup_dist = v;
                    else throw std::runtime_error("BASIN_DEDUP_DIST requires TYPE BASINS or BASINSV2");
                    continue;
                }
                if (key=="STOP_NO_IMPROVE_ROUNDS") {
                    std::size_t v; iss>>v;
                    if (!is_v1) throw std::runtime_error("STOP_NO_IMPROVE_ROUNDS requires TYPE BASINS (use STOP_NO_IMPROVE_ITERATIONS for BASINSV2)");
                    spec.basins.stop_no_improve_rounds = v;
                    continue;
                }
                if (key=="SEED_TOP_N") {
                    std::size_t v; iss>>v;
                    if (v == 0) throw std::runtime_error("SEED_TOP_N must be >= 1");
                    if (!is_v1) throw std::runtime_error("SEED_TOP_N requires TYPE BASINS (BASINSV2 uses Sobol+filtered EXPLORE seeding)");
                    spec.basins.seed_top_n = v;
                    continue;
                }
                if (key=="CLAMP_TO_NOISE_FLOOR") {
                    std::string v; iss>>v;
                    const bool b = parse_bool(v, "CLAMP_TO_NOISE_FLOOR");
                    if (is_v1)      spec.basins.clamp_to_noise_floor    = b;
                    else if (is_v2) spec.basins_v2.clamp_to_noise_floor = b;
                    else throw std::runtime_error("CLAMP_TO_NOISE_FLOOR requires TYPE BASINS or BASINSV2");
                    continue;
                }
                if (key=="CHECKPOINT_EVERY") {
                    std::string tok; iss>>tok;
                    if (tok.empty())
                        throw std::runtime_error("CHECKPOINT_EVERY requires a value (e.g. \"60s\" or \"100\")");
                    double secs = 0.0; std::size_t evs = 0;
                    if (tok.back() == 's' || tok.back() == 'S') {
                        secs = std::stod(tok.substr(0, tok.size()-1));
                        if (secs < 0.0) throw std::runtime_error("CHECKPOINT_EVERY seconds must be >= 0");
                    } else {
                        evs = std::stoul(tok);
                    }
                    if (is_v1) {
                        spec.basins.checkpoint_seconds = secs;
                        spec.basins.checkpoint_evals   = evs;
                    } else if (is_v2) {
                        spec.basins_v2.checkpoint_seconds = secs;
                        spec.basins_v2.checkpoint_evals   = evs;
                    } else {
                        throw std::runtime_error("CHECKPOINT_EVERY requires TYPE BASINS or BASINSV2");
                    }
                    continue;
                }

                /* BASINSV2-only knobs. */
                if (is_v2) {
                    if (key=="BASIN_MERGE_DIST") {
                        double v; iss>>v;
                        if (v < 0.0) throw std::runtime_error("BASIN_MERGE_DIST must be >= 0");
                        spec.basins_v2.basin_merge_dist = v; continue;
                    }
                    if (key=="BASIN_MERGE_LOSS_THRESHOLD") {
                        double v; iss>>v;
                        if (v < 0.0) throw std::runtime_error("BASIN_MERGE_LOSS_THRESHOLD must be >= 0");
                        spec.basins_v2.basin_merge_loss_threshold = v; continue;
                    }
                    if (key=="ARCHIVE_MAX") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.archive_max = v; continue;
                    }
                    if (key=="INACTIVE_AFTER_EVALS") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.inactive_after_evals = v; continue;
                    }
                    if (key=="VERIFY_CADENCE") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.verify_cadence = v; continue;
                    }
                    if (key=="VERIFY_MAX_ATTEMPTS") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.verify_max_attempts = v; continue;
                    }
                    if (key=="VERIFY_EVALS_CAP") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.verify_evals_cap = v; continue;
                    }
                    if (key=="MERGE_CADENCE") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.merge_cadence = v; continue;
                    }
                    if (key=="EXPLORE_KICK_CADENCE") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.explore_kick_cadence = v; continue;
                    }
                    if (key=="EXPLORE_DIST_FLOOR") {
                        double v; iss>>v;
                        if (v < 0.0) throw std::runtime_error("EXPLORE_DIST_FLOOR must be >= 0");
                        spec.basins_v2.explore_dist_floor = v; continue;
                    }
                    if (key=="SUMMARY_CADENCE") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.summary_cadence = v; continue;
                    }
                    if (key=="STARVATION_CAP") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.starvation_cap = v; continue;
                    }
                    if (key=="STOP_NO_IMPROVE_ITERATIONS") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.stop_no_improve_iterations = v; continue;
                    }
                    if (key=="PLATEAU_EPS_FACTOR") {
                        double v; iss>>v;
                        if (v <= 0.0) throw std::runtime_error("PLATEAU_EPS_FACTOR must be > 0");
                        spec.basins_v2.plateau_eps_factor = v; continue;
                    }
                    if (key=="PATH_MAX_POINTS") {
                        std::size_t v; iss>>v;
                        spec.basins_v2.path_max_points = v; continue;
                    }
                    if (key=="PATH_LOSS_EPS") {
                        double v; iss>>v;
                        if (v < 0.0) throw std::runtime_error("PATH_LOSS_EPS must be >= 0");
                        spec.basins_v2.path_loss_eps = v; continue;
                    }
                    if (key=="PATH_DIST_EPS") {
                        double v; iss>>v;
                        if (v < 0.0) throw std::runtime_error("PATH_DIST_EPS must be >= 0");
                        spec.basins_v2.path_dist_eps = v; continue;
                    }
                    if (key=="PRIORITY_WEIGHTS") {
                        /* Six floats: improve neglect distance tol_gap state age */
                        double w_imp, w_neg, w_dist, w_tol, w_st, w_age;
                        if (!(iss >> w_imp >> w_neg >> w_dist >> w_tol >> w_st >> w_age))
                            throw std::runtime_error("PRIORITY_WEIGHTS expects 6 floats: improve neglect distance tol_gap state age");
                        spec.basins_v2.w_improve  = w_imp;
                        spec.basins_v2.w_neglect  = w_neg;
                        spec.basins_v2.w_distance = w_dist;
                        spec.basins_v2.w_tol_gap  = w_tol;
                        spec.basins_v2.w_state    = w_st;
                        spec.basins_v2.w_age      = w_age;
                        continue;
                    }
                    if (key=="VERBOSE_DIAGNOSTICS") {
                        std::string v; iss>>v;
                        spec.basins_v2.verbose_diagnostics = parse_bool(v, "VERBOSE_DIAGNOSTICS");
                        continue;
                    }
                }

                throw std::runtime_error("unknown key '" + key + "' inside STRATEGY block");
            }

            /* EXPLORE / REFINE block keys (per-stage BASINS knobs).
             * Only meaningful for TYPE BASINS -- PRESTAGE doesn't iterate
             * stages so it doesn't need ladders/budgets. */
            if (block == EXPLORE_BLK || block == REFINE_BLK) {
                const bool is_explore = (block == EXPLORE_BLK);
                const bool is_v1 = (spec.strategy_type == StrategyType::BASINS);
                const bool is_v2 = (spec.strategy_type == StrategyType::BASINSV2);
                /* FTOL_LADDER and BUDGET_* meaningful for BASINS / BASINSV2. */
                if (key=="FTOL_LADDER") {
                    if (!is_v1 && !is_v2)
                        throw std::runtime_error("FTOL_LADDER only valid for TYPE BASINS or BASINSV2");
                    std::vector<double> rungs; double v;
                    while (iss >> v) {
                        if (v <= 0.0) throw std::runtime_error("FTOL_LADDER entries must be positive");
                        rungs.push_back(v);
                    }
                    if (rungs.empty()) throw std::runtime_error("FTOL_LADDER requires at least one positive value");
                    if (is_v1) {
                        if (is_explore) spec.basins.explore_ftol_ladder = std::move(rungs);
                        else            spec.basins.refine_ftol_ladder  = std::move(rungs);
                    } else {
                        if (is_explore) spec.basins_v2.explore_ftol_ladder = std::move(rungs);
                        else            spec.basins_v2.refine_ftol_ladder  = std::move(rungs);
                    }
                    continue;
                }
                if (key=="BUDGET_MAX_EVALS") {
                    if (!is_v1 && !is_v2)
                        throw std::runtime_error("BUDGET_MAX_EVALS only valid for TYPE BASINS or BASINSV2");
                    std::size_t v; iss>>v;
                    if (is_v1) {
                        if (is_explore) spec.basins.explore_budget_max_evals = v;
                        else            spec.basins.refine_budget_max_evals  = v;
                    } else {
                        if (is_explore) spec.basins_v2.explore_budget_max_evals = v;
                        else            spec.basins_v2.refine_budget_max_evals  = v;
                    }
                    continue;
                }
                if (key=="BUDGET_MIN_EVALS") {
                    if (!is_v1 && !is_v2)
                        throw std::runtime_error("BUDGET_MIN_EVALS only valid for TYPE BASINS or BASINSV2");
                    std::size_t v; iss>>v;
                    if (is_v1) {
                        if (is_explore) spec.basins.explore_budget_min_evals = v;
                        else            spec.basins.refine_budget_min_evals  = v;
                    } else {
                        if (is_explore) spec.basins_v2.explore_budget_min_evals = v;
                        else            spec.basins_v2.refine_budget_min_evals  = v;
                    }
                    continue;
                }
                if (key=="BUDGET_RATE_FACTOR") {
                    if (!is_v1 && !is_v2)
                        throw std::runtime_error("BUDGET_RATE_FACTOR only valid for TYPE BASINS or BASINSV2");
                    double v; iss>>v;
                    if (v < 0.0) throw std::runtime_error("BUDGET_RATE_FACTOR must be >= 0");
                    if (is_v1) {
                        if (is_explore) spec.basins.explore_budget_rate_factor = v;
                        else            spec.basins.refine_budget_rate_factor  = v;
                    } else {
                        if (is_explore) spec.basins_v2.explore_budget_rate_factor = v;
                        else            spec.basins_v2.refine_budget_rate_factor  = v;
                    }
                    continue;
                }
                throw std::runtime_error("unknown key '" + key + "' inside "
                    + std::string(is_explore ? "EXPLORE" : "REFINE") + " block");
            }

            /* INNER block -- only opt attributes, dispatched above. */
            if (block == INNER_BLK) {
                throw std::runtime_error("unknown key '" + key + "' inside INNER block "
                    "(only " + opt_attr_keys_joined("/") + " are valid)");
            }

            /* Anything still unhandled inside an OPTIMIZER-level block is
             * a user mistake (LIB/ALGO/MAXEVAL etc. now live inside
             * STRATEGY). LOSS / PC_LAMMPS / OPTIM / FAILSCORE / OUTPUT /
             * BDUMP and friends are ROOT-level and fall through. */
            if (block != ROOT) {
                throw std::runtime_error("unknown key '" + key + "' in this block");
            }

            /* ==================  global (ROOT) section  =========== */

            /* optimizer-independent settings ----------------------- */
            // if      (key=="THREADS"){ double v; iss>>v; spec.param["THREADS"]=v; }
            if (key=="FAILSCORE"){ iss>>FAIL_SCORE; }
            /* PC_SCRIPT_TIMEOUT: per-pc_script soft wall-clock limit (SECONDS)
             * injected as LAMMPS `timer timeout` after each clear. MAXTIME is
             * the legacy alias, kept so existing .hpo configs keep parsing.
             * NOTE: the OPTIMIZER-block MAXTIME (whole-run stop) is a DIFFERENT
             * setting parsed in optimizer_block.cpp -- do not conflate. If both
             * spellings appear here, the later line wins (each overwrites
             * MAX_TIME); hpo.cpp warns when both were given. */
            else if (key=="PC_SCRIPT_TIMEOUT" || key=="MAXTIME") {
                iss>>MAX_TIME;
                MAX_TIME_key = key;
                if (key=="MAXTIME") MAX_TIME_seen_legacy = true;
                else                MAX_TIME_seen_new    = true;
            }
            else if (key=="PC_ISOLATION") {
                iss>>PC_ISOLATION;
                PC_ISOLATION_set = true;
                if (PC_ISOLATION != "THREAD" && PC_ISOLATION != "PROCESS")
                    throw std::runtime_error(
                        "PC_ISOLATION must be THREAD or PROCESS, got '"
                        + PC_ISOLATION + "'");
            }
            else if (key=="PC_WORKER_TIMEOUT") {
                std::string v;
                if (!(iss>>v))
                    throw std::runtime_error(
                        "PC_WORKER_TIMEOUT requires a value in seconds");
                PC_WORKER_TIMEOUT = parse_pc_worker_timeout(v, "PC_WORKER_TIMEOUT");
            }
            else if (key=="FAILSCORE_GRADING") {
                std::string v; iss>>v;
                if (v=="ON")       FAILSCORE_GRADING = true;
                else if (v=="OFF") FAILSCORE_GRADING = false;
                else throw std::runtime_error(
                        "FAILSCORE_GRADING must be ON or OFF, got '" + v + "'");
            }
            else if (key=="BUMPGATE") {
                std::string v; iss>>v;
                if (v=="STRICT")         BUMPGATE_STRICT = true;
                else if (v=="CANDIDATE") BUMPGATE_STRICT = false;
                else throw std::runtime_error(
                        "BUMPGATE must be CANDIDATE or STRICT, got '" + v + "'");
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
                        const double v = std::stod(toks[i2+1]);
                        if (toks[i2]=="--gate") failprobe_gate = v;
                        else {
                            if (!std::isfinite(v) || !(v > 0.0))
                                throw std::runtime_error(
                                    "PC_FAILPROBE --s0 must be finite and > 0");
                            failprobe_s0 = v;
                        }
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
            else if (key=="QFLUSH")   { iss>>QFLUSH_THRESHOLD; }
            /* D14: these are moduli (`iter % every`), so 0 is an integer
             * division by zero -- SIGFPE on the first evaluation, with no
             * message. "0" is also the spelling a user naturally reaches for
             * to mean "do not write these files", which makes it a trap
             * rather than a typo. Reject it at the parse site, where the
             * line number is still known, in the same shape as DIGITS
             * below. */
            else if (key=="OUTPUT")   { iss>>outputEvery;
                if (!iss || outputEvery < 1)
                    throw std::runtime_error(
                        "OUTPUT must be >= 1 (it is a cadence: write every "
                        "Nth evaluation). To write as little as possible, "
                        "use a large value such as OUTPUT 1000000."); }
            else if (key=="BOUTPUT")  { iss>>bestOutputEvery;
                if (!iss || bestOutputEvery < 1)
                    throw std::runtime_error(
                        "BOUTPUT must be >= 1 (it is a cadence: write every "
                        "Nth new best)."); }
            else if (key=="DIGITS")   { iss>>outputDigits;
                if (outputDigits < 1 || outputDigits > 15)
                    throw std::runtime_error("DIGITS must be 1-15"); }
            else if (key=="DUMP")     { iss>>dumpPotEvery>>dumpPotDir;
                if (!iss || dumpPotEvery < 1)
                    throw std::runtime_error(
                        "DUMP requires a cadence >= 1 and a directory");
                handle_directory_creation(dumpPotDir); }
            else if (key=="BDUMP")    { iss>>dumpBestPotEvery>>dumpBestPotDir;
                if (!iss || dumpBestPotEvery < 1)
                    throw std::runtime_error(
                        "BDUMP requires a cadence >= 1 and a directory");
                handle_directory_creation(dumpBestPotDir); }

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
                    if (!tadah::hpo::try_parse_double(tok, p))
                        throw std::runtime_error(
                            "LOSS " + type + ": parameter '" + tok +
                            "' is not a number");
                    params.push_back(p);
                }
                auto id = tadah::LossFunction::from_string(type);
                globalLoss = tadah::LossFunction(id,params);
            }

            /* optimisation variables --------------------------------*/
            else if (key=="OPTIM")   { process_optim_config(iss,nline); }
            else if (key=="OPTIM_SCALE") {
                std::string v; iss >> v;
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
                const double tgt = parse_metric_number(target_token);
                const double wgt = parse_metric_number(weight_token);
                if (wgt < 0) throw std::runtime_error(key + " weight must be nonnegative");
                HPOTargets::SkipCaps caps;
                std::string flag;
                while (iss >> flag) {
                    if (flag != "--skip-above" && flag != "--skip-rel")
                        throw std::runtime_error(key + ": unknown flag '" + flag + "'");
                    std::string token;
                    if (!(iss >> token)) throw std::runtime_error(key + " " + flag + " requires a positive number");
                    const double value = parse_metric_number(token);
                    if (!(value > 0)) throw std::runtime_error(key + " " + flag + " requires a positive number");
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
        catch(const std::exception& e)
        {
            throw std::runtime_error("Error on line "+core::to_string(nline)+
                                     ": "+e.what()+"\nLine: "+line);
        }
    }

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

    /* PC_OBSERVABLE always engages the analytical gradient path. The
     * MVP's only observable (cohesive_energy) provides closed-form
     * dO/dW; combined with the Phase 2 fit-side walker, every theta has an
     * analytical chain entry. NUMERICAL with PC_OBSERVABLE would run
     * Ceres finite differences over the whole HPO objective -- correct
     * but slow, and contrary to the redesign's purpose. Reject it at
     * parse time with a clear message. ANALYTICAL and HYBRID are both
     * accepted; HYBRID's masked-FD fallback never fires for
     * PC_OBSERVABLE (every theta is registered), so the two strategies
     * are observationally identical here. */
    auto reject_numerical_with_observable =
        [&](const OptimizerSpec& sp, const char* tag) {
            if (sp.grad_strategy
                && *sp.grad_strategy == GradStrategy::Numerical) {
                throw std::runtime_error(
                    std::string("--hpotarget defines PC_OBSERVABLE but ")
                    + tag + " GRAD_STRATEGY=NUMERICAL is set. PC_OBSERVABLE "
                    "always uses analytical gradients. Set GRAD_STRATEGY "
                    "to HYBRID or ANALYTICAL (they're equivalent for "
                    "PC_OBSERVABLE), or drop the GRAD_STRATEGY line "
                    "entirely.");
            }
        };
    if (!observables.empty()) {
        reject_numerical_with_observable(spec, "OPTIMIZER's");
        if (spec.inner)
            reject_numerical_with_observable(*spec.inner, "INNER's");
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
        [&](const OptimizerSpec& sp, const std::string& fallback_lib,
            const char* tag) {
            if (sp.grad_strategy) {
                throw std::runtime_error(
                    std::string("--hpotarget defines PC_LAMMPS but ") + tag
                    + " GRAD_STRATEGY is set. PC_LAMMPS no longer "
                    "supports analytical/numerical/hybrid gradients "
                    "(2026-05-13). Drop the GRAD_STRATEGY line.");
            }
            if (sp.algo == "NOT SET") return;
            /* INNER block typically only sets ALGO; lib is inherited
             * from the enclosing STRATEGY at run time. Use the outer
             * lib here so the caps lookup works for inner specs that
             * don't carry their own lib string. */
            const std::string lib =
                (sp.lib == "NOT SET") ? fallback_lib : sp.lib;
            if (lib == "NOT SET") return;
            const bool dlib_bfgs_exception =
                (lib == "DLIB" && sp.algo == "BFGS");
            if (dlib_bfgs_exception) return;
            const auto& caps = optimizer_caps(lib, sp.algo);
            if (caps.is_gradient_based) {
                throw std::runtime_error(
                    std::string("--hpotarget defines PC_LAMMPS but ") + tag
                    + " optimiser " + lib + ":" + sp.algo +
                    " is gradient-based. PC_LAMMPS requires a "
                    "gradient-free optimiser (NLOPT GN_*/LN_*, "
                    "TADAH RANDOM/GRID/ANNEAL). The only gradient-based "
                    "exception is DLIB:BFGS (uses per-theta-per-iter "
                    "dtheta, not a full gradient).");
            }
        };
    if (!LS.empty()) {
        reject_pc_lammps_grad(spec, "NOT SET", "OPTIMIZER's");
        if (spec.inner)
            reject_pc_lammps_grad(*spec.inner, spec.lib, "INNER's");
    }
}

double HPOTargets::parse_pc_worker_timeout(const std::string& token,
                                           const std::string& origin) {
  double v = 0.0;
  if (!try_parse_double(token, v) || !std::isfinite(v) || v < 0.0)
    throw std::runtime_error(
        origin + " must be a finite number of seconds >= 0 (0 derives the "
        "deadline from each script's timeout), got '" + token + "'");
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
        try { d.x0 = std::stod(o.config[i]); } catch (const std::exception&) {}
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
      "#   origin       OPTIM = explicit bounds; AUTO FRAC / AUTO RATIO = derived from the seed\n"
      "#   x0           starting point (real units)\n"
      "#   scale, c     shape of the internal coordinate: LIN, LOG (ln x) or ASINH (asinh(x/c));\n"
      "#                c is the ASINH linear scale, 1 for the other shapes\n"
      "#   u0           x0 in optimiser units (0..1 under OPTIM_SCALE AUTO, raw or ln under NONE)\n"
      "#   internal_lo, internal_hi   optimiser-side bounds (0 and 1 under AUTO)\n"
      "#   version      revision of the search space. 1 = the boxes parsed from the config; it\n"
      "#                would only increase if a bound-adaptation step rewrote the boxes during\n"
      "#                the run (no such step exists yet), so a journal or checkpoint reference\n"
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
  hpo << std::setprecision(10);
  for (std::size_t i = 0; i < ss.dim(); ++i) {
    const DimSpec& d = ss.at(i);
    hpo << "OPTIM " << d.key << " (" << (d.user_index ? d.user_index : 1) << ") "
        << d.lo << " " << d.hi;
    if (d.tf.kind == ScaleKind::Log)   hpo << " LOG";
    if (d.tf.kind == ScaleKind::Asinh) hpo << " ASINH " << d.tf.c;
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
  if (spec.strategy_type == StrategyType::BASINS) {
    bind(spec.basins.explore.get());
    bind(spec.basins.refine.get());
  } else if (spec.strategy_type == StrategyType::BASINSV2) {
    bind(spec.basins_v2.explore.get());
    bind(spec.basins_v2.refine.get());
  }
}

void HPOTargets::process_optim_config(std::istringstream& iss, size_t nline) {
  std::string k;
  if (!(iss >> k)) {
    throw std::runtime_error("Failed to read key from input.");
  }

  if (k == "WEIGHTS") {
    throw std::runtime_error(
      "OPTIM WEIGHTS is not a valid search target: the linear WEIGHTS are "
      "re-fitted by least squares on every evaluation, so optimising them "
      "is a silent no-op. To search linear coefficients, pin them with "
      "FIXINDEX + FIXWEIGHT in the training configuration and use "
      "`OPTIM FIXWEIGHT (<weight-index>) <low> <high>`.");
  }

  Opti& omap = optimap[k];
  size_t s = context.size(k);
  omap.resize(s);

  std::vector<std::string> tokens = parse_optim_line(iss);

  /* At least the (indices) group plus one token (AUTO, or <low> <high>);
   * the exact grammar is checked below. */
  if (tokens.size() < 2) {
    throw std::runtime_error(
      "OPTIM " + k + " on line " + core::to_string(nline) +
      ": expected `(indices) <low> <high>` or `(indices) AUTO ...` (got "
      + core::to_string(tokens.size() - (tokens.empty() ? 0 : 1)) + " token(s))");
  }

  std::vector<size_t> user_indices;
  try {
    user_indices = core::parse_indices_strict(tokens.back());
    if (std::find(user_indices.begin(), user_indices.end(), 0) != user_indices.end())
      throw std::runtime_error("OPTIM indices start at 1");
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Error parsing indices on process_optim_config line "
      + core::to_string(nline) + ": " + std::string(e.what()) +
      "\nTokens: " + tokens.back());
  }
  tokens.pop_back();

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
  auto parse_num = [&](const std::string& tok, const char* what) {
    try {
      std::size_t pos = 0;
      const double v = std::stod(tok, &pos);
      if (pos != tok.size()) throw std::invalid_argument("trailing chars");
      return v;
    } catch (const std::exception&) {
      throw std::runtime_error("OPTIM " + k + " on line " + core::to_string(nline) +
                               ": " + what + " must be a number (got '" + tok + "')");
    }
  };
  /* trailing shape tag (both forms) */
  auto take_tag = [&](std::vector<std::string>& toks) {
    if (toks.size() >= 2 && upper(toks[toks.size() - 2]) == "ASINH") {
      tf.kind = ScaleKind::Asinh;
      tf.c = parse_num(toks.back(), "ASINH <c>");
      if (!std::isfinite(tf.c) || !(tf.c > 0.0))
        throw std::runtime_error("OPTIM " + k + " on line " + core::to_string(nline) +
                                 ": ASINH requires a positive linear scale c (got " +
                                 toks.back() + ")");
      toks.pop_back(); toks.pop_back();
      return;
    }
    if (!toks.empty()) {
      const std::string tag = upper(toks.back());
      if (tag == "LIN")      { tf.kind = ScaleKind::Linear; toks.pop_back(); }
      else if (tag == "LOG") { tf.kind = ScaleKind::Log;    toks.pop_back(); }
      else if (tag == "ASINH")
        throw std::runtime_error("OPTIM " + k + " on line " + core::to_string(nline) +
                                 ": ASINH needs one argument <c> (the linear scale)");
    }
  };
  const std::string grammar =
    " -- the grammar is OPTIM <KEY> (indices) <low> <high> [LIN | LOG | ASINH <c>]"
    " or OPTIM <KEY> (indices) AUTO [RATIO <f> | FRAC <p>] [FLOOR <a>] [GROUP_FLOOR <r>]"
    " [LIN | LOG | ASINH <c>]";

  const bool auto_box = !tokens.empty() && upper(tokens.front()) == "AUTO";
  double low = 0.0, high = 0.0;
  struct AutoRule { bool geometric = false; double factor = 0.0; double floor = 0.0; double group_floor = 0.0; } rule;
  if (auto_box) {
    tokens.erase(tokens.begin());
    take_tag(tokens);
    bool mode_set = false;
    for (std::size_t t = 0; t < tokens.size(); t += 2) {
      const std::string opt = upper(tokens[t]);
      if (t + 1 >= tokens.size())
        throw std::runtime_error("OPTIM " + k + " on line " + core::to_string(nline) +
                                 ": AUTO option '" + tokens[t] + "' needs a value" + grammar);
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
      } else if (opt == "GROUP_FLOOR") {
        if (v < 0.0) throw std::runtime_error("OPTIM " + k + ": AUTO GROUP_FLOOR must be >= 0");
        rule.group_floor = v;
      } else {
        throw std::runtime_error("OPTIM " + k + " on line " + core::to_string(nline) +
                                 ": unexpected AUTO option '" + tokens[t] + "'" + grammar);
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
  } else {
    take_tag(tokens);
    if (tokens.size() != 2) {
      throw std::runtime_error(
        "OPTIM " + k + " on line " + core::to_string(nline) +
        ": unexpected token(s) '" + core::join(tokens) + "'" + grammar);
    }
    try {
      high = std::stod(tokens[1]);
    } catch (const std::exception& e) {
      throw std::runtime_error(
        "Error converting high bound to double on process_optim_config line "
        + core::to_string(nline) + ": " + std::string(e.what()) +
        "\nTokens: " + tokens[1]);
    }
    try {
      low = std::stod(tokens[0]);
    } catch (const std::exception& e) {
      throw std::runtime_error(
        "Error converting low bound to double on process_optim_config line "
        + core::to_string(nline) + ": " + std::string(e.what()) +
        "\nTokens: " + tokens[0]);
    }
  }

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
          "OPTIM FIXWEIGHT (" + core::to_string(ui) + ") on line "
          + core::to_string(nline) + ": weight " + core::to_string(ui)
          + " is not pinned by FIXINDEX (" + core::join(fi_str)
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
        group_max = std::max(group_max, std::fabs(std::stod(omap.config[i])));
    }
  }

  for (const auto& ui : user_indices) {
    try {
      size_t i = ui_offset(omap.config, ui);
      if (i>=s) throw std::runtime_error("Out of bound index: " + core::to_string(ui));
      double lo_i = low, hi_i = high;
      if (auto_box) {
        if (!core::is_number(omap.config[i]))
          throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                   "): AUTO needs a numeric CONFIG seed (got '" +
                                   omap.config[i] + "')");
        const double x0 = std::stod(omap.config[i]);
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
          if (!(d > 0.0))
            throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                     "): AUTO FRAC gives a zero-width box (seed 0 or FRAC 0); " +
                                     ((rule.group_floor > 0.0 && !(group_max > 0.0))
                                        ? "every seed on this line is 0, so GROUP_FLOOR has "
                                          "nothing to scale; set FLOOR"
                                        : "set FLOOR or GROUP_FLOOR"));
          lo_i = x0 - d; hi_i = x0 + d;
          omap.origin[i] = BoxOrigin::AutoLinear;
        }
        if (tf.kind == ScaleKind::Log && lo_i <= 0.0)
          throw std::runtime_error("OPTIM " + k + " (" + core::to_string(ui) +
                                   "): AUTO box [" + core::to_string(lo_i) + ", " +
                                   core::to_string(hi_i) + "] is not positive but the dim is "
                                   "tagged LOG; use RATIO (geometric) or drop the tag");
      }
      omap.opt[i] = true;
      omap.highbound[i] = hi_i;
      omap.lowbound[i] = lo_i;
      omap.scale[i] = tf;
    } catch (const std::exception& e) {
      throw std::runtime_error(
        "Error processing user index on process_optim_config line "
        + core::to_string(nline) + ": " + std::string(e.what()) +
        "\nTokens: " + core::to_string(ui));
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

