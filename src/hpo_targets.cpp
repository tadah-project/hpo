#include <tadah/hpo/hpo_targets.h>
#include <tadah/models/descriptors/dm/embed_spec.h>
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
#include <tadah/core/blueprint_parser.h>
#include <tadah/core/blueprint_validator.h>
#include <tadah/core/registry.h>
#include <tadah/core/retired_keys.h>
#include <tadah/core/number_format.h>
#include <tadah/hpo/loss_function.h>
#include "hpotarget_parser.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <set>
#include <optional>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <system_error>
#include <unordered_map>
#include <unistd.h>   // access(): the archive directories
#include <fcntl.h>    // open(): the output files
#include <sys/stat.h> // stat(): the output files
#include <cerrno>
#include <cstring>

namespace tadah {
namespace hpo {

namespace fs = std::filesystem;

namespace {

/* A seed the run draws once, where a block that draws random numbers has no
 * SEED: 64 bits from std::random_device, two draws (as RestartPolicy draws
 * the seed of its rounds). */
std::uint64_t draw_run_seed()
{
  std::random_device rd;
  const std::uint64_t hi = rd(), lo = rd();
  return (hi << 32) ^ lo;
}

/* Whether the WARM seed file `warm` is the file `best` that a run writes
 * its best potential to: the path `warm` leads to (symbolic links
 * followed) is the directory entry `best` names in its directory, which
 * the run removes and replaces (a link there is replaced, not the file it
 * leads to). Paths are compared as the system resolves them, so
 * best_pot.tadah, ./best_pot.tadah, its absolute path and a link to it are
 * one. */
bool names_best_pot_file(const std::string& warm, const std::string& best)
{
  std::error_code ec;
  const fs::path w = fs::weakly_canonical(fs::path(warm), ec);
  if (ec) return false;
  const fs::path b(best);
  const fs::path dir = b.has_parent_path() ? b.parent_path() : fs::path(".");
  const fs::path d = fs::weakly_canonical(dir, ec);
  if (ec) return false;
  return w == d / b.filename();
}

/* Whether the WARM path `warm` names the directory entry `best` itself,
 * a symbolic link there not followed: a RESTART round opens the path
 * again, after the run has replaced that entry, link and all, with its
 * best potential. */
bool names_best_pot_entry(const std::string& warm, const std::string& best)
{
  const auto entry = [](const fs::path& p, std::error_code& ec) {
    const fs::path dir =
        p.has_parent_path() ? p.parent_path() : fs::path(".");
    return fs::weakly_canonical(dir, ec) / p.filename();
  };
  std::error_code ec;
  const fs::path w = entry(fs::path(warm), ec);
  if (ec) return false;
  const fs::path b = entry(fs::path(best), ec);
  if (ec) return false;
  return w == b;
}

}  // namespace

namespace {
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
    if (ispec.chain.empty()) return "METHOD CONFIG, by default";
    std::string m;
    for (const auto& e : ispec.chain) {
        if (!m.empty()) m += ", ";
        m += e.method;
    }
    return m;
}
}

void ParsedTarget::Opti::resize(size_t s) {
  if (opt.size() >= s) return;
  opt.resize(s, false);
  line.resize(s, 0);
  line_text.resize(s);
  config.resize(s);
  scale.resize(s);
  tagged.resize(s, false);
  origin.resize(s, BoxOrigin::Explicit);
  lowbound.resize(s, 0);
  highbound.resize(s, 0);
}

ParsedTarget::ParsedTarget(core::Context& c) : context(c) {}

void ParsedTarget::prepare_optim_reader() {
  optim_seed_ = mlip::prepare_optim_seed(context, "tadah hpo");
  optim_reader_.emplace(context, optim_seed_);
}

HPOTargets::HPOTargets(core::Context &c, std::string fn,
                       const mlip::KeyProvenance *provenance,
                       TargetsPurpose purpose)
    : ParsedTarget(c) {
  const bool check = purpose == TargetsPurpose::Check;
  models::resolve_embed_blocks(context);
  /* The training record of the potential files, taken before anything
   * below can insert a default into the context. */
  training_record = mlip::training_record(context, provenance);
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
          "potential files of tadah hpo (best_pot.tadah, BDUMP and DUMP "
          "archives): " + w;
      std::cerr << "[tadah hpo] WARNING: " << msg << "\n";
      HPOLog::instance().warning(msg);
    }
  }

  /* The radial grids expanded (a -c potential's values its descriptors do
   * not read left out) and the density form: RHOFORM (the DM_REAM density
   * form) is mandatory and has no default, read before the OPTIM lines
   * (non-negative AMPGRIDMB boxes) and the WARM seeds (form check) that
   * depend on it. MLIP's OPTIM reader reads the OPTIM lines, as tadah
   * analysis check_bounds reads them. */
  prepare_optim_reader();

  /* The target file, read by the registry-driven parser
   * (src/hpotarget_parser.cpp). LS keeps configuration order: the
   * outvar/loss columns, the per-evaluation rows and the potential-file
   * header are emitted in LS order, and failprobe_idx is recorded against
   * it at parse time. */
  hpotarget::parse_target_file(*this, fn);
  /* ---------- search space ----------------------------------------
   * One SearchSpace from the OPTIM lines (real boxes + per-line shape
   * tags), OPTIM_SCALE and LOG_HP; bound to the outer spec and, below,
   * to every sub-spec. spec.low/high are the INTERNAL bounds. */
  /* Env override TADAH_OPTIM_SCALE=AUTO|NONE (an A/B of the coordinate
   * system without editing the target file). Read here because the space
   * is built in this constructor. An empty value is ignored, as an empty
   * value of the other overrides is (the shell's way of clearing it). */
  if (const char* e = std::getenv("TADAH_OPTIM_SCALE"); e && *e) {
    std::string v = e;
    for (auto& ch : v) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
    if      (v == "AUTO") scale_policy_.normalise = true;
    else if (v == "NONE") scale_policy_.normalise = false;
    else throw std::runtime_error("TADAH_OPTIM_SCALE must be AUTO or NONE (got '" +
                                  std::string(e) + "')");
    std::ostringstream oss;
    oss << "OPTIM_SCALE " << v << " taken from the TADAH_OPTIM_SCALE environment "
           "override" << (scale_policy_set_ ? " (the OPTIM_SCALE line of the "
                                              "target file is not read)" : "");
    std::cerr << "[tadah hpo] " << oss.str() << "\n";
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
  spec.failscore_grading = FAILSCORE_GRADING;

  /* ---------- NOISE_AUTO: heuristic for finite-difference fd_step ---
   *
   * The NOISE block's AUTO derives PARAM fd_step from the search-space
   * geometry:
   *
   *     fd_step = max(1e-3 * box_diagonal / sqrt(dim), 1e-6)
   *
   * where box_diagonal is the L2 norm of (high - low) in the
   * post-transform space. The hand-set PARAM fd_step always wins so
   * users keep an escape hatch. It is derived where the STRATEGY-level
   * phase -- which carries the NOISE block: each MULTISTART start, the
   * PRESTAGE refinement -- takes finite-difference gradients
   * (takes_fd_gradients: a gradient-based backend, or an MLSL/AUGLAG
   * whose local search is one, under GRAD_STRATEGY NUMERICAL or HYBRID;
   * the parse has resolved an unset GRAD_STRATEGY by now). Where it takes
   * none -- a derivative-free optimiser, GRAD_STRATEGY ANALYTICAL --
   * validate_hpo_setup refuses the AUTO, which would have no effect (it
   * printed "ignored" to stderr), and so it does beside a PARAM fd_step
   * written in the block (the written step won, with a note on stderr).
   * The parse rejects a
   * configuration without LIB/ALGO. */
  if (spec.noise.autodetect &&
      takes_fd_gradients(PhaseSpec{"the STRATEGY-level optimiser", spec})) {
    if (spec.param.count("fd_step") || spec.param_raw.count("fd_step")) {
      /* refused by validate_hpo_setup */
    } else if (spec.low.empty()) {
      std::cerr << "[tadah hpo] NOISE AUTO: the search space has no "
                   "dimension, so no PARAM fd_step is set\n";
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
      /* Provenance: a box too narrow for this step is refused at startup
       * naming NOISE AUTO (fd_box_error), and the journal names it. */
      spec.fd_step_from_noise_auto = true;
      std::ostringstream oss;
      oss << "NOISE AUTO: PARAM fd_step is " << fd << " (1e-3 x the "
          << "diagonal " << diag << " of the search space / sqrt("
          << spec.low.size() << " dimensions))";
      std::cout << "[tadah hpo] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
  }

  /* Propagate the search space into every sub-spec FIRST, before
   * resolving any INIT_STRATEGY chain on them: strategies like RANDOM/
   * LHS/SOBOL sample within bounds, so an unbound sub-spec produces
   * 0-dim points. (Bound inheritance must precede init resolution.) */
  bind_space_recursive();

  /* The seed of a PRESTAGE exploration that sets none. A run's random
   * numbers come from the SEEDs written in its target file: the
   * refinement's starts (SEED + i) and every RESTART round (SEED + r*S,
   * the exploration's SEED + r) derive theirs from them. An exploration
   * whose optimiser reads SEED and that has no seed of its own -- no
   * SEED and, for NLopt, no INIT block SEED (nlopt_seed_from_init) --
   * takes the STRATEGY-level SEED, so a run with SEED in its OPTIMIZER
   * or STRATEGY block repeats its exploration as it repeats its
   * refinement. The refinement draws streams of its own key
   * (phase_stream_seed, seed_stream.h), so the two phases do not share
   * random numbers. An exploration whose run reads no SEED (DLIB:GFS, an
   * NLopt algorithm that draws no random numbers: reads_seed,
   * phase_specs.h, which needs the bound search space for
   * GN_DIRECT_L_RAND) takes none. The journal names the exploration's
   * seed and where it comes from (seed_from_strategy). */
  if (spec.prestage && spec.seed && !spec.prestage->seed &&
      !spec.prestage->nlopt_seed_from_init() && reads_seed(*spec.prestage)) {
    spec.prestage->seed = spec.seed;
    spec.prestage->seed_from_strategy = true;
  }

  /* ---------- resolve INIT_STRATEGY chain into optim_init -----------
   *
   * Apply to OUTER and to the PRESTAGE exploration when its EXPLORE block
   * has an INIT block. An INNER block takes no INIT block (the parse
   * refuses one): NLopt's MLSL and AUGLAG choose the starting points of the INNER
   * search, so an INNER spec has no chain to resolve. Each spec's INIT
   * block (METHOD / SEED / K_TOTAL / WARM_ON_MISMATCH lines) is honoured
   * independently of every other spec.
   *
   * Diagnostics tag the location ("OUTER" / "PRESTAGE") so the user can
   * see which block fired which strategy. */
  auto resolve_init = [&](OptimizerSpec& s, const std::string& phase) {
    auto strategy = build_init_strategy(s.init_strategy_spec);
    /* The INIT block as the lines below name it: by its line, or, without
     * an INIT block, as the INIT chain of its phase (OUTER: the
     * STRATEGY-level optimiser, PRESTAGE: the EXPLORE block). */
    const std::string init_name =
        s.init_strategy_spec.line > 0
            ? "the INIT block on line " +
                  std::to_string(s.init_strategy_spec.line)
            : "the INIT chain of " + phase;
    /* A SEED with no seed-consuming METHOD in the chain is a silent
     * no-op for the starting point: an empty chain falls back to
     * CONFIG, and CONFIG/WARM ignore ctx.seed. For an NLopt phase without
     * a SEED of its own the seed still reaches nlopt::srand in make_nlopt,
     * but only stochastic non-LDS NLopt algos (G_MLSL, CRS, ISRES, ESCH,
     * ...) ever read that RNG. Surface it so "I set SEED and nothing
     * changed" is answerable from the log. */
    /* SEED inside the INIT block seeds the init-strategy chain -- and
     * NLopt only as a fallback, when the phase has no SEED of its own
     * (make_nlopt); it does not seed any other optimiser. An optimiser
     * that draws random numbers of its own (TADAH:RANDOM, the annealers,
     * ASA) then draws them with a seed drawn once for the run
     * (draw_optimiser_seed, below), which a user who set the INIT block's
     * SEED may not expect: the WARNING names the block whose SEED the
     * optimiser takes. */
    if (!s.seed && s.init_strategy_spec.seed && s.lib != "NLOPT" &&
        reads_seed(s)) {
      std::ostringstream oss;
      oss << "the SEED of the INIT block";
      if (s.init_strategy_spec.line > 0)
        oss << " on line " << s.init_strategy_spec.line;
      oss << " seeds the starting points of the block only. " << s.lib << ":"
          << s.algo << " draws its own random numbers from the SEED of the "
          << s.key_lines.at("ALGO").block << " block; that block has none, "
             "so a seed is drawn once for the run";
      std::cerr << "[tadah hpo] " << oss.str() << "\n";
      HPOLog::instance().warning(oss.str());
    }
    if (s.init_strategy_spec.seed) {
      if (!chain_draws_from_seed(s.init_strategy_spec)) {
        std::ostringstream oss;
        oss << "the SEED of " << init_name << " changes no starting point: "
               "its methods (" << chain_methods(s.init_strategy_spec)
            << ") draw no random numbers; a RANDOM, LHS or SOBOL method "
               "draws with it.";
        if (s.lib == "NLOPT" && !s.seed && reads_seed(s))
          oss << " (" << s.lib << ":" << s.algo << ", without a SEED of its "
                 "own, still takes its random numbers from it.)";
        std::cerr << "[tadah hpo] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }
    /* A chain that draws its points without a SEED draws them with a seed
     * the run draws once, kept as the block's SEED (and so, for NLopt
     * without a SEED of its own, nlopt_seed_from_init, NLopt's) and named
     * with the SEED line that draws the same points again; a check says
     * that a run draws one, without the number (TargetsPurpose). A chain
     * of CONFIG and WARM points draws nothing and takes none. */
    std::uint64_t seed = s.init_strategy_spec.seed.value_or(0);
    if (!s.init_strategy_spec.seed &&
        chain_draws_from_seed(s.init_strategy_spec)) {
      seed = draw_run_seed();
      s.init_strategy_spec.seed = seed;
      s.init_strategy_spec.seed_drawn = true;
      const std::string n = std::to_string(seed);
      const bool nlopt_takes_it = s.lib == "NLOPT" && !s.seed && reads_seed(s);
      const std::string nlopt_note =
          nlopt_takes_it ? s.lib + ":" + s.algo + ", without a SEED of its "
                           "own, takes its random numbers from the same seed"
                         : std::string();
      std::ostringstream oss;
      oss << "the INIT block";
      if (s.init_strategy_spec.line > 0)
        oss << " on line " << s.init_strategy_spec.line;
      if (check) {
        oss << " has no SEED, so a run draws a seed for its starting points "
               "once and names it";
        if (nlopt_takes_it) oss << " (" << nlopt_note << ")";
      } else {
        oss << " has no SEED, so its starting points are drawn with a seed "
               "drawn once for the run, " << n << " (SEED " << n
            << " in the INIT block draws the same points again";
        if (nlopt_takes_it) oss << "; " << nlopt_note;
        oss << ")";
      }
      std::cout << "[tadah hpo] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
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
      oss << init_name << " (" << chain_methods(s.init_strategy_spec)
          << ") gave no starting point (a WARM point that WARM_ON_MISMATCH "
             "dropped): TADAH:ANNEAL, ANNEAL_PT and ANNEAL_PA start from the "
             "centre of the OPTIM box, not from the WARM point; "
             "WARM_ON_MISMATCH error in the INIT block makes this an error";
      std::cerr << "[tadah hpo] " << oss.str() << "\n";
      HPOLog::instance().warning(oss.str());
    }
    if (!s.optim_init.empty()) {
      std::ostringstream oss;
      oss << init_name << " (" << chain_methods(s.init_strategy_spec)
          << ") gives " << s.optim_init.size() << " starting point"
          << (s.optim_init.size()==1 ? "" : "s");
      std::cout << "[tadah hpo] " << oss.str() << "\n";
      HPOLog::instance().info(oss.str());
    }
    return strategy;
  };

  resolve_init(spec, "the STRATEGY-level optimiser");

  /* An optimiser that draws random numbers (reads_seed: TADAH:RANDOM,
   * the annealers, ASA, an NLopt algorithm that draws them) in a block
   * without SEED -- for NLopt, and without the SEED of its INIT block,
   * which it would take (nlopt_seed_from_init) -- draws them with a seed
   * the run draws once, kept as the block's SEED -- a seed of its own
   * drawn when it starts (std::random_device for the annealers, ASA and
   * TADAH:RANDOM, NLopt's own clock seed) would leave the run
   * unrepeatable -- and named with the SEED line that draws the same
   * numbers again. MULTISTART's starts, the PRESTAGE refinement and the
   * RESTART rounds derive their seeds from it as from a SEED line
   * (OptimizerSpec::derive_restart_seeds). After the INIT chains, whose
   * drawn seed NLopt takes. The line names the block that holds the
   * optimiser's ALGO line, where its SEED goes: the STRATEGY-level
   * optimiser's keys may be written in OPTIMIZER, STRATEGY or REFINE, and
   * a file may have no STRATEGY block; a parsed spec has an ALGO line (the
   * parser refuses a block without one). A check says that a run draws a
   * seed, without the number (TargetsPurpose). */
  const auto draw_optimiser_seed = [check](OptimizerSpec& s) {
    if (s.seed || s.nlopt_seed_from_init() || !reads_seed(s))
      return;
    const std::string& block = s.key_lines.at("ALGO").block;
    const std::uint64_t seed = draw_run_seed();
    s.seed = seed;
    s.seed_drawn = true;
    const std::string n = std::to_string(seed);
    const std::string head = std::string("the ") + block +
                             " block has no SEED, and " + s.lib + ":" +
                             s.algo + " draws random numbers, so ";
    const std::string msg =
        check ? head + "a run draws a seed for them once and names it"
              : head + "it draws them with a seed drawn once for the run, " +
                    n + " (SEED " + n + " in the " + block +
                    " block draws the same numbers again)";
    std::cout << "[tadah hpo] " << msg << "\n";
    HPOLog::instance().info(msg);
  };

  /* INIT points a phase does not evaluate -- more than a PLAIN search, a
   * PRESTAGE exploration or the starts of a MULTISTART take -- are warned
   * about at startup (start_point_warnings, backend_rules.h), once every
   * chain is resolved. */

  if (spec.prestage) {
    /* If PRESTAGE has no explicit chain and no INIT_STRATEGY entries,
     * it inherits the OUTER's optim_init (legacy behaviour: a CONFIG
     * point flows in). If PRESTAGE has its own chain, that chain wins
     * and the OUTER's optim_init is *not* prepended -- PRESTAGE owns its
     * starting points. */
    if (!spec.prestage->init_strategy_spec.chain.empty())
      resolve_init(*spec.prestage, "the PRESTAGE exploration (EXPLORE block)");
    spec.share_init_with_nested_phases();
  }
  draw_optimiser_seed(spec);
  if (spec.prestage) draw_optimiser_seed(*spec.prestage);

  /* The RESTART block's SEED seeds the METHOD chain that draws each
   * restart's points (round r with SEED + r, RestartPolicy). Like an INIT
   * block's SEED, it does nothing without a RANDOM/LHS/SOBOL method; say so,
   * when the RESTART block is active at all. */
  if (spec.restart.active() && spec.restart.init_strategy.seed &&
      !chain_draws_from_seed(spec.restart.init_strategy)) {
    std::ostringstream oss;
    oss << "RESTART: the SEED of the RESTART block changes no restart "
           "point: its methods ("
        << chain_methods(spec.restart.init_strategy)
        << ") draw no random numbers; a RANDOM, LHS or SOBOL method draws "
           "with it.";
    std::cerr << "[tadah hpo] " << oss.str() << "\n";
    HPOLog::instance().warning(oss.str());
  }

  /* The RESTART chain's seeds (CONFIG, WARM), resolved once here as a
   * round draws them (RestartPolicy: the outer spec, the run context, the
   * block's K_TOTAL and WARM_ON_MISMATCH), so a WARM file that is missing
   * or unreadable, or whose RHOFORM the run refuses, or a WARM seed the
   * block's `WARM_ON_MISMATCH error` refuses, is a startup error and not
   * the first stagnation's, the EMBF0 note comes at startup, and the
   * density gates check these points as seeds (collect_gate_seeds). The
   * sampled points of the draw are dropped: each round draws its own. */
  if (spec.restart.active()) {
    const InitStrategySpec& rs = spec.restart.init_strategy;
    auto refresh = build_init_strategy(rs);
    InitContext ictx{spec, *this, rs.seed.value_or(0), &context,
                     InitStream::Restart};
    std::vector<InitOrigin> origins;
    const auto pts = refresh->generate_tagged(
        ictx, rs.k_total.value_or(refresh->default_k()), origins);
    for (std::size_t i = 0; i < pts.size() && i < origins.size(); ++i) {
      if (!origins[i].is_seed()) continue;
      spec.restart.seed_points.push_back(pts[i]);
      spec.restart.seed_origins.push_back(origins[i]);
    }
  }

  /* A WARM seed that is the file the run writes its best potential to:
   * the run removes it before its first evaluation
   * (HPO_Host::prepare_best_publication) and writes its best potential
   * there, so the seed it held does not outlive the start of the run; a
   * RESTART round, which reads its WARM file again, reads the run's best
   * potential so far. A WARNING each, on stderr and in the journal (and
   * so in the summary of tadah hpo --check). */
  warn_warm_of_best_pot();

  /* Fail fast on the initial-step policy (registry-driven). Seeds vs
   * box and the width audit run in validate_hpo_setup(), with the other
   * startup rules. */
  run_scale_validation(spec, /*check_start_points=*/false);
}

std::string HPOTargets::repeat_reuse_blocker() const {
  if (!LS.empty()) {
    std::string why =
        "PC_LAMMPS: its scripts run on LAMMPS instances kept from one "
        "evaluation to the next, which keep what a script set there (LAMMPS "
        "keeps variables across clear)";
    for (const auto& sim : LS)
      if (!sim.hotvars.empty()) {
        why += ", and each --hotvar starts from the last value inside its "
               "range that an earlier evaluation of its chain read back";
        break;
      }
    return why;
  }
  for (const auto& obs : observables) {
    const std::string carried = obs->carried_state();
    if (!carried.empty()) return "PC_OBSERVABLE " + obs->kind() + ": " + carried;
  }
  const std::string rel = " --skip-rel: the filter compares with the best "
                          "value so far, which a later evaluation can lower";
  if (ermse_targets.has_relative_cap()) return "PC_ERMSE" + rel;
  if (frmse_targets.has_relative_cap()) return "PC_FRMSE" + rel;
  if (srmse_targets.has_relative_cap()) return "PC_SRMSE_GPA" + rel;
  if (use_emae && skip_emae.rel_above) return "PC_EMAE" + rel;
  if (use_errmse && skip_errmse.rel_above) return "PC_ErRMSE" + rel;
  return "";
}

double ParsedTarget::parse_pc_worker_timeout(const std::string& token,
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

namespace {
/* Whether the process can create entries in the directory @p dir: write
 * and search permission (access(2), the real user's). */
bool writable_directory(const fs::path& dir) {
  return ::access(dir.c_str(), W_OK | X_OK) == 0;
}

/* A path as the operating system finds it when the run opens or creates
 * it (resolve_physically). */
struct PhysicalPath {
  fs::path path;               // absolute, without a link, '.' or '..'
  bool exists = false;         // something exists at path
  bool through_links = false;  // a symbolic link was followed to reach it
  bool names_directory = false;  // it ends in '/', '/.' or '/..' (as given,
                                 // or as a link target spliced in at its
                                 // end): only a directory has such a name
  fs::path own_link;           // the target the path's own link stores, its
                               // trailing '/' and '.' set aside (empty: the
                               // path names no link)
};

/* @p given as the operating system resolves it to open or create it: from
 * the root, one component after the other, every symbolic link followed --
 * the one the path ends with too, to a file not yet created -- and '.' and
 * '..' taken in the directory reached so far. Throws, with the reason
 * alone, where the system cannot resolve it: more than 40 links, the
 * system's limit ("too many levels of symbolic links"), '.' or '..' in a
 * directory that does not exist or is no directory, and a component it
 * cannot read ("'<path>' cannot be read: Permission denied").
 * @p will_exist (absolute and resolved; empty: none) is a directory that
 * does not exist yet and that the run creates first, the other archive
 * line's: it is taken as an existing directory ('.' and '..' in it, as the
 * system takes them once it exists), and what lies below it as missing.
 * (fs::weakly_canonical takes the part of a path that does not exist
 * lexically, and stops at a link to a file not yet created.) */
PhysicalPath resolve_physically(const std::string& given,
                                const fs::path& will_exist = fs::path()) {
  std::error_code ec;
  const fs::path abs = fs::absolute(fs::path(given), ec);
  if (ec) throw std::runtime_error(ec.message());
  PhysicalPath out;
  /* The path's own link: its last named component, its trailing '/' and
   * '.' set aside ("lnk/" names the link lnk, as mkdir takes it). */
  {
    fs::path last = abs;
    while (last.has_parent_path() && last != last.root_path() &&
           (last.filename().empty() || last.filename() == "."))
      last = last.parent_path();
    std::error_code lec;
    if (fs::is_symlink(fs::symlink_status(last, lec)))
      out.own_link = fs::read_symlink(last, lec);
  }
  std::deque<fs::path> todo;
  for (const fs::path& c : abs.relative_path()) todo.push_back(c);
  fs::path cur = abs.root_path();
  /* The first component on the way that does not exist, or is no
   * directory though more follow (empty: none); below it the path is
   * taken as written. */
  fs::path stop;
  bool stop_exists = false;
  int links = 0;
  /* Whether the components taken last were '', '.' or '..': the path then
   * names a directory. */
  bool trailing = false;
  const auto missing = [&] {
    return std::string(stop_exists && cur == stop ? "is not a directory"
                                                  : "does not exist");
  };
  while (!todo.empty()) {
    const fs::path c = todo.front();
    todo.pop_front();
    if (c.empty()) {
      trailing = true;
      continue;
    }
    if (c == ".") {
      if (!stop.empty())
        throw std::runtime_error("it walks '.' in '" + cur.string() +
                                 "', which " + missing());
      trailing = true;
      continue;
    }
    if (c == "..") {
      if (!stop.empty())
        throw std::runtime_error("it walks '..' out of '" + cur.string() +
                                 "', which " + missing());
      cur = cur.parent_path();
      trailing = true;
      continue;
    }
    trailing = false;
    const fs::path next = cur / c;
    if (!stop.empty()) {
      cur = next;
      continue;
    }
    const fs::file_status ls = fs::symlink_status(next, ec);
    if (ec && ls.type() != fs::file_type::not_found)
      throw std::runtime_error("'" + next.string() +
                               "' cannot be read: " + ec.message());
    if (fs::is_symlink(ls)) {
      if (++links > 40)
        throw std::runtime_error("too many levels of symbolic links");
      const fs::path target = fs::read_symlink(next, ec);
      if (ec)
        throw std::runtime_error("'" + next.string() +
                                 "' cannot be read: " + ec.message());
      out.through_links = true;
      if (target.is_absolute()) cur = target.root_path();
      std::deque<fs::path> parts;
      for (const fs::path& t : target.relative_path()) parts.push_back(t);
      todo.insert(todo.begin(), parts.begin(), parts.end());
      continue;
    }
    if (ls.type() == fs::file_type::not_found) {
      if (!will_exist.empty() && next == will_exist) {
        cur = next;  // there once the run has created it
        continue;
      }
      stop = next;
      stop_exists = false;
    } else if (!fs::is_directory(ls) &&
               std::any_of(todo.begin(), todo.end(), [](const fs::path& t) {
                 return !t.empty() && t != ".";
               })) {
      stop = next;
      stop_exists = true;
    }
    cur = next;
  }
  out.path = cur;
  out.exists = stop.empty();
  out.names_directory = trailing;
  return out;
}

/* @p dir resolved (resolve_physically, with @p will_exist), or nullopt
 * when it cannot be. */
std::optional<PhysicalPath> resolved_or_none(
    const std::string& dir, const fs::path& will_exist = fs::path()) {
  try {
    return resolve_physically(dir, will_exist);
  } catch (const std::exception&) {
    return std::nullopt;
  }
}

/* The directory of the archive line @p dir as the run finds it before it
 * creates any: its resolved path when the run creates it (it does not
 * exist, and the path names no link), else empty. */
fs::path created_by_the_run(const std::string& dir) {
  const auto r = resolved_or_none(dir);
  return r && !r->exists && r->own_link.empty() ? r->path : fs::path();
}
}  // namespace

void HPOTargets::create_archive_directories() const {
  const bool dump = dumpPotEvery != std::numeric_limits<size_t>::max();
  const bool bdump = dumpBestPotEvery != std::numeric_limits<size_t>::max();
  /* The DUMP directory first, unless the system puts it inside the BDUMP
   * directory. */
  bool bdump_first = false;
  if (dump && bdump) {
    const fs::path b = created_by_the_run(dumpBestPotDir);
    const auto d = resolved_or_none(dumpPotDir, b);
    /* The DUMP directory is the BDUMP one ("archive/.") or inside it. */
    bdump_first =
        !b.empty() && d && (d->path == b || d->path.parent_path() == b);
  }
  if (bdump_first) create_archive_directory("BDUMP", dumpBestPotDir);
  if (dump) create_archive_directory("DUMP", dumpPotDir);
  if (bdump && !bdump_first) create_archive_directory("BDUMP", dumpBestPotDir);
}

void ParsedTarget::check_archive_directories() const {
  struct Archive {
    const char* key;
    const std::string& dir;
    bool active;
    const ArchiveLine& line;
  };
  const Archive a[2] = {
      {"DUMP", dumpPotDir,
       dumpPotEvery != std::numeric_limits<size_t>::max(), dumpPotLine},
      {"BDUMP", dumpBestPotDir,
       dumpBestPotEvery != std::numeric_limits<size_t>::max(),
       dumpBestPotLine}};
  /* In the order of their lines, so that the first line in error is the
   * one named. */
  const int first = a[1].line.number < a[0].line.number ? 1 : 0;
  for (int k : {first, 1 - first}) {
    const Archive& self = a[k];
    const Archive& other = a[1 - k];
    if (!self.active) continue;
    /* The other line's directory, where the system puts it, when the run
     * creates it: one inside it is created after it. Whether the run can
     * create it is checked at its own line. */
    const fs::path created_first =
        other.active ? created_by_the_run(other.dir) : fs::path();
    try {
      check_archive_directory(self.key, self.dir, created_first);
    } catch (const std::exception& e) {
      throw std::runtime_error(
          config_line_error(target_file, self.line.number, e.what(),
                            self.line.text));
    }
  }
}

void ParsedTarget::check_archive_directory(const std::string& key,
                                         const std::string& dir,
                                         const fs::path& created_first) {
  const std::string what = key + " directory '" + dir + "'";
  PhysicalPath r;
  try {
    r = resolve_physically(dir, created_first);
  } catch (const std::exception& e) {
    throw std::runtime_error(what + " cannot be resolved: " + e.what());
  }
  /* The other line's directory itself ("archive/."), which the run
   * creates first. */
  if (!created_first.empty() && r.path == created_first) return;
  std::error_code ec;
  if (r.exists) {
    const fs::file_status st = fs::status(r.path, ec);
    if (ec)
      throw std::runtime_error(what + " cannot be read: " + ec.message());
    if (!fs::is_directory(st))
      throw std::runtime_error(what + " exists and is not a directory");
    if (!writable_directory(r.path))
      throw std::runtime_error(what + " is not writable");
    return;
  }
  /* mkdir does not follow a link the path itself names: a link to a
   * directory that does not exist stops it. */
  if (!r.own_link.empty())
    throw std::runtime_error(what + " is a symbolic link to '" +
                             r.own_link.string() + "', which does not exist");
  /* A missing directory: the run creates the last directory of the path,
   * in the directory the system puts it in -- after the other archive
   * line's directory, when that is it. */
  const fs::path parent = r.path.parent_path();
  if (!created_first.empty() && parent == created_first) return;
  const std::string cannot = what + " cannot be created: ";
  const fs::file_status ps = fs::status(parent, ec);
  if (ps.type() == fs::file_type::not_found)
    throw std::runtime_error(cannot + "the directory '" + parent.string() +
                             "' does not exist (tadah hpo creates only the "
                             "last directory of the path)");
  if (ec)
    throw std::runtime_error(cannot + "'" + parent.string() +
                             "' cannot be read: " + ec.message());
  if (!fs::is_directory(ps))
    throw std::runtime_error(cannot + "'" + parent.string() +
                             "' is not a directory");
  if (!writable_directory(parent))
    throw std::runtime_error(cannot + "the directory '" + parent.string() +
                             "' is not writable");
}

namespace {
/* The working directory, for the messages on the run's output files. */
std::string working_directory() {
  std::error_code ec;
  const fs::path cwd = fs::current_path(ec);
  return ec ? std::string(".") : cwd.string();
}

/* What the run does with an output file in the working directory. */
enum class OutputUse {
  Rewritten,  // opens it to write, once (the journal, a search-space file)
  Reopened,   // opens it to write, again for each row (a result file)
  Locked,     // opens it to write and locks it (the lock of a best potential)
  Replaced    // replaces or removes the name (a best potential)
};

/* Whether the run can open the existing file @p file as it does -- to
 * write, and to read too (@p read_too: a lock) --: 0, or the errno of the
 * failure. Links are followed as the open follows them. A directory is
 * refused (EISDIR), and so is a socket, which the run's open cannot open
 * (ENXIO). A named pipe is not opened -- opening and closing its write end
 * would end the stream of its reader --, and its permission is checked
 * instead (faccessat with the effective ids): the run's own open of it
 * waits for its reader. A regular file, and a character or block device,
 * are opened as the run opens them, without creating or truncating them,
 * without waiting and without taking a controlling terminal, and closed:
 * nothing is written. */
int open_as_the_run(const std::string& file, bool read_too) {
  struct stat st;
  if (::stat(file.c_str(), &st) != 0) return errno;
  if (S_ISDIR(st.st_mode)) return EISDIR;
  if (S_ISSOCK(st.st_mode)) return ENXIO;
  if (S_ISFIFO(st.st_mode)) {
    const int mode = read_too ? (R_OK | W_OK) : W_OK;
    return ::faccessat(AT_FDCWD, file.c_str(), mode, AT_EACCESS) == 0 ? 0
                                                                       : errno;
  }
  const int fd = ::open(file.c_str(),
                        (read_too ? O_RDWR : (O_WRONLY | O_APPEND)) |
                            O_NONBLOCK | O_NOCTTY | O_CLOEXEC);
  if (fd < 0) return errno;
  ::close(fd);
  return 0;
}

/* Throws when the run cannot write the output file @p file of the working
 * directory as @p use says (@p locked: the best potential a lock is of).
 * A file the run writes through (Rewritten, Reopened, Locked) that exists
 * is opened as the run opens it (open_as_the_run): the system decides,
 * following every link, and the refusal is the open's own error ("...
 * cannot be opened: Permission denied (tadah hpo rewrites it)"); a file
 * the run reopens for each row (Reopened) cannot be a named pipe. One that does not
 * exist is checked where the system would create it
 * (resolve_physically): a name that ends in '/' or '/.' cannot be a file,
 * and the file must be in a directory that exists and the run can write
 * into (the run creates the file, no directory). A file the run replaces
 * (Replaced) is checked by its own name: a symbolic link there is
 * replaced, not followed, and only a directory of that name stops the
 * run. */
void check_output_file(const std::string& file, OutputUse use,
                       const std::string& locked = std::string()) {
  const std::string where =
      "'" + file + "' in the working directory '" + working_directory() + "'";
  std::error_code ec;
  if (use == OutputUse::Replaced) {
    const fs::file_status ls = fs::symlink_status(file, ec);
    if (ec && ls.type() != fs::file_type::not_found)
      throw std::runtime_error(where + " cannot be read: " + ec.message());
    if (fs::is_directory(ls))
      throw std::runtime_error(where + " is a directory: tadah hpo writes a "
                               "file of that name");
    return;
  }
  const std::string purpose = use == OutputUse::Locked
                                  ? "tadah hpo opens it to lock " + locked
                                  : std::string("tadah hpo rewrites it");
  /* A file the run reopens for each row cannot be a pipe: the reader gets
   * the end of the file when the run closes it, and the next open waits
   * for a reader for ever. */
  if (use == OutputUse::Reopened) {
    struct stat st;
    if (::stat(file.c_str(), &st) == 0 && S_ISFIFO(st.st_mode))
      throw std::runtime_error(where + " is a named pipe: tadah hpo reopens "
                               "the file for each row it writes, and a pipe "
                               "would block it");
  }
  const int err = open_as_the_run(file, use == OutputUse::Locked);
  if (err == 0) return;
  if (err != ENOENT)
    throw std::runtime_error(where + " cannot be opened: " +
                             std::strerror(err) + " (" + purpose + ")");
  /* Nothing to open: where the system would create the file. */
  PhysicalPath r;
  try {
    r = resolve_physically(file);
  } catch (const std::exception& e) {
    throw std::runtime_error(where + " cannot be resolved: " + e.what());
  }
  const std::string subject =
      r.through_links
          ? where + " is a symbolic link to '" + r.path.string() + "', which"
          : where;
  const std::string cannot = subject + " cannot be created: ";
  if (r.names_directory)
    throw std::runtime_error(cannot +
                             "a file name cannot end in '/' or '/.'");
  if (r.exists)
    throw std::runtime_error(where + " cannot be opened: " +
                             std::strerror(err) + " (" + purpose + ")");
  const fs::path dir = r.path.parent_path();
  const fs::file_status ds = fs::status(dir, ec);
  if (ds.type() == fs::file_type::not_found)
    throw std::runtime_error(cannot + "the directory '" + dir.string() +
                             "' does not exist");
  if (ec)
    throw std::runtime_error(cannot + "'" + dir.string() +
                             "' cannot be read: " + ec.message());
  if (!fs::is_directory(ds))
    throw std::runtime_error(cannot + "'" + dir.string() +
                             "' is not a directory");
  if (!writable_directory(dir))
    throw std::runtime_error(cannot + "the directory '" + dir.string() +
                             "' is not writable");
}
}  // namespace

void HPOTargets::warn_warm_of_best_pot() const {
  struct Chain {
    const InitStrategySpec* spec;
    const char* block;
    std::size_t line;
    bool restart;
  };
  std::vector<Chain> chains = {
      {&spec.init_strategy_spec, "INIT", spec.init_strategy_spec.line, false}};
  if (spec.prestage)
    chains.push_back({&spec.prestage->init_strategy_spec, "INIT",
                      spec.prestage->init_strategy_spec.line, false});
  if (spec.restart.active())
    chains.push_back(
        {&spec.restart.init_strategy, "RESTART", spec.restart.line, true});
  const OutputNames names = output_names();
  for (const Chain& c : chains) {
    for (const auto& e : c.spec->chain) {
      if (e.method != "WARM" || e.args.empty()) continue;
      /* An INIT block reads its file once, at startup: the file the path
       * leads to then. A RESTART round opens the path again: its entry,
       * which the run replaces, counts too. */
      std::string best;
      for (const std::string* f : {&names.canonicalBestPot, &names.localBestPot})
        if (names_best_pot_file(e.args[0], *f) ||
            (c.restart && names_best_pot_entry(e.args[0], *f))) {
          best = *f;
          break;
        }
      if (best.empty()) continue;
      std::string w = std::string("the ") + c.block + " block" +
                      (c.line > 0 ? " on line " + std::to_string(c.line)
                                  : std::string()) +
                      ": METHOD WARM " + e.args[0] +
                      (e.line > 0 ? " (line " + std::to_string(e.line) + ")"
                                  : std::string()) +
                      " reads " + best + ", the file this run writes its "
                      "best potential to: the run removes it before its "
                      "first evaluation";
      w += c.restart
               ? ", so each RESTART round starts from the best potential the "
                 "run has written by then, not from the file's content at "
                 "startup."
               : " and writes its best potential there, so only a copy under "
                 "another name keeps this starting potential.";
      std::cerr << "[tadah hpo] WARNING: " << w << "\n";
      HPOLog::instance().warning(w);
    }
  }
}

HPOTargets::OutputNames HPOTargets::output_names() {
  const std::string r = hpo_rank_suffix();
  return {"loss.tadah" + r,      "outvar.tadah" + r,
          "params.tadah" + r,    "stats.tadah" + r,
          "best_loss.tadah" + r, "best_outvar.tadah" + r,
          "best_params.tadah" + r, "best_stats.tadah" + r,
          "best_pot.tadah",      "best_pot.tadah" + r};
}

bool HPOTargets::writes_search_space_file() const {
  return static_cast<bool>(space_);
}

bool HPOTargets::writes_resolved_optim_file() const {
  if (!space_) return false;
  for (std::size_t i = 0; i < space_->dim(); ++i)
    if (space_->at(i).origin != BoxOrigin::Explicit) return true;
  return false;
}

void HPOTargets::check_run_outputs(const std::string& journal) {
  std::error_code ec;
  if (!fs::is_directory(fs::status(".", ec)) || !writable_directory("."))
    throw std::runtime_error(
        "the working directory '" + working_directory() +
        "' is not writable: tadah hpo writes its journal (hpo_run.log), its "
        "result files and best_pot.tadah there");
  check_output_file(journal, OutputUse::Rewritten);
  const OutputNames n = output_names();
  for (const std::string* f :
       {&n.loss, &n.outvar, &n.params, &n.stats, &n.bestLoss, &n.bestOutvar,
        &n.bestParams, &n.bestStats})
    check_output_file(*f, OutputUse::Reopened);
  for (const std::string* f : {&n.canonicalBestPot, &n.localBestPot}) {
    check_output_file(*f, OutputUse::Replaced);
    check_output_file(*f + ".lock", OutputUse::Locked, *f);
  }
}

void HPOTargets::check_search_space_files() const {
  if (writes_search_space_file())
    check_output_file(kSearchSpaceFile, OutputUse::Rewritten);
  if (writes_resolved_optim_file())
    check_output_file(kResolvedOptimFile, OutputUse::Rewritten);
}

void HPOTargets::create_archive_directory(const std::string& key,
                                          const std::string& dir) {
  std::error_code ec;
  if (fs::create_directory(dir, ec)) {
    std::cout << "[tadah hpo] " << key << " directory '" << dir
              << "' created" << std::endl;
    return;
  }
  if (!ec && fs::is_directory(dir, ec)) {
    std::cout << "[tadah hpo] " << key << " directory '" << dir
              << "' already exists" << std::endl;
    return;
  }
  throw std::runtime_error(
      key + " directory '" + dir + "' could not be created: " +
      (ec ? ec.message() : std::string("it exists and is not a directory")));
}

mlip::TrainingRecord HPOTargets::record_of(const core::Context& pot) const {
  mlip::TrainingRecord r = training_record;
  for (std::string& line : r.lines) {
    std::istringstream ls(line);
    std::string key;
    ls >> key;
    const auto o = optimap.find(core::to_upper(key));
    if (o == optimap.end() ||
        std::none_of(o->second.opt.begin(), o->second.opt.end(),
                     [](bool b) { return b; }))
      continue;
    if (!pot.has_stored(key)) continue;
    line = key;
    for (std::size_t n = 0; const auto t = pot.stored_value(key, n); ++n)
      line += " " + *t;
  }
  return r;
}

const std::vector<std::string>& ParsedTarget::density_gate_keys() {
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
    if (kv.first == "FIXWEIGHT" && context.contains("FIXINDEX"))
      fixidx = core::parse_fixindex(context, std::nullopt);
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
      /* FIXWEIGHT's index is a model weight, which a FIXWEIGHT with one
       * value names too. */
      if (o.opt.size() > 1 || kv.first == "FIXWEIGHT")
        d.label += "[" + std::to_string(d.user_index) + "]";
      d.lo = o.lowbound[i];
      d.hi = o.highbound[i];
      d.tf = i < o.scale.size() ? o.scale[i] : DimTransform{};
      d.tagged = i < o.tagged.size() && o.tagged[i];
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
     * TSV (a reader strips the padding of each field). 10 significant digits -- the
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
    std::ofstream tsv(dir + "/" + kSearchSpaceFile);
    tsv <<
      "# tadah hpo search space -- one row per optimised dimension, in optimiser order\n"
      "# (the same order as the params.tadah columns and the journal tables).\n"
      "#   dim          1-based position in the parameter vector\n"
      "#   label        KEY[n]: n is the OPTIM index (1-based; for FIXWEIGHT the coefficient\n"
      "#                named in FIXINDEX); single-value keys other than FIXWEIGHT carry no index\n"
      "#   lo, hi       real bounds the optimiser may not leave\n"
      "#   origin       OPTIM = explicit bounds; AUTO FRAC / AUTO RATIO = derived from the seed;\n"
      "#                AUTO FRAC>=0 = AUTO FRAC with lo = max(0, x0 - d): every box of a non-negative\n"
      "#                amplitude (AMPGRIDMB under RHOFORM NONNEG / MONOTONE) and of the relative\n"
      "#                part b of FSIGMA and SSIGMA, and a box of another key whose values are >= 0\n"
      "#                (the centres) where x0 - d < 0;\n"
      "#                AUTO FRAC>0 = AUTO FRAC with lo = x0/2, a key whose values are > 0 (the widths,\n"
      "#                RCUTENV, WATOM under RHOFORM NONNEG / MONOTONE) where x0 - d is not above 0\n"
      "#   x0           starting point (real units)\n"
      "#   scale, c     shape of the internal coordinate: LIN, LOG (ln x) or ASINH (asinh(x/c));\n"
      "#                c is the ASINH linear scale, 1 for the other shapes\n"
      "#   u0           x0 in optimiser units (0..1 under OPTIM_SCALE AUTO, raw or ln under NONE)\n"
      "#   internal_lo, internal_hi   optimiser-side bounds (0 and 1 under AUTO)\n"
      "#   version      revision of the search space: 1, the boxes read from the target file;\n"
      "#                every journal reference to a version matches this one box set\n"
      "# Columns are tab-separated and space-padded to equal width; a parser strips the whitespace\n"
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
  if (!writes_resolved_optim_file()) return;
  std::ofstream hpo(dir + "/" + kResolvedOptimFile);
  hpo << "# Resolved OPTIM lines (explicit bounds for every searched dimension).\n"
         "# Written by tadah hpo because at least one box was derived by an AUTO rule;\n"
         "# in place of every OPTIM line of the target file, these lines give exactly\n"
         "# this search space.\n";
  /* Lossless numbers (shortest round-trip text): pasted back, every bound
   * must parse to the very double this run searched -- 10 digits moved AUTO
   * bounds by up to 5e-11 relative. */
  for (std::size_t i = 0; i < ss.dim(); ++i) {
    const DimSpec& d = ss.at(i);
    hpo << "OPTIM " << d.key << " (" << (d.user_index ? d.user_index : 1) << ") "
        << core::format_roundtrip(d.lo) << " " << core::format_roundtrip(d.hi);
    /* A tagged LIN line keeps its tag: untagged, LOG_HP could log-scale
     * the box pasted back. */
    if (d.tf.kind == ScaleKind::Linear && d.tagged) hpo << " LIN";
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
    sub->failscore_grading = spec.failscore_grading;
    if (sub->inner) {
      sub->inner->bind_space(space_);
      sub->inner->scale     = spec.scale;
      sub->inner->failscore = spec.failscore;
      sub->inner->failscore_grading = spec.failscore_grading;
    }
  };
  bind(spec.inner.get());
  bind(spec.prestage.get());
}

void HPOTargets::set_failscore_grading(bool on) {
  FAILSCORE_GRADING = on;
  const auto set = [on](OptimizerSpec& sp) {
    sp.failscore_grading = on;
    sp.failscore_grading_by_env = true;
  };
  set(spec);
  for (OptimizerSpec* sub : {spec.inner.get(), spec.prestage.get()}) {
    if (!sub) continue;
    set(*sub);
    if (sub->inner) set(*sub->inner);
  }
}

namespace {

/* The search coordinate of a box of an OPTIM line. */
DimTransform dim_transform_of(mlip::OptimShape shape, double c) {
  DimTransform tf;
  switch (shape) {
    case mlip::OptimShape::Linear: tf.kind = ScaleKind::Linear; break;
    case mlip::OptimShape::Log:    tf.kind = ScaleKind::Log;    break;
    case mlip::OptimShape::Asinh:  tf.kind = ScaleKind::Asinh; tf.c = c; break;
  }
  return tf;
}

BoxOrigin box_origin_of(mlip::OptimBoxOrigin o) {
  switch (o) {
    case mlip::OptimBoxOrigin::Explicit:         return BoxOrigin::Explicit;
    case mlip::OptimBoxOrigin::AutoLinear:       return BoxOrigin::AutoLinear;
    case mlip::OptimBoxOrigin::AutoGeometric:    return BoxOrigin::AutoGeometric;
    case mlip::OptimBoxOrigin::AutoLinearNonneg: return BoxOrigin::AutoLinearNonneg;
    case mlip::OptimBoxOrigin::AutoLinearPositive: return BoxOrigin::AutoLinearPositive;
  }
  return BoxOrigin::Explicit;
}

}  // namespace

void ParsedTarget::process_optim_config(std::istringstream& iss,
                                      std::size_t nline,
                                      const std::string& line_text) {
  /* The line read as tadah analysis check_bounds reads it (MLIP's OPTIM
   * reader: the refusals of the key, of the index list, of the box and of
   * a value two lines select, and the AUTO boxes); the parser names the line
   * of every error ("line N of <file>: "). The key is upper-cased, so
   * optimap, the labels and every rule and lookup keyed on it
   * (startup_validation, the gates, the neighbour-list cutoff) see one
   * spelling whatever the case the line used. */
  const mlip::OptimLine l = optim_reader_->read(iss, nline);
  Opti& omap = optimap[l.key];
  omap.resize(l.config.size());
  omap.config = l.config;
  const DimTransform tf = dim_transform_of(l.shape, l.c);
  for (const mlip::OptimBox& b : l.boxes) {
    omap.opt[b.slot] = true;
    omap.line[b.slot] = nline;
    omap.line_text[b.slot] = line_text;
    omap.lowbound[b.slot] = b.lo;
    omap.highbound[b.slot] = b.hi;
    omap.scale[b.slot] = tf;
    omap.tagged[b.slot] = l.tagged;
    omap.origin[b.slot] = box_origin_of(b.origin);
  }
}
} // namespace hpo
} // namespace tadah

