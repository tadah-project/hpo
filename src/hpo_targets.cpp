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
      throw std::runtime_error(key + ": " + kw + " is given twice on this "
                               "line; each option is given once");
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
// @p what names the line in a refusal: "OPTIM <KEY>" or the metric key.
std::string read_index_selector(std::istringstream& iss, const std::string& what) {
    iss >> std::ws;
    if (iss.get() != '(')
        throw std::runtime_error(what + ": the key is followed by its indices "
                                 "in parentheses, such as " + what + " (1)");
    std::string indices;
    if (!std::getline(iss, indices, ')') || iss.eof())
        throw std::runtime_error(what + ": the indices have no closing "
                                 "parenthesis");
    if (indices.find('(') != std::string::npos)
        throw std::runtime_error(what + ": the indices hold another opening "
                                 "parenthesis");
    if (core::trim(indices).empty())
        throw std::runtime_error(what + ": the parentheses hold no index");
    const int next = iss.peek();
    if (next != std::char_traits<char>::eof() && !std::isspace(static_cast<unsigned char>(next)))
        throw std::runtime_error(what + ": a space follows the closing "
                                 "parenthesis of the indices");
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

std::vector<std::string> HPOTargets::parse_optim_line(std::istringstream& iss,
                                                      const std::string& what) {
    const auto indices = read_index_selector(iss, what);
    std::vector<std::string> elements;
    std::string token;
    while (iss >> token) {
        if (token.find_first_of("()") != std::string::npos)
            throw std::runtime_error(what + ": the line takes one list of "
                                     "indices in parentheses (unexpected '" +
                                     token + "')");
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
  tagged.resize(s, false);
  origin.resize(s, BoxOrigin::Explicit);
  lowbound.resize(s, 0);
  highbound.resize(s, 0);
}

HPOTargets::HPOTargets(core::Context &c, std::string fn,
                       const mlip::KeyProvenance *provenance)
    : context(c) {
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
      std::cerr << "[HPO] WARNING: " << msg << "\n";
      HPOLog::instance().warning(msg);
    }
  }

  left_out_ = models::D_mJoin::expand_grids(context);

  /* RHOFORM (the DM_REAM density form) is mandatory and has no default:
   * read it before the OPTIM lines (non-negative AMPGRIDMB boxes) and the
   * WARM seeds (form check) that depend on it. Stored value only. */
  if (models::context_has_ream(context))
    rho_form_ = models::require_rhoform(context, "tadah hpo");
  else if(models::embed_specs(context).size()==1)
    rho_form_=models::embed_specs(context)[0].form;

  /* LS keeps configuration order: the outvar/loss columns, the
   * per-evaluation rows and the potential-file header are emitted in LS
   * order, and failprobe_idx is recorded against it at parse time. */
  read(fn);
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
   * read() has resolved an unset GRAD_STRATEGY by now). Where it takes
   * none -- a derivative-free optimiser, GRAD_STRATEGY ANALYTICAL --
   * validate_hpo_setup refuses the AUTO, which would have no effect (it
   * printed "ignored" to stderr), and so it does beside a PARAM fd_step
   * written in the block (the written step won, with a note on stderr).
   * read() rejects a
   * configuration without LIB/ALGO. */
  if (spec.noise.autodetect &&
      takes_fd_gradients(PhaseSpec{"the STRATEGY-level optimiser", spec})) {
    if (spec.param.count("fd_step") || spec.param_raw.count("fd_step")) {
      /* refused by validate_hpo_setup */
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
      /* Provenance: a box too narrow for this step is refused at startup
       * naming NOISE AUTO (fd_box_error), and the journal names it. */
      spec.fd_step_from_noise_auto = true;
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
   * has an INIT block. An INNER block takes no INIT block (read() refuses
   * one): NLopt's MLSL and AUGLAG choose the starting points of the INNER
   * search, so an INNER spec has no chain to resolve. Each spec's INIT
   * block (METHOD / SEED / K_TOTAL / WARM_ON_MISMATCH lines) is honoured
   * independently of every other spec.
   *
   * Diagnostics tag the location ("OUTER" / "PRESTAGE") so the user can
   * see which block fired which strategy. */
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
        if (s.lib == "NLOPT" && !s.seed && reads_seed(s))
          oss << " (Without a SEED of its own, the NLopt optimiser still "
                 "takes its random seed from it.)";
        std::cerr << "[HPO] " << oss.str() << "\n";
        HPOLog::instance().warning(oss.str());
      }
    }
    /* A chain that draws its points without a SEED draws them with a seed
     * the run draws once, kept as the block's SEED (and so, for NLopt
     * without a SEED of its own, nlopt_seed_from_init, NLopt's) and named
     * with the SEED line that draws the same points again. A chain of
     * CONFIG and WARM points draws nothing and takes none. */
    std::uint64_t seed = s.init_strategy_spec.seed.value_or(0);
    if (!s.init_strategy_spec.seed &&
        chain_draws_from_seed(s.init_strategy_spec)) {
      seed = draw_run_seed();
      s.init_strategy_spec.seed = seed;
      s.init_strategy_spec.seed_drawn = true;
      const std::string n = std::to_string(seed);
      std::ostringstream oss;
      oss << "the INIT block";
      if (s.init_strategy_spec.line > 0)
        oss << " on line " << s.init_strategy_spec.line;
      oss << " has no SEED, so its starting points are drawn with a seed "
             "drawn once for the run, " << n << " (SEED " << n
          << " in the INIT block draws the same points again";
      if (s.lib == "NLOPT" && !s.seed && reads_seed(s))
        oss << "; " << s.lib << ":" << s.algo << ", without a SEED of its "
               "own, takes its random numbers from the same seed";
      oss << ")";
      std::cout << "[HPO] " << oss.str() << "\n";
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

  /* An NLopt algorithm that draws random numbers (reads_seed) in a block
   * without SEED, and without the SEED of its INIT block, which it would
   * take (nlopt_seed_from_init), draws them with a seed the run draws once,
   * kept as the block's SEED -- NLopt's own clock seed would leave the run
   * unrepeatable -- and named with the SEED line that draws the same
   * numbers again. After the INIT chains, whose drawn seed NLopt takes.
   * The line names the block that holds the optimiser's ALGO line, where
   * its SEED goes: the STRATEGY-level optimiser's keys may be written in
   * OPTIMIZER, STRATEGY or REFINE, and a file may have no STRATEGY block;
   * a parsed spec has an ALGO line (the parser refuses a block without
   * one). */
  const auto draw_nlopt_seed = [](OptimizerSpec& s) {
    if (s.lib != "NLOPT" || s.seed || s.nlopt_seed_from_init() ||
        !reads_seed(s))
      return;
    const std::string& block = s.key_lines.at("ALGO").block;
    const std::uint64_t seed = draw_run_seed();
    s.seed = seed;
    s.seed_drawn = true;
    const std::string n = std::to_string(seed);
    const std::string msg =
        std::string("the ") + block + " block has no SEED, and " + s.lib +
        ":" + s.algo + " draws random numbers, so it draws them with a seed "
        "drawn once for the run, " + n + " (SEED " + n + " in the " + block +
        " block draws the same numbers again)";
    std::cout << "[HPO] " << msg << "\n";
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
      resolve_init(*spec.prestage, "PRESTAGE");
    spec.share_init_with_nested_phases();
  }
  draw_nlopt_seed(spec);
  if (spec.prestage) draw_nlopt_seed(*spec.prestage);

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
                               InitStrategySpec& a,
                               std::size_t nline, const std::string& line)
{
    if (key=="METHOD" || key=="STRATEGY") {
        InitStrategySpec::Entry e;
        /* The line names the entry's refusals when the chain is drawn. */
        e.line = nline;
        e.line_text = line;
        e.key = key;
        iss >> e.method;
        if (e.method.empty())
            throw std::runtime_error(key + " requires a method name");
        std::string tok;
        while (iss >> tok) e.args.push_back(tok);
        /* Arity and K are checked here, at the line, not when the chain
         * is resolved after the whole file has been read. */
        validate_init_entry(e);
        a.chain.push_back(std::move(e));
        return true;
    }
    if (key=="K_TOTAL") {
        /* A cap of the points the METHOD lines give: 1 or more, and below
         * their number, checked when the file is read (HPOTargets::read). */
        a.k_total = parse_uint(single_value(iss, key), key, 1);
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
 * attribute but three, which belong to the enclosing block:
 *   - SEED: NLopt has one random stream per run, which make_nlopt seeds
 *     from the enclosing block (its SEED, else its INIT block's SEED), and
 *     the INNER search draws from that stream. An INNER SEED could only
 *     reseed it in place of the enclosing block's SEED.
 *   - GRAD_STRATEGY: MLSL and AUGLAG hand their local search the gradient
 *     of their own objective, which the enclosing block's GRAD_STRATEGY
 *     obtains (nlopt_adapter.h). An INNER GRAD_STRATEGY would never be
 *     used.
 *   - STOPVAL: mlsl.c and auglag.c set the local search's stopval to the
 *     enclosing run's (HPO sets AUGLAG no constraint), so an INNER STOPVAL
 *     never takes effect (inner_run_spec(), phase_specs.h).
 * Returns why `key` is not accepted inside INNER, or nullptr when it is. */
static const char* inner_rejects_attr(const std::string& key)
{
    if (key == "SEED")
        return "NLopt draws the INNER search from the enclosing run's random "
               "stream";
    if (key == "GRAD_STRATEGY")
        return "the enclosing block's GRAD_STRATEGY applies to its local "
               "search";
    if (key == "STOPVAL")
        return "NLopt's MLSL and AUGLAG replace the INNER search's STOPVAL "
               "with the enclosing block's";
    return nullptr;
}

/* The keys of the top level of the target file (HPOTargets::read, the ROOT
 * section): a block refuses one naming itself as still open. MAXTIME is an
 * optimiser key too, which the blocks of an optimiser take. */
static bool is_top_level_key(const std::string& key)
{
    static const std::set<std::string> keys = {
        "FAILSCORE", "PC_SCRIPT_TIMEOUT", "MAXTIME", "PC_ISOLATION",
        "PC_WORKER_TIMEOUT", "FAILSCORE_GRADING", "BUMPGATE", "RHOGATE",
        "RHOBARGATE", "PC_FAILPROBE", "QFLUSH", "OUTPUT", "BOUTPUT",
        "DIGITS", "DUMP", "BDUMP", "LOSS", "OPTIM", "OPTIM_SCALE",
        "PC_LAMMPS", "PC_OBSERVABLE", "PC_EMAE", "PC_ERMSE", "PC_ErRMSE",
        "PC_ERSQ", "PC_SRMSE_GPA", "PC_FRMSE"};
    if (!keys.count(key)) return false;
    const auto& opt = opt_attr_keys();
    return std::find(opt.begin(), opt.end(), key) == opt.end();
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
/* Why PARALLEL (and MULTISTART_PARALLEL, its form before the STRATEGY
 * block) is refused: it asked for the starts of TYPE MULTISTART to run at
 * the same time, and they run one after another; the workers of the run
 * are THREADS. */
static constexpr const char* kParallelRemoved =
    "the starts of STRATEGY TYPE MULTISTART run one after another; THREADS "
    "sets the number of workers that run the work of the evaluations (the "
    "LAMMPS scripts and the training and prediction of the potential).";

static bool reject_old_keyword(const std::string& key)
{
    struct Rename { const char* old_; const char* hint; };
    static const Rename map[] = {
        {"LOCAL",                       "renamed to INNER, the block of the local search that an NLopt MLSL or AUGLAG algorithm runs."},
        {"ENDLOCAL",                    "renamed to ENDINNER."},
        {"PRESTAGE",                    "replaced by `STRATEGY { TYPE PRESTAGE EXPLORE { ... } [REFINE { ... }] }` (REFINE is optional; without it the STRATEGY-level LIB/ALGO refines)."},
        {"ENDPRESTAGE",                 "replaced by ENDEXPLORE / ENDREFINE / ENDSTRATEGY (see STRATEGY block)."},
        {"MULTISTART",                  "moved into `STRATEGY { TYPE MULTISTART N_STARTS N }`."},
        {"MULTISTART_PARALLEL",         kParallelRemoved},
        {"PARALLEL",                    kParallelRemoved},
        {"MULTISTART_BUDGET_PER_START", "moved into `STRATEGY { TYPE MULTISTART BUDGET_PER_START N }`."},
        {"STAGE_GLOBAL",                "STRATEGY TYPE PRESTAGE runs an exploration and then a refinement."},
        {"RESTART_AFTER_STAGNATION",    "moved into `STRATEGY { RESTART { AFTER_STAGNATION N } ENDRESTART }`."},
        {"RESTART_MAX",                 "moved into `STRATEGY { RESTART { MAX N } ENDRESTART }`."},
        {"RESTART_INIT_STRATEGY",       "moved into `RESTART { METHOD <method> ... }` (re-uses INIT-block grammar)."},
        {"INIT_STRATEGY",               "moved into `INIT { METHOD <method> ... } ENDINIT` block."},
        {"INIT_K",                      "renamed to K_TOTAL inside an INIT block."},
        {"INIT_SEED",                   "renamed to SEED inside an INIT block."},
        {"INIT_INCLUDE_CONFIG",         "the values of the training configuration for the search parameters are a starting point only when the INIT block has a `METHOD CONFIG` line."},
        {"INCLUDE_CONFIG",              "the values of the training configuration for the search parameters are a starting point only when the INIT block has a `METHOD CONFIG` line."},
        {"NOISE_AUTO",                  "moved into `NOISE { AUTO true|false } ENDNOISE` block."},
        {"NOISE_CALIBRATION",           "moved into `NOISE { CALIBRATION N } ENDNOISE` block."},
        {"PC_SRMSE",                    "replaced by PC_SRMSE_GPA, the root mean square of the nine entries of each residual stress tensor in GPa (PC_SRMSE was that of the six entries x <= y in eV/A^3). PC_SRMSE_GPA of a residual is 160.2 times its PC_SRMSE when the six entries have equal mean squares, 130.8 times for a residual on the diagonal only (a pressure) and 185.0 times for a shear residual only: a target or --skip-above cap of PC_SRMSE times 160.2 is about that of PC_SRMSE_GPA, and a weight under LOSS L2 divided by about 25 670 (160.2 squared)."},
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
    if (!ifs)
        throw std::runtime_error("the target file " + fn +
                                 " (--hpotarget) cannot be opened");

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
     *           +-- RESTART     RESTART ... ENDRESTART   (decorator; populates spec.restart)
     *
     * There is one OPTIMIZER block, at the top level. An END line closes
     * the innermost open block, and the file ends with every block closed.
     * Each other form is refused at its line: a second OPTIMIZER block
     * would reopen the first and replace the settings both give, an
     * OPTIMIZER line inside a block would go back to the OPTIMIZER level,
     * an ENDOPTIMIZER inside another block would close the OPTIMIZER block
     * around it, and a block left open would be read as complete. A line
     * that opens or closes a block holds its key alone: a word after it
     * would be dropped (`OPTIMIZER LIB NLOPT`). */
    enum Block { ROOT, OUTER, INIT_BLK, NOISE_BLK, STRATEGY_BLK,
                 EXPLORE_BLK, REFINE_BLK, INNER_BLK, RESTART_BLK };
    /* The name of each block type: NAME opens it, ENDNAME closes it. */
    struct BlockName { Block id; const char* name; };
    static constexpr BlockName kBlockNames[] = {
        {OUTER, "OPTIMIZER"},       {INIT_BLK, "INIT"},
        {NOISE_BLK, "NOISE"},       {STRATEGY_BLK, "STRATEGY"},
        {EXPLORE_BLK, "EXPLORE"},   {REFINE_BLK, "REFINE"},
        {INNER_BLK, "INNER"},       {RESTART_BLK, "RESTART"}};
    auto block_name = [](Block b) -> std::string {
        for (const auto& bn : kBlockNames)
            if (bn.id == b) return bn.name;
        return "";
    };
    /* The block an END line closes (STRATEGY_BLK for ENDSTRATEGY), ROOT
     * for any other key. */
    auto block_ended_by = [](const std::string& key) -> Block {
        for (const auto& bn : kBlockNames)
            if (key == std::string("END") + bn.name) return bn.id;
        return ROOT;
    };
    /* The open blocks, outermost first, each with the line that opens it
     * and its text. `block` is the innermost, ROOT when none is open. */
    struct OpenBlock { Block id; std::size_t line; std::string text; };
    std::vector<OpenBlock> open_blocks;
    Block            block      = ROOT;
    OptimizerSpec*   active_opt    = nullptr;  // recipient of LIB/ALGO/...
    InitStrategySpec* active_init  = nullptr;  // recipient of STRATEGY/K_TOTAL/...
    OptimizerSpec*   inner_parent  = nullptr;  // outer spec of an INNER block
    /* The line of the INNER block of each optimiser: one block each. */
    std::map<const OptimizerSpec*, std::size_t> inner_lines;

    /* STRATEGY keys that belong to one TYPE (strategy_rules.h), checked
     * when the STRATEGY block ends: TYPE may come after them. The EXPLORE
     * and REFINE blocks need TYPE PRESTAGE before them, checked when they
     * open; TYPE is set once, so it stays. */
    struct TypeKeySeen { std::string key; std::size_t line; std::string text; };
    std::vector<TypeKeySeen> type_keys;
    bool strategy_type_given = false;
    /* The TYPE line (its number and text) and whether the block has an
     * N_STARTS line: TYPE MULTISTART needs N_STARTS 2 or more, checked
     * when the block ends (N_STARTS may follow TYPE). */
    std::size_t type_line = 0;
    std::string type_text;
    bool n_starts_given = false;
    /* The N_STARTS line and its text: whether the INIT chain gives the
     * starting points it asks for is known once every INIT block has been
     * read (checked when the file ends). */
    std::size_t n_starts_line = 0;
    std::string n_starts_text;
    /* The last ENDRESTART line and its text (0 before one): a RESTART
     * block -- the RESTART blocks of the STRATEGY block make one -- needs
     * AFTER_STAGNATION and MAX, checked when the STRATEGY block ends. */
    std::size_t endrestart_line = 0;
    std::string endrestart_text;
    /* The K_TOTAL line of each set of starting points (an INIT chain, the
     * RESTART chain) and its text: whether it caps anything is known once
     * every block that adds to the chain has been read. */
    std::map<const InitStrategySpec*, std::pair<std::size_t, std::string>>
        k_total_lines;
    /* The line of the STRATEGY block (0 before it): there is one. A second
     * block reopened the first and could change the TYPE its keys had been
     * checked against. */
    std::size_t strategy_line = 0;
    /* The line of the OPTIMIZER block (0 before it): there is one. */
    std::size_t optimizer_line = 0;
    /* The line of each PC_EMAE, PC_ErRMSE and PC_ERSQ line: one of each
     * (PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA: ValidationMetricTargets::add). */
    std::map<std::string, std::pair<std::size_t, std::string>> metric_lines;
    /* Where each key was set, per thing it configures (an optimiser, the
     * starting points of one, the NOISE, RESTART or STRATEGY settings, the
     * search space, the run for a top-level key): the line, the block and
     * the block around it. A key is set once for what it configures --
     * OPTIMIZER, STRATEGY and REFINE configure the STRATEGY-level
     * optimiser, the INIT blocks inside OPTIMIZER, STRATEGY and REFINE its
     * starting points, a repeated block continues the first -- so a second
     * line, which would replace the first's value, is refused. A STEP <KEY>
     * override is set once for its key and indices, and layers over the
     * block-wide STEP and over overrides of other indices. METHOD lines add
     * to their chain, and FAILSCORE, LOSS and PC_SCRIPT_TIMEOUT apply to the
     * PC_LAMMPS lines after them: none of them is set this way. */
    struct AttrSet { std::size_t line; Block block; Block parent; };
    std::map<std::pair<const void*, std::string>, AttrSet> attr_set;
    /* What a top-level key configures: the run. */
    static const char run_settings = 0;

    /* The block the open INNER block sits in: an INNER block opens only
     * inside STRATEGY, EXPLORE or REFINE. */
    auto inner_enclosing = [&]() -> std::string {
        const Block around = open_blocks.size() >= 2
                                 ? open_blocks[open_blocks.size() - 2].id
                                 : ROOT;
        return around == STRATEGY_BLK ? "STRATEGY" :
               around == EXPLORE_BLK  ? "EXPLORE"  : "REFINE";
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
    /* Opens a block of type `b` at the line being read; closes the
     * innermost open block. */
    auto open_block = [&](Block b) {
        open_blocks.push_back({b, nline, line});
        block = b;
    };
    auto close_block = [&]() {
        open_blocks.pop_back();
        block = open_blocks.empty() ? ROOT : open_blocks.back().id;
    };
    /* The line that opens or closes a block holds its key alone; `rest`
     * stands after the key. */
    auto require_key_alone = [&](std::istringstream& rest,
                                 const std::string& key) {
        std::string extra;
        if (!(rest >> extra)) return;
        if (const Block ends = block_ended_by(key); ends != ROOT)
            throw std::runtime_error(
                key + " closes the " + block_name(ends) + " block and takes "
                "no value (unexpected '" + extra + "')");
        throw std::runtime_error(
            key + " opens a block and takes no value (unexpected '" + extra +
            "'); each key stands on its own line inside the block");
    };
    /* A key the innermost open block does not take: a top-level key is
     * named as such, with the block that is still open; another key is
     * unknown there. @p valid names the keys the block takes, or is empty. */
    auto refuse_in_block = [&](const std::string& key,
                               const std::string& valid) {
        const OpenBlock& in = open_blocks.back();
        const std::string name = block_name(in.id);
        const std::string where = "the " + name + " block opened on line " +
                                  std::to_string(in.line);
        if (is_top_level_key(key))
            return std::runtime_error(key + " is a top-level key, and " +
                                      where + " is still open (END" + name +
                                      " closes it)");
        return std::runtime_error(
            "unknown key '" + key + "' inside " + where + " (" +
            (valid.empty() ? std::string() : "only " + valid + " are valid; ") +
            "END" + name + " closes it)");
    };
    /* The density gate lines, for the arming rule at the end of the file
     * (an OPTIM line that arms a gate may follow it). */
    struct GateLine { const char* key; std::size_t line; std::string text; };
    std::vector<GateLine> gate_lines;

    /* Every parse error names the line it is about. */
    auto at_line = [](std::size_t n, const std::string& what,
                      const std::string& text) {
        return config_line_error(n, what, text);
    };
    /* An error about an earlier line than the one being read: a STRATEGY
     * key found at the end of its block to have no effect. */
    struct EarlierLineError : std::runtime_error {
        std::size_t line;
        std::string text;
        EarlierLineError(std::size_t l, std::string t, const std::string& what)
            : std::runtime_error(what), line(l), text(std::move(t)) {}
    };
    /* The keys of one TYPE must match the TYPE of the block. */
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
    };

    auto strip_comment = [](std::string& s){
        auto p = s.find('#'); if (p!=std::string::npos) s.resize(p);
    };
    auto rtrim = [](std::string& s){
        s.erase(std::find_if(s.rbegin(), s.rend(),
                 [](int c){return !std::isspace(c);} ).base(), s.end());
    };

    /* Records that `attr` of `target` is set at this line, or refuses the
     * line when an earlier one set it; `what` names what two blocks of
     * different types that set it both configure. */
    auto set_once = [&](const void* target, const std::string& attr,
                        const char* what) {
        const Block parent = open_blocks.size() >= 2
                                 ? open_blocks[open_blocks.size() - 2].id
                                 : ROOT;
        const auto [it, fresh] = attr_set.emplace(
            std::make_pair(target, attr), AttrSet{nline, block, parent});
        if (fresh) return;
        const AttrSet& first = it->second;
        if (block == ROOT)
            throw std::runtime_error(
                attr + " is set twice (first on line " +
                std::to_string(first.line) + "); each key is given once");
        if (first.block == block && first.parent == parent)
            throw std::runtime_error(
                attr + " is set twice in the " + block_name(block) +
                " block (first on line " + std::to_string(first.line) +
                "); each key is given once");
        /* Blocks of one type inside two blocks (the INIT blocks inside
         * OPTIMIZER and inside STRATEGY) are named by the block around
         * them. */
        const bool by_parent = first.block == block;
        const auto where = [&](Block b, Block around) {
            return std::string("the ") + block_name(b) + " block" +
                   (by_parent ? std::string(" inside ") + block_name(around)
                              : std::string());
        };
        throw std::runtime_error(
            attr + " is set on line " + std::to_string(first.line) + " in " +
            where(first.block, first.parent) + " and again here in " +
            where(block, parent) + "; both blocks configure the same " +
            what + ", so each key is given once");
    };

    /* FAILSCORE, LOSS and PC_SCRIPT_TIMEOUT (MAXTIME): a line after one
     * reads its value when it uses it -- a PC_LAMMPS or PC_FAILPROBE line
     * without --failscore the FAILSCORE, one without --maxtime the
     * PC_SCRIPT_TIMEOUT, a --varloss term or a PC_OBSERVABLE line that
     * names no loss function the LOSS -- and the run-wide uses (the
     * ceiling of every loss, the validation metrics' loss) the last
     * FAILSCORE and LOSS. A second line is read only after a line has read
     * the first: otherwise the first would have no effect on any line. A
     * PC_SCRIPT_TIMEOUT, or a LOSS in a file without a metric line, that
     * no line after it reads is refused at the end of the file: it would
     * have no effect. (A line that gives its own value does not read the
     * one above it: such lines alone after a LOSS, FAILSCORE or
     * PC_SCRIPT_TIMEOUT line left it without effect.) The line, its text
     * and spelling, and whether a line has read it. */
    struct LineSetting {
        std::size_t line; std::string text; std::string spelling; bool read;
    };
    std::map<std::string, LineSetting> line_settings;
    /* @p readers: the lines that read the key. */
    auto set_for_lines = [&](const std::string& key, const char* readers) {
        const std::string name = key == "MAXTIME" ? "PC_SCRIPT_TIMEOUT" : key;
        const auto it = line_settings.find(name);
        if (it != line_settings.end() && !it->second.read)
            throw std::runtime_error(
                key + " is set twice (first on line " +
                std::to_string(it->second.line) +
                (it->second.spelling != key ? ", as " + it->second.spelling
                                            : std::string()) +
                ") and no line between reads the first (" + readers +
                "); each key is given once");
        line_settings[name] = {nline, line, key, false};
    };
    /* The line read now uses the setting @p name when @p uses. */
    auto line_reads_setting = [&](const char* name, bool uses) {
        if (!uses) return;
        if (const auto it = line_settings.find(name);
            it != line_settings.end())
            it->second.read = true;
    };
    const char* const kFailscoreReaders =
        "a PC_LAMMPS or PC_FAILPROBE line without --failscore";
    const char* const kTimeoutReaders =
        "a PC_LAMMPS or PC_FAILPROBE line without --maxtime";
    const char* const kLossReaders =
        "a --varloss or PC_OBSERVABLE term that names no loss function";
    /* Where the script limit of the PC_LAMMPS or PC_FAILPROBE line @p sim
     * comes from: its --maxtime, the last PC_SCRIPT_TIMEOUT (MAXTIME) line
     * above it, or the default. */
    auto maxtime_source = [&](const LammpsSim& sim) -> std::string {
        if (!sim.reads_default_maxtime) return "its --maxtime";
        const auto it = line_settings.find("PC_SCRIPT_TIMEOUT");
        if (it == line_settings.end()) return "the default";
        return it->second.spelling + " on line " +
               std::to_string(it->second.line);
    };
    /* The line of the last LOSS line (0: none), which the metric lines
     * take. */
    std::size_t loss_line = 0;

    while (std::getline(ifs,line))
    {
        ++nline;
        strip_comment(line);   rtrim(line);
        if (line.empty()) continue;

        while (!line.empty() && line.back()=='\\')
        {
            line.pop_back();
            std::string cont;
            /* A blank or comment line after the backslash is joined: only
             * the end of the file leaves the line without a next one. */
            if (!std::getline(ifs,cont))
                throw std::runtime_error(at_line(nline,
                    "the line ends with \\, which continues it on the next "
                    "line, and the file has no next line", line + "\\"));
            ++nline;  strip_comment(cont);  rtrim(cont);
            line += cont;
        }

        std::istringstream iss(line);
        std::string key;  iss>>key;
        if (key.empty()) continue;

        try {
            /* ============== block delimiters ====================== */

            /* An END line closes the innermost open block. */
            if (const Block ends = block_ended_by(key); ends != ROOT) {
                if (open_blocks.empty())
                    throw std::runtime_error(key + " without matching " +
                                             block_name(ends));
                const OpenBlock& in = open_blocks.back();
                if (in.id != ends)
                    throw std::runtime_error(
                        key + " does not close the innermost open block, "
                        "the " + block_name(in.id) + " block opened on line " +
                        std::to_string(in.line) + " (END" +
                        block_name(in.id) + " closes it)");
                require_key_alone(iss, key);
            }

            if (key=="OPTIMIZER") {
                if (!open_blocks.empty()) {
                    const OpenBlock& in = open_blocks.back();
                    throw std::runtime_error(
                        "OPTIMIZER block must appear at the top level of the "
                        "file (it is inside the " + block_name(in.id) +
                        " block opened on line " + std::to_string(in.line) +
                        ")");
                }
                if (optimizer_line != 0)
                    throw std::runtime_error(
                        "OPTIMIZER block given more than once (the first opens "
                        "on line " + std::to_string(optimizer_line) + "); "
                        "one OPTIMIZER block holds every optimiser setting");
                require_key_alone(iss, key);
                optimizer_line = nline;
                open_block(OUTER);
                refresh_active_opt();
                continue;
            }
            if (key=="ENDOPTIMIZER") {
                /* The STRATEGY-level optimiser is complete here: a short-form
                 * LIB/ALGO may follow the STRATEGY block. Its startup
                 * refusals name this line too (OptimizerSpec::block_line). */
                spec.block_line = nline;
                spec.block_line_text = line;
                require_driver(spec, DriverPhase::Run);
                close_block();  active_opt=nullptr;  continue;
            }

            if (key=="INIT") {
                if (block != OUTER && block != STRATEGY_BLK &&
                    block != EXPLORE_BLK && block != REFINE_BLK)
                    throw std::runtime_error(
                        "INIT block must appear inside OPTIMIZER, STRATEGY, EXPLORE, or REFINE");
                /* INIT populates the active OptimizerSpec's init_strategy_spec. */
                if (!active_opt)
                    throw std::logic_error("INIT block requires an active OptimizerSpec");
                require_key_alone(iss, key);
                active_init = &active_opt->init_strategy_spec;
                /* The block's line: its first INIT block when it has two
                 * (they make one chain). */
                if (active_init->line == 0) {
                    active_init->line = nline;
                    active_init->line_text = line;
                }
                /* Suspend active_opt so SEED/STRATEGY don't get gobbled by
                 * dispatch_opt_attr; restored on ENDINIT. */
                inner_parent = active_opt;     // reuse field for stash
                active_opt   = nullptr;
                open_block(INIT_BLK);
                continue;
            }
            if (key=="ENDINIT") {
                active_init = nullptr;
                close_block();
                active_opt = inner_parent;
                inner_parent = nullptr;
                continue;
            }

            if (key=="NOISE") {
                if (block != OUTER)
                    throw std::runtime_error("NOISE block must appear directly inside OPTIMIZER");
                require_key_alone(iss, key);
                /* A NOISE block takes AUTO and CALIBRATION only: no
                 * optimiser receives its lines, so an optimiser key there
                 * is refused as unknown. */
                open_block(NOISE_BLK);
                refresh_active_opt();
                continue;
            }
            if (key=="ENDNOISE") {
                close_block();
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
                    if (active_init && dispatch_init_attr(key, iss, *active_init, nline, line))
                        continue;
                    /* fall through to error */
                }
                if (block != OUTER)
                    throw std::runtime_error("STRATEGY block must appear directly inside OPTIMIZER");
                if (strategy_line != 0)
                    throw std::runtime_error(
                        "STRATEGY block given more than once (the first opens "
                        "on line " + std::to_string(strategy_line) + "); one "
                        "STRATEGY block holds every STRATEGY key");
                require_key_alone(iss, key);
                strategy_line = nline;
                open_block(STRATEGY_BLK);
                refresh_active_opt();
                continue;
            }
            if (key=="ENDSTRATEGY") {
                check_type_keys();
                /* N_STARTS 1 is refused at its line; without N_STARTS a
                 * MULTISTART would run one start, which is TYPE PLAIN. */
                if (spec.strategy_type == StrategyType::MULTISTART &&
                    !n_starts_given)
                    throw EarlierLineError(
                        type_line, type_text,
                        "STRATEGY TYPE MULTISTART needs N_STARTS 2 or more: "
                        "without N_STARTS it runs one start, which is "
                        "STRATEGY TYPE PLAIN");
                /* A restart comes after AFTER_STAGNATION evaluations without
                 * a new best, at most MAX times: a block without one of them
                 * would restart nothing (0 is refused at its line). */
                if (endrestart_line != 0 && !spec.restart.active()) {
                    const bool no_after = spec.restart.after_stagnation == 0;
                    const bool no_max = spec.restart.max == 0;
                    const std::string missing =
                        no_after && no_max
                            ? "no AFTER_STAGNATION line and no MAX line"
                        : no_after ? "no AFTER_STAGNATION line"
                                   : "no MAX line";
                    throw EarlierLineError(
                        endrestart_line, endrestart_text,
                        "the RESTART block never restarts the search: a "
                        "restart needs AFTER_STAGNATION and MAX, and the block "
                        "has " + missing + "; a search without restarts has "
                        "no RESTART block");
                }
                close_block();
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
                require_key_alone(iss, key);
                if (!spec.prestage)
                    spec.prestage = std::make_unique<OptimizerSpec>();
                open_block(EXPLORE_BLK);
                refresh_active_opt();
                continue;
            }
            if (key=="ENDEXPLORE") {
                spec.prestage->block_line = nline;
                spec.prestage->block_line_text = line;
                require_driver(*spec.prestage, DriverPhase::Explore);
                close_block();
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
                require_key_alone(iss, key);
                open_block(REFINE_BLK);
                refresh_active_opt();
                continue;
            }
            if (key=="ENDREFINE") {
                close_block();
                refresh_active_opt();
                continue;
            }

            if (key=="INNER") {
                if (block != STRATEGY_BLK && block != EXPLORE_BLK && block != REFINE_BLK)
                    throw std::runtime_error(
                        "INNER block must appear inside STRATEGY, EXPLORE or "
                        "REFINE: it holds the local search that an NLopt MLSL "
                        "or AUGLAG algorithm runs");
                if (!active_opt)
                    throw std::logic_error("INNER block has no parent OptimizerSpec");
                if (active_opt->inner)
                    throw std::runtime_error(
                        "INNER block given twice for one optimiser (the "
                        "first on line " +
                        std::to_string(inner_lines[active_opt]) +
                        "); STRATEGY and REFINE configure the same "
                        "optimiser, and so do two EXPLORE blocks");
                inner_lines[active_opt] = nline;
                require_key_alone(iss, key);
                active_opt->inner = std::make_unique<OptimizerSpec>();
                inner_parent = active_opt;
                active_opt = active_opt->inner.get();
                open_block(INNER_BLK);
                continue;
            }
            if (key=="ENDINNER") {
                /* The INNER optimiser is complete here, like an EXPLORE
                 * block's at ENDEXPLORE. */
                active_opt->block_line = nline;
                active_opt->block_line_text = line;
                require_inner_driver(*active_opt, inner_enclosing());
                close_block();
                inner_parent = nullptr;
                refresh_active_opt();
                continue;
            }

            if (key=="RESTART") {
                if (block != STRATEGY_BLK)
                    throw std::runtime_error("RESTART block must appear inside STRATEGY");
                require_key_alone(iss, key);
                active_init = &spec.restart.init_strategy;
                if (spec.restart.line == 0) {
                    spec.restart.line = nline;
                    spec.restart.line_text = line;
                }
                /* Suspend active_opt while inside RESTART so SEED/STRATEGY
                 * go to active_init, not the OUTER spec. A RESTART with no
                 * STRATEGY lines falls back to a single CONFIG point. */
                inner_parent = active_opt;
                active_opt   = nullptr;
                open_block(RESTART_BLK);
                continue;
            }
            if (key=="ENDRESTART") {
                endrestart_line = nline;
                endrestart_text = line;
                active_init = nullptr;
                close_block();
                active_opt = inner_parent;
                inner_parent = nullptr;
                continue;
            }

            /* ============== old-grammar rejection ================= */
            reject_old_keyword(key);

            /* ============== block-specific dispatch =============== */

            /* SEED, GRAD_STRATEGY and STOPVAL inside INNER: the INNER
             * search draws from the random stream of the enclosing block's
             * run, takes the gradient its GRAD_STRATEGY obtains and stops at
             * its STOPVAL (inner_rejects_attr). */
            if (block == INNER_BLK && !inner_takes_attr(key)) {
                throw std::runtime_error(
                    key + " is not accepted inside an INNER block: " +
                    inner_rejects_attr(key) + "; the enclosing " +
                    inner_enclosing() + " block takes " + key);
            }

            /* Common opt-attribute keys (LIB/ALGO/MAXEVAL/...) accepted in
             * any block where active_opt is set. */
            /* The attribute a line of an optimiser sets: its key; PARAM
             * <name> for a PARAM line; none for a STEP <KEY> override, which
             * refines the block-wide STEP of some dimensions. */
            std::string attr = key;
            if (key == "PARAM" || key == "STEP") {
                std::istringstream peek(line);
                std::string k0, first;
                peek >> k0 >> first;
                if (key == "PARAM") attr = "PARAM " + first;
                else if (!first.empty() && !is_number_shaped(first)) attr.clear();
            }
            if (active_opt && dispatch_opt_attr(key, iss, *active_opt)) {
                /* The line of a key whose refusal is about it alone
                 * (OptimizerSpec::key_lines, with_key_line); a STEP <KEY>
                 * override keeps its own (StepOverride::line). */
                if (attr.empty() && key == "STEP" &&
                    !active_opt->step_overrides.empty()) {
                    StepOverride& ov = active_opt->step_overrides.back();
                    ov.line = nline;
                    ov.line_text = line;
                } else if (key == "LIB" || key == "ALGO" || key == "STEP" ||
                           key == "GRAD_STRATEGY") {
                    active_opt->key_lines[key] =
                        KeyLine{nline, line, block_name(block)};
                }
                /* A STEP <KEY> override: once for its key and indices
                 * ("WATOM(1)" and "WATOM (1)" are one). */
                if (attr.empty() && key == "STEP" &&
                    !active_opt->step_overrides.empty()) {
                    const StepOverride& ov = active_opt->step_overrides.back();
                    std::set<std::size_t> idx(ov.indices.begin(),
                                              ov.indices.end());
                    attr = "STEP " + ov.key;
                    if (!idx.empty()) {
                        std::string list;
                        for (const std::size_t i : idx)
                            list += (list.empty() ? "" : ",") +
                                    std::to_string(i);
                        attr += "(" + list + ")";
                    }
                }
                if (!attr.empty()) set_once(active_opt, attr, "optimiser");
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
                if (active_init && dispatch_init_attr(key, iss, *active_init, nline, line)) {
                    /* METHOD (and its alias STRATEGY) adds to the chain */
                    if (key != "METHOD" && key != "STRATEGY")
                        set_once(active_init, key, "starting points");
                    if (key == "K_TOTAL")
                        k_total_lines[active_init] = {nline, line};
                    continue;
                }
                if (block == RESTART_BLK) {
                    /* Each 1 or more: 0 restarts nothing. */
                    const char* const kNoRestart =
                        "0 never restarts the search; a search without "
                        "restarts has no RESTART block";
                    if (key=="AFTER_STAGNATION") {
                        spec.restart.after_stagnation = parse_uint(
                            single_value(iss, key), key, 1,
                            std::numeric_limits<std::uint64_t>::max(),
                            kNoRestart);
                        set_once(&spec.restart, key, "restarts");
                        continue;
                    }
                    if (key=="MAX") {
                        spec.restart.max = parse_uint(
                            single_value(iss, key), key, 1,
                            std::numeric_limits<std::uint64_t>::max(),
                            kNoRestart);
                        set_once(&spec.restart, key, "restarts");
                        continue;
                    }
                }
                throw refuse_in_block(key, "");
            }

            /* NOISE block keys */
            if (block == NOISE_BLK) {
                if (key=="AUTO") {
                    /* AUTO alone means AUTO true. */
                    std::string v;
                    iss >> v;
                    expect_end(iss, "AUTO takes one value");
                    spec.noise.autodetect = v.empty() ? true : parse_bool(v, "NOISE.AUTO");
                    set_once(&spec.noise, key, "noise settings");
                    continue;
                }
                if (key=="CALIBRATION") {
                    /* N evaluations, at most the int budget MAXEVAL is
                     * (optimizer_block.cpp). */
                    const std::size_t v = parse_uint(
                        single_value(iss, key), key, 0,
                        std::numeric_limits<int>::max());
                    if (v == 1)
                        throw std::runtime_error("NOISE.CALIBRATION must be 0 (off) or >= 2");
                    spec.noise.calibration_n = v;
                    set_once(&spec.noise, key, "noise settings");
                    continue;
                }
                throw refuse_in_block(key, "AUTO and CALIBRATION");
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
                    set_once(&spec.log_threshold, key, "search space");
                    spec.log_threshold = v;
                    continue;
                }
                throw std::runtime_error(
                    block == ROOT
                        ? std::string("LOG_HP is written once, inside "
                                      "OPTIMIZER or STRATEGY: it holds for "
                                      "the whole search space and is not a "
                                      "top-level key")
                        : "LOG_HP is written once, inside OPTIMIZER or "
                          "STRATEGY: it holds for the whole search space, "
                          "and the " + block_name(block) + " block does not "
                          "take it");
            }

            /* STRATEGY block keys */
            if (block == STRATEGY_BLK) {
                if (key=="TYPE") {
                    const std::string t = single_value(iss, key);
                    if      (t=="PLAIN")      spec.strategy_type = StrategyType::PLAIN;
                    else if (t=="MULTISTART") spec.strategy_type = StrategyType::MULTISTART;
                    else if (t=="PRESTAGE")   spec.strategy_type = StrategyType::PRESTAGE;
                    else throw std::runtime_error("STRATEGY TYPE must be PLAIN, MULTISTART, or PRESTAGE (got '" + t + "')");
                    set_once(&spec.strategy_type, key, "strategy");
                    strategy_type_given = true;
                    type_line = nline;
                    type_text = line;
                    continue;
                }
                /* The keys of one TYPE are checked against the block's TYPE
                 * when it ends (TYPE may follow them). */
                if (is_strategy_type_key(key))
                    type_keys.push_back({key, nline, line});
                /* MULTISTART knobs. */
                if (key=="N_STARTS") {
                    /* One start is TYPE PLAIN: N_STARTS 1 would make a
                     * MULTISTART that is one. */
                    spec.multistart.n_starts = parse_uint(
                        single_value(iss, key), key, 2,
                        std::numeric_limits<std::uint64_t>::max(),
                        "one start is STRATEGY TYPE PLAIN");
                    set_once(&spec.strategy_type, key, "strategy");
                    n_starts_given = true;
                    n_starts_line = nline;
                    n_starts_text = line;
                    continue;
                }
                if (key=="BUDGET_PER_START") {
                    /* Each start's MAXEVAL: the int budget MAXEVAL is
                     * (optimizer_block.cpp). */
                    spec.multistart.budget_per_start =
                        parse_uint(single_value(iss, key), key, 1,
                                   std::numeric_limits<int>::max());
                    set_once(&spec.strategy_type, key, "strategy");
                    continue;
                }
                /* PRESTAGE knob: the candidates the refinement starts
                 * from, a set of starting points (init_strategy.h). */
                if (key=="TOPK") {
                    spec.stage_topk = parse_uint(single_value(iss, key), key, 1,
                                                 kMaxStartingPoints);
                    set_once(&spec.strategy_type, key, "strategy");
                    continue;
                }
                throw refuse_in_block(key, "");
            }

            /* EXPLORE / REFINE block keys: only the optimiser attributes
             * dispatched above (LIB/ALGO/MAXEVAL/...) are valid there. */
            if (block == EXPLORE_BLK || block == REFINE_BLK) {
                throw refuse_in_block(key, "");
            }

            /* INNER block -- only opt attributes, dispatched above. */
            if (block == INNER_BLK) {
                throw refuse_in_block(key, inner_attr_keys_joined("/"));
            }

            /* Anything still unhandled inside an OPTIMIZER-level block is
             * a user mistake (LIB/ALGO/MAXEVAL etc. now live inside
             * STRATEGY). LOSS / PC_LAMMPS / OPTIM / FAILSCORE / OUTPUT /
             * BDUMP and friends are ROOT-level and fall through. */
            if (block != ROOT) {
                throw refuse_in_block(key, "");
            }

            /* A top-level key that takes one value is set once: a second
             * line would replace the first's value. FAILSCORE, LOSS and
             * PC_SCRIPT_TIMEOUT (MAXTIME) are set again only after a line
             * has read them (set_for_lines); the metric and gate keys
             * refuse a repeat themselves. */
            {
                static const std::set<std::string> once = {
                    "QFLUSH", "OUTPUT", "BOUTPUT", "DIGITS", "DUMP",
                    "BDUMP", "FAILSCORE_GRADING", "BUMPGATE", "PC_ISOLATION",
                    "PC_WORKER_TIMEOUT", "OPTIM_SCALE", "RHOGATE",
                    "RHOBARGATE"};
                if (once.count(key)) set_once(&run_settings, key, "run");
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
                set_for_lines(key, kFailscoreReaders);
                FAIL_SCORE = parse_positive(single_value(iss, key), key,
                                            "finite number");
            }
            /* PC_SCRIPT_TIMEOUT, and MAXTIME at the top level, another name
             * of the key: the wall-clock limit of each section of a
             * PC_LAMMPS and PC_FAILPROBE script, a LAMMPS time spec (off,
             * unlimited or a positive [[HH:]MM:]SS) injected as LAMMPS
             * `timer timeout` after each clear. Each such line takes the
             * value of the last line of the key above it (set_for_lines);
             * its --maxtime gives it its own. MAXTIME inside an optimiser
             * block is a different setting (optimizer_block.cpp). */
            else if (key=="PC_SCRIPT_TIMEOUT" || key=="MAXTIME") {
                /* A LAMMPS time spec that is off, unlimited or positive:
                 * it becomes `timer timeout <spec>`, and its seconds size
                 * the PROCESS worker deadline of the tasks of each line
                 * that reads it. */
                set_for_lines(key, kTimeoutReaders);
                MAX_TIME = single_value(iss, key);
                MAX_TIME_seconds = parse_script_timeout(MAX_TIME, key);
                if (key=="MAXTIME" && MAXTIME_line == 0) MAXTIME_line = nline;
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
                PC_WORKER_TIMEOUT_line = nline;
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
                RHOGATE = parse_rhogate(iss);
                RHOGATE->line = nline;
                RHOGATE->line_text = line;
                gate_lines.push_back({"RHOGATE", nline, line});
            }
            else if (key=="RHOBARGATE") {
                RHOBARGATE = parse_rhobargate(iss);
                RHOBARGATE->line = nline;
                RHOBARGATE->line_text = line;
                gate_lines.push_back({"RHOBARGATE", nline, line});
            }

            /* PC_FAILPROBE: minimiser-free potential-sanity probe (graded-
             * FAILSCORE Tier 3). Reads the probe-only options (--gate, --s0)
             * here; the LammpsSim parser skips them with their value, so a
             * word after one is named with it. Loss components on the probe
             * are forbidden -- it must not shape the feasible-region
             * objective. */
            else if (key=="PC_FAILPROBE")
            {
                if (failprobe_idx >= 0)
                    throw std::runtime_error("only one PC_FAILPROBE line is allowed");
                std::string cmd; std::getline(iss,cmd);
                std::istringstream ts(cmd);
                std::vector<std::string> toks;
                { std::string t; while (ts>>t) toks.push_back(t); }
                bool gate_seen = false;
                for (std::size_t i2 = 0; i2 < toks.size(); ++i2) {
                    if (toks[i2]=="--gate" || toks[i2]=="--s0") {
                        const std::string what = "PC_FAILPROBE " + toks[i2];
                        /* Each is given once: a second replaced the
                         * first's value. */
                        bool& seen = toks[i2]=="--gate" ? gate_seen
                                                        : failprobe_s0_given;
                        if (seen)
                            throw std::runtime_error(
                                what + " is given twice on this line; each "
                                "option is given once");
                        seen = true;
                        if (i2+1 >= toks.size())
                            throw std::runtime_error("PC_FAILPROBE " + toks[i2]
                                                     + " requires a value");
                        if (toks[i2]=="--gate")
                            failprobe_gate = parse_finite(toks[i2+1], what);
                        else
                            failprobe_s0 = parse_positive(toks[i2+1], what);
                        ++i2;
                    }
                }
                LS.emplace_back(cmd, MAX_TIME, FAIL_SCORE, globalLoss,
                                /*failprobe=*/true);
                LS.back().line = nline;
                LS.back().line_text = line;
                LS.back().maxtime_source = maxtime_source(LS.back());
                line_reads_setting("FAILSCORE",
                                   LS.back().reads_default_failscore);
                line_reads_setting("PC_SCRIPT_TIMEOUT",
                                   LS.back().reads_default_maxtime);
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
            /* These are moduli (`iter % every`), so 0 is an integer
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
                /* <cadence> <directory>. The directory is the run's
                 * output: the two are checked together once the file is
                 * read (check_archive_directories), each error at its
                 * line, for the run and for `tadah hpo --check` alike, and
                 * created by the run just before its first evaluation
                 * (create_archive_directories); parsing creates nothing. */
                std::string every, dir;
                if (!(iss >> every >> dir))
                    throw std::runtime_error(
                        key + " requires a cadence and a directory");
                expect_end(iss, key + " takes a cadence and a directory");
                const std::size_t n = parse_uint(every, key + " cadence", 1);
                if (key == "DUMP") {
                    dumpPotEvery = n; dumpPotDir = dir;
                    dumpPotLine = {nline, line};
                } else {
                    dumpBestPotEvery = n; dumpBestPotDir = dir;
                    dumpBestPotLine = {nline, line};
                }
            }

            /* loss -------------------------------------------------- */
            else if (key=="LOSS")
            {
                set_for_lines(key, kLossReaders);
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
                auto id = tadah::LossFunction::from_name(type, "LOSS");
                globalLoss = tadah::LossFunction(id,params);
                loss_line = nline;
            }

            /* optimisation variables --------------------------------*/
            else if (key=="OPTIM")   { process_optim_config(iss, nline); }
            else if (key=="OPTIM_SCALE") {
                const std::string tok = single_value(iss, key);
                std::string v = tok;
                for (auto& ch : v) ch = static_cast<char>(std::toupper(static_cast<unsigned char>(ch)));
                if      (v == "AUTO") scale_policy_.normalise = true;
                else if (v == "NONE") scale_policy_.normalise = false;
                else throw std::runtime_error(
                    "OPTIM_SCALE must be AUTO (every OPTIM dimension is mapped "
                    "to [0, 1]) or NONE (each dimension stays in its OPTIM "
                    "units: the value for LIN, its natural logarithm for LOG, "
                    "asinh(x/c) for ASINH <c>), got '" + tok + "'");
                scale_policy_set_ = true;
            }

            /* LAMMPS sims ------------------------------------------ */
            else if (key=="PC_LAMMPS")
            {
                std::string cmd; std::getline(iss,cmd);
                LS.emplace_back(cmd, MAX_TIME, FAIL_SCORE, globalLoss);
                LS.back().line = nline;
                LS.back().line_text = line;
                LS.back().maxtime_source = maxtime_source(LS.back());
                line_reads_setting("FAILSCORE",
                                   LS.back().reads_default_failscore);
                line_reads_setting("LOSS", LS.back().reads_default_loss);
                line_reads_setting("PC_SCRIPT_TIMEOUT",
                                   LS.back().reads_default_maxtime);
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
                /* Its loss function is that of the LOSS line above it
                 * unless it gives --loss, as for a --varloss term. */
                std::string cmd; std::getline(iss, cmd);
                observables.push_back(parse_pc_observable_line(cmd, globalLoss));
                observable_lines.push_back({nline, line});
                line_reads_setting("LOSS",
                                   observables.back()->reads_default_loss());
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
                  || key=="PC_SRMSE_GPA"
                  || key=="PC_FRMSE")
            {
                const bool dataset_metric = key == "PC_ERMSE" ||
                                            key == "PC_FRMSE" ||
                                            key == "PC_SRMSE_GPA";
                std::vector<std::size_t> indices;
                iss >> std::ws;
                if (iss.peek() == '(') {
                    if (!dataset_metric)
                        throw std::runtime_error(key + ": indexed targets are supported only for PC_ERMSE, PC_FRMSE and PC_SRMSE_GPA");
                    indices = core::parse_indices_strict(read_index_selector(iss, key));
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
                    auto& targets = key == "PC_ERMSE" ? ermse_targets
                                  : key == "PC_FRMSE" ? frmse_targets
                                                      : srmse_targets;
                    targets.add(indices, {tgt, wgt, caps, nline, line});
                    if (key == "PC_ERMSE") use_ermse = true;
                    else if (key == "PC_FRMSE") use_frmse = true;
                    else use_srmse = true;
                }
                else if (const auto [first, fresh] =
                             metric_lines.emplace(key, std::make_pair(nline, line));
                         !fresh)
                    /* A second line would replace the first's target,
                     * weight and caps. */
                    throw std::runtime_error(
                        key + ": repeated declaration (first declared on "
                        "line " + std::to_string(first->second.first) + ")");
                else if (key=="PC_EMAE")   { emae=tgt;   w_emae=wgt;   use_emae=true;   skip_emae=caps; }
                else if (key=="PC_ErRMSE") { errmse=tgt; w_errmse=wgt; use_errmse=true; skip_errmse=caps; }
                else if (key=="PC_ERSQ")   { ersq=tgt;   w_ersq=wgt;   use_ersq=true;   skip_ersq=caps; }
            }
            else
                throw std::runtime_error("unknown key '" + key + "'");
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

    /* A block the file leaves open: at the line that opens it, the
     * innermost first. Each block is closed past this point, and its END
     * line has checked what it sets (check_type_keys, require_driver,
     * require_inner_driver). */
    if (!open_blocks.empty()) {
        const OpenBlock& in = open_blocks.back();
        throw std::runtime_error(at_line(in.line,
            block_name(in.id) + " block not closed with END" +
                block_name(in.id),
            in.text));
    }
    /* K_TOTAL caps the starting points the METHOD lines of its chain give,
     * at most the chain's default_k() (the K of RANDOM, LHS and SOBOL, one
     * point of CONFIG and of WARM, which gives at most one; without a
     * METHOD line the CONFIG point): one at or above their number caps
     * nothing. The INIT blocks of one optimiser make one chain, and so do
     * the RESTART blocks. An EXPLORE chain without a METHOD line is not
     * resolved: the exploration takes the points of the STRATEGY-level
     * chain (share_init_with_nested_phases), which its K_TOTAL does not
     * cap. */
    std::vector<std::pair<std::pair<std::size_t, std::string>,
                          const InitStrategySpec*>> k_totals;
    for (const auto& [init, at] : k_total_lines) k_totals.push_back({at, init});
    std::sort(k_totals.begin(), k_totals.end(),
              [](const auto& a, const auto& b) {
                  return a.first.first < b.first.first;
              });
    for (const auto& [at, init] : k_totals) {
        if (!init->k_total) continue;
        const std::size_t points = build_init_strategy(*init)->default_k();
        std::string list;
        for (const auto& e : init->chain) {
            list += (list.empty() ? "" : ", ") + e.method;
            for (const auto& a : e.args) list += " " + a;
        }
        const std::size_t cap = *init->k_total;
        const std::string head =
            "K_TOTAL " + std::to_string(cap) + " caps nothing: ";
        const std::string keeps =
            ", and K_TOTAL keeps the first K_TOTAL starting points";
        const bool explore =
            spec.prestage && init == &spec.prestage->init_strategy_spec;
        if (init->chain.empty() && explore)
            throw std::runtime_error(at_line(at.first,
                head + "an INIT block inside EXPLORE without a METHOD line "
                "gives the exploration the starting points of the INIT "
                "blocks inside OPTIMIZER, STRATEGY and REFINE (the CONFIG "
                "point without one), and K_TOTAL caps only the points of the "
                "block's own METHOD lines", at.second));
        if (init->chain.empty())
            throw std::runtime_error(at_line(at.first,
                head + "a block without a METHOD line gives one starting "
                "point (the CONFIG point)" + keeps, at.second));
        /* The blocks of the chain, which a K_TOTAL in a block without a
         * METHOD line caps too. */
        const std::string blocks =
            init == &spec.restart.init_strategy ? "the RESTART blocks"
            : explore ? "the INIT blocks inside EXPLORE"
                      : "the INIT blocks inside OPTIMIZER, STRATEGY and REFINE";
        if (cap >= points)
            throw std::runtime_error(at_line(at.first,
                head + "the METHOD lines of " + blocks + " give at most " +
                    std::to_string(points) +
                    (points == 1 ? " starting point (" : " starting points (") +
                    list + ")" + keeps,
                at.second));
    }

    /* MULTISTART runs one start from each of N_STARTS starting points of
     * the INIT chain (the INIT blocks inside OPTIMIZER and STRATEGY). The
     * most points the chain can give are known here: the K of its RANDOM,
     * LHS and SOBOL lines and one point of each CONFIG and WARM line (the
     * CONFIG point without a METHOD line), at most K_TOTAL. A chain that
     * gives fewer than N_STARTS is refused at the N_STARTS line. A WARM
     * point may still be dropped when the chain is drawn (WARM_ON_MISMATCH),
     * and a RESTART round draws points of its own: a start the run cannot
     * make there is warned about when it runs (MultiStartPolicy). */
    if (spec.strategy_type == StrategyType::MULTISTART && n_starts_line != 0) {
        const InitStrategySpec& init = spec.init_strategy_spec;
        const bool has_warm =
            std::any_of(init.chain.begin(), init.chain.end(),
                        [](const InitStrategySpec::Entry& e) {
                            return e.method == "WARM";
                        });
        const std::size_t drawn = build_init_strategy(init)->default_k();
        const std::size_t points =
            init.k_total ? std::min(drawn, *init.k_total) : drawn;
        const std::size_t want = spec.multistart.n_starts;
        if (points < want) {
            const std::string k = std::to_string(want);
            const std::string n = std::to_string(points);
            std::string list;
            for (const auto& e : init.chain) {
                list += (list.empty() ? "" : ", ") + e.method;
                for (const auto& a : e.args) list += " " + a;
            }
            const std::string from =
                init.chain.empty()
                    ? "the CONFIG point: the chain has no METHOD line"
                : init.k_total
                    ? list + ", capped by K_TOTAL " +
                          std::to_string(*init.k_total)
                    : list;
            const std::string more =
                "a chain of " + k + " or more starting points (METHOD LHS " +
                k + ", RANDOM " + k + " or SOBOL " + k + ")";
            std::string advice;
            if (!init.k_total)
                advice = more + " runs " + k + " starts";
            else if (drawn > want)
                advice = "with K_TOTAL " + k + ", or without K_TOTAL, " + k +
                         " starts run";
            else if (drawn == want)
                advice = "without K_TOTAL, " + k + " starts run";
            else
                advice = more + " without K_TOTAL runs " + k + " starts";
            throw std::runtime_error(at_line(n_starts_line,
                "N_STARTS " + k + " needs " + k + " starting points, and the "
                "INIT chain gives " + (has_warm ? "at most " : "") + n +
                " (" + from + "); " +
                (points == 1 ? "one start is STRATEGY TYPE PLAIN, and " + advice
                             : advice + ", and N_STARTS " + n +
                                   " starts from each of the " + n),
                n_starts_text));
        }
    }


    /* A PC_SCRIPT_TIMEOUT (MAXTIME) that no PC_LAMMPS or PC_FAILPROBE line
     * without --maxtime after it reads, and a LOSS that no --varloss or
     * PC_OBSERVABLE term without a loss function of its own after it reads
     * in a file without a metric line, would have no effect
     * (set_for_lines); refused at its line. A FAILSCORE is the ceiling of
     * every loss as well. */
    {
        const bool has_metric_lines = use_emae || use_ermse || use_errmse ||
                                      use_ersq || use_frmse || use_srmse;
        if (const auto it = line_settings.find("PC_SCRIPT_TIMEOUT");
            it != line_settings.end() && !it->second.read)
            throw std::runtime_error(at_line(it->second.line,
                it->second.spelling + " sets the script timeout of the "
                "PC_LAMMPS and PC_FAILPROBE lines after it that give no "
                "--maxtime, and no such line follows it; it would have no "
                "effect",
                it->second.text));
        if (const auto it = line_settings.find("LOSS");
            it != line_settings.end() && !it->second.read &&
            !has_metric_lines)
            throw std::runtime_error(at_line(it->second.line,
                "LOSS sets the loss function of the --varloss and "
                "PC_OBSERVABLE terms after it that name none, and of the "
                "metric lines (PC_EMAE, PC_ERMSE, PC_ErRMSE, PC_ERSQ, "
                "PC_FRMSE, PC_SRMSE_GPA), and the file has no metric line and "
                "no such term after it; it would have no effect",
                it->second.text));
    }

    /* A target the loss function of the metric lines -- the last LOSS line
     * -- cannot measure (MAPE 0, RMSLE at or below -1e-12): every
     * evaluation would score the term the same (MAPE: |m - 0| / |0|), or
     * fail it (RMSLE: ln of a negative number). Refused at the metric's
     * line, the first in the file; the LOSS line may follow it. */
    {
        struct MetricTarget {
            std::size_t line; std::string text; std::string what; double target;
        };
        std::vector<MetricTarget> targets;
        auto add_metric = [&](bool used, const char* key, double target) {
            if (!used) return;
            const auto it = metric_lines.find(key);
            if (it != metric_lines.end())
                targets.push_back({it->second.first, it->second.second, key,
                                   target});
        };
        add_metric(use_emae, "PC_EMAE", emae);
        add_metric(use_errmse, "PC_ErRMSE", errmse);
        add_metric(use_ersq, "PC_ERSQ", ersq);
        auto add_dataset_metric = [&](const ValidationMetricTargets& t,
                                      const char* key) {
            if (const auto& g = t.global_assignment())
                targets.push_back({g->line, g->text, key, g->target});
            for (const auto& [index, a] : t.indexed_assignments())
                targets.push_back({a.line, a.text,
                                   std::string(key) + " (" +
                                       std::to_string(index) + ")",
                                   a.target});
        };
        add_dataset_metric(ermse_targets, "PC_ERMSE");
        add_dataset_metric(frmse_targets, "PC_FRMSE");
        add_dataset_metric(srmse_targets, "PC_SRMSE_GPA");
        std::stable_sort(targets.begin(), targets.end(),
                         [](const MetricTarget& a, const MetricTarget& b) {
                             return a.line < b.line;
                         });
        for (const auto& t : targets) {
            const std::string why = globalLoss.target_refusal(t.target);
            if (why.empty()) continue;
            std::ostringstream what;
            what << t.what << " " << t.target << ": the metric lines take the "
                 << "loss function of the last LOSS line (line " << loss_line
                 << "): " << why;
            throw std::runtime_error(at_line(t.line, what.str(), t.text));
        }
    }
    /* A density gate that no OPTIM line arms cannot act: no dimension of
     * the search changes rho(r) or its window -- a DM_REAM density key or a
     * DENSITY parameter of a DM_Embed component -- so every candidate has
     * the density of the training configuration and its one verdict (OPTIM
     * WATOM only scales the density, and the gate verdicts are
     * scale-invariant). Refused at the gate's line, the first in the file;
     * the OPTIM line that arms it may stand anywhere. */
    {
        bool armed = false;
        for(const auto &kv:optimap) if(kv.first.find(".DENSITY.")!=std::string::npos)
          for(bool selected:kv.second.opt)armed=armed||selected;
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
                "that changes the density or its window (" + keys + " of a "
                "DM_REAM many-body term, or a DENSITY parameter "
                "<ID>.DENSITY.<parameter> of a DM_Embed component), so every "
                "candidate has the density of the training configuration and "
                "its one verdict (OPTIM WATOM only scales the density; the "
                "verdict does not change). The " + g.key + " line takes "
                "effect with an OPTIM line on one of these keys, anywhere in "
                "the file.", g.text));
        }
    }

    /* The DUMP and BDUMP directories, together: one may be inside the
     * other's new directory, whatever the order of their lines. */
    check_archive_directories();

    /* A file without an OPTIMIZER block names no optimiser. */
    require_driver(spec, DriverPhase::Run);

    /* PC_OBSERVABLE lines do not share a target file with PC_LAMMPS or
     * PC_FAILPROBE lines: the run serves one of the two kinds of term.
     * Refused at the first line of the group that comes later in the file,
     * naming the lines of the other group. */
    if (!observables.empty() && !LS.empty()) {
        struct Group { std::string key; std::size_t n = 0, first = 0; };
        /* "1 PC_LAMMPS line (line 3)", "2 PC_LAMMPS lines (the first on
         * line 3)". */
        auto describe = [](const Group& g) {
            return core::to_string(g.n) + " " + g.key +
                   (g.n == 1 ? " line (line " : " lines (the first on line ") +
                   core::to_string(g.first) + ")";
        };
        auto and_list = [](const std::vector<std::string>& items) {
            std::string out;
            for (std::size_t i = 0; i < items.size(); ++i)
                out += (i == 0 ? "" : i + 1 == items.size()
                                          ? (items.size() == 2 ? " and "
                                                               : ", and ")
                                          : ", ") + items[i];
            return out;
        };
        Group obs{"PC_OBSERVABLE"}, lammps{"PC_LAMMPS"}, probe{"PC_FAILPROBE"};
        for (const auto& l : observable_lines) {
            if (obs.n++ == 0) obs.first = l.number;
        }
        const LammpsSim* first_script = nullptr;
        for (const auto& sim : LS) {
            Group& g = sim.is_failprobe ? probe : lammps;
            if (g.n++ == 0) g.first = sim.line;
            if (!first_script || sim.line < first_script->line)
                first_script = &sim;
        }
        const std::string tail =
            "; PC_OBSERVABLE lines do not share a target file with PC_LAMMPS "
            "or PC_FAILPROBE lines";
        if (first_script->line > obs.first)
            throw std::runtime_error(at_line(first_script->line,
                std::string(first_script->is_failprobe ? "PC_FAILPROBE"
                                                       : "PC_LAMMPS") +
                    ": this target file also has " + describe(obs) + tail,
                first_script->line_text));
        std::vector<std::string> others;
        for (const Group* g : {&lammps, &probe})
            if (g->n) others.push_back(describe(*g));
        throw std::runtime_error(at_line(observable_lines.front().number,
            "PC_OBSERVABLE: this target file also has " + and_list(others) +
                tail,
            observable_lines.front().text));
    }

    /* A LAMMPS variable a line without a --hotvar defines on LAMMPS'
     * command line (--invar, -var, -v), which its script reads and which
     * another line declares with --hotvar: the run injects that line's
     * value into this script as `variable <name> equal <value>` (a cold
     * script is injected the hot values it reads, script_reads_variable,
     * the test the run makes), and LAMMPS refuses it for a variable of
     * another style, so every evaluation would fail. Refused at the line
     * that defines it. (A --hotvar defined on its own line is refused by
     * LammpsSim.) */
    for (const auto& cold : LS) {
        if (cold.is_failprobe || !cold.hotvars.empty()) continue;
        for (const auto& v : cold.line_variables)
            for (const auto& hot : LS) {
                if (&hot == &cold) continue;
                for (const auto& hv : hot.hotvars) {
                    if (hv.name != v.name ||
                        !script_reads_variable(cold.script_lines, v.name))
                        continue;
                    throw std::runtime_error(at_line(cold.line,
                        "PC_LAMMPS: the script " + cold.script + " reads " +
                        v.name + ", the --hotvar of line " +
                        std::to_string(hot.line) + ", and this line defines " +
                        v.name + " (" + v.option + " " + v.name +
                        "): the run injects `variable " + v.name +
                        " equal <value>` into the script, which LAMMPS "
                        "refuses for a variable of another style",
                        cold.line_text));
                }
            }
    }

    /* One line declares each --hotvar name: an evaluation keeps one value
     * of each name (hot_values_for_iter, keyed by the name only) for the
     * scripts that read it, and the line whose script finished last would
     * set it -- whether the two lines run one script or two. Refused at the
     * later line (LS keeps the order of the file). */
    {
        std::unordered_map<std::string, std::size_t> hot_owner;
        for (std::size_t s = 0; s < LS.size(); ++s) {
            for (const auto& hv : LS[s].hotvars) {
                auto it = hot_owner.find(hv.name);
                if (it != hot_owner.end() && it->second != s) {
                    const LammpsSim& first = LS[it->second];
                    throw std::runtime_error(at_line(LS[s].line,
                        "PC_LAMMPS --hotvar " + hv.name + ": line " +
                            core::to_string(first.line) + " declares --hotvar " +
                            hv.name + " as well (script " + first.script +
                            "); one line declares each --hotvar name, as an "
                            "evaluation keeps one value of each name for the "
                            "scripts that read it, and the line that finished "
                            "last would set it",
                        LS[s].line_text));
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
    /* The refusals below name the line of the key they refuse (the
     * GRAD_STRATEGY or ALGO line of the block, OptimizerSpec::key_lines:
     * with_key_line), or without it the line the block closes at
     * (OptimizerSpec::block_line: ENDOPTIMIZER, ENDEXPLORE, ENDINNER). */
    auto reject_numerical_with_observable =
        [&](const OptimizerSpec& sp, const char* tag) {
            if (sp.grad_strategy
                && *sp.grad_strategy == GradStrategy::Numerical) {
                throw std::runtime_error(with_key_line(sp, "GRAD_STRATEGY",
                    std::string("--hpotarget defines PC_OBSERVABLE but ")
                    + tag + " GRAD_STRATEGY=NUMERICAL is set. PC_OBSERVABLE "
                    "is built for the analytical gradient: use GRAD_STRATEGY "
                    "ANALYTICAL (an unset GRAD_STRATEGY resolves to it in a "
                    "block that obtains gradients; it needs the settings of "
                    "the analytical gradient, such as OALGO 4 with a fixed "
                    "LAMBDA >= 0, and startup names a missing one) or "
                    "GRAD_STRATEGY HYBRID, which uses finite differences "
                    "where the analytical gradient is unavailable."));
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
     * local search obtains them, whatever the D or N of its name) -- the
     * STRATEGY-level block and EXPLORE.
     * grad_strategy_resolved puts it in the journal. An INNER block runs
     * no optimiser of its own: MLSL and AUGLAG hand their local search the
     * gradient of their own objective, which the enclosing block's
     * GRAD_STRATEGY obtains (nlopt_adapter.h), so an INNER block takes no
     * GRAD_STRATEGY and is not resolved. The derivative-free blocks are
     * left alone too: they never ask for a gradient, and ANNEAL and ASA
     * reject any GRAD_STRATEGY. */
    if (!observables.empty()) {
        for (const auto& b : optimiser_blocks) {
            /* (an INNER block of another LIB than NLopt is refused at
             * startup, require_inner_backend: its enclosing block has no
             * local search to resolve a GRAD_STRATEGY for) */
            if (b.inner || b.sp->grad_strategy ||
                (b.sp->inner && b.sp->inner->lib != "NLOPT") ||
                !obtains_gradients(*b.sp))
                continue;
            b.sp->grad_strategy = GradStrategy::Analytical;
            b.sp->grad_strategy_resolved = true;
        }
    }

    /* The seed of a PRESTAGE exploration that sets none is given once the
     * search space is bound (HPOTargets::HPOTargets): whether its optimiser
     * reads SEED can depend on its dimensions (reads_seed, phase_specs.h). */

    /* PC_LAMMPS: no gradients.
     *
     * The HPO-side analytical-gradient wiring for PC_LAMMPS (the
     * IFT-adjoint chain) is not engaged. PC_LAMMPS configs must
     * use gradient-free optimisers -- every registered optimiser whose run
     * takes no gradient (searches_without_gradient(), strategy_rules.h).
     * DLIB:BFGS is refused as every gradient-based optimiser is: it takes
     * the central-difference gradient of the loss (GradientProvider,
     * GRAD_STRATEGY NUMERICAL), as an NLopt LD_* algorithm would, across
     * the relaxations of the scripts.
     *
     * A block is refused when it asks the host for gradients
     * (obtains_gradients), not for the D of its name: NLopt's MLSL and
     * AUGLAG evaluate their own points without a gradient, and their local
     * search decides (mlsl.c, auglag.c). With an INNER block that block
     * decides, and is checked as a block of its own (its message names
     * its optimiser); without one, NLopt's default local optimiser
     * decides -- LD_MMA for GD_MLSL, GD_MLSL_LDS, LD_AUGLAG and
     * LD_AUGLAG_EQ -- and the message names it. GD_MLSL* and LD_AUGLAG*
     * with a derivative-free INNER block used to be refused for their D
     * although they take no gradient.
     *
     * An INNER block takes a local NLopt algorithm (require_inner_backend(),
     * at startup: no GN_*, TADAH or DLIB block), so its refusal names the
     * derivative-free local NLopt algorithms, every one, from the registry
     * (nlopt_inner_searches_without_gradient()), and an INNER block with an
     * MLSL or AUGLAG has no INNER block of its own to replace its default
     * local optimiser with.
     *
     * Also reject any explicit GRAD_STRATEGY={NUMERICAL,HYBRID,
     * ANALYTICAL} when PC_LAMMPS is present -- these tags only make
     * sense with PC_OBSERVABLE.
     *
     * A PC_FAILPROBE line runs a LAMMPS script too (LS holds it), so the
     * rules hold for a file whose only script line is the probe; the
     * refusals name the line the file has, PC_LAMMPS when it has one. */
    bool has_pc_lammps = false;
    for (std::size_t i = 0; i < LS.size(); ++i)
        if (static_cast<int>(i) != failprobe_idx)
            has_pc_lammps = true;
    /* The script line a refusal names: the first PC_LAMMPS line, else the
     * PC_FAILPROBE line, with its line number when the parser read it
     * ("PC_LAMMPS on line 12"). */
    std::string script_line = has_pc_lammps ? "PC_LAMMPS" : "PC_FAILPROBE";
    for (std::size_t i = 0; i < LS.size(); ++i) {
        if (has_pc_lammps == (static_cast<int>(i) == failprobe_idx)) continue;
        if (LS[i].line > 0)
            script_line += " on line " + std::to_string(LS[i].line);
        break;
    }
    auto reject_pc_lammps_grad =
        [&](const OptimizerSpec& sp, const char* tag, bool inner) {
            if (sp.grad_strategy) {
                throw std::runtime_error(with_key_line(sp, "GRAD_STRATEGY",
                    "--hpotarget defines " + script_line + " but " + tag
                    + " GRAD_STRATEGY is set. A PC_LAMMPS or PC_FAILPROBE "
                    "line does not support analytical/numerical/hybrid "
                    "gradients. Drop the GRAD_STRATEGY line."));
            }
            /* read() has required LIB and ALGO of every block by now, the
             * EXPLORE and INNER blocks included (require_driver,
             * require_inner_driver). */
            /* An INNER block of another LIB than NLopt is refused at startup
             * as not NLopt (require_inner_backend), whatever it names; its
             * enclosing MLSL or AUGLAG has no local search to ask about. */
            if (inner && sp.lib != "NLOPT") return;
            if (sp.inner && sp.inner->lib != "NLOPT") return;
            if (!obtains_gradients(sp)) return;
            const std::string requires_free = inner
                ? " A PC_LAMMPS or PC_FAILPROBE line requires a "
                  "gradient-free optimiser, and an INNER search is a local "
                  "NLopt algorithm; the INNER block takes a derivative-free "
                  "one (" + nlopt_inner_searches_without_gradient() + ")."
                : " A PC_LAMMPS or PC_FAILPROBE line requires a "
                  "gradient-free optimiser (" +
                      searches_without_gradient() + "); an NLopt MLSL or "
                      "AUGLAG algorithm is accepted with a derivative-free "
                      "INNER block.";
            if (optimizer_caps(sp).accepts_inner) {
                /* An MLSL or AUGLAG that takes a gradient for its local
                 * search: an INNER block's is refused at that block. */
                if (sp.inner) return;
                /* obtains_gradients: there is a default local search. */
                const OptimizerSpec local = inner_run_spec(sp).value();
                throw std::runtime_error(with_key_line(sp, "ALGO",
                    "--hpotarget defines " + script_line + " but " + tag
                    + " optimiser " + sp.lib + ":" + sp.algo +
                    " takes a gradient: NLopt's MLSL and AUGLAG take one for "
                    "their local search, and NLopt's default local "
                    "optimiser, " + local.lib + ":" + local.algo +
                    ", is gradient-based" +
                    (inner ? std::string(".")
                           : std::string("; an INNER block with a "
                                         "derivative-free one takes "
                                         "none.")) +
                    requires_free));
            }
            throw std::runtime_error(with_key_line(sp, "ALGO",
                "--hpotarget defines " + script_line + " but " + tag
                + " optimiser " + sp.lib + ":" + sp.algo +
                " is gradient-based." + requires_free));
        };
    if (!LS.empty())
        for (const auto& b : optimiser_blocks)
            reject_pc_lammps_grad(*b.sp, b.tag, b.inner);
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
  if (use_ersq && skip_ersq.rel_above) return "PC_ERSQ" + rel;
  return "";
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

void HPOTargets::check_archive_directories() const {
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
          config_line_error(self.line.number, e.what(), self.line.text));
    }
  }
}

void HPOTargets::check_archive_directory(const std::string& key,
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
                 "another name keeps this seed.";
      std::cerr << "[HPO] WARNING: " << w << "\n";
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
    std::cout << "Directory created: " << dir << std::endl;
    return;
  }
  if (!ec && fs::is_directory(dir, ec)) {
    std::cout << "Directory already exists: " << dir << std::endl;
    return;
  }
  throw std::runtime_error(
      key + " directory '" + dir + "' could not be created: " +
      (ec ? ec.message() : std::string("it exists and is not a directory")));
}

/* Positive-only OPTIM keys, from the same table the refit hpotarget writer
 * uses (MLIP hpotarget_writer.h) so HPO and the writer cannot disagree;
 * keys unknown to that table are sign-free. */
bool HPOTargets::optim_key_positive_only(const std::string& key) {
  for (const auto& g : mlip::engines::hpotarget_group_specs())
    if (key == g.key) return g.positive_only;
  return false;
}

std::string HPOTargets::optim_fitted_key_error(const std::string& key) {
  const std::string k = core::to_upper(key);
  if (k == "WEIGHTS")
    return "OPTIM WEIGHTS: WEIGHTS holds the coefficients each evaluation "
           "fits, so a searched value would be replaced by the fit; FIXINDEX "
           "and FIXWEIGHT in the training configuration hold coefficients "
           "fixed, and OPTIM FIXWEIGHT (<coefficient>) <low> <high> searches "
           "them";
  if (k == "EMBF0")
    return "OPTIM EMBF0: EMBF0 (ZERO or FREE) chooses the embedding "
           "reference of a DM_REAM many-body term, a choice and not a number "
           "to search; the training configuration sets it";
  return "";
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
    {"FIXINDEX", "FIXINDEX names the coefficients FIXWEIGHT holds fixed "
                 "(OPTIM FIXWEIGHT searches their values)"},
    {"MODEL",    "MODEL names the model and its basis function or kernel "
                 "(and an integer option)"},
  };
  const std::string k = core::to_upper(key);
  for (const auto& e : structure)
    if (k == e.key)
      return "OPTIM " + k + " is not a valid search target: " + e.what +
             ", structure data rather than a continuous parameter; the "
             "training configuration sets it.";
  return "";
}

std::string HPOTargets::optim_discrete_index_error(const std::string& key,
                                                   std::size_t index) {
  if (core::to_upper(key) == "OALGO" && index == 1)
    return "OPTIM OALGO (1) is not a valid search target: the first OALGO "
           "value selects the training algorithm (1 GELSD, 2 GELS, 3 SVD, "
           "4 Cholesky), a discrete choice rather than a continuous "
           "parameter; the training configuration sets it. The second value, "
           "the rcond M of options 1 and 3, is continuous: OPTIM OALGO (2) "
           "<low> <high>.";
  return "";
}

std::string HPOTargets::optim_component_key_error(const core::Context& ctx,
                                                  const std::string& key) {
  static const struct { const char* key; bool density; const char* field; }
      legacy[] = {{"CGRIDMB", true, "CENTRES"},   {"SGRIDMB", true, ""},
                  {"AMPGRIDMB", true, "AMPLITUDES"},
                  {"CEMBFUNC", false, "CENTRES"}, {"SEMBFUNC", false, ""},
                  {"RCUTMB", true, "CUTOFF.RADIUS"}};
  const std::string k = core::to_upper(key);
  const auto* row = std::find_if(std::begin(legacy), std::end(legacy),
                                 [&](const auto& r) { return k == r.key; });
  if (row == std::end(legacy) || ctx.has_stored(k)) return "";
  const auto specs = models::embed_specs(ctx);
  if (specs.empty()) return "";
  std::string paths;
  for (const auto& s : specs) {
    const auto& c = row->density ? s.density : s.embedding;
    std::string field = row->field;
    if (field.empty())  // a width: the coordinate the block declares
      field = c.basis == models::EmbedBasis::Knot5
                  ? ""
                  : c.coordinates == models::EmbedCoordinates::Widths ? "WIDTHS"
                                                                      : "ETA";
    const std::string id = core::to_upper(s.id);
    const std::string channel = row->density ? ".DENSITY." : ".EMBEDDING.";
    paths += (paths.empty() ? "" : ", ") +
             (field.empty()
                  ? id + channel + "* (a Knot5 component has no width)"
                  : id + channel + field +
                        (field == "CUTOFF.RADIUS" && s.cutoff == "NONE"
                             ? " (a parameter when the CUTOFF TYPE is not "
                               "NONE)"
                             : ""));
  }
  return "OPTIM " + k + ": the potential has no " + k + " key: its many-body "
         "term is DM_Embed component(s) (DESCRIPTOR blocks), whose " +
         (row->density ? "density" : "embedding") + " parameter is the named "
         "path " + paths + "; an OPTIM line names the path, OPTIM <path> "
         "(indices) <low> <high> (the starter target file of tadah refit "
         "names the paths; a target file written for a DM_REAM many-body "
         "term names " + k + ")";
}

std::string HPOTargets::optim_internal_key_error(const std::string& key) {
  const std::string k = core::to_upper(key);
  if (core::BlueprintParser::getInternalKeysMapRef().count(k) == 0) return "";
  return "OPTIM " + k + " is not a valid search target: " + k +
         " is an internal key, a value tadah computes itself (no "
         "configuration sets it, so no evaluation can take a candidate "
         "value of it).";
}

std::string HPOTargets::optim_startup_only_key_error(const std::string& key) {
  static const char* const once[] = {
      "LSCALE",  "ESHIFT",  "ESHIFT_ATOM",   "ESHIFT_DBATOM",
      "EFILTER", "FFILTER", "ZERO_COM_FORCE"};
  const std::string k = core::to_upper(key);
  for (const char* o : once)
    if (k == o)
      return "OPTIM " + k + " is not a valid search target: " + k +
             " prepares the data and is applied once, when the run starts "
             "and reads them, so no evaluation would take a candidate value "
             "of it: its value is the training configuration's.";
  return "";
}

std::optional<HPOTargets::FitValueRule> HPOTargets::optim_fit_value_rule(
    const std::string& key, std::size_t index) {
  const std::string k = core::to_upper(key);
  if (k == "LAMBDA")
    return FitValueRule{false,
                        "LAMBDA is the ridge penalty, a number >= 0: every "
                        "fit refuses a negative value other than -1, and "
                        "-1, which requests the evidence approximation, is "
                        "no value of a box"};
  if (k == "ESIGMA")
    return FitValueRule{true,
                        "ESIGMA is the energy error scale, a number > 0 "
                        "(meV/atom, or meV per sqrt(atom) under ENORM "
                        "SQRT), and every fit refuses another"};
  if (k == "WDBFILE")
    return FitValueRule{true,
                        "a searched WDBFILE value is the priority of a "
                        "dataset in the fit, a number > 0 (WDBFILE 0 in the "
                        "training configuration leaves a dataset out of the "
                        "fit)"};
  if (k == "FSIGMA" || k == "SSIGMA") {
    const bool force = k == "FSIGMA";
    if (index == 1)
      return FitValueRule{
          true, "the absolute part a of " + k + " is the " +
                    (force ? std::string("force") : std::string("stress")) +
                    " error scale of a structure whose " +
                    (force ? "forces are zero, a number > 0 (eV/A)"
                           : "stresses are zero, a number > 0 (GPa)") +
                    ", and every fit refuses another"};
    if (index == 2)
      return FitValueRule{false, "the relative part b of " + k +
                                     " is a number >= 0, and every fit "
                                     "refuses a negative one"};
  }
  return std::nullopt;
}

std::string HPOTargets::optim_switch_key_error(const std::string& key) {
  const std::string k = core::to_upper(key);
  if (k == "ENORM")
    return "OPTIM ENORM is not a valid search target: ENORM (ATOM or SQRT) "
           "chooses how the residual of an energy is normalised, a choice "
           "rather than a continuous parameter; the training configuration "
           "sets it.";
  if (k == "ENERGY")
    return "OPTIM ENERGY is not a valid search target: ENERGY (true or "
           "false) says whether the fit has the energies, a switch rather "
           "than a continuous parameter; the training configuration sets it.";
  return "";
}

std::string HPOTargets::optim_retired_key_error(const std::string& key) {
  /* The keys the blueprint marks retired (BlueprintKey::isRetired: the
   * reason, the replacement and the conversion of each are its text); a
   * test holds this list to the blueprint's. */
  static const char* const retired[] = {"EWEIGHT",      "FWEIGHT",
                                        "SWEIGHT",      "WDBFILE_AUTO",
                                        "EWEIGHT_TEMP", "UNCERTAINTY",
                                        "WCOV",         "WNOISE",
                                        "ERROR"};
  const std::string k = core::to_upper(key);
  for (const char* r : retired) {
    if (k != r) continue;
    const auto& info = core::BlueprintParser::getKeysMapRef().at(k);
    if (!info.isRetired())
      throw std::logic_error("OPTIM " + k + ": the blueprint does not mark "
                             "the key retired");
    return "OPTIM " + k + " is not a valid search target: " +
           core::retired_key_text(k, info);
  }
  return "";
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

std::string HPOTargets::optim_integer_index_error(
    const std::string& key, const std::vector<std::string>& config,
    std::size_t pos, std::size_t index) {
  const std::string k = core::to_upper(key);
  const auto& keys = core::BlueprintParser::getKeysMapRef();
  const auto it = keys.find(k);
  if (it == keys.end() || it->second.value_type.empty() || pos >= config.size())
    return "";
  /* The key's tokens as they are match one of its formats, but not with
   * the value at pos replaced by a fraction: no format takes a fraction
   * there. */
  const std::vector<std::string>& formats = it->second.value_type;
  std::vector<std::string> fraction = config;
  fraction[pos] = "0.5";
  if (!core::BlueprintValidator::validateOnce(formats, core::join(config)) ||
      core::BlueprintValidator::validateOnce(formats, core::join(fraction)))
    return "";
  std::string takes;
  for (const std::string& f : formats)
    takes += (takes.empty() ? "" : " or ") + f;
  return "OPTIM " + k + " (" + core::to_string(index) +
         ") is not a valid search target: value " + core::to_string(index) +
         " of " + k + " is an integer (" + k + " takes " + takes +
         "), a discrete value rather than a continuous parameter; the "
         "training configuration sets it.";
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
    std::ofstream tsv(dir + "/" + kSearchSpaceFile);
    tsv <<
      "# tadah hpo search space -- one row per optimised dimension, in optimiser order\n"
      "# (the same order as the params.tadah columns and the journal tables).\n"
      "#   dim          1-based position in the parameter vector\n"
      "#   label        KEY[n]: n is the OPTIM index (1-based; for FIXWEIGHT the model weight\n"
      "#                named in FIXINDEX); single-value keys other than FIXWEIGHT carry no index\n"
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
  if (!writes_resolved_optim_file()) return;
  std::ofstream hpo(dir + "/" + kResolvedOptimFile);
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
  spec.failscore_grading = on;
  for (OptimizerSpec* sub : {spec.inner.get(), spec.prestage.get()}) {
    if (!sub) continue;
    sub->failscore_grading = on;
    if (sub->inner) sub->inner->failscore_grading = on;
  }
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

  /* WEIGHTS, which each evaluation fits, and EMBF0, a choice; before
   * context.size(k) below, which would materialise the blueprint default
   * EMBF0 ZERO into the run context. */
  if (const std::string e = optim_fitted_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* Structure keys, also before context.size(k): RHOFORM has no blueprint
   * default, so size() would log an ERROR and throw a generic message. */
  if (const std::string e = optim_structure_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* Internal keys, before context.size(k), which would materialise the
   * blueprint default into the run context. */
  if (const std::string e = optim_internal_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* A retired key (the context refuses to read one) and a choice or a
   * switch of the fit, before context.size(k). */
  if (const std::string e = optim_retired_key_error(k); !e.empty())
    throw std::runtime_error(e);
  if (const std::string e = optim_switch_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* A key applied once, when the run starts and reads its data. */
  if (const std::string e = optim_startup_only_key_error(k); !e.empty())
    throw std::runtime_error(e);
  /* A legacy many-body key of a DM_Embed potential, before context.size(k),
   * which refuses an absent key without naming the component path. */
  if (const std::string e = optim_component_key_error(context, k); !e.empty())
    throw std::runtime_error(e);

  const bool named=k.find('.')!=std::string::npos;
  /* OPTIM FIXWEIGHT searches the FIXWEIGHT values, the coefficients FIXINDEX
   * names, and neither key has a default: both are in the training
   * configuration, checked before context.size(k), which would log an
   * ERROR line and refuse an absent key without saying what it is for. */
  if (k == "FIXWEIGHT") {
    const bool has_index = context.has_stored("FIXINDEX");
    const bool has_values = context.has_stored("FIXWEIGHT");
    if (!has_index && !has_values)
      throw std::runtime_error(
          "OPTIM FIXWEIGHT requires FIXINDEX and FIXWEIGHT in the training "
          "configuration (FIXINDEX names the coefficients the FIXWEIGHT "
          "values hold, and OPTIM FIXWEIGHT searches those values).");
    if (!has_values)
      throw std::runtime_error(
          "OPTIM FIXWEIGHT requires FIXWEIGHT in the training configuration "
          "(the FIXWEIGHT values hold the coefficients FIXINDEX names, and "
          "OPTIM FIXWEIGHT searches those values).");
    if (!has_index)
      throw std::runtime_error(
          "OPTIM FIXWEIGHT requires FIXINDEX in the training configuration "
          "(FIXINDEX names the coefficients the FIXWEIGHT values hold).");
  }
  /* A key of the training configuration, and one the configuration gives
   * (a line or a default), before context.size(k), which logs an ERROR line
   * and refuses either without saying what OPTIM needs. contains() inserts
   * the blueprint default, as size() does. A path of a DM_Embed parameter
   * is checked by embed_binding below. */
  if (!named) {
    const auto& keys = core::BlueprintParser::getKeysMapRef();
    if (keys.find(k) == keys.end()) {
      const std::string closest = core::BlueprintParser::findClosestKey(k, keys);
      throw std::runtime_error(
          "OPTIM " + k + ": " + k + " is not a key of the training "
          "configuration" +
          (closest.empty() ? std::string() : " (closest key: " + closest + ")"));
    }
    if (!context.contains(k))
      throw std::runtime_error(
          "OPTIM " + k + ": the training configuration has no " + k +
          " line, and OPTIM " + k + " searches the values the training "
          "configuration gives");
  }

  Opti& omap = optimap[k];
  std::optional<models::EmbedParameterBinding> binding;
  if(named) binding=models::embed_binding(context,k);
  size_t s = named?binding->values.size():context.size(k);
  omap.resize(s);

  std::vector<std::string> tokens = parse_optim_line(iss, "OPTIM " + k);

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
  bool tag_written = false;   // the line writes LIN, LOG or ASINH <c>
  std::string tag_c_tok;      // the c of ASINH <c>, as written
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
      tag_c_tok = toks.back();
      toks.pop_back(); toks.pop_back();
      tag_written = true;
      return;
    }
    if (!toks.empty()) {
      const std::string tag = upper(toks.back());
      if (tag == "LIN") {
        tf.kind = ScaleKind::Linear; toks.pop_back(); tag_written = true;
      } else if (tag == "LOG") {
        tf.kind = ScaleKind::Log; toks.pop_back(); tag_written = true;
      }
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
  const bool named_amp=named && k.size()>=11 && k.compare(k.size()-11,11,".AMPLITUDES")==0;
  const bool nonneg_amp = named ? named_amp && binding->positive :
      models::optim_key_lower_bound(k, rho_form_) == models::OptimLowerBound::NonNegative;
  auto low_rule = named ? std::optional<models::OptimLowBoundRule>{} : models::optim_low_bound_rule(k, context);
  if(named && binding->positive)
    low_rule=models::OptimLowBoundRule{!named_amp,named_amp?"nonnegative amplitudes":"a positive scale or radius",k};
  /* What makes the amplitudes of a non-negative amplitude line >= 0: the
   * RHOFORM line of a DM_REAM density, or the FORM of the DENSITY block of
   * the DM_Embed component a path names (each component has its own). */
  auto form_label = [&]() -> std::string {
    if (named) {
      for (const auto& es : models::embed_specs(context))
        if (core::to_upper(es.id) == core::to_upper(binding->owner))
          return std::string("FORM ") + models::rhoform_name(es.form) +
                 " of the DENSITY of DESCRIPTOR " + es.id;
    } else if (rho_form_) {
      return std::string("RHOFORM ") + models::rhoform_name(*rho_form_);
    }
    throw std::logic_error("OPTIM " + k + ": a non-negative amplitude line "
                           "without a density form");
  };

  const bool auto_box = !tokens.empty() && upper(tokens.front()) == "AUTO";
  double low = 0.0, high = 0.0;
  std::string low_tok, high_tok;  // the explicit bounds as written
  struct AutoRule { bool geometric = false; double factor = 0.0; double floor = 0.0; double group_floor = 0.0; } rule;
  std::string factor_tok;   // the RATIO or FRAC value as written
  std::string group_floor_tok;
  bool floor_set = false;   // FLOOR or GROUP_FLOOR given
  bool floor_seen = false, group_floor_seen = false;
  bool ratio_seen = false, frac_seen = false;
  bool mode_set = false;    // RATIO or FRAC given
  if (auto_box) {
    tokens.erase(tokens.begin());
    take_tag(tokens);
    /* The name of each option first: a word that is no AUTO option is
     * named as such whatever follows it. */
    for (std::size_t t = 0; t < tokens.size(); t += 2) {
      const std::string opt = upper(tokens[t]);
      if (opt != "RATIO" && opt != "FRAC" && opt != "FLOOR" &&
          opt != "GROUP_FLOOR")
        throw std::runtime_error("OPTIM " + k + ": '" + tokens[t] + "' is not "
                                 "an AUTO option (RATIO, FRAC, FLOOR or "
                                 "GROUP_FLOOR)" + grammar);
      if (t + 1 >= tokens.size())
        throw std::runtime_error("OPTIM " + k + ": AUTO option '" + tokens[t] +
                                 "' needs a value" + grammar);
      /* Each is given once: a second replaced the first's value. */
      bool& seen = opt == "RATIO" ? ratio_seen
                 : opt == "FRAC"  ? frac_seen
                 : opt == "FLOOR" ? floor_seen
                                  : group_floor_seen;
      if (seen)
        throw std::runtime_error("OPTIM " + k + ": AUTO " + opt + " is given "
                                 "twice on this line; each option is given once");
      seen = true;
      if ((opt == "RATIO" || opt == "FRAC") && mode_set)
        throw std::runtime_error("OPTIM " + k + ": AUTO takes RATIO or FRAC, "
                                 "not both");
      const double v = parse_num(tokens[t + 1], ("AUTO " + opt).c_str());
      if (opt == "RATIO") {
        if (!(v > 1.0)) throw std::runtime_error("OPTIM " + k + ": AUTO RATIO must be > 1 (got " + tokens[t + 1] + ")");
        rule.geometric = true; rule.factor = v; mode_set = true;
        factor_tok = tokens[t + 1];
      } else if (opt == "FRAC") {
        if (!(v >= 0.0))
          throw std::runtime_error("OPTIM " + k + ": AUTO FRAC must be >= 0 (got " +
                                   tokens[t + 1] + "); FRAC 0 with FLOOR or "
                                   "GROUP_FLOOR gives a constant half-width");
        rule.geometric = false; rule.factor = v; mode_set = true;
        factor_tok = tokens[t + 1];
      } else {
        if (v < 0.0)
          throw std::runtime_error("OPTIM " + k + ": AUTO " + opt +
                                   " must be >= 0 (got " + tokens[t + 1] + ")");
        if (opt == "FLOOR") {
          rule.floor = v;
        } else {
          rule.group_floor = v;
          group_floor_tok = tokens[t + 1];
        }
        floor_set = true;
      }
    }
    if (k.rfind("RCUT", 0) == 0)
      throw std::runtime_error("OPTIM " + k + ": a cutoff key takes explicit "
                               "bounds, OPTIM " + k + " (indices) <low> <high>, "
                               "and no AUTO box: the neighbour list is sized "
                               "from the explicit high bound");
    if (!mode_set) {
      rule.geometric = named ? binding->positive && !named_amp : optim_key_positive_only(k);
      rule.factor    = rule.geometric ? 2.0 : 0.5;
    }
    /* FLOOR and GROUP_FLOOR bound the half-width of the linear rule; the
     * geometric box [x0/f, x0*f] has none, so they would be ignored. A
     * value the fit takes only > 0 (ESIGMA, FSIGMA (1), SSIGMA (1),
     * WDBFILE: optim_fit_value_rule) has a geometric box by default, the
     * relative part FSIGMA (2) or SSIGMA (2) a linear one: the default is
     * that of each index of the line. */
    std::string geometric_default;  // the key (and index) of such a box
    if (mode_set) {
      if (rule.geometric) geometric_default = k;
    } else if (rule.geometric) {
      geometric_default = k;
    } else if (!named) {
      for (const std::size_t ui : user_indices) {
        const auto v = optim_fit_value_rule(k, ui);
        if (!v || !v->strict) continue;
        geometric_default =
            (k == "FSIGMA" || k == "SSIGMA") ? k + " (" + core::to_string(ui) + ")"
                                             : k;
        break;
      }
    }
    if (!geometric_default.empty() && floor_set)
      throw std::runtime_error(
          "OPTIM " + geometric_default + ": FLOOR and GROUP_FLOOR set the smallest "
          "half-width of an AUTO FRAC box; " +
          (mode_set ? "AUTO RATIO " + factor_tok + " makes the geometric "
                      "box [seed/" + factor_tok + ", seed*" + factor_tok +
                      "], which they do not change, so they are given with "
                      "FRAC <p> only"
                    : "the AUTO box of " + geometric_default + " is geometric by default "
                      "(RATIO 2), which they do not change, so they are "
                      "given with FRAC <p> only"));
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
    low_tok = tokens[0];
    high_tok = tokens[1];
    /* An empty or reversed box, at the line and with the bounds as
     * written. */
    if (!(high > low))
      throw std::runtime_error("OPTIM " + k + ": <high> " + high_tok +
                               " is not greater than <low> " + low_tok);
  }

  /* The indices as the line wrote them (for FIXWEIGHT, model weights). */
  const std::vector<size_t> typed_indices = user_indices;
  if (k == "FIXWEIGHT") {
    /* OPTIM FIXWEIGHT indices name coefficients (those FIXINDEX lists),
     * not positions in the FIXWEIGHT value list. Map each user index to
     * its FIXWEIGHT slot through FIXINDEX; declaration order defines the
     * pairing. Back-compatible with `tadah refit` targets, which list
     * FIXINDEX 1-K so the coefficient index == slot. (The training
     * configuration has FIXINDEX and FIXWEIGHT: checked above.) */
    /* core::parse_fixindex, as the trainers read FIXINDEX; the weight
     * count is not known here (startup validation checks the bounds). */
    std::vector<std::string> fi_str(context.size("FIXINDEX"));
    context.get("FIXINDEX", fi_str);
    const std::vector<size_t> fixidx =
        core::parse_fixindex(context, std::nullopt);
    for (auto& ui : user_indices) {
      auto it = std::find(fixidx.begin(), fixidx.end(), ui);
      if (it == fixidx.end()) {
        throw std::runtime_error(
          "OPTIM FIXWEIGHT (" + core::to_string(ui) + "): coefficient " +
          core::to_string(ui) + " is not in FIXINDEX (" + core::join(fi_str) +
          "); the indices of OPTIM FIXWEIGHT name coefficients, those "
          "FIXINDEX lists, not positions among the FIXWEIGHT values");
      }
      ui = static_cast<size_t>(it - fixidx.begin()) + 1;  // 1-based slot
    }
  }

  if(named) {
    omap.config.clear();
    for(double v:binding->values) omap.config.push_back(core::format_roundtrip(v));
  } else context.get<std::vector<std::string>>(k, omap.config);

  /* LAMBDA -1 requests the evidence approximation, which finds the penalty
   * in each fit: a searched LAMBDA would replace it, and the starting
   * point -1 is no value of a box. */
  if (k == "LAMBDA" && !omap.config.empty() &&
      core::is_number(omap.config[0]) &&
      core::to_double(omap.config[0]) == -1.0)
    throw std::runtime_error(
        "OPTIM LAMBDA: the training configuration has LAMBDA -1, which "
        "requests the evidence approximation: each fit then finds the "
        "penalty itself, and OPTIM LAMBDA would replace it with the values "
        "of the box. LAMBDA >= 0 in the training configuration is the "
        "starting point of a search of LAMBDA; without OPTIM LAMBDA the "
        "evidence approximation finds it in every evaluation.");

  /* The checks a box [lo_b, hi_b] of the index @p ui (a FIXWEIGHT slot)
   * meets below, past its AUTO derivation: what the fit takes of the value
   * (optim_fit_value_rule), the rank threshold of OALGO and the low-bound
   * rule of the key. A hint of a message names a box only when the line's
   * own checks take it. */
  /* The low-bound rule of the index @p ui: on a D2_mJoin line, that of the
   * descriptor whose entry it is (a D2_BP's width has none). */
  auto entry_low_rule = [&](std::size_t ui) {
    if (!named && low_rule)
      return models::optim_low_bound_rule(k, context, ui - 1);
    return low_rule;
  };
  auto box_passes = [&](double lo_b, double hi_b, std::size_t ui) {
    const std::optional<FitValueRule> vr =
        named ? std::nullopt : optim_fit_value_rule(k, ui);
    if (vr && vr->strict && !(lo_b > 0.0)) return false;
    if (vr && !vr->strict && lo_b < 0.0) return false;
    if (!named && k == "OALGO" && ui == 2 && !(lo_b > 0.0 && hi_b < 1.0))
      return false;
    if (const auto r = entry_low_rule(ui); r && !r->admits(lo_b))
      return false;
    return true;
  };
  /* An AUTO FRAC box of the value: clamped at 0 for a non-negative
   * amplitude and for a value the fit takes >= 0 other than LAMBDA. */
  auto frac_clamped = [&](const std::optional<FitValueRule>& vr) {
    return nonneg_amp || (vr && !vr->strict && k != "LAMBDA");
  };
  /* Whether an AUTO FRAC box with a FLOOR above 0 around the seed x0 of
   * the index passes the line's checks. */
  auto floor_box_passes = [&](double x0, std::size_t ui,
                              const std::optional<FitValueRule>& vr) {
    const double a = 0.5;  // a FLOOR above 0
    double lo_b = x0 - a;
    if (frac_clamped(vr) && lo_b < 0.0) lo_b = 0.0;
    return box_passes(lo_b, x0 + a, ui);
  };
  /* The explicit bounds that give the value of index @p ui (written
   * @p typed) a box the line's checks take: from 0 when a box from 0
   * passes, above 0 otherwise. */
  auto explicit_hint = [&](std::size_t ui, std::size_t typed) {
    const std::string line = "OPTIM " + k + " (" + core::to_string(typed) + ")";
    if (box_passes(0.0, 0.5, ui))
      return "; the explicit bounds `" + line + " 0 <high>` give it a box";
    return "; the explicit bounds `" + line + " <low> <high>` with <low> > 0 "
           "give it a box";
  };

  /* An explicit box tagged LOG reaches 0 or below, or one that ASINH <c>
   * maps to an empty or infinite range: refused at the line (SearchSpace
   * keeps both checks as its own invariants). */
  if (!auto_box) {
    if (tf.kind == ScaleKind::Log && !(low > 0.0)) {
      bool every_index_takes_it = true;
      for (const size_t ui : user_indices)
        every_index_takes_it = every_index_takes_it && box_passes(low, high, ui);
      throw std::runtime_error(
          "OPTIM " + k + ": the line is tagged LOG, and <low> " + low_tok +
          " is not > 0, and a box tagged LOG lies above 0" +
          (every_index_takes_it
               ? std::string("; a line without a tag, or tagged LIN or "
                             "ASINH <c>, takes this box")
               : std::string()));
    }
    if (tf.kind == ScaleKind::Asinh) {
      const double tlo = tf.fwd(low), thi = tf.fwd(high);
      if (!metric_value_is_finite(tlo) || !metric_value_is_finite(thi) ||
          !(thi > tlo))
        throw std::runtime_error(
            "OPTIM " + k + ": under ASINH " + tag_c_tok + " the box [" +
            low_tok + ", " + high_tok + "] maps to [" +
            core::format_roundtrip(tlo) + ", " + core::format_roundtrip(thi) +
            "], which has no finite width; a c of the size of the values of "
            "the box gives one");
    }
  }

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
    for (std::size_t j = 0; j < user_indices.size(); ++j) {
      const size_t i = ui_offset(omap.config, user_indices[j]);
      if (i >= s || !core::is_number(omap.config[i])) continue;  // reported below
      if (core::to_double(omap.config[i]) > 0.0) continue;
      throw std::runtime_error(
          "OPTIM " + k + " (" + core::to_string(typed_indices[j]) +
          "): AUTO RATIO under " + form_label() + " needs every seed on the "
          "line > 0 (got " + omap.config[i] + "); FRAC, the default, gives a "
          "box clamped at 0");
    }
  }

  /* The numeric values of the key, those its indices count. */
  std::size_t numeric_values = 0;
  for (const auto& t : omap.config)
    if (core::is_number(t)) ++numeric_values;

  for (std::size_t j = 0; j < user_indices.size(); ++j) {
    const size_t ui = user_indices[j];
    {
      const size_t i = ui_offset(omap.config, ui);
      if (const std::string e =
              optim_integer_index_error(k, omap.config, i, typed_indices[j]);
          !e.empty())
        throw std::runtime_error(e);
    }
    /* A refusal of this entry says why; the catch below names the entry
     * once, "OPTIM <KEY> (<i>): <why>". */
    try {
      size_t i = ui_offset(omap.config, ui);
      if (i>=s) {
        /* Values of a potential its descriptors do not read were left out
         * (D_mJoin::expand_grids): the indices address the values kept. */
        std::string why;
        bool left_out = false;
        for (const auto& lo : left_out_) {
          if (lo.key != k) continue;
          left_out = true;
          std::vector<std::string> out, rest = lo.kept;
          for (const auto& v : lo.given) {
            const auto it = std::find(rest.begin(), rest.end(), v);
            if (it != rest.end()) rest.erase(it);
            else out.push_back(v);
          }
          why = " (of the potential's " + k + " " + core::join(lo.given) +
                ", the descriptors do not read " + core::join(out) +
                (out.size() == 1 ? ", which was" : ", which were") +
                " left out, and OPTIM indices address the values kept, " +
                core::join(lo.kept);
          if (ui >= 1 && ui <= lo.given.size()) {
            const std::string& v = lo.given[ui - 1];
            const auto it = std::find(lo.kept.begin(), lo.kept.end(), v);
            if (it != lo.kept.end())
              why += "; the potential's value at index " + core::to_string(ui) +
                     ", " + v + ", is index " +
                     core::to_string(static_cast<std::size_t>(
                         it - lo.kept.begin()) + 1);
          }
          why += ")";
        }
        /* FSIGMA and SSIGMA of one value: the relative part b is index 2,
         * which the configuration holds once it gives both values. */
        if (why.empty() && (k == "FSIGMA" || k == "SSIGMA") && ui == 2 &&
            s == 1)
          why = " (its absolute part a; its relative part b, index 2, is "
                "searched when the training configuration gives both, " + k +
                " <a> <b>)";
        const std::string has =
            numeric_values == 1
                ? "1 value"
                : core::to_string(numeric_values) + " values";
        const std::string range =
            numeric_values == 1
                ? "the index of OPTIM " + k + " is 1"
                : "the indices of OPTIM " + k + " run from 1 to " +
                      core::to_string(numeric_values);
        /* A FIXWEIGHT slot is an entry of FIXINDEX, which
         * core::parse_fixindex pairs with the FIXWEIGHT values one to
         * one. */
        if (k == "FIXWEIGHT")
          throw std::logic_error(
              "OPTIM FIXWEIGHT: coefficient " +
              core::to_string(typed_indices[j]) + " is entry " +
              core::to_string(ui) + " of FIXINDEX, past the FIXWEIGHT "
              "values");
        if (numeric_values == 0)
          throw std::runtime_error(
              k + " has no numeric value in the training configuration, so "
              "no index of OPTIM " + k + " selects one");
        throw std::runtime_error(
            k + " has " + has +
            (left_out ? " that the descriptors read"
                      : " in the training configuration") +
            ", so " + range + why);
      }
      /* One OPTIM line per index: a second box for an index would replace
       * the first without a word. Lines of one key that select other
       * indices add to it (in any case, the key being upper-cased). */
      if (omap.opt[i])
        throw std::runtime_error(
            "the OPTIM " + k + " line on line " + core::to_string(omap.line[i]) +
            " selects this index as well; one OPTIM line selects each index "
            "(OPTIM keys match in any case)");
      double lo_i = low, hi_i = high;
      /* What the fit takes of this value (> 0, >= 0), whose box keeps to
       * it: the AUTO box of a value > 0 is geometric unless RATIO or FRAC
       * says otherwise, that of a value >= 0 linear and clamped at 0. */
      const std::optional<FitValueRule> value_rule =
          named ? std::nullopt : optim_fit_value_rule(k, ui);
      const bool geometric =
          mode_set ? rule.geometric
                   : rule.geometric || (value_rule && value_rule->strict);
      const double factor =
          mode_set ? rule.factor : (geometric ? 2.0 : 0.5);
      const std::string factor_text =
          mode_set ? factor_tok : core::format_roundtrip(factor);
      const std::size_t typed = typed_indices[j];
      if (auto_box) {
        /* A number: ui_offset gives the position of the ui-th number of
         * the key (ui >= 1, index 0 refused by the selector). */
        const double x0 = core::to_double(omap.config[i]);
        const std::string& seed = omap.config[i];
        if (nonneg_amp && x0 < 0.0)
          throw std::runtime_error("the seed " + seed + " is negative, but " +
                                   form_label() + " requires every AMPGRIDMB "
                                   "amplitude >= 0");
        /* A value every fit takes >= 0 (optim_fit_value_rule) is the seed
         * of its box: a negative one is refused as such, before a FRAC box
         * clamped at 0 would be reversed ([0, x0 + d] with x0 + d < 0) or
         * leave the seed out ([0, x0 + d] with x0 + d > 0). */
        if (value_rule && !value_rule->strict && x0 < 0.0)
          throw std::runtime_error("the seed " + seed + " is negative, but " +
                                   value_rule->why);
        if (geometric) {
          if (!(x0 != 0.0)) {
            /* The key, with the index for FSIGMA and SSIGMA, whose two
             * values have different defaults. */
            const std::string subject =
                (k == "FSIGMA" || k == "SSIGMA")
                    ? k + " (" + core::to_string(typed) + ")"
                    : k;
            const std::string head =
                mode_set
                    ? "AUTO RATIO " + factor_text + " gives the box [seed/" +
                          factor_text + ", seed*" + factor_text +
                          "], which needs a seed other than 0, and the seed "
                          "is 0"
                    : "AUTO without RATIO or FRAC gives " + subject +
                          " the box [seed/2, seed*2] (RATIO 2), which needs a "
                          "seed other than 0, and the seed is 0";
            throw std::runtime_error(
                head + (floor_box_passes(x0, ui, value_rule)
                            ? std::string("; AUTO FRAC <p> FLOOR <a> gives a "
                                          "box of half-width a or more "
                                          "around it")
                            : explicit_hint(ui, typed)));
          }
          const double a = x0 / factor, b = x0 * factor;
          lo_i = std::min(a, b); hi_i = std::max(a, b);
          omap.origin[i] = BoxOrigin::AutoGeometric;
          if (!(hi_i > lo_i))
            throw std::runtime_error(
                "the AUTO box [" + core::format_roundtrip(lo_i) + ", " +
                core::format_roundtrip(hi_i) + "] around the seed " + seed +
                " has zero width in double precision; a larger RATIO gives it "
                "a width");
        } else {
          const double d = std::max({factor * std::fabs(x0), rule.floor,
                                     rule.group_floor * group_max});
          /* d is 0 only when every seed of the line is 0: the group floor
           * 1e-3 is relative to the largest of them. */
          if (!(d > 0.0) && nonneg_amp && !floor_set)
            throw std::runtime_error(
                "every seed on this line is 0, so the AUTO box has zero "
                "width (FRAC and the GROUP_FLOOR 1e-3 that " + form_label() +
                " implies for " + k + " are relative to the seeds); AUTO "
                "FLOOR <a> with a > 0 gives the box [0, a], and the explicit "
                "bounds `OPTIM " + k + " (" + core::to_string(typed) +
                ") 0 <high>` the box [0, <high>]");
          if (!(d > 0.0)) {
            std::string head;
            std::string hint;
            if (group_max > 0.0) {
              head = "AUTO FRAC gives a zero-width box (the seed " + seed +
                     " times FRAC " + factor_text + " is 0)";
              hint = "; FLOOR <a> or GROUP_FLOOR <r> above 0 gives it a "
                     "smallest half-width";
            } else {
              head = "every seed on this line is 0, so AUTO FRAC gives a "
                     "zero-width box";
              if (rule.group_floor > 0.0)
                head += ", and GROUP_FLOOR " + group_floor_tok + ", a "
                        "fraction of the largest seed on the line, adds "
                        "nothing";
              hint = "; FLOOR <a> above 0 gives it the half-width a";
            }
            if (!floor_box_passes(x0, ui, value_rule))
              hint = explicit_hint(ui, typed);
            throw std::runtime_error(head + hint);
          }
          lo_i = x0 - d; hi_i = x0 + d;
          omap.origin[i] = BoxOrigin::AutoLinear;
          /* non-negative amplitudes: lo = max(0, x0 - d), before the LOG
           * check below (a LOG box cannot start at 0) */
          if (frac_clamped(value_rule)) {
            if (lo_i < 0.0) lo_i = 0.0;
            omap.origin[i] = BoxOrigin::AutoLinearNonneg;
          }
          /* A half-width below the spacing of the doubles at the seed
           * rounds the box to the seed (the seed of a clamped box is >= 0:
           * a negative one is refused above). */
          if (!(hi_i > lo_i))
            throw std::runtime_error(
                "the AUTO box around the seed " + seed + " has zero width in "
                "double precision (half-width " + core::format_roundtrip(d) +
                "); a larger FRAC or FLOOR gives it a width");
        }
        /* A box past the largest double (x0*f, x0 + d). */
        if (!metric_value_is_finite(lo_i) || !metric_value_is_finite(hi_i))
          throw std::runtime_error(
              "the AUTO box [" + core::format_roundtrip(lo_i) + ", " +
              core::format_roundtrip(hi_i) + "] around the seed " + seed +
              " is not finite; a smaller RATIO or FRAC gives a finite box");
        if (tf.kind == ScaleKind::Log && lo_i <= 0.0)
          throw std::runtime_error(
              "the line is tagged LOG, and the AUTO box [" +
              core::format_roundtrip(lo_i) + ", " +
              core::format_roundtrip(hi_i) + "] reaches 0 or below, and a box "
              "tagged LOG lies above 0" +
              (x0 > 0.0 ? std::string("; AUTO RATIO <f> gives a box above 0 "
                                      "around this seed")
                        : std::string()) +
              (box_passes(lo_i, hi_i, ui)
                   ? std::string("; a line without a tag, or tagged LIN or "
                                 "ASINH <c>, takes this box")
                   : std::string()));
      }
      /* A value of the fit a fit refuses at or below 0, or below 0
       * (optim_fit_value_rule): every such candidate would be refused by
       * its evaluation. */
      if (value_rule && value_rule->strict && !(lo_i > 0.0))
        throw std::runtime_error("the low bound " +
                                 core::format_roundtrip(lo_i) +
                                 " is not > 0, but " + value_rule->why +
                                 ", so the low bound of its box is > 0");
      if (value_rule && !value_rule->strict && lo_i < 0.0)
        throw std::runtime_error("the low bound " +
                                 core::format_roundtrip(lo_i) +
                                 " is negative, but " + value_rule->why +
                                 ", so the low bound of its box is >= 0");
      /* OALGO's second value, the rank threshold M of OALGO 1 and 3: only
       * 0 < M < 1 sets a threshold (below 0 OALGO 1 takes the machine
       * precision, at 0 or below OALGO 3 leaves out only a singular value
       * of 0, whatever M), and a fit refuses 1 or more. */
      if (!named && k == "OALGO" && ui == 2 && !(lo_i > 0.0 && hi_i < 1.0))
        throw std::runtime_error(
            "the box [" + core::format_roundtrip(lo_i) + ", " +
            core::format_roundtrip(hi_i) +
            "] is not inside (0, 1), but the second value of OALGO is the "
            "rank threshold M of OALGO 1 and 3, which only a value 0 < M < "
            "1 sets: every M below 0 is the machine precision under OALGO "
            "1, every M of 0 or below leaves out only a singular value of 0 "
            "under OALGO 3, and a fit refuses an M of 1 or more, so the box "
            "of a search is inside (0, 1)");
      /* The low-bound rule of this entry (entry_low_rule: on a D2_mJoin
       * line, that of the descriptor whose entry it is): an explicit low
       * bound as written, an AUTO box with both ends (the line wrote
       * neither). */
      if (const auto entry_rule = entry_low_rule(ui);
          entry_rule && !entry_rule->admits(lo_i)) {
        const std::string requires_ =
            entry_rule->by + " requires " + entry_rule->requirement;
        if (auto_box)
          throw std::runtime_error(
              "the AUTO box [" + core::format_roundtrip(lo_i) + ", " +
              core::format_roundtrip(hi_i) + "] has the low bound " +
              core::format_roundtrip(lo_i) + ", which " +
              (entry_rule->strict ? "is not > 0" : "is negative") + ", but " +
              requires_ + ", so the low bound of its box is " +
              (entry_rule->strict ? "> 0" : ">= 0"));
        throw std::runtime_error(
            "the low bound " + low_tok +
            (entry_rule->strict ? " is not > 0" : " is negative") + ", but " +
            requires_ + ", so the low bound of its box is " +
            (entry_rule->strict ? std::string("> 0")
                                : std::string(">= 0 (its AUTO FRAC box, the "
                                              "default, is clamped at 0)")));
      }
      omap.opt[i] = true;
      omap.line[i] = nline;
      omap.highbound[i] = hi_i;
      omap.lowbound[i] = lo_i;
      omap.scale[i] = tf;
      omap.tagged[i] = tag_written;
    } catch (const std::exception& e) {
      throw std::runtime_error("OPTIM " + k + " (" +
                               core::to_string(typed_indices[j]) + "): " +
                               e.what());
    }
  }
  /* Every box has high > low: an explicit box is checked at its bounds, an
   * AUTO box where it is derived. */
  for (size_t i = 0; i < omap.lowbound.size(); ++i)
    if (omap.opt[i] && !(omap.highbound[i] > omap.lowbound[i]))
      throw std::logic_error("OPTIM " + k + ": a box with high <= low passed "
                             "the checks of its line");
}
} // namespace hpo
} // namespace tadah

