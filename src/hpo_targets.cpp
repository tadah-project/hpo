#include <tadah/hpo/hpo_targets.h>
#include <tadah/hpo/observables/observable_parser.h>
#include <tadah/hpo/hpo_log.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/models/descriptors/d_base.h>
#include <tadah/models/descriptors/d_mjoin.h>
#include <tadah/core/registry.h>
#include <tadah/hpo/loss_function.h>

#include <algorithm>
#include <random>
#include <fstream>
#include <sstream>
#include <stdexcept>

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

std::vector<std::string> HPOTargets::parse_optim_line(std::istringstream& iss) {
  std::vector<std::string> elements;
  std::string token, enclosed;
  bool in_parens = false;
  bool has_parens = false;
  std::string processed_line;
  std::string indices = "";

  while (iss >> token) {
    processed_line += token + " ";

    if (token.front() == '(' || in_parens) {
      has_parens = true;
      if (token.front() == '(') {
        if (in_parens) {
          throw std::runtime_error(
            "Nested parentheses are not supported up to: " + processed_line);
        }
        in_parens = true;
        token.erase(token.begin());
      }
      size_t close_pos = token.find(')');
      if (close_pos != std::string::npos) {
        enclosed += token.substr(0, close_pos);
        in_parens = false;
        if (enclosed.empty()) {
          throw std::runtime_error(
            "Empty parentheses are not valid up to: " + processed_line);
        }
        indices += enclosed;
        enclosed.clear();
      } else {
        enclosed += token + " ";
      }
    } else {
      if (token.find(')') != std::string::npos && !in_parens) {
        throw std::runtime_error(
          "Unmatched closing parenthesis up to: " + processed_line);
      }
      elements.push_back(token);
    }
  }

  if (in_parens) {
    throw std::runtime_error(
      "Unmatched opening parenthesis up to: " + processed_line);
  }

  if (!has_parens) {
    throw std::runtime_error("No parentheses found up to: " + processed_line);
  }

  elements.push_back(indices);
  return elements;
}

void HPOTargets::Opti::resize(size_t s) {
  if (opt.size() >= s) return;
  opt.resize(s, false);
  config.resize(s);
  labels.resize(s, "");
  lowbound.resize(s, 0);
  highbound.resize(s, 0);
}

void HPOTargets::Opti::print(const std::string &key) const {
  int p = 1;
  int w = 8 + p;
  size_t start = core::is_number(config[0]) ? 0 : 1;

  std::cout << "\nConfig Key: " << key << std::endl;
  std::cout << std::setw(w) << "Label:";
  for (size_t i = start, idx = start; i < labels.size(); ++i) {
    if (core::is_number(config[i])) {
      std::cout << std::setw(w) << labels[idx++];
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "Index:";
  for (size_t i = start, idx = 1; i < config.size(); ++i) {
    if (core::is_number(config[i])) {
      std::cout << std::setw(w) << idx++;
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "Params:";
  for (size_t i = start; i < config.size(); ++i) {
    if (core::is_number(config[i])) {
      double num = std::stod(config[i]);
      if (core::is_integer(num)) {
        std::cout << std::setw(w) << std::right << static_cast<int>(num);
      } else {
        std::cout << std::setw(w) << std::right
          << std::scientific << std::setprecision(p) << num;
      }
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "H bound:";
  for (size_t i = start; i < opt.size(); ++i) {
    if (!core::is_number(config[i])) continue;
    if (opt[i]) {
      if (core::is_integer(highbound[i])) {
        std::cout << std::setw(w) << static_cast<int>(highbound[i]);
      } else {
        std::cout << std::setw(w) << std::scientific 
          << std::setprecision(p) << highbound[i];
      }
    } else {
      std::cout << std::setw(w) << std::right << " ";
    }
  }
  std::cout << std::endl;

  std::cout << std::setw(w) << "L bound:";
  for (size_t i = start; i < opt.size(); ++i) {
    if (!core::is_number(config[i])) continue;
    if (opt[i]) {
      if (core::is_integer(lowbound[i])) {
        std::cout << std::setw(w) << static_cast<int>(lowbound[i]);
      } else {
        std::cout << std::setw(w) << 
          std::scientific << std::setprecision(p) << lowbound[i];
      }
    } else {
      std::cout << std::setw(w) << std::right << " ";
    }
  }
  std::cout << std::endl;
}

void HPOTargets::create_labels() {
  for (auto& pair : optimap) {
    std::string key = pair.first;
    Opti& o = pair.second;

    auto suffix_prefix = determine_suffix_prefix(key);
    std::string type = "TYPE" + suffix_prefix.first;
    std::string rcut = "RCUT" + suffix_prefix.first;
    std::string cgrid = "CGRID" + suffix_prefix.first;
    std::string sgrid = "SGRID" + suffix_prefix.first;
    std::string watom = "WATOM";
    std::string weights = "WEIGHTS";

    size_t i = key == type ? 1 : 0;

    auto d_labels = type=="TYPE" ? get_labels(key)
      : get_d_labels(type, suffix_prefix.second);

    size_t nparams = 0;
    for (auto it = d_labels.begin(); it != d_labels.end(); ++it) {
      std::string label = it->first;
      auto params = it->second;

      if (key == rcut) {
        o.labels[i++] = label;
      } else if (key == weights) {
        o.labels[i++] = label;
      } else if (key == type) {
        nparams = params.size();
        for (size_t j = 0; j < nparams; j++) o.labels[i++] = label;
      } else if (key == cgrid || key == sgrid) {
        process_grid_labels(label, o, i, params, key);
      } else if (key == watom) {
        process_atoms_labels(o, i, watom);
      } else {
        if (watom != "\0") std::cerr << "Cannot create label for: "
          << label << std::endl;
        i++;
      }
    }
  }
}

std::vector<std::pair<std::string, std::vector<double>>> HPOTargets::
get_d_labels(const std::string& type, const std::string& prefix) {
  std::vector<std::pair<std::string, std::vector<double>>> d_label_obj;

  // if (config.exist(type)) { }
  size_t s = context.size(type);
  std::vector<std::string> type_config(s);
  context.get<std::vector<std::string>>(type, type_config);

  std::vector<double> d_label_param;
  std::string current_label;
  bool processing = false;

  for (const auto& token : type_config) {
    if (token.rfind(prefix, 0) == 0) {
      if (token == prefix + "mJoin") continue;

      if (processing) {
        d_label_obj.emplace_back(current_label, d_label_param);
        d_label_param.clear();
      }

      current_label = token;
      processing = true;
    } else if (processing && core::is_number(token)) {
      d_label_param.push_back(std::stod(token));
    }
  }

  if (processing) {
    d_label_obj.emplace_back(current_label, d_label_param);
  }

  return d_label_obj;
}
std::vector<std::pair<std::string, std::vector<double>>> HPOTargets::
get_labels(const std::string& key) {
  std::vector<std::pair<std::string, std::vector<double>>> label_obj;

  size_t s = context.size(key);
  std::vector<std::string> key_config(s);
  context.get<std::vector<std::string>>(key, key_config);

  std::vector<double> label_param;

  for (const auto& token : key_config) {
    label_param.push_back(std::stod(token));
  }
  label_obj.emplace_back(key, label_param);

  return label_obj;
}

std::pair<std::string, std::string> HPOTargets::
determine_suffix_prefix(const std::string& key) {
  if (key.size() >= 2 && key.compare(key.size() - 2, 2, "2B") == 0) {
    return {"2B", "D2_"};
  }
  else if (key.size() >= 2 && key.compare(key.size() - 2, 2, "MB") == 0) {
    return {"MB", "DM_"};
  } else {
    return {"", ""};
  }
}

void HPOTargets::process_grid_labels(
  const std::string& label, Opti& o, size_t& i,
  const std::vector<double>& params, const std::string& key) {
  std::unique_ptr<models::D_Base> d(tadah::core::factory<models::D_Base>(label));
  int arg_pos = d->get_arg_pos(key);

  if (arg_pos > -1) {
    size_t nparams = params[arg_pos];
    for (size_t j = 0; j < nparams; j++) {
      o.labels[i++] = label;
    }
  }
}

void HPOTargets::process_atoms_labels(Opti& o, size_t& i, std::string& watom) {
  for (size_t j = 0; j < context.size("ATOM"); ++j) {
    o.labels[i++] = context.get<std::string>("ATOM", j);
  }
  watom = "\0"; // Avoid further processing
}

HPOTargets::HPOTargets(core::Context &c, std::string fn) : context(c) {
  models::D_mJoin::expand_grids(context);

  read(fn);
  std::sort(LS.begin(), LS.end(), LammpsSim::greater);
  create_labels();
  for (auto& pair : optimap) {
    std::string key = pair.first;
    Opti &o = pair.second;
    o.print(key);
  }

  /* ---------- gather bounds --------------------------------------- */
  std::vector<double> low, high;
  for (auto& p : optimap)
  {
    const Opti& o = p.second;
    for (size_t i=0;i<o.opt.size();++i)
      if (o.opt[i]) {
        low .push_back(o.lowbound [i]);
        high.push_back(o.highbound[i]);
      }
  }

  spec.low        = std::move(low);
  spec.high       = std::move(high);
  spec.is_int.assign(spec.low.size(), false);   // no integer vars yet
  spec.analyse_logscale();   // mutates low/high in place; fills log_scaled

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

  /* Propagate the resolved search space and (analyse_logscale-applied)
   * bounds into the LOCAL and PRESTAGE sub-specs FIRST, before resolving
   * any INIT_STRATEGY chain on them: strategies like RANDOM/LHS/SOBOL
   * sample within bounds, so an unbounded sub-spec produces 0-dim
   * points. (Bound inheritance must precede init resolution.) */
  auto inherit_bounds_only = [&](OptimizerSpec& sub) {
    sub.low        = spec.low;
    sub.high       = spec.high;
    sub.is_int     = spec.is_int;
    sub.log_scaled = spec.log_scaled;
  };
  if (spec.inner)    inherit_bounds_only(*spec.inner);
  if (spec.prestage) {
    inherit_bounds_only(*spec.prestage);
    if (spec.prestage->inner) inherit_bounds_only(*spec.prestage->inner);
  }
  /* BASINS' EXPLORE/REFINE stage sub-specs need bounds too. */
  if (spec.strategy_type == StrategyType::BASINS) {
    if (spec.basins.explore) {
        inherit_bounds_only(*spec.basins.explore);
        if (spec.basins.explore->inner) inherit_bounds_only(*spec.basins.explore->inner);
    }
    if (spec.basins.refine) {
        inherit_bounds_only(*spec.basins.refine);
        if (spec.basins.refine->inner) inherit_bounds_only(*spec.basins.refine->inner);
    }
  }

  /* ---------- resolve INIT_STRATEGY chain into optim_init -----------
   *
   * Apply to OUTER, PRESTAGE, and PRESTAGE->LOCAL (LOCAL inside OUTER
   * is the NLopt MLSL local sub-algo and inherits the outer's
   * starting point -- it does not need its own chain). Each spec's
   * INIT_STRATEGY / INIT_SEED / INIT_INCLUDE_CONFIG / WARM_ON_MISMATCH
   * is honoured independently of every other spec.
   *
   * Diagnostics tag the location ("OUTER" / "PRESTAGE" / "PRESTAGE.LOCAL")
   * so the user can see which block fired which strategy. */
  auto resolve_init = [&](OptimizerSpec& s, const char* where) {
    auto strategy = build_init_strategy(s.init_strategy_spec);
    std::uint64_t seed = s.init_strategy_spec.seed.value_or(
        static_cast<std::uint64_t>(std::random_device{}()));
    InitContext ictx{s, *this, seed};
    std::size_t k = s.init_strategy_spec.k_total.value_or(strategy->default_k());
    s.optim_init = strategy->generate(ictx, k);
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
      std::cerr << "[HPO] WARNING: INIT_STRATEGY chain produced "
                << spec.optim_init.size() << " seeds but " << spec.lib
                << ":" << spec.algo << " consumes only the first; "
                << (spec.optim_init.size() - 1)
                << " seed(s) will be discarded. "
                << "Set MULTISTART, add a PRESTAGE block, or pick a "
                   "multi-seed backend (" << format_caps_list(multi_seed)
                << ") to use them.\n";
    }
  }

  if (spec.prestage) {
    /* If PRESTAGE has no explicit chain and no INIT_STRATEGY entries,
     * it inherits the OUTER's optim_init (legacy behaviour: a CONFIG
     * point flows in). If PRESTAGE has its own chain, that chain wins
     * and the OUTER's optim_init is *not* prepended -- PRESTAGE owns its
     * starting points. */
    if (spec.prestage->init_strategy_spec.chain.empty() &&
        spec.prestage->init_strategy_spec.include_config) {
      /* Default: inherit from OUTER. */
      if (spec.prestage->optim_init.empty())
        spec.prestage->optim_init = spec.optim_init;
    } else {
      resolve_init(*spec.prestage, "PRESTAGE");
    }
    /* PRESTAGE's nested LOCAL: inherit OUTER's seed by default, or
     * resolve its own chain when one was given. */
    if (spec.prestage->inner) {
      if (spec.prestage->inner->init_strategy_spec.chain.empty() &&
          spec.prestage->inner->init_strategy_spec.include_config) {
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
      if (s.init_strategy_spec.chain.empty() &&
          s.init_strategy_spec.include_config) {
        if (s.optim_init.empty())
          s.optim_init = spec.optim_init;
      } else {
        resolve_init(s, tag);
      }
    };
    if (spec.basins.explore) resolve_or_inherit(*spec.basins.explore, "BASINS.EXPLORE");
    if (spec.basins.refine ) resolve_or_inherit(*spec.basins.refine,  "BASINS.REFINE");
  }
}

// -------------------------------------------------------------------
// HPOTargets::read   - complete, no omissions
// -------------------------------------------------------------------
/* Helper: parse a boolean token (true/false/1/0/yes/no/on/off, any case).
 * Throws on unknown spellings. */
static bool parse_bool(const std::string& v, const std::string& ctx)
{
    std::string lo; lo.reserve(v.size());
    for (char c : v) lo.push_back(static_cast<char>(std::tolower(c)));
    if (lo=="true" || lo=="1" || lo=="yes" || lo=="on")  return true;
    if (lo=="false"|| lo=="0" || lo=="no"  || lo=="off") return false;
    throw std::runtime_error(ctx + ": expected boolean (true/false/1/0/yes/no/on/off), got '" + v + "'");
}

/* Helper: parse one optimiser-attribute key/value into an OptimizerSpec.
 * Returns true if `key` was consumed; false otherwise (caller can try
 * other dispatches).  These keys are valid wherever an OptimizerSpec is
 * the active receiver -- STRATEGY top-level, EXPLORE, REFINE, INNER. */
static bool dispatch_opt_attr(const std::string& key,
                              std::istringstream& iss,
                              OptimizerSpec& a)
{
    if      (key=="LIB")       iss >> a.lib;
    else if (key=="ALGO")      iss >> a.algo;
    else if (key=="MAXEVAL")   { uint64_t v; iss>>v; a.stop.maxeval=v; }
    else if (key=="MAXTIME")   { double   v; iss>>v; a.stop.maxtime=v; }
    else if (key=="STOPVAL")   { double   v; iss>>v; a.stop.stopval=v; }
    else if (key=="FTOL_REL")  { double   v; iss>>v; a.stop.ftol_rel=v; }
    else if (key=="FTOL_ABS")  { double   v; iss>>v; a.stop.ftol_abs=v; }
    else if (key=="XTOL_REL")  { double   v; iss>>v; a.stop.xtol_rel=v; }
    else if (key=="XTOL_ABS")  { double   v; iss>>v; a.stop.xtol_abs=v; }
    else if (key=="POPULATION"){ unsigned v; iss>>v; a.pop=v; }
    else if (key=="STEP")      { double   v; iss>>v; a.step=v; }
    else if (key=="SEED")      { unsigned long v; iss>>v; a.seed=v; }
    else if (key=="VECTOR_ARR"){ unsigned v; iss>>v; a.vector_storage=v; }
    else if (key=="THREADS")   { unsigned v; iss>>v; a.num_threads=v; }
    else if (key=="GRAD_STRATEGY") {
        std::string v; iss>>v;
        if      (v=="NUMERICAL")  a.grad_strategy = GradStrategy::Numerical;
        else if (v=="ANALYTICAL") a.grad_strategy = GradStrategy::Analytical;
        else if (v=="HYBRID")     a.grad_strategy = GradStrategy::Hybrid;
        else if (v=="AUTODIFF")   throw std::runtime_error(
            "GRAD_STRATEGY=AUTODIFF was renamed to ANALYTICAL in Phase 2; "
            "the prior stub now backs a closed-form analytical gradient. "
            "Use GRAD_STRATEGY ANALYTICAL (strict -- every theta must have a "
            "registered analytical derivative) or GRAD_STRATEGY HYBRID "
            "(analytical-where-registered + finite-difference fallback).");
        else throw std::runtime_error(
            "GRAD_STRATEGY must be one of NUMERICAL, ANALYTICAL, HYBRID "
            "(got '" + v + "')");
    }
    else if (key=="PARAM")     { std::string p; double v; iss>>p>>v; a.param[p]=v; }
    else return false;
    return true;
}

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
                "(reserved keyword). Use RANDOM/LHS/SOBOL/PRIOR/WARM/CONFIG.");
        std::string tok;
        while (iss >> tok) e.args.push_back(tok);
        a.chain.push_back(std::move(e));
        return true;
    }
    if (key=="K_TOTAL") { std::size_t v; iss>>v; a.k_total=v; return true; }
    if (key=="SEED")    { std::uint64_t v; iss>>v; a.seed=v; return true; }
    if (key=="INCLUDE_CONFIG") {
        std::string v; iss>>v;
        a.include_config = parse_bool(v, "INCLUDE_CONFIG");
        return true;
    }
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
        {"PRESTAGE",                    "replaced by `STRATEGY { TYPE PRESTAGE EXPLORE { ... } REFINE { ... } }`."},
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
        {"RESTART_INIT_STRATEGY",       "moved into `RESTART { STRATEGY <method> ... }` (re-uses INIT-block grammar)."},
        {"INIT_STRATEGY",               "moved into `INIT { STRATEGY <method> ... } ENDINIT` block."},
        {"INIT_K",                      "renamed to K_TOTAL inside an INIT block."},
        {"INIT_SEED",                   "renamed to SEED inside an INIT block."},
        {"INIT_INCLUDE_CONFIG",         "renamed to INCLUDE_CONFIG inside an INIT block."},
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
                /* Restart never auto-prepends CONFIG: a stagnated run by
                 * definition has the user's config in its history. */
                spec.restart.init_strategy.include_config = false;
                active_init = &spec.restart.init_strategy;
                /* Suspend active_opt while inside RESTART so SEED/STRATEGY/
                 * INCLUDE_CONFIG go to active_init, not the OUTER spec. */
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
            if (active_opt && dispatch_opt_attr(key, iss, *active_opt))
                continue;

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
                if (key=="LOG_HP") { double v; iss>>v; spec.log_threshold=v; continue; }

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
                    "(only LIB/ALGO/MAXEVAL/FTOL_*/STEP/SEED/PARAM/POPULATION/THREADS/VECTOR_ARR are valid)");
            }

            /* Anything still unhandled inside an OPTIMIZER-level block is
             * a user mistake (LIB/ALGO/MAXEVAL etc. now live inside
             * STRATEGY). LOSS / PC_LAMMPS / OPTIM / FAILSCORE / OUTPUT /
             * KEEP and friends are ROOT-level and fall through. */
            if (block != ROOT) {
                throw std::runtime_error("unknown key '" + key + "' in this block");
            }

            /* ==================  global (ROOT) section  =========== */

            /* optimizer-independent settings ----------------------- */
            // if      (key=="THREADS"){ double v; iss>>v; spec.param["THREADS"]=v; }
            if (key=="FAILSCORE"){ iss>>FAIL_SCORE; }
            else if (key=="MAXTIME")  { iss>>MAX_TIME;  }
            else if (key=="QFLUSH")   { iss>>QFLUSH_THRESHOLD; }
            else if (key=="OUTPUT")   { iss>>outputEvery; }
            else if (key=="BOUTPUT")  { iss>>bestOutputEvery; }
            else if (key=="DIGITS")   { iss>>outputDigits;
                if (outputDigits < 1 || outputDigits > 15)
                    throw std::runtime_error("DIGITS must be 1-15"); }
            else if (key=="DUMP")     { iss>>dumpPotEvery>>dumpPotDir;   handle_directory_creation(dumpPotDir); }
            else if (key=="KEEP")     { iss>>dumpBestPotEvery>>dumpBestPotDir; handle_directory_creation(dumpBestPotDir); }

            /* loss -------------------------------------------------- */
            else if (key=="LOSS")
            {
                std::string type; iss>>type;
                std::vector<double> params;
                if (type=="HUBER" || type=="TUKEY"){ double p; iss>>p; params={p}; }
                auto id = tadah::LossFunction::from_string(type);
                globalLoss = tadah::LossFunction(id,params);
            }

            /* optimisation variables --------------------------------*/
            else if (key=="OPTIM")   { process_optim_config(iss,nline); }

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
                double tgt, wgt;
                iss >> tgt >> wgt;
                if (!iss)
                    throw std::runtime_error(key + " requires <target> <weight>");
                /* Optional flags. Order is irrelevant; both are optional. */
                HPOTargets::SkipCaps caps;
                std::string flag;
                while (iss >> flag) {
                    if (flag == "--skip-above") {
                        double v; iss >> v;
                        if (!iss || !(v > 0.0))
                            throw std::runtime_error(key + " --skip-above requires a positive number");
                        caps.abs_above = v;
                    } else if (flag == "--skip-rel") {
                        double v; iss >> v;
                        if (!iss || !(v > 0.0))
                            throw std::runtime_error(key + " --skip-rel requires a positive number");
                        caps.rel_above = v;
                    } else {
                        throw std::runtime_error(key + ": unknown flag '" + flag +
                            "' (only --skip-above and --skip-rel are accepted)");
                    }
                }
                if      (key=="PC_EMAE")   { emae=tgt;   w_emae=wgt;   use_emae=true;   skip_emae=caps; }
                else if (key=="PC_ERMSE")  { ermse=tgt;  w_ermse=wgt;  use_ermse=true;  skip_ermse=caps; }
                else if (key=="PC_ErRMSE") { errmse=tgt; w_errmse=wgt; use_errmse=true; skip_errmse=caps; }
                else if (key=="PC_ERSQ")   { ersq=tgt;   w_ersq=wgt;   use_ersq=true;   skip_ersq=caps; }
                else if (key=="PC_SRMSE")  { srmse=tgt;  w_srmse=wgt;  use_srmse=true;  skip_srmse=caps; }
                else if (key=="PC_FRMSE")  { frmse=tgt;  w_frmse=wgt;  use_frmse=true;  skip_frmse=caps; }
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

void HPOTargets::process_optim_config(std::istringstream& iss, size_t nline) {
  std::string k;
  if (!(iss >> k)) {
    throw std::runtime_error("Failed to read key from input.");
  }

  Opti& omap = optimap[k];
  size_t s = context.size(k);
  omap.resize(s);

  std::vector<std::string> tokens = parse_optim_line(iss);

  if (tokens.size() < 3) {
    throw std::runtime_error(
      "Insufficient tokens from parse_optim_line. Tokens: "
      + core::to_string(tokens.size()));
  }

  std::vector<size_t> user_indices;
  try {
    user_indices = core::parse_indices(tokens.back());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Error parsing indices on process_optim_config line "
      + core::to_string(nline) + ": " + std::string(e.what()) +
      "\nTokens: " + tokens.back());
  }
  tokens.pop_back();

  double high;
  try {
    high = std::stod(tokens.back());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Error converting high bound to double on process_optim_config line "
      + core::to_string(nline) + ": " + std::string(e.what()) +
      "\nTokens: " + tokens.back());
  }
  tokens.pop_back();

  double low;
  try {
    low = std::stod(tokens.back());
  } catch (const std::exception& e) {
    throw std::runtime_error(
      "Error converting low bound to double on process_optim_config line "
      + core::to_string(nline) + ": " + std::string(e.what()) +
      "\nTokens: " + tokens.back());
  }
  tokens.pop_back();

  context.get<std::vector<std::string>>(k, omap.config);

  for (const auto& ui : user_indices) {
    try {
      size_t i = ui_offset(omap.config, ui);
      if (i>=s) throw std::runtime_error("Out of bound index: " + core::to_string(ui));
      omap.opt[i] = true;
      omap.highbound[i] = high;
      omap.lowbound[i] = low;
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

