#include <tadah/hpo/refit/refit_parser.h>

#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/strict_parse.h>
#include <tadah/core/context.h>

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <vector>

namespace tadah {
namespace hpo {
namespace refit {

namespace {

std::string up(std::string s) {
  for (char &c : s) c = static_cast<char>(std::toupper((unsigned char)c));
  return s;
}

// True if the token is meant as a number (is_number_shaped), so the INIT and
// MAX_ADD readers stop pulling operands at the next keyword or channel token;
// the operand itself is then read with the number grammar (parse_finite,
// parse_uint), which names a malformed one.
bool looks_number(const std::string &s) { return is_number_shaped(s); }

// The keys a stage's OPTIMIZER block (or its INNER block) reads. A stage reads
// only some of the optimiser attributes of `tadah hpo`; any other one it would
// silently ignore is an error naming the keys it does read (the rule of the
// `tadah hpo` STRATEGY keys that would have no effect).
struct StageKeys {
  const char *stage;                  // EXPLORE, REFINE, FUSE
  std::vector<std::string> outer;     // the OPTIMIZER block's keys
  std::vector<std::string> inner;     // the INNER block's (empty: no INNER)
};

std::string reads_list(const std::vector<std::string> &keys) {
  // MAXITER is listed with MAXEVAL, of which it is an alias.
  std::string out;
  for (const auto &k : keys) {
    if (k == "MAXITER") continue;
    if (!out.empty()) out += ", ";
    out += k;
    if (k == "MAXEVAL" &&
        std::find(keys.begin(), keys.end(), "MAXITER") != keys.end())
      out += "/MAXITER";
  }
  return out;
}

// A key the refit OPTIMIZER blocks know: an optimiser attribute of the shared
// grammar, the Ceres keys GTOL and MAXITER, or INNER. Anything else is left
// to the "unknown key" error of read_optimizer_block().
bool known_optimizer_key(const std::string &key) {
  const auto &attrs = opt_attr_keys();
  return key == "INNER" || key == "GTOL" || key == "MAXITER" ||
         std::find(attrs.begin(), attrs.end(), key) != attrs.end();
}

// read_optimizer_block() prefixes these messages with the block
// ("REFINE: ", "EXPLORE INNER: ").
KeyCheckFn stage_key_check(const StageKeys &sk) {
  return [sk](const std::string &key, bool in_inner) {
    const std::vector<std::string> &reads = in_inner ? sk.inner : sk.outer;
    if (std::find(reads.begin(), reads.end(), key) != reads.end()) return;
    if (!known_optimizer_key(key)) return;
    // SEED and FTOL_ABS say what to use instead.
    if (key == "SEED")
      throw std::runtime_error(
          "SEED is not accepted in a tadah refit OPTIMIZER block: the "
          "refit's random numbers come from REFIT_COMPACT_SEED; set "
          "REFIT_COMPACT_SEED in the configuration instead.");
    if (key == "FTOL_ABS" && std::string(sk.stage) != "EXPLORE")
      throw std::runtime_error(
          "FTOL_ABS is not supported — Ceres' function_tolerance is RELATIVE "
          "and has no absolute form. Use FTOL_REL.");
    throw std::runtime_error(key +
                             " is not used by a tadah refit stage (it reads " +
                             reads_list(reads) + ")");
  };
}

const StageKeys kExploreKeys{"EXPLORE",
                             {"LIB", "ALGO", "MAXEVAL", "INNER"},
                             {"LIB", "ALGO", "MAXEVAL", "FTOL_REL", "XTOL_REL"}};
const StageKeys kRefineKeys{
    "REFINE", {"LIB", "ALGO", "MAXEVAL", "MAXITER", "FTOL_REL", "GTOL"}, {}};
const StageKeys kFuseKeys{
    "FUSE", {"LIB", "ALGO", "MAXEVAL", "MAXITER", "FTOL_REL"}, {}};

// ALGO is what a stage's OPTIMIZER block, and its INNER block, is read for:
// it must be set (NONE / UNIFORM are written out). An INNER block without
// ALGO meant no local search, while an EXPLORE optimiser without an INNER
// block runs the default one.
void require_algo(const OptimizerSpec &o, const std::string &where,
                  const char *block, const char *off) {
  if (o.algo == "NOT SET")
    throw std::runtime_error(where + ": ALGO is not set; set ALGO in the " +
                             block + " block (" + off + ")");
}

bool as_bool(const std::string &s, const std::string &what) {
  const std::string u = up(s);
  if (u == "TRUE" || u == "ON" || u == "1" || u == "YES") return true;
  if (u == "FALSE" || u == "OFF" || u == "0" || u == "NO") return false;
  throw std::runtime_error(what + ": expected true|false, got '" + s + "'");
}

// Pull a stage block's tokens out of the Context (empty if absent).
std::vector<std::string> stage_tokens(const tadah::core::Context &c,
                                      const std::string &key) {
  std::vector<std::string> v;
  if (c.contains(key) && c.size(key) > 0) {
    v.resize(c.size(key));
    c.get(key, v);
  }
  return v;
}

void parse_init(const std::vector<std::string> &tok, RefitConfig &cfg) {
  size_t i = 0;
  while (i < tok.size()) {
    InitBasis b;
    const std::string ch = up(tok[i++]);
    if (ch == "PHI")
      b.channel = InitBasis::PHI;
    else if (ch == "RHO")
      b.channel = InitBasis::RHO;
    else if (ch == "F")
      b.channel = InitBasis::F;
    else
      throw std::runtime_error("INIT: expected channel PHI|RHO|F, got '" +
                               tok[i - 1] + "'");
    if (i >= tok.size())
      throw std::runtime_error("INIT " + ch + ": missing placement keyword");
    const std::string pl = up(tok[i++]);
    if (pl == "MANUAL") {
      b.placement = PinMode::Manual;
      if (i >= tok.size() || !looks_number(tok[i]))
        throw std::runtime_error("INIT " + ch +
                                 " MANUAL requires at least <mu>");
      /* looks_number() only tells an operand from the next keyword; the
       * operand itself must be finite. */
      b.mu = parse_finite(tok[i++], "INIT " + ch + " MANUAL <mu>");
      if (i < tok.size() && looks_number(tok[i])) {
        b.eta = parse_finite(tok[i++], "INIT " + ch + " MANUAL <eta>");
        b.has_eta = true;
      }
    } else if (pl == "XMIN") {
      b.placement = PinMode::XMin;
    } else if (pl == "XMAX") {
      b.placement = PinMode::XMax;
    } else if (pl == "YMIN") {
      b.placement = PinMode::YMin;
    } else if (pl == "YMAX") {
      b.placement = PinMode::YMax;
    } else if (pl == "SMAX") {
      b.placement = PinMode::SMax;
    } else if (pl == "CMAX") {
      b.placement = PinMode::CMax;
    } else {
      throw std::runtime_error(
          "INIT " + ch + ": unknown placement '" + tok[i - 1] +
          "' (MANUAL|XMIN|XMAX|YMIN|YMAX|SMAX|CMAX)");
    }
    if (i < tok.size() && up(tok[i]) == "PIN") {
      b.pinned = true;
      ++i;
    }
    cfg.init.push_back(b);
  }
}

void parse_explore(const std::vector<std::string> &tok, RefitConfig &cfg) {
  size_t i = 0;
  while (i < tok.size()) {
    const std::string u = up(tok[i]);
    if (u == "MAX_ADD") {
      // 1 value (all functions) or 3 values (phi rho F). Pull the run of
      // numeric operands (stops at the next keyword, e.g. OPTIMIZER). Each
      // is an integer >= 0 (strict_parse.h): std::stoi read '+3' as 3, and
      // a negative value silently meant no cap -- leaving MAX_ADD out is
      // how to write that.
      std::vector<int> v;
      size_t j = i + 1;
      while (j < tok.size() && looks_number(tok[j]) && v.size() < 3) {
        v.push_back(static_cast<int>(
            parse_uint(tok[j], "EXPLORE MAX_ADD", 0,
                       std::numeric_limits<int>::max())));
        ++j;
      }
      if (v.size() != 1 && v.size() != 3)
        throw std::runtime_error(
            "EXPLORE: MAX_ADD requires 1 value (all functions) or 3 (phi rho F)");
      if (v.size() == 1)
        cfg.explore.max_add = {v[0], v[0], v[0]};
      else
        cfg.explore.max_add = {v[0], v[1], v[2]};
      i = j;
    } else if (u == "ENABLED") {
      if (i + 1 >= tok.size())
        throw std::runtime_error("EXPLORE: ENABLED requires true|false");
      cfg.explore.enabled = as_bool(tok[i + 1], "EXPLORE ENABLED");
      i += 2;
    } else if (u == "OPTIMIZER") {
      read_optimizer_block(tok, i, cfg.explore.opt, nullptr,
                           stage_key_check(kExploreKeys),  // outer + INNER
                           kExploreKeys.stage);
      cfg.explore.has_opt = true;
    } else {
      throw std::runtime_error("EXPLORE: unknown key '" + tok[i] +
                               "' (expected MAX_ADD, ENABLED or OPTIMIZER)");
    }
  }
}

// REFINE/FUSE OPTIMIZER blocks accept the Ceres-flavoured command-specific keys
// GTOL (-> param["GTOL"], Ceres gradient_tolerance; REFINE only, FUSE does not
// read it) and MAXITER (alias of MAXEVAL -> Ceres max_num_iterations). The
// function tolerance key is FTOL_REL (handled by dispatch_opt_attr -> Ceres
// function_tolerance): Ceres' function_tolerance is RELATIVE — it stops when
// (new_cost - old_cost) < ftol * old_cost. Ceres has NO absolute function
// tolerance, so FTOL_ABS (and the bare FTOL) are rejected with a pointer to
// FTOL_REL: FTOL_ABS by the stage's key check (stage_key_check), before
// dispatch_opt_attr would consume it, and the bare FTOL here.
bool ceres_extra(const std::string &key, std::istringstream &iss,
                 OptimizerSpec &a) {
  if (key == "GTOL") {
    a.param["GTOL"] = parse_finite(next_value(iss, key), key);
    return true;
  }
  if (key == "MAXITER") {
    // The alias of MAXEVAL, with its range: the int iteration cap Ceres
    // takes (apply_stage_optimizers).
    a.stop.maxeval = parse_uint(next_value(iss, key), key, 0,
                                std::numeric_limits<int>::max());
    return true;
  }
  if (key == "FTOL" || key == "FTOL_ABS")
    throw std::runtime_error(
        "'" + key +
        "' is not supported — Ceres' function_tolerance is RELATIVE and has no "
        "absolute form. Use FTOL_REL.");
  return false;
}

void parse_ceres_stage(const std::vector<std::string> &tok, OptimizerSpec &opt,
                       bool &has_opt, bool &enabled, const StageKeys &keys) {
  const char *stage = keys.stage;
  size_t i = 0;
  while (i < tok.size()) {
    const std::string u = up(tok[i]);
    if (u == "OPTIMIZER") {
      read_optimizer_block(tok, i, opt, ceres_extra, stage_key_check(keys),
                           stage);
      has_opt = true;
    } else if (u == "ENABLED") {
      if (i + 1 >= tok.size())
        throw std::runtime_error(std::string(stage) + ": ENABLED requires true|false");
      enabled = as_bool(tok[i + 1], std::string(stage) + " ENABLED");
      i += 2;
    } else {
      throw std::runtime_error(std::string(stage) + ": unknown key '" + tok[i] +
                               "' (expected OPTIMIZER or ENABLED)");
    }
  }
}

[[noreturn]] void algo_err(const char *stage, const std::string &lib,
                           const std::string &algo, const char *legal) {
  throw std::runtime_error(std::string(stage) + ": unsupported optimiser LIB '" +
                           (lib.empty() ? "(unset)" : lib) + "' ALGO '" +
                           (algo.empty() ? "(unset)" : algo) + "'. Legal: " +
                           legal);
}

}  // namespace

std::string refit_explore_outer_algo(const OptimizerSpec &s) {
  const std::string lib = up(s.lib == "NOT SET" ? "" : s.lib);
  const std::string algo = up(s.algo == "NOT SET" ? "" : s.algo);
  if (algo == "UNIFORM" || algo == "NONE") return "UNIFORM";
  if (lib == "DLIB" || algo == "DLIB_GLOBAL") {
    if (algo == "GLOBAL" || algo == "DLIB_GLOBAL") return "DLIB_GLOBAL";
    algo_err("EXPLORE outer", lib, algo, "DLIB GLOBAL");
  }
  if (algo == "MLSL" || algo == "GN_MLSL_LDS" || algo == "GN_MLSL") return "MLSL";
  if (algo == "DIRECT" || algo == "GN_DIRECT_L" || algo == "GN_DIRECT")
    return "DIRECT";
  if (algo == "CRS2" || algo == "GN_CRS2_LM" || algo == "GN_CRS2") return "CRS2";
  algo_err("EXPLORE outer", lib, algo,
           "NLOPT GN_MLSL_LDS|GN_DIRECT_L|GN_CRS2_LM, DLIB GLOBAL, or UNIFORM");
}

std::string refit_explore_inner_algo(const OptimizerSpec &s) {
  const std::string lib = up(s.lib == "NOT SET" ? "" : s.lib);
  const std::string algo = up(s.algo == "NOT SET" ? "" : s.algo);
  if (algo.empty() || algo == "NONE") return "NONE";
  if (lib == "DLIB") {
    if (algo == "LBFGS" || algo == "DLIB_LBFGS") return "DLIB_LBFGS";
    algo_err("EXPLORE inner", lib, algo, "DLIB LBFGS");
  }
  if (algo == "SBPLX" || algo == "LN_SBPLX") return "SBPLX";
  if (algo == "NELDERMEAD" || algo == "LN_NELDERMEAD") return "NELDERMEAD";
  if (algo == "BOBYQA" || algo == "LN_BOBYQA") return "BOBYQA";
  if (algo == "PRAXIS" || algo == "LN_PRAXIS") return "PRAXIS";
  if (algo == "NLOPT_LBFGS" || algo == "LD_LBFGS") return "NLOPT_LBFGS";
  algo_err("EXPLORE inner", lib, algo,
           "NLOPT LN_SBPLX|LN_NELDERMEAD|LN_BOBYQA|LN_PRAXIS|LD_LBFGS, "
           "DLIB LBFGS, or NONE");
}

std::string refit_refine_algo(const OptimizerSpec &s) {
  const std::string lib = up(s.lib == "NOT SET" ? "" : s.lib);
  const std::string algo = up(s.algo == "NOT SET" ? "" : s.algo);
  if (algo == "NONE") return "NONE";
  if (lib == "CERES" || algo == "CERES_LM" || algo == "CERES_DOGLEG" ||
      algo == "LM" || algo == "DOGLEG") {
    if (algo == "LM" || algo == "CERES_LM") return "CERES_LM";
    if (algo == "DOGLEG" || algo == "CERES_DOGLEG") return "CERES_DOGLEG";
    algo_err("REFINE", lib, algo, "CERES LM|DOGLEG");
  }
  if (lib == "DLIB") {
    if (algo == "LBFGS" || algo == "DLIB_LBFGS") return "DLIB_LBFGS";
    algo_err("REFINE", lib, algo, "DLIB LBFGS");
  }
  if (lib == "NLOPT") {
    if (algo == "LD_LBFGS" || algo == "NLOPT_LBFGS") return "NLOPT_LBFGS";
    algo_err("REFINE", lib, algo, "NLOPT LD_LBFGS");
  }
  algo_err("REFINE", lib, algo,
           "CERES LM|DOGLEG, NLOPT LD_LBFGS, DLIB LBFGS, or NONE");
}

std::string refit_fuse_algo(const OptimizerSpec &s) {
  const std::string lib = up(s.lib == "NOT SET" ? "" : s.lib);
  const std::string algo = up(s.algo == "NOT SET" ? "" : s.algo);
  if (algo == "NONE") return "NONE";
  if (lib == "CERES" || algo == "CERES_LM" || algo == "CERES_DOGLEG" ||
      algo == "LM" || algo == "DOGLEG") {
    if (algo == "LM" || algo == "CERES_LM") return "CERES_LM";
    if (algo == "DOGLEG" || algo == "CERES_DOGLEG") return "CERES_DOGLEG";
  }
  algo_err("FUSE", lib, algo,
           "CERES LM|DOGLEG or NONE (FUSE is a least-squares stage)");
}

RefitConfig RefitParser::parse(const tadah::core::Context &c) {
  // Old grammar: a top-level OPTIMIZER block (GLOBAL/EXPLORE/POLISH/JOINT
  // subblocks) is captured by CORE under the OPTIMIZER key. The pipeline is
  // now four top-level stage blocks, so flag it with migration guidance.
  if (c.contains("OPTIMIZER") && c.size("OPTIMIZER") > 0) {
    throw std::runtime_error(
        "tadah refit: a top-level OPTIMIZER block is no longer supported. The "
        "stage pipeline is now INIT -> EXPLORE -> REFINE -> FUSE, each its own "
        "block, with the OPTIMIZER block NESTED inside EXPLORE/REFINE/FUSE. "
        "Migration: GLOBAL folded into EXPLORE rung 0 (set its global ALGO in "
        "EXPLORE's OPTIMIZER, with an INNER local); POLISH -> REFINE; "
        "JOINT -> FUSE. See HPO_REFIT_2B_EAM/REDESIGN/configs for examples.");
  }

  RefitConfig cfg;
  parse_init(stage_tokens(c, "INIT"), cfg);
  parse_explore(stage_tokens(c, "EXPLORE"), cfg);
  parse_ceres_stage(stage_tokens(c, "REFINE"), cfg.refine.opt,
                    cfg.refine.has_opt, cfg.refine.enabled, kRefineKeys);
  parse_ceres_stage(stage_tokens(c, "FUSE"), cfg.fuse.opt, cfg.fuse.has_opt,
                    cfg.fuse.enabled, kFuseKeys);

  // Every OPTIMIZER block, and an INNER block, names its ALGO. The keys a
  // stage does not use (SEED, FTOL_ABS on the Ceres stages, THREADS, ...)
  // were rejected as they were read (stage_key_check).
  if (cfg.explore.has_opt) {
    require_algo(cfg.explore.opt, "EXPLORE", "OPTIMIZER",
                 "UNIFORM for no global search");
    if (cfg.explore.opt.inner)
      require_algo(*cfg.explore.opt.inner, "EXPLORE INNER", "INNER",
                   "NONE for no local search");
  }
  if (cfg.refine.has_opt)
    require_algo(cfg.refine.opt, "REFINE", "OPTIMIZER", "NONE to skip the stage");
  if (cfg.fuse.has_opt)
    require_algo(cfg.fuse.opt, "FUSE", "OPTIMIZER", "NONE to skip the stage");

  // Validate (and surface) the per-stage ALGO choices early — these throw on an
  // illegal LIB/ALGO pair so the user sees the error before any fitting.
  if (cfg.explore.has_opt) {
    (void)refit_explore_outer_algo(cfg.explore.opt);
    if (cfg.explore.opt.inner)
      (void)refit_explore_inner_algo(*cfg.explore.opt.inner);
  }
  if (cfg.refine.has_opt) (void)refit_refine_algo(cfg.refine.opt);
  if (cfg.fuse.has_opt) (void)refit_fuse_algo(cfg.fuse.opt);

  // EXPLORE cannot be disabled (it builds the ladder) — still a hard error.
  // REFINE/FUSE: ENABLED false is a master kill-switch that WINS over a
  // configured optimiser (the engine sets that stage's algo to NONE and logs
  // that the configured optimiser is ignored) — it is NOT a contradiction.
  if (!cfg.explore.enabled)
    throw std::runtime_error(
        "EXPLORE: ENABLED false is not allowed — EXPLORE builds the basis "
        "ladder and cannot be disabled.");

  return cfg;
}

}  // namespace refit
}  // namespace hpo
}  // namespace tadah
