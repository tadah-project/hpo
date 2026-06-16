#include <tadah/hpo/refit/refit_parser.h>

#include <tadah/hpo/optimizer_block.h>
#include <tadah/core/context.h>

#include <cctype>
#include <cstdlib>
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

// True if the whole token parses as a floating-point number (so the INIT line
// reader can stop pulling MANUAL operands at the next keyword/channel token).
bool looks_number(const std::string &s) {
  if (s.empty()) return false;
  char *end = nullptr;
  std::strtod(s.c_str(), &end);
  return end == s.c_str() + s.size();
}

int as_int(const std::string &s, const char *what) {
  try {
    size_t p = 0;
    int v = std::stoi(s, &p);
    if (p != s.size()) throw std::invalid_argument(s);
    return v;
  } catch (...) {
    throw std::runtime_error(std::string(what) + ": expected an integer, got '" +
                             s + "'");
  }
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
      b.mu = std::strtod(tok[i++].c_str(), nullptr);
      if (i < tok.size() && looks_number(tok[i])) {
        b.eta = std::strtod(tok[i++].c_str(), nullptr);
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
      // numeric operands (stops at the next keyword, e.g. OPTIMIZER).
      std::vector<int> v;
      size_t j = i + 1;
      while (j < tok.size() && looks_number(tok[j]) && v.size() < 3) {
        v.push_back(as_int(tok[j], "EXPLORE MAX_ADD"));
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
      read_optimizer_block(tok, i, cfg.explore.opt);  // outer + INNER
      cfg.explore.has_opt = true;
    } else {
      throw std::runtime_error("EXPLORE: unknown key '" + tok[i] +
                               "' (expected MAX_ADD, ENABLED or OPTIMIZER)");
    }
  }
}

// REFINE/FUSE OPTIMIZER blocks accept the Ceres-flavoured command-specific keys
// GTOL (-> param["GTOL"], Ceres gradient_tolerance) and MAXITER (alias of
// MAXEVAL -> Ceres max_num_iterations). The function tolerance key is FTOL_REL
// (handled by dispatch_opt_attr -> Ceres function_tolerance): Ceres'
// function_tolerance is RELATIVE — it stops when (new_cost - old_cost) <
// ftol * old_cost. Ceres has NO absolute function tolerance, so FTOL_ABS (and
// the bare FTOL) are rejected here with a pointer to FTOL_REL. (FTOL_ABS is
// caught post-parse in RefitParser::parse, since dispatch_opt_attr consumes it
// before this hook runs; the bare FTOL reaches here.)
bool ceres_extra(const std::string &key, std::istringstream &iss,
                 OptimizerSpec &a) {
  if (key == "GTOL") {
    double v;
    iss >> v;
    a.param["GTOL"] = v;
    return true;
  }
  if (key == "MAXITER") {
    uint64_t v;
    iss >> v;
    a.stop.maxeval = v;
    return true;
  }
  if (key == "FTOL" || key == "FTOL_ABS")
    throw std::runtime_error(
        "REFINE/FUSE: '" + key +
        "' is not supported — Ceres' function_tolerance is RELATIVE and has no "
        "absolute form. Use FTOL_REL.");
  return false;
}

void parse_ceres_stage(const std::vector<std::string> &tok, OptimizerSpec &opt,
                       bool &has_opt, bool &enabled, const char *stage) {
  size_t i = 0;
  while (i < tok.size()) {
    const std::string u = up(tok[i]);
    if (u == "OPTIMIZER") {
      read_optimizer_block(tok, i, opt, ceres_extra);
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
                    cfg.refine.has_opt, cfg.refine.enabled, "REFINE");
  parse_ceres_stage(stage_tokens(c, "FUSE"), cfg.fuse.opt, cfg.fuse.has_opt,
                    cfg.fuse.enabled, "FUSE");

  // Validate (and surface) the per-stage ALGO choices early — these throw on an
  // illegal LIB/ALGO pair so the user sees the error before any fitting.
  if (cfg.explore.has_opt) {
    (void)refit_explore_outer_algo(cfg.explore.opt);
    if (cfg.explore.opt.inner)
      (void)refit_explore_inner_algo(*cfg.explore.opt.inner);
  }
  if (cfg.refine.has_opt) (void)refit_refine_algo(cfg.refine.opt);
  if (cfg.fuse.has_opt) (void)refit_fuse_algo(cfg.fuse.opt);

  // Reject FTOL_ABS on the Ceres stages: dispatch_opt_attr consumes it (it is a
  // valid attribute for the NLopt EXPLORE stage) before ceres_extra can, so the
  // guard there never fires for it — catch it here. Ceres has no absolute
  // function tolerance; FTOL_REL is the only function-tolerance key.
  auto reject_ftol_abs = [](const OptimizerSpec &o, const char *stage) {
    if (o.stop.ftol_abs)
      throw std::runtime_error(
          std::string(stage) +
          ": FTOL_ABS is not supported — Ceres' function_tolerance is RELATIVE "
          "and has no absolute form. Use FTOL_REL.");
  };
  if (cfg.refine.has_opt) reject_ftol_abs(cfg.refine.opt, "REFINE");
  if (cfg.fuse.has_opt) reject_ftol_abs(cfg.fuse.opt, "FUSE");

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
