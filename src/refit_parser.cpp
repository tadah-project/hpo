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

// The keys some ALGO of a stage reads, in its OPTIMIZER block (or its INNER
// block). A stage reads only some of the optimiser attributes of `tadah hpo`;
// any other one it would silently ignore is an error naming the keys it does
// read (the rule of the `tadah hpo` STRATEGY keys that would have no effect).
// This is the first gate, as each key is read; once the block's ALGO is known
// a key that ALGO does not read is an error too (reject_keys_not_read).
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

// The keys a stage's OPTIMIZER block and its INNER block set, in order and
// in the spelling used (MAXITER, not MAXEVAL): a key that has no effect with
// the block's ALGO is named as written (reject_keys_not_read).
struct SeenKeys {
  std::vector<std::string> outer, inner;
};

// read_optimizer_block() prefixes these messages with the block
// ("REFINE: ", "EXPLORE INNER: ").
KeyCheckFn stage_key_check(const StageKeys &sk, SeenKeys *seen) {
  return [sk, seen](const std::string &key, bool in_inner) {
    const std::vector<std::string> &reads = in_inner ? sk.inner : sk.outer;
    if (std::find(reads.begin(), reads.end(), key) != reads.end()) {
      (in_inner ? seen->inner : seen->outer).push_back(key);
      return;
    }
    if (!known_optimizer_key(key)) return;
    // SEED and FTOL_ABS (in FUSE, where no ALGO reads it) say what to use
    // instead.
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

// FTOL_ABS: dlib's LBFGS (EXPLORE INNER, REFINE) stops on it.
const StageKeys kExploreKeys{
    "EXPLORE",
    {"LIB", "ALGO", "MAXEVAL", "INNER"},
    {"LIB", "ALGO", "MAXEVAL", "FTOL_REL", "FTOL_ABS", "XTOL_REL"}};
const StageKeys kRefineKeys{"REFINE",
                            {"LIB", "ALGO", "MAXEVAL", "MAXITER", "FTOL_REL",
                             "FTOL_ABS", "GTOL"},
                            {}};
const StageKeys kFuseKeys{
    "FUSE", {"LIB", "ALGO", "MAXEVAL", "MAXITER", "FTOL_REL", "GTOL"}, {}};

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

// The tolerances of every refit block, FTOL_REL, FTOL_ABS, XTOL_REL and GTOL:
// a finite number (parse_finite, whose message a malformed token keeps) that
// is >= 0, named as written: "GTOL must be a number >= 0, got '-1'". Ceres
// refuses a negative tolerance in its option check and returns FAILURE
// without iterating, so a REFINE or FUSE with one did nothing; NLopt switches
// a tolerance below 0 off, and dlib requires one >= 0. 0 switches the test
// off. read_optimizer_block() offers the key to the stage's hook before
// dispatch_opt_attr(), which reads any finite number.
double tolerance_value(const std::string &key, std::istringstream &iss) {
  const std::string tok = next_value(iss, key);
  const double v = parse_finite(tok, key);
  if (v < 0.0)
    throw std::runtime_error(key + " must be a number >= 0, got '" + tok + "'");
  return v;
}

bool tolerance_attr(const std::string &key, std::istringstream &iss,
                    OptimizerSpec &a) {
  if (key == "FTOL_REL") a.stop.ftol_rel = tolerance_value(key, iss);
  else if (key == "FTOL_ABS") a.stop.ftol_abs = tolerance_value(key, iss);
  else if (key == "XTOL_REL") a.stop.xtol_rel = tolerance_value(key, iss);
  else if (key == "GTOL") a.param["GTOL"] = tolerance_value(key, iss);
  else return false;
  return true;
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

void parse_explore(const std::vector<std::string> &tok, RefitConfig &cfg,
                   SeenKeys &seen) {
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
      read_optimizer_block(tok, i, cfg.explore.opt, tolerance_attr,
                           stage_key_check(kExploreKeys, &seen),  // + INNER
                           kExploreKeys.stage);
      cfg.explore.has_opt = true;
    } else {
      throw std::runtime_error("EXPLORE: unknown key '" + tok[i] +
                               "' (expected MAX_ADD, ENABLED or OPTIMIZER)");
    }
  }
}

// REFINE/FUSE OPTIMIZER blocks accept the Ceres-flavoured command-specific keys
// GTOL (-> param["GTOL"], Ceres gradient_tolerance) and MAXITER (alias of
// MAXEVAL -> Ceres max_num_iterations). The function tolerance key is
// FTOL_REL (-> Ceres function_tolerance). GTOL, FTOL_REL and FTOL_ABS (dlib's
// LBFGS in REFINE) are tolerances, read by tolerance_attr. Ceres' function_tolerance is RELATIVE — it stops when
// (new_cost - old_cost) < ftol * old_cost. Ceres has NO absolute function
// tolerance, so FTOL_ABS with a Ceres ALGO is rejected with a pointer to
// FTOL_REL (in FUSE by the stage's key check, stage_key_check, before it is
// read; in REFINE once the ALGO is known, reject_keys_not_read), and so is
// the bare FTOL here.
bool ceres_extra(const std::string &key, std::istringstream &iss,
                 OptimizerSpec &a) {
  if (key == "FTOL")
    throw std::runtime_error(
        "'" + key +
        "' is not supported — Ceres' function_tolerance is RELATIVE and has no "
        "absolute form. Use FTOL_REL.");
  if (tolerance_attr(key, iss, a)) return true;  // GTOL among them
  if (key == "MAXITER") {
    // The alias of MAXEVAL, with its range: the int iteration cap Ceres
    // takes (apply_stage_optimizers), at least one iteration.
    a.stop.maxeval = parse_uint(next_value(iss, key), key, 1,
                                std::numeric_limits<int>::max());
    return true;
  }
  return false;
}

void parse_ceres_stage(const std::vector<std::string> &tok, OptimizerSpec &opt,
                       bool &has_opt, bool &enabled, const StageKeys &keys,
                       SeenKeys &seen) {
  const char *stage = keys.stage;
  size_t i = 0;
  while (i < tok.size()) {
    const std::string u = up(tok[i]);
    if (u == "OPTIMIZER") {
      read_optimizer_block(tok, i, opt, ceres_extra,
                           stage_key_check(keys, &seen), stage);
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

// The library an ALGO spelling belongs to, as `tadah hpo` names it; none
// for NONE and UNIFORM, which run no optimiser.
constexpr const char *kNoOptimiser = "";

// One ALGO spelling of a stage: the compact fitter's algorithm it selects,
// the library the stage block must name in LIB, and whether the stage's
// legal list names it (the NLopt long names, and the dlib and Ceres names).
struct AlgoSpelling {
  const char *spelling;   // uppercase
  const char *canonical;  // the compact fitter's name
  const char *lib;        // NLOPT | DLIB | CERES, or kNoOptimiser
  bool listed;            // named by the stage's legal list
};
constexpr bool kListed = true;
constexpr bool kAlias = false;

// Every spelling of each stage (the NLopt long names and the fitter's short
// names alike), with the library of `tadah hpo` it belongs to. The ALGO
// names are the refit's own: its dlib global search is GLOBAL (`tadah hpo`
// calls dlib's GFS), and its LBFGS is dlib's L-BFGS (a CERES algorithm in
// `tadah hpo`; NLopt's is LD_LBFGS). UNIFORM, the refit's own uniform
// start, runs no global search: like NONE, it belongs to no library.
const std::vector<AlgoSpelling> kExploreOuterAlgos = {
    {"UNIFORM", "UNIFORM", kNoOptimiser, kListed},
    {"NONE", "UNIFORM", kNoOptimiser, kAlias},
    {"GLOBAL", "DLIB_GLOBAL", "DLIB", kListed},
    {"DLIB_GLOBAL", "DLIB_GLOBAL", "DLIB", kAlias},
    {"MLSL", "MLSL", "NLOPT", kAlias},
    {"GN_MLSL_LDS", "MLSL", "NLOPT", kListed},
    {"DIRECT", "DIRECT", "NLOPT", kAlias},
    {"GN_DIRECT_L", "DIRECT", "NLOPT", kListed},
    {"CRS2", "CRS2", "NLOPT", kAlias},
    {"GN_CRS2_LM", "CRS2", "NLOPT", kListed}};

// NLopt spellings the EXPLORE optimiser refuses, with the algorithm the refit
// runs in their place. They were aliases of it: GN_MLSL ran GN_MLSL_LDS,
// GN_DIRECT GN_DIRECT_L and GN_CRS2 GN_CRS2_LM. GN_MLSL and GN_DIRECT are
// other NLopt algorithms, which `tadah hpo` runs as named, and GN_CRS2 is
// none. The refit's own short names (MLSL, DIRECT, CRS2) stay.
struct RefusedSpelling {
  const char *spelling;  // uppercase
  const char *what;      // what the spelling is
  const char *short_name;
  const char *runs;      // the refit's algorithm of that name
};
const std::vector<RefusedSpelling> kExploreRefusedAlgos = {
    {"GN_MLSL", "NLopt's MLSL with pseudo-random starts", "MLSL",
     "GN_MLSL_LDS"},
    {"GN_DIRECT", "NLopt's DIRECT without the local bias of DIRECT_L",
     "DIRECT", "GN_DIRECT_L"},
    {"GN_CRS2", "not an NLopt algorithm", "CRS2", "GN_CRS2_LM"}};
const std::vector<AlgoSpelling> kExploreInnerAlgos = {
    {"NONE", "NONE", kNoOptimiser, kListed},
    {"LBFGS", "DLIB_LBFGS", "DLIB", kListed},
    {"DLIB_LBFGS", "DLIB_LBFGS", "DLIB", kAlias},
    {"SBPLX", "SBPLX", "NLOPT", kAlias},
    {"LN_SBPLX", "SBPLX", "NLOPT", kListed},
    {"NELDERMEAD", "NELDERMEAD", "NLOPT", kAlias},
    {"LN_NELDERMEAD", "NELDERMEAD", "NLOPT", kListed},
    {"BOBYQA", "BOBYQA", "NLOPT", kAlias},
    {"LN_BOBYQA", "BOBYQA", "NLOPT", kListed},
    {"PRAXIS", "PRAXIS", "NLOPT", kAlias},
    {"LN_PRAXIS", "PRAXIS", "NLOPT", kListed},
    {"NLOPT_LBFGS", "NLOPT_LBFGS", "NLOPT", kAlias},
    {"LD_LBFGS", "NLOPT_LBFGS", "NLOPT", kListed}};
const std::vector<AlgoSpelling> kRefineAlgos = {
    {"NONE", "NONE", kNoOptimiser, kListed},
    {"LM", "CERES_LM", "CERES", kListed},
    {"CERES_LM", "CERES_LM", "CERES", kAlias},
    {"DOGLEG", "CERES_DOGLEG", "CERES", kListed},
    {"CERES_DOGLEG", "CERES_DOGLEG", "CERES", kAlias},
    {"LBFGS", "DLIB_LBFGS", "DLIB", kListed},
    {"DLIB_LBFGS", "DLIB_LBFGS", "DLIB", kAlias},
    {"LD_LBFGS", "NLOPT_LBFGS", "NLOPT", kListed},
    {"NLOPT_LBFGS", "NLOPT_LBFGS", "NLOPT", kAlias}};
const std::vector<AlgoSpelling> kFuseAlgos = {
    {"NONE", "NONE", kNoOptimiser, kListed},
    {"LM", "CERES_LM", "CERES", kListed},
    {"CERES_LM", "CERES_LM", "CERES", kAlias},
    {"DOGLEG", "CERES_DOGLEG", "CERES", kListed},
    {"CERES_DOGLEG", "CERES_DOGLEG", "CERES", kAlias}};

// The listed algorithms of a stage in library `lib`, "A|B", or "" when the
// stage has none there (or `lib` is not a library).
std::string listed_in(const std::vector<AlgoSpelling> &algos,
                      const std::string &lib) {
  std::string out;
  for (const auto &a : algos) {
    if (!a.listed || *a.lib == '\0' || lib != a.lib) continue;
    if (!out.empty()) out += "|";
    out += a.spelling;
  }
  return out;
}

std::string library_kind(const std::string &lib) {
  if (lib == "NLOPT") return "an NLopt algorithm";
  if (lib == "DLIB") return "a dlib algorithm";
  return "a Ceres algorithm";
}

// The fitter's algorithm for `s` in a stage, with the LIB rule of `tadah hpo`:
// a block that runs an optimiser names the library of its ALGO in LIB (a
// missing or different LIB is an error), and a block with ALGO NONE or
// UNIFORM runs none, so a LIB there would have no effect. `block` names the
// block in the LIB messages ("EXPLORE INNER"), `stage` in the
// unsupported-ALGO one ("EXPLORE inner"); `none_means` says what ALGO NONE
// (or UNIFORM) does in the block.
std::string stage_algo(const OptimizerSpec &s,
                       const std::vector<AlgoSpelling> &algos,
                       const char *block, const char *stage,
                       const char *none_means, const char *legal) {
  const std::string lib = up(s.lib == "NOT SET" ? "" : s.lib);
  const std::string algo = up(s.algo == "NOT SET" ? "" : s.algo);
  for (const auto &a : algos) {
    if (algo != a.spelling) continue;
    if (std::string(a.lib).empty()) {  // ALGO NONE or UNIFORM
      if (!lib.empty())
        throw std::runtime_error(std::string(block) +
                                 ": LIB has no effect with ALGO " +
                                 a.spelling + " (" + none_means +
                                 "); remove LIB");
      return a.canonical;
    }
    if (lib.empty())
      throw std::runtime_error(std::string(block) + ": LIB is not set; set LIB " +
                               a.lib + " (ALGO " + s.algo + " is " +
                               library_kind(a.lib) + ")");
    if (lib != a.lib) {
      // The LIB may be the one meant and the ALGO the slip (LIB NLOPT ALGO
      // LBFGS for NLopt's L-BFGS): name that library's algorithms of the
      // stage too, when it has any.
      const std::string in_lib = listed_in(algos, lib);
      throw std::runtime_error(
          std::string(block) + ": LIB " + s.lib + " does not match ALGO " +
          s.algo + ", " + library_kind(a.lib) + "; set LIB " + a.lib +
          (in_lib.empty() ? std::string()
                          : ", or with LIB " + lib + " set ALGO " + in_lib));
    }
    return a.canonical;
  }
  algo_err(stage, lib, algo, legal);
}

// What ALGO NONE (UNIFORM in EXPLORE) does in each block, for the messages.
constexpr const char *kExploreOff = "no global search runs";
constexpr const char *kInnerOff = "no local search runs";
constexpr const char *kStageOff = "the stage is skipped";

// The stop criteria each ALGO of a block reads (beyond LIB and ALGO), by
// the compact fitter's name of the ALGO; a block's key outside its ALGO's
// set would have no effect and is an error (reject_keys_not_read). What
// CompactBlipFitter does with each:
//  - EXPLORE seed: MLSL, DIRECT, CRS2 and DLIB_GLOBAL their MAXEVAL; UNIFORM
//    runs no search;
//  - EXPLORE INNER: the NLopt searches MAXEVAL, FTOL_REL and XTOL_REL;
//    dlib's LBFGS MAXEVAL and FTOL_ABS (objective_delta_stop_strategy, an
//    absolute change of the objective); NONE runs no search. Under the
//    EXPLORE optimiser MLSL the block's MAXEVAL, FTOL_REL and XTOL_REL also
//    set MLSL's local LN_SBPLX searches (kMlslLocalReads), whatever its ALGO;
//  - REFINE: Ceres' LM and DOGLEG MAXEVAL (alias MAXITER: iterations),
//    FTOL_REL and GTOL; NLopt's LD_LBFGS MAXEVAL and FTOL_REL; dlib's LBFGS
//    MAXEVAL and FTOL_ABS; NONE skips the stage;
//  - FUSE: Ceres' LM and DOGLEG MAXEVAL/MAXITER, FTOL_REL and GTOL.
struct AlgoReads {
  const char *canonical;
  std::vector<std::string> reads;
};
const std::vector<std::string> kCeresReads = {"MAXEVAL", "MAXITER", "FTOL_REL",
                                              "GTOL"};
const std::vector<std::string> kNloptLocalReads = {"MAXEVAL", "FTOL_REL",
                                                   "XTOL_REL"};
const std::vector<std::string> kDlibLbfgsReads = {"MAXEVAL", "FTOL_ABS"};
const std::vector<std::string> kMlslLocalReads = kNloptLocalReads;
const std::vector<AlgoReads> kExploreOuterReads = {
    {"UNIFORM", {}},
    {"MLSL", {"MAXEVAL"}},
    {"DIRECT", {"MAXEVAL"}},
    {"CRS2", {"MAXEVAL"}},
    {"DLIB_GLOBAL", {"MAXEVAL"}}};
const std::vector<AlgoReads> kExploreInnerReads = {
    {"NONE", {}},
    {"SBPLX", kNloptLocalReads},
    {"NELDERMEAD", kNloptLocalReads},
    {"BOBYQA", kNloptLocalReads},
    {"PRAXIS", kNloptLocalReads},
    {"NLOPT_LBFGS", kNloptLocalReads},
    {"DLIB_LBFGS", kDlibLbfgsReads}};
const std::vector<AlgoReads> kRefineReads = {
    {"NONE", {}},
    {"CERES_LM", kCeresReads},
    {"CERES_DOGLEG", kCeresReads},
    {"NLOPT_LBFGS", {"MAXEVAL", "FTOL_REL"}},
    {"DLIB_LBFGS", kDlibLbfgsReads}};
const std::vector<AlgoReads> kFuseReads = {{"NONE", {}},
                                           {"CERES_LM", kCeresReads},
                                           {"CERES_DOGLEG", kCeresReads}};

std::vector<std::string> reads_of(const std::vector<AlgoReads> &table,
                                  const std::string &canonical) {
  for (const auto &a : table)
    if (canonical == a.canonical) return a.reads;
  throw std::logic_error("tadah refit: no read set for ALGO " + canonical +
                         " (internal error)");
}

bool has_key(const std::vector<std::string> &keys, const std::string &key) {
  return std::find(keys.begin(), keys.end(), key) != keys.end();
}

// Another optimiser that reads a block's keys besides the block's own ALGO:
// under the EXPLORE optimiser MLSL, MLSL's local LN_SBPLX searches read the
// INNER block's MAXEVAL, FTOL_REL and XTOL_REL. `outer` is the EXPLORE ALGO
// as written; empty `reads`: none.
struct AlsoReads {
  std::string outer;
  std::vector<std::string> reads;
};

// The first of `keys` (the block's, as written) other than LIB, ALGO and
// INNER that neither `reads` (what the block's ALGO reads, from the tables
// above) nor `also` reads is an error naming it and the ALGO. (LIB is the
// LIB rule's, stage_algo(); EXPLORE's INNER block configures the per-rung
// local search whatever the EXPLORE optimiser.) An ALGO that runs no
// optimiser (NONE, UNIFORM: `reads` is empty) says what it does instead
// (`off`); MAXITER, Ceres' iterations, with an ALGO that counts evaluations
// names MAXEVAL; FTOL_REL with dlib's LBFGS, which tests an absolute change,
// names FTOL_ABS; FTOL_ABS with a Ceres ALGO names FTOL_REL. Where `also`
// reads keys the ALGO does not, the message names them as MLSL's.
void reject_keys_not_read(const std::string &block, const OptimizerSpec &s,
                          const std::vector<std::string> &keys,
                          const std::vector<std::string> &reads,
                          const char *off, const AlsoReads &also = {}) {
  const std::string algo = up(s.algo);
  bool also_adds = false;
  for (const auto &k : also.reads)
    if (!has_key(reads, k)) also_adds = true;
  for (const auto &key : keys) {
    if (key == "LIB" || key == "ALGO" || key == "INNER") continue;
    if (has_key(reads, key) || has_key(also.reads, key)) continue;
    const std::string mlsl =
        also_adds ? "; under ALGO " + also.outer + " the block " +
                        (reads.empty() ? "sets" : "also sets") +
                        " MLSL's local LN_SBPLX searches, which read " +
                        reads_list(also.reads)
                  : std::string();
    if (reads.empty())
      throw std::runtime_error(block + ": " + key + " has no effect with ALGO " +
                               algo + " (" + off + mlsl + "); remove " + key);
    if (key == "MAXITER")
      throw std::runtime_error(block + ": MAXITER counts Ceres iterations; "
                               "ALGO " + algo + " counts objective "
                               "evaluations: write MAXEVAL");
    if (key == "FTOL_REL" && has_key(reads, "FTOL_ABS"))
      throw std::runtime_error(block + ": FTOL_REL has no effect with ALGO " +
                               algo + ": dlib's L-BFGS stops on an absolute "
                               "change of the objective; write FTOL_ABS");
    if (key == "FTOL_ABS" && has_key(reads, "GTOL"))
      throw std::runtime_error(
          block + ": FTOL_ABS is not supported — Ceres' function_tolerance is "
          "RELATIVE and has no absolute form. Use FTOL_REL.");
    throw std::runtime_error(block + ": " + key + " has no effect with ALGO " +
                             algo + " (it reads " + reads_list(reads) + mlsl +
                             "); remove " + key);
  }
}

// REFINE or FUSE with ENABLED false skips the stage, so an OPTIMIZER block
// beside it would have no effect.
void reject_disabled_with_optimizer(const char *stage, bool enabled,
                                    bool has_opt) {
  if (!enabled && has_opt)
    throw std::runtime_error(
        std::string(stage) + ": ENABLED false skips the stage, so its "
        "OPTIMIZER block has no effect; remove the OPTIMIZER block, or write "
        "ALGO NONE in it instead of ENABLED false");
}

}  // namespace

bool refit_curve_objective(const tadah::core::Context &c) {
  std::string obj =
      (c.contains("REFIT_OBJECTIVE") && c.size("REFIT_OBJECTIVE") > 0)
          ? c.get<std::string>("REFIT_OBJECTIVE")
          : std::string("energy");
  for (auto &ch : obj)
    ch = static_cast<char>(std::tolower(static_cast<unsigned char>(ch)));
  if (obj == "curve") return true;
  if (obj == "energy") return false;
  throw std::runtime_error(
      "tadah refit: REFIT_OBJECTIVE must be 'energy' (default) or 'curve'; "
      "got '" + obj + "'");
}

std::string refit_explore_outer_algo(const OptimizerSpec &s) {
  // A refused spelling first, before the LIB rule, whatever the LIB.
  const std::string algo = up(s.algo == "NOT SET" ? "" : s.algo);
  for (const auto &r : kExploreRefusedAlgos)
    if (algo == r.spelling)
      throw std::runtime_error(std::string("EXPLORE: ALGO ") + s.algo +
                               " is not run by tadah refit (" + r.what +
                               "); its " + r.short_name + " is " + r.runs +
                               ": set ALGO " + r.runs);
  return stage_algo(s, kExploreOuterAlgos, "EXPLORE", "EXPLORE outer",
                    kExploreOff,
                    "NLOPT GN_MLSL_LDS|GN_DIRECT_L|GN_CRS2_LM, DLIB GLOBAL, or "
                    "UNIFORM");
}

std::string refit_explore_inner_algo(const OptimizerSpec &s) {
  // An INNER block without ALGO (a spec built outside RefitParser, which
  // requires ALGO) runs no local search.
  if (s.algo == "NOT SET" || s.algo.empty()) return "NONE";
  return stage_algo(s, kExploreInnerAlgos, "EXPLORE INNER", "EXPLORE inner",
                    kInnerOff,
                    "NLOPT LN_SBPLX|LN_NELDERMEAD|LN_BOBYQA|LN_PRAXIS|LD_LBFGS, "
                    "DLIB LBFGS, or NONE");
}

std::string refit_refine_algo(const OptimizerSpec &s) {
  return stage_algo(s, kRefineAlgos, "REFINE", "REFINE", kStageOff,
                    "CERES LM|DOGLEG, NLOPT LD_LBFGS, DLIB LBFGS, or NONE");
}

std::string refit_fuse_algo(const OptimizerSpec &s) {
  return stage_algo(s, kFuseAlgos, "FUSE", "FUSE", kStageOff,
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
  SeenKeys explore_keys, refine_keys, fuse_keys;
  parse_init(stage_tokens(c, "INIT"), cfg);
  parse_explore(stage_tokens(c, "EXPLORE"), cfg, explore_keys);
  parse_ceres_stage(stage_tokens(c, "REFINE"), cfg.refine.opt,
                    cfg.refine.has_opt, cfg.refine.enabled, kRefineKeys,
                    refine_keys);
  parse_ceres_stage(stage_tokens(c, "FUSE"), cfg.fuse.opt, cfg.fuse.has_opt,
                    cfg.fuse.enabled, kFuseKeys, fuse_keys);

  // ENABLED false skips REFINE or FUSE: an OPTIMIZER block beside it would
  // have no effect (it used to be ignored with a WARNING).
  reject_disabled_with_optimizer("REFINE", cfg.refine.enabled,
                                 cfg.refine.has_opt);
  reject_disabled_with_optimizer("FUSE", cfg.fuse.enabled, cfg.fuse.has_opt);

  // Every OPTIMIZER block, and an INNER block, names its ALGO. The keys a
  // stage does not use (SEED, FTOL_ABS in FUSE, THREADS, ...) were rejected
  // as they were read (stage_key_check).
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

  // A stop criterion the block's ALGO does not read has no effect and is
  // rejected (reject_keys_not_read; with ALGO NONE or UNIFORM every one).
  // The one key set another stage reads: under the EXPLORE optimiser MLSL,
  // EXPLORE INNER's MAXEVAL, FTOL_REL and XTOL_REL also set MLSL's local
  // LN_SBPLX searches, whatever the INNER ALGO. (The FUSE curve refine of
  // REFIT_OBJECTIVE curve runs with FUSE's own settings, fuse_curve_config;
  // it ran with REFINE's, which were accepted there.)
  if (cfg.explore.has_opt) {
    const std::string outer = refit_explore_outer_algo(cfg.explore.opt);
    reject_keys_not_read("EXPLORE", cfg.explore.opt, explore_keys.outer,
                         reads_of(kExploreOuterReads, outer), kExploreOff);
    if (cfg.explore.opt.inner) {
      AlsoReads mlsl;
      if (outer == "MLSL") mlsl = {up(cfg.explore.opt.algo), kMlslLocalReads};
      reject_keys_not_read(
          "EXPLORE INNER", *cfg.explore.opt.inner, explore_keys.inner,
          reads_of(kExploreInnerReads,
                   refit_explore_inner_algo(*cfg.explore.opt.inner)),
          kInnerOff, mlsl);
    }
  }
  if (cfg.refine.has_opt)
    reject_keys_not_read("REFINE", cfg.refine.opt, refine_keys.outer,
                         reads_of(kRefineReads, refit_refine_algo(cfg.refine.opt)),
                         kStageOff);
  if (cfg.fuse.has_opt)
    reject_keys_not_read("FUSE", cfg.fuse.opt, fuse_keys.outer,
                         reads_of(kFuseReads, refit_fuse_algo(cfg.fuse.opt)),
                         kStageOff);

  // EXPLORE cannot be disabled (it builds the ladder) — still a hard error.
  if (!cfg.explore.enabled)
    throw std::runtime_error(
        "EXPLORE: ENABLED false is not allowed — EXPLORE builds the basis "
        "ladder and cannot be disabled.");

  return cfg;
}

}  // namespace refit
}  // namespace hpo
}  // namespace tadah
