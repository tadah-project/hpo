#include <tadah/hpo/optimizer_block.h>
#include <tadah/hpo/strict_parse.h>
#include <tadah/core/utils.h>
#include <tadah/core/number_format.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

bool parse_bool(const std::string &v, const std::string &ctx) {
  std::string lo;
  lo.reserve(v.size());
  for (char c : v) lo.push_back(static_cast<char>(std::tolower(c)));
  if (lo == "true" || lo == "1" || lo == "yes" || lo == "on") return true;
  if (lo == "false" || lo == "0" || lo == "no" || lo == "off") return false;
  throw std::runtime_error(
      ctx + ": expected boolean (true/false/1/0/yes/no/on/off), got '" + v +
      "'");
}

const std::vector<std::string> &opt_attr_keys() {
  static const std::vector<std::string> keys = {
      "LIB",      "ALGO",     "MAXEVAL",       "MAXTIME",  "STOPVAL",
      "FTOL_REL", "FTOL_ABS", "XTOL_REL",      "XTOL_ABS", "POPULATION",
      "STEP",     "SEED",     "VECTOR_ARR",    "THREADS",  "GRAD_STRATEGY",
      "PARAM"};
  return keys;
}

std::string opt_attr_keys_joined(const char *sep) {
  std::string out;
  for (const auto &k : opt_attr_keys()) {
    if (!out.empty()) out += sep;
    out += k;
  }
  return out;
}

namespace {
std::string up(std::string s) {
  for (char &c : s) c = static_cast<char>(std::toupper((unsigned char)c));
  return s;
}
}  // namespace

bool param_takes_another_value(const std::string &key,
                               const std::vector<std::string> &sofar,
                               const std::string &next) {
  /* Value grammar for `PARAM <key> ...`, family-agnostic on purpose:
   * optimizer_block.cpp is shared with `tadah refit` and must not include an
   * optimiser family's key registry. The annealers' registry is the authority
   * on what the tokens MEAN; this decides only how many to read.
   *
   * It must be a GRAMMAR and not a maximum count. read_optimizer_block()
   * presents the whole block as one flat token stream, so a key with "up to 3
   * values" would happily read `JITTER SEED 1234` out of
   *   PARAM INIT_MODE JITTER   SEED 1234
   * and silently consume the next key. Deciding per token -- is this one
   * actually part of my value? -- is correct in both framings. */
  if (sofar.empty()) return true;          // every PARAM takes >= 1 value
  if (key != "INIT_MODE") return false;

  // INIT_MODE UNIFORM | JITTER [AUTO <p> | <frac>]. A token shaped like a
  // number is the fraction; the registry judges it (the number grammar).
  if (up(sofar[0]) != "JITTER") return false;
  if (sofar.size() == 1) return up(next) == "AUTO" || is_number_shaped(next);
  if (sofar.size() == 2 && up(sofar[1]) == "AUTO") return true;   // the p value
  return false;
}

namespace {
/* The value of a numeric attribute: exactly one token, parsed strictly
 * (strict_parse.h). What follows it is the caller's: HPOTargets::read()
 * rejects a leftover token on the line, and read_optimizer_block() reads
 * the next key there. */
std::uint64_t attr_uint(std::istringstream &iss, const std::string &key,
                        std::uint64_t max =
                            std::numeric_limits<std::uint64_t>::max(),
                        std::uint64_t min = 0) {
  return parse_uint(next_value(iss, key), key, min, max);
}
unsigned attr_unsigned(std::istringstream &iss, const std::string &key) {
  return static_cast<unsigned>(
      attr_uint(iss, key, std::numeric_limits<unsigned>::max()));
}
/* MAXEVAL is an evaluation budget, and NLopt counts evaluations in an int
 * (nlopt::opt::set_maxeval), as `tadah refit` hands it to NLopt and Ceres:
 * 1 to INT_MAX. MAXEVAL 4294967299 used to reach NLopt as 3. MAXEVAL 0
 * meant something different to every backend (no limit to NLopt, no
 * evaluation to TADAH:RANDOM, the annealers and ASA, an error to DLIB:BFGS
 * and CERES); the key is left out for no limit, which a backend without a
 * stopping rule of its own refuses at startup (backend_rules.h). */
std::uint64_t attr_budget(std::istringstream &iss, const std::string &key) {
  return attr_uint(iss, key, std::numeric_limits<int>::max(), 1);
}
/* THREADS is the width of the worker pool (an int, ScriptExecutor): at
 * least one worker. THREADS 0 used to become one worker without a word. */
unsigned attr_threads(std::istringstream &iss, const std::string &key) {
  return static_cast<unsigned>(
      attr_uint(iss, key, std::numeric_limits<int>::max(), 1));
}
double attr_finite(std::istringstream &iss, const std::string &key) {
  return parse_finite(next_value(iss, key), key);
}
/* MAXTIME, the optimiser's whole-run wall time: a positive number of
 * seconds; the key is left out for no limit. 0 or a negative value used to
 * mean no limit to NLopt and TADAH and a startup error to ASA and the
 * annealers. */
double attr_seconds(std::istringstream &iss, const std::string &key) {
  return parse_positive(next_value(iss, key), key, "number of seconds");
}
}  // namespace

bool dispatch_opt_attr(const std::string &key, std::istringstream &iss,
                       OptimizerSpec &a) {
  if (key == "LIB")
    a.lib = next_value(iss, key);
  else if (key == "ALGO")
    a.algo = next_value(iss, key);
  else if (key == "MAXEVAL")
    a.stop.maxeval = attr_budget(iss, key);
  else if (key == "MAXTIME")
    a.stop.maxtime = attr_seconds(iss, key);
  else if (key == "STOPVAL")
    a.stop.stopval = attr_finite(iss, key);
  else if (key == "FTOL_REL")
    a.stop.ftol_rel = attr_finite(iss, key);
  else if (key == "FTOL_ABS")
    a.stop.ftol_abs = attr_finite(iss, key);
  else if (key == "XTOL_REL")
    a.stop.xtol_rel = attr_finite(iss, key);
  else if (key == "XTOL_ABS")
    a.stop.xtol_abs = attr_finite(iss, key);
  else if (key == "POPULATION")
    a.pop = attr_unsigned(iss, key);
  else if (key == "STEP") {
    /* Two forms:
     *   STEP <number>                      -- the block-wide fraction
     *   STEP <KEY>[(indices)] <number>     -- per-key / per-index override
     * A token shaped like a number is the block-wide fraction and must be
     * a finite one (parse_finite): `STEP inf` used to pass here and fail
     * only when the run validated it, without the line, and `STEP 5x` was
     * taken for an override key. The range, 0 < STEP <= 0.5, is the
     * backend's (validate_optimizer_scaling). */
    const std::string first = next_value(iss, key);
    if (is_number_shaped(first)) {
      a.step = parse_finite(first, key);
    } else {
      StepOverride ov;
      /* KEY, KEY(3), KEY(1-4,7) or KEY (1,2) with a space before '(' */
      std::string spec = first;
      std::string rest_tok;
      if (spec.find('(') == std::string::npos) {
        std::streampos save = iss.tellg();
        if (iss >> rest_tok && !rest_tok.empty() && rest_tok.front() == '(') {
          spec += rest_tok;
        } else {
          iss.clear();
          iss.seekg(save);
        }
      }
      const std::size_t lp = spec.find('(');
      if (lp != std::string::npos) {
        const std::size_t rp = spec.find(')', lp);
        if (rp == std::string::npos || rp != spec.size() - 1)
          throw std::runtime_error(
              "STEP override '" + spec + "': expected KEY(indices) <frac>");
        ov.key = spec.substr(0, lp);
        const std::string idx = spec.substr(lp + 1, rp - lp - 1);
        if (idx.empty())
          throw std::runtime_error(
              "STEP override '" + spec + "': empty index list");
        /* The strict grammar of the OPTIM lines: the lenient
         * core::parse_indices dropped a bad segment with a message on
         * stderr, and a list left empty meant every index of the key. */
        try {
          ov.indices = core::parse_indices_strict(idx);
          if (std::find(ov.indices.begin(), ov.indices.end(), 0) !=
              ov.indices.end())
            throw std::runtime_error("indices start at 1");
        } catch (const std::exception& e) {
          throw std::runtime_error("STEP override '" + spec +
                                   "': bad index list (" + e.what() + ")");
        }
      } else {
        ov.key = spec;
      }
      const std::string what = "STEP " + spec;
      ov.frac = parse_finite(next_value(iss, what), what);
      a.step_overrides.push_back(std::move(ov));
    }
  } else if (key == "SEED")
    a.seed = attr_uint(iss, key);
  else if (key == "VECTOR_ARR")
    a.vector_storage = attr_unsigned(iss, key);
  else if (key == "THREADS")
    a.num_threads = attr_threads(iss, key);
  else if (key == "GRAD_STRATEGY") {
    std::string v;
    iss >> v;
    if (v == "NUMERICAL")
      a.grad_strategy = GradStrategy::Numerical;
    else if (v == "ANALYTICAL")
      a.grad_strategy = GradStrategy::Analytical;
    else if (v == "HYBRID")
      a.grad_strategy = GradStrategy::Hybrid;
    else if (v == "AUTODIFF")
      throw std::runtime_error(
          "GRAD_STRATEGY=AUTODIFF was renamed to ANALYTICAL in Phase 2; "
          "the prior stub now backs a closed-form analytical gradient. "
          "Use GRAD_STRATEGY ANALYTICAL (strict -- every theta must have a "
          "registered analytical derivative) or GRAD_STRATEGY HYBRID "
          "(analytical-where-registered + finite-difference fallback).");
    else
      throw std::runtime_error(
          "GRAD_STRATEGY must be one of NUMERICAL, ANALYTICAL, HYBRID "
          "(got '" +
          v + "')");
  } else if (key == "PARAM") {
    std::string p;
    iss >> p;
    /* Read at most this key's declared arity. Reading "the rest" is NOT
     * an option: read_block_until() hands us a flat token stream over the
     * whole block, so everything after the value -- the next key, the
     * ENDSTRATEGY terminator -- is still in `iss`. A bounded read is
     * correct in both framings; the line-framed caller
     * (HPOTargets::read) additionally rejects whatever it leaves behind. */
    std::vector<std::string> vals;
    const std::string ukey = up(p);
    for (;;) {
      const std::streampos save = iss.tellg();
      std::string t;
      if (!(iss >> t)) { iss.clear(); break; }
      if (!param_takes_another_value(ukey, vals, t)) {
        /* Not ours -- put it back for the caller. Same rewind idiom the STEP
         * override branch above uses. */
        iss.clear();
        iss.seekg(save);
        break;
      }
      vals.push_back(std::move(t));
    }
    if (p.empty() || vals.empty())
      throw std::runtime_error("PARAM requires a name and a value");
    const std::string& raw = vals.front();
    /* Numeric values land in `param`; a word is kept verbatim in
     * `param_raw` for families with enum-valued keys (numeric-only
     * families reject raw entries with a hard error). The pre-registry
     * parser silently stored 0.0 for a non-numeric value. A value shaped
     * like a number must be a finite one: `PARAM fd_step inf` failed only
     * when the optimiser ran, and `PARAM fd_step 5x` was kept as a word
     * that no backend read. */
    if (is_number_shaped(raw))
      a.param[p] = parse_finite(raw, "PARAM " + p);
    else
      a.param_raw[p] = raw;
    /* The numbers of INIT_MODE JITTER <fraction> | JITTER AUTO <tolerance>
     * (param_takes_another_value takes a number-shaped token as one) are
     * read with the number grammar here, at the line; the registry judges
     * their range when the annealer starts. */
    if (ukey == "INIT_MODE")
      for (std::size_t i = 1; i < vals.size(); ++i)
        if (up(vals[i]) != "AUTO")
          (void)parse_finite(vals[i],
                             i == 2 ? "PARAM " + p + " JITTER AUTO <tolerance>"
                                    : "PARAM " + p + " JITTER <fraction>");
    /* Extra value tokens are kept for the family registry to accept or
     * reject (validate_optimizer_scaling). They used to be swallowed by
     * `iss >> p >> raw`, so `PARAM INIT_MODE JITTER AUTO 0.05` silently
     * ran at the default. Recorded only when there IS an extra, so the
     * map stays empty for every single-value config. */
    if (vals.size() > 1) a.param_tokens[p] = std::move(vals);
  } else
    return false;
  return true;
}

namespace {

/* "<where>: <message>", or the message alone when the caller names no
 * block. */
std::runtime_error in_block(const std::string &where, const std::string &msg) {
  return std::runtime_error(where.empty() ? msg : where + ": " + msg);
}

// Consume attributes from a flat token stream until `endTok` (case-insensitive)
// is reached. Standard attributes go to dispatch_opt_attr(); an `INNER` opener
// recurses into spec.inner (only at the OPTIMIZER level — INNER does not nest).
// Every error of the block, the check's and a key's value's included, is
// prefixed with `where` ("REFINE", "EXPLORE INNER"). On return `pos` points
// just past `endTok`.
void read_block_until(const std::vector<std::string> &tok, std::size_t &pos,
                      const std::string &endTok, OptimizerSpec &spec,
                      const ExtraAttrFn &extra, const KeyCheckFn &check,
                      bool allow_inner, const std::string &where) {
  const char *block = allow_inner ? "OPTIMIZER" : "INNER";
  while (pos < tok.size()) {
    const std::string u = up(tok[pos]);
    if (u == endTok) {
      ++pos;
      return;
    }
    // The caller's gate sees every key, INNER included, before it is read.
    try {
      if (check) check(u, /*in_inner=*/!allow_inner);
    } catch (const std::exception &e) {
      throw in_block(where, e.what());
    }
    if (allow_inner && u == "INNER") {
      ++pos;  // consume INNER
      spec.inner = std::make_unique<OptimizerSpec>();
      read_block_until(tok, pos, "ENDINNER", *spec.inner, extra, check,
                       /*allow_inner=*/false,
                       where.empty() ? where : where + " INNER");
      continue;
    }
    // Standard / command-specific attribute. dispatch_opt_attr() reads a fixed
    // (small) number of value tokens from a stream built over the remaining
    // tokens; we then recover how many it consumed by counting the leftover.
    std::string rest;
    for (std::size_t j = pos; j < tok.size(); ++j) rest += tok[j] + " ";
    std::istringstream iss(rest);
    std::string key;
    iss >> key;  // the attribute key (original case)
    const std::string ukey = up(key);
    bool consumed = false;
    try {
      consumed = dispatch_opt_attr(ukey, iss, spec) ||
                 (extra && extra(ukey, iss, spec));
    } catch (const std::exception &e) {
      throw in_block(where, e.what());
    }
    if (!consumed)
      throw in_block(where, "unknown key '" + key + "' in the " +
                                std::string(block) + " block");
    if (iss.fail())
      throw in_block(where, "bad value for '" + key + "' in the " +
                                std::string(block) + " block");
    std::size_t before = tok.size() - pos;  // key + everything after it
    std::size_t remaining = 0;
    std::string t;
    while (iss >> t) ++remaining;
    pos += (before - remaining);
  }
  throw in_block(where, std::string(block) + " block not closed with " +
                            endTok);
}

}  // namespace

void read_optimizer_block(const std::vector<std::string> &tok,
                          std::size_t &pos, OptimizerSpec &spec,
                          const ExtraAttrFn &extra, const KeyCheckFn &check,
                          const std::string &where) {
  if (pos >= tok.size() || up(tok[pos]) != "OPTIMIZER")
    throw std::runtime_error("read_optimizer_block: expected OPTIMIZER");
  ++pos;  // consume OPTIMIZER
  read_block_until(tok, pos, "ENDOPTIMIZER", spec, extra, check,
                   /*allow_inner=*/true, where);
}

}  // namespace hpo
}  // namespace tadah
