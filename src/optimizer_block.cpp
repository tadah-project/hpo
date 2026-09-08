#include <tadah/hpo/optimizer_block.h>
#include <tadah/core/utils.h>

#include <cctype>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>

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

bool dispatch_opt_attr(const std::string &key, std::istringstream &iss,
                       OptimizerSpec &a) {
  if (key == "LIB")
    iss >> a.lib;
  else if (key == "ALGO")
    iss >> a.algo;
  else if (key == "MAXEVAL") {
    uint64_t v;
    iss >> v;
    a.stop.maxeval = v;
  } else if (key == "MAXTIME") {
    double v;
    iss >> v;
    a.stop.maxtime = v;
  } else if (key == "STOPVAL") {
    double v;
    iss >> v;
    a.stop.stopval = v;
  } else if (key == "FTOL_REL") {
    double v;
    iss >> v;
    a.stop.ftol_rel = v;
  } else if (key == "FTOL_ABS") {
    double v;
    iss >> v;
    a.stop.ftol_abs = v;
  } else if (key == "XTOL_REL") {
    double v;
    iss >> v;
    a.stop.xtol_rel = v;
  } else if (key == "XTOL_ABS") {
    double v;
    iss >> v;
    a.stop.xtol_abs = v;
  } else if (key == "POPULATION") {
    unsigned v;
    iss >> v;
    a.pop = v;
  } else if (key == "STEP") {
    /* Two forms:
     *   STEP <number>                      -- the block-wide fraction
     *   STEP <KEY>[(indices)] <number>     -- per-key / per-index override
     * A non-numeric first token that is not a key form used to leave v=0
     * and surface later as NLopt's 'zero step size'; both forms are now
     * parsed strictly. */
    std::string first;
    iss >> first;
    if (first.empty())
      throw std::runtime_error("STEP requires a value");
    auto parse_num = [](const std::string& tok, double& out) {
      try {
        std::size_t pos = 0;
        out = std::stod(tok, &pos);
        return pos == tok.size();
      } catch (const std::exception&) {
        return false;
      }
    };
    double v = 0.0;
    if (parse_num(first, v)) {
      a.step = v;
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
        try {
          ov.indices = core::parse_indices(idx);
        } catch (const std::exception& e) {
          throw std::runtime_error("STEP override '" + spec +
                                   "': bad index list (" + e.what() + ")");
        }
      } else {
        ov.key = spec;
      }
      std::string num;
      iss >> num;
      if (num.empty() || !parse_num(num, ov.frac))
        throw std::runtime_error(
            "STEP override for '" + ov.key + "' expects a numeric fraction "
            "(got '" + num + "')");
      a.step_overrides.push_back(std::move(ov));
    }
  } else if (key == "SEED") {
    std::uint64_t v;
    iss >> v;
    a.seed = v;
  } else if (key == "VECTOR_ARR") {
    unsigned v;
    iss >> v;
    a.vector_storage = v;
  } else if (key == "THREADS") {
    unsigned v;
    iss >> v;
    a.num_threads = v;
  } else if (key == "GRAD_STRATEGY") {
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
    std::string p, raw;
    iss >> p >> raw;
    if (p.empty() || raw.empty())
      throw std::runtime_error("PARAM requires a name and a value");
    /* Numeric values land in `param`; anything else is kept verbatim in
     * `param_raw` for families with enum-valued keys (numeric-only
     * families reject raw entries with a hard error). The pre-registry
     * parser silently stored 0.0 for a non-numeric value. */
    bool numeric = false;
    try {
      std::size_t pos = 0;
      const double v = std::stod(raw, &pos);
      if (pos == raw.size()) {
        a.param[p] = v;
        numeric = true;
      }
    } catch (const std::exception&) {
      /* not numeric */
    }
    if (!numeric) a.param_raw[p] = raw;
  } else
    return false;
  return true;
}

namespace {

std::string up(std::string s) {
  for (char &c : s) c = static_cast<char>(std::toupper((unsigned char)c));
  return s;
}

// Consume attributes from a flat token stream until `endTok` (case-insensitive)
// is reached. Standard attributes go to dispatch_opt_attr(); an `INNER` opener
// recurses into spec.inner (only at the OPTIMIZER level — INNER does not nest).
// On return `pos` points just past `endTok`.
void read_block_until(const std::vector<std::string> &tok, std::size_t &pos,
                      const std::string &endTok, OptimizerSpec &spec,
                      const ExtraAttrFn &extra, bool allow_inner) {
  while (pos < tok.size()) {
    const std::string u = up(tok[pos]);
    if (u == endTok) {
      ++pos;
      return;
    }
    if (allow_inner && u == "INNER") {
      ++pos;  // consume INNER
      spec.inner = std::make_unique<OptimizerSpec>();
      read_block_until(tok, pos, "ENDINNER", *spec.inner, extra,
                       /*allow_inner=*/false);
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
    const bool consumed = dispatch_opt_attr(ukey, iss, spec) ||
                          (extra && extra(ukey, iss, spec));
    if (!consumed)
      throw std::runtime_error("OPTIMIZER block: unknown key '" + key + "'");
    if (iss.fail())
      throw std::runtime_error("OPTIMIZER block: bad value for '" + key + "'");
    std::size_t before = tok.size() - pos;  // key + everything after it
    std::size_t remaining = 0;
    std::string t;
    while (iss >> t) ++remaining;
    pos += (before - remaining);
  }
  throw std::runtime_error("OPTIMIZER block not closed with " + endTok);
}

}  // namespace

void read_optimizer_block(const std::vector<std::string> &tok,
                          std::size_t &pos, OptimizerSpec &spec,
                          const ExtraAttrFn &extra) {
  if (pos >= tok.size() || up(tok[pos]) != "OPTIMIZER")
    throw std::runtime_error("read_optimizer_block: expected OPTIMIZER");
  ++pos;  // consume OPTIMIZER
  read_block_until(tok, pos, "ENDOPTIMIZER", spec, extra, /*allow_inner=*/true);
}

}  // namespace hpo
}  // namespace tadah
