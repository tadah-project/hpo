#include <tadah/hpo/optimizer_block.h>

#include "hpotarget_reads.h"
#include <tadah/hpo/strict_parse.h>
#include <tadah/core/utils.h>
#include <tadah/core/number_format.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <limits>
#include <map>
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

namespace R = hpotarget_registry;

/* The keys of the optimiser group of the registry, in its order. */
const std::vector<std::string> &opt_attr_keys() {
  static const std::vector<std::string> keys = [] {
    std::vector<std::string> out;
    for (const R::Key &k : R::kKeys)
      if (k.group == "OPTIMIZER") out.emplace_back(k.name);
    return out;
  }();
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

/* A key of the optimiser group (its name exact) is read by its handler
 * (hpotarget_handlers.cpp), which reads exactly the values the key takes:
 * the number and integer grammar of its arguments (the registry's), the
 * STEP and PARAM grammars, the words of GRAD_STRATEGY. What follows them
 * is the caller's: HPOTargets::read() rejects a leftover token on the
 * line, and read_optimizer_block() reads the next key there. */
bool dispatch_opt_attr(const std::string &key, std::istringstream &iss,
                       OptimizerSpec &a) {
  for (const R::Key &k : R::kKeys) {
    if (k.group != "OPTIMIZER" || k.name != key) continue;
    R::KeyRead read{k, key, iss};
    read.spec = &a;
    R::call(k, read);
    return true;
  }
  return false;
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
  /* Each key is set once for this block's optimiser, as in a target file
   * (HPOTargets::read): the attribute a key sets -> the spelling it was
   * first set with. MAXITER is MAXEVAL's alias (the refit's Ceres stages),
   * PARAM <name> is one attribute per name, and a STEP <KEY> override
   * layers over the block-wide STEP (the refit takes no STEP: its stage
   * check refuses it). An OPTIMIZER block takes one INNER block. */
  std::map<std::string, std::string> set_as;
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
      // One INNER block: a second one would replace the first.
      if (spec.inner)
        throw in_block(where, "the INNER block is given twice; an OPTIMIZER "
                              "block takes one INNER block");
      ++pos;  // consume INNER
      spec.inner = std::make_unique<OptimizerSpec>();
      read_block_until(tok, pos, "ENDINNER", *spec.inner, extra, check,
                       /*allow_inner=*/false,
                       where.empty() ? where : where + " INNER");
      continue;
    }
    // Command-specific / standard attribute. The command's hook is offered
    // the key first, so it can read a standard attribute by its own rule
    // (the refit's tolerances are >= 0); dispatch_opt_attr() reads the rest.
    // Each reads a fixed (small) number of value tokens from a stream built
    // over the remaining tokens; we then recover how many it consumed by
    // counting the leftover.
    std::string rest;
    for (std::size_t j = pos; j < tok.size(); ++j) rest += tok[j] + " ";
    std::istringstream iss(rest);
    std::string key;
    iss >> key;  // the attribute key (original case)
    const std::string ukey = up(key);
    bool consumed = false;
    try {
      consumed = (extra && extra(ukey, iss, spec)) ||
                 dispatch_opt_attr(ukey, iss, spec);
    } catch (const std::exception &e) {
      throw in_block(where, e.what());
    }
    if (!consumed)
      throw in_block(where, "unknown key '" + key + "' in the " +
                                std::string(block) + " block");
    if (iss.fail())
      throw in_block(where, "invalid value for '" + key + "' in the " +
                                std::string(block) + " block: " +
                                (pos + 1 < tok.size()
                                     ? "the value at or after '" +
                                           tok[pos + 1] + "' is not one it takes"
                                     : std::string("no value given")));
    {
      const std::string first = pos + 1 < tok.size() ? tok[pos + 1] : "";
      std::string attr = ukey == "MAXITER" ? std::string("MAXEVAL") : ukey;
      if (ukey == "PARAM") attr = "PARAM " + first;
      const bool step_override = ukey == "STEP" && !is_number_shaped(first);
      if (!step_override) {
        const auto [it, fresh] = set_as.emplace(attr, ukey);
        if (!fresh) {
          // The key as written; MAXEVAL and its alias MAXITER in the order
          // written.
          const bool alias = it->second != ukey;
          const std::string spellings =
              !alias ? ""
              : it->second == "MAXEVAL"
                  ? " (as MAXEVAL and as MAXITER, its alias)"
                  : " (as MAXITER, its alias, and as MAXEVAL)";
          throw in_block(where, (alias ? attr : ukey) + " is set twice in "
                                    "the " + std::string(block) + " block" +
                                    spellings + "; each key is given once");
        }
      }
    }
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
