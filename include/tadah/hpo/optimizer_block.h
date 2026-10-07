#ifndef TADAH_HPO_OPTIMIZER_BLOCK_H
#define TADAH_HPO_OPTIMIZER_BLOCK_H

/** \file optimizer_block.h
 *  \brief Shared leaf utilities for parsing an `OPTIMIZER` block into an
 *  OptimizerSpec. These are the genuinely-common pieces between the
 *  `tadah hpo --hpotarget` parser (HPOTargets::read) and the `tadah refit`
 *  stage parser (RefitParser): the optimiser-attribute vocabulary
 *  (LIB/ALGO/MAXEVAL/stop-criteria/... -> OptimizerSpec) and a balanced
 *  OPTIMIZER...ENDOPTIMIZER reader (incl. a nested INNER...ENDINNER).
 *
 *  The HPO parser keeps its own (line-oriented) state machine and only calls
 *  dispatch_opt_attr(); the refit parser additionally uses
 *  read_optimizer_block() because its block body reaches it as a flat token
 *  stream (CORE's config parser joins the block's lines before splitting on
 *  whitespace, so line boundaries are lost — fine, the grammar is positional).
 */

#include <tadah/hpo/optimizer_spec.h>

#include <cstddef>
#include <functional>
#include <sstream>
#include <string>
#include <vector>

namespace tadah {
namespace hpo {

/** Parse a boolean token (true/false/1/0/yes/no/on/off, any case). Throws on
 *  unknown spellings; `ctx` is used in the error message. */
bool parse_bool(const std::string &v, const std::string &ctx);

/** The optimiser-attribute vocabulary consumed by dispatch_opt_attr(), in
 *  the order documented. Single source of truth for "unknown key" messages
 *  (previously hand-listed in the INNER block error) and the docs table. */
const std::vector<std::string> &opt_attr_keys();
std::string opt_attr_keys_joined(const char *sep = "/");

/** Value grammar for `PARAM <key> ...`: given the value tokens already read
 *  (`sofar`) and the next token on the stream, should that token be taken as a
 *  further value of this key? Almost every key answers false after one value;
 *  `INIT_MODE` accepts `UNIFORM` | `JITTER [AUTO <p> | <frac>]`.
 *
 *  This is a grammar rather than a maximum count because the two callers frame
 *  their input differently: HPOTargets::read() hands dispatch_opt_attr() one
 *  LINE, while read_optimizer_block() hands it a flat token stream spanning
 *  the whole block. "Read up to N values" would silently consume the next key
 *  in the second framing whenever fewer than N are actually present; deciding
 *  per token is correct in both. `key` is expected uppercase. */
bool param_takes_another_value(const std::string &key,
                               const std::vector<std::string> &sofar,
                               const std::string &next);

/** Parse one optimiser-attribute key/value into an OptimizerSpec. Returns true
 *  if `key` was consumed (its value(s) read from `iss`), false otherwise so the
 *  caller can try other dispatches. Valid wherever an OptimizerSpec is the
 *  active receiver (HPO STRATEGY/EXPLORE/REFINE/INNER; refit OPTIMIZER block).
 *  `key` is expected uppercase. */
bool dispatch_opt_attr(const std::string &key, std::istringstream &iss,
                       OptimizerSpec &a);

/** Command-specific attribute hook for read_optimizer_block(): tried before
 *  dispatch_opt_attr(), so it can add keys (refit's GTOL) and read a standard
 *  attribute by its own rule (refit's tolerances are >= 0). Receives the
 *  uppercased key and a stream positioned right after it (so it can pull its
 *  value(s)); returns true if it consumed the key, false to leave the key to
 *  dispatch_opt_attr(). */
using ExtraAttrFn = std::function<bool(const std::string &key,
                                       std::istringstream &iss,
                                       OptimizerSpec &a)>;

/** Key gate for read_optimizer_block(): called with every key of the block
 *  (uppercased, `INNER` included) before it is read; `in_inner` is true for
 *  the keys of the INNER block. It throws to reject a key the caller does
 *  not use (the `tadah refit` stages); read_optimizer_block() prefixes the
 *  message with the block, as it does every error of the block. */
using KeyCheckFn = std::function<void(const std::string &key, bool in_inner)>;

/** Consume a balanced `OPTIMIZER ... ENDOPTIMIZER` block from the flat token
 *  stream `tok`, starting at `tok[pos] == "OPTIMIZER"`. Fills `spec`; a nested
 *  `INNER ... ENDINNER` fills `spec.inner`. Every key is first shown to
 *  `check`, then offered to `extra` (command-specific keys, e.g. refit's
 *  GTOL, and standard ones the command reads by its own rule); the keys it
 *  leaves go through dispatch_opt_attr(); an unconsumed key throws.
 *  `where` names the block in every error it raises, the INNER block's as
 *  "<where> INNER" ("REFINE: MAXEVAL must be ...", "EXPLORE INNER: unknown
 *  key 'X' in the INNER block"); empty, the messages are unprefixed.
 *  On return `pos` points just past `ENDOPTIMIZER`. Keyword matching is
 *  case-insensitive. */
void read_optimizer_block(const std::vector<std::string> &tok, std::size_t &pos,
                          OptimizerSpec &spec,
                          const ExtraAttrFn &extra = nullptr,
                          const KeyCheckFn &check = nullptr,
                          const std::string &where = "");

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_OPTIMIZER_BLOCK_H
