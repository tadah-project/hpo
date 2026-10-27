#ifndef TADAH_HPO_CUTOFF_CHANNELS_H
#define TADAH_HPO_CUTOFF_CHANNELS_H

// The descriptors whose cutoff tadah hpo takes care of.
//
// A knot channel (D2_Knot5; a DM_REAM density of knots) has the cutoff of
// its largest knot: tadah hpo derives RCUT2B/RCUTMB from it at startup and
// on every evaluation (rcut_derivation.h). A bump channel (D2_Blip,
// D2_Blip5; a DM_REAM density of bumps) under a hard cutoff has every bump
// support end at or before the cutoff, which the startup checks and the
// per-candidate gate keep (bump_support_gate.h); under a smooth one a
// centre beyond the cutoff is warned about. A channel is the whole TYPE2B or
// TYPEMB line when it has one descriptor, or one descriptor of a joined line
// (D2_mJoin, DM_mJoin; mlip::joined_descriptors), with its own context: its
// share of the list keys and its own value of the RCUT key. Its values are
// entries of the whole keys, as an OPTIM line numbers them: its value i of
// a key K is entry first(K) + i of K (0-based).

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include <tadah/core/context.h>
#include <tadah/mlip/joined_descriptors.h>
#include <tadah/models/eam/pot_curve_edit.h>

namespace tadah {
namespace hpo {
namespace detail {

struct CutoffChannel {
  std::string sfx;            ///< "2B" or "MB"
  bool joined = false;        ///< a descriptor of a joined line
  std::size_t position = 0;   ///< its place on the joined line, 0-based
  std::string label;          ///< its descriptor (D2_Knot5, D2_Blip, ...)
  /** Its own context: the whole configuration for a line of one
   *  descriptor; for a descriptor of a joined line, its share of the keys
   *  (joined_descriptors), without the TYPE line of the other body order,
   *  so that a rule of one channel reads this channel only. */
  core::Context context;
  /** For each key the joined line splits, the number of the key's values
   *  before its own (mlip::JoinedDescriptor::first_entries); empty for a
   *  line of one descriptor. */
  std::map<std::string, std::size_t> first_entries;

  /** The 0-based entry of the whole key @p key that its value 0 is. */
  std::size_t first(const std::string &key) const {
    const auto it = first_entries.find(key);
    return it == first_entries.end() ? 0 : it->second;
  }
  /** The 0-based entry of its RCUT value in the whole RCUT key. */
  std::size_t rcut_index() const { return first("RCUT" + sfx); }

  /** "TYPE2B descriptor 2 (D2_Blip)", the descriptor as a message names
   *  it; "TYPE2B" for a line of one descriptor. */
  std::string where() const {
    return joined ? "TYPE" + sfx + " descriptor " +
                        std::to_string(position + 1) + " (" + label + ")"
                  : "TYPE" + sfx;
  }
  /** "TYPE2B descriptor 2 (D2_Blip): " for a descriptor of a joined line,
   *  "" for a line of one descriptor. */
  std::string prefix() const {
    return joined ? where() + ": " : std::string();
  }
  /** "RCUT2B (2)" for a descriptor of a joined line, "RCUT2B" otherwise. */
  std::string rcut_name() const {
    return joined
               ? "RCUT" + sfx + " (" + std::to_string(rcut_index() + 1) + ")"
               : "RCUT" + sfx;
  }
};

namespace cutoff_channels_detail {
/* The channels of `ctx`'s TYPE`sfx` line that `is_one` accepts on a
 * context of one descriptor (derived_knot_rcut, bump_channel_halfwidth). */
template <class Pred>
std::vector<CutoffChannel> channels(const core::Context &ctx,
                                    const std::string &sfx, Pred is_one) {
  std::vector<CutoffChannel> out;
  const std::string type_key = "TYPE" + sfx;
  if (!mlip::is_joined_line(ctx, type_key)) {
    if (!is_one(ctx)) return out;
    CutoffChannel ch;
    ch.sfx = sfx;
    ch.label = ctx.get<std::string>(type_key, 0);
    ch.context = ctx;
    out.push_back(std::move(ch));
    return out;
  }
  const std::string other = sfx == "2B" ? "TYPEMB" : "TYPE2B";
  for (auto &jd : mlip::joined_descriptors(ctx, type_key)) {
    core::Context own = jd.context;
    if (own.contains(other)) own.remove(other);
    if (!is_one(own)) continue;
    CutoffChannel ch;
    ch.sfx = sfx;
    ch.joined = true;
    ch.position = jd.position;
    ch.label = jd.label;
    ch.first_entries = std::move(jd.first_entries);
    ch.context = std::move(own);
    out.push_back(std::move(ch));
  }
  return out;
}
}  // namespace cutoff_channels_detail

/** \brief The knot channels of the TYPE`sfx` line of @p ctx: the line,
 *  when models::potedit::derived_knot_rcut() derives its cutoff, or each
 *  descriptor of a joined line for which it derives one on its own
 *  context. */
inline std::vector<CutoffChannel> knot_channels(const core::Context &ctx,
                                                const std::string &sfx) {
  return cutoff_channels_detail::channels(
      ctx, sfx, [&sfx](const core::Context &c) {
        return models::potedit::derived_knot_rcut(c, sfx).has_value();
      });
}

/** \brief Whether the TYPE`sfx` line of @p ctx has a knot channel
 *  (knot_channels()). */
inline bool has_knot_channel(const core::Context &ctx,
                             const std::string &sfx) {
  return !knot_channels(ctx, sfx).empty();
}

/** \brief Value @p i of the RCUT key of @p ch, set to @p value in the whole
 *  context @p ctx: the whole key for a line of one descriptor (one value,
 *  as normalize_knot_rcut writes it), its own entry for a descriptor of a
 *  joined line. Returns whether the key changed; Context::add refreshes
 *  RCUT2BMAX / RCUTMBMAX / RCUTMAX. The comparison is exact (the text of a
 *  Context value is lossless). */
inline bool set_channel_rcut(core::Context &ctx, const CutoffChannel &ch,
                             double value) {
  const std::string key = "RCUT" + ch.sfx;
  if (!ch.joined) {
    if (ctx.contains(key) && ctx.size(key) > 0 &&
        ctx.get<double>(key) == value)
      return false;
    ctx.remove(key);
    ctx.add(key, value);
    return true;
  }
  const std::size_t at = ch.rcut_index();
  std::vector<double> values(ctx.contains(key) ? ctx.size(key) : 0);
  for (std::size_t i = 0; i < values.size(); ++i)
    values[i] = ctx.get<double>(key, i);
  if (at >= values.size())
    throw std::logic_error(ch.prefix() + key + " has " +
                           std::to_string(values.size()) +
                           " value(s), and the descriptor's is value " +
                           std::to_string(at + 1) +
                           " (D_mJoin::parse_context refuses such a line)");
  if (values[at] == value) return false;
  values[at] = value;
  ctx.remove(key);
  ctx.add(key, values);
  return true;
}

/** \brief The bump channels of the TYPE`sfx` line of @p ctx: the line,
 *  when models::potedit::bump_channel_halfwidth() gives its support
 *  half-width, or each descriptor of a joined line for which it gives one
 *  on its own context. */
inline std::vector<CutoffChannel> bump_channels(const core::Context &ctx,
                                                const std::string &sfx) {
  return cutoff_channels_detail::channels(
      ctx, sfx, [&sfx](const core::Context &c) {
        return models::potedit::bump_channel_halfwidth(c, sfx).has_value();
      });
}

}  // namespace detail
}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_CUTOFF_CHANNELS_H
