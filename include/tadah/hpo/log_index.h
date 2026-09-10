#pragma once
/*
 * log_index -- the ONE radix boundary between Tadah's internal indices and
 * everything a person reads.
 *
 * THE RULE: the radix follows AUTHORSHIP.
 *
 *   An index the USER TYPES in a config is 1-based everywhere, journal and
 *   files alike (OPTIM `KEY[n]`, ASA `PARAM Name[i]`, hpo_search_space.tsv's
 *   `dim`). That half shipped in 08b814d / 198234c.
 *
 *   An index the PROGRAM ASSIGNS ITSELF -- a ladder slot, an exchange pair, a
 *   walker lineage, a pc-worker pool slot -- is 0-based in memory and 1-based
 *   only where it is rendered for a person. It has to stay 0-based in memory
 *   because it round-trips as an IDENTITY through warmstart::current_chain()
 *   -> EvalResult::chain_id -> durable_best::Key -> `# HPO_SOURCE_SLOT` ->
 *   the resume checkpoint. Shifting it at the source would silently
 *   repartition the per-chain warm-start cache (the failure class that
 *   HPO d33e83e fixed) and make durable_best::publish_if_better throw.
 *
 * So: `ui()` on the way OUT, `from_ui()` on the way BACK IN, and nothing that
 * builds a cache key, a comparison key or a checkpoint field ever calls either.
 *
 * Everything that renders a slot/pair/walker index routes through here, so
 * there is exactly one `+1` in the tree rather than a dozen independent ones.
 * In particular grid() owns the rows-of-8 "[lo-hi]" layout that six call sites
 * used to hand-roll, each with its own index-derived column width.
 *
 * NOTE grid()'s `cell` callback is handed the ZERO-BASED index on purpose.
 * Callers index arrays and std::map with it (HPOLog::best_slot_* are keyed by
 * the raw slot), so making them un-convert would reintroduce exactly the
 * off-by-one this module exists to prevent.
 */

#include <cstddef>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <string>

namespace tadah {
namespace hpo {
namespace logidx {

/* The one +1 in the tree. Introduced as the identity so that the refactor
 * routing every emitter through this module could be proven output-neutral
 * first; the radix flip is then exactly this line. */
inline std::size_t ui(std::size_t zero_based) { return zero_based + 1; }

/* The one -1: parse a user-facing index back to the internal one. */
inline std::size_t from_ui(std::size_t user_facing, const char* what)
{
    const std::size_t lowest = ui(0);
    if (user_facing < lowest)
        throw std::runtime_error(
            std::string("malformed ") + what + ": index " +
            std::to_string(user_facing) + " is below the lowest valid value " +
            std::to_string(lowest));
    return user_facing - lowest;
}

inline std::string uis(std::size_t zero_based)
{
    return std::to_string(ui(zero_based));
}

/* Column width for labels of a `count`-sized family: wide enough for the
 * largest label actually printed. Derived from the rendered label rather
 * than from `count - 1`, so it stays correct when the radix flips (at
 * count = 10 the labels run 1..10 and need two columns, not one). */
inline int width(std::size_t count)
{
    return static_cast<int>(uis(count ? count - 1 : 0).size());
}

/* "[ 1- 8]" from ZERO-BASED inclusive bounds. */
inline std::string range(std::size_t lo0, std::size_t hi0, int w)
{
    std::ostringstream o;
    o << "[" << std::setw(w) << uis(lo0) << "-" << std::setw(w) << uis(hi0)
      << "]";
    return o.str();
}

/* The rows-of-`stride` grid every per-slot / per-pair table uses.
 * `cell(os, i)` receives the ZERO-BASED index (see the note above). */
template <class Cell>
std::string grid(const std::string& header, std::size_t count, Cell&& cell,
                 std::size_t stride = 8)
{
    std::ostringstream o;
    o << header;
    const int w = width(count);
    for (std::size_t lo = 0; lo < count; lo += stride) {
        const std::size_t hi = std::min(lo + stride, count) - 1;
        o << "\n" << range(lo, hi, w);
        for (std::size_t i = lo; i <= hi; ++i) cell(o, i);
    }
    return o.str();
}

/* Fallback labels for a parameter dimension whose name is unavailable.
 * Dimensions are user-authored, so these are 1-based independently of ui():
 * the surrounding run already prints KEY[n] and hpo_search_space.tsv's `dim`
 * 1-based, and the scattered "dim[i]" fallbacks contradict it. Correcting
 * those call sites CHANGES output, so it happens in the radix-flip step
 * rather than in the output-neutral refactor that introduced this module. */
inline std::string dim_label(std::size_t i)
{
    return "dim[" + std::to_string(i + 1) + "]";
}
inline std::string param_label(std::size_t i)
{
    return "param[" + std::to_string(i + 1) + "]";
}

}  // namespace logidx
}  // namespace hpo
}  // namespace tadah
