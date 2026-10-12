#pragma once
/* seed_stream.h -- the random streams the optimisers draw from their SEED.
 *
 * Every consumer of an optimiser's SEED draws a std::mt19937_64 of its own,
 * seeded from (SEED, salt, index) through a std::seed_seq: the ANNEAL
 * family's chains, T0 probes, PT swap coins, PT warm-up and PA resampler
 * (anneal_core.h, which names these salts SALT_*), TADAH:RANDOM's draws
 * (kRandomSearch) and ASA's generator (kAsa). Streams of different salts,
 * or of different indices, are seeded from different seed sequences, and
 * none is seeded as std::mt19937_64(seed) is, the stream the starting-point
 * methods RANDOM and LHS draw (init_strategies.cpp: an INIT block's or a
 * RESTART round's METHOD chain). So an optimiser does not draw the numbers
 * of a METHOD chain or of another backend: TADAH:RANDOM and ASA seeded
 * std::mt19937_64 with the integer and drew it through uniform(0, 1), as
 * the RANDOM method does, so equal seeds of two keys gave the same numbers
 * -- with an INIT block's SEED equal to TADAH:RANDOM's, its first draw was
 * its start point again; with the RESTART block's SEED equal to the
 * optimiser's, every round's first draw was the round's fresh point. NLopt
 * draws from its own generator (nlopt::srand) and SOBOL draws no random
 * numbers (its seed selects a block of the sequence).
 *
 * Two consumers of the same kind given the same seed -- two METHOD chains,
 * two phases running one backend -- draw the same numbers: a seed names a
 * stream. HPO's own derived seeds keep the runs of one SEED apart
 * (MULTISTART start i: SEED + i; RESTART round r: SEED + r*S, the RESTART
 * block's points RESTART SEED + r; OptimizerSpec::derive_restart_seeds).
 */
#include <cstddef>
#include <cstdint>
#include <random>

namespace tadah::hpo {

namespace seed_streams {
constexpr std::uint64_t kAnnealChain     = 0xC4A1C;   /* chain i proposals+coins */
constexpr std::uint64_t kAnnealProbes    = 0x70B35;   /* T0 calibration probes   */
constexpr std::uint64_t kAnnealPtSwap    = 0x5757;    /* PT swap coins (epoch)   */
constexpr std::uint64_t kAnnealPaResample = 0x9E5A;   /* PA resampler (tstep)    */
constexpr std::uint64_t kAnnealWarmup    = 0x57A21;   /* PT step warm-up + pilot */
constexpr std::uint64_t kRandomSearch    = 0x4A4D5E;  /* TADAH:RANDOM's draws    */
constexpr std::uint64_t kAsa             = 0xA5A5A;   /* ASA's generator         */

/* Whether the salts are pairwise distinct. */
constexpr bool all_distinct()
{
    constexpr std::uint64_t s[] = {kAnnealChain, kAnnealProbes, kAnnealPtSwap,
                                   kAnnealPaResample, kAnnealWarmup,
                                   kRandomSearch, kAsa};
    constexpr std::size_t n = sizeof(s) / sizeof(s[0]);
    for (std::size_t i = 0; i < n; ++i)
        for (std::size_t j = i + 1; j < n; ++j)
            if (s[i] == s[j]) return false;
    return true;
}
static_assert(all_distinct(), "every seed stream has a salt of its own");
}  // namespace seed_streams

/* The stream of the consumer `salt` (seed_streams) for `seed`, its `idx`-th
 * when the consumer has several (an annealer's chain per slot, ...). */
inline std::mt19937_64 seed_stream(std::uint64_t seed, std::uint64_t salt,
                                   std::uint64_t idx = 0)
{
    std::seed_seq seq{
        static_cast<std::uint32_t>(seed), static_cast<std::uint32_t>(seed >> 32),
        static_cast<std::uint32_t>(salt), static_cast<std::uint32_t>(salt >> 32),
        static_cast<std::uint32_t>(idx),  static_cast<std::uint32_t>(idx >> 32)};
    return std::mt19937_64(seq);
}

}  // namespace tadah::hpo
