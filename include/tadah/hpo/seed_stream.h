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
 * A PRESTAGE refinement draws streams of its own (StreamPhase,
 * phase_stream_seed below): its optimiser seeds every stream from a key
 * derived from its SEED, so it does not draw the numbers of the
 * exploration, whatever the two SEEDs -- equal ones, or the seeds RESTART
 * derives for a round (the refinement SEED + r*S, the exploration
 * EXPLORE SEED + r), which meet for SEEDs that differ (SEED 7 and
 * EXPLORE SEED 8: round 1's refinement ran with round 0's EXPLORE SEED).
 * Before, a refinement with the exploration's optimiser drew its numbers
 * for an equal seed: a TADAH:RANDOM refinement evaluated the exploration's
 * draws again.
 *
 * A RESTART round's starting-point methods draw streams of their own too
 * (kRestartPoints, InitStream::Restart, init_strategy.h): its RANDOM and
 * LHS methods seed seed_stream(SEED + r, kRestartPoints), not
 * std::mt19937_64(SEED + r), so a round does not draw the points of an
 * INIT block whatever the two SEEDs: an INIT block's SEED s and a round
 * whose RESTART SEED + r is s drew the same points (INIT SEED 5, RESTART
 * SEED 4: restart 1 started where round 0 had). SOBOL draws no random
 * numbers: its seed selects a block of the sequence in both blocks.
 *
 * Two consumers of the same kind in one phase given the same seed -- two
 * INIT chains of one phase, two optimisers of one kind -- draw the same
 * numbers: a seed names a stream. HPO's own derived seeds keep the runs of
 * one SEED apart (MULTISTART start i: SEED + i; RESTART round r: SEED +
 * r*S, the RESTART block's points RESTART SEED + r;
 * OptimizerSpec::derive_restart_seeds).
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
constexpr std::uint64_t kRefinementPhase = 0x2EF1E;   /* a refinement's key      */
constexpr std::uint64_t kRestartPoints   = 0x2E57A;   /* RESTART rounds' RANDOM/LHS */

/* Whether the salts are pairwise distinct. */
constexpr bool all_distinct()
{
    constexpr std::uint64_t s[] = {kAnnealChain, kAnnealProbes, kAnnealPtSwap,
                                   kAnnealPaResample, kAnnealWarmup,
                                   kRandomSearch, kAsa, kRefinementPhase,
                                   kRestartPoints};
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

/* The phase an optimiser runs in, as far as its random numbers go
 * (OptimizerSpec::stream_phase): a search -- a PLAIN or MULTISTART run, a
 * PRESTAGE exploration -- or a PRESTAGE refinement
 * (prestage_refinement_spec, phase_specs.h). */
enum class StreamPhase { Search, Refinement };

/* The seed every random stream of an optimiser run is drawn from, for its
 * SEED `seed` (or the INIT block's, NLopt's fallback) in `phase`: `seed`
 * itself in a search, so a search draws what it always drew, and in a
 * refinement the first number of seed_stream(seed, kRefinementPhase), a
 * key no search's SEED names but by a 64-bit coincidence. The journal
 * names the SEED, not the key. TADAH:RANDOM, ASA, the ANNEAL family and
 * NLopt (nlopt::srand) read it. */
inline std::uint64_t phase_stream_seed(std::uint64_t seed, StreamPhase phase)
{
    if (phase == StreamPhase::Search) return seed;
    return seed_stream(seed, seed_streams::kRefinementPhase)();
}

}  // namespace tadah::hpo
