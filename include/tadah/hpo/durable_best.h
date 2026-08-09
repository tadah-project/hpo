#pragma once

/*
 * Crash-safe best-potential publication.
 *
 * A candidate is compared while holding an advisory process lock, written to
 * a unique file in the destination directory, fsync'd, and atomically renamed
 * over the destination.  The directory is fsync'd before the lock is released
 * so a successful return means the rename is durable across a power loss.
 */

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>

namespace tadah::hpo::durable_best {

struct Key {
    double      loss = 0.0;
    int         rank = 0;
    std::size_t slot = 0;
    std::size_t eval = 0;
};

class Error : public std::runtime_error {
public:
    using std::runtime_error::runtime_error;
};

/** Strict total ordering used by every process. Lower loss wins; exact-loss
 * ties choose the lexicographically smaller (rank, slot, eval) tuple. */
bool precedes(const Key& candidate, const Key& incumbent) noexcept;

/** Parse the machine-readable metadata emitted by PotHeaderWriter.
 * Returns nullopt for legacy/unrecognised payloads. */
std::optional<Key> parse_key(const std::string& payload);

/** Read and parse the current file while holding the same process/thread
 * locks used by publication. Returns nullopt when the file is absent or uses
 * the legacy format. */
std::optional<Key> read_key_file(const std::string& path);

/** Remove a prior run's artifact under the publication lock and fsync the
 * containing directory. Call only for a confirmed fresh run; checkpoint
 * resumes deliberately retain their incumbent. */
void reset_for_fresh_run(const std::string& path);

/** Publish payload at path only when key beats the current file's key.
 * Legacy files without machine metadata are replaced. Returns true when a
 * rename was committed, false when an equal-or-better file already existed.
 * Throws Error on locking, I/O, fsync, or rename failure. */
bool publish_if_better(const std::string& path,
                       const Key& key,
                       const std::string& payload);

} // namespace tadah::hpo::durable_best
