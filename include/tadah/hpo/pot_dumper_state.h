#ifndef TADAH_HPO_POT_DUMPER_STATE_H
#define TADAH_HPO_POT_DUMPER_STATE_H

/*
 *  PotDumperState -- checkpoint participant for DUMP / BDUMP counters.
 *
 *  Background
 *  ----------
 *  DUMP <N> <dir>   archives a snapshot of the potential to <dir>/pot_*.tadah
 *  every N evaluations. BDUMP <N> <dir> does the same for "every Nth new
 *  best". The actual write logic lives inline in HPO_Host::submit_result
 *  and HPO_Host::flush_remaining (hpo_host_impl.hpp:1241-1266 / 1348-1372)
 *  and is keyed off the global iteration counter `e.iter` and best counter
 *  `e.best_iter` respectively. Both counters are global atomics on
 *  HPO_Host (g_iter_ / g_best_iter_), so once those values are restored
 *  on resume the DUMP/BDUMP modulo schedule continues seamlessly.
 *
 *  Role of this participant
 *  ------------------------
 *  Surfaces the DUMP/BDUMP configuration in the checkpoint file so that:
 *    1. A user inspecting `checkpoint.tadah` can see "yes, this run was
 *       dumping every 50 evals to ./pots/ when it stopped".
 *    2. A future inline-refactor that does want per-dumper state (e.g.
 *       last-flushed counter, ring-buffer rotation index) has a place
 *       to put it.
 *
 *  This file does not change the user-visible DUMP/BDUMP behaviour --
 *  the existing inline code path is left intact.
 */

#include <tadah/core/checkpoint.h>

#include <cstddef>
#include <limits>
#include <string>

namespace tadah {
namespace hpo {

class PotDumperState : public core::ICheckpointable {
public:
    PotDumperState(std::size_t dump_every,
                   std::string dump_dir,
                   std::size_t bdump_every,
                   std::string bdump_dir)
        : dump_every_(dump_every),
          dump_dir_(std::move(dump_dir)),
          bdump_every_(bdump_every),
          bdump_dir_(std::move(bdump_dir)) {}

    std::string tag() const override { return "POT_DUMPER"; }

    void save(core::Checkpoint& cp) const override {
        cp.section(tag());
        cp.put_size("dump_every",  dump_every_);
        cp.put     ("dump_dir",    dump_dir_);
        cp.put_size("bdump_every", bdump_every_);
        cp.put     ("bdump_dir",   bdump_dir_);
    }

    void load(const core::Checkpoint& cp) override {
        if (auto v = cp.get_size(tag(), "dump_every"))  dump_every_  = *v;
        if (auto s = cp.get(tag(), "dump_dir"))         dump_dir_    = *s;
        if (auto v = cp.get_size(tag(), "bdump_every")) bdump_every_ = *v;
        if (auto s = cp.get(tag(), "bdump_dir"))        bdump_dir_   = *s;
    }

    std::size_t dump_every()  const { return dump_every_; }
    const std::string& dump_dir()  const { return dump_dir_; }
    std::size_t bdump_every() const { return bdump_every_; }
    const std::string& bdump_dir() const { return bdump_dir_; }

private:
    std::size_t  dump_every_;
    std::string  dump_dir_;
    std::size_t  bdump_every_;
    std::string  bdump_dir_;
};

}   // namespace hpo
}   // namespace tadah

#endif
