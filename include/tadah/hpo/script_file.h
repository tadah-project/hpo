#pragma once
#ifndef TADAH_HPO_SCRIPT_FILE_H
#define TADAH_HPO_SCRIPT_FILE_H

/* RAII on-disk rendering of a preprocessed PC_LAMMPS script.
 *
 * PC_LAMMPS scripts are executed through LAMMPS's own Input::file(path)
 * reader -- the only executor in which jump/label/next/if-then-"jump"
 * control flow has native LAMMPS semantics. The old Input::one()-per-line
 * loop could not loop backwards: jump's rewind(infile) is a no-op on stdin
 * in library mode, and the sticky label_active state then silently
 * swallowed the rest of the script (and poisoned the cached instance's
 * NEXT eval -- the run26 SIGSEGV storm).
 *
 * The file is rendered ONCE from LammpsSim::prep_script_lines() output
 * (logical newline-terminated commands; includes already expanded,
 * comments stripped, & continuations joined, `timer timeout` lines
 * injected) and reused for every evaluation of the owning cache entry, so
 * there is no per-eval I/O. It lives in the node-local temp directory
 * (workers may run with a network-mounted cwd).
 *
 * Cleanup: the dtor best-effort unlinks. THREAD mode: PcSlotState is
 * thread_local and dies at pool-thread join. PROCESS mode: slot state is
 * stack-local in pc_worker_main and dies on SHUTDOWN/EOF return. A
 * SIGKILLed worker (hard deadline) orphans its files in node-local tmp --
 * accepted. */

#include <atomic>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>   /* getpid */

namespace tadah {
namespace hpo {

/* Sentinel command injected by prep_script_lines() before every mid-script
 * `clear` that follows a timer-armed section:
 *   if "$(timeremain) == 0.0" then "TADAH_PC_TIMEOUT_ABORT"
 * After a fired `timer timeout` the guard executes the sentinel as an
 * unknown command, aborting Input::file() with an exception whose message
 * carries this token; the runner classifies that as TIMEOUT_ABORT. This
 * preserves the per-section soft-timeout semantics (and cost profile) of
 * the old per-line is_timeout() check, which Input::file() cannot provide
 * between commands. */
inline constexpr const char* PC_TIMEOUT_ABORT_SENTINEL = "TADAH_PC_TIMEOUT_ABORT";

class ScriptFile {
public:
  ScriptFile() = default;   /* empty; valid() == false */

  explicit ScriptFile(const std::vector<std::string>& script_lines) {
    namespace fs = std::filesystem;
    static std::atomic<unsigned long long> counter{0};
    const auto n = counter.fetch_add(1, std::memory_order_relaxed);
    const fs::path p = fs::temp_directory_path() /
        ("tadah_pc_" + std::to_string(::getpid()) + "_" +
         std::to_string(n) + ".in");
    {
      std::ofstream out(p, std::ios::binary | std::ios::trunc);
      for (const auto& line : script_lines) {
        out << line;
        if (line.empty() || line.back() != '\n') out << '\n';
        if (!has_extern_ &&
            line.find("pair_coeff") != std::string::npos &&
            line.find("EXTERN") != std::string::npos)
          has_extern_ = true;
      }
      out.flush();
      if (!out) {
        std::error_code ec;
        fs::remove(p, ec);
        throw std::runtime_error("ScriptFile: cannot write " + p.string());
      }
    }
    path_ = p.string();
  }

  ScriptFile(ScriptFile&& o) noexcept
      : path_(std::move(o.path_)), has_extern_(o.has_extern_) {
    o.path_.clear();
  }
  ScriptFile& operator=(ScriptFile&& o) noexcept {
    if (this != &o) {
      remove_();
      path_ = std::move(o.path_);
      has_extern_ = o.has_extern_;
      o.path_.clear();
    }
    return *this;
  }
  ScriptFile(const ScriptFile&) = delete;
  ScriptFile& operator=(const ScriptFile&) = delete;

  ~ScriptFile() { remove_(); }

  const std::string& path() const { return path_; }
  bool has_extern() const { return has_extern_; }
  bool valid() const { return !path_.empty(); }

private:
  void remove_() {
    if (path_.empty()) return;
    std::error_code ec;
    std::filesystem::remove(path_, ec);   /* best-effort */
    path_.clear();
  }

  std::string path_;
  bool has_extern_ = false;
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_SCRIPT_FILE_H
