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
 * there is no per-eval I/O. It is created in the node-local temp directory
 * (workers may run with a network-mounted cwd).
 *
 * No file is left behind: the rendering is created by mkostemps() under
 * a name that nobody else holds (tadah_pc_<pid>_<random>.in, created with
 * O_EXCL and mode 0600; the C library tries other random names while one
 * is taken, so stale files of an earlier process whose pid was reused, or
 * of a process in another PID namespace sharing the directory, neither
 * block it nor get touched). The name is unlinked right after the file is
 * created, and the content lives in an open descriptor (O_CLOEXEC) that
 * path() names as /proc/self/fd/<fd>. On Linux, opening that path opens
 * the file afresh, with its own offset, so every Input::file() call -- on
 * any thread of this process -- reads the whole script from the start. The
 * kernel frees the file when the last descriptor closes, however the
 * process ends: a PC_ISOLATION worker SIGKILLed at its hard deadline,
 * crashing inside LAMMPS or killed with its parent (PR_SET_PDEATHSIG) used
 * to leave a named tadah_pc_<pid>_<n>.in behind, because only the
 * destructor removed it. (Only a kill between the creation and the unlink
 * that follows it could still leave a file.) path() is valid in THIS
 * process only.
 *
 * Identical renderings are shared, process-wide, while one is alive: every
 * THREAD-mode slot renders the same script, and one descriptor per distinct
 * script (not per slot and script) keeps a wide THREADS run far from the
 * descriptor limit. */

#include <cerrno>
#include <cstdlib>    /* mkostemps */
#include <cstring>
#include <filesystem>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <unistd.h>   /* getpid, write, unlink, close */

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
    std::string text;
    for (const auto& line : script_lines) {
      text += line;
      if (line.empty() || line.back() != '\n') text += '\n';
      if (!has_extern_ &&
          line.find("pair_coeff") != std::string::npos &&
          line.find("EXTERN") != std::string::npos)
        has_extern_ = true;
    }
    rendering_ = Rendering::shared(text);
  }

  ScriptFile(ScriptFile&& o) noexcept
      : rendering_(std::move(o.rendering_)), has_extern_(o.has_extern_) {}
  ScriptFile& operator=(ScriptFile&& o) noexcept {
    if (this != &o) {
      rendering_ = std::move(o.rendering_);
      has_extern_ = o.has_extern_;
    }
    return *this;
  }
  ScriptFile(const ScriptFile&) = delete;
  ScriptFile& operator=(const ScriptFile&) = delete;

  ~ScriptFile() = default;

  /* /proc/self/fd/<fd>: readable by this process while this object (or
   * another rendering of the same text) is alive. */
  const std::string& path() const {
    static const std::string none;
    return rendering_ ? rendering_->path : none;
  }
  bool has_extern() const { return has_extern_; }
  bool valid() const { return static_cast<bool>(rendering_); }

private:
  /* One unlinked temp file holding `text`, open as `fd`. */
  struct Rendering {
    int fd = -1;
    std::string path;

    explicit Rendering(const std::string& text) {
      namespace fs = std::filesystem;
      static constexpr int kSuffixLen = 3;   /* ".in" */
      const fs::path dir = fs::temp_directory_path();
      std::string name = (dir / ("tadah_pc_" + std::to_string(::getpid()) +
                                 "_XXXXXX.in")).string();
      /* mkostemps replaces XXXXXX with a random part and creates the file
       * with O_EXCL and mode 0600, trying other names while one exists. */
      fd = ::mkostemps(name.data(), kSuffixLen, O_CLOEXEC);
      if (fd < 0)
        throw std::runtime_error("ScriptFile: cannot create a file in " +
                                 dir.string() + ": " + std::strerror(errno));
      /* The name goes at once: from here on the file exists only through
       * the descriptor. */
      if (::unlink(name.c_str()) != 0) {
        const int err = errno;
        ::close(fd);
        throw std::runtime_error("ScriptFile: cannot unlink " + name + ": " +
                                 std::strerror(err));
      }
      std::size_t off = 0;
      while (off < text.size()) {
        const ssize_t w = ::write(fd, text.data() + off, text.size() - off);
        if (w < 0) {
          if (errno == EINTR) continue;
          const int err = errno;
          ::close(fd);
          throw std::runtime_error("ScriptFile: cannot write the rendered "
                                   "script (" + name + ", unlinked): " +
                                   std::strerror(err));
        }
        off += static_cast<std::size_t>(w);
      }
      path = "/proc/self/fd/" + std::to_string(fd);
      /* LAMMPS opens path() by name; fail here, with the reason, if this
       * system has no /proc/self/fd. */
      const int probe = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
      if (probe < 0) {
        const int err = errno;
        ::close(fd);
        throw std::runtime_error("ScriptFile: cannot reopen the rendered "
                                 "script through " + path + ": " +
                                 std::strerror(err));
      }
      ::close(probe);
    }
    ~Rendering() { if (fd >= 0) ::close(fd); }
    Rendering(const Rendering&) = delete;
    Rendering& operator=(const Rendering&) = delete;

    /* The live rendering of `text`, or a new one. */
    static std::shared_ptr<const Rendering> shared(const std::string& text) {
      static std::mutex m;
      static std::unordered_map<std::string, std::weak_ptr<const Rendering>>
          live;
      std::lock_guard<std::mutex> lock(m);
      if (auto it = live.find(text); it != live.end()) {
        if (auto r = it->second.lock()) return r;
      }
      for (auto it = live.begin(); it != live.end();) {
        if (it->second.expired()) it = live.erase(it);
        else ++it;
      }
      auto r = std::make_shared<const Rendering>(text);
      live[text] = r;
      return r;
    }
  };

  std::shared_ptr<const Rendering> rendering_;
  bool has_extern_ = false;
};

}  // namespace hpo
}  // namespace tadah

#endif  // TADAH_HPO_SCRIPT_FILE_H
