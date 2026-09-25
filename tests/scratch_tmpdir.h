#pragma once
/* ScratchTmpdir: point TMPDIR at a fresh, empty directory for the life of
 * one test, so the test can count what the code under test leaves in the
 * temporary directory without seeing other processes' files.
 *
 * std::filesystem::temp_directory_path() reads TMPDIR on every call, and a
 * PC_ISOLATION worker (fork + execv) inherits the environment at spawn, so
 * both the test process and the workers it starts afterwards use the
 * scratch directory. The destructor restores TMPDIR (or unsets it if it was
 * unset) and removes the directory. Not thread-safe: construct it before
 * the code under test starts threads or workers, destroy it after. */

#include <cstdlib>
#include <filesystem>
#include <optional>
#include <stdexcept>
#include <string>
#include <system_error>

#include <unistd.h>

namespace hpo_test {

class ScratchTmpdir {
public:
    explicit ScratchTmpdir(const std::string& tag)
    {
        namespace fs = std::filesystem;
        if (const char* old = std::getenv("TMPDIR")) old_ = std::string(old);
        static unsigned long n = 0;
        dir_ = fs::temp_directory_path() /
               ("tadah_test_" + tag + "_" + std::to_string(::getpid()) +
                "_" + std::to_string(n++));
        fs::remove_all(dir_);
        fs::create_directories(dir_);
        if (::setenv("TMPDIR", dir_.c_str(), 1) != 0)
            throw std::runtime_error("ScratchTmpdir: setenv TMPDIR failed");
        if (fs::temp_directory_path() != dir_)
            throw std::runtime_error("ScratchTmpdir: TMPDIR is not honoured");
    }

    ~ScratchTmpdir()
    {
        if (old_) ::setenv("TMPDIR", old_->c_str(), 1);
        else      ::unsetenv("TMPDIR");
        std::error_code ec;
        std::filesystem::remove_all(dir_, ec);
    }

    ScratchTmpdir(const ScratchTmpdir&) = delete;
    ScratchTmpdir& operator=(const ScratchTmpdir&) = delete;

    const std::filesystem::path& path() const { return dir_; }

    /* Number of entries of the directory whose name starts with prefix. */
    std::size_t count(const std::string& prefix) const
    {
        std::size_t k = 0;
        for (const auto& e : std::filesystem::directory_iterator(dir_))
            if (e.path().filename().string().rfind(prefix, 0) == 0) ++k;
        return k;
    }

private:
    std::filesystem::path dir_;
    std::optional<std::string> old_;
};

}  // namespace hpo_test
