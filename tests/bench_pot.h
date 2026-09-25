#pragma once
/* The Ta potential of the hidden LAMMPS-driven cases ([.][!benchmark]:
 * bench_pair_tadah.cpp, test_hpo_phase3_pilot.cpp, test_lammps_hessian.cpp).
 *
 *  1. TADAH_BENCH_POT, when set. The user asks for these cases to run, so
 *     a file that cannot be read is an error.
 *  2. Otherwise TADAH_BENCH_DEFAULT_POT (HPO/tests/CMakeLists.txt:
 *     $TADAH_PATH/HPO_test/pot.tadah at configure time), when it exists.
 *     If it exists but cannot be read, that is an error too.
 *  3. Otherwise there is no potential: path is empty, absent_reason says
 *     why, and the case calls TADAH_SKIP(absent_reason) (tadah_test_skip.hpp)
 *     so that it is reported as skipped, not passed.
 *
 * An error is a std::runtime_error, which Catch reports as a failure of
 * the test case. A potential that is found must also load and run: the
 * cases check that with REQUIRE, never by skipping. */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

namespace hpo_test {

struct BenchPot {
    std::string path;           // empty: no potential; the case skips
    std::string absent_reason;  // why, when path is empty
};

inline BenchPot find_bench_pot()
{
    const auto readable = [](const std::string& p) {
        std::ifstream in(p);
        return in.good() && !std::filesystem::is_directory(p);
    };
    const char* env = std::getenv("TADAH_BENCH_POT");
    if (env && *env) {
        if (!readable(env))
            throw std::runtime_error(std::string("TADAH_BENCH_POT=") + env +
                                     " cannot be read");
        return {env, ""};
    }
#ifdef TADAH_BENCH_DEFAULT_POT
    const std::string def = TADAH_BENCH_DEFAULT_POT;
    if (!std::filesystem::exists(def))
        return {"", "no potential: TADAH_BENCH_POT is unset and the default "
                    "potential " + def + " does not exist"};
    if (!readable(def))
        throw std::runtime_error("the default benchmark potential " + def +
                                 " cannot be read");
    return {def, ""};
#else
    return {"", "no potential: TADAH_BENCH_POT is unset, and TADAH_PATH was "
                "unset when the tests were configured (no default potential)"};
#endif
}

}  // namespace hpo_test
