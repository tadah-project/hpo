#pragma once
/* The Ta potential of the LAMMPS-driven cases (bench_pair_tadah.cpp,
 * test_hpo_phase3_pilot.cpp, test_lammps_hessian.cpp; CTest
 * hpo_bench_pair_tadah, hpo_phase3_pilot, hpo_lammps_hessian, label bench):
 *
 *  1. TADAH_BENCH_POT, when set: the user names a potential to run them
 *     with, so a file that cannot be read is an error.
 *  2. Otherwise the potential shipped with the tests,
 *     TADAH_BENCH_DEFAULT_POT (tests/data/bench_ta.tadah, whose header
 *     gives its provenance). It is part of the source tree: a missing or
 *     unreadable one is an error too.
 *
 * An error is a std::runtime_error, which Catch reports as a failure of
 * the test case: there is no skip. The potential must also load and run:
 * the cases check that with REQUIRE. */

#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#ifndef TADAH_BENCH_DEFAULT_POT
#error "TADAH_BENCH_DEFAULT_POT (HPO/tests/CMakeLists.txt) names the shipped potential"
#endif

namespace hpo_test {

inline std::string bench_pot_path()
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
        return env;
    }
    const std::string shipped = TADAH_BENCH_DEFAULT_POT;
    if (!readable(shipped))
        throw std::runtime_error("the shipped benchmark potential " + shipped +
                                 " cannot be read");
    return shipped;
}

}  // namespace hpo_test
