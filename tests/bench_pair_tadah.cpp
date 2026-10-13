// bench_pair_tadah.cpp
//
// Phase 2A.5 microbench gate. Drives pair_tadah::compute on a 256-atom
// bcc Ta cell and lets a CI step gate PRs on median throughput.
//
// The test is tagged [.][!benchmark], so it is hidden (not discovered);
// the CTest entry hpo_bench_pair_tadah (label bench) runs it with Catch2's
// default benchmark settings, which takes a few seconds. To run it by hand:
//
//   build_tadah/_deps/tadah.hpo-build/tests/test_hpo \
//       "[bench][pair_tadah]" --benchmark-samples 50
//
// Catch2 v2.13 (this build) ships console/compact/junit/xml reporters
// only — there is no csv reporter despite the original plan note. The
// default console reporter prints the bench summary inline; parse from
// stdout for CI gating, or use --reporter xml --out bench.xml.
//
// The potential (bench_pot.h): TADAH_BENCH_POT when set (a run-time
// override), else the one shipped with the tests, tests/data/bench_ta.tadah
// (the Quick Start Ta potential: RCUT2B 5.3 Å, RCUTMB 5.0 Å). One that
// cannot be read, parsed or run is a failure; there is no skip.
//
// Cell resolution (in order):
//   1. TADAH_BENCH_CELL_DATA env var: path to a LAMMPS read_data file.
//   2. Inline default: 4×4×8 bcc lattice at a₀ = 3.301 Å (256 atoms,
//      box 13.2 × 13.2 × 26.4 Å). Every dimension exceeds 2·rcut for the
//      shipped potential (5.3 Å), so neighbour lists are
//      non-degenerate and the timing reflects realistic pair counts.

#include "catch2/catch.hpp"

#include "bench_pot.h"

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>

#include <force.h>
#include <pair.h>

#include <cstdlib>
#include <fstream>
#include <string>
#include <vector>

namespace {

std::string env_or(const char* name) {
    const char* v = std::getenv(name);
    return v ? std::string(v) : std::string();
}

// 256-atom bcc Ta cell at a₀ ≈ 3.301 Å. 4×4×8 supercell — every box
// dimension exceeds 2·rcut for the shipped 5.3-Å pot so the neighbour
// list isn't degenerate. `run 0` builds the neighbour list and runs an
// initial compute so subsequent pair->compute calls in the BENCHMARK
// body time only the hot path.
std::vector<std::string> inline_bcc_ta_script() {
    return {
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 3.301",
        "region box block 0 4 0 4 0 8 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "run 0",
    };
}

std::vector<std::string> data_file_script(const std::string& cell_path) {
    return {
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "read_data " + cell_path,
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 bin",
        "neigh_modify delay 10 check yes",
        "run 0",
    };
}

}  // namespace

TEST_CASE("pair_tadah microbench gate (256-atom bcc Ta)",
          "[bench][pair_tadah][.][!benchmark]")
{
    const std::string pot_path = hpo_test::bench_pot_path();
    tadah::core::PeriodicTable::initialize();   // ATOM symbols of the pot

    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx =
        tadah::core::CoreUserConfigParser().parse(pot_path));

    LammpsRunner runner;
    auto lmp = runner.create_instance(
        {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);

    const std::string cell_data = env_or("TADAH_BENCH_CELL_DATA");
    std::vector<std::string> setup = cell_data.empty()
        ? inline_bcc_ta_script()
        : data_file_script(cell_data);
    REQUIRE(runner.run_script(lmp, setup, pot_ctx));

    LAMMPS_NS::Pair* pair = lmp->force->pair;
    REQUIRE(pair != nullptr);

    BENCHMARK("pair_tadah::compute on 256-atom bcc Ta")
    {
        pair->compute(/*eflag=*/1, /*vflag=*/1);
        // Anti-DCE: combine outputs the compiler can't prove are unused.
        return pair->eng_vdwl + pair->virial[0];
    };
}
