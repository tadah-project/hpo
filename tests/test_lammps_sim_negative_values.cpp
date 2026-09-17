// test_lammps_sim_negative_values.cpp
//
// PC_LAMMPS option values may be negative numbers. A token that starts
// with '-' is an option only when it is not a number: a target such as a
// cohesive energy (-8.1 eV) or a phase-enthalpy difference (-0.0674 eV/atom)
// is a value, and so are negative --hotvar bounds. Options (--outvar,
// --script, ...) must still terminate a value list.

#include "catch2/catch.hpp"

#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unistd.h>

namespace {

struct TempScript {
    std::string path;
    TempScript() {
        static int counter = 0;
        path = (std::filesystem::temp_directory_path()
                / ("test_lammps_sim_negative_" + std::to_string(::getpid())
                   + "_" + std::to_string(++counter) + ".in")).string();
        std::ofstream os(path);
        os << "clear\nrun 0\n"
              "variable E equal 0.0\n"
              "variable F equal 0.0\n"
              "variable a0 equal 3.3\n";
    }
    ~TempScript() { std::remove(path.c_str()); }
};

tadah::hpo::LammpsSim make_sim(std::string cmd) {
    return tadah::hpo::LammpsSim(cmd, "1", 1.0e6, tadah::LossFunction{});
}

}  // namespace

TEST_CASE("PC_LAMMPS --varloss accepts negative numeric values",
          "[lammps_sim][parse][negative]")
{
    TempScript s;

    SECTION("negative target") {
        auto sim = make_sim("--script " + s.path + " --varloss E -0.0674 1.0e6");
        REQUIRE(sim.n_varloss.size() == 1);
        CHECK(sim.n_varloss[0] == "E");
        CHECK(sim.t_varloss[0] == Approx(-0.0674));
        CHECK(sim.w_varloss[0] == Approx(1.0e6));
    }
    SECTION("negative target in exponent and leading-dot forms") {
        auto sim = make_sim("--script " + s.path
                            + " --varloss E -1.65e-5 1.0 --varloss F -.5 2.0");
        REQUIRE(sim.n_varloss.size() == 2);
        CHECK(sim.t_varloss[0] == Approx(-1.65e-5));
        CHECK(sim.t_varloss[1] == Approx(-0.5));
        CHECK(sim.w_varloss[1] == Approx(2.0));
    }
    SECTION("negative weight") {
        auto sim = make_sim("--script " + s.path + " --varloss E 1.0 -2.0");
        REQUIRE(sim.n_varloss.size() == 1);
        CHECK(sim.w_varloss[0] == Approx(-2.0));
    }
    SECTION("negative target followed by another option") {
        auto sim = make_sim("--script " + s.path
                            + " --varloss E -8.1 800 --outvar F");
        REQUIRE(sim.n_varloss.size() == 1);
        CHECK(sim.t_varloss[0] == Approx(-8.1));
        CHECK(sim.w_varloss[0] == Approx(800.0));
        REQUIRE(sim.n_outvar.size() == 2);
        CHECK(sim.n_outvar[1] == "F");
    }
}

TEST_CASE("PC_LAMMPS --hotvar accepts negative bounds and initial value",
          "[lammps_sim][parse][negative]")
{
    TempScript s;
    auto sim = make_sim("--script " + s.path
                        + " --varloss E 1.0 1.0 --hotvar a0 -1.0 1.0 -0.5");
    REQUIRE(sim.hotvars.size() == 1);
    CHECK(sim.hotvars[0].name == "a0");
    CHECK(sim.hotvars[0].min == Approx(-1.0));
    CHECK(sim.hotvars[0].max == Approx(1.0));
    REQUIRE(sim.hotvars[0].initial.has_value());
    CHECK(*sim.hotvars[0].initial == Approx(-0.5));
}

TEST_CASE("PC_LAMMPS options still terminate a --varloss value list",
          "[lammps_sim][parse][negative]")
{
    TempScript s;
    SECTION("an option in place of the weight is a missing weight") {
        REQUIRE_THROWS_WITH(make_sim("--script " + s.path
                                     + " --varloss E 1.0 --outvar F"),
                            Catch::Contains("missing <weight>"));
    }
    SECTION("an option in place of the target is a missing target") {
        REQUIRE_THROWS_WITH(make_sim("--script " + s.path
                                     + " --varloss E --outvar F"),
                            Catch::Contains("missing <target>"));
    }
}
