// test_lammps_sim_negative_values.cpp
//
// PC_LAMMPS option values may be negative numbers. A token that starts
// with '-' is an option only when it is not a number: a target such as a
// cohesive energy (-8.1 eV) or a phase-enthalpy difference (-0.0674 eV/atom)
// is a value, and so are negative --hotvar bounds. Options (--outvar,
// --script, ...) must still terminate a value list. The reject line of a
// hot value outside its window prints every number losslessly.

#include "catch2/catch.hpp"

#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/loss_function.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <limits>
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

TEST_CASE("PC_LAMMPS --hotvar: the reject line prints the value and the "
          "window losslessly", "[lammps_sim][hotvar]")
{
    /* std::to_string's six decimals printed a value one ulp above a max of
     * 3.6 as 3.600000 -- the max itself, so the line read as a value
     * inside its window -- and one below 5e-7 as 0.000000. The line now
     * carries the shortest round-trip form (core::format_roundtrip), as
     * BUMPGATE's reject line does. */
    tadah::hpo::LammpsSim::HotVarDecl hv;
    hv.name = "a0";
    hv.min = 3.0;
    hv.max = 3.6;
    CHECK(tadah::hpo::hotvar_reject_what(hv, std::nextafter(3.6, 4.0)) ==
          "a0=3.6000000000000005 outside [3, 3.6]");
    hv.name = "rho";
    hv.min = 1e-6;
    hv.max = 1.0;
    CHECK(tadah::hpo::hotvar_reject_what(hv, 2.5e-7) ==
          "rho=2.5e-07 outside [1e-06, 1]");
    CHECK(tadah::hpo::hotvar_reject_what(hv, -std::numeric_limits<double>::infinity()) ==
          "rho=-inf outside [1e-06, 1]");
    const double third = 1.0 / 3.0;
    hv.min = 0.5;
    const std::string t = tadah::hpo::hotvar_reject_what(hv, third);
    CHECK(std::stod(t.substr(4, t.find(' ') - 4)) == third);
}
