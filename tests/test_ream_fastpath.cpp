// test_ream_fastpath.cpp
//
// DM_REAM through the real pair_style tadah (LammpsRunner): the EAMFAST
// collapse kernel vs the generic kernel, under both embedding references
// (EMBF0 ZERO, the default, and EMBF0 FREE). Fixture: the compact Knot5 Ta
// refit of HPO_REFIT_2B_EAM (13 pair knots, 8 density knots, 15 embedding
// knots, several of them negative, so the raw F(0) != 0). Covers:
//   - isolated atom: pe == 0.0 exactly under ZERO on both kernels; pe equals
//     the raw F(0) under FREE;
//   - rattled bcc: forces, fnorm and every pressure component bitwise equal
//     between ZERO and FREE on the same kernel (F' is unchanged);
//   - (E_ZERO - E_FREE)/N == -F(0);
//   - EAMFAST vs generic parity under ZERO and FREE: energy (relative
//     1e-9, as for pure 2B), fnorm, virial pressure and forces;
//   - a potential without an EMBF0 line is evaluated as ZERO;
//   - pair_coeff prints a LAMMPS WARNING (screen + log.lammps) for such a
//     potential on every load, and nothing new for a declared one (BUG-08);
//   - pair_coeff states the density form (RHOFORM) on every load, and a
//     potential without RHOFORM fails with the migration text in the
//     pair_coeff ERROR (log.lammps and the exception the driver catches),
//     also on pair_coeff * * EXTERN after a driver's failed load();
//   - the density forms through both kernels (a Blip5 fixture, since
//     MONOTONE refuses the knot family): NONNEG equals SIGNED exactly,
//     MONOTONE differs and its EAMFAST run matches the generic one.

#include "catch2/catch.hpp"

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <force.h>
#include <input.h>
#include <pair.h>
#include <pair_tadah.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <string>
#include <tuple>
#include <unistd.h>
#include <vector>

namespace {

const char* TA_REAM_POT_BODY =
    "AMPGRIDMB         0.009708244690774269   0.009674844385395990   -0.025225004679995621   -0.009325942537791374   -1.222676951476143525   -0.001952139603524065   0.015322895526324260   1.252682486527901595\n"
    "ATOM              Ta\n"
    "BIAS              false\n"
    "CEMBFUNC          0.905701494783395344   1.097482295774922179   -2.965526708599611094   -2.904894547222190315   0.180799209360483659   -3.026068634475253205   1.635708786836001183   0.527197892951199676   -0.026314766355414316   1.032650453676995594   0.032521730189707422   1.288054212113806107   0.772710595184760929   0.130355308372253259   0.085105065336974997\n"
    "CGRID2B           5.456548172178544931   5.240680385246352380   2.866382601547265985   3.054828424872450210   4.483083941769357317   1.716808776326026242   2.393707442941635843   2.844960054504627855   3.749657037231673318   3.936065544763269308   3.219822611526424794   4.726770772280009858   1.716808818354101041\n"
    "CGRIDMB           5.484743755492014650   4.425251978455539970   5.122289321044224408   3.859860916521183238   2.509299068926699405   3.395167852661002073   4.646000519334722689   2.504754665526272017\n"
    "FIXINDEX          1-28\n"
    "FIXWEIGHT         -0.014641348094898586   -0.051535959944426050   -41.156393764756018072   3.808410339336090633   0.188032523218672837   -3058.915888454833293508   -3.511803288240705534   36.979581171997139677   -0.549561123341020474   -0.334973899538970332   2.145274625352853182   0.142458455160975661   35.268028553999059227   222.913408871893182095   223.783694639092914258   -168.806095957493624837   90.719128161416477951   5304.663560913277251530   78.628875194741951304   7.352619952010690518   29.817892005298819669   5136.174514981846186856   -298.436161734381755650   -23120.745531707521877252   -51.786960138537409648   -121.723616579840154373   -25539.485778202957590111   38206.820714959903853014\n"
    "FORCE             false\n"
    "INIT2B            true\n"
    "INITMB            true\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR   BF_Linear\n"
    "NORM              false\n"
    "OALGO             2\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCTYPEMB          Cut_Dummy\n"
    "RCUT2B            5.581907420240000306\n"
    "RCUTMB            5.581907420240000306\n"
    "RHOFORM           SIGNED\n"
    "STRESS            false\n"
    "TYPE2B            D2_Knot5   13   0   Ta   Ta\n"
    "TYPEMB            DM_REAM   2   8   0   8   15   0   0   0   Ta   Ta\n"
    "WATOM             1.000000000000000000\n"
    "WEIGHTS           -0.014641348094898586   -0.051535959944426050   -41.156393764756018072   3.808410339336090633   0.188032523218672837   -3058.915888454833293508   -3.511803288240705534   36.979581171997139677   -0.549561123341020474   -0.334973899538970332   2.145274625352853182   0.142458455160975661   35.268028553999059227   222.913408871893182095   223.783694639092914258   -168.806095957493624837   90.719128161416477951   5304.663560913277251530   78.628875194741951304   7.352619952010690518   29.817892005298819669   5136.174514981846186856   -298.436161734381755650   -23120.745531707521877252   -51.786960138537409648   -121.723616579840154373   -25539.485778202957590111   38206.820714959903853014\n";

/* Unique per process (ctest runs cases as concurrent processes). With
 * rhoform false the body's RHOFORM line is left out. */
std::string write_pot(bool eamfast, const char* embf0, bool rhoform = true) {
    const std::string path = "ream_fast_" + std::to_string(::getpid()) + "_"
        + (eamfast ? "on" : "off") + "_" + (embf0 ? embf0 : "absent")
        + (rhoform ? "" : "_no_rhoform") + ".tadah";
    std::string body = TA_REAM_POT_BODY;
    if (!rhoform) {
        const std::string line = "RHOFORM           SIGNED\n";
        const auto at = body.find(line);
        REQUIRE(at != std::string::npos);
        body.erase(at, line.size());
    }
    std::ofstream ofs(path);
    ofs << body;
    ofs << "EAMFAST           " << (eamfast ? "true" : "false") << "\n";
    if (embf0) ofs << "EMBF0             " << embf0 << "\n";
    return path;
}

std::vector<std::string> isolated_atom() {
    return {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "region box block 0 40 0 40 0 40 units box",
        "create_box 1 box",
        "create_atoms 1 single 20 20 20 units box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 nsq",
        "variable e equal pe",
        "run 0 post no",
    };
}

/* 4x4x4 bcc Ta (128 atoms) at 3.304 A, deterministically rattled. */
std::vector<std::string> rattled_bcc_ta() {
    std::vector<std::string> s = {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 3.304",
        "region box block 0 4 0 4 0 4 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 nsq",
        "displace_atoms all random 0.05 0.05 0.05 777 units box",
        "compute pv all pressure NULL virial",
        "variable e equal pe",
        "variable n equal atoms",
        "variable fn equal fnorm",
        "variable p1 equal c_pv[1]",
        "variable p2 equal c_pv[2]",
        "variable p3 equal c_pv[3]",
        "variable p4 equal c_pv[4]",
        "variable p5 equal c_pv[5]",
        "variable p6 equal c_pv[6]",
    };
    for (int i = 1; i <= 128; i += 9) {
        const std::string k = std::to_string(i);
        s.push_back("variable fx" + k + " equal fx[" + k + "]");
        s.push_back("variable fy" + k + " equal fy[" + k + "]");
        s.push_back("variable fz" + k + " equal fz[" + k + "]");
    }
    s.push_back("run 0 post no");
    return s;
}

struct Run {
    double e_iso = 0.0;
    double raw_f0 = 0.0;
    double s0 = 0.0;   // sum_m |w_m g_m(0)| (all embedding families: g >= 0)
    bool fast = false;
    double e = 0.0;
    int n = 0;
    std::vector<std::string> exact;   // forces, fnorm, pressures (lossless)
};

Run run_case(LammpsRunner& runner, bool eamfast, const char* embf0) {
    Run r;
    const std::string pot_path = write_pot(eamfast, embf0);
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx =
        tadah::core::CoreUserConfigParser().parse(pot_path));

    auto lmp = runner.create_instance(
        0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);

    REQUIRE(runner.run_script(lmp, isolated_atom(), pot_ctx));
    r.e_iso = std::stod(runner.get_variable(lmp, "e"));

    REQUIRE(runner.run_script(lmp, rattled_bcc_ta(), pot_ctx));
    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pt != nullptr);
    auto* lt = pt->lammps_tadah();
    r.fast = lt->eam_fast;
    tadah::core::t_type w;
    w.resize(lt->c.size("WEIGHTS"));
    lt->c.get<tadah::core::t_type>("WEIGHTS", w);
    r.raw_f0 = *lt->S.dmb->raw_embedding_f0(w);
    tadah::core::t_type wabs = w;
    for (std::size_t i = 0; i < wabs.size(); ++i) wabs(i) = std::fabs(w(i));
    r.s0 = *lt->S.dmb->raw_embedding_f0(wabs);
    r.e = std::stod(runner.get_variable(lmp, "e"));
    r.n = std::stoi(runner.get_variable(lmp, "n"));
    for (const char* v : {"fn", "p1", "p2", "p3", "p4", "p5", "p6"})
        r.exact.push_back(runner.get_variable(lmp, v));
    for (int i = 1; i <= 128; i += 9)
        for (const char* c : {"fx", "fy", "fz"})
            r.exact.push_back(
                runner.get_variable(lmp, c + std::to_string(i)));
    std::remove(pot_path.c_str());
    return r;
}

/* The largest kernel-to-kernel differences of two runs' observables:
 * fnorm relative to itself; the six virial pressure components relative to
 * the largest of them; the sampled force components relative to the
 * largest sampled magnitude (a component near zero is measured against the
 * scale of the forces, not against itself). */
struct Parity {
    double fnorm = 0.0, pressure = 0.0, force = 0.0;
};

Parity parity(const Run& a, const Run& b) {
    REQUIRE(a.exact.size() == b.exact.size());
    REQUIRE(a.exact.size() > 7);
    std::vector<double> x(a.exact.size()), y(b.exact.size());
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = std::stod(a.exact[i]);
        y[i] = std::stod(b.exact[i]);
    }
    const auto max_rel = [&](std::size_t lo, std::size_t hi) {
        double scale = 0.0, diff = 0.0;
        for (std::size_t i = lo; i < hi; ++i) {
            scale = std::max({scale, std::fabs(x[i]), std::fabs(y[i])});
            diff = std::max(diff, std::fabs(x[i] - y[i]));
        }
        REQUIRE(scale > 0.0);
        return diff / scale;
    };
    Parity p;
    p.fnorm = max_rel(0, 1);          // exact[0]: fnorm
    p.pressure = max_rel(1, 7);       // exact[1..6]: c_pv[1..6]
    p.force = max_rel(7, x.size());   // sampled fx, fy, fz
    return p;
}

std::string slurp(const std::string& path) {
    std::ifstream ifs(path);
    return std::string(std::istreambuf_iterator<char>(ifs),
                       std::istreambuf_iterator<char>());
}

int count_of(const std::string& hay, const std::string& needle) {
    int n = 0;
    for (auto p = hay.find(needle); p != std::string::npos;
         p = hay.find(needle, p + needle.size()))
        ++n;
    return n;
}

/* Removes a scratch file however the test case ends (a failed REQUIRE
 * throws past a trailing std::remove). */
struct RemoveOnExit {
    std::string path;
    ~RemoveOnExit() { std::remove(path.c_str()); }
};

/* log.lammps of one isolated-atom run that loads @p pot_path by file name
 * (pair_coeff * * <file> Ta, the plain LAMMPS user path); @p expect_ok:
 * whether the run must succeed. */
std::string log_of_file_load(LammpsRunner& runner, const std::string& pot_path,
                             const std::string& tag, bool expect_ok = true) {
    const std::string stem = "ream_notice_" + std::to_string(::getpid())
        + "_" + tag;
    const std::string script = stem + ".in", log = stem + ".log";
    {
        std::ofstream ofs(script);
        for (const auto& line : isolated_atom())
            ofs << (line == "pair_coeff * * EXTERN Ta"
                        ? "pair_coeff * * " + pot_path + " Ta"
                        : line)
                << "\n";
    }
    {
        auto lmp = runner.create_instance(
            0, {"lmp", "-log", log, "-screen", "none"});
        REQUIRE(lmp != nullptr);
        REQUIRE(runner.run_script(lmp, script) == expect_ok);
    }   // LAMMPS destroyed: log.lammps flushed and closed
    const std::string text = slurp(log);
    std::remove(script.c_str());
    std::remove(log.c_str());
    return text;
}

}  // namespace

TEST_CASE("DM_REAM fast path: EMBF0 ZERO/FREE on EAMFAST and generic kernels",
          "[pair_tadah][ream][eamfast][embf0]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;

    const Run zf = run_case(runner, true,  "ZERO");
    const Run zg = run_case(runner, false, "ZERO");
    const Run ff = run_case(runner, true,  "FREE");
    const Run fg = run_case(runner, false, "FREE");
    const Run af = run_case(runner, true,  nullptr);   // no EMBF0 line

    REQUIRE(zf.fast);
    REQUIRE_FALSE(zg.fast);
    REQUIRE(ff.fast);
    REQUIRE_FALSE(fg.fast);

    const double f0 = ff.raw_f0;
    REQUIRE(f0 != 0.0);                       // the fixture is sensitive
    /* Rounding floor of a sum that cancels down from S0 = sum |w g(0)|. */
    const double eps = std::numeric_limits<double>::epsilon();
    const double floor1 = 64.0 * eps * ff.s0;

    /* Isolated atom: exactly 0 under ZERO on both kernels, raw F(0) under
     * FREE. */
    REQUIRE(zf.e_iso == 0.0);
    REQUIRE(zg.e_iso == 0.0);
    REQUIRE(std::fabs(ff.e_iso - f0) <= floor1);
    REQUIRE(std::fabs(fg.e_iso - f0) <= floor1);

    /* Forces and virial: bitwise identical between the references. */
    REQUIRE(zf.exact == ff.exact);
    REQUIRE(zg.exact == fg.exact);

    /* Energies differ by -F(0) per atom. */
    REQUIRE(zf.n == 128);
    const double tol = zg.n * floor1 + 64.0 * eps * std::fabs(fg.e);
    REQUIRE(std::fabs((zg.e - fg.e) + zg.n * f0) < tol);
    REQUIRE(std::fabs((zf.e - ff.e) + zf.n * f0) < tol);

    /* EAMFAST vs generic, under both references: the energy to the
     * pure-2B collapse tolerance; fnorm, the virial pressure and the
     * sampled forces too. The tables are certified to rel 1e-9
     * (kCollapseRelTol); a force or a virial component sums many pair and
     * embedding terms, and differs by about 1.3e-9 of the force scale (1.4e-9
     * of the pressure scale) here. 1e-8 leaves an order of magnitude for
     * the rounding, where a wrong table, derivative or embedding shift in
     * either kernel shows at the percent level. */
    REQUIRE(zf.e == Approx(zg.e).epsilon(1e-9));
    REQUIRE(ff.e == Approx(fg.e).epsilon(1e-9));
    for (const auto& [name, fast, gen] :
         {std::tuple<const char*, const Run*, const Run*>{"ZERO", &zf, &zg},
          std::tuple<const char*, const Run*, const Run*>{"FREE", &ff, &fg}}) {
        INFO("EMBF0 " << name);
        const Parity p = parity(*fast, *gen);
        CAPTURE(p.fnorm, p.pressure, p.force);
        CHECK(p.fnorm <= 1e-9);
        CHECK(p.pressure <= 1e-8);
        CHECK(p.force <= 1e-8);
    }

    /* No EMBF0 line: evaluated as ZERO. */
    REQUIRE(af.e_iso == 0.0);
    REQUIRE(af.e == zf.e);
    REQUIRE(af.exact == zf.exact);
}

TEST_CASE("pair_coeff warns when a DM_REAM potential has no EMBF0 line",
          "[pair_tadah][ream][embf0][notice]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string warn = "has no EMBF0 line";
    const std::string absent_banner =
        "EMBF0 ZERO (key absent: default ZERO, F(0) = 0 by construction; "
        "add EMBF0 FREE to reproduce tadah <= 1.5.0 energies)";

    SECTION("pair_coeff * * <file>: WARNING only for the key-less potential") {
        const std::string p_abs = write_pot(true, nullptr);
        const std::string p_zero = write_pot(true, "ZERO");
        const std::string p_free = write_pot(true, "FREE");
        const std::string l_abs = log_of_file_load(runner, p_abs, "absent");
        const std::string l_zero = log_of_file_load(runner, p_zero, "zero");
        const std::string l_free = log_of_file_load(runner, p_free, "free");
        for (const auto& p : {p_abs, p_zero, p_free}) std::remove(p.c_str());

        REQUIRE(count_of(l_abs, "WARNING: pair tadah: potential '" + p_abs +
                                    "' " + warn) == 1);
        REQUIRE(count_of(l_abs, absent_banner) == 1);
        // linear model: the energies-only text, with the migration
        REQUIRE(count_of(l_abs, "forces and stresses are unchanged") == 1);
        REQUIRE(count_of(l_abs, "add 'EMBF0 FREE' to the potential") == 1);

        for (const auto* l : {&l_zero, &l_free}) {
            REQUIRE(count_of(*l, warn) == 0);
            REQUIRE(count_of(*l, "key absent") == 0);
        }
        REQUIRE(count_of(l_zero, "Tadah! DM_REAM embedding reference EMBF0 "
                                 "ZERO (F(0) = 0 by construction)\n") == 1);
        REQUIRE(count_of(l_free, "Tadah! DM_REAM embedding reference EMBF0 "
                                 "FREE (raw embedding of tadah <= 1.5.0)\n")
                == 1);
    }

    SECTION("pair_coeff * * EXTERN: WARNING on every load") {
        const std::string pot_path = write_pot(true, nullptr);
        tadah::core::Context pot_ctx;
        REQUIRE_NOTHROW(pot_ctx =
            tadah::core::CoreUserConfigParser().parse(pot_path));
        std::remove(pot_path.c_str());
        const std::string log = "ream_notice_" + std::to_string(::getpid())
            + "_extern.log";
        {
            auto lmp = runner.create_instance(
                0, {"lmp", "-log", log, "-screen", "none"});
            REQUIRE(lmp != nullptr);
            REQUIRE(runner.run_script(lmp, isolated_atom(), pot_ctx));
            REQUIRE(runner.run_script(lmp, isolated_atom(), pot_ctx));
        }
        const std::string text = slurp(log);
        std::remove(log.c_str());
        REQUIRE(count_of(text, "WARNING: pair tadah: potential 'EXTERN "
                               "(driver-provided)' " + warn) == 2);
        REQUIRE(count_of(text, absent_banner) == 2);
        // the parsed context itself is untouched (LammpsTadah copies it)
        REQUIRE_FALSE(pot_ctx.has_stored("EMBF0"));
    }
}

TEST_CASE("pair_coeff states RHOFORM, and a potential without it fails with "
          "the migration text",
          "[pair_tadah][ream][rhoform]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string banner =
        "Tadah! DM_REAM density form RHOFORM SIGNED (signed amplitudes, the "
        "density before tadah 1.6.0)\n";
    const std::string absent = "DM_REAM: RHOFORM is required: SIGNED";

    SECTION("pair_coeff * * <file>: the RHOFORM line next to EMBF0") {
        const std::string pot_path = write_pot(true, "ZERO");
        const RemoveOnExit rm_pot{pot_path};
        const std::string text = log_of_file_load(runner, pot_path, "rhoform");
        REQUIRE(count_of(text, banner) == 1);
        const auto e = text.find("Tadah! DM_REAM embedding reference EMBF0");
        REQUIRE(e != std::string::npos);
        REQUIRE(text.find(banner) > e);
    }

    SECTION("pair_coeff * * EXTERN: the RHOFORM line on every load") {
        const std::string pot_path = write_pot(true, "ZERO");
        const RemoveOnExit rm_pot{pot_path};
        tadah::core::Context pot_ctx;
        REQUIRE_NOTHROW(pot_ctx =
            tadah::core::CoreUserConfigParser().parse(pot_path));
        const std::string log = "ream_rhoform_" + std::to_string(::getpid())
            + "_extern.log";
        const RemoveOnExit rm_log{log};
        {
            auto lmp = runner.create_instance(
                0, {"lmp", "-log", log, "-screen", "none"});
            REQUIRE(lmp != nullptr);
            REQUIRE(runner.run_script(lmp, isolated_atom(), pot_ctx));
            REQUIRE(runner.run_script(lmp, isolated_atom(), pot_ctx));
        }
        const std::string text = slurp(log);
        REQUIRE(count_of(text, banner) == 2);
    }

    SECTION("pair_coeff * * <file> without RHOFORM: the reason is in the "
            "ERROR") {
        const std::string pot_path = write_pot(true, "ZERO", false);
        const RemoveOnExit rm_pot{pot_path};
        // log.lammps: the pair_coeff ERROR carries the load failure's text
        const std::string text =
            log_of_file_load(runner, pot_path, "no_rhoform", false);
        REQUIRE(count_of(text, "ERROR: pair_coeff: failed to load Tadah "
                               "potential from '" + pot_path + "': " +
                               absent) == 1);
        REQUIRE(count_of(text, "add 'RHOFORM SIGNED' to use an existing "
                               "potential unchanged") == 1);
        REQUIRE(count_of(text, "Tadah! DM_REAM density form") == 0);

        // the exception a library driver catches carries it as well
        const std::string script = "ream_rhoform_" +
            std::to_string(::getpid()) + "_no_rhoform.in";
        const RemoveOnExit rm_script{script};
        {
            std::ofstream ofs(script);
            for (const auto& line : isolated_atom())
                ofs << (line == "pair_coeff * * EXTERN Ta"
                            ? "pair_coeff * * " + pot_path + " Ta"
                            : line)
                    << "\n";
        }
        std::string what;
        {
            auto lmp = runner.create_instance(
                0, {"lmp", "-log", "none", "-screen", "none"});
            REQUIRE(lmp != nullptr);
            try {
                lmp->input->file(script.c_str());
            } catch (const std::exception& ex) {
                what = ex.what();
            }
        }
        REQUIRE(what.find("pair_coeff: failed to load Tadah potential from '"
                          + pot_path + "': " + absent) != std::string::npos);
    }

    SECTION("pair_coeff * * EXTERN after a failed driver load(): the reason "
            "is in the ERROR") {
        // The legacy per-line driver pattern: pair_style, then load() on the
        // pair style, then pair_coeff * * EXTERN with nothing staged.
        const std::string pot_path = write_pot(true, "ZERO", false);
        const RemoveOnExit rm_pot{pot_path};
        tadah::core::Context pot_ctx;
        REQUIRE_NOTHROW(pot_ctx =
            tadah::core::CoreUserConfigParser().parse(pot_path));
        LAMMPS_NS::PairTadah::clear_staged_extern();
        std::string what;
        {
            auto lmp = runner.create_instance(
                0, {"lmp", "-log", "none", "-screen", "none"});
            REQUIRE(lmp != nullptr);
            for (const auto& line : isolated_atom()) {
                if (line == "pair_coeff * * EXTERN Ta") break;
                lmp->input->one(line.c_str());
            }
            auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
            REQUIRE(pt != nullptr);
            REQUIRE_FALSE(pt->load(pot_ctx));
            REQUIRE(pt->last_load_error().find(absent) != std::string::npos);
            try {
                lmp->input->one("pair_coeff * * EXTERN Ta");
            } catch (const std::exception& ex) {
                what = ex.what();
            }
        }
        REQUIRE(what.find("pair_coeff * * EXTERN: no staged or pre-loaded "
                          "Tadah potential (last load() failed: " + absent)
                != std::string::npos);
    }
}

namespace {

/* Synthetic single-species Blip5 DM_REAM potential (Tadah.LAMMPS
 * tests_data/pots/ta_ream_blip5.tadah): every density amplitude > 0, SGRIDMB
 * > 0, WATOM 1 and Cut_Dummy, so it is valid under all three forms. */
const char* TA_REAM_BLIP5_BODY =
    "ATOM              Ta\n"
    "BIAS              false\n"
    "INIT2B            true\n"
    "INITMB            true\n"
    "MODEL             M_BLR   BF_Linear\n"
    "NORM              false\n"
    "TYPE2B            D2_Blip5   4   4   Ta   Ta\n"
    "CGRID2B           2.5   3.2   3.9   4.5\n"
    "SGRID2B           3.0   3.0   3.0   3.0\n"
    "TYPEMB            DM_REAM   3   3   3   3   5   5   0   0   Ta   Ta\n"
    "CGRIDMB           2.8   3.6   4.4\n"
    "SGRIDMB           3.0   3.0   3.0\n"
    "AMPGRIDMB         0.5   0.3   0.1\n"
    "CEMBFUNC          -0.2   0.3   0.8   1.3   1.8\n"
    "SEMBFUNC          2.0   2.0   2.0   2.0   2.0\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCTYPEMB          Cut_Dummy\n"
    "RCUT2B            5.5\n"
    "RCUTMB            5.5\n"
    "WATOM             1.0\n"
    "EMBF0             ZERO\n"
    "WEIGHTS           0.8   -0.3   0.1   -0.05   -2.0   1.5   -0.7   0.4   0.2\n";

/* 3x3x3 bcc Ta at 4.2 A (expanded, so the lattice density lies inside the
 * embedding bumps and the energy depends on the density form), rattled. */
std::vector<std::string> rattled_bcc_expanded() {
    std::vector<std::string> s = {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 4.2",
        "region box block 0 3 0 3 0 3 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 2.0 nsq",
        "displace_atoms all random 0.05 0.05 0.05 777 units box",
        "variable e equal pe",
        "variable fn equal fnorm",
        "variable p equal press",
        "run 0 post no",
    };
    return s;
}

struct FormRun {
    bool fast = false;
    double e = 0.0, fn = 0.0, p = 0.0;
    std::vector<std::string> exact;   // e, fnorm, press (lossless)
};

FormRun run_form(LammpsRunner& runner, const char* form, bool eamfast) {
    tadah::core::Context pot_ctx;
    {
        const std::string path = "ream_form_" + std::to_string(::getpid())
            + "_" + form + (eamfast ? "_on" : "_off") + ".tadah";
        const RemoveOnExit rm{path};
        {
            std::ofstream ofs(path);
            ofs << TA_REAM_BLIP5_BODY << "RHOFORM           " << form << "\n"
                << "EAMFAST           " << (eamfast ? "true" : "false") << "\n";
        }
        REQUIRE_NOTHROW(pot_ctx =
            tadah::core::CoreUserConfigParser().parse(path));
    }
    auto lmp = runner.create_instance(
        0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, rattled_bcc_expanded(), pot_ctx));
    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pt != nullptr);
    FormRun r;
    r.fast = pt->lammps_tadah()->eam_fast;
    for (const char* v : {"e", "fn", "p"})
        r.exact.push_back(runner.get_variable(lmp, v));
    r.e = std::stod(r.exact[0]);
    r.fn = std::stod(r.exact[1]);
    r.p = std::stod(r.exact[2]);
    return r;
}

}  // namespace

TEST_CASE("DM_REAM density forms through pair_style tadah: NONNEG equals "
          "SIGNED, MONOTONE EAMFAST matches the generic kernel",
          "[pair_tadah][ream][eamfast][rhoform]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const FormRun sf = run_form(runner, "SIGNED", true);
    const FormRun sg = run_form(runner, "SIGNED", false);
    const FormRun nf = run_form(runner, "NONNEG", true);
    const FormRun ng = run_form(runner, "NONNEG", false);
    const FormRun mf = run_form(runner, "MONOTONE", true);
    const FormRun mg = run_form(runner, "MONOTONE", false);

    REQUIRE(sf.fast);
    REQUIRE(nf.fast);
    REQUIRE(mf.fast);
    REQUIRE_FALSE(sg.fast);
    REQUIRE_FALSE(ng.fast);
    REQUIRE_FALSE(mg.fast);

    /* Every amplitude >= 0 and Cut_Dummy >= 0: the NONNEG density is the
     * SIGNED one, and both kernels run the same code on it. */
    REQUIRE(nf.exact == sf.exact);
    REQUIRE(ng.exact == sg.exact);

    /* MONOTONE is a different density (integrated-bump steps). */
    REQUIRE(std::fabs(mg.e - sg.e) > 1e-3 * std::fabs(sg.e));

    /* EAMFAST vs generic under every form (the pure-2B / EMBF0 tolerance for
     * the energy; forces and pressure carry the table's interpolation). */
    for (const auto* pr : {&sf, &mf}) {
        const FormRun& f = *pr;
        const FormRun& g = pr == &sf ? sg : mg;
        REQUIRE(f.e == Approx(g.e).epsilon(1e-9));
        REQUIRE(f.fn == Approx(g.fn).epsilon(1e-6));
        REQUIRE(f.p == Approx(g.p).epsilon(1e-6));
    }
}

TEST_CASE("pair_coeff: an element the potential does not have is a LAMMPS "
          "ERROR with the reason, in log.lammps",
          "[pair_tadah][pair_coeff]") {
    /* pair_coeff * * <file> W on a Ta potential: LammpsTadah::initialize()
     * throws "Symbol not in potential file: W". The exception left coeff()
     * outside error->all, so log.lammps recorded nothing and the reason
     * reached only a driver's catch (and stderr through LAMMPS' main). */
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string pot_path = write_pot(true, "ZERO");
    const RemoveOnExit rm_pot{pot_path};
    const std::string stem = "pair_coeff_symbol_" + std::to_string(::getpid());
    const std::string script = stem + ".in", log = stem + ".log";
    const RemoveOnExit rm_script{script}, rm_log{log};
    {
        std::ofstream ofs(script);
        for (const auto& line : isolated_atom())
            ofs << (line == "pair_coeff * * EXTERN Ta"
                        ? "pair_coeff * * " + pot_path + " W"
                        : line)
                << "\n";
    }
    const std::string reason = "pair_coeff: failed to initialise Tadah "
        "potential '" + pot_path + "': Symbol not in potential file: W";
    std::string what;
    {
        auto lmp = runner.create_instance(
            0, {"lmp", "-log", log, "-screen", "none"});
        REQUIRE(lmp != nullptr);
        try {
            lmp->input->file(script.c_str());
        } catch (const std::exception& ex) {
            what = ex.what();
        }
    }   // LAMMPS destroyed: log.lammps flushed and closed
    CHECK(what.find(reason) != std::string::npos);
    const std::string text = slurp(log);
    INFO(text);
    CHECK(count_of(text, "ERROR: " + reason) == 1);
}

TEST_CASE("PairTadah::load(Context) keeps a failure's reason for pair_coeff "
          "and prints nothing itself",
          "[pair_tadah][pair_coeff][rhoform]") {
    /* The reason reaches LAMMPS' error->all through pair_coeff * * EXTERN
     * (last_load_error()); the overloads also wrote it to stderr, so a
     * failed load was reported twice. */
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string pot_path = write_pot(true, "ZERO", false);
    const RemoveOnExit rm_pot{pot_path};
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx =
        tadah::core::CoreUserConfigParser().parse(pot_path));
    LAMMPS_NS::PairTadah::clear_staged_extern();
    auto lmp = runner.create_instance(
        0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    for (const auto& line : isolated_atom()) {
        if (line == "pair_coeff * * EXTERN Ta") break;
        lmp->input->one(line.c_str());
    }
    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pt != nullptr);
    std::ostringstream err;
    std::streambuf* old = std::cerr.rdbuf(err.rdbuf());
    bool ok_copy = true, ok_move = true;
    try {
        ok_copy = pt->load(pot_ctx);
        ok_move = pt->load(tadah::core::Context(pot_ctx));
    } catch (...) {
        std::cerr.rdbuf(old);
        throw;
    }
    std::cerr.rdbuf(old);
    CHECK_FALSE(ok_copy);
    CHECK_FALSE(ok_move);
    CHECK(pt->last_load_error().find("DM_REAM: RHOFORM is required")
          != std::string::npos);
    CHECK(err.str().find("PairTadah::load") == std::string::npos);
}
