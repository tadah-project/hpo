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
//   - EAMFAST vs generic parity (relative 1e-9, as for pure 2B);
//   - a potential without an EMBF0 line is evaluated as ZERO;
//   - pair_coeff prints a LAMMPS WARNING (screen + log.lammps) for such a
//     potential on every load, and nothing new for a declared one (BUG-08).

#include "catch2/catch.hpp"

#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <force.h>
#include <pair.h>
#include <pair_tadah.h>

#include <cmath>
#include <cstdio>
#include <limits>
#include <fstream>
#include <iterator>
#include <string>
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
    "STRESS            false\n"
    "TYPE2B            D2_Knot5   13   0   Ta   Ta\n"
    "TYPEMB            DM_REAM   2   8   0   8   15   0   0   0   Ta   Ta\n"
    "WATOM             1.000000000000000000\n"
    "WEIGHTS           -0.014641348094898586   -0.051535959944426050   -41.156393764756018072   3.808410339336090633   0.188032523218672837   -3058.915888454833293508   -3.511803288240705534   36.979581171997139677   -0.549561123341020474   -0.334973899538970332   2.145274625352853182   0.142458455160975661   35.268028553999059227   222.913408871893182095   223.783694639092914258   -168.806095957493624837   90.719128161416477951   5304.663560913277251530   78.628875194741951304   7.352619952010690518   29.817892005298819669   5136.174514981846186856   -298.436161734381755650   -23120.745531707521877252   -51.786960138537409648   -121.723616579840154373   -25539.485778202957590111   38206.820714959903853014\n";

/* Unique per process (ctest runs cases as concurrent processes). */
std::string write_pot(bool eamfast, const char* embf0) {
    const std::string path = "ream_fast_" + std::to_string(::getpid()) + "_"
        + (eamfast ? "on" : "off") + "_" + (embf0 ? embf0 : "absent")
        + ".tadah";
    std::ofstream ofs(path);
    ofs << TA_REAM_POT_BODY;
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

/* log.lammps of one isolated-atom run that loads @p pot_path by file name
 * (pair_coeff * * <file> Ta, the plain LAMMPS user path). */
std::string log_of_file_load(LammpsRunner& runner, const std::string& pot_path,
                             const std::string& tag) {
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
        REQUIRE(runner.run_script(lmp, script));
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

    /* EAMFAST vs generic (same tolerance as the pure-2B collapse). */
    REQUIRE(zf.e == Approx(zg.e).epsilon(1e-9));

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
