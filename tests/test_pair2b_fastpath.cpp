// test_pair2b_fastpath.cpp
//
// EAMFAST pure-2B (phi-only) collapse: parity of the compute_2b_fast kernel
// against the generic compute_2b, driven end-to-end through LammpsRunner on
// a rattled bcc Kr cell (D2_Knot5, 21 knots, Cut_Dummy, linear, INITMB
// false -- the Kr HPO champion). Covers:
//   - static energy/force-norm/pressure parity (EAMFAST true vs false),
//   - box/relax trajectory equivalence (relaxed a0),
//   - adjoint capture under the fast path (pure-2B falls back to the
//     generic kernel, no error, snapshot filled).

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
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

namespace {

/* Kr pure-2B champion (ROUND6): physical scale, mechanically stable bcc. */
const char* KR_POT_BODY =
    "ATOM              Kr\n"
    "BIAS              false\n"
    "CGRID2B           5.654180679071199478 5.411252065569847503 3.217331521801811256 3.930824515846677247 2.537779659971619584 2.250748637725905610 3.737568302103229101 2.292155276555360999 4.166118987621620029 3.895659538414396827 3.882828370167631160 5.104012407403072160 1.892622625426633620 1.861528429044043031 6.197172052531381148 6.089092122840964905 6.744489232873557505 7.244744465733268690 7.327761008518089092 8.274486017059761167 8.116941074816644175\n"
    "EWEIGHT           1.0\n"
    "FORCE             false\n"
    "FWEIGHT           1.0\n"
    "INIT2B            true\n"
    "INITMB            false\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR   BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCUT2B            8.274486017059761167\n"
    "STRESS            false\n"
    "SWEIGHT           1.0\n"
    "TYPE2B            D2_Knot5   21   0   Kr   Kr\n"
    "WATOM             1.000000000000000000\n"
    "WEIGHTS           0.002022341035452596 -0.234834468777806615 -0.176955931611558204 -0.101273818977953803 -0.640091273014962159 -0.371627635837921522 -0.122730970028676681 -0.639591836790447621 0.889938055528681660 -0.501552854167752749 -0.306677826118867403 0.142464681149275907 -0.541937682987483349 -0.085416359679197792 0.085834571424044798 0.000135043203549349 0.000000000000000000 -0.011937057559490772 0.000000000000000000 0.000000000000000000 0.000000690371680869\n";

/* Pot files land in the shared ctest cwd, and each TEST_CASE runs as its
 * own concurrent process under `ctest -j` -- the name must be unique per
 * process or parallel cases clobber each other's pots mid-parse. */
std::string write_pot(bool eamfast) {
    const std::string path = "pair2b_" + std::to_string(::getpid())
                           + (eamfast ? "_on.tadah" : "_off.tadah");
    std::ofstream ofs(path);
    ofs << KR_POT_BODY;
    ofs << "EAMFAST           " << (eamfast ? "true" : "false") << "\n";
    return path;
}

/* 5x5x5 bcc Kr (250 atoms): box 20.5 A > 2*rcut = 16.55 A, then a
 * deterministic rattle so forces are non-zero. */
std::vector<std::string> rattled_bcc_kr(double amp) {
    return {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 4.1024",
        "region box block 0 5 0 5 0 5 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 83.798",
        "pair_style tadah",
        "pair_coeff * * EXTERN Kr",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        "displace_atoms all random " + std::to_string(amp) + " "
            + std::to_string(amp) + " " + std::to_string(amp)
            + " 777 units box",
        "variable e equal pe",
        "variable fn equal fnorm",
        "variable fm equal fmax",
        "variable p equal press",
        "run 0 post no",
    };
}

std::vector<std::string> boxrelax_bcc_kr() {
    return {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "lattice bcc 4.0",
        "region box block 0 5 0 5 0 5 units lattice",
        "create_box 1 box",
        "create_atoms 1 region box",
        "mass 1 83.798",
        "pair_style tadah",
        "pair_coeff * * EXTERN Kr",
        "neighbor 2.0 bin",
        "neigh_modify delay 0 every 1 check yes",
        "fix 1 all box/relax iso 0.0 vmax 0.001",
        "min_style cg",
        "min_modify dmax 0.01 line quadratic",
        "minimize 0 1e-10 2000 20000",
        "unfix 1",
        "variable a0 equal lx/5.0",
    };
}

double getd(LammpsRunner& runner, const LammpsRunner::LMPPtr& lmp,
            const char* name) {
    return std::stod(runner.get_variable(lmp, name));
}

}  // namespace

TEST_CASE("pair2b fast path: static + box/relax parity vs generic kernel",
          "[pair_tadah][pair2b][eamfast]") {
    /* Building the potential resolves "Kr"; required when this case runs
     * first in its process (ctest runs one case per process). */
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;

    struct Obs { double e, fn, fm, p, a0; bool fast; };
    Obs obs[2];

    for (int mode = 0; mode < 2; ++mode) {
        const bool eamfast = (mode == 0);
        const std::string pot_path = write_pot(eamfast);
        tadah::core::Context pot_ctx;
        REQUIRE_NOTHROW(pot_ctx =
            tadah::core::CoreUserConfigParser().parse(pot_path));

        auto lmp = runner.create_instance(
            0, {"lmp", "-log", "none", "-screen", "none"});
        REQUIRE(lmp != nullptr);

        REQUIRE(runner.run_script(lmp, rattled_bcc_kr(0.08), pot_ctx));

        auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
        REQUIRE(pt != nullptr);
        obs[mode].fast = pt->lammps_tadah()->eam_fast;

        obs[mode].e  = getd(runner, lmp, "e");
        obs[mode].fn = getd(runner, lmp, "fn");
        obs[mode].fm = getd(runner, lmp, "fm");
        obs[mode].p  = getd(runner, lmp, "p");

        REQUIRE(runner.run_script(lmp, boxrelax_bcc_kr(), pot_ctx));
        obs[mode].a0 = getd(runner, lmp, "a0");

        std::remove(pot_path.c_str());
    }

    /* The gate must actually differ between the two pots. */
    REQUIRE(obs[0].fast);
    REQUIRE_FALSE(obs[1].fast);

    /* Static parity: table is node-exact, mid-node Hermite error ~1e-9. */
    REQUIRE(obs[0].e  == Approx(obs[1].e).epsilon(1e-9));
    REQUIRE(obs[0].fn == Approx(obs[1].fn).epsilon(1e-7).margin(1e-8));
    REQUIRE(obs[0].fm == Approx(obs[1].fm).epsilon(1e-7).margin(1e-8));
    REQUIRE(obs[0].p  == Approx(obs[1].p).epsilon(1e-6).margin(1e-6));

    /* Same relaxed lattice constant through a full box/relax trajectory. */
    REQUIRE(obs[0].a0 == Approx(obs[1].a0).margin(1e-7));
}

TEST_CASE("pair2b fast path: adjoint capture falls back to the generic kernel",
          "[pair_tadah][pair2b][eamfast][adjoint]") {
    tadah::core::PeriodicTable::initialize();
    LammpsRunner runner;
    const std::string pot_path = write_pot(/*eamfast=*/true);
    tadah::core::Context pot_ctx;
    REQUIRE_NOTHROW(pot_ctx =
        tadah::core::CoreUserConfigParser().parse(pot_path));

    auto lmp = runner.create_instance(
        0, {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp != nullptr);
    REQUIRE(runner.run_script(lmp, rattled_bcc_kr(0.05), pot_ctx));

    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pt != nullptr);
    auto* lt = pt->lammps_tadah();
    REQUIRE(lt->eam_fast);

    /* Toggle capture and recompute: pure-2B must NOT error (the MB fast
     * path would) and must fill the snapshot via the generic kernel. */
    lt->capture_adjoint_state = true;
    lt->adjoint_snapshot.box_dofs = 1;
    REQUIRE_NOTHROW(lmp->force->pair->compute(1, 1));
    REQUIRE(lt->adjoint_snapshot.valid);
    REQUIRE(lt->adjoint_snapshot.n_desc == lt->S.descriptor_size());
    lt->capture_adjoint_state = false;

    std::remove(pot_path.c_str());
}
