// test_pair_tadah_coincident.cpp
//
// pair_style tadah refuses two atoms at one position (report_leftovers.md
// section 3; MK's decision: option A in NNFinder plus the same check in
// pair_tadah). LAMMPS builds its own neighbour lists, so NNFinder never sees
// a LAMMPS run. Every pair_tadah kernel divides the pair force by the
// distance, so a pair at distance 0 gave NaN forces and pressure while `run
// 0` completed (and, under MD, atoms with NaN positions that dropped out of
// the interactions; report section 3.3). The check raises error->one naming
// the LAMMPS atom IDs of the two atoms; in this in-process instance it
// arrives as a LAMMPSAbortException, a std::exception whose message the
// cases read. A pair 1e-12 A apart is not refused.
//
// The kernels compute() dispatches to, one potential each (the first four
// as in test_pair2b_fastpath.cpp, which shows that they take these paths):
//   compute_2b                  D2_LJ, M_KRR Kern_Linear, EAMFAST false
//   compute_2b_fast             the same, EAMFAST true
//   compute_2b_mb_half          D2_LJ + DM_REAM, M_BLR BF_Linear, EAMFAST false
//   compute_2b_mb_half_eamfast  the same, EAMFAST true
//   compute_2b_mb_full          D2_BP, M_BLR BF_Polynomial2 (not linear)
// Each case checks the LammpsTadah flags that compute() dispatches on, so a
// potential that took another kernel fails at that REQUIRE, not at the
// refusal. compute_dimers (DIMER true, rigid molecules) is not covered.
//
// The cell: a 3.3 A cube with atom 1 at (0, 0, 0), atom 2 at its centre and
// atom 3 at (dx, 0, 0), created in that order (atom IDs 1, 2, 3).

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <atom.h>
#include <force.h>
#include <input.h>
#include <pair.h>
#include <pair_tadah.h>

#include <exception>
#include <iomanip>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using tadah::core::Context;
using tadah::core::PeriodicTable;
using LMPPtr = LammpsRunner::LMPPtr;

/* The docs quickstart potential (as TA_LJ_QUICKSTART_BODY in
 * test_pair2b_fastpath.cpp): D2_LJ Ta, Cut_Dummy, RCUT2B 6.5, M_KRR
 * Kern_Linear, INITMB false. Linear and two-body only: compute_2b, or
 * compute_2b_fast with EAMFAST true. */
const char* const kLjBody =
    "ATOM     Ta\n"
    "BIAS     false\n"
    "DIMER     false     0     false\n"
    "EWEIGHT     1.0\n"
    "FWEIGHT     1.0\n"
    "INIT2B     true\n"
    "INITMB     false\n"
    "LAMBDA     0\n"
    "MODEL     M_KRR     Kern_Linear     1\n"
    "NORM     false\n"
    "OALGO     1\n"
    "RCTYPE2B     Cut_Dummy\n"
    "RCUT2B     6.5\n"
    "SWEIGHT     1.0\n"
    "TYPE2B     D2_LJ     Ta     Ta\n"
    "UNCERTAINTY     false\n"
    "WATOM     73\n"
    "WEIGHTS     14.204213349318765225     2604.396965191186154698\n";

/* D2_LJ + DM_REAM, linear (as TA_LJ_REAM_BODY in test_pair2b_fastpath.cpp):
 * compute_2b_mb_half, or compute_2b_mb_half_eamfast with EAMFAST true. */
const char* const kLjReamBody =
    "ATOM              Ta\n"
    "AMPGRIDMB         0.40 0.25 0.12 0.05\n"
    "BIAS              false\n"
    "CEMBFUNC          0.0 0.4 0.8 1.2 1.6\n"
    "CGRIDMB           1.0 2.0 3.0 4.0\n"
    "DIMER             false 0 false\n"
    "EWEIGHT           1.0\n"
    "FWEIGHT           1.0\n"
    "INIT2B            true\n"
    "INITMB            true\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR BF_Linear\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Dummy\n"
    "RCTYPEMB          Cut_Cos\n"
    "RCUT2B            6.5\n"
    "RCUTMB            6.0\n"
    "RHOFORM           SIGNED\n"
    "SEMBFUNC          1.5 1.5 1.5 1.5 1.5\n"
    "SGRIDMB           1.0 1.0 1.0 1.0\n"
    "SWEIGHT           1.0\n"
    "TYPE2B            D2_LJ Ta Ta\n"
    "TYPEMB            DM_REAM 0 4 4 4 5 5 0 0 Ta Ta\n"
    "WATOM             1\n"
    "WEIGHTS           1036.907574500269861425 190120.978458956589292954 "
    "-1.2 -0.9 -0.6 -0.3 0.1\n";

/* Two Gaussians (D2_BP, Cut_Cos 5.5) under a quadratic basis
 * (BF_Polynomial2: weights for x1 x1, x1 x2, x2 x2): not linear, so
 * compute_2b_mb_full. */
const char* const kBpPoly2Body =
    "ATOM              Ta\n"
    "BIAS              false\n"
    "CGRID2B           2.6 3.4\n"
    "DIMER             false 0 false\n"
    "EWEIGHT           1.0\n"
    "FWEIGHT           1.0\n"
    "INIT2B            true\n"
    "INITMB            false\n"
    "LAMBDA            0\n"
    "MODEL             M_BLR BF_Polynomial2\n"
    "NORM              false\n"
    "RCTYPE2B          Cut_Cos\n"
    "RCUT2B            5.5\n"
    "SGRID2B           1.0 1.0\n"
    "SWEIGHT           1.0\n"
    "TYPE2B            D2_BP 2 2 Ta Ta\n"
    "WATOM             1\n"
    "WEIGHTS           0.5 -0.25 0.125\n";

/* A potential and the kernel compute() takes for it, by the LammpsTadah
 * flags it dispatches on (dimer false in every case). */
struct Kernel {
    const char* name;
    const char* body;
    bool eamfast;   // the EAMFAST line appended to the body
    bool fast;      // LammpsTadah::eam_fast after pair_coeff
    bool linear;
    bool initmb;
};

const Kernel kKernels[] = {
    {"compute_2b", kLjBody, false, false, true, false},
    {"compute_2b_fast", kLjBody, true, true, true, false},
    {"compute_2b_mb_half", kLjReamBody, false, false, true, true},
    {"compute_2b_mb_half_eamfast", kLjReamBody, true, true, true, true},
    {"compute_2b_mb_full", kBpPoly2Body, false, false, false, false},
};

Context potential(const Kernel& k) {
    std::istringstream in(std::string(k.body) + "EAMFAST           " +
                          (k.eamfast ? "true" : "false") + "\n");
    return tadah::core::CoreUserConfigParser().parse(in);
}

std::string num(double v) {
    std::ostringstream o;
    o << std::setprecision(17) << v;
    return o.str();
}

/* The cell up to pair_coeff (no run): atoms 1 and 3 are dx apart. */
std::vector<std::string> setup_script(double dx) {
    return {
        "clear",
        "units metal",
        "boundary p p p",
        "atom_style atomic",
        "atom_modify map array",
        "region box block 0 3.3 0 3.3 0 3.3 units box",
        "create_box 1 box",
        "create_atoms 1 single 0 0 0 units box",
        "create_atoms 1 single 1.65 1.65 1.65 units box",
        "create_atoms 1 single " + num(dx) + " 0 0 units box",
        "mass 1 180.94788",
        "pair_style tadah",
        "pair_coeff * * EXTERN Ta",
        "neighbor 1.0 nsq",
    };
}

LMPPtr make_instance() {
    return LammpsRunner().create_instance(
        0, {"lmp", "-log", "none", "-screen", "none"});
}

/* Stages @p pot for `pair_coeff * * EXTERN` (PairTadah::stage_extern, as
 * LammpsRunner::run_script does) while it lives. */
struct StagedPotential {
    explicit StagedPotential(const Context& pot) {
        LAMMPS_NS::PairTadah::stage_extern(
            std::make_shared<tadah::LammpsTadah>(pot));
    }
    ~StagedPotential() { LAMMPS_NS::PairTadah::clear_staged_extern(); }
    StagedPotential(const StagedPotential&) = delete;
    StagedPotential& operator=(const StagedPotential&) = delete;
};

/* Executes @p lines one at a time with @p pot staged: "" when every line
 * ran, else the failing line and the message of its exception. */
std::string execute(const LMPPtr& lmp, const std::vector<std::string>& lines,
                    const Context& pot) {
    StagedPotential staged(pot);
    for (const std::string& line : lines) {
        try {
            lmp->input->one(line);
        } catch (const std::exception& e) {
            return "'" + line + "': " + e.what();
        }
    }
    return "";
}

/* The message of the exception `run 0 post no` throws, "" when it
 * completes. */
std::string run0_error(const LMPPtr& lmp) {
    try {
        lmp->input->one("run 0 post no");
    } catch (const std::exception& e) {
        return e.what();
    }
    return "";
}

/* The potential of the instance takes kernel @p k (see compute()). */
void require_kernel(const LMPPtr& lmp, const Kernel& k) {
    auto* pt = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);
    REQUIRE(pt != nullptr);
    const tadah::LammpsTadah* lt = pt->lammps_tadah();
    REQUIRE(lt != nullptr);
    CAPTURE(k.name, lt->dimer, lt->eam_fast, lt->linear, lt->initmb);
    REQUIRE_FALSE(lt->dimer);
    REQUIRE(lt->eam_fast == k.fast);
    REQUIRE(lt->linear == k.linear);
    REQUIRE(lt->initmb == k.initmb);
}

bool contains(const std::string& s, const std::string& part) {
    return s.find(part) != std::string::npos;
}

}  // namespace

/* Pins: in each pair_tadah kernel, `run 0` on the 3.3 A Ta cube with atoms 1
 * and 3 both at (0, 0, 0) throws, and the message names the LAMMPS atom IDs,
 * the smaller first ("atom IDs 1 and 3"), and says the atoms are at the same
 * position. With atom 3 at (1e-12, 0, 0) instead, `run 0` completes: a close
 * pair is not refused.
 * Before the fix: `run 0` completed with NaN forces in every kernel (the
 * pair force divided by r = 0), so err was empty and both message CHECKs
 * failed in each section; the 1e-12 runs passed. */
TEST_CASE("pair_style tadah: run 0 refuses two atoms at the same position in "
          "every kernel, naming the atom IDs",
          "[pair_tadah][coincident]") {
    PeriodicTable::initialize();
    for (const Kernel& k : kKernels) {
        DYNAMIC_SECTION(k.name) {
            const Context pot = potential(k);
            {
                auto lmp = make_instance();
                REQUIRE(lmp != nullptr);
                const std::string setup = execute(lmp, setup_script(1e-12), pot);
                INFO("setup: " << setup);
                REQUIRE(setup.empty());
                require_kernel(lmp, k);
                const std::string err = run0_error(lmp);
                INFO("run 0 with atoms 1 and 3 1e-12 A apart: " << err);
                CHECK(err.empty());
            }
            {
                auto lmp = make_instance();
                REQUIRE(lmp != nullptr);
                const std::string setup = execute(lmp, setup_script(0.0), pot);
                INFO("setup: " << setup);
                REQUIRE(setup.empty());
                require_kernel(lmp, k);
                const std::string err = run0_error(lmp);
                INFO("run 0 with atoms 1 and 3 at the same position: " << err);
                CHECK(contains(err, "atom IDs 1 and 3"));
                CHECK(contains(err, "same position"));
            }
        }
    }
}

/* Pins: PairTadah::single(), which `compute pair/local` calls for each pair
 * (pair_tadah enables it for a linear two-body potential), refuses a pair at
 * distance 0 through error->one naming the atom IDs of the two atoms it is
 * given ("atom IDs 1 and 3", "same position"), and evaluates a pair 1e-12 A
 * apart. The pair is set up on the separated cell (atom 3 at (0.9, 0, 0),
 * after a `run 0` that completes) and single() is called on atoms 1 and 3 by
 * their local indices (Atom::map).
 * Before the fix: single() divided the pair force by r = 0 and returned, so
 * nothing was thrown and both message CHECKs failed. */
TEST_CASE("pair_style tadah: single() refuses a pair at distance 0, naming "
          "the atom IDs",
          "[pair_tadah][coincident]") {
    PeriodicTable::initialize();
    const Kernel& k = kKernels[0];   // linear, two-body: single() enabled
    auto lmp = make_instance();
    REQUIRE(lmp != nullptr);
    const std::string setup = execute(lmp, setup_script(0.9), potential(k));
    INFO("setup: " << setup);
    REQUIRE(setup.empty());
    require_kernel(lmp, k);
    const std::string run = run0_error(lmp);
    INFO("run 0: " << run);
    REQUIRE(run.empty());

    LAMMPS_NS::Pair* pair = lmp->force->pair;
    REQUIRE(pair->single_enable == 1);
    const int i = lmp->atom->map(1);
    const int j = lmp->atom->map(3);
    REQUIRE(i >= 0);
    REQUIRE(j >= 0);
    double fforce = 0.0;
    CHECK_NOTHROW(pair->single(i, j, 1, 1, 1e-24, 1.0, 1.0, fforce));

    std::string msg;
    try {
        pair->single(i, j, 1, 1, 0.0, 1.0, 1.0, fforce);
    } catch (const std::exception& e) {
        msg = e.what();
    }
    INFO("single() at r = 0: " << msg);
    CHECK(contains(msg, "atom IDs 1 and 3"));
    CHECK(contains(msg, "same position"));
}
