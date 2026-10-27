// test_analytical_coverage.cpp
//
// Which search parameters of each descriptor the analytical gradient of
// tadah hpo differentiates, in the energy rows and in the force rows of the
// design matrix: the claims of the descriptor walker
// (mlip::assemble_dphi_train_both_rows), which the analytical gradient
// routes each parameter by (compute_analytical_gradient_from_trainer:
// a parameter without the claims of the rows the run has goes to finite
// differences) and by which GRAD_STRATEGY ANALYTICAL refuses one at
// startup. The table of the GRAD_STRATEGY documentation is this one.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/mlip/analytics/dphi_dtheta_walk.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/trainer.h>

#include <map>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace {

using tadah::core::Context;

/* A two-atom bcc Ta cell (a = 3.3 A), the second atom moved off its site,
 * with its neighbour list to 5 A: pairs at many distances inside the
 * cutoffs below. */
tadah::mlip::StructureDB ta_cell()
{
    using tadah::core::Element;
    using tadah::mlip::Atom;
    tadah::mlip::Structure st;
    st.cell.load(3.3, 0.0, 0.0, 0.0, 3.3, 0.0, 0.0, 0.0, 3.3);
    st.add_atom(Atom(Element("Ta"), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0));
    st.add_atom(Atom(Element("Ta"), 1.72, 1.61, 1.66, 0.0, 0.0, 0.0));
    tadah::mlip::NNFinder().calc(st, 5.0);
    tadah::mlip::StructureDB db;
    db.add(st);
    return db;
}

/* The claims of the descriptors of the training configuration `config`
 * for each search parameter of `keys` (index 1): "EF" for the energy and
 * the force rows, "E" for the energy rows only, "-" for none. */
std::map<std::string, std::string> claims(const std::string& config,
                                          const std::vector<std::string>& keys)
{
    std::istringstream in(config);
    Context c = tadah::core::CoreUserConfigParser().parse(in);
    tadah::mlip::memory::MLIPWorkspaceManager ws;
    tadah::mlip::Trainer trainer(c, ws);
    const auto db = ta_cell();
    const std::size_t n_cols = trainer.DCS.descriptor_size();
    std::map<std::string, std::string> out;
    for (const std::string& key : keys) {
        std::vector<double> energy(n_cols, 0.0);
        std::vector<double> force(3 * db(0).natoms() * n_cols, 0.0);
        const auto claimed = tadah::mlip::assemble_dphi_train_both_rows(
            db, trainer.DCS, key, 1, energy.data(), force.data(), n_cols,
            {0});
        out[key] = claimed.first ? (claimed.second ? "EF" : "E")
                                 : (claimed.second ? "F" : "-");
    }
    return out;
}

const std::string kModel =
    "ATOM Ta\nWATOM 1.0\nMODEL M_BLR BF_Linear\nVERBOSE 0\n";

/* A two-body descriptor `type` (TYPE2B) with its lines, alone. */
std::string two_body(const std::string& type, const std::string& lines)
{
    return kModel + "INIT2B true\nINITMB false\nTYPE2B " + type + "\n" +
           lines + "RCTYPE2B Cut_Cos\nRCUT2B 4.5\n";
}

/* A many-body descriptor `type` (TYPEMB) with its lines, alone. */
std::string many_body(const std::string& type, const std::string& lines)
{
    return kModel + "INIT2B false\nINITMB true\nTYPEMB " + type + "\n" +
           lines + "RCTYPEMB Cut_Cos\nRCUTMB 4.5\n";
}

}  // namespace

/* Pins: the search parameters each descriptor differentiates (index 1 of
 * each key, on a two-atom Ta cell, every pair inside the Cut_Cos cutoff of
 * 4.5 A): "EF" in the energy and the force rows, "E" in the energy rows
 * only, "-" in none.
 *   - D2_Blip, D2_Blip5, D2_BP: CGRID2B, SGRID2B, RCUT2B and WATOM, EF;
 *   - D2_Knot5: CGRID2B, RCUT2B and WATOM, EF (no SGRID2B);
 *   - D2_LJ, D2_MIE, D2_ZBL: RCUT2B and WATOM, EF;
 *   - DM_Blip, DM_EAD: CGRIDMB, SGRIDMB, RCUTMB and WATOM, E;
 *   - DM_mBlip: CGRIDMB, SGRIDMB, RCUTMB, WATOM, CEMBFUNC and SEMBFUNC, EF;
 *   - DM_mBlipEnv: those and RCUTENV, EF;
 *   - DM_mRLR, DM_mSQRT: CGRIDMB, SGRIDMB, RCUTMB and WATOM, E;
 *   - D2_mJoin: none;
 *   - a cutoff that does not depend on RCUT2B (Cut_Dummy): no RCUT2B.
 * No descriptor differentiates a key of the fit (LAMBDA) or another
 * descriptor's keys.
 * Before the fix: passes as well (a pin of the coverage the GRAD_STRATEGY
 * documentation gives, which startup applies to GRAD_STRATEGY ANALYTICAL). */
TEST_CASE("analytical gradient: the search parameters each descriptor "
          "differentiates in the energy and the force rows",
          "[HPO][analytical_gradient][coverage]")
{
    tadah::core::PeriodicTable::initialize();
    const std::vector<std::string> keys2 = {"CGRID2B", "SGRID2B", "RCUT2B",
                                            "WATOM", "LAMBDA", "CGRIDMB"};
    const std::vector<std::string> keysm = {
        "CGRIDMB", "SGRIDMB", "RCUTMB", "WATOM", "CEMBFUNC",
        "SEMBFUNC", "RCUTENV", "LAMBDA", "CGRID2B"};
    using Claims = std::map<std::string, std::string>;
    struct Case { std::string name, config; Claims expected; };
    const std::string grid2 = "CGRID2B 1.0 2.0\nSGRID2B 1.0 1.2\n";
    const Claims blip2 = {{"CGRID2B", "EF"}, {"SGRID2B", "EF"},
                          {"RCUT2B", "EF"},  {"WATOM", "EF"},
                          {"LAMBDA", "-"},   {"CGRIDMB", "-"}};
    const Claims knot = {{"CGRID2B", "EF"}, {"SGRID2B", "-"},
                         {"RCUT2B", "EF"},  {"WATOM", "EF"},
                         {"LAMBDA", "-"},   {"CGRIDMB", "-"}};
    const Claims pair = {{"CGRID2B", "-"}, {"SGRID2B", "-"},
                         {"RCUT2B", "EF"}, {"WATOM", "EF"},
                         {"LAMBDA", "-"},  {"CGRIDMB", "-"}};
    const Claims none2 = {{"CGRID2B", "-"}, {"SGRID2B", "-"},
                          {"RCUT2B", "-"},  {"WATOM", "-"},
                          {"LAMBDA", "-"},  {"CGRIDMB", "-"}};
    const std::string gridm = "CGRIDMB 2.0 3.0\nSGRIDMB 1.0 1.0\n";
    const std::string embm = "CEMBFUNC 1.0 2.0\nSEMBFUNC 1.0 1.0\n";
    const Claims energy_m = {{"CGRIDMB", "E"},   {"SGRIDMB", "E"},
                             {"RCUTMB", "E"},    {"WATOM", "E"},
                             {"CEMBFUNC", "-"},  {"SEMBFUNC", "-"},
                             {"RCUTENV", "-"},   {"LAMBDA", "-"},
                             {"CGRID2B", "-"}};
    const Claims mblip = {{"CGRIDMB", "EF"},  {"SGRIDMB", "EF"},
                          {"RCUTMB", "EF"},   {"WATOM", "EF"},
                          {"CEMBFUNC", "EF"}, {"SEMBFUNC", "EF"},
                          {"RCUTENV", "-"},   {"LAMBDA", "-"},
                          {"CGRID2B", "-"}};
    Claims menv = mblip;
    menv["RCUTENV"] = "EF";
    const std::vector<Case> cases = {
        {"D2_Blip", two_body("D2_Blip 2 2 Ta Ta", grid2), blip2},
        {"D2_Blip5", two_body("D2_Blip5 2 2 Ta Ta", grid2), blip2},
        {"D2_BP", two_body("D2_BP 2 2 Ta Ta", grid2), blip2},
        {"D2_Knot5", two_body("D2_Knot5 2 2 Ta Ta",
                              "CGRID2B 1.5 2.0\nKDEG2B 5 5\n"), knot},
        {"D2_LJ", two_body("D2_LJ Ta Ta", ""), pair},
        {"D2_MIE", two_body("D2_MIE 12 6 Ta Ta", ""), pair},
        {"D2_ZBL", two_body("D2_ZBL 0.5292 0.88534 0.23 0.23 Ta Ta", ""),
         pair},
        {"D2_mJoin",
         kModel + "INIT2B true\nINITMB false\nTYPE2B D2_mJoin\n"
                  "TYPE2B D2_BP 1 1 Ta Ta\nRCUT2B 4.5\nRCTYPE2B Cut_Cos\n"
                  "CGRID2B 1.0\nSGRID2B 1.0\n"
                  "TYPE2B D2_BP 1 1 Ta Ta\nRCUT2B 4.5\nRCTYPE2B Cut_Cos\n"
                  "CGRID2B 2.0\nSGRID2B 1.2\n",
         none2},
        {"DM_Blip", many_body("DM_Blip 0 2 2 Ta Ta", gridm), energy_m},
        {"DM_EAD", many_body("DM_EAD 0 2 2 Ta Ta", gridm), energy_m},
        {"DM_mBlip", many_body("DM_mBlip 0 1 1 2 2 Ta Ta",
                               "CGRIDMB 0.0\nSGRIDMB 0.2\n" + embm),
         mblip},
        {"DM_mBlipEnv",
         many_body("DM_mBlipEnv 0 1 1 2 2 Ta Ta",
                   "CGRIDMB 0.0\nSGRIDMB 0.2\nCEMBFUNC 1.0 2.0\n"
                   "SEMBFUNC 1.0 1.0\nRCUTENV 4.0\n"),
         menv},
        {"DM_mRLR", many_body("DM_mRLR 0 2 2 2 2 Ta Ta", gridm + embm),
         energy_m},
        {"DM_mSQRT", many_body("DM_mSQRT 0 2 2 2 2 Ta Ta", gridm + embm),
         energy_m},
    };
    for (const Case& c : cases) {
        INFO(c.name << "\n" << c.config);
        const std::vector<std::string>& keys =
            c.expected.count("RCUT2B") ? keys2 : keysm;
        const Claims got = claims(c.config, keys);
        for (const auto& [key, want] : c.expected) {
            INFO(key);
            CHECK(got.at(key) == want);
        }
    }
    /* RCUT2B enters through the cutoff function: Cut_Dummy does not
     * depend on it. */
    std::string dummy = two_body("D2_BP 2 2 Ta Ta", grid2);
    dummy.replace(dummy.find("Cut_Cos"), 7, "Cut_Dummy");
    INFO(dummy);
    CHECK(claims(dummy, keys2).at("RCUT2B") == "-");
}

namespace {

/* A two-body descriptor `type2` (TYPE2B) and a many-body descriptor
 * `typem` (TYPEMB), with their lines, in one configuration. */
std::string joined(const std::string& type2, const std::string& lines2,
                   const std::string& typem, const std::string& linesm)
{
    return kModel + "INIT2B true\nINITMB true\nTYPE2B " + type2 + "\n" +
           lines2 + "RCTYPE2B Cut_Cos\nRCUT2B 4.5\nTYPEMB " + typem + "\n" +
           linesm + "RCTYPEMB Cut_Cos\nRCUTMB 4.5\n";
}

}  // namespace

/* Pins: the search parameters a two-body and a many-body descriptor
 * differentiate together (index 1, on the two-atom Ta cell, Cut_Cos 4.5 A
 * for both): each key of one descriptor as that descriptor alone, and
 * WATOM, which both read, in the force rows only when both differentiate
 * it there:
 *   - D2_Blip and DM_Blip: CGRID2B, SGRID2B and RCUT2B, EF; CGRIDMB,
 *     SGRIDMB and RCUTMB, E; WATOM, E (DM_Blip has no force rows);
 *   - D2_Blip and DM_mBlip: CGRID2B, CGRIDMB, CEMBFUNC and WATOM, EF.
 * No descriptor differentiates LAMBDA.
 * Before the fix: D2_Blip and DM_Blip claimed WATOM for the force rows,
 * "EF": the claims of the two descriptors were joined by an OR, so the
 * force rows of D2_Blip hid the missing DM_Blip part, and the analytical
 * gradient used force rows without it. */
TEST_CASE("analytical gradient: the search parameters a two-body and a "
          "many-body descriptor differentiate together",
          "[HPO][analytical_gradient][coverage]")
{
    tadah::core::PeriodicTable::initialize();
    using Claims = std::map<std::string, std::string>;
    const std::string grid2 = "CGRID2B 1.0 2.0\nSGRID2B 1.0 1.2\n";
    struct Case { std::string name, config; Claims expected; };
    const std::vector<Case> cases = {
        {"D2_Blip and DM_Blip",
         joined("D2_Blip 2 2 Ta Ta", grid2, "DM_Blip 0 2 2 Ta Ta",
                "CGRIDMB 2.0 3.0\nSGRIDMB 1.0 1.0\n"),
         {{"CGRID2B", "EF"}, {"SGRID2B", "EF"}, {"RCUT2B", "EF"},
          {"CGRIDMB", "E"},  {"SGRIDMB", "E"},  {"RCUTMB", "E"},
          {"WATOM", "E"},    {"LAMBDA", "-"}}},
        {"D2_Blip and DM_mBlip",
         joined("D2_Blip 2 2 Ta Ta", grid2, "DM_mBlip 0 1 1 2 2 Ta Ta",
                "CGRIDMB 0.0\nSGRIDMB 0.2\nCEMBFUNC 1.0 2.0\n"
                "SEMBFUNC 1.0 1.0\n"),
         {{"CGRID2B", "EF"}, {"CGRIDMB", "EF"}, {"CEMBFUNC", "EF"},
          {"WATOM", "EF"},   {"LAMBDA", "-"}}},
    };
    for (const Case& c : cases) {
        INFO(c.name << "\n" << c.config);
        std::vector<std::string> keys;
        for (const auto& kv : c.expected) keys.push_back(kv.first);
        const Claims got = claims(c.config, keys);
        for (const auto& [key, want] : c.expected) {
            INFO(key);
            CHECK(got.at(key) == want);
        }
    }
}
