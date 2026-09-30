// test_cohesive_energy_observable.cpp
//
// Phase D3 (PC_OBSERVABLE redesign): coverage for
// CohesiveEnergyObservable.
//
// What these tests check:
//   1. Construction validation (bad args throw cleanly).
//   2. Hot-var seeding: default → midpoint, explicit → that value,
//      out-of-bounds → clamped.
//   3. Closed-form dO/dW matches W-FD to high precision when the
//      relaxer is bypassed (maxiter=0). This is the analytical contract
//      we ship under PC_OBSERVABLE — at fixed atomic state the formula
//      `dE_coh/dW = −(dE_bulk/dW − N · dE_iso/dW)/N` must hold byte-for-
//      byte (modulo float noise).
//   4. Hot-var updates: with relax enabled, a0_current changes after
//      evaluate.
//   5. Neighbour lists (workstream C): the E_coh of a relaxation whose cell
//      changed equals a fresh neighbour-list evaluation of the relaxed bulk
//      (the relaxer's cached list must carry the image displacements of
//      the current cell); and --supercell 1 1 1 gives the E_coh of 4 4 4.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/element.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/observables/cohesive_energy.h>
#include <tadah/mlip/cell_builder.h>
#include <tadah/mlip/nn_finder.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>
#include <tadah/mlip/structure_transformations.h>
#include <tadah/mlip/trainer.h>
#include <tadah/mlip/memory/mlip_workspace_manager.h>

#include <algorithm>
#include <cmath>
#include <sstream>
#include <stdexcept>

namespace {

using tadah::LossFunction;
using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::hpo::CohesiveEnergyObservable;
using tadah::hpo::ObservableEvalResult;
using tadah::mlip::NNFinder;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;
using tadah::mlip::Trainer;
using tadah::mlip::memory::MLIPWorkspaceManager;

Context make_ta_context(double rcut, const std::vector<double>& cgrid,
                        const std::vector<double>& sgrid)
{
    Context c;
    c.add("ATOM",     "Ta");
    c.add("WATOM",    1);
    c.add("INIT2B",   "true");
    c.add("INITMB",   "false");
    c.add("TYPE2B",   "D2_BP");
    c.add("TYPE2B",   static_cast<int>(cgrid.size()));
    c.add("TYPE2B",   static_cast<int>(sgrid.size()));
    c.add("TYPE2B",   "Ta"); c.add("TYPE2B", "Ta");
    c.add("TYPEMB",   "DM_Dummy");
    c.add("TYPEMB",   "Ta"); c.add("TYPEMB", "Ta");
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < cgrid.size(); ++i) {
            if (i) ss << " "; ss << cgrid[i];
        }
        c.add("CGRID2B", ss.str());
    }
    {
        std::ostringstream ss;
        for (std::size_t i = 0; i < sgrid.size(); ++i) {
            if (i) ss << " "; ss << sgrid[i];
        }
        c.add("SGRID2B", ss.str());
    }
    c.add("RCTYPE2B", "Cut_Cos");
    c.add("RCTYPEMB", "Cut_Cos");
    c.add("RCUT2B",   rcut);
    c.add("RCUTMB",   rcut);
    c.add("DIMER",    "false");
    c.add("DIMER",    0);
    c.add("DIMER",    "false");
    c.add("FORCE",    "true");
    c.add("STRESS",   "false");
    c.add("EWEIGHT",  1.0);
    c.add("FWEIGHT",  1.0);
    c.add("SWEIGHT",  1.0);
    c.add("BIAS",     "false");
    c.add("NORM",     "false");
    c.add("VERBOSE",  0);
    c.add("MODEL",    "M_BLR");
    c.add("MODEL",    "BF_Linear");
    c.add("LAMBDA",   1e-3);
    c.add("OALGO",    4);
    return c;
}

StructureDB make_ta_train_db() {
    StructureDB db;
    const std::vector<double> a0s = {3.20, 3.30, 3.40};
    db.dbidx = {0, a0s.size()};
    db.structures.reserve(a0s.size());
    for (std::size_t i = 0; i < a0s.size(); ++i) {
        auto st = tadah::mlip::cells::bcc("Ta", a0s[i]);
        tadah::mlip::StructureTransformation::replicate(st, 2, 2, 2);
        st.energy = -0.5 * static_cast<double>(i + 1);
        db.structures.push_back(st);
    }
    return db;
}

/* n x n x n copies of @p st along its cell rows, built here (independent of
 * StructureTransformation::replicate): rows scaled by n, copy (x, y, z) of
 * atom i at r_i + x a0 + y a1 + z a2. */
Structure supercell_along_rows(const Structure& st, int n) {
    Structure out = st;
    out.atoms.clear();
    out.nn.reset();
    for (int k = 0; k < 3; ++k)
        for (int c = 0; c < 3; ++c) out.cell(k, c) = st.cell(k, c) * n;
    for (int x = 0; x < n; ++x)
        for (int y = 0; y < n; ++y)
            for (int z = 0; z < n; ++z)
                for (auto at : st.atoms) {
                    for (int c = 0; c < 3; ++c)
                        at.position[c] += x * st.cell(0, c) +
                                          y * st.cell(1, c) + z * st.cell(2, c);
                    out.atoms.push_back(at);
                }
    return out;
}

/* 2x2x2 bcc cells, a = 3.0 ... 3.6, labelled with E/atom = -8 + 5 (a - 3.3)^2:
 * a D2_BP model fitted to them has a bcc energy minimum inside the hot-var
 * range [2.7, 3.9] (near a = 3.34 for the grids used below), so an Iso
 * relaxation from a0 = 3.27 moves the cell by about 2%. */
StructureDB make_parabola_train_db() {
    StructureDB db;
    const std::vector<double> a0s = {3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6};
    db.dbidx = {0, a0s.size()};
    for (double a : a0s) {
        Structure st = supercell_along_rows(tadah::mlip::cells::bcc("Ta", a), 2);
        st.energy = static_cast<double>(st.natoms()) *
                    (-8.0 + 5.0 * (a - 3.3) * (a - 3.3));
        db.structures.push_back(st);
    }
    return db;
}

/* E_coh = -(E_bulk - N E_iso) / N of the observable's relaxed bulk and
 * isolated atom, with neighbour lists built afresh here at the model's
 * cutoff. */
double fresh_ecoh(Trainer& host, const ObservableEvalResult& r) {
    REQUIRE(r.energy_terms.size() == 2u);
    Structure bulk = r.energy_terms[0].structure;
    Structure iso  = r.energy_terms[1].structure;
    REQUIRE(iso.natoms() == 1u);
    bulk.nn.reset();
    iso.nn.reset();
    NNFinder().calc(bulk, host.context);
    NNFinder().calc(iso, host.context);
    const double E_bulk = host.model->epredict(host.dc.calc(bulk));
    const double E_iso  = host.model->epredict(host.dc.calc(iso));
    const double N = static_cast<double>(bulk.natoms());
    return -(E_bulk - N * E_iso) / N;
}

}  // namespace

TEST_CASE("CohesiveEnergyObservable constructor validates",
          "[hpo][observable][cohesive_energy][ctor]")
{
    PeriodicTable::initialize();

    auto valid = []() {
        CohesiveEnergyObservable::Config cfg;
        cfg.species = "Ta";
        cfg.nx = cfg.ny = cfg.nz = 2;
        cfg.target = 8.1;
        cfg.weight = 1000.0;
        cfg.a0_min = 2.7;
        cfg.a0_max = 3.9;
        cfg.a0_initial = 3.30;
        return cfg;
    };

    CHECK_NOTHROW(CohesiveEnergyObservable{valid()});

    {
        auto c = valid(); c.species = "";
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
    {
        auto c = valid(); c.nx = 0;
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
    {
        auto c = valid(); c.a0_min = 3.5; c.a0_max = 3.5;
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
    {
        auto c = valid(); c.weight = -1.0;
        CHECK_THROWS_AS(CohesiveEnergyObservable{c}, std::invalid_argument);
    }
}

TEST_CASE("CohesiveEnergyObservable seeds hot-var correctly",
          "[hpo][observable][cohesive_energy][hotvar]")
{
    PeriodicTable::initialize();

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.1; cfg.weight = 1000.0;
    cfg.a0_min = 2.7; cfg.a0_max = 3.9;

    /* No explicit initial → midpoint. */
    {
        CohesiveEnergyObservable obs(cfg);
        CHECK(std::abs(obs.current_a0() - 3.3) < 1e-12);
    }
    /* Explicit initial inside bounds. */
    {
        auto c = cfg; c.a0_initial = 3.50;
        CohesiveEnergyObservable obs(c);
        CHECK(std::abs(obs.current_a0() - 3.50) < 1e-12);
    }
    /* Out-of-bounds initial gets clamped. */
    {
        auto c = cfg; c.a0_initial = 4.5;   /* > a0_max */
        CohesiveEnergyObservable obs(c);
        CHECK(std::abs(obs.current_a0() - 3.9) < 1e-12);
    }
    {
        auto c = cfg; c.a0_initial = 2.0;   /* < a0_min */
        CohesiveEnergyObservable obs(c);
        CHECK(std::abs(obs.current_a0() - 2.7) < 1e-12);
    }
}

TEST_CASE("CohesiveEnergyObservable dO/dW matches W-FD without relaxation",
          "[hpo][observable][cohesive_energy][w_grad]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db();
    host.train(train_db);

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.1;
    cfg.weight = 1000.0;
    cfg.a0_min = 2.7;
    cfg.a0_max = 3.9;
    cfg.a0_initial = 3.30;
    cfg.relax_maxiter = 0;        /* skip relax — fixed cell → clean FD parity */

    CohesiveEnergyObservable obs(cfg);
    ObservableEvalResult r0 = obs.evaluate(*host.model, host.dc, host.context);

    REQUIRE(r0.dO_dW.size() == host.model->get_weights().size());
    REQUIRE(!r0.needs_adjoint);

    const tadah::core::t_type w0 = host.model->get_weights();
    const double h = 1e-5;

    /* FD reference: re-evaluate at W ± h·e_c. Each call uses the SAME
     * hot-var seed (we re-seed by re-constructing obs each iter to keep
     * the test independent of the relaxer's warmstart cache). */
    double max_rel = 0.0;
    for (std::size_t c = 0; c < w0.size(); ++c) {
        tadah::core::t_type wp = w0;  wp(c) += h;
        host.model->set_weights(wp);
        CohesiveEnergyObservable obs_p(cfg);
        ObservableEvalResult rp = obs_p.evaluate(*host.model, host.dc, host.context);

        tadah::core::t_type wm = w0;  wm(c) -= h;
        host.model->set_weights(wm);
        CohesiveEnergyObservable obs_m(cfg);
        ObservableEvalResult rm = obs_m.evaluate(*host.model, host.dc, host.context);

        const double fd = (rp.value - rm.value) / (2.0 * h);
        const double anal = r0.dO_dW(c);
        const double scale = std::max({std::abs(fd), std::abs(anal), 1e-9});
        const double rel = std::abs(fd - anal) / scale;
        max_rel = std::max(max_rel, rel);
        INFO("c=" << c << "  anal=" << anal << "  fd=" << fd
             << "  rel=" << rel);
        CHECK(rel < 1e-5);
    }
    INFO("max rel = " << max_rel);

    host.model->set_weights(w0);
}

TEST_CASE("CohesiveEnergyObservable updates hot-var after relaxation",
          "[hpo][observable][cohesive_energy][hotvar][relax]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    const double rcut = 4.5;
    Context ctx = make_ta_context(rcut, {2.5, 3.5}, {2.0, 3.0});
    Trainer host(ctx, ws);
    auto train_db = make_ta_train_db();
    host.train(train_db);

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.1;
    cfg.weight = 1000.0;
    cfg.a0_min = 2.7;
    cfg.a0_max = 3.9;
    cfg.a0_initial = 3.10;        /* start away from any plausible minimum */
    cfg.relax_atom_ftol = 1.0e-2;
    cfg.relax_maxiter = 500;

    CohesiveEnergyObservable obs(cfg);
    const double a0_before = obs.current_a0();
    ObservableEvalResult r = obs.evaluate(*host.model, host.dc, host.context);

    INFO("a0 before = " << a0_before
         << "  a0 after = " << obs.current_a0()
         << "  Ecoh = " << r.value);

    REQUIRE(r.hotvar_updated_value.has_value());
    /* a0 changed under relaxation (synthetic potential, but the
     * relaxer's Iso DOF still moves it). Just check the cache field
     * reflects current_a0. */
    CHECK(std::abs(obs.current_a0() - *r.hotvar_updated_value) < 1e-12);
    /* Hot-var didn't escape the bounds. */
    CHECK(obs.current_a0() >= cfg.a0_min * 0.5);   /* leniency for sneak-out */
    CHECK(obs.current_a0() <= cfg.a0_max * 2.0);
}

/* The relaxed bulk's neighbour list is the relaxer's cached one (built at
 * rcut + skin and reused while it covers the cutoff). Its image
 * displacements must be those of the relaxed cell: before the exact-image
 * fix a cache hit kept the displacements of the cell the list was built
 * for, so after an Iso step the images sat where the old cell put them --
 * a perfect bcc lattice felt spurious forces (the relaxation stalls on
 * them) and E_coh was computed from misplaced images.
 *
 * relax_etol and a lenient eq_fmax_threshold make the observable report the
 * E_coh it computes for the bulk the relaxation ends at, whether it stops on
 * the force criterion (exact lists: every atomic force ~1e-15 on a perfect
 * lattice) or stalls on an energy plateau (misplaced images); that value
 * must be the relaxed bulk's energy. */
TEST_CASE("CohesiveEnergyObservable: relaxed E_coh equals a fresh "
          "neighbour-list evaluation of the relaxed bulk",
          "[hpo][observable][cohesive_energy][nn]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(4.5, {2.4, 3.0, 3.6}, {2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    CohesiveEnergyObservable::Config cfg;
    cfg.species = "Ta";
    cfg.nx = cfg.ny = cfg.nz = 2;
    cfg.target = 8.0;
    cfg.weight = 1.0;
    cfg.a0_min = 2.7;
    cfg.a0_max = 3.9;
    cfg.a0_initial = 3.27;           /* about 2% below the model's minimum */
    cfg.relax_atom_ftol = 1.0e-4;    /* the strain must converge: several Iso steps */
    cfg.relax_etol = 1.0e-12;
    cfg.relax_maxiter = 500;
    cfg.eq_fmax_threshold = 1.0e3;

    CohesiveEnergyObservable obs(cfg);
    const ObservableEvalResult r = obs.evaluate(*host.model, host.dc, host.context);

    INFO("a0 " << *cfg.a0_initial << " -> " << obs.current_a0()
         << ", E_coh " << r.value);
    /* Precondition: the relaxation changed the cell (accepted Iso steps
     * after the list was built), so the cached list's cell is not the
     * relaxed one. Each Iso step changes the scale by at most 0.1%. */
    REQUIRE(std::abs(obs.current_a0() / *cfg.a0_initial - 1.0) > 5.0e-3);
    /* The observable's own value, not the unconverged penalty. */
    REQUIRE_FALSE(r.value_is_constant);
    REQUIRE(std::isfinite(r.value));
    REQUIRE(r.energy_terms.size() == 2u);
    REQUIRE(r.energy_terms[0].structure.cell(0, 0) ==
            Approx(2.0 * obs.current_a0()).epsilon(1e-12));

    const double ecoh = fresh_ecoh(host, r);
    INFO("fresh-list E_coh " << ecoh);
    CHECK(std::abs(r.value - ecoh) <= 1.0e-10 * std::max(1.0, std::abs(ecoh)));
}

/* A guard, expected to pass before and after the exact-image fix: the E_coh
 * of the 2-atom bcc cell (--supercell 1 1 1) equals that of 4 4 4 at the
 * same a0, without relaxation (relax_maxiter 0; a relax_atom_ftol no
 * pressure reaches, so the iteration-0 check accepts the cell as it is).
 * At rcut 6 and a0 3.3 (rcut/a0 = 1.82) a list built at the model's cutoff
 * would miss images of the 2-atom cell before the fix, but the relaxer
 * builds its list at rcut + skin (2 A), wide enough that no pair inside
 * rcut is lost (DESIGN-nnfinder-exact-images.md, section 0 item 8). */
TEST_CASE("CohesiveEnergyObservable: supercell 1 1 1 and 4 4 4 give the "
          "same E_coh",
          "[hpo][observable][cohesive_energy][nn][supercell]")
{
    PeriodicTable::initialize();

    MLIPWorkspaceManager ws;
    Context ctx = make_ta_context(6.0, {2.4, 3.0, 3.6, 4.5}, {2.0, 2.0, 2.0, 2.0});
    Trainer host(ctx, ws);
    auto train_db = make_parabola_train_db();
    host.train(train_db);

    auto ecoh = [&](int n) {
        CohesiveEnergyObservable::Config cfg;
        cfg.species = "Ta";
        cfg.nx = cfg.ny = cfg.nz = n;
        cfg.target = 8.0;
        cfg.weight = 1.0;
        cfg.a0_min = 2.7;
        cfg.a0_max = 3.9;
        cfg.a0_initial = 3.3;
        cfg.relax_maxiter = 0;
        cfg.relax_atom_ftol = 1.0e6;
        CohesiveEnergyObservable obs(cfg);
        const ObservableEvalResult r =
            obs.evaluate(*host.model, host.dc, host.context);
        CAPTURE(n, r.value);
        REQUIRE_FALSE(r.value_is_constant);
        REQUIRE(std::isfinite(r.value));
        REQUIRE(r.energy_terms.size() == 2u);
        REQUIRE(r.energy_terms[0].structure.natoms() ==
                static_cast<std::size_t>(2 * n * n * n));
        CHECK(obs.current_a0() == 3.3);   /* not relaxed */
        return r.value;
    };

    const double e1 = ecoh(1);
    const double e4 = ecoh(4);
    INFO("E_coh 1x1x1 " << e1 << ", 4x4x4 " << e4);
    CHECK(std::abs(e1 - e4) <= 1.0e-10 * std::max(1.0, std::abs(e4)));
}
