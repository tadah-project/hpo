// test_val_filter_forwarding.cpp
//
// Regression coverage for the rule "EFILTER / FFILTER on the --validation
// file are honoured; the same keys on the master (training) context are
// not". Exercises the small forward_val_filters helper directly, then
// pipes the resulting stripped context through apply_load_transforms to
// verify the end-to-end behaviour the HPO host depends on.
//
// We do NOT spin up a full HPO_Host here: that would require a model and
// real DBFILEs. The forwarding helper is the only new logic on the
// validation path, and apply_load_transforms is already covered by
// MLIP/tests/test_load_transforms.cpp; combining them gives full coverage
// of the user-visible contract.

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/val_filter_forwarding.h>
#include <tadah/mlip/atom.h>
#include <tadah/mlip/load_transforms.h>
#include <tadah/mlip/structure.h>
#include <tadah/mlip/structure_db.h>

using tadah::core::Context;
using tadah::core::PeriodicTable;
using tadah::core::Vec3d;
using tadah::mlip::apply_load_transforms;
using tadah::mlip::Atom;
using tadah::mlip::Structure;
using tadah::mlip::StructureDB;

namespace {

Structure make_structure(const char* label, double E,
                         std::vector<std::pair<std::string, Vec3d>> atoms,
                         double a = 5.0)
{
    Structure st;
    st.label = label;
    st.energy = E;
    st.cell.set_zero();
    st.cell(0, 0) = a;
    st.cell(1, 1) = a;
    st.cell(2, 2) = a;
    st.stress.set_zero();
    for (const auto& p : atoms) {
        Atom a_obj(PeriodicTable::find_by_symbol(p.first),
                   p.second[0], p.second[1], p.second[2]);
        st.add_atom(a_obj);
    }
    return st;
}

}  // namespace

TEST_CASE("forward_val_filters copies EFILTER from val_ctx",
          "[HPO][val_filter]")
{
    Context val_ctx;
    val_ctx.add("EFILTER", "-12.0");
    val_ctx.add("EFILTER", "-2.0");

    Context vt;
    tadah::hpo::detail::forward_val_filters(vt, val_ctx);

    REQUIRE(vt.contains("EFILTER"));
    REQUIRE(vt.size("EFILTER") == 2);
    CHECK(vt.get<double>("EFILTER", 0) == Approx(-12.0));
    CHECK(vt.get<double>("EFILTER", 1) == Approx(-2.0));
}

TEST_CASE("forward_val_filters copies FFILTER from val_ctx",
          "[HPO][val_filter]")
{
    Context val_ctx;
    val_ctx.add("FFILTER", "20.0");

    Context vt;
    tadah::hpo::detail::forward_val_filters(vt, val_ctx);

    REQUIRE(vt.contains("FFILTER"));
    CHECK(vt.get<double>("FFILTER") == Approx(20.0));
}

TEST_CASE("forward_val_filters does not invent filters when val_ctx has none",
          "[HPO][val_filter]")
{
    Context val_ctx;  // empty: caller may have set EFILTER on master_ctx_,
                      // but the helper must not see it.

    Context vt;
    tadah::hpo::detail::forward_val_filters(vt, val_ctx);

    CHECK_FALSE(vt.contains("EFILTER"));
    CHECK_FALSE(vt.contains("FFILTER"));
}

TEST_CASE("forward_val_filters skips malformed EFILTER (size != 2)",
          "[HPO][val_filter]")
{
    Context val_ctx;
    val_ctx.add("EFILTER", "-2.0");  // only one bound — invalid

    Context vt;
    tadah::hpo::detail::forward_val_filters(vt, val_ctx);

    CHECK_FALSE(vt.contains("EFILTER"));
}

TEST_CASE("Validation pipeline: filter from val_ctx drops configs; "
          "ESHIFT from training propagates",
          "[HPO][val_filter][integration]")
{
    // Build a "validation" StructureDB with three Ti configs:
    //   - good_a:  E/N = -3.0  (-3.0 / 1)
    //   - good_b:  E/N = -2.5  (-10.0 / 4)
    //   - hot:     E/N = +5.0  (+5.0 / 1) — should be dropped by EFILTER
    StructureDB val;
    val.add(make_structure("good_a", -3.0, {{"Ti", Vec3d(0, 0, 0)}}));
    val.add(make_structure(
        "good_b", -10.0,
        {{"Ti", Vec3d(0, 0, 0)},
         {"Ti", Vec3d(1, 0, 0)},
         {"Ti", Vec3d(0, 1, 0)},
         {"Ti", Vec3d(0, 0, 1)}}));
    val.add(make_structure("hot", +5.0, {{"Ti", Vec3d(0, 0, 0)}}));
    val.dbidx = {0, val.size()};

    // The HPO host would have populated this from the parsed --validation
    // file. The user put EFILTER there directly.
    Context val_ctx;
    val_ctx.add("EFILTER", "-12.0");
    val_ctx.add("EFILTER", "-2.0");

    // Mirror what HPO_Host builds: LSCALE+resolved ESHIFT (from training
    // report), then forward filters from val_ctx.
    Context vt;
    vt.add("LSCALE", "1.0");
    vt.add("ESHIFT", "-3.0");  // resolved per-Ti baseline from training
    tadah::hpo::detail::forward_val_filters(vt, val_ctx);

    const std::size_t n_before = val.size();
    auto report = apply_load_transforms(val, vt);

    // EFILTER ran first against ABSOLUTE energies, before ESHIFT.
    // hot's E/N = +5.0 lies above emax=-2.0, so it must be dropped.
    CHECK(report.n_dropped_energy == 1);
    CHECK(report.n_dropped_force == 0);
    CHECK(n_before - val.size() == 1);
    REQUIRE(val.size() == 2);

    // ESHIFT then runs on the survivors.
    // good_a: -3.0 - 1*(-3.0) = 0
    // good_b: -10.0 - 4*(-3.0) = +2
    CHECK(val(0).energy == Approx(0.0));
    CHECK(val(1).energy == Approx(2.0));
}
