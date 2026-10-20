// test_fit_switch_guard.cpp
//
// detail::ScopedFitSwitches and detail::validation_switches()
// (include/tadah/hpo/fit_switch_guard.h; branch fix/pc-frmse-force-off).
//
// HPO validation predicts what its metrics measure: forces whenever
// PC_FRMSE is set, stresses whenever PC_SRMSE_GPA is set, and only then,
// whatever FORCE/STRESS training fitted. validation_switches(trg) is that
// rule, {trg.use_frmse, trg.use_srmse}. ScopedFitSwitches sets FORCE and/or
// STRESS on a context for one scope (a switch given as nullopt is left
// alone) and restores the EXACT prior state when the scope ends: a key
// that was not stored is not stored afterwards, a stored value comes back
// verbatim -- also when an exception leaves the scope, and nested guards
// unwind in order.
//
// "Exact prior state" is checked on the whole context: snapshot() is
// operator<<, which writes every stored key with its tokens verbatim, so a
// key the guard added, dropped or respelled shows up.
//
// Before the fix the guard read the prior state with Context::contains(),
// which inserts the blueprint default of a missing key (FORCE and STRESS
// default to "false"): a key that was absent came back stored "false", and
// a potential written from such a context gained explicit "FORCE false" /
// "STRESS false" lines.
//
// Fast unit tests, no data, no host:
//   ./test_hpo "[fit_switches][unit]"

#include "catch2/catch.hpp"

#include <tadah/core/context.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/fit_switch_guard.h>
#include <tadah/hpo/hpo_targets.h>

#include <unistd.h>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>

namespace {

namespace fs = std::filesystem;
using tadah::core::Context;
using tadah::hpo::detail::ScopedFitSwitches;

/* What value_of() reports for a key the context does not store. */
const std::string kNotStored = "<not stored>";

/* The token @p key stores in @p c, or kNotStored; never inserts a
 * default (Context::stored_value). */
std::string value_of(const Context& c, const char* key) {
    return c.stored_value(key).value_or(kNotStored);
}

std::string stored(const char* text) { return text; }

/* Every stored key with its tokens, verbatim (Context's operator<<). */
std::string snapshot(const Context& c) {
    std::ostringstream os;
    os << c;
    return os.str();
}

/* A context with a few ordinary keys and neither FORCE nor STRESS stored. */
Context context_without_switches() {
    Context c;
    c.add("INIT2B", "true");
    c.add("RCUT2B", 4.5);
    c.add("MODEL", "M_BLR");
    c.add("MODEL", "BF_Linear");
    REQUIRE_FALSE(c.has_stored("FORCE"));
    REQUIRE_FALSE(c.has_stored("STRESS"));
    return c;
}

/* The same context with FORCE = @p force and STRESS = @p stress stored. */
Context context_with_switches(const char* force, const char* stress) {
    Context c = context_without_switches();
    c.add("FORCE", force);
    c.add("STRESS", stress);
    REQUIRE(value_of(c, "FORCE") == stored(force));
    REQUIRE(value_of(c, "STRESS") == stored(stress));
    return c;
}

/* A temporary directory for hpotarget files, removed with its contents. */
struct TempDir {
    fs::path dir;
    TempDir() {
        dir = fs::temp_directory_path() /
              ("tadah_fit_switch_guard_" + std::to_string(::getpid()) + "_" +
               std::to_string(std::chrono::steady_clock::now()
                                  .time_since_epoch()
                                  .count()));
        fs::create_directories(dir);
    }
    ~TempDir() {
        std::error_code ec;
        fs::remove_all(dir, ec);
    }
};

/* The -c context HPOTargets is parsed against (the minimal one of
 * test_hpo_targets_order.cpp), with the TRAINING switches FORCE/STRESS =
 * @p fit: validation_switches() must not follow them. */
Context targets_context(bool fit) {
    Context c;
    c.add("ATOM",    "Ta");
    c.add("WATOM",   1);
    c.add("INIT2B",  "false");
    c.add("INITMB",  "false");
    c.add("BIAS",    "false");
    c.add("NORM",    "false");
    c.add("MODEL",   "M_BLR");
    c.add("MODEL",   "BF_Linear");
    c.add("LAMBDA",  1e-3);
    c.add("OALGO",   4);
    c.add("FORCE",   fit ? "true" : "false");
    c.add("STRESS",  fit ? "true" : "false");
    return c;
}

}  // namespace

/* Pins: a guard on a context that stores neither FORCE nor STRESS sets the
 * switches it is given for the scope, and afterwards the context stores
 * neither again -- the whole context is as before.
 * Before the fix: after the scope FORCE (and STRESS, when given) is stored
 * "false" -- Context::contains() inserted the blueprint default before the
 * prior state was saved -- so has_stored() is true and the snapshot gains
 * "FORCE false" / "STRESS false" lines. */
TEST_CASE("ScopedFitSwitches: FORCE and STRESS that were not stored are not "
          "stored after the scope", "[fit_switches][unit]")
{
    Context c = context_without_switches();
    const std::string before = snapshot(c);

    std::optional<bool> force, stress;
    SECTION("FORCE true, STRESS true") {
        force = true;
        stress = true;
    }
    SECTION("FORCE false, STRESS false") {
        force = false;
        stress = false;
    }
    SECTION("FORCE true only") {
        force = true;
    }
    SECTION("STRESS false only") {
        stress = false;
    }
    CAPTURE(force.has_value(), stress.has_value());

    {
        ScopedFitSwitches g(c, force, stress);
        if (force)
            CHECK(value_of(c, "FORCE") == stored(*force ? "true" : "false"));
        else
            CHECK_FALSE(c.has_stored("FORCE"));
        if (stress)
            CHECK(value_of(c, "STRESS") ==
                  stored(*stress ? "true" : "false"));
        else
            CHECK_FALSE(c.has_stored("STRESS"));
    }

    CHECK_FALSE(c.has_stored("FORCE"));
    CHECK_FALSE(c.has_stored("STRESS"));
    CHECK(value_of(c, "FORCE") == kNotStored);
    CHECK(value_of(c, "STRESS") == kNotStored);
    CHECK(snapshot(c) == before);
}

/* Pins: a stored FORCE/STRESS comes back verbatim (one token, the same
 * text) after a guard that set the opposite or the same value, and the
 * rest of the context is untouched.
 * Holds before the fix. Fails if the destructor only removes the key, or
 * puts back the blueprint default ("false") or the guard's own value
 * instead of the saved one. */
TEST_CASE("ScopedFitSwitches: a stored FORCE or STRESS comes back verbatim",
          "[fit_switches][unit]")
{
    const char* f0 = "true";
    const char* s0 = "false";
    bool f = false, s = true;
    SECTION("stored FORCE true / STRESS false, set to the opposite") {
        f0 = "true"; s0 = "false"; f = false; s = true;
    }
    SECTION("stored FORCE false / STRESS true, set to the opposite") {
        f0 = "false"; s0 = "true"; f = true; s = false;
    }
    SECTION("stored FORCE true / STRESS true, set to false") {
        f0 = "true"; s0 = "true"; f = false; s = false;
    }
    SECTION("stored FORCE true / STRESS true, set to the same values") {
        f0 = "true"; s0 = "true"; f = true; s = true;
    }
    CAPTURE(f0, s0, f, s);

    Context c = context_with_switches(f0, s0);
    const std::string before = snapshot(c);
    {
        ScopedFitSwitches g(c, f, s);
        CHECK(value_of(c, "FORCE") == stored(f ? "true" : "false"));
        CHECK(value_of(c, "STRESS") == stored(s ? "true" : "false"));
        CHECK(c.size("FORCE") == 1u);
        CHECK(c.size("STRESS") == 1u);
    }
    CHECK(value_of(c, "FORCE") == stored(f0));
    CHECK(value_of(c, "STRESS") == stored(s0));
    CHECK(c.size("FORCE") == 1u);
    CHECK(c.size("STRESS") == 1u);
    CHECK(snapshot(c) == before);
}

/* Pins: a scope left by an exception restores the prior state like a
 * normal exit -- absent keys absent, stored keys verbatim -- and the
 * exception passes through unchanged.
 * Before the fix: the "not stored" SECTION fails (FORCE and STRESS come
 * back stored "false", as in the first case). The "stored" SECTION holds
 * before the fix; it fails if the restore is done anywhere but in the
 * destructor (for example after the guarded call returns). */
TEST_CASE("ScopedFitSwitches: a scope left by an exception restores the "
          "prior state", "[fit_switches][unit]")
{
    Context c;
    bool was_stored = false;
    SECTION("FORCE and STRESS not stored") {
        c = context_without_switches();
    }
    SECTION("FORCE false and STRESS true stored") {
        c = context_with_switches("false", "true");
        was_stored = true;
    }
    CAPTURE(was_stored);
    const std::string before = snapshot(c);

    std::string inside_force, inside_stress, caught = "<nothing caught>";
    try {
        ScopedFitSwitches g(c, true, false);
        inside_force  = value_of(c, "FORCE");
        inside_stress = value_of(c, "STRESS");
        throw std::runtime_error("validation prediction failed");
    } catch (const std::runtime_error& e) {
        caught = e.what();
    }
    CHECK(caught == stored("validation prediction failed"));
    CHECK(inside_force == stored("true"));
    CHECK(inside_stress == stored("false"));

    if (was_stored) {
        CHECK(value_of(c, "FORCE") == stored("false"));
        CHECK(value_of(c, "STRESS") == stored("true"));
    } else {
        CHECK_FALSE(c.has_stored("FORCE"));
        CHECK_FALSE(c.has_stored("STRESS"));
    }
    CHECK(snapshot(c) == before);
}

/* Pins: two nested guards (outer FORCE true; inner FORCE false and STRESS
 * true) restore in order: after the inner scope FORCE is the outer's
 * "true" and STRESS is as before both guards; after the outer scope the
 * context is as before both.
 * Before the fix: the "not stored" SECTION fails -- after the inner scope
 * STRESS is stored "false" instead of absent, and after the outer scope
 * FORCE and STRESS are stored "false". The "stored" SECTION holds before
 * the fix; it fails if a guard restores a value other than the one it
 * saved (the outer's "true" after the inner scope). */
TEST_CASE("ScopedFitSwitches: nested guards restore in order",
          "[fit_switches][unit]")
{
    Context c;
    bool was_stored = false;
    SECTION("FORCE and STRESS not stored") {
        c = context_without_switches();
    }
    SECTION("FORCE false and STRESS false stored") {
        c = context_with_switches("false", "false");
        was_stored = true;
    }
    CAPTURE(was_stored);
    const std::string before = snapshot(c);
    const std::string stress0 = value_of(c, "STRESS");

    {
        ScopedFitSwitches outer(c, true, std::nullopt);
        CHECK(value_of(c, "FORCE") == stored("true"));
        CHECK(value_of(c, "STRESS") == stress0);
        {
            ScopedFitSwitches inner(c, false, true);
            CHECK(value_of(c, "FORCE") == stored("false"));
            CHECK(value_of(c, "STRESS") == stored("true"));
        }
        CHECK(value_of(c, "FORCE") == stored("true"));
        CHECK(value_of(c, "STRESS") == stress0);
        CHECK(c.has_stored("STRESS") == was_stored);
    }
    CHECK(c.has_stored("FORCE") == was_stored);
    CHECK(c.has_stored("STRESS") == was_stored);
    CHECK(snapshot(c) == before);
}

/* Pins: a switch given as nullopt is not touched -- neither inside the
 * scope nor after it, absent or stored -- and a guard with both nullopt
 * leaves the whole context as it is.
 * Holds before the fix. Fails if the guard sets a nullopt switch (for
 * example to value_or(false)) or reads it with contains()/get<bool>(),
 * which inserts the blueprint default: the absent key then becomes stored
 * inside the scope. */
TEST_CASE("ScopedFitSwitches: a switch given as nullopt is left alone",
          "[fit_switches][unit]")
{
    SECTION("both nullopt, nothing stored") {
        Context c = context_without_switches();
        const std::string before = snapshot(c);
        {
            ScopedFitSwitches g(c, std::nullopt, std::nullopt);
            CHECK_FALSE(c.has_stored("FORCE"));
            CHECK_FALSE(c.has_stored("STRESS"));
            CHECK(snapshot(c) == before);
        }
        CHECK(snapshot(c) == before);
    }
    SECTION("both nullopt, both stored") {
        Context c = context_with_switches("true", "false");
        const std::string before = snapshot(c);
        {
            ScopedFitSwitches g(c, std::nullopt, std::nullopt);
            CHECK(snapshot(c) == before);
        }
        CHECK(snapshot(c) == before);
    }
    SECTION("FORCE nullopt and not stored, STRESS set") {
        Context c = context_without_switches();
        {
            ScopedFitSwitches g(c, std::nullopt, true);
            CHECK_FALSE(c.has_stored("FORCE"));
            CHECK(value_of(c, "STRESS") == stored("true"));
        }
        CHECK_FALSE(c.has_stored("FORCE"));
    }
    SECTION("STRESS nullopt and not stored, FORCE set") {
        Context c = context_without_switches();
        {
            ScopedFitSwitches g(c, false, std::nullopt);
            CHECK(value_of(c, "FORCE") == stored("false"));
            CHECK_FALSE(c.has_stored("STRESS"));
        }
        CHECK_FALSE(c.has_stored("STRESS"));
    }
    SECTION("FORCE nullopt and stored true, STRESS set") {
        Context c = context_with_switches("true", "true");
        {
            ScopedFitSwitches g(c, std::nullopt, false);
            CHECK(value_of(c, "FORCE") == stored("true"));
            CHECK(value_of(c, "STRESS") == stored("false"));
        }
        CHECK(value_of(c, "FORCE") == stored("true"));
        CHECK(value_of(c, "STRESS") == stored("true"));
    }
}

/* Pins: validation_switches(trg) is {use_frmse, use_srmse} for all four
 * metric combinations, parsed from an hpotarget (PC_ERMSE always, PC_FRMSE
 * and/or PC_SRMSE_GPA), and the training FORCE/STRESS of the -c context (both
 * true, both false) do not enter it: a force and stress fit validated on
 * PC_ERMSE alone predicts neither forces nor stresses.
 * Holds before the fix (validation_switches() already exists). Fails if
 * the rule also follows the training switches (for example {use_frmse ||
 * FORCE, use_srmse || STRESS}: the PC_ERMSE-only and single-metric rows
 * with FORCE/STRESS true), or mixes up the two flags. */
TEST_CASE("validation_switches: FORCE and STRESS follow PC_FRMSE and "
          "PC_SRMSE_GPA, not the training switches", "[fit_switches][unit]")
{
    tadah::core::PeriodicTable::initialize();
    TempDir td;
    int n = 0;
    for (const bool fit : {true, false})
        for (const bool use_f : {false, true})
            for (const bool use_s : {false, true}) {
                CAPTURE(fit, use_f, use_s);
                const fs::path cfg =
                    td.dir / ("config_" + std::to_string(++n) + ".hpo");
                {
                    std::ofstream os(cfg);
                    os << "OPTIMIZER\nLIB NLOPT\nALGO LN_SBPLX\nMAXEVAL 1\n"
                          "ENDOPTIMIZER\n"
                          "LOSS L2\nFAILSCORE 1e6\n"
                          "OPTIM WATOM (1) 0.1 5.0\n"
                          "PC_ERMSE 0 1\n";
                    if (use_f) os << "PC_FRMSE 0 1\n";
                    if (use_s) os << "PC_SRMSE_GPA 0 1\n";
                    REQUIRE(os.good());
                }
                Context ctx = targets_context(fit);
                tadah::hpo::HPOTargets trg(ctx, cfg.string());
                REQUIRE(trg.use_frmse == use_f);
                REQUIRE(trg.use_srmse == use_s);

                const tadah::hpo::detail::FitSwitches sw =
                    tadah::hpo::detail::validation_switches(trg);
                CHECK(sw.force == use_f);
                CHECK(sw.stress == use_s);
            }
}
