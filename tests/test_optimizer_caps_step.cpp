/* OptimizerCaps STEP policy (StepUse) -- table-driven registry facts.
 * The registry is the single source of truth for which backends honour
 * an OPTIMIZER-block STEP and how; scale_validation.cpp and the journal
 * derive their behaviour from it, so these facts are pinned here. */
#include "catch2/catch.hpp"
#include <tadah/hpo/optimizer_caps.h>

#include <string>
#include <vector>

using namespace tadah::hpo;

TEST_CASE("OptimizerCaps: StepUse classification per backend",
          "[HPO][caps][step]")
{
    struct Row { const char* lib; const char* algo; StepUse use; };
    const std::vector<Row> rows = {
        {"NLOPT", "LN_SBPLX",        StepUse::PerDimVector},
        {"NLOPT", "LN_NELDERMEAD",   StepUse::PerDimVector},
        {"NLOPT", "LN_BOBYQA",       StepUse::PerDimRescale},
        {"NLOPT", "LN_COBYLA",       StepUse::PerDimRescale},
        {"NLOPT", "LN_NEWUOA",       StepUse::MinCollapsed},
        {"NLOPT", "LN_NEWUOA_BOUND", StepUse::MinCollapsed},
        {"NLOPT", "LN_PRAXIS",       StepUse::MinCollapsed},
        {"NLOPT", "LD_MMA",          StepUse::TrustHalfWidth},
        {"NLOPT", "LD_CCSAQ",        StepUse::TrustHalfWidth},
        {"NLOPT", "G_MLSL_LDS",      StepUse::ForwardedToInner},
        {"NLOPT", "G_MLSL",          StepUse::ForwardedToInner},
        {"NLOPT", "LD_AUGLAG",       StepUse::ForwardedToInner},
        {"NLOPT", "GN_CRS2_LM",      StepUse::Ignored},
        {"NLOPT", "GN_ISRES",        StepUse::Ignored},
        {"NLOPT", "GN_ESCH",         StepUse::Ignored},
        {"NLOPT", "GN_DIRECT",       StepUse::Ignored},
        {"NLOPT", "GD_STOGO",        StepUse::Ignored},
        {"NLOPT", "LD_LBFGS",        StepUse::Ignored},
        {"NLOPT", "LD_SLSQP",        StepUse::Ignored},
        {"NLOPT", "LD_TNEWTON",      StepUse::Ignored},
        {"TADAH", "GRID",            StepUse::GridCount},
        {"TADAH", "RANDOM",          StepUse::Rejected},
        {"TADAH", "ANNEAL",          StepUse::WidthFraction},
        {"TADAH", "ANNEAL_PT",       StepUse::WidthFraction},
        {"TADAH", "ANNEAL_PA",       StepUse::WidthFraction},
        {"ASA",   "ASA",             StepUse::Rejected},
        {"DLIB",  "GFS",             StepUse::Rejected},
        {"DLIB",  "BFGS",            StepUse::Rejected},
        {"CERES", "LBFGS",           StepUse::Rejected},
        {"CERES", "STEEPEST_DESCENT",StepUse::Rejected},
    };
    for (const auto& r : rows) {
        INFO(r.lib << ":" << r.algo);
        const auto& c = optimizer_caps(r.lib, r.algo);
        CHECK(c.step_use == r.use);
        CHECK(std::string(c.step_note).size() > 10);
    }
}

TEST_CASE("OptimizerCaps: step_use_consumes and names",
          "[HPO][caps][step]")
{
    CHECK(step_use_consumes(StepUse::PerDimVector));
    CHECK(step_use_consumes(StepUse::PerDimRescale));
    CHECK(step_use_consumes(StepUse::MinCollapsed));
    CHECK(step_use_consumes(StepUse::TrustHalfWidth));
    CHECK(step_use_consumes(StepUse::ForwardedToInner));
    CHECK_FALSE(step_use_consumes(StepUse::Rejected));
    CHECK_FALSE(step_use_consumes(StepUse::Ignored));
    CHECK_FALSE(step_use_consumes(StepUse::GridCount));
    CHECK_FALSE(step_use_consumes(StepUse::WidthFraction));
    CHECK(std::string(step_use_name(StepUse::PerDimRescale)).find("rescaling")
          != std::string::npos);
    CHECK(std::string(step_use_name(StepUse::GridCount)).find("GRID_POINTS")
          != std::string::npos);
}

TEST_CASE("OptimizerCaps: every registered backend carries a STEP note",
          "[HPO][caps][step]")
{
    auto all = filter_optimizer_caps([](const OptimizerCaps&) { return true; });
    REQUIRE(all.size() > 30);
    for (const auto* c : all) {
        INFO(c->lib << ":" << c->algo);
        CHECK(c->step_note != nullptr);
        CHECK(std::string(c->step_note).size() > 0);
        if (c->ignores_bounds) CHECK(std::string(c->bounds_note).size() > 0);
    }
}

TEST_CASE("OptimizerCaps: bounds-ignoring backends are flagged",
          "[HPO][caps][bounds]")
{
    CHECK(optimizer_caps("NLOPT", "LN_NEWUOA").ignores_bounds);
    CHECK_FALSE(optimizer_caps("NLOPT", "LN_NEWUOA_BOUND").ignores_bounds);
    CHECK_FALSE(optimizer_caps("NLOPT", "LN_BOBYQA").ignores_bounds);
    CHECK(optimizer_caps("CERES", "LBFGS").ignores_bounds);
    CHECK(optimizer_caps("CERES", "NLCG").ignores_bounds);
    CHECK_FALSE(optimizer_caps("DLIB", "BFGS").ignores_bounds);
    CHECK_FALSE(optimizer_caps("TADAH", "ANNEAL").ignores_bounds);
}
