/*
 * Table-driven coverage of the ANNEAL-family key registry
 * (blueprint/anneal_keys.toml -> anneal_keys_gen.h -> Knobs::parse).
 *
 * The loops below iterate the generated tables, so a key or rule added
 * to the TOML is exercised here with no test edit:
 *   - every key parses successfully at a valid value for every kind it
 *     belongs to (with any cross-key context it requires),
 *   - every key is rejected with a steering message for kinds it does
 *     not belong to,
 *   - every table-declared range rejects an out-of-range value with the
 *     registry's message text,
 *   - every cross-key rule fires with its exact message,
 *   - non-numeric values, unknown names, and renamed keys are hard
 *     errors.
 */

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_keys_gen.h>
#include <tadah/hpo/optimizer_spec.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

using namespace tadah::hpo;
namespace reg = anneal_registry;

namespace {

const std::vector<std::pair<anneal::AlgoKind, unsigned>> kAllKinds = {
    {anneal::AlgoKind::Serial, reg::KIND_SERIAL},
    {anneal::AlgoKind::PT, reg::KIND_PT},
    {anneal::AlgoKind::PA, reg::KIND_PA},
};

OptimizerSpec registry_base_spec(anneal::AlgoKind k)
{
    OptimizerSpec sp;
    sp.lib = "TADAH";
    sp.algo = anneal::algo_name(k);
    sp.low = {-1.0};
    sp.high = {1.0};
    sp.is_int = {false};
    sp.log_scaled = {false};
    if (k == anneal::AlgoKind::PA) sp.pop = 16;
    return sp;
}

/* Cross-key context that makes an explicitly-set key legal.  This is the
 * only per-key knowledge in the file; a new key with no entry here is
 * assumed to be legal standalone (the valid-value loop will fail loudly
 * if that assumption breaks). */
void enable_context(OptimizerSpec& sp, const std::string& key,
                    anneal::AlgoKind kind)
{
    if (key == "NS" || key == "NT" || key == "C" || key == "V_FLOOR_FRAC")
        sp.param["CORANA"] = 1.0;
    if (key.rfind("LADDER_ADAPT_", 0) == 0 || key == "LADDER_MIN_GAP" ||
        key == "LADDER_FREEZE")
        sp.param["LADDER_ADAPT"] = 1.0;
    if (key == "STEP_TRUST_FACTOR") sp.param["STEP_ADAPT_EVERY"] = 8.0;
    if (key == "T_MIN" && kind == anneal::AlgoKind::PT)
        sp.param["RT"] = 0.5;
}

/* A value that passes the key's declared range (customs get literals). */
double good_value(const reg::AnnealKeyInfo& ki)
{
    const std::string name = ki.name;
    if (name == "REPLICAS") return 4.0;
    if (name == "T0_PROBES") return 4.0;
    if (name == "STEP_TRUST_FACTOR") return 8.0;
    if (name == "RT") return 0.5;
    if (name == "T_HOT" || name == "T_MIN" || name == "T_COLD") return 0.5;
    if (ki.type == reg::AnnealKeyType::Bool) return 1.0;
    if (!ki.has_min && !ki.has_max)
        return 1.0;
    if (ki.type == reg::AnnealKeyType::Count) {
        if (ki.has_min) return ki.min;
        return ki.max;  /* only-max counts: the top of the enum range */
    }
    if (ki.has_min && ki.has_max) return (ki.min + ki.max) / 2.0;
    if (ki.has_min) return ki.min + 1.0;
    return ki.max / 2.0;
}

/* Values that violate the declared range (empty for custom ranges). */
std::vector<double> bad_values(const reg::AnnealKeyInfo& ki)
{
    std::vector<double> bad;
    if (ki.type == reg::AnnealKeyType::Bool) {
        bad.push_back(2.0);
        return bad;
    }
    if (!ki.range_msg) return bad;
    if (ki.has_min) bad.push_back(ki.min_open ? ki.min : ki.min - 1.0);
    if (ki.has_max) bad.push_back(ki.max_open ? ki.max : ki.max + 1.0);
    /* Counts cannot represent negatives through the range path (the
     * integer-form check fires first with its own message). */
    if (ki.type == reg::AnnealKeyType::Count)
        bad.erase(std::remove_if(bad.begin(), bad.end(),
                                 [](double v) { return v < 0.0; }),
                  bad.end());
    return bad;
}

bool kind_rejected(const char* key, unsigned kind_bit)
{
    for (const auto& rule : reg::kRules)
        if (rule.type == reg::AnnealRuleType::KindReject &&
            std::string(rule.key) == key && (rule.kind_mask & kind_bit))
            return true;
    return false;
}

}  // namespace

TEST_CASE("registry: every key parses at a valid value for every kind it "
          "belongs to", "[HPO][anneal_core][registry]")
{
    for (const auto& ki : reg::kKeys) {
        for (const auto& [kind, bit] : kAllKinds) {
            if (!(ki.kinds & bit)) continue;
            if (kind_rejected(ki.name, bit)) continue;
            INFO("key " << ki.name << " kind " << anneal::algo_name(kind));
            auto sp = registry_base_spec(kind);
            enable_context(sp, ki.name, kind);
            sp.param[ki.name] = good_value(ki);
            CHECK_NOTHROW(anneal::Knobs::parse(sp, kind));
        }
    }
}

TEST_CASE("registry: every key steers to its family members for kinds it "
          "does not belong to", "[HPO][anneal_core][registry]")
{
    for (const auto& ki : reg::kKeys) {
        for (const auto& [kind, bit] : kAllKinds) {
            if (ki.kinds & bit) continue;
            INFO("key " << ki.name << " kind " << anneal::algo_name(kind));
            auto sp = registry_base_spec(kind);
            sp.param[ki.name] = good_value(ki);
            REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, kind),
                                Catch::Contains("belongs to"));
        }
    }
}

TEST_CASE("registry: every declared range rejects out-of-range values with "
          "the registry message", "[HPO][anneal_core][registry]")
{
    for (const auto& ki : reg::kKeys) {
        const auto bad = bad_values(ki);
        if (bad.empty()) continue;
        /* One kind suffices: range text is kind-independent. */
        const auto [kind, bit] = *std::find_if(
            kAllKinds.begin(), kAllKinds.end(),
            [&](const auto& kb) { return ki.kinds & kb.second; });
        if (kind_rejected(ki.name, bit)) continue;
        const std::string expect = ki.range_msg
            ? std::string(ki.range_msg).substr(0, std::string(ki.range_msg).find(" ("))
            : std::string("must be 0 or 1");
        for (double v : bad) {
            INFO("key " << ki.name << " value " << v);
            auto sp = registry_base_spec(kind);
            enable_context(sp, ki.name, kind);
            sp.param[ki.name] = v;
            REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, kind),
                                Catch::Contains(expect));
        }
    }
}

TEST_CASE("registry: every cross-key rule fires with its exact message",
          "[HPO][anneal_core][registry]")
{
    for (const auto& rule : reg::kRules) {
        const auto* ki = reg::find_key(rule.key);
        REQUIRE(ki != nullptr);
        anneal::AlgoKind kind = anneal::AlgoKind::Serial;
        unsigned bit = reg::KIND_SERIAL;
        if (rule.type == reg::AnnealRuleType::KindReject) {
            for (const auto& [k, b] : kAllKinds)
                if (rule.kind_mask & b) { kind = k; bit = b; }
        } else {
            /* The first kind the key belongs to (rules themselves are
             * kind-independent apart from KindReject). */
            for (const auto& [k, b] : kAllKinds)
                if (ki->kinds & b) { kind = k; bit = b; break; }
        }
        (void)bit;
        INFO("rule on " << rule.key << " kind " << anneal::algo_name(kind));
        auto sp = registry_base_spec(kind);
        sp.param[rule.key] = good_value(*ki);
        switch (rule.type) {
            case reg::AnnealRuleType::RequiresFlag:
                /* flag absent -> rule fires */
                break;
            case reg::AnnealRuleType::ConflictsFlag:
                sp.param[rule.arg] = 1.0;
                break;
            case reg::AnnealRuleType::UnusedWhenSet:
                sp.param[rule.arg] = good_value(*reg::find_key(rule.arg));
                break;
            case reg::AnnealRuleType::UnusedWhenFlagOff:
                sp.param[rule.arg] = 0.0;
                break;
            case reg::AnnealRuleType::KindReject:
                break;
        }
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, kind),
                            Catch::Contains(rule.msg));
    }
}

TEST_CASE("registry: non-numeric, unknown, and renamed PARAMs are hard "
          "errors", "[HPO][anneal_core][registry]")
{
    SECTION("token value for a numeric key") {
        auto sp = registry_base_spec(anneal::AlgoKind::Serial);
        sp.param_raw["RT"] = "fast";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("must be a number"));
    }
    SECTION("token value for an unknown key names the key, not the value") {
        auto sp = registry_base_spec(anneal::AlgoKind::Serial);
        sp.param_raw["NO_SUCH_KEY"] = "abc";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("unknown PARAM"));
    }
    SECTION("unknown numeric key lists the accepted keys") {
        auto sp = registry_base_spec(anneal::AlgoKind::Serial);
        sp.param["NO_SUCH_KEY"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("accepted keys"));
    }
    SECTION("renamed key is caught in numeric and token form") {
        auto sp = registry_base_spec(anneal::AlgoKind::Serial);
        sp.param["T_INIT"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                            Catch::Contains("renamed to T_HOT"));
        auto sp2 = registry_base_spec(anneal::AlgoKind::Serial);
        sp2.param_raw["T_INIT"] = "hot";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp2, anneal::AlgoKind::Serial),
                            Catch::Contains("renamed to T_HOT"));
    }
}

TEST_CASE("registry: defaults in the generated header match the parsed "
          "defaults", "[HPO][anneal_core][registry]")
{
    /* A bare spec must produce knob values equal to the registry
     * defaults (spot checks across the three types + a PT override). */
    auto serial = registry_base_spec(anneal::AlgoKind::Serial);
    const auto kb = anneal::Knobs::parse(serial, anneal::AlgoKind::Serial);
    CHECK(kb.L == static_cast<std::size_t>(anneal_defaults::L));
    CHECK(kb.step_frac == anneal_defaults::STEP_FRAC);
    CHECK(kb.corana == (anneal_defaults::CORANA == 1.0));
    CHECK(kb.rt == anneal_defaults::RT);

    auto pt = registry_base_spec(anneal::AlgoKind::PT);
    const auto kbp = anneal::Knobs::parse(pt, anneal::AlgoKind::PT);
    CHECK(kbp.rt == anneal_defaults::RT_PT);
    CHECK(kbp.swap_window ==
          static_cast<std::size_t>(anneal_defaults::SWAP_WINDOW));
}
