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
#include <cctype>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <sstream>
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
    /* Serial CHAIN_LENGTH/NT default to AUTO, which requires MAXEVAL. */
    sp.stop.maxeval = 1000;
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
    (void)kind;
    if (key == "NS" || key == "NT" || key == "CORANA_C" ||
        key == "V_FLOOR_FRAC")
        sp.param_raw["CORANA"] = "ON";
    if (key.rfind("LADDER_ADAPT_", 0) == 0 || key == "LADDER_MIN_GAP" ||
        key == "LADDER_FREEZE")
        sp.param_raw["LADDER_ADAPT"] = "ON";
    if (key == "STEP_TRUST_FACTOR" || key == "STEP_ADAPT_EVERY")
        sp.param_raw["STEP_MODE"] = "ADAPTIVE";
    /* The INIT_JITTER_* shape/cost keys are meaningful only under the
     * multi-token `INIT_MODE JITTER AUTO <p>` form. The rule table can only
     * express "requires INIT_MODE == JITTER" (AUTO is an EXTRA on the enum
     * value, not an enum value of its own), so the rest is a hand-written
     * check in Knobs::parse and the context here has to match it. */
    if (key.rfind("INIT_JITTER_", 0) == 0) {
        sp.param_raw["INIT_MODE"] = "JITTER";
        sp.param_tokens["INIT_MODE"] = {"JITTER", "AUTO", "0.05"};
    }
}

/* A value that passes the key's declared range (customs get literals). */
double good_value(const reg::AnnealKeyInfo& ki)
{
    const std::string name = ki.name;
    if (name == "REPLICAS") return 4.0;
    if (name == "T0_PROBES") return 4.0;
    if (name == "STEP_TRUST_FACTOR") return 8.0;
    if (name == "RT") return 0.5;
    if (name == "T_INIT" || name == "T_HOT" || name == "T_MIN" ||
        name == "T_COLD")
        return 0.5;
    if (ki.type == reg::AnnealKeyType::Bool) return 1.0;
    if (ki.type == reg::AnnealKeyType::Enum) return 0.0;  /* value index */
    if (!ki.has_min && !ki.has_max)
        return 1.0;
    if (ki.type == reg::AnnealKeyType::Count) {
        if (ki.has_min) return ki.min;
        return ki.max;  /* only-max counts: the top of the range */
    }
    if (ki.has_min && ki.has_max) return (ki.min + ki.max) / 2.0;
    if (ki.has_min) return ki.min + 1.0;
    return ki.max / 2.0;
}

/* Write a valid value for the key into the spec. Since 1.5.0 the value
 * vocabulary is token-first: enum keys take their token, booleans take
 * ON/OFF, and only genuinely numeric keys go to `param`. */
void set_good(OptimizerSpec& sp, const reg::AnnealKeyInfo& ki)
{
    if (ki.type == reg::AnnealKeyType::Enum)
        sp.param_raw[ki.name] =
            ki.enum_values[static_cast<std::size_t>(good_value(ki))];
    else if (ki.type == reg::AnnealKeyType::Bool)
        sp.param_raw[ki.name] = good_value(ki) != 0.0 ? "ON" : "OFF";
    else
        sp.param[ki.name] = good_value(ki);
}

/* Values that violate the declared range (empty for custom ranges and
 * enum keys, which get their own dedicated checks). */
std::vector<double> bad_values(const reg::AnnealKeyInfo& ki)
{
    std::vector<double> bad;
    if (ki.type == reg::AnnealKeyType::Enum) return bad;
    if (ki.type == reg::AnnealKeyType::Bool) {
        /* Booleans no longer have a numeric range: ANY number is rejected
         * with the ON/OFF migration message, which has its own test. */
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
            set_good(sp, ki);
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
            /* A rename hint for this kind takes precedence over the
             * membership steering (e.g. RT/T_MIN/NT under PT). */
            bool renamed_hint = false;
            for (const auto& ren : reg::kRenamed)
                if ((ren.kinds & bit) &&
                    std::string(ren.old_name) == ki.name)
                    renamed_hint = true;
            if (renamed_hint) continue;
            INFO("key " << ki.name << " kind " << anneal::algo_name(kind));
            auto sp = registry_base_spec(kind);
            set_good(sp, ki);
            /* It was "... belongs to ...". */
            REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, kind),
                                Catch::Contains(": PARAM " +
                                                std::string(ki.name) +
                                                " is a key of TADAH:"));
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
            /* The first kind where the rule can actually fire: both the
             * rule's key and, for UnusedWhenSet, the `other` key must be
             * live keys for that kind (e.g. the T0_*-vs-T_HOT rules can
             * only fire for PT/PA, where T_HOT exists). */
            unsigned need = ki->kinds;
            if (rule.type == reg::AnnealRuleType::UnusedWhenSet) {
                const auto* other = reg::find_key(rule.arg);
                REQUIRE(other != nullptr);
                need &= other->kinds;
            }
            REQUIRE(need != 0u);
            for (const auto& [k, b] : kAllKinds)
                if (need & b) { kind = k; bit = b; break; }
        }
        (void)bit;
        INFO("rule on " << rule.key << " kind " << anneal::algo_name(kind));
        auto sp = registry_base_spec(kind);
        set_good(sp, *ki);
        switch (rule.type) {
            case reg::AnnealRuleType::RequiresFlag:
                /* flag absent -> rule fires */
                break;
            case reg::AnnealRuleType::ConflictsFlag:
                sp.param_raw[rule.arg] = "ON";   /* flags are ON/OFF (1.5.0) */
                break;
            case reg::AnnealRuleType::UnusedWhenSet:
                set_good(sp, *reg::find_key(rule.arg));
                break;
            case reg::AnnealRuleType::UnusedWhenFlagOff:
                sp.param_raw[rule.arg] = "OFF";
                break;
            case reg::AnnealRuleType::RequiresEnum:
                /* enum stays at its default, which the rule requires
                 * it not to be */
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
        auto sp = registry_base_spec(anneal::AlgoKind::PT);
        sp.param["STEP_AUTO"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::PT),
                            Catch::Contains("replaced by PARAM STEP_MODE"));
        auto sp2 = registry_base_spec(anneal::AlgoKind::PT);
        sp2.param_raw["STEP_AUTO"] = "on";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp2, anneal::AlgoKind::PT),
                            Catch::Contains("replaced by PARAM STEP_MODE"));
    }
}

/* Pins: the refusal of an unknown PARAM lists, in registry order, the keys
 * the annealer takes: the keys of its kind but those a kind_reject rule
 * refuses for it, so TADAH:ANNEAL_PA does not list RESET_TO_BEST, which it
 * refuses ("PARAM RESET_TO_BEST is not defined for population annealing"),
 * and TADAH:ANNEAL_PT does.
 * Before the fix: the list held every key of the kind, RESET_TO_BEST under
 * TADAH:ANNEAL_PA included. */
TEST_CASE("registry: the accepted keys of an unknown-key refusal leave out "
          "a key the annealer refuses", "[HPO][anneal_core][registry]")
{
    for (const auto& [kind, mask] : kAllKinds) {
        INFO("kind " << anneal::algo_name(kind));
        auto sp = registry_base_spec(kind);
        sp.param["NO_SUCH_KEY"] = 1.0;
        std::string what;
        try {
            (void)anneal::Knobs::parse(sp, kind);
        } catch (const std::exception& e) {
            what = e.what();
        }
        const std::string head = "accepted keys: ";
        const auto at = what.find(head);
        REQUIRE(at != std::string::npos);
        std::istringstream is(what.substr(at + head.size()));
        std::vector<std::string> listed;
        for (std::string k; is >> k;) listed.push_back(k);
        std::vector<std::string> expected;
        for (const auto& k : reg::kKeys) {
            if (!(k.kinds & mask)) continue;
            const bool refused = std::any_of(
                std::begin(reg::kRules), std::end(reg::kRules),
                [&](const reg::AnnealKeyRule& r) {
                    return r.type == reg::AnnealRuleType::KindReject &&
                           (r.kind_mask & mask) &&
                           std::strcmp(r.key, k.name) == 0;
                });
            if (!refused) expected.push_back(k.name);
        }
        CHECK(listed == expected);
        const bool lists_reset =
            std::find(listed.begin(), listed.end(), "RESET_TO_BEST") !=
            listed.end();
        CHECK(lists_reset == (kind != anneal::AlgoKind::PA));
    }
}

TEST_CASE("registry: defaults in the generated header match the parsed "
          "defaults", "[HPO][anneal_core][registry]")
{
    /* A bare spec must produce knob values equal to the registry
     * defaults (spot checks across the types, kinds, and AUTO/enum
     * forms). */
    auto serial = registry_base_spec(anneal::AlgoKind::Serial);
    const auto kb = anneal::Knobs::parse(serial, anneal::AlgoKind::Serial);
    CHECK(kb.chain_length == 0);   /* AUTO sentinel: resolved at run time */
    CHECK(kb.NT == 0);             /* AUTO sentinel (Corana) */
    CHECK(kb.NS == static_cast<std::size_t>(anneal_defaults::NS));
    CHECK(kb.corana_c == anneal_defaults::CORANA_C);
    CHECK(kb.step_frac == anneal_defaults::STEP_FRAC);
    CHECK(kb.corana == (anneal_defaults::CORANA == 1.0));
    CHECK(kb.rt == anneal_defaults::RT_SERIAL);
    CHECK(kb.t0_accept == anneal_defaults::T0_ACCEPT);

    auto pt = registry_base_spec(anneal::AlgoKind::PT);
    const auto kbp = anneal::Knobs::parse(pt, anneal::AlgoKind::PT);
    CHECK(kbp.step_mode == anneal_enums::STEP_MODE::FROZEN);
    CHECK(kbp.init_mode == anneal_enums::INIT_MODE::JITTER);
    CHECK(kbp.swap_window ==
          static_cast<std::size_t>(anneal_defaults::SWAP_WINDOW));
    /* FROZEN normalizes the ADAPTIVE-only trust region to its off
     * sentinel (the RAM adapts per proposal; it has no cadence knob). */
    CHECK(kbp.step_trust_factor == 0.0);

    auto pa = registry_base_spec(anneal::AlgoKind::PA);
    const auto kba = anneal::Knobs::parse(pa, anneal::AlgoKind::PA);
    CHECK(kba.chain_length ==
          static_cast<std::size_t>(anneal_defaults::CHAIN_LENGTH_PA));
    CHECK(kba.rt == anneal_defaults::RT_PA);

    SECTION("enum keys reject numeric and unknown-token values") {
        auto sp = registry_base_spec(anneal::AlgoKind::PT);
        sp.param["STEP_MODE"] = 1.0;
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::PT),
                            Catch::Contains("takes a token"));
        auto sp2 = registry_base_spec(anneal::AlgoKind::PT);
        sp2.param_raw["STEP_MODE"] = "SOMETIMES";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp2, anneal::AlgoKind::PT),
                            Catch::Contains("must be one of"));
    }
    SECTION("every registry rename hint fires for its kinds") {
        for (const auto& ren : reg::kRenamed) {
            for (const auto& [kind, bit] : kAllKinds) {
                if (!(ren.kinds & bit)) continue;
                INFO("renamed " << ren.old_name << " kind "
                                << anneal::algo_name(kind));
                auto sp = registry_base_spec(kind);
                sp.param[ren.old_name] = 1.0;
                REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, kind),
                                    Catch::Contains(ren.msg));
            }
        }
    }
}

TEST_CASE("registry: the INIT_MODE JITTER numbers take the decimal number "
          "grammar", "[HPO][anneal_core][registry][hex]")
{
    /* PARAM INIT_MODE JITTER <fraction> | JITTER AUTO <p>: the extras are
     * numbers of the one grammar every key reads, the decimal one LAMMPS
     * reads. std::stod read hexadecimal floats, so `JITTER 0x1p-3` ran with
     * a fraction of 0.125. */
    auto jitter = [](std::vector<std::string> toks) {
        auto sp = registry_base_spec(anneal::AlgoKind::PT);
        sp.param_raw["INIT_MODE"] = "JITTER";
        sp.param_tokens["INIT_MODE"] = std::move(toks);
        return sp;
    };
    CHECK(anneal::Knobs::parse(jitter({"JITTER", "0.125"}),
                               anneal::AlgoKind::PT)
              .init_jitter_frac == 0.125);
    CHECK(anneal::Knobs::parse(jitter({"JITTER", "AUTO", "5e-2"}),
                               anneal::AlgoKind::PT)
              .init_jitter_tol == 0.05);
    REQUIRE_THROWS_WITH(
        anneal::Knobs::parse(jitter({"JITTER", "0x1p-3"}), anneal::AlgoKind::PT),
        Catch::Contains("PARAM INIT_MODE JITTER fraction must be a finite "
                        "number (got '0x1p-3')"));
    REQUIRE_THROWS_WITH(
        anneal::Knobs::parse(jitter({"JITTER", "AUTO", "0x1p-4"}),
                             anneal::AlgoKind::PT),
        Catch::Contains("PARAM INIT_MODE JITTER AUTO tolerance must be a "
                        "finite number (got '0x1p-4')"));
}

/* Pins: the comments of algorithms_pt.h name no key that the registry
 * retires for TADAH:ANNEAL_PT (its [[renamed]] entries for pt: RT,
 * T_MIN, T0_PACCEPT, STEP_AUTO, LADDER_ADAPT_UPDATES, ...); the keys and
 * their defaults are those of blueprint/anneal_keys.toml. One-letter names
 * (L, C) are left out: they are words of the formulas too.
 * Before the fix: the header described RT ladder cooling (RT < 1, a PT ->
 * SA hybrid ending at T_MIN), T0_PACCEPT, and a PARAM table with
 * LADDER_ADAPT_UPDATES, RT, STEP_AUTO, STEP_AUTO_SWEEPS and
 * STEP_ADAPT_EVERY; two body comments named RT cooling. */
TEST_CASE("registry: algorithms_pt.h names no key the registry retires for "
          "TADAH:ANNEAL_PT", "[HPO][anneal_pt][registry][lint]")
{
    namespace fs = std::filesystem;
    /* __FILE__ is an absolute path (CMake GLOBs the test sources). */
    const fs::path header = fs::path(__FILE__).parent_path().parent_path() /
                            "include" / "tadah" / "hpo" / "algorithms_pt.h";
    std::ifstream in(header);
    REQUIRE(in.good());
    std::vector<std::string> lines;
    for (std::string l; std::getline(in, l);) lines.push_back(l);
    REQUIRE(lines.size() > 100u);
    const auto word = [](char c) {
        return std::isalnum(static_cast<unsigned char>(c)) != 0 || c == '_';
    };
    std::size_t checked = 0;
    for (const auto& r : reg::kRenamed) {
        if (!(r.kinds & reg::KIND_PT) || std::strlen(r.old_name) < 2) continue;
        ++checked;
        const std::string name = r.old_name;
        for (std::size_t i = 0; i < lines.size(); ++i) {
            const std::string& l = lines[i];
            for (auto at = l.find(name); at != std::string::npos;
                 at = l.find(name, at + 1)) {
                const bool starts = at == 0 || !word(l[at - 1]);
                const bool ends = at + name.size() == l.size() ||
                                  !word(l[at + name.size()]);
                INFO("algorithms_pt.h:" << i + 1 << ": " << l);
                CHECK_FALSE((starts && ends));
            }
        }
    }
    CHECK(checked >= 8u);
}
