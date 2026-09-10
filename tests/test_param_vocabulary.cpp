// test_param_vocabulary.cpp
//
// The 1.5.0 PARAM value vocabulary: ON / OFF / AUTO instead of magic numbers.
//
// Before 1.5.0 the same literal meant different things on different keys --
// `SWAP_WINDOW 0` meant "cumulative tables only" while `REPLICAS 0` meant
// "derive from THREADS" and `CORANA 0` meant false. The rules now are:
//
//   * booleans take ON / OFF; a number is a hard error carrying a migration
//     hint (accepting 1 silently would make the two spellings
//     indistinguishable in a config, which is the ambiguity being removed);
//   * a numeric key may declare AUTO ("derive at run time") and/or OFF
//     ("feature disabled") in the registry, resolving to an internal value
//     that is outside the key's own legal numeric range;
//   * 0 never carries special meaning again -- where it used to be a mode
//     switch it is now simply out of range.
//
// The bool loop is registry-driven, so a boolean added to
// blueprint/anneal_keys.toml is covered here with no test edit.

#include "catch2/catch.hpp"

#include <tadah/hpo/anneal_core.h>
#include <tadah/hpo/anneal_keys_gen.h>
#include <tadah/hpo/optimizer_spec.h>

#include <iostream>
#include <sstream>

#include <string>

using namespace tadah::hpo;
namespace reg = anneal_registry;

namespace {

/* The kind a key belongs to, preferring serial so the spec stays minimal. */
anneal::AlgoKind kind_for(const reg::AnnealKeyInfo& ki)
{
    if (ki.kinds & reg::KIND_SERIAL) return anneal::AlgoKind::Serial;
    if (ki.kinds & reg::KIND_PT)     return anneal::AlgoKind::PT;
    return anneal::AlgoKind::PA;
}

OptimizerSpec spec_for(anneal::AlgoKind kind)
{
    OptimizerSpec sp;
    sp.lib  = "TADAH";
    sp.algo = kind == anneal::AlgoKind::Serial ? "ANNEAL"
            : kind == anneal::AlgoKind::PT     ? "ANNEAL_PT" : "ANNEAL_PA";
    sp.low  = {-1.0};
    sp.high = {1.0};
    sp.is_int = {false};
    sp.log_scaled = {false};
    sp.stop.maxeval = 1000;   /* serial CHAIN_LENGTH/NT AUTO need a budget */
    return sp;
}

}  // namespace

TEST_CASE("vocabulary: every boolean key takes ON/OFF and rejects 0/1",
          "[HPO][anneal_core][registry][vocabulary]")
{
    std::size_t checked = 0;
    for (const auto& ki : reg::kKeys) {
        if (ki.type != reg::AnnealKeyType::Bool) continue;
        const auto kind = kind_for(ki);
        INFO("bool key " << ki.name << " kind " << anneal::algo_name(kind));
        ++checked;

        /* ON / OFF parse. (Cross-key rules may still reject the key in this
         * context; what matters is that the VALUE is never the complaint.) */
        for (const char* tok : {"ON", "OFF"}) {
            auto sp = spec_for(kind);
            sp.param_raw[ki.name] = tok;
            try { anneal::Knobs::parse(sp, kind); }
            catch (const std::exception& e) {
                REQUIRE_THAT(std::string(e.what()),
                             !Catch::Matchers::Contains("takes ON or OFF"));
            }
        }

        /* A number is rejected, and the message must carry the migration
         * hint -- not merely "invalid value". */
        for (double v : {0.0, 1.0}) {
            auto sp = spec_for(kind);
            sp.param[ki.name] = v;
            REQUIRE_THROWS_WITH(
                anneal::Knobs::parse(sp, kind),
                Catch::Contains(std::string(ki.name) + " takes ON or OFF")
                    && Catch::Contains("0/1 booleans were removed in 1.5.0"));
        }

        /* A word that is not ON/OFF is rejected too. */
        auto bad = spec_for(kind);
        bad.param_raw[ki.name] = "true";
        REQUIRE_THROWS_WITH(anneal::Knobs::parse(bad, kind),
                            Catch::Contains("takes ON or OFF"));
    }
    CHECK(checked >= 5u);   /* CORANA RESET_TO_BEST REANNEAL LADDER_* */
}

TEST_CASE("vocabulary: AUTO and OFF resolve to the registry's internal value",
          "[HPO][anneal_core][registry][vocabulary]")
{
    std::size_t auto_keys = 0, off_keys = 0;
    for (const auto& ki : reg::kKeys) {
        if (!ki.accepts_auto && !ki.accepts_off) continue;
        const auto kind = kind_for(ki);
        INFO("key " << ki.name << " kind " << anneal::algo_name(kind));

        /* Some of these keys carry cross-key rules (NT requires CORANA ON,
         * STEP_TRUST_FACTOR requires STEP_MODE ADAPTIVE). Setting up every
         * rule's context is test_anneal_registry.cpp's job; here all that
         * matters is that the VALUE is never what the parser objects to. */
        const auto value_accepted = [&](const char* tok) {
            auto sp = spec_for(kind);
            sp.param_raw[ki.name] = tok;
            try { anneal::Knobs::parse(sp, kind); }
            catch (const std::exception& e) {
                const std::string what = e.what();
                INFO("message: " << what);
                CHECK(what.find("must be a number") == std::string::npos);
                CHECK(what.find("got '" + std::string(tok) + "'")
                      == std::string::npos);
            }
        };
        if (ki.accepts_auto) {
            ++auto_keys;
            value_accepted("AUTO");
            value_accepted("auto");   /* case-insensitive */
        }
        if (ki.accepts_off) {
            ++off_keys;
            value_accepted("OFF");
        }
        /* A key that accepts only AUTO must not silently accept OFF, and
         * vice versa -- they are different intents. */
        if (!ki.accepts_off) {
            auto sp = spec_for(kind);
            sp.param_raw[ki.name] = "OFF";
            REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, kind),
                                Catch::Contains(ki.name));
        }
        if (!ki.accepts_auto) {
            auto sp = spec_for(kind);
            sp.param_raw[ki.name] = "AUTO";
            REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, kind),
                                Catch::Contains(ki.name));
        }
    }
    CHECK(auto_keys >= 3u);   /* REPLICAS CHAIN_LENGTH NT */
    CHECK(off_keys  >= 3u);   /* SWAP_WINDOW STEP_TRUST_FACTOR ESS_TARGET */
}

TEST_CASE("vocabulary: REPLICAS AUTO resolves, REPLICAS 0 does not",
          "[HPO][anneal_core][registry][vocabulary]")
{
    /* The concrete case that motivated the change: 0 used to mean "derive
     * from THREADS" on REPLICAS and "disabled" on SWAP_WINDOW. */
    auto ok = spec_for(anneal::AlgoKind::PT);
    ok.param_raw["REPLICAS"] = "AUTO";
    const auto kb = anneal::Knobs::parse(ok, anneal::AlgoKind::PT);
    CHECK(kb.replicas == 0u);          /* 0 = "resolve later from THREADS" */

    auto bad = spec_for(anneal::AlgoKind::PT);
    bad.param["REPLICAS"] = 0.0;
    REQUIRE_THROWS_WITH(anneal::Knobs::parse(bad, anneal::AlgoKind::PT),
                        Catch::Contains("REPLICAS must be >= 2"));
}

TEST_CASE("vocabulary: SWAP_WINDOW OFF resolves, SWAP_WINDOW 0 does not",
          "[HPO][anneal_core][registry][vocabulary]")
{
    auto ok = spec_for(anneal::AlgoKind::PT);
    ok.param_raw["SWAP_WINDOW"] = "OFF";
    const auto kb = anneal::Knobs::parse(ok, anneal::AlgoKind::PT);
    CHECK(kb.swap_window == 0u);       /* 0 = cumulative tables only */

    auto bad = spec_for(anneal::AlgoKind::PT);
    bad.param["SWAP_WINDOW"] = 0.0;
    REQUIRE_THROWS_WITH(anneal::Knobs::parse(bad, anneal::AlgoKind::PT),
                        Catch::Contains("SWAP_WINDOW"));
}

TEST_CASE("vocabulary: a token on a plain numeric key names what is accepted",
          "[HPO][anneal_core][registry][vocabulary]")
{
    /* STEP_FRAC declares neither AUTO nor OFF, so a token must be rejected
     * with a message that says what the key does take. */
    auto sp = spec_for(anneal::AlgoKind::Serial);
    sp.param_raw["STEP_FRAC"] = "AUTO";
    REQUIRE_THROWS_WITH(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial),
                        Catch::Contains("STEP_FRAC must be a number"));
}



TEST_CASE("vocabulary: the help text advertises the TOKEN, not the sentinel",
          "[HPO][anneal][vocabulary][caps]")
{
    /* Keys whose AUTO/OFF sentinel IS their numeric default used to print
     * "(default 0)" -- a value the parser then rejects, sending the reader
     * to try exactly the spelling 1.5.0 removed. The capability text is the
     * first place anyone looks, so what it prints is a contract. */
    std::ostringstream captured;
    auto* old = std::cout.rdbuf(captured.rdbuf());
    tadah::hpo::OptimizerSpec::print_caps("TADAH", "ANNEAL_PT");
    std::cout.rdbuf(old);
    const std::string text = captured.str();

    REQUIRE_FALSE(text.empty());
    /* Ladder scaling: AUTO by default, and OFF is the documented escape. */
    const auto at = text.find("PARAM INIT_JITTER_LADDER");
    REQUIRE(at != std::string::npos);
    const std::string line = text.substr(at, text.find('\n', at) - at);
    INFO("rendered: " << line);
    CHECK(line.find("(default AUTO)") != std::string::npos);
    CHECK(line.find("[accepts AUTO/OFF]") != std::string::npos);
    CHECK(line.find("default -1") == std::string::npos);

    /* And the pre-existing AUTO key that motivated the vocabulary change. */
    const auto me = text.find("PARAM INIT_JITTER_MAXEVAL");
    REQUIRE(me != std::string::npos);
    const std::string me_line = text.substr(me, text.find('\n', me) - me);
    INFO("rendered: " << me_line);
    CHECK(me_line.find("(default AUTO)") != std::string::npos);
    CHECK(me_line.find("(default 0)") == std::string::npos);
}
