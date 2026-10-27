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

/* Pins: every default `tadah explain` prints for an annealing key is a
 * value the annealer takes, or says in words what the key takes without
 * a line (default_doc of the registry): an ON/OFF key prints ON or OFF,
 * a key without a fixed default prints its rule ("(default: calibrated
 * with probes)"), and a key the annealer refuses (RESET_TO_BEST of
 * TADAH:ANNEAL_PA) is not listed.
 * Before the fix: "PARAM CORANA (default 0)", "PARAM REANNEAL (default
 * 1)" and "PARAM LADDER_FREEZE (default 1)", which the parser refuses
 * ("0/1 booleans were removed in 1.5.0"), "PARAM T_INIT (default auto)",
 * a token T_INIT refuses, and TADAH:ANNEAL_PA listed RESET_TO_BEST with
 * "(default auto)". */
TEST_CASE("vocabulary: every printed default of an annealing key is a "
          "value it takes", "[HPO][anneal][vocabulary][caps]")
{
    namespace reg = tadah::hpo::anneal_registry;
    const auto help = [](const char* algo) {
        std::ostringstream captured;
        auto* old = std::cout.rdbuf(captured.rdbuf());
        tadah::hpo::OptimizerSpec::print_caps("TADAH", algo);
        std::cout.rdbuf(old);
        return captured.str();
    };
    /* The listing line of a key ("     PARAM <key> (default ...)" or
     * "     PARAM <key> <A|B> ..."), not a mention of the key in the prose
     * of the help ("     PARAM CORANA ON selects ..."). */
    const auto line_of = [](const std::string& text, const std::string& key) {
        const std::string head = "\n     PARAM " + key + " ";
        for (auto found = text.find(head); found != std::string::npos;
             found = text.find(head, found + 1)) {
            const char next = text[found + head.size()];
            if (next != '(' && next != '<') continue;
            const auto at = found + 6;
            return text.substr(at, text.find('\n', at) - at);
        }
        return std::string();
    };
    struct Family { const char* algo; unsigned mask; };
    for (const Family f : {Family{"ANNEAL", reg::KIND_SERIAL},
                           Family{"ANNEAL_PT", reg::KIND_PT},
                           Family{"ANNEAL_PA", reg::KIND_PA}}) {
        INFO(f.algo);
        const std::string text = help(f.algo);
        REQUIRE_FALSE(text.empty());
        CHECK(text.find("(default auto)") == std::string::npos);
        for (const auto& k : reg::kKeys) {
            if (!(k.kinds & f.mask) || k.type != reg::AnnealKeyType::Bool)
                continue;
            bool refused = false;
            for (const auto& r : reg::kRules)
                refused = refused ||
                          (r.type == reg::AnnealRuleType::KindReject &&
                           std::string_view(r.key) == k.name &&
                           (r.kind_mask & f.mask));
            const std::string line = line_of(text, k.name);
            INFO(k.name << ": " << line);
            if (refused) {
                CHECK(line.empty());
                continue;
            }
            REQUIRE_FALSE(line.empty());
            CHECK((line.find("(default ON)") != std::string::npos ||
                   line.find("(default OFF)") != std::string::npos ||
                   line.find("(default: ON") != std::string::npos ||
                   line.find("(default: OFF") != std::string::npos));
        }
    }
    const std::string serial = help("ANNEAL");
    const std::string pt = help("ANNEAL_PT");
    const std::string pa = help("ANNEAL_PA");
    CHECK(line_of(serial, "CORANA") == "PARAM CORANA (default OFF)");
    CHECK(line_of(serial, "REANNEAL") == "PARAM REANNEAL (default ON)");
    CHECK(line_of(serial, "T_INIT") ==
          "PARAM T_INIT (default: calibrated with probes)");
    CHECK(line_of(serial, "T_MIN") ==
          "PARAM T_MIN (default: 1e-6 times the starting temperature)");
    CHECK(line_of(serial, "CHAIN_LENGTH") ==
          "PARAM CHAIN_LENGTH (default: AUTO, from MAXEVAL) [accepts AUTO]");
    CHECK(line_of(serial, "RESET_TO_BEST") ==
          "PARAM RESET_TO_BEST (default: OFF; ON with CORANA ON)");
    CHECK(line_of(pt, "LADDER_FREEZE") == "PARAM LADDER_FREEZE (default ON)");
    CHECK(line_of(pt, "T_HOT") ==
          "PARAM T_HOT (default: calibrated with probes)");
    CHECK(line_of(pt, "T_COLD") ==
          "PARAM T_COLD (default: T_HOT divided by LADDER_SPAN)");
    /* A value default, OFF, as for every other ON/OFF key (it read
     * "(default: OFF)", a default in words). */
    CHECK(line_of(pt, "RESET_TO_BEST") ==
          "PARAM RESET_TO_BEST (default OFF)");
    CHECK(line_of(pt, "REPLICAS") ==
          "PARAM REPLICAS (default: AUTO, the larger of THREADS and 2) "
          "[accepts AUTO]");
    CHECK(line_of(pa, "RESET_TO_BEST").empty());
    CHECK(line_of(pa, "T_HOT") ==
          "PARAM T_HOT (default: calibrated with probes)");
}

/* AUTO of CHAIN_LENGTH and NT is the serial TADAH:ANNEAL's: it derives the
 * value from MAXEVAL and the cooling schedule. TADAH:ANNEAL_PA has static
 * defaults for both (50 and 5) and no such derivation, so it refuses AUTO,
 * naming the key and the optimiser that takes it; the serial TADAH:ANNEAL
 * accepts it as before, and the help text offers AUTO for the serial
 * optimiser only.
 * Before the fix: TADAH:ANNEAL_PA accepted AUTO and resolved it to the
 * registry's internal 0, so every temperature step had 0 evaluations: a
 * run planned '~18446744073709551615 temperature step(s)' and stopped after
 * its initialisation ('MAXEVAL exhausted after 36 host eval(s) (0
 * temperature step(s))' with MAXEVAL 200). */
TEST_CASE("vocabulary: CHAIN_LENGTH AUTO and NT AUTO are the serial "
          "TADAH:ANNEAL's", "[HPO][anneal_core][registry][vocabulary]")
{
    SECTION("TADAH:ANNEAL_PA refuses them") {
        for (const char* key : {"CHAIN_LENGTH", "NT"}) {
            for (const char* tok : {"AUTO", "auto"}) {
                INFO(key << " " << tok);
                auto sp = spec_for(anneal::AlgoKind::PA);
                sp.param_raw[key] = tok;
                if (std::string(key) == "NT") sp.param_raw["CORANA"] = "ON";
                REQUIRE_THROWS_WITH(
                    anneal::Knobs::parse(sp, anneal::AlgoKind::PA),
                    Catch::Contains(
                        std::string("TADAH:ANNEAL_PA: PARAM ") + key +
                        " must be a number (got '" + tok + "'; AUTO is "
                        "accepted by TADAH:ANNEAL only)"));
            }
        }
    }
    SECTION("the serial TADAH:ANNEAL accepts them") {
        auto sp = spec_for(anneal::AlgoKind::Serial);
        sp.param_raw["CHAIN_LENGTH"] = "AUTO";
        CHECK_NOTHROW(anneal::Knobs::parse(sp, anneal::AlgoKind::Serial));
        auto corana = spec_for(anneal::AlgoKind::Serial);
        corana.param_raw["CORANA"] = "ON";
        corana.param_raw["NT"] = "AUTO";
        CHECK_NOTHROW(anneal::Knobs::parse(corana, anneal::AlgoKind::Serial));
    }
    SECTION("the help text offers AUTO for the serial optimiser only") {
        const auto line_of = [](const char* algo, const std::string& key) {
            std::ostringstream captured;
            auto* old = std::cout.rdbuf(captured.rdbuf());
            tadah::hpo::OptimizerSpec::print_caps("TADAH", algo);
            std::cout.rdbuf(old);
            const std::string text = captured.str();
            const auto at = text.find("PARAM " + key + " ");
            REQUIRE(at != std::string::npos);
            return text.substr(at, text.find('\n', at) - at);
        };
        for (const std::string key : {"CHAIN_LENGTH", "NT"}) {
            INFO(key);
            CHECK(line_of("ANNEAL", key).find("[accepts AUTO]") !=
                  std::string::npos);
            const std::string pa = line_of("ANNEAL_PA", key);
            INFO(pa);
            CHECK(pa.find("AUTO") == std::string::npos);
        }
    }
}
