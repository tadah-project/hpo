/* hpotarget_parser.cpp -- the registry-driven parser of the target file
 * (hpotarget_parser.h describes it). The messages of its refusals come
 * from the generated tables where the registry holds them, and are built
 * here where a part of them is the parser's own (the set-once forms, the
 * K_TOTAL, metric, gate and script checks at the end of the file). */

#include "hpotarget_parser.h"

#include <tadah/core/number_format.h>
#include <tadah/hpo/init_strategy.h>
#include <tadah/hpo/lammps_sim.h>
#include <tadah/hpo/optimizer_caps.h>
#include <tadah/hpo/phase_specs.h>
#include <tadah/hpo/strategy_rules.h>
#include <tadah/hpo/strict_parse.h>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <set>
#include <sstream>
#include <unordered_map>

namespace tadah {
namespace hpo {
namespace hpotarget {

namespace {

using R::BlockId;
using R::fill;

std::string str(std::string_view s) { return std::string(s); }
std::string num(std::size_t n) { return std::to_string(n); }

/* The rule of the registry of id `id`. */
const R::Rule& rule(std::string_view id)
{
    for (const R::Rule& r : R::kRules)
        if (r.id_text == id) return r;
    throw std::logic_error("hpotarget parser: the registry has no rule " +
                           str(id));
}

/* The {what} rule `id` gives the entry `entry`. */
std::string_view rule_what(std::string_view id, std::string_view entry)
{
    for (const R::RuleWhat& w : rule(id).what)
        if (w.id_text == entry) return w.text;
    throw std::logic_error("hpotarget parser: the rule " + str(id) +
                           " has no {what} of " + str(entry));
}

/* The block an END line closes, nullptr for any other key. */
const R::Block* block_ended_by(const std::string& key)
{
    for (const R::Block& b : R::kBlocks)
        if (b.end == key) return &b;
    return nullptr;
}

bool is_top_level_key(const std::string& key)
{
    for (std::string_view k : R::kTopLevelKeys)
        if (k == key) return true;
    return false;
}

/* The key of the block `b` named `name`, nullptr for none. */
const R::Key* block_key(const R::Block& b, const std::string& name)
{
    for (R::KeyId id : b.keys) {
        const R::Key& k = R::key_entry(id);
        if (k.name == name) return &k;
    }
    return nullptr;
}

/* A key of the optimiser group named `name`, nullptr for none. */
const R::Key* group_key(const std::string& name)
{
    for (const R::Key& k : R::kKeys)
        if (!k.group.empty() && k.name == name) return &k;
    return nullptr;
}

/* The readers of the settings FAILSCORE, PC_SCRIPT_TIMEOUT and LOSS, which
 * a second line of the key is refused for naming. */
const char* readers_of(const std::string& name)
{
    if (name == "FAILSCORE")
        return "a PC_LAMMPS or PC_FAILPROBE line without --failscore";
    if (name == "PC_SCRIPT_TIMEOUT")
        return "a PC_LAMMPS or PC_FAILPROBE line without --maxtime";
    if (name == "LOSS")
        return "a --varloss or PC_OBSERVABLE term that names no loss "
               "function";
    throw std::logic_error("hpotarget parser: " + name + " is no setting "
                           "the lines after it read");
}

}  // namespace

/* ---------------------------------------------------------- state ---- */

Parser::Parser(ParsedTarget& target, const std::string& file)
    : t(target), fn(file) {}

BlockId Parser::block() const
{
    return open_blocks.empty() ? BlockId::ROOT : open_blocks.back().id;
}

BlockId Parser::parent() const
{
    return open_blocks.size() >= 2 ? open_blocks[open_blocks.size() - 2].id
                                   : BlockId::ROOT;
}

const R::Block& Parser::block_entry(BlockId id)
{
    for (const R::Block& b : R::kBlocks)
        if (b.id == id) return b;
    throw std::logic_error("hpotarget parser: ROOT is no block");
}

std::string Parser::block_name(BlockId id)
{
    return id == BlockId::ROOT ? std::string("ROOT")
                               : str(block_entry(id).name);
}

std::string Parser::inner_enclosing() const
{
    const BlockId around = parent();
    return around == BlockId::STRATEGY  ? "STRATEGY"
           : around == BlockId::EXPLORE ? "EXPLORE"
                                        : "REFINE";
}

void Parser::refresh_active_opt()
{
    switch (block()) {
        case BlockId::OPTIMIZER:
        case BlockId::STRATEGY:
        case BlockId::REFINE:  /* PRESTAGE: REFINE is the OUTER spec */
            active_opt = &t.spec;
            break;
        case BlockId::EXPLORE:  /* PRESTAGE's one-pass exploration */
            active_opt = t.spec.prestage.get();
            break;
        case BlockId::INNER:
        case BlockId::INIT:
        case BlockId::NOISE:
        case BlockId::RESTART:
        case BlockId::ROOT:
            active_opt = nullptr;
            break;
    }
}

std::runtime_error Parser::refuse_in_block(const std::string& key) const
{
    const OpenBlock& in = open_blocks.back();
    const R::Block& b = block_entry(in.id);
    if (is_top_level_key(key))
        return std::runtime_error(fill(R::structure::root_key_in_block_msg,
                                       {{"key", key},
                                        {"name", b.name},
                                        {"open_line", num(in.line)},
                                        {"end", b.end}}));
    std::string keys;
    for (R::KeyId id : b.keys)
        keys += (keys.empty() ? "" : "/") + str(R::key_entry(id).name);
    return std::runtime_error(fill(b.unknown_msg, {{"key", key},
                                                   {"open_line", num(in.line)},
                                                   {"keys", keys}}));
}

void Parser::set_once(const void* target, const std::string& attr,
                      std::string_view what)
{
    const BlockId blk = block();
    const BlockId around = parent();
    const auto [it, fresh] = attr_set.emplace(std::make_pair(target, attr),
                                              AttrSet{nline, blk, around});
    if (fresh) return;
    const AttrSet& first = it->second;
    if (blk == BlockId::ROOT)
        throw std::logic_error("hpotarget parser: a top-level key is set "
                               "once by its repeat rule");
    if (first.block == blk && first.parent == around)
        throw std::runtime_error(attr + " is set twice in the " +
                                 block_name(blk) + " block (first on line " +
                                 num(first.line) +
                                 "); each key is given once");
    if (first.block == blk) {
        /* Blocks of one type inside two blocks: the INIT blocks inside
         * OPTIMIZER and inside STRATEGY. */
        if (blk != BlockId::INIT)
            throw std::logic_error("hpotarget parser: two " +
                                   block_name(blk) + " blocks inside two "
                                   "blocks set " + attr);
        throw std::runtime_error(fill(
            rule("RULE.set_once_for_starting_points").msg,
            {{"key", attr},
             {"first_line", num(first.line)},
             {"parent", block_name(first.parent)},
             {"enclosing", block_name(around)}}));
    }
    throw std::runtime_error(fill(rule("RULE.set_once_for_optimiser").msg,
                                  {{"key", attr},
                                   {"first_line", num(first.line)},
                                   {"open", block_name(first.block)},
                                   {"block", block_name(blk)},
                                   {"what", what}}));
}

void Parser::line_reads_setting(const char* name, bool uses)
{
    if (!uses) return;
    if (const auto it = line_settings.find(name); it != line_settings.end())
        it->second.read = true;
}

std::string Parser::maxtime_source(const LammpsSim& sim) const
{
    if (!sim.reads_default_maxtime) return "its --maxtime";
    const auto it = line_settings.find("PC_SCRIPT_TIMEOUT");
    if (it == line_settings.end()) return "the default";
    return it->second.spelling + " on line " + num(it->second.line);
}

/* ---------------------------------------------------------- lines ---- */

void Parser::run()
{
    t.target_file = fn;
    std::ifstream ifs(fn);
    if (!ifs)
        throw std::runtime_error(
            fill(R::lexing::missing_file_msg, {{"file", fn}}));
    const char comment = R::lexing::comment.at(0);
    const char continuation = R::lexing::continuation.at(0);
    const auto strip = [comment](std::string& s) {
        const auto p = s.find(comment);
        if (p != std::string::npos) s.resize(p);
        s.erase(std::find_if(s.rbegin(), s.rend(),
                             [](int c) { return !std::isspace(c); })
                    .base(),
                s.end());
    };
    while (std::getline(ifs, line)) {
        ++nline;
        strip(line);
        if (line.empty()) continue;
        while (!line.empty() && line.back() == continuation) {
            line.pop_back();
            std::string cont;
            /* A blank or comment line after the continuation is joined:
             * only the end of the file leaves the line without a next
             * one. */
            if (!std::getline(ifs, cont))
                throw std::runtime_error(config_line_error(
                    fn, nline, str(R::lexing::unfinished_msg),
                    line + str(R::lexing::continuation)));
            ++nline;
            strip(cont);
            line += cont;
        }
        try {
            read_line();
        } catch (const EarlierLineError& e) {
            throw std::runtime_error(
                config_line_error(fn, e.line, e.what(), e.text));
        } catch (const std::exception& e) {
            throw std::runtime_error(
                config_line_error(fn, nline, e.what(), line));
        }
    }
    end_of_file();
}

void Parser::read_line()
{
    std::istringstream iss(line);
    std::string key;
    iss >> key;
    if (key.empty()) return;
    for (const R::CheckStage stage : R::structure::check_order) {
        switch (stage) {
            case R::CheckStage::BlockLines:
                if (block_lines(key, iss)) return;
                break;
            case R::CheckStage::Removed:
                for (const R::Removed& r : R::kRemoved)
                    if (r.name == key)
                        throw std::runtime_error(fill(
                            R::structure::removed_msg,
                            {{"key", key}, {"hint", r.hint}}));
                break;
            case R::CheckStage::InnerRefusal:
                if (block() == BlockId::INNER) {
                    const R::Block& b = block_entry(BlockId::INNER);
                    for (const R::Refusal& r : b.refuse)
                        if (r.key == key)
                            throw std::runtime_error(fill(
                                b.refuse_msg,
                                {{"key", key},
                                 {"why", r.why},
                                 {"enclosing", inner_enclosing()}}));
                }
                break;
            case R::CheckStage::GroupKeys:
                if (active_opt) {
                    if (const R::Key* k = group_key(key)) {
                        key_line(*k, key, iss, Place::Group);
                        return;
                    }
                }
                break;
            case R::CheckStage::InitRestartKeys:
            case R::CheckStage::NoiseKeys:
            case R::CheckStage::StrategyKeys: {
                const BlockId stage_block[] = {BlockId::INIT,
                                               BlockId::RESTART};
                bool here = false;
                if (stage == R::CheckStage::InitRestartKeys)
                    here = block() == stage_block[0] ||
                           block() == stage_block[1];
                else if (stage == R::CheckStage::NoiseKeys)
                    here = block() == BlockId::NOISE;
                else
                    here = block() == BlockId::STRATEGY;
                if (!here) break;
                if (const R::Key* k = block_key(block_entry(block()), key)) {
                    /* The keys of one STRATEGY TYPE are checked against the
                     * block's TYPE when it ends (TYPE may follow them). */
                    for (const R::StrategyType& st : R::kStrategyTypes)
                        for (R::KeyId id : st.keys)
                            if (id == k->id &&
                                (type_keys.empty() ||
                                 type_keys.back().line != nline))
                                type_keys.push_back({key, nline, line});
                    key_line(*k, key, iss, Place::Block);
                    return;
                }
                throw refuse_in_block(key);
            }
            case R::CheckStage::LogHp: {
                const R::Key& k = R::key_entry(R::KeyId::OPTIMIZER_LOG_HP);
                if (key != k.name) break;
                if (k.blocks & R::block_bit(block())) {
                    key_line(k, key, iss, Place::Block);
                    return;
                }
                const std::string note =
                    block() == BlockId::ROOT
                        ? std::string(" and is not a top-level key")
                        : ", and the " + block_name(block()) +
                              " block does not take it";
                throw std::runtime_error(
                    fill(k.misplaced_msg, {{"note", note}}));
            }
            case R::CheckStage::BlockUnknown:
                if (block() != BlockId::ROOT) throw refuse_in_block(key);
                break;
            case R::CheckStage::RootKeys:
                if (const R::Key* k = R::find_key(key, BlockId::ROOT)) {
                    key_line(*k, key, iss, Place::Root);
                    return;
                }
                throw std::runtime_error(
                    fill(R::structure::unknown_root_msg, {{"key", key}}));
        }
    }
}

/* ---------------------------------------------------- block lines ---- */

bool Parser::block_lines(const std::string& key, std::istringstream& iss)
{
    if (const R::Block* b = block_ended_by(key)) {
        end_line(*b, key, iss);
        return true;
    }
    if (const R::Block* b = R::find_block(key)) return open_line(*b, key, iss);
    return false;
}

void Parser::end_line(const R::Block& b, const std::string& key,
                      std::istringstream& iss)
{
    for (const R::CloseStep step : R::structure::close_order) {
        switch (step) {
            case R::CloseStep::Match: {
                if (open_blocks.empty())
                    throw std::runtime_error(
                        fill(R::structure::end_without_open_msg,
                             {{"end", key}, {"name", b.name}}));
                const OpenBlock& in = open_blocks.back();
                if (in.id != b.id)
                    throw std::runtime_error(
                        fill(R::structure::end_mismatch_msg,
                             {{"end", key},
                              {"open", block_name(in.id)},
                              {"open_line", num(in.line)}}));
                break;
            }
            case R::CloseStep::KeyAlone: {
                std::string extra;
                if (iss >> extra)
                    throw std::runtime_error(
                        fill(R::structure::end_takes_no_value_msg,
                             {{"end", key}, {"name", b.name},
                              {"tok", extra}}));
                break;
            }
            case R::CloseStep::CloseSteps:
                /* The END line of a block of an optimiser completes it:
                 * the startup refusals of its settings name this line
                 * (OptimizerSpec::block_line); the last ENDRESTART line
                 * names a RESTART block that never restarts. */
                if (b.id == BlockId::OPTIMIZER || b.id == BlockId::EXPLORE ||
                    b.id == BlockId::INNER) {
                    OptimizerSpec& s = b.id == BlockId::OPTIMIZER ? t.spec
                                       : b.id == BlockId::EXPLORE
                                           ? *t.spec.prestage
                                           : *active_opt;
                    s.source_file = fn;
                    s.block_line = nline;
                    s.block_line_text = line;
                } else if (b.id == BlockId::RESTART) {
                    endrestart_line = nline;
                    endrestart_text = line;
                }
                for (const R::BlockCloseStep s : b.close_steps) close_step(s);
                break;
            case R::CloseStep::Close:
                open_blocks.pop_back();
                if (b.id == BlockId::INIT || b.id == BlockId::RESTART)
                    active_init = nullptr;
                inner_parent = nullptr;
                refresh_active_opt();
                break;
        }
    }
}

void Parser::close_step(R::BlockCloseStep step)
{
    switch (step) {
        case R::BlockCloseStep::OptimizerDriver:
            require_driver(t.spec, DriverPhase::Run);
            break;
        case R::BlockCloseStep::ExploreDriver:
            require_driver(*t.spec.prestage, DriverPhase::Explore);
            break;
        case R::BlockCloseStep::InnerDriver:
            require_inner_driver(*active_opt, inner_enclosing());
            break;
        case R::BlockCloseStep::StrategyTypeKeys:
            /* The keys of one TYPE must match the TYPE of the block. */
            for (const auto& k : type_keys) {
                try {
                    require_strategy_key(k.key, t.spec.strategy_type,
                                         strategy_type_given);
                } catch (const std::exception& e) {
                    throw EarlierLineError(k.line, k.text, e.what());
                }
            }
            type_keys.clear();
            break;
        case R::BlockCloseStep::MultistartNStarts:
            /* N_STARTS 1 is refused at its line; without N_STARTS a
             * MULTISTART would run one start, which is TYPE PLAIN. */
            if (t.spec.strategy_type == StrategyType::MULTISTART &&
                !n_starts_given)
                throw EarlierLineError(
                    type_line, type_text,
                    str(R::key_entry(R::KeyId::STRATEGY_TYPE).checks[0]));
            break;
        case R::BlockCloseStep::RestartRestarts:
            /* A restart comes after AFTER_STAGNATION evaluations without a
             * new best, at most MAX times: a block without one of them
             * would restart nothing (0 is refused at its line). */
            if (endrestart_line != 0 && !t.spec.restart.active()) {
                const bool no_after = t.spec.restart.after_stagnation == 0;
                const bool no_max = t.spec.restart.max == 0;
                const std::string missing =
                    no_after && no_max
                        ? "no AFTER_STAGNATION line and no MAX line"
                    : no_after ? "no AFTER_STAGNATION line"
                               : "no MAX line";
                throw EarlierLineError(
                    endrestart_line, endrestart_text,
                    fill(block_entry(BlockId::RESTART).close_checks[0],
                         {{"missing", missing}}));
            }
            break;
    }
}

bool Parser::open_line(const R::Block& b, const std::string& key,
                       std::istringstream& iss)
{
    for (const R::OpenStep step : R::structure::open_order) {
        switch (step) {
            case R::OpenStep::Alias:
                /* A word the enclosing block reads as one of its keys
                 * (STRATEGY inside INIT and RESTART is METHOD). */
                if (block() != BlockId::ROOT)
                    for (const R::Alias& a : block_entry(block()).aliases)
                        if (a.word == key) {
                            key_line(R::key_entry(a.key), key, iss,
                                     Place::Block);
                            return true;
                        }
                break;
            case R::OpenStep::Parents:
                if (!(b.parents & R::block_bit(block()))) {
                    const std::string around =
                        open_blocks.empty() ? std::string()
                                            : block_name(block());
                    const std::string around_line =
                        open_blocks.empty()
                            ? std::string()
                            : num(open_blocks.back().line);
                    throw std::runtime_error(
                        fill(b.misplaced_msg, {{"parent", around},
                                               {"parent_line", around_line}}));
                }
                /* A block of an optimiser's settings opens where one is
                 * set (no file reaches the message). */
                if ((b.id == BlockId::INIT || b.id == BlockId::INNER) &&
                    !active_opt)
                    throw std::logic_error(str(b.unreachable_msgs[0]));
                break;
            case R::OpenStep::Requires:
                /* EXPLORE and REFINE: the TYPE line comes before them. */
                if (!b.requires_checks.empty() &&
                    t.spec.strategy_type != StrategyType::PRESTAGE)
                    throw std::runtime_error(str(b.requires_checks[0]));
                break;
            case R::OpenStep::Repeat: {
                std::size_t first = 0;
                if (b.repeat == R::BlockRepeat::Error)
                    first = b.id == BlockId::OPTIMIZER ? optimizer_line
                                                       : strategy_line;
                else if (b.repeat == R::BlockRepeat::ErrorPerParent &&
                         active_opt->inner)
                    first = inner_lines[active_opt];
                if (first != 0 ||
                    (b.repeat == R::BlockRepeat::ErrorPerParent &&
                     active_opt->inner))
                    throw std::runtime_error(
                        fill(b.repeat_msg, {{"first_line", num(first)}}));
                break;
            }
            case R::OpenStep::KeyAlone: {
                std::string extra;
                if (iss >> extra)
                    throw std::runtime_error(
                        fill(R::structure::open_takes_no_value_msg,
                             {{"name", b.name}, {"tok", extra}}));
                break;
            }
            case R::OpenStep::Open:
                switch (b.id) {
                    case BlockId::OPTIMIZER:
                        optimizer_line = nline;
                        break;
                    case BlockId::STRATEGY:
                        strategy_line = nline;
                        break;
                    case BlockId::INIT:
                        active_init = &active_opt->init_strategy_spec;
                        /* The block's line: its first INIT block when it
                         * has two (they make one chain). */
                        if (active_init->line == 0) {
                            active_init->line = nline;
                            active_init->line_text = line;
                        }
                        inner_parent = active_opt;
                        break;
                    case BlockId::EXPLORE:
                        if (!t.spec.prestage)
                            t.spec.prestage = std::make_unique<OptimizerSpec>();
                        break;
                    case BlockId::INNER:
                        inner_lines[active_opt] = nline;
                        active_opt->inner = std::make_unique<OptimizerSpec>();
                        inner_parent = active_opt;
                        break;
                    case BlockId::RESTART:
                        active_init = &t.spec.restart.init_strategy;
                        if (t.spec.restart.line == 0) {
                            t.spec.source_file = fn;
                            t.spec.restart.line = nline;
                            t.spec.restart.line_text = line;
                        }
                        inner_parent = active_opt;
                        break;
                    case BlockId::NOISE:
                    case BlockId::REFINE:
                    case BlockId::ROOT:
                        break;
                }
                open_blocks.push_back({b.id, nline, line});
                if (b.id == BlockId::INNER)
                    active_opt = inner_parent->inner.get();
                else
                    refresh_active_opt();
                break;
        }
    }
    return true;
}

/* ------------------------------------------------------ key lines ---- */

void Parser::key_line(const R::Key& k, const std::string& word,
                      std::istringstream& iss, Place place)
{
    R::Span<R::LineStep> steps = k.steps;
    if (steps.empty())
        steps = place == Place::Root    ? R::structure::line_steps_root
                : place == Place::Group ? R::structure::line_steps_group
                                        : R::structure::line_steps_block;
    R::KeyRead r{k, word, iss};
    r.parser = this;
    if (place == Place::Group) r.spec = active_opt;
    /* What a line of the group sets once: its key, PARAM <name> for a
     * PARAM line, none for a STEP <KEY> override (named once it is read). */
    std::string attr = word;
    if (place == Place::Group && (word == "PARAM" || word == "STEP")) {
        std::istringstream peek(line);
        std::string k0, first;
        peek >> k0 >> first;
        if (word == "PARAM")
            attr = "PARAM " + first;
        else if (!first.empty() && !is_number_shaped(first))
            attr.clear();
    }
    for (const R::LineStep step : steps) {
        switch (step) {
            case R::LineStep::Value:
                r.step = R::LineStep::Value;
                R::call(k, r);
                if (place == Place::Group) group_after_value(r, attr);
                break;
            case R::LineStep::Options:
                r.step = R::LineStep::Options;
                R::call(k, r);
                break;
            case R::LineStep::Repeat:
                repeat_step(r, place, attr);
                break;
            case R::LineStep::Leftover: {
                std::string extra;
                for (const R::Group& g : R::kGroups)
                    if (g.name == k.group && (iss >> extra))
                        throw std::runtime_error(
                            fill(g.leftover_msg,
                                 {{"key", word}, {"tok", extra}}));
                break;
            }
        }
    }
}

void Parser::group_after_value(R::KeyRead& r, std::string& attr)
{
    /* The line of a key whose refusal is about it alone
     * (OptimizerSpec::key_lines, with_key_line); a STEP <KEY> override
     * keeps its own (StepOverride::line). */
    const bool override_line = attr.empty() && r.word == "STEP" &&
                               !active_opt->step_overrides.empty();
    if (override_line) {
        StepOverride& ov = active_opt->step_overrides.back();
        ov.line = nline;
        ov.line_text = line;
    } else if (r.word == "LIB" || r.word == "ALGO" || r.word == "STEP" ||
               r.word == "GRAD_STRATEGY") {
        active_opt->source_file = fn;
        active_opt->key_lines[r.word] = KeyLine{nline, line,
                                                block_name(block())};
    }
    /* A STEP <KEY> override: once for its key and indices ("WATOM(1)" and
     * "WATOM (1)" are one). */
    if (override_line) {
        const StepOverride& ov = active_opt->step_overrides.back();
        std::set<std::size_t> idx(ov.indices.begin(), ov.indices.end());
        attr = "STEP " + ov.key;
        if (!idx.empty()) {
            std::string list;
            for (const std::size_t i : idx)
                list += (list.empty() ? "" : ",") + std::to_string(i);
            attr += "(" + list + ")";
        }
    }
}

void Parser::repeat_step(R::KeyRead& r, Place place, const std::string& attr)
{
    const R::Key& k = r.key;
    if (place == Place::Group) {
        /* A key of the group is set once for its optimiser, a PARAM name
         * and a STEP override for their name and indices. */
        if (!attr.empty())
            set_once(active_opt, attr,
                     rule_what("RULE.set_once_for_optimiser",
                               "GROUP." + str(k.group)));
        return;
    }
    if (place == Place::Block) {
        if (k.repeat == R::Repeat::Error) set_once(r.target, attr, r.what);
        return;
    }
    switch (k.repeat) {
        case R::Repeat::Error: {
            const auto [it, fresh] = root_lines.emplace(k.id, nline);
            if (!fresh)
                throw std::runtime_error(
                    fill(k.repeat_msg, {{"first_line", num(it->second)}}));
            break;
        }
        case R::Repeat::ErrorUntilRead: {
            /* A line after it reads it; a second line is read only after a
             * line has read the first. */
            const std::string name =
                k.has_alias ? str(R::key_entry(k.alias_of).name) : str(k.name);
            const auto it = line_settings.find(name);
            if (it != line_settings.end() && !it->second.read)
                throw std::runtime_error(
                    fill(rule("RULE.settings_taken_by_lines").msg,
                         {{"key", r.word},
                          {"first_line", num(it->second.line)},
                          {"note", it->second.spelling != r.word
                                       ? ", as " + it->second.spelling
                                       : std::string()},
                          {"list", readers_of(name)}}));
            line_settings[name] = {nline, line, r.word, false};
            break;
        }
        case R::Repeat::Add:
        case R::Repeat::LastWins:
        case R::Repeat::WarnLastWins:
            break;
    }
}

/* ---------------------------------------------------- end of file ---- */

void Parser::end_of_file()
{
    for (const R::EofStep step : R::structure::eof_order) eof_step(step);
}

void Parser::eof_step(R::EofStep step)
{
    const auto at_line = [this](std::size_t n, const std::string& what,
                                const std::string& text) {
        return std::runtime_error(config_line_error(fn, n, what, text));
    };
    switch (step) {
    case R::EofStep::OpenBlock:
        /* A block the file leaves open: at the line that opens it, the
         * innermost first. */
        if (!open_blocks.empty()) {
            const OpenBlock& in = open_blocks.back();
            const R::Block& b = block_entry(in.id);
            throw at_line(in.line,
                          fill(R::structure::unclosed_msg,
                               {{"name", b.name}, {"end", b.end}}),
                          in.text);
        }
        break;
    case R::EofStep::KTotalCaps: {
        /* K_TOTAL caps the starting points the METHOD lines of its chain
         * give (chain_points); one at or above their number caps nothing.
         * An EXPLORE chain without a METHOD line is not resolved. */
        std::vector<std::pair<std::pair<std::size_t, std::string>,
                              const InitStrategySpec*>> k_totals;
        for (const auto& [init, at] : k_total_lines)
            k_totals.push_back({at, init});
        std::sort(k_totals.begin(), k_totals.end(),
                  [](const auto& a, const auto& b) {
                      return a.first.first < b.first.first;
                  });
        for (const auto& [at, init] : k_totals) {
            if (!init->k_total) continue;
            const ChainPoints count = chain_points(*init);
            const std::size_t points = count.distinct;
            const std::string list =
                count.lines + (count.notes.empty() ? "" : "; " + count.notes);
            const std::size_t cap = *init->k_total;
            const std::string head =
                "K_TOTAL " + std::to_string(cap) + " caps nothing: ";
            const std::string keeps =
                ", and K_TOTAL keeps the first K_TOTAL starting points";
            const bool explore =
                t.spec.prestage && init == &t.spec.prestage->init_strategy_spec;
            if (init->chain.empty() && explore)
                throw at_line(at.first,
                    head + "an INIT block inside EXPLORE without a METHOD "
                    "line gives the exploration the starting points of the "
                    "INIT blocks inside OPTIMIZER, STRATEGY and REFINE (the "
                    "CONFIG point without one), and K_TOTAL caps only the "
                    "points of the block's own METHOD lines", at.second);
            if (init->chain.empty())
                throw at_line(at.first,
                    head + "a block without a METHOD line gives one starting "
                    "point (the CONFIG point)" + keeps, at.second);
            const std::string blocks =
                init == &t.spec.restart.init_strategy
                    ? "the RESTART blocks"
                : explore ? "the INIT blocks inside EXPLORE"
                          : "the INIT blocks inside OPTIMIZER, STRATEGY and "
                            "REFINE";
            if (cap >= points)
                throw at_line(at.first,
                    head + "the METHOD lines of " + blocks + " give at most " +
                        std::to_string(points) +
                        (points == 1 ? " starting point ("
                                     : " starting points (") +
                        list + ")" + keeps,
                    at.second);
        }
        break;
    }
    case R::EofStep::ScriptTimeoutUnread:
        /* A PC_SCRIPT_TIMEOUT (MAXTIME) that no PC_LAMMPS or PC_FAILPROBE
         * line without --maxtime after it reads would have no effect. */
        if (const auto it = line_settings.find("PC_SCRIPT_TIMEOUT");
            it != line_settings.end() && !it->second.read)
            throw at_line(it->second.line,
                          fill(R::key_entry(R::KeyId::ROOT_PC_SCRIPT_TIMEOUT)
                                   .checks[0],
                               {{"key", it->second.spelling}}),
                          it->second.text);
        break;
    case R::EofStep::LossUnread: {
        /* A LOSS that no term without a loss function of its own after it
         * reads, in a file without a metric line, would have no effect. */
        const bool has_metric_lines = t.use_emae || t.use_ermse ||
                                      t.use_errmse || t.use_ersq ||
                                      t.use_frmse || t.use_srmse;
        if (const auto it = line_settings.find("LOSS");
            it != line_settings.end() && !it->second.read &&
            !has_metric_lines)
            throw at_line(it->second.line,
                "LOSS sets the loss function of the --varloss and "
                "PC_OBSERVABLE terms after it that name none, and of the "
                "metric lines (PC_EMAE, PC_ERMSE, PC_ErRMSE, PC_ERSQ, "
                "PC_FRMSE, PC_SRMSE_GPA), and the file has no metric line and "
                "no such term after it; it would have no effect",
                it->second.text);
        break;
    }
    case R::EofStep::MetricTargetLoss: {
        /* A target the loss function of the metric lines -- the last LOSS
         * line -- cannot measure, refused at the metric's line, the first
         * in the file. */
        struct MetricTarget {
            std::size_t line;
            std::string text;
            std::string what;
            double target;
        };
        std::vector<MetricTarget> targets;
        const auto add_metric = [&](bool used, const char* key,
                                    double target) {
            if (!used) return;
            const auto it = metric_lines.find(key);
            if (it != metric_lines.end())
                targets.push_back(
                    {it->second.first, it->second.second, key, target});
        };
        add_metric(t.use_emae, "PC_EMAE", t.emae);
        add_metric(t.use_errmse, "PC_ErRMSE", t.errmse);
        add_metric(t.use_ersq, "PC_ERSQ", t.ersq);
        const auto add_dataset_metric = [&](const ValidationMetricTargets& m,
                                            const char* key) {
            if (const auto& g = m.global_assignment())
                targets.push_back({g->line, g->text, key, g->target});
            for (const auto& [index, a] : m.indexed_assignments())
                targets.push_back({a.line, a.text,
                                   std::string(key) + " (" +
                                       std::to_string(index) + ")",
                                   a.target});
        };
        add_dataset_metric(t.ermse_targets, "PC_ERMSE");
        add_dataset_metric(t.frmse_targets, "PC_FRMSE");
        add_dataset_metric(t.srmse_targets, "PC_SRMSE_GPA");
        std::stable_sort(targets.begin(), targets.end(),
                         [](const MetricTarget& a, const MetricTarget& b) {
                             return a.line < b.line;
                         });
        for (const auto& m : targets) {
            const std::string why = t.globalLoss.target_refusal(m.target);
            if (why.empty()) continue;
            std::ostringstream what;
            what << m.what << " " << m.target << ": the metric lines take "
                 << "the loss function of the last LOSS line (line "
                 << loss_line << "): " << why;
            throw at_line(m.line, what.str(), m.text);
        }
        break;
    }
    case R::EofStep::DensityKeyArming: {
        /* A density gate that no OPTIM line arms cannot act; refused at the
         * gate's line, the first in the file. */
        bool armed = false;
        for (const auto& kv : t.optimap)
            if (kv.first.find(".DENSITY.") != std::string::npos)
                for (bool selected : kv.second.opt)
                    armed = armed || selected;
        for (const auto& k : ParsedTarget::density_gate_keys()) {
            const auto it = t.optimap.find(k);
            if (it == t.optimap.end()) continue;
            for (bool b : it->second.opt) armed = armed || b;
        }
        if (!armed && !gate_lines.empty()) {
            const GateLine& g = gate_lines.front();
            std::string keys;
            const auto& dk = ParsedTarget::density_gate_keys();
            for (std::size_t i = 0; i < dk.size(); ++i)
                keys += (i == 0 ? "" : i + 1 == dk.size() ? " or " : ", ") +
                        dk[i];
            throw at_line(g.line,
                          fill(rule("RULE.density_key_arming").msg,
                               {{"key", g.key}, {"keys", keys}}),
                          g.text);
        }
        break;
    }
    case R::EofStep::ArchiveDirectories:
        /* The DUMP and BDUMP directories, together. */
        t.check_archive_directories();
        break;
    case R::EofStep::OptimizerRequired:
        /* A file without an OPTIMIZER block names no optimiser. */
        require_driver(t.spec, DriverPhase::Run);
        break;
    case R::EofStep::ObservableWithScripts:
        /* PC_OBSERVABLE lines do not share a target file with PC_LAMMPS or
         * PC_FAILPROBE lines: refused at the first line of the group that
         * comes later in the file, naming the lines of the other group. */
        if (!t.observables.empty() && !t.LS.empty()) {
            struct Group {
                std::string key;
                std::size_t n = 0, first = 0;
            };
            auto describe = [](const Group& g) {
                return core::to_string(g.n) + " " + g.key +
                       (g.n == 1 ? " line (line " : " lines (the first on line ") +
                       core::to_string(g.first) + ")";
            };
            auto and_list = [](const std::vector<std::string>& items) {
                std::string out;
                for (std::size_t i = 0; i < items.size(); ++i)
                    out += (i == 0 ? "" : i + 1 == items.size()
                                              ? (items.size() == 2 ? " and "
                                                                   : ", and ")
                                              : ", ") + items[i];
                return out;
            };
            Group obs{"PC_OBSERVABLE"}, lammps{"PC_LAMMPS"},
                probe{"PC_FAILPROBE"};
            for (const auto& l : t.observable_lines)
                if (obs.n++ == 0) obs.first = l.number;
            const LammpsSim* first_script = nullptr;
            for (const auto& sim : t.LS) {
                Group& g = sim.is_failprobe ? probe : lammps;
                if (g.n++ == 0) g.first = sim.line;
                if (!first_script || sim.line < first_script->line)
                    first_script = &sim;
            }
            const R::Rule& r = rule("RULE.PC_OBSERVABLE_excludes_PC_LAMMPS");
            if (first_script->line > obs.first)
                throw at_line(first_script->line,
                              fill(r.msg, {{"key", first_script->is_failprobe
                                                       ? "PC_FAILPROBE"
                                                       : "PC_LAMMPS"},
                                           {"list", describe(obs)}}),
                              first_script->line_text);
            std::vector<std::string> others;
            for (const Group* g : {&lammps, &probe})
                if (g->n) others.push_back(describe(*g));
            throw at_line(t.observable_lines.front().number,
                          fill(r.msg, {{"key", "PC_OBSERVABLE"},
                                       {"list", and_list(others)}}),
                          t.observable_lines.front().text);
        }
        break;
    case R::EofStep::HotvarInjected:
        /* A LAMMPS variable a line without a --hotvar defines on LAMMPS'
         * command line, which its script reads and which another line
         * declares with --hotvar: refused at the line that defines it. */
        for (const auto& cold : t.LS) {
            if (cold.is_failprobe || !cold.hotvars.empty()) continue;
            for (const auto& v : cold.line_variables)
                for (const auto& hot : t.LS) {
                    if (&hot == &cold) continue;
                    for (const auto& hv : hot.hotvars) {
                        if (hv.name != v.name ||
                            !script_reads_variable(cold.script_lines, v.name))
                            continue;
                        throw at_line(cold.line,
                            "PC_LAMMPS: the script " + cold.script +
                                " reads " + v.name + ", the --hotvar of line " +
                                std::to_string(hot.line) +
                                ", and this line defines " + v.name + " (" +
                                v.option + " " + v.name +
                                "): the run injects `variable " + v.name +
                                " equal <value>` into the script, which "
                                "LAMMPS refuses for a variable of another "
                                "style",
                            cold.line_text);
                    }
                }
        }
        break;
    case R::EofStep::HotvarUnique: {
        /* One line declares each --hotvar name; refused at the later line. */
        std::unordered_map<std::string, std::size_t> hot_owner;
        for (std::size_t s = 0; s < t.LS.size(); ++s) {
            for (const auto& hv : t.LS[s].hotvars) {
                auto it = hot_owner.find(hv.name);
                if (it != hot_owner.end() && it->second != s) {
                    const LammpsSim& first = t.LS[it->second];
                    throw at_line(t.LS[s].line,
                                  fill(rule("RULE.PC_LAMMPS_hotvar_unique").msg,
                                       {{"name", hv.name},
                                        {"line", core::to_string(first.line)},
                                        {"script", first.script}}),
                                  t.LS[s].line_text);
                }
                hot_owner.emplace(hv.name, s);
            }
        }
        break;
    }
    case R::EofStep::ObservableNumerical:
    case R::EofStep::ObservableAnalytical:
    case R::EofStep::ScriptGradients: {
        /* Every block that configures an optimiser, each named as the
         * startup checks name its phase (run_phase_name()). */
        struct OptimiserBlock {
            OptimizerSpec* sp;
            std::string phase;
            bool inner;
        };
        const std::string run_phase = run_phase_name(t.spec);
        const std::string explore_phase =
            "the PRESTAGE exploration (EXPLORE block)";
        std::vector<OptimiserBlock> blocks = {{&t.spec, run_phase, false}};
        if (t.spec.inner)
            blocks.push_back(
                {t.spec.inner.get(), "the INNER block of " + run_phase, true});
        if (t.spec.prestage) {
            blocks.push_back({t.spec.prestage.get(), explore_phase, false});
            if (t.spec.prestage->inner)
                blocks.push_back({t.spec.prestage->inner.get(),
                                  "the INNER block of " + explore_phase, true});
        }
        if (step == R::EofStep::ObservableNumerical) {
            /* GRAD_STRATEGY NUMERICAL with PC_OBSERVABLE lines. */
            if (t.observables.empty()) break;
            std::string observable_line = "PC_OBSERVABLE";
            if (!t.observable_lines.empty() &&
                t.observable_lines.front().number > 0)
                observable_line +=
                    " on line " +
                    std::to_string(t.observable_lines.front().number);
            for (const auto& b : blocks)
                if (b.sp->grad_strategy &&
                    *b.sp->grad_strategy == GradStrategy::Numerical)
                    throw std::runtime_error(with_key_line(*b.sp,
                        "GRAD_STRATEGY",
                        b.phase + ": GRAD_STRATEGY NUMERICAL is refused in a "
                        "target file with PC_OBSERVABLE lines (" +
                        observable_line + "): PC_OBSERVABLE terms are built "
                        "for the analytical gradient; GRAD_STRATEGY "
                        "ANALYTICAL takes it (an unset GRAD_STRATEGY resolves "
                        "to it in a block that obtains gradients; it needs "
                        "the settings of the analytical gradient, such as "
                        "OALGO 4 with a fixed LAMBDA >= 0, and startup names "
                        "a missing one), and GRAD_STRATEGY HYBRID takes "
                        "finite differences where the analytical gradient is "
                        "unavailable"));
            break;
        }
        if (step == R::EofStep::ObservableAnalytical) {
            /* An unset GRAD_STRATEGY resolves to ANALYTICAL in each block
             * that runs an optimiser and obtains gradients. */
            if (t.observables.empty()) break;
            for (const auto& b : blocks) {
                if (b.inner || b.sp->grad_strategy ||
                    (b.sp->inner && b.sp->inner->lib != "NLOPT") ||
                    !obtains_gradients(*b.sp))
                    continue;
                b.sp->grad_strategy = GradStrategy::Analytical;
                b.sp->grad_strategy_resolved = true;
            }
            break;
        }
        /* PC_LAMMPS: no gradients; a PC_FAILPROBE line runs a LAMMPS
         * script too. */
        if (t.LS.empty()) break;
        bool has_pc_lammps = false;
        for (std::size_t i = 0; i < t.LS.size(); ++i)
            if (static_cast<int>(i) != t.failprobe_idx) has_pc_lammps = true;
        std::string script_line = has_pc_lammps ? "PC_LAMMPS" : "PC_FAILPROBE";
        for (std::size_t i = 0; i < t.LS.size(); ++i) {
            if (has_pc_lammps == (static_cast<int>(i) == t.failprobe_idx))
                continue;
            if (t.LS[i].line > 0)
                script_line += " on line " + std::to_string(t.LS[i].line);
            break;
        }
        const std::string pc_file =
            "a target file with PC_LAMMPS or PC_FAILPROBE lines (" +
            script_line + ")";
        for (const auto& b : blocks) {
            const OptimizerSpec& sp = *b.sp;
            const std::string& phase = b.phase;
            const bool inner = b.inner;
            if (sp.grad_strategy)
                throw std::runtime_error(with_key_line(sp, "GRAD_STRATEGY",
                    phase + ": GRAD_STRATEGY is refused in " + pc_file +
                    ": tadah hpo takes no gradient (ANALYTICAL, NUMERICAL or "
                    "HYBRID) of a loss that runs LAMMPS scripts, and such a "
                    "file takes a gradient-free optimiser"));
            if (inner && sp.lib != "NLOPT") continue;
            if (sp.inner && sp.inner->lib != "NLOPT") continue;
            if (!obtains_gradients(sp)) continue;
            const std::string inner_free =
                " A target file with PC_LAMMPS or PC_FAILPROBE lines (" +
                script_line + ") takes a gradient-free optimiser, and an "
                "INNER block takes a local NLopt algorithm; the gradient-free "
                "ones are " + nlopt_inner_searches_without_gradient() + ".";
            if (optimizer_caps(sp).accepts_inner) {
                if (sp.inner) continue;
                const OptimizerSpec local = inner_run_spec(sp).value();
                const std::string head =
                    phase + ": " + sp.lib + ":" + sp.algo +
                    " takes a gradient: NLopt's MLSL and AUGLAG take one for "
                    "their local search, and NLopt's default local "
                    "optimiser, " + local.lib + ":" + local.algo +
                    ", is gradient-based";
                throw std::runtime_error(with_key_line(sp, "ALGO",
                    inner ? head + "." + inner_free
                          : head + "; with an INNER block that names a "
                                   "gradient-free local NLopt algorithm it "
                                   "takes none. A target file with PC_LAMMPS "
                                   "or PC_FAILPROBE lines (" + script_line +
                                   ") takes a gradient-free optimiser (" +
                                   searches_without_gradient() + ")."));
            }
            throw std::runtime_error(with_key_line(sp, "ALGO",
                inner ? phase + ": " + sp.lib + ":" + sp.algo +
                            " is gradient-based." + inner_free
                      : phase + ": " + sp.lib + ":" + sp.algo +
                            " is gradient-based, and " + pc_file +
                            " takes a gradient-free optimiser (" +
                            searches_without_gradient() + "); an NLopt MLSL "
                            "or AUGLAG algorithm is accepted with a "
                            "gradient-free INNER block."));
        }
        break;
    }
    }
}

void parse_target_file(ParsedTarget& target, const std::string& fn)
{
    Parser(target, fn).run();
}

}  // namespace hpotarget

namespace hpotarget_registry {

hpotarget::Parser& KeyRead::p() const
{
    if (!parser)
        throw std::logic_error("the handler of " + std::string(key.name) +
                               " reads a line of a target file, and was "
                               "called without its parser");
    return *parser;
}

}  // namespace hpotarget_registry
}  // namespace hpo
}  // namespace tadah
