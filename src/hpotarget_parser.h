#pragma once
/* hpotarget_parser.h -- the registry-driven parser of the target file.
 *
 * It reads a target file into a ParsedTarget (HPOTargets reads its target
 * file with it), driven by the generated tables of the registry (blueprint/README.md, "The generated tables"):
 *   - the lexer: comments, continued lines and the frame of a refusal
 *     ([lexing]);
 *   - each line tried in structure::check_order: the block lines (a line
 *     that opens a block in structure::open_order, an END line in
 *     close_order, with the checks of Block::close_steps), the removed
 *     keywords, the refusals of INNER, the keys of the optimiser group,
 *     the keys of INIT and RESTART, of NOISE, LOG_HP, the keys of
 *     STRATEGY, the unknown-key refusal of a block and the top-level keys;
 *   - the line of a key in the steps of its place (structure::line_steps_*,
 *     Key::steps): its repeat rule (Key::repeat), its value and its options
 *     (the key's handler, hpotarget_handlers.cpp) and the leftover check of
 *     the group;
 *   - the checks at the end of the file in structure::eof_order.
 * Private to HPO: the tests reach it through HPOTargets, whose constructor
 * calls parse_target_file(). */

#include "hpotarget_reads.h"

#include <tadah/hpo/hpo_targets.h>

#include <cstddef>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace tadah {
namespace hpo {
namespace hpotarget {

namespace R = hpotarget_registry;

/* Reads the target file `fn` into `target` (whose OPTIM reader is
 * prepared: ParsedTarget::prepare_optim_reader). Throws std::runtime_error
 * with the refusal of the first line or check that refuses the file. */
void parse_target_file(ParsedTarget& target, const std::string& fn);

/* A refusal of a line before the one being read (a STRATEGY key found at
 * the end of its block to have no effect): framed at that line. */
struct EarlierLineError : std::runtime_error {
    std::size_t line;
    std::string text;
    EarlierLineError(std::size_t l, std::string t, const std::string& what)
        : std::runtime_error(what), line(l), text(std::move(t)) {}
};

/* The state of the parse of one file, which the handlers read and set. */
class Parser {
public:
    Parser(ParsedTarget& target, const std::string& fn);
    /* Reads the file. */
    void run();

    ParsedTarget& t;
    const std::string& fn;
    /* The line being read: its number (the last physical line of a
     * continued one) and its text (the joined line). */
    std::size_t nline = 0;
    std::string line;

    /* The open blocks, outermost first, each with the line that opens it
     * and its text. */
    struct OpenBlock {
        R::BlockId id;
        std::size_t line;
        std::string text;
    };
    std::vector<OpenBlock> open_blocks;
    /* The innermost open block (ROOT when none), and the one around it. */
    R::BlockId block() const;
    R::BlockId parent() const;
    static const R::Block& block_entry(R::BlockId id);
    static std::string block_name(R::BlockId id);

    /* The optimiser the keys of the group set (none outside the blocks of
     * the group), the starting points the keys of INIT and RESTART set,
     * and the optimiser around an INIT, INNER or RESTART block. */
    OptimizerSpec* active_opt = nullptr;
    InitStrategySpec* active_init = nullptr;
    OptimizerSpec* inner_parent = nullptr;
    /* The line of the INNER block of each optimiser: one block each. */
    std::map<const OptimizerSpec*, std::size_t> inner_lines;
    /* The block an INNER block sits in: STRATEGY, EXPLORE or REFINE. */
    std::string inner_enclosing() const;

    /* The STRATEGY keys of one TYPE, checked when the STRATEGY block ends;
     * the TYPE line and whether the block has N_STARTS; the last
     * ENDRESTART line. */
    struct TypeKeySeen {
        std::string key;
        std::size_t line;
        std::string text;
    };
    std::vector<TypeKeySeen> type_keys;
    bool strategy_type_given = false;
    std::size_t type_line = 0;
    std::string type_text;
    bool n_starts_given = false;
    std::size_t endrestart_line = 0;
    std::string endrestart_text;
    /* The K_TOTAL line of each set of starting points. */
    std::map<const InitStrategySpec*, std::pair<std::size_t, std::string>>
        k_total_lines;
    /* The lines of the OPTIMIZER and STRATEGY blocks (0 before one). */
    std::size_t optimizer_line = 0;
    std::size_t strategy_line = 0;
    /* The line of each PC_EMAE, PC_ErRMSE and PC_ERSQ line. */
    std::map<std::string, std::pair<std::size_t, std::string>> metric_lines;
    /* The first line of each top-level key set once. */
    std::map<R::KeyId, std::size_t> root_lines;

    /* Where each key was set, per thing it configures (set_once). */
    struct AttrSet {
        std::size_t line;
        R::BlockId block;
        R::BlockId parent;
    };
    std::map<std::pair<const void*, std::string>, AttrSet> attr_set;
    /* Records that `attr` of `target` is set at this line, or refuses the
     * line when an earlier one set it; `what` names what two blocks of
     * different types that set it configure. */
    void set_once(const void* target, const std::string& attr,
                  std::string_view what);

    /* FAILSCORE, LOSS and PC_SCRIPT_TIMEOUT (MAXTIME): the last line of
     * each, its spelling and whether a line after it has read it. */
    struct LineSetting {
        std::size_t line;
        std::string text;
        std::string spelling;
        bool read;
    };
    std::map<std::string, LineSetting> line_settings;
    /* The line read now uses the setting `name` when `uses`. */
    void line_reads_setting(const char* name, bool uses);
    /* Where the script limit of a PC_LAMMPS or PC_FAILPROBE line comes
     * from. */
    std::string maxtime_source(const LammpsSim& sim) const;
    /* The line of the last LOSS line (0: none). */
    std::size_t loss_line = 0;

    /* The density gate lines, for the arming check at the end of the
     * file. */
    struct GateLine {
        std::string key;
        std::size_t line;
        std::string text;
    };
    std::vector<GateLine> gate_lines;
    /* What a gate line reads between its value and its options. */
    std::vector<std::string> gate_tokens;
    std::optional<RhoGateSpec> rhogate;
    std::optional<RhoBarGateSpec> rhobargate;
    /* What a metric line reads between its values and its options. */
    struct MetricLine {
        std::vector<std::size_t> indices;
        double target = 0.0;
        double weight = 0.0;
    } metric;

    /* A key the innermost open block does not take: a top-level key is
     * named as such, with the block that is still open; another key is
     * unknown there. */
    std::runtime_error refuse_in_block(const std::string& key) const;

private:
    void read_line();
    bool block_lines(const std::string& key, std::istringstream& iss);
    void end_line(const R::Block& b, const std::string& key,
                  std::istringstream& iss);
    bool open_line(const R::Block& b, const std::string& key,
                   std::istringstream& iss);
    void close_step(R::BlockCloseStep step);
    void refresh_active_opt();

    /* The place of a key's line, which gives its steps. */
    enum class Place { Root, Group, Block };
    void key_line(const R::Key& k, const std::string& word,
                  std::istringstream& iss, Place place);
    void repeat_step(R::KeyRead& r, Place place, const std::string& attr);
    void group_after_value(R::KeyRead& r, std::string& attr);

    void end_of_file();
    void eof_step(R::EofStep step);
};

}  // namespace hpotarget
}  // namespace hpo
}  // namespace tadah
