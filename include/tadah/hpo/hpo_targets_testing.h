#pragma once
/* hpo_targets_testing.h -- test seam of the target-file parsers
 * (ParsedTarget::read and the registry-driven parser of
 * src/hpotarget_parser.cpp). Not part of the public API.
 *
 * make_targets() builds the targets of a run (HPOTargets) whose target
 * file the parser `parser` reads: Old, ParsedTarget::read (the parser of
 * `tadah hpo`), or New, the registry-driven parser; the rest of the
 * construction is the constructor's. So two parsers of one file are
 * compared on the targets they give, each on its own training
 * configuration (the parse inserts the defaults of the keys it reads).
 *
 * describe_parse() writes what the parse of a target file gave as a
 * canonical text, so two parses can be compared as text: the comparison of
 * two parsers of one file, and the expected results of a corpus of target
 * files. One line per setting, `<path> = <value>`, in a fixed order:
 *
 *   - a string is quoted, with \" \\ \n \r \t and \u00XX escapes;
 *   - a number is the shortest text that reads back as the same double
 *     (core::format_roundtrip: "1000", "0.25", "1e-20", "inf", "nan");
 *     a count or an index is an integer;
 *   - a flag is true or false, a value never set `none`, an enumerator its
 *     word (NUMERICAL, PRESTAGE, ...);
 *   - a list of plain values stands on one line, `[a, b]`; a list of
 *     records gives its `.size` and then each record, `<path>[i].<field>`;
 *     a map gives its `.size` and its entries in key order,
 *     `<path>[<key>]`;
 *   - a LAMMPS script is given by its number of lines and the FNV-1a hash
 *     of its lines (`.size`, `.fnv1a`), not line by line.
 *
 * The dump holds every member of ParsedTarget the parse sets, with the
 * OptimizerSpec of the run and its INNER and PRESTAGE specs, each PC_LAMMPS
 * and PC_FAILPROBE line, each PC_OBSERVABLE term with its options, the
 * boxes of the OPTIM lines (optimap) and the training configuration the
 * targets read (context[<KEY>]: read() inserts the defaults of the keys it
 * reads). It leaves out what the constructor derives once read() has
 * returned and what does not depend on the file: the search space and the
 * internal bounds, the starting points of the INIT and RESTART chains, the
 * copies of FAILSCORE and FAILSCORE_GRADING in the specs, the training
 * record and NUMERIC. A seed the constructor draws (an INIT block or an
 * optimiser without SEED) and the seed an exploration takes from the
 * STRATEGY-level SEED are shown as read() leaves them (`none`), and so is
 * the PARAM fd_step that NOISE AUTO derives (left out of `param`), so the
 * dump of one file does not change from one construction to the next.
 * The environment override TADAH_OPTIM_SCALE, which the constructor
 * applies, shows in scale_policy.
 *
 * Throws std::logic_error for a PC_OBSERVABLE kind it does not know, whose
 * options it could not show.
 *
 * read_shadowed() is the parser shadow: a validation build configured with
 * the CMake option TADAH_HPO_PARSER_SHADOW (OFF by default; HPO's
 * CMakeLists.txt) reads every target file with it, the parser the targets
 * name (ParsedTarget::read for `tadah hpo`) first and the other one second,
 * and stops at the first file the two read differently. The program's
 * build has no shadow. */

#include <tadah/hpo/hpo_targets.h>

#include <cstddef>
#include <functional>
#include <memory>
#include <string>

/* Defined: make_targets() reads a target file with the parser it names. */
#define TADAH_HPOTARGET_PARSER_CHOICE 1
/* Defined: read_shadowed(), the parser shadow, and its counter. */
#define TADAH_HPOTARGET_PARSER_SHADOW_SEAM 1

namespace tadah::hpo {
namespace testing {

/* The parsers of the target file. */
enum class TargetParser : unsigned char { Old, New };

/* A parser of the target file, which reads the file `fn` into the target,
 * and the name the shadow gives it. */
struct NamedParser {
    std::function<void(ParsedTarget& target, const std::string& fn)> read;
    std::string name;
};

/* The parser `parser` names: Old, ParsedTarget::read; New, the
 * registry-driven parser. */
NamedParser parser_of(TargetParser parser);

/* The targets of a run whose target file `parser` reads (in a build with
 * the shadow, compared with the other parser: read_shadowed()). */
std::unique_ptr<HPOTargets> make_targets(
    core::Context& c, const std::string& fn, TargetParser parser,
    TargetsPurpose purpose = TargetsPurpose::Run);

/* The targets of a run whose target file read_shadowed() reads with
 * `primary` and `shadow`, in any build. */
std::unique_ptr<HPOTargets> make_targets(
    core::Context& c, const std::string& fn, const NamedParser& primary,
    const NamedParser& shadow, TargetsPurpose purpose = TargetsPurpose::Run);

/* Reads the target file `fn` into `target` with `primary`, and with
 * `shadow` into a target of its own, built on a copy of the training
 * configuration and of the OPTIM reader of `target` as they stand before
 * the read (so neither parse sees what the other inserts). What each parser
 * writes to standard output and standard error is recorded; the primary's
 * reaches the streams as well, the shadow's does not.
 *   - The two read the file alike: returns, `target` holds what the
 *     primary read. Both refuse it alike: rethrows the primary's refusal.
 *   - Else throws std::logic_error naming the option
 *     TADAH_HPO_PARSER_SHADOW, the file, both parsers and the difference:
 *     one parser refuses the file and the other reads it ("<name>: refuses
 *     it: <what>", "<name>: reads the file"); the two refusals differ in
 *     their text or their type; the first line where the descriptions of
 *     the two parses (describe_parse) differ ("<name>: <line>"); what the
 *     two wrote to standard output or standard error.
 * Each call counts in shadow_reads(). */
void read_shadowed(ParsedTarget& target, const std::string& fn,
                   const NamedParser& primary, const NamedParser& shadow);

/* The number of reads read_shadowed() has compared in this process: in the
 * program's build it never runs, in a build with the shadow once for each
 * construction of HPOTargets. */
std::size_t shadow_reads();

std::string describe_parse(const ParsedTarget& targets);

}  // namespace testing
}  // namespace tadah::hpo
