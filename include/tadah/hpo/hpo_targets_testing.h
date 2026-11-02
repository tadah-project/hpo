#pragma once
/* hpo_targets_testing.h -- test seam of the target-file parser
 * (HPOTargets::read). Not part of the public API.
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
 * The dump holds every member of HPOTargets that read() sets, with the
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
 * options it could not show. */

#include <string>

namespace tadah::hpo {

class HPOTargets;

namespace testing {

std::string describe_parse(const HPOTargets& targets);

}  // namespace testing
}  // namespace tadah::hpo
