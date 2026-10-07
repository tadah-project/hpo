#pragma once
/*
 * Strict parsing of the numbers in the hpotarget grammar (HPOTargets::read,
 * the optimiser attributes shared with `tadah refit`, the PC_LAMMPS
 * options). A key's value is one whole token that must represent what the
 * key holds; anything else is an error naming the key and the token, in
 * the form
 *     <what> must be an integer >= 1, got '<tok>'
 * HPOTargets::read() prefixes "Error on line N: " and appends the line.
 * `iss >> v` and std::stod accepted a prefix ("5x" -> 5), stored 0 for a
 * non-number, wrapped a negative into a huge unsigned value and left the
 * target untouched when the value was missing.
 *
 * Every real number of every key -- in the hpotarget file, on a PC_LAMMPS,
 * PC_FAILPROBE or PC_OBSERVABLE line, in a `tadah refit` stage block --
 * has one grammar, the decimal syntax LAMMPS' utils::is_double accepts
 * (parse_number). std::stod also read hexadecimal floats (0x1p4), inf and
 * nan, and skipped leading whitespace.
 */

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <sstream>
#include <string>

namespace tadah::hpo {

/* The form of every error HPOTargets::read() reports about a line of the
 * hpotarget file:
 *     Error on line <line>: <what>
 *     Line: <text>
 * The startup refusals of a phase's settings name the line its block
 * closes at in the same form (with_block_line, phase_specs.h). */
inline std::string config_line_error(std::size_t line, const std::string& what,
                                     const std::string& text)
{
    return "Error on line " + std::to_string(line) + ": " + what +
           "\nLine: " + text;
}

/* A non-negative integer: decimal digits only (no sign, fraction, exponent,
 * whitespace or trailing characters), within [min, max]. Otherwise throws
 * std::runtime_error:
 *     "<what> must be an integer >= <min>, got '<tok>'"         (max unbounded)
 *     "<what> must be an integer from <min> to <max>, got '<tok>'"
 * followed by " (<note>)" when `note` is not empty. */
std::uint64_t parse_uint(const std::string& tok, const std::string& what,
                         std::uint64_t min = 0,
                         std::uint64_t max =
                             std::numeric_limits<std::uint64_t>::max(),
                         const std::string& note = "");

/* The number grammar of every key, the syntax LAMMPS' utils::is_double
 * accepts: an optional '+' or '-', decimal digits with at most one '.' and
 * at least one digit, then an optional exponent [eE][+-]<digits> -- "5",
 * "-0.5", ".5", "5.", "+1e-3", "2.5E+07". Nothing else: no hexadecimal
 * float ("0x1p4"), no inf or nan, no whitespace or trailing character. The
 * value must be a finite double, and a normal one or zero: a number that
 * overflows (1e400) is rejected, as LAMMPS' utils::numeric rejects it, and
 * so is one whose magnitude is below the smallest normal double (1e-320,
 * or 1e-400, which underflows to 0) -- HPO's own rule: utils::numeric
 * accepts such a number (a subnormal, or 0). On success sets `value` and
 * returns true; otherwise returns false and leaves `value` alone. The
 * conversion (std::from_chars) does not depend on the locale, and the
 * finiteness and subnormal tests are on the bits (metric_value_is_finite),
 * which neither -ffast-math nor denormals-are-zero can defeat. */
bool parse_number(const std::string& tok, double& value);

/* A finite real number (parse_number). Otherwise throws
 *     "<what> must be a finite number, got '<tok>'"
 * followed by " (too small: ...)" when the token is in the grammar and its
 * magnitude is below the smallest normal double (e.g. 1e-320). */
double parse_finite(const std::string& tok, const std::string& what);

/* A positive finite real number (parse_number, > 0). Otherwise throws
 *     "<what> must be a positive <noun>, got '<tok>'"
 * followed by the too-small note of parse_finite. */
double parse_positive(const std::string& tok, const std::string& what,
                      const std::string& noun = "number");

/* " (too small: ...)" when `tok` is in the number grammar and its
 * magnitude is below the smallest normal double, so parse_number()
 * rejects a finite-looking number; "" otherwise. For the messages of the
 * callers of parse_number(). */
std::string number_underflow_note(const std::string& tok);

/* True when `tok` is meant as a number: it starts like one -- a digit, or
 * a sign or '.' followed by a digit or '.' ("5x", "-.5x", "1e-320",
 * "0x1p4") -- or it is an inf or nan word, with or without a sign ("inf",
 * "-Infinity", "NaN"). Keys and words ("WATOM", "AUTO", "inhouse", "INFO")
 * are not. Where a position takes a number or a word (STEP <frac> | STEP
 * <KEY> <frac>, a PARAM value, an optional value on a PC_LAMMPS line), such
 * a token is a number and the number grammar (parse_number) judges it. */
bool is_number_shaped(const std::string& tok);

/* The next token of `iss`, the value of `key`; throws
 * "<key> requires a value" when there is none. */
std::string next_value(std::istringstream& iss, const std::string& key);

/* Throws "<what> (unexpected '<tok>')" when `iss` holds another token. */
void expect_end(std::istringstream& iss, const std::string& what);

/* The one value of a single-value key: next_value(), then
 * expect_end(iss, key + " takes a single value"). */
std::string single_value(std::istringstream& iss, const std::string& key);

/* A LAMMPS time spec, a strict subset of what `timer timeout` reads
 * (utils::timespec2seconds): "off" or "unlimited" (no limit: nullopt), or
 * [[HH:]MM:]SS -- one to three non-empty ':'-separated numbers of the
 * number grammar (parse_number) -- converted to seconds as
 * (HH*60 + MM)*60 + SS, the seconds LAMMPS computes for it. Like LAMMPS, a
 * component may carry a fraction or a sign. LAMMPS reads the components
 * with std::stod and also takes what this rejects: an empty component
 * ("1::30", ":30"; skipped), a fourth one ("1:2:3:4"; ignored), and stod's
 * forms (hexadecimal, inf, a subnormal). Otherwise throws
 *     "<what> must be 'off', 'unlimited' or a time [[HH:]MM:]SS, got '<tok>'" */
std::optional<double> parse_lammps_timespec(const std::string& tok,
                                            const std::string& what);

/* A script timeout (PC_SCRIPT_TIMEOUT, its alias MAXTIME, --maxtime): a
 * LAMMPS time spec (parse_lammps_timespec) that is "off", "unlimited" or a
 * positive time. LAMMPS accepts 0 and a negative time, but either leaves
 * $(timeremain) at 0, so the section guard injected before every `clear`
 * stops each multi-section script (and 0 stops any script at its first
 * timer check): the evaluations would score FAILSCORE. Such a limit throws
 *     "<what> must be a positive time, 'off' or 'unlimited' (a limit <= 0
 *      stops every script), got '<tok>'"
 * The PROCESS worker deadline derived from it, max(3t, t + 120 s), must
 * fit in 64-bit milliseconds (pc_worker_deadline_fits: t below about
 * 3.07e15 s); a larger time throws
 *     "<what> is too large for the worker deadline, got '<tok>' (...)" */
std::optional<double> parse_script_timeout(const std::string& tok,
                                           const std::string& what);

}  // namespace tadah::hpo
