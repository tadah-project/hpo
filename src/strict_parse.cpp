#include <tadah/hpo/strict_parse.h>

#include <tadah/hpo/pc_worker_pool.h>      // pc_worker_deadline_fits
#include <tadah/hpo/validation_metric.h>   // metric_value_is_finite

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace tadah::hpo {

namespace {

bool is_digit(char c) { return std::isdigit(static_cast<unsigned char>(c)) != 0; }

/* The syntax of the number grammar (strict_parse.h, parse_number): the
 * union of the four patterns of LAMMPS' utils::is_double,
 * [+-]?(\d+\.?\d*|\d*\.?\d+)([eE][+-]?\d+)?. */
bool has_number_syntax(const std::string& t)
{
    std::size_t i = 0;
    const std::size_t n = t.size();
    if (i < n && (t[i] == '+' || t[i] == '-')) ++i;
    std::size_t mantissa_digits = 0;
    while (i < n && is_digit(t[i])) { ++i; ++mantissa_digits; }
    if (i < n && t[i] == '.') {
        ++i;
        while (i < n && is_digit(t[i])) { ++i; ++mantissa_digits; }
    }
    if (mantissa_digits == 0) return false;
    if (i < n && (t[i] == 'e' || t[i] == 'E')) {
        ++i;
        if (i < n && (t[i] == '+' || t[i] == '-')) ++i;
        std::size_t exponent_digits = 0;
        while (i < n && is_digit(t[i])) { ++i; ++exponent_digits; }
        if (exponent_digits == 0) return false;
    }
    return i == n;
}

/* An inf or nan word, as strtod reads one: an optional sign, then "inf",
 * "infinity", "nan" or "nan(...)", in any case. */
bool is_inf_nan_word(const std::string& tok)
{
    std::string w = tok;
    if (!w.empty() && (w[0] == '+' || w[0] == '-')) w.erase(0, 1);
    std::transform(w.begin(), w.end(), w.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
    if (w == "inf" || w == "infinity" || w == "nan") return true;
    if (w.size() < 5 || w.compare(0, 4, "nan(") != 0 || w.back() != ')')
        return false;
    return std::all_of(w.begin() + 4, w.end() - 1, [](unsigned char c) {
        return std::isalnum(c) != 0 || c == '_';
    });
}

/* A subnormal double: exponent field 0, mantissa not 0. Read from the bits,
 * like metric_value_is_finite: with denormals-are-zero on (the start-up
 * code of an executable linked with -ffast-math sets it) a subnormal
 * compares equal to 0, so `v != 0.0 && |v| < DBL_MIN` let it through. */
bool is_subnormal(double v)
{
    static_assert(sizeof(double) == sizeof(std::uint64_t));
    static_assert(std::numeric_limits<double>::is_iec559);
    std::uint64_t bits;
    std::memcpy(&bits, &v, sizeof(bits));
    return (bits & UINT64_C(0x7ff0000000000000)) == 0 &&
           (bits & UINT64_C(0x000fffffffffffff)) != 0;
}

}  // namespace

bool parse_number(const std::string& tok, double& value)
{
    if (!has_number_syntax(tok)) return false;
    const char* first = tok.data();
    const char* const last = first + tok.size();
    if (*first == '+') ++first;   /* from_chars takes no '+' */
    double v = 0.0;
    const auto r = std::from_chars(first, last, v);
    /* errc::result_out_of_range: an overflow, or an underflow to zero. */
    if (r.ec != std::errc{} || r.ptr != last || !metric_value_is_finite(v))
        return false;
    /* A subnormal result: below the smallest normal double. */
    if (is_subnormal(v)) return false;
    value = v;
    return true;
}

std::string number_underflow_note(const std::string& tok)
{
    double v = 0.0;
    if (!has_number_syntax(tok) || parse_number(tok, v)) return "";
    /* In the grammar, yet rejected: it overflows or underflows. strtod
     * tells which: inf for an overflow, 0 or a subnormal for an underflow
     * (the bits, whatever the floating-point mode). */
    if (!metric_value_is_finite(std::strtod(tok.c_str(), nullptr)))
        return "";
    return " (too small: its magnitude is below the smallest normal "
           "double, 2.2250738585072014e-308)";
}

std::uint64_t parse_uint(const std::string& tok, const std::string& what,
                         std::uint64_t min, std::uint64_t max,
                         const std::string& note)
{
    std::uint64_t v = 0;
    const char* first = tok.data();
    const char* last  = first + tok.size();
    /* from_chars reads decimal digits only: no whitespace, no '+', and for
     * an unsigned type no '-'; ptr != last catches a fraction, an exponent
     * and trailing junk, errc::result_out_of_range an overflow. */
    const auto r = std::from_chars(first, last, v);
    const bool ok = !tok.empty() && r.ec == std::errc{} && r.ptr == last &&
                    v >= min && v <= max;
    if (ok) return v;
    std::string msg = what + " must be an integer ";
    if (max == std::numeric_limits<std::uint64_t>::max())
        msg += ">= " + std::to_string(min);
    else
        msg += "from " + std::to_string(min) + " to " + std::to_string(max);
    msg += ", got '" + tok + "'";
    /* The note explains why a whole number below `min` is refused (0
     * restarts nothing); any other token gets the range alone. */
    const bool below_min = !tok.empty() && r.ec == std::errc{} &&
                           r.ptr == last && v < min;
    if (!note.empty() && below_min) msg += " (" + note + ")";
    throw std::runtime_error(msg);
}

double parse_finite(const std::string& tok, const std::string& what)
{
    double v = 0.0;
    if (!parse_number(tok, v))
        throw std::runtime_error(what + " must be a finite number, got '" +
                                 tok + "'" + number_underflow_note(tok));
    return v;
}

double parse_positive(const std::string& tok, const std::string& what,
                      const std::string& noun)
{
    double v = 0.0;
    if (!parse_number(tok, v) || !(v > 0.0))
        throw std::runtime_error(what + " must be a positive " + noun +
                                 ", got '" + tok + "'" +
                                 number_underflow_note(tok));
    return v;
}

bool is_number_shaped(const std::string& tok)
{
    if (tok.empty()) return false;
    if (is_digit(tok[0])) return true;
    if ((tok[0] == '+' || tok[0] == '-' || tok[0] == '.') && tok.size() > 1 &&
        (is_digit(tok[1]) || tok[1] == '.'))
        return true;
    return is_inf_nan_word(tok);
}

std::string next_value(std::istringstream& iss, const std::string& key)
{
    std::string tok;
    if (!(iss >> tok))
        throw std::runtime_error(key + " requires a value");
    return tok;
}

void expect_end(std::istringstream& iss, const std::string& what)
{
    std::string extra;
    if (iss >> extra)
        throw std::runtime_error(what + " (unexpected '" + extra + "')");
}

std::string single_value(std::istringstream& iss, const std::string& key)
{
    std::string tok = next_value(iss, key);
    expect_end(iss, key + " takes a single value");
    return tok;
}

std::optional<double> parse_lammps_timespec(const std::string& tok,
                                            const std::string& what)
{
    if (tok == "off" || tok == "unlimited") return std::nullopt;
    std::string note;   // set when a component underflows a double
    const auto bad = [&]() {
        return std::runtime_error(
            what + " must be 'off', 'unlimited' or a time [[HH:]MM:]SS, got '" +
            tok + "'" + note);
    };
    std::vector<double> parts;
    std::size_t from = 0;
    while (true) {
        const std::size_t colon = tok.find(':', from);
        const std::string part = tok.substr(
            from, colon == std::string::npos ? std::string::npos : colon - from);
        double v = 0.0;
        if (!parse_number(part, v)) {
            note = number_underflow_note(part);
            throw bad();
        }
        parts.push_back(v);
        if (colon == std::string::npos) break;
        from = colon + 1;
    }
    /* utils::timespec2seconds reads at most three components. */
    if (parts.size() > 3) throw bad();
    double seconds = 0.0;
    for (const double p : parts) seconds = seconds * 60.0 + p;
    /* Finite components can still add up to an infinite time (1e307:0:0). */
    if (!metric_value_is_finite(seconds)) throw bad();
    return seconds;
}

std::optional<double> parse_script_timeout(const std::string& tok,
                                           const std::string& what)
{
    const std::optional<double> seconds = parse_lammps_timespec(tok, what);
    if (seconds && !(*seconds > 0.0))
        throw std::runtime_error(
            what + " must be a positive time, 'off' or 'unlimited' (a limit "
            "<= 0 stops every script), got '" + tok + "'");
    if (!pc_worker_deadline_fits(0.0, seconds))
        throw std::runtime_error(
            what + " is too large for the worker deadline, got '" + tok +
            "' (the PC_ISOLATION PROCESS deadline max(3t, t + 120 s) is kept "
            "in 64-bit milliseconds: t must stay below about 3.07e15 s)");
    return seconds;
}

}  // namespace tadah::hpo
