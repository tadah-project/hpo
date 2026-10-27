#pragma once
/* A count and its noun for the messages and journal lines of tadah hpo, in
 * the singular for a count of one: count_text(1, "probe", "probes") is
 * "1 probe", count_text(3, "probe", "probes") "3 probes". */

#include <string>

namespace tadah::hpo {

inline std::string count_text(unsigned long long n, const std::string& singular,
                              const std::string& plural)
{
    return std::to_string(n) + " " + (n == 1 ? singular : plural);
}

}  // namespace tadah::hpo
