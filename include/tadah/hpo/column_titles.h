#pragma once
/*
 * column_titles -- the geometry of the value columns of loss.tadah,
 * outvar.tadah, params.tadah and stats.tadah, and the fitting of an indexed
 * title into it. Shared by HPO_Host, which writes the file headers, and the
 * journal, which names the columns (HPOLog::write_loss_section), so both
 * print the same title.
 */

#include <algorithm>
#include <cstddef>
#include <string>

namespace tadah::hpo {

/* The narrowest value column: it holds L(ErRMSE)[avg] (14) and a space. */
inline constexpr std::size_t min_column_width = 15;

/* The width of a value column at DIGITS `digits`, the digits after the
 * decimal point of its scientific notation (sign, digit, point, the digits,
 * e+XX and a gap), never below min_column_width. */
inline std::size_t column_width(int digits)
{
    return static_cast<std::size_t>(
        std::max(digits + 9, static_cast<int>(min_column_width)));
}

/* The longest title a column holds at DIGITS `digits`: L(VAR) always keeps
 * a gap of at least one space, and VAR is cut the same way. */
inline std::size_t column_title_budget(int digits)
{
    return column_width(digits) - 4;
}

namespace header_detail {
  /* Fit an indexed label "NAME[i]" into maxlen characters by shortening
   * NAME and keeping the index: "AMPGRIDMB[12]" -> "AMPGRID*[12]". Plain
   * truncation would cut every member of a family to the same "NAME[*",
   * leaving the collision resolver to number them ~2, ~3, ..., which is
   * unique but says nothing. Labels that fit, and labels without an index
   * (or whose index alone does not fit), are returned unchanged:
   * unique_label (hpo_host_impl.hpp) cuts those. */
  inline std::string fit_keep_index(std::string name, std::size_t maxlen)
  {
    if (name.size() <= maxlen) return name;
    const std::size_t br = name.rfind('[');
    if (br == std::string::npos || br == 0) return name;
    const std::string idx = name.substr(br);            /* "[12]" */
    if (idx.size() + 2 > maxlen) return name;
    return name.substr(0, maxlen - idx.size() - 1) + "*" + idx;
  }
}  // namespace header_detail

}  // namespace tadah::hpo
