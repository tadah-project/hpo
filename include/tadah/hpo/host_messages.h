#pragma once
/*
 * host_messages.h -- journal lines of HPO_Host that a script may read, built
 * in one place so that the host writes them and the tests pin them from the
 * same text.
 *
 *   evaluation_error_warning   the WARNING of an evaluation that stopped on
 *                              an error (HPO_Host::operator(): it scores
 *                              FAILSCORE)
 *   result_gap_warning         the WARNING of the result writer when
 *                              evaluations did not return within its
 *                              gap-skip time and their rows are given up
 */

#include <cstddef>
#include <cstdint>
#include <string>

namespace tadah::hpo {

/* "eval <iter>: the evaluation stopped on an error and scores FAILSCORE:
 * <what>". */
inline std::string evaluation_error_warning(std::size_t iter,
                                            const std::string& what)
{
    return "eval " + std::to_string(iter) +
           ": the evaluation stopped on an error and scores FAILSCORE: " +
           what;
}

/* "eval <iter>: the evaluation stopped on an error that carries no message
 * and scores FAILSCORE": an exception that is not a std::exception. */
inline std::string evaluation_error_warning(std::size_t iter)
{
    return "eval " + std::to_string(iter) +
           ": the evaluation stopped on an error that carries no message "
           "and scores FAILSCORE";
}

/* The evaluations [first, next) did not return within @p seconds:
 * "result files: evaluation <first> did not return within <s> s; its rows
 * are not written to the result files", or "evaluations <first> to
 * <next - 1> ... their rows ...", followed by " (stages still running:
 * [<stuck_stages>])" when @p stuck_stages is not empty. */
inline std::string result_gap_warning(std::size_t first, std::size_t next,
                                      std::int64_t seconds,
                                      const std::string& stuck_stages)
{
    const bool one = next == first + 1;
    return "result files: " +
           (one ? "evaluation " + std::to_string(first)
                : "evaluations " + std::to_string(first) + " to " +
                      std::to_string(next - 1)) +
           " did not return within " + std::to_string(seconds) + " s; " +
           (one ? "its rows are" : "their rows are") +
           " not written to the result files" +
           (stuck_stages.empty()
                ? std::string{}
                : " (stages still running: [" + stuck_stages + "])");
}

}  // namespace tadah::hpo
