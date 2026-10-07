#pragma once
/*
 * run_end.h -- why an optimiser search ended: one vocabulary for the
 * drivers that name their own endings.
 *
 * Every optimiser run records why it ended (HPOLog::set_status); the run's
 * journal trailer ("Optimiser status") is the status of the run that ended
 * the whole run (run_driver.h), and a normal end without one is an internal
 * error there. The drivers of TADAH (RANDOM, GRID, ANNEAL, ANNEAL_PT,
 * ANNEAL_PA), DLIB (GFS, BFGS) and CERES record "<LIB>:<ALGO>: <reason>"
 * with a fixed reason text from this table, so a script or a fingerprint
 * can match it; the numbers (the evaluations made, a tolerance and whether
 * it is the default) are in the journal's stop line, never in the status.
 * NLopt keeps its own result strings ("MAXEVAL_REACHED -- maximum
 * evaluations exhausted", nlopt_result_str) and ASA "ASA: <reason>"; a
 * gradient that is not finite ends a search with its own status
 * (gradient_stop_status, gradient_provider.h), and RESTART with
 * "STOPPED -- stagnation: ...".
 *
 *   EndReason     status text                 the stop
 *   MaxEval       MAXEVAL reached             MAXEVAL spent
 *   MaxTime       MAXTIME exceeded            MAXTIME passed
 *   StopVal       STOPVAL reached             a loss at or below STOPVAL
 *   FtolAbs       FTOL_ABS reached            DLIB:BFGS: the loss changed by
 *                                             less than FTOL_ABS from one
 *                                             iteration to the next
 *   FtolRel       FTOL_REL reached            CERES: function_tolerance
 *   XtolRel       XTOL_REL reached            CERES: parameter_tolerance
 *   GradTol       tol_gradinf reached         PARAM tol_gradinf (DLIB:BFGS:
 *                                             the gradient's L2 norm; CERES:
 *                                             gradient_tolerance, its max
 *                                             norm)
 *   MaxIter       max_iter reached            CERES: PARAM max_iter
 *   TMin          T_MIN reached               the annealing schedule cooled
 *                                             below T_MIN
 *   FtolPlateau   FTOL plateau                the annealers' FTOL_ABS /
 *                                             FTOL_REL plateau rule
 *   GridDone      all grid points evaluated   TADAH:GRID's whole grid
 *   NoDims        no optimisable dimensions   an annealer with no OPTIM
 *                                             dimension to move
 */

#include <tadah/hpo/hpo_log.h>

#include <iostream>
#include <string>

namespace tadah::hpo {

enum class EndReason {
    MaxEval, MaxTime, StopVal, FtolAbs, FtolRel, XtolRel, GradTol, MaxIter,
    TMin, FtolPlateau, GridDone, NoDims
};

/* The fixed text of `r` (the table above). */
inline const char* end_reason_text(EndReason r)
{
    switch (r) {
        case EndReason::MaxEval:     return "MAXEVAL reached";
        case EndReason::MaxTime:     return "MAXTIME exceeded";
        case EndReason::StopVal:     return "STOPVAL reached";
        case EndReason::FtolAbs:     return "FTOL_ABS reached";
        case EndReason::FtolRel:     return "FTOL_REL reached";
        case EndReason::XtolRel:     return "XTOL_REL reached";
        case EndReason::GradTol:     return "tol_gradinf reached";
        case EndReason::MaxIter:     return "max_iter reached";
        case EndReason::TMin:        return "T_MIN reached";
        case EndReason::FtolPlateau: return "FTOL plateau";
        case EndReason::GridDone:    return "all grid points evaluated";
        case EndReason::NoDims:      return "no optimisable dimensions";
    }
    return "";   // unreachable: every enumerator is listed
}

/* The run's status for a search of `backend` ("DLIB:BFGS") that `r`
 * ended: "DLIB:BFGS: MAXEVAL reached". */
inline std::string end_status(const std::string& backend, EndReason r)
{
    return backend + ": " + end_reason_text(r);
}

/* Records that `r` ended the search of `backend`: the run's status
 * (end_status), for a driver that writes its own stop line. */
inline void record_end(const std::string& backend, EndReason r)
{
    HPOLog::instance().set_status(end_status(backend, r));
}

/* The stop line "<backend> stopped: <reason> after <detail>" on stdout
 * ("[HPO] ...") and in the journal, then the run's status (record_end).
 * `detail` says what the search spent and, for a tolerance, its value:
 * "51 evaluation(s): the loss changed by less than FTOL_ABS = 1e-07
 * (default) from one iteration to the next". */
inline void report_end(const std::string& backend, EndReason r,
                       const std::string& detail)
{
    const std::string msg = backend + " stopped: " + end_reason_text(r) +
                            " after " + detail;
    std::cout << "[HPO] " << msg << "\n";
    HPOLog::instance().info(msg);
    record_end(backend, r);
}

}  // namespace tadah::hpo
