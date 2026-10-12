// test_lammps_runner.cpp
//
// The communicator of the LAMMPS instances LammpsRunner creates. Every LAMMPS
// instance HPO creates runs on one communicator, the build's:
// tadah::hpo::hpo_lammps_comm(), the LAMMPS STUBS' 0 in the serial build and
// MPI_COMM_SELF in the MPI build (TADAH_HPO_MPI), where each rank runs
// single-rank instances. A caller that chose it could pass the serial
// build's 0 in the MPI build, where a literal 0 is a null communicator and
// LAMMPS aborts the process in MPI_Comm_rank (MPI_ERR_COMM) -- what nine test
// cases of the MPI build's suite did, while the serial suite passed them.
//
// The calls are checked through detection traits, so this file compiles
// whichever create_instance the runner has.

#include "catch2/catch.hpp"

#include <tadah/hpo/hpo_mpi.h>
#include <tadah/hpo/runners.h>

#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace {

using Tokens = std::vector<std::string>;

/* Whether r.create_instance(0, tokens) -- a literal 0, the serial build's
 * communicator -- is a well-formed call. */
template <class R, class = void>
struct takes_literal_zero : std::false_type {};
template <class R>
struct takes_literal_zero<R, std::void_t<decltype(
    std::declval<const R&>().create_instance(0, std::declval<Tokens>()))>>
    : std::true_type {};

/* Whether r.create_instance(hpo_lammps_comm(), tokens) is. */
template <class R, class = void>
struct takes_build_comm : std::false_type {};
template <class R>
struct takes_build_comm<R, std::void_t<decltype(
    std::declval<const R&>().create_instance(
        tadah::hpo::hpo_lammps_comm(), std::declval<Tokens>()))>>
    : std::true_type {};

/* Whether r.create_instance(tokens) is. */
template <class R, class = void>
struct takes_tokens : std::false_type {};
template <class R>
struct takes_tokens<R, std::void_t<decltype(
    std::declval<const R&>().create_instance(std::declval<Tokens>()))>>
    : std::true_type {};

/* Whether r.create_instance(tokens, &error) is. */
template <class R, class = void>
struct takes_tokens_and_error : std::false_type {};
template <class R>
struct takes_tokens_and_error<R, std::void_t<decltype(
    std::declval<const R&>().create_instance(std::declval<Tokens>(),
                                             std::declval<std::string*>()))>>
    : std::true_type {};

/* Creates an instance with @p runner's create_instance(tokens, &error) and
 * checks its communicator; a template, so that the call is compiled only
 * for a runner that has it. */
template <class R>
void check_instance_communicator(const R& runner) {
  if constexpr (takes_tokens_and_error<R>::value) {
    std::string error;
    auto lmp = runner.create_instance(
        Tokens{"lmp", "-log", "none", "-screen", "none"}, &error);
    INFO("create_instance error: " << error);
    REQUIRE(lmp != nullptr);
    CHECK(lmp->world == tadah::hpo::hpo_lammps_comm());
#ifdef TADAH_HPO_MPI
    int size = 0;
    MPI_Comm_size(lmp->world, &size);
    CHECK(size == 1);
    int same = MPI_UNEQUAL;
    MPI_Comm_compare(lmp->world, MPI_COMM_SELF, &same);
    CHECK(same == MPI_IDENT);
#endif
    lmp->input->one("variable six equal 2*3");
    CHECK(runner.get_variable(lmp, "six") == "6");
  } else {
    FAIL("LammpsRunner::create_instance(tokens, &error) does not exist: the "
         "caller chooses the communicator");
  }
}

}  // namespace

/* Pins: LammpsRunner::create_instance takes no communicator. A call that
 * passes one -- a literal 0, or hpo_lammps_comm() itself -- does not
 * compile, in either build; create_instance(tokens) and
 * create_instance(tokens, &error) do.
 * Before the fix: fails, all four checks: create_instance(MPI_Comm, tokens,
 * error = nullptr) took the communicator from the caller, so both calls
 * with one compiled -- create_instance(0, ...) in the MPI build too, a null
 * communicator that aborts the process at run time -- and the calls
 * without one did not. */
TEST_CASE("LammpsRunner::create_instance takes no communicator",
          "[HPO][lammps][runner][mpi_build]") {
  CHECK_FALSE(takes_literal_zero<LammpsRunner>::value);
  CHECK_FALSE(takes_build_comm<LammpsRunner>::value);
  CHECK(takes_tokens<LammpsRunner>::value);
  CHECK(takes_tokens_and_error<LammpsRunner>::value);
}

/* Pins: create_instance(tokens, &error) returns an instance whose world is
 * the build's communicator, hpo_lammps_comm() -- in the MPI build the
 * MPI_COMM_SELF handle itself (MPI_Comm_compare: MPI_IDENT, size 1), in the
 * serial build the STUBS' 0 -- and the instance runs a command ("variable
 * six equal 2*3" reads back as "6").
 * Before the fix: fails ("LammpsRunner::create_instance(tokens, &error)
 * does not exist"). */
TEST_CASE("LammpsRunner::create_instance runs the instance on the build's "
          "communicator", "[HPO][lammps][runner][mpi_build]") {
  check_instance_communicator(LammpsRunner{});
}
