// LammpsRunner::get_variable readback precision.
//
// Every PC_LAMMPS outvar, hot value, failvar and gradient observable reaches
// HPO as the TEXT get_variable() returns. LAMMPS' Variable::retrieve()
// renders equal-style variables with "%.15g", so each readback lost up to
// ~2 significant digits before the loss, the warm-start cache and the
// '# HOT' headers ever saw the value. get_variable() now evaluates numeric
// (equal-style) variables with compute_equal() and renders the double
// losslessly; every other style is text and comes back verbatim.
//
// A LAMMPS error raised while a variable is evaluated leaves the variables
// of that evaluation marked as being evaluated; a later read reports the
// variable's own error again, not a false circular dependency.
//
// In-process LAMMPS instance, no potential needed (pure variable commands).

#include "catch2/catch.hpp"

#include <tadah/core/number_format.h>
#include <tadah/hpo/runners.h>

#include <input.h>
#include <variable.h>

#include <cfloat>
#include <cstring>
#include <exception>
#include <string>
#include <vector>

namespace {

LammpsRunner::LMPPtr make_instance() {
  LammpsRunner runner;
  return runner.create_instance(
      0, std::vector<std::string>{"lmp", "-log", "none", "-screen", "none"});
}

bool same_bits(double a, double b) {
  return std::memcmp(&a, &b, sizeof a) == 0;
}

}  // namespace

TEST_CASE("get_variable reads equal-style variables losslessly",
          "[HPO][lammps][readback][lossless]")
{
  auto lmp = make_instance();
  REQUIRE(lmp);
  LammpsRunner runner;

  lmp->input->one("variable a equal 0.1+0.2");
  CHECK(runner.get_variable(lmp, "a") == "0.30000000000000004");
  CHECK(same_bits(std::stod(runner.get_variable(lmp, "a")), 0.1 + 0.2));
  /* What retrieve() (and so the old readback) gave for the same variable. */
  CHECK(std::string(lmp->input->variable->retrieve("a")) == "0.3");

  lmp->input->one("variable third equal 1/3");
  CHECK(runner.get_variable(lmp, "third") ==
        tadah::core::format_roundtrip(1.0 / 3.0));
  CHECK(same_bits(std::stod(runner.get_variable(lmp, "third")), 1.0 / 3.0));

  /* Integral results stay plain integers. */
  lmp->input->one("variable n equal 2*3");
  CHECK(runner.get_variable(lmp, "n") == "6");

  /* Same underflow-to-zero rounding as retrieve() for a subnormal result
   * (LAMMPS' own literal parser rejects a subnormal literal like 1e-310). */
  lmp->input->one("variable tiny equal 1e-200*1e-110");
  CHECK(runner.get_variable(lmp, "tiny") == "0");
  CHECK(std::string(lmp->input->variable->retrieve("tiny")) == "0");

  /* Re-defining an equal-style variable is re-evaluated on every read. */
  lmp->input->one("variable a equal 2.0/7.0");
  CHECK(same_bits(std::stod(runner.get_variable(lmp, "a")), 2.0 / 7.0));
}

TEST_CASE("get_variable reads internal-style variables exactly",
          "[HPO][lammps][readback][lossless]")
{
  auto lmp = make_instance();
  REQUIRE(lmp);
  LammpsRunner runner;

  lmp->input->one("variable q internal 0");
  auto* var = lmp->input->variable;
  const int iq = var->find("q");
  REQUIRE(iq >= 0);
  const double value = 0.12345678901234568;  // 17 significant digits
  var->internal_set(iq, value);
  CHECK(same_bits(std::stod(runner.get_variable(lmp, "q")), value));
  CHECK_FALSE(same_bits(std::stod(var->retrieve("q")), value));
}

TEST_CASE("get_variable returns text-style variables verbatim and undefined "
          "ones as empty", "[HPO][lammps][readback][lossless]")
{
  auto lmp = make_instance();
  REQUIRE(lmp);
  LammpsRunner runner;

  /* String-style: text by definition, never re-rendered. */
  lmp->input->one("variable s string 0.30000000000000004441");
  CHECK(runner.get_variable(lmp, "s") == "0.30000000000000004441");
  lmp->input->one("variable w string hello");
  CHECK(runner.get_variable(lmp, "w") == "hello");

  /* Index-style: the current item, verbatim. */
  lmp->input->one("variable i index 1.50 2.5");
  CHECK(runner.get_variable(lmp, "i") == "1.50");

  /* Format-style: LAMMPS applies the user's format. */
  lmp->input->one("variable third equal 1/3");
  lmp->input->one("variable f format third %.3f");
  CHECK(runner.get_variable(lmp, "f") == "0.333");

  CHECK(runner.get_variable(lmp, "no_such_variable").empty());
}

namespace {

/* The message of the exception get_variable raises for @p name ("" when it
 * reads the variable). */
std::string read_error(const LammpsRunner& runner,
                       const LammpsRunner::LMPPtr& lmp,
                       const std::string& name) {
  try {
    (void)runner.get_variable(lmp, name);
  } catch (const std::exception& e) {
    return e.what();
  }
  return {};
}

}  // namespace

/* Pins: after a LAMMPS error raised while a variable was being evaluated --
 * by an earlier get_variable, or by a command of the script -- a read that
 * evaluates that variable again raises its own error again ("Divide by 0 in
 * variable formula"), not "has a circular dependency": the variable itself
 * (bad), a variable that references it (uses = v_bad+1.0) and one that
 * references that one (uses2 = 2*v_uses), read one after the other and
 * the first again; a variable that does not reference it reads as usual
 * (ok = 2.5).
 * Before the fix: the error left the variables of the interrupted evaluation
 * marked as being evaluated (LAMMPS Variable::eval_in_progress, which only
 * `clear` and a redefinition reset), so every later read that reached one
 * of them raised "Variable bad: has a circular dependency" (or "Variable
 * uses: ...") instead, and the real cause was lost. The message checks
 * fail. */
TEST_CASE("get_variable after a LAMMPS error raises the variable's own error "
          "and not a circular dependency", "[HPO][lammps][readback]")
{
  auto lmp = make_instance();
  REQUIRE(lmp);
  LammpsRunner runner;
  lmp->input->one("variable zero equal 0.0");
  lmp->input->one("variable bad equal 1.0/v_zero");
  lmp->input->one("variable uses equal v_bad+1.0");
  lmp->input->one("variable uses2 equal 2*v_uses");
  lmp->input->one("variable ok equal 2.5");
  const char* divide = "Divide by 0 in variable formula";

  SECTION("an earlier get_variable") {
    CHECK_THAT(read_error(runner, lmp, "uses"), Catch::Contains(divide));
  }
  SECTION("a command of the script") {
    std::string msg;
    try {
      lmp->input->one("print \"${uses2}\"");
    } catch (const std::exception& e) {
      msg = e.what();
    }
    REQUIRE_THAT(msg, Catch::Contains(divide));
  }
  for (const char* name : {"bad", "uses", "uses2", "bad"}) {
    INFO("reading " << name);
    const std::string msg = read_error(runner, lmp, name);
    CHECK_THAT(msg, Catch::Contains(divide) &&
                        !Catch::Contains("circular dependency"));
  }
  CHECK(runner.get_variable(lmp, "ok") == "2.5");
}
