// test_hpotarget_script_contract.cpp
//
// [lammps_script] of the target-file registry (blueprint/hpotarget_lines.toml)
// says what the LAMMPS script of a PC_LAMMPS line provides, and its fixture is
// the script the reference page shows. An evaluation runs the script in a
// LAMMPS instance it keeps for the next evaluation of the same line, with
// the candidate potential staged (pc_task_exec.h, LammpsRunner::run_script):
// `pair_style tadah` with `pair_coeff * * EXTERN <element> ...` adopts that
// potential, and `clear` removes the atoms and settings the last evaluation
// left (its variables stay).
//
// The case runs the fixture as two evaluations of one line run it: in one
// LAMMPS instance, the first with a linear Ta potential, the second with the
// same potential, its weights doubled. Each variable that a --varloss or
// --outvar option of [lammps_script]'s examples names is an equal-style
// variable of the script, finite after the first run and doubled after the
// second: the script took the potential of each evaluation.
// The registry is read at run time (TADAH_HPOTARGET_REGISTRY_DIR names
// another directory to read it from, as for the conformance cases).

#include "catch2/catch.hpp"

#include <toml.hpp>

#include <tadah/core/context.h>
#include <tadah/core/core_user_config_parser.h>
#include <tadah/core/periodic_table.h>
#include <tadah/hpo/runners.h>
#include <tadah/hpo/script_file.h>
#include <tadah/tadah.lammps/lammps_tadah.h>

#include <input.h>
#include <variable.h>

#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#ifndef TADAH_HPO_BLUEPRINT_DIR
#error "TADAH_HPO_BLUEPRINT_DIR names HPO/blueprint (tests/CMakeLists.txt)"
#endif

namespace {

namespace fs = std::filesystem;

fs::path registry_dir()
{
    if (const char* d = std::getenv("TADAH_HPOTARGET_REGISTRY_DIR"))
        if (*d) return d;
    return TADAH_HPO_BLUEPRINT_DIR;
}

/* The Ta D2_LJ potential of test_pair_tadah_coincident.cpp (kLjBody):
 * linear, two-body, without a bias, so its energy is proportional to its
 * weights. */
std::string lj_potential(double scale)
{
    std::ostringstream w;
    w.precision(17);
    w << 14.204213349318765225 * scale << " "
      << 2604.396965191186154698 * scale;
    return "ATOM     Ta\n"
           "BIAS     false\n"
           "DIMER     false     0     false\n"
           "INIT2B     true\n"
           "INITMB     false\n"
           "LAMBDA     0\n"
           "MODEL     M_KRR     Kern_Linear     1\n"
           "NORM     false\n"
           "OALGO     1\n"
           "RCTYPE2B     Cut_Dummy\n"
           "RCUT2B     6.5\n"
           "TYPE2B     D2_LJ     Ta     Ta\n"
           "WATOM     73\n"
           "WEIGHTS     " + w.str() + "\n";
}

std::shared_ptr<tadah::LammpsTadah> potential(double scale)
{
    std::istringstream in(lj_potential(scale));
    return std::make_shared<tadah::LammpsTadah>(
        tadah::core::CoreUserConfigParser().parse(in));
}

std::vector<std::string> lines_of(const std::string& text)
{
    std::vector<std::string> out;
    std::istringstream in(text);
    for (std::string line; std::getline(in, line);)
        out.push_back(line);
    return out;
}

/* The names that the --varloss and --outvar options of `line` give. */
std::vector<std::string> variables_of(const std::string& line)
{
    std::vector<std::string> out;
    std::istringstream in(line);
    std::vector<std::string> tok;
    for (std::string t; in >> t;) tok.push_back(t);
    for (std::size_t i = 0; i + 1 < tok.size(); ++i)
        if (tok[i] == "--varloss" || tok[i] == "--outvar")
            out.push_back(tok[i + 1]);
    return out;
}

double value_of(const LammpsRunner::LMPPtr& lmp, const std::string& name)
{
    auto* var = lmp->input->variable;
    const int idx = var->find(name.c_str());
    INFO("variable " << name);
    REQUIRE(idx >= 0);
    REQUIRE(var->equalstyle(idx));
    const char* raw = var->retrieve(name.c_str());
    REQUIRE(raw != nullptr);
    return std::strtod(raw, nullptr);
}

}  // namespace

TEST_CASE("hpotarget registry: the LAMMPS script of [lammps_script] takes "
          "the candidate potential of each evaluation",
          "[hpotarget_registry][hpotarget_lines][lammps_script]")
{
    tadah::core::PeriodicTable::initialize();
    const toml::value doc =
        toml::parse((registry_dir() / "hpotarget_lines.toml").string());
    REQUIRE(doc.contains("lammps_script"));
    const toml::value& ls = doc.at("lammps_script");
    const std::string fixture = ls.at("fixture").as_string().str;
    INFO("fixture " << fixture);
    const std::string text =
        doc.at("fixture").at(fixture).at("text").as_string().str;

    std::vector<std::string> names;
    for (const auto& ex : ls.at("examples").as_array())
        for (const auto& n : variables_of(ex.at("text").as_string().str))
            names.push_back(n);
    REQUIRE_FALSE(names.empty());

    const tadah::hpo::ScriptFile script(lines_of(text));
    CHECK(script.has_extern());

    const LammpsRunner runner;
    const auto lmp = runner.create_instance(
        {"lmp", "-log", "none", "-screen", "none"});
    REQUIRE(lmp);

    tadah::hpo::ScriptRunStatus why = tadah::hpo::ScriptRunStatus::OK;
    std::string error;
    REQUIRE(runner.run_script(lmp, script, potential(1.0), &why, &error));
    std::vector<double> first;
    for (const auto& n : names) {
        first.push_back(value_of(lmp, n));
        INFO(n << " = " << first.back());
        CHECK(std::isfinite(first.back()));
        CHECK(first.back() != 0.0);
    }

    /* The next evaluation, in the same instance: the script starts again
     * from clear and takes the new potential. */
    INFO(error);
    REQUIRE(runner.run_script(lmp, script, potential(2.0), &why, &error));
    for (std::size_t i = 0; i < names.size(); ++i) {
        INFO(names[i]);
        CHECK(value_of(lmp, names[i]) ==
              Approx(2.0 * first[i]).epsilon(1e-12));
    }
}
