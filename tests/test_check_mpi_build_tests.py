#!/usr/bin/env python3
"""Unit tests of check_mpi_build_tests.py (CTest hpo_mpi_build_tests_lint_tool).

The check reads HPO's test sources as the MPI build (TADAH_HPO_MPI) compiles
them and fails a TEST_CASE that builds a PcWorkerPool or requests
PC_ISOLATION PROCESS there without SKIP_WITHOUT_PC_PROCESS_WORKERS() as its
first statement, such a pool or request outside every TEST_CASE body, and
any use of the LAMMPS class other than as a pointer, a reference or a
smart-pointer holder (an instance comes from LammpsRunner::create_instance).
Synthetic sources, checked as text and as a directory through the command
line. A test that passed against the check before it matched the classes
and every spelling of a request says so ("a pin"); one whose expectation
changed says what it was.
"""
from __future__ import annotations

import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

HERE = Path(__file__).resolve().parent
TOOL = HERE / "check_mpi_build_tests.py"

sys.dont_write_bytecode = True   # no __pycache__ in the source tree
_spec = importlib.util.spec_from_file_location("check_mpi_build_tests", TOOL)
check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check   # dataclasses look their module up
_spec.loader.exec_module(check)

SKIP = "SKIP_WITHOUT_PC_PROCESS_WORKERS();"
LAMMPS_LINE = '"PC_LAMMPS --script pe.in --varloss E 0 1\\n"'


def case(body: str, name: str = "a case") -> str:
    return f'TEST_CASE("{name}", "[t]") {{\n{body}\n}}\n'


def problems(text: str) -> list:
    return check.check_text(text, "t.cpp")[0]


class ProcessRule(unittest.TestCase):
    def test_a_named_pool_without_the_skip_is_refused(self):
        # The message quotes the line (it quoted 'PcWorkerPool pool(').
        p = problems(case("  PcWorkerPool pool(1, specs, 0.0);\n"
                          "  CHECK(pool.worker_pid(0) > 0);", "pool case"))
        self.assertEqual(len(p), 1, p)
        self.assertIn('t.cpp:1: TEST_CASE "pool case" builds a worker pool '
                      "('PcWorkerPool pool(1, specs, 0.0);', line 2)", p[0])
        self.assertIn("SKIP_WITHOUT_PC_PROCESS_WORKERS()", p[0])

    def test_the_skip_as_first_statement_satisfies_it(self):
        self.assertEqual(problems(case(
            "  /* why */ " + SKIP + "\n  PcWorkerPool pool(1, specs, 0.0);")),
            [])

    def test_the_skip_after_another_statement_does_not(self):
        p = problems(case("  Journal journal;\n  " + SKIP +
                          "\n  PcWorkerPool pool(1, specs, 0.0);"))
        self.assertEqual(len(p), 1, p)

    def test_the_skip_inside_a_section_does_not(self):
        p = problems(case('  SECTION("s") { ' + SKIP + " }\n"
                          "  PcWorkerPool pool(1, specs, 0.0);"))
        self.assertEqual(len(p), 1, p)

    def test_every_way_of_building_a_pool_counts(self):
        # Five of these -- the copy-initialised pool (C++17 elides the
        # copy), the braced temporary, std::optional's emplace, the
        # temporary in a check and the vector of pools -- passed the check
        # while it matched a named or allocated pool only.
        for stmt in ("auto p = std::make_unique<PcWorkerPool>(1, s, 0.0);",
                     "auto p = std::make_shared<tadah::hpo::PcWorkerPool>(1, s, 0.0);",
                     "auto* p = new PcWorkerPool(1, s, 0.0);",
                     "PcWorkerPool healthy{1, s, 0.0};",
                     "pc_worker_pool_init(2, std::move(per_slot), 0.0);",
                     "auto pool = PcWorkerPool(2, specs_for(2, sp), 0.0);",
                     "const auto pool = tadah::hpo::PcWorkerPool{1, s, 0.0};",
                     "std::optional<PcWorkerPool> pool; pool.emplace(1, s, 0.0);",
                     "CHECK(PcWorkerPool(1, s, 0.0).width() == 1);",
                     "auto* p = new PcWorkerPool{1, s, 0.0};",
                     "tadah :: hpo :: PcWorkerPool pool(1, s, 0.0);",
                     "std::vector<PcWorkerPool> pools;"):
            with self.subTest(stmt=stmt):
                self.assertEqual(len(problems(case("  " + stmt))), 1)

    def test_a_pool_inside_a_throws_check_counts(self):
        # A pool built in a CHECK_THROWS* is still built: a worker that
        # fails it would pass the check in the MPI build for the wrong
        # reason. It was not counted (a temporary pool, the width check).
        for stmt in ("CHECK_THROWS_AS(PcWorkerPool(0, {}, 0.0), "
                     "std::invalid_argument);",
                     "REQUIRE_THROWS_WITH(PcWorkerPool(1, s, 0.0), "
                     "Catch::Contains(\"handshake\"));"):
            with self.subTest(stmt=stmt):
                self.assertEqual(len(problems(case("  " + stmt))), 1)

    def test_pointers_references_holders_and_scopes_are_no_pool(self):
        # Before the fix: passes as well (a pin: these build no pool).
        self.assertEqual(problems(case(
            "  auto use = [](PcWorkerPool& p, const PcWorkerPool* q,\n"
            "                PcWorkerPool const & r) {};\n"
            "  std::unique_ptr<PcWorkerPool> held;\n"
            "  std::shared_ptr<tadah::hpo::PcWorkerPool> shared;\n"
            "  CHECK(PcWorkerPool::PROTOCOL_VERSION > 0u);\n"
            "  CHECK(tadah::hpo::PcWorkerPool::PC_MAX_CONSEC_DEATHS > 0);")),
            [])

    def test_a_pool_in_a_comment_or_a_string_is_not_counted(self):
        self.assertEqual(problems(case(
            '  // PcWorkerPool pool(1, specs, 0.0);\n'
            '  /* PcWorkerPool pool(1, specs, 0.0); */\n'
            '  INFO("PcWorkerPool pool(1, specs, 0.0);");')), [])

    def test_a_process_request_for_pc_lammps_lines_is_refused(self):
        # The message quotes the request (it said "requests PC_ISOLATION
        # PROCESS for its PC_LAMMPS lines (line 2)").
        p = problems(case(
            '  const std::string root = "PC_LAMMPS --script pe.in "\n'
            '      "--varloss E 0 1\\nPC_ISOLATION PROCESS\\n";\n'
            "  REQUIRE(hpo_run(ctx) == 0);"))
        self.assertEqual(len(p), 1, p)
        self.assertIn('requests PC_ISOLATION PROCESS ("PC_LAMMPS --script '
                      'pe.in --varloss E 0 1 PC_ISOLATION PROCESS", line 2)',
                      p[0])

    def test_an_environment_request_counts(self):
        p = problems(case(
            '  ::setenv("TADAH_PC_ISOLATION", "PROCESS", 1);\n'
            '  TaRun run(opt, "PC_LAMMPS --script pe.in --varloss E 0 1\\n");'))
        self.assertEqual(len(p), 1, p)
        self.assertEqual(problems(case(
            '  ::setenv("TADAH_PC_ISOLATION", "THREAD", 1);\n'
            '  TaRun run(opt, "PC_LAMMPS --script pe.in --varloss E 0 1\\n");')),
            [])

    def test_a_request_counts_without_pc_lammps_lines(self):
        # The MPI build refuses every explicit PROCESS, whatever lines the
        # configuration has. A request without a "PC_LAMMPS " literal in
        # the case was not counted.
        for body in ('  TaRun run(opt + "PC_ISOLATION PROCESS\\n");\n'
                     '  REQUIRE(hpo_run(run.ctx) == 0);',
                     '  ::setenv("TADAH_PC_ISOLATION", "PROCESS", 1);\n'
                     '  TaRun run(opt);\n  REQUIRE(hpo_run(run.ctx) == 0);'):
            with self.subTest(body=body):
                p = problems(case(body))
                self.assertEqual(len(p), 1, p)
                self.assertIn("requests PC_ISOLATION PROCESS", p[0])

    def test_every_spelling_of_a_request_counts(self):
        # The first, fourth, fifth and sixth were seen while the check
        # matched a literal holding "PC_ISOLATION PROCESS" (or one setenv
        # statement); the other nine passed it.
        for stmt in ('TaRun run(opt + "PC_ISOLATION PROCESS\\n");',
                     'TaRun run(opt + "PC_ISOLATION  PROCESS\\n");',
                     'TaRun run(opt + "PC_ISOLATION\\tPROCESS\\n");',
                     'TaRun run(opt + "x\\n  PC_ISOLATION PROCESS  # pool\\n");',
                     'TaRun run(opt + "PC_ISOLATION " "PROCESS" "\\n");',
                     'TaRun run(opt + R"(PC_LAMMPS x\nPC_ISOLATION PROCESS\n)");',
                     'const std::string isolation = "PROCESS";\n'
                     '  TaRun run(opt + "PC_ISOLATION " + isolation + "\\n");',
                     'const char* mode = "PROCESS";\n'
                     '  ::setenv("TADAH_PC_ISOLATION", mode, 1);',
                     'const std::vector<std::string> modes{"THREAD", "PROCESS"};',
                     'run_tadah_hpo(bin, dir, "PROCESS", target, script, {});',
                     'run_tadah_hpo(bin, dir, " PROCESS ", target, script, {});',
                     'std::system("env TADAH_PC_ISOLATION=PROCESS tadah hpo");',
                     'run_self("[x]", dir, {"TADAH_PC_ISOLATION=PROCESS"});'):
            with self.subTest(stmt=stmt):
                p = problems(case("  " + stmt + "\n  " + LAMMPS_LINE + ";"))
                self.assertEqual(len(p), 1, p)
                self.assertIn("requests PC_ISOLATION PROCESS", p[0])

    def test_text_that_mentions_process_is_no_request(self):
        # The second INFO, the skip reason and the line with two values
        # were counted while any literal that held "PC_ISOLATION PROCESS"
        # was a request; the rest passed then as well (pins), the TEST_CASE
        # name included.
        for stmt in ('CHECK_THAT(j, Catch::Contains("PC_ISOLATION: PROCESS '
                     '-- pc_lammps tasks execute in"));',
                     'INFO(name << ", PC_ISOLATION " << mode);',
                     'INFO("PC_ISOLATION PROCESS in the MPI build");',
                     'if (mode == "PROCESS") CHECK(workers >= 1);',
                     'CHECK(("PROCESS" != mode) == thread);',
                     'TADAH_SKIP("PC_ISOLATION PROCESS is serial-build only");',
                     'const fs::path dir = sd.dir / (name + "_PROCESS");',
                     'TaRun run(opt + "PC_ISOLATION THREAD\\n");',
                     'TaRun run(opt + "PC_ISOLATION PROCESS THREAD\\n");',
                     'SECTION("PROCESS") { CHECK(true); }'):
            with self.subTest(stmt=stmt):
                self.assertEqual(problems(case(
                    "  " + stmt + "\n  " + LAMMPS_LINE + ";")), [])
        self.assertEqual(problems(case(
            "  CHECK(true);", "TADAH_PC_ISOLATION=PROCESS, no PC_LAMMPS "
            "line")), [])

    def test_a_case_asserting_the_refusal_needs_no_skip(self):
        self.assertEqual(problems(case(
            '  TaRun run(opt, "PC_LAMMPS --script pe.in\\nPC_ISOLATION '
            'PROCESS\\n");\n'
            '  CHECK_THAT(err, Catch::Contains("PC_ISOLATION PROCESS is not '
            'supported in "\n                                  "the MPI build '
            '(TADAH_HPO_MPI)"));')), [])

    def test_a_refusal_in_an_enclosing_block_exempts_its_sections(self):
        # Before the fix: passes as well (a pin: test_hpo_termination.cpp's
        # PC_ISOLATION journal case, refusal at the top of its MPI branch).
        self.assertEqual(problems(case(
            '  const std::string refusal = "PC_ISOLATION PROCESS is not '
            'supported in the MPI build";\n'
            '  SECTION("config") {\n'
            '    TaRun run(opt + "PC_ISOLATION PROCESS\\n");\n'
            "    CHECK_THROWS_WITH(hpo_run(run.ctx), refusal);\n  }\n"
            '  SECTION("environment") {\n'
            '    ::setenv("TADAH_PC_ISOLATION", "PROCESS", 1);\n'
            "    CHECK_THROWS_WITH(hpo_run(run.ctx), refusal);\n  }")), [])

    def test_the_refusal_exempts_its_own_block_only(self):
        # A sibling block that requests PROCESS and checks something else
        # fails in the MPI build. The refusal anywhere in the case exempted
        # the whole case.
        p = problems(case(
            '  SECTION("refused") {\n'
            '    TaRun run(opt + "PC_ISOLATION PROCESS\\n");\n'
            '    CHECK_THROWS_WITH(hpo_run(run.ctx), Catch::Contains(\n'
            '        "is not supported in the MPI build"));\n  }\n'
            '  SECTION("runs") {\n'
            '    TaRun run(opt + ' + LAMMPS_LINE + ' "PC_ISOLATION PROCESS\\n");\n'
            "    REQUIRE(hpo_run(run.ctx) == 0);\n  }"))
        self.assertEqual(len(p), 1, p)
        self.assertIn('", line 8)', p[0])

    def test_a_mode_compared_with_process_is_no_request(self):
        self.assertEqual(problems(case(
            '  run(bin, "PC_LAMMPS --script pe.in\\n", mode);\n'
            '  if (mode == "PROCESS") CHECK(workers >= 1);')), [])


class OutsideCases(unittest.TestCase):
    """A pool or a request outside every TEST_CASE body: the check cannot
    tell which case reaches it, so it is refused in the MPI build's view.
    The check read case bodies only: each of these passed it."""

    def test_a_request_outside_every_case_is_refused(self):
        for text in ('static const std::string kProc = '
                     '"PC_ISOLATION PROCESS\\n";\n',
                     'void run_process(TaRun& r) {\n'
                     '  ::setenv("TADAH_PC_ISOLATION", "PROCESS", 1);\n}\n',
                     'struct Modes { std::vector<std::string> all{"THREAD", '
                     '"PROCESS"}; };\n',
                     '#define PROCESS_LINE "PC_ISOLATION PROCESS\\n"\n',
                     'namespace {\nconst char* const kMode = "PROCESS";\n}\n'):
            with self.subTest(text=text):
                p = problems(text + case("  CHECK(true);"))
                self.assertEqual(len(p), 1, p)
                self.assertIn("t.cpp:", p[0])
                self.assertIn("requests PC_ISOLATION PROCESS", p[0])
                self.assertIn("outside every TEST_CASE body in the MPI "
                              "build's view", p[0])

    def test_a_pool_outside_every_case_is_refused(self):
        for text in ("void build(int w) {\n"
                     "  PcWorkerPool pool(w, specs_for(w, sp), 0.0);\n}\n",
                     "std::unique_ptr<PcWorkerPool> make(int w) {\n"
                     "  return std::make_unique<PcWorkerPool>(w, s, 0.0);\n}\n",
                     "#define POOL(w) PcWorkerPool pool(w, s, 0.0)\n"):
            with self.subTest(text=text):
                p = problems(text + case("  CHECK(true);"))
                self.assertEqual(len(p), 1, p)
                self.assertIn("builds a worker pool", p[0])
                self.assertIn("outside every TEST_CASE body", p[0])

    def test_a_helper_that_takes_the_mode_or_the_pool_is_fine(self):
        # Before the fix: passes as well (a pin: run_tadah_hpo's shape).
        self.assertEqual(problems(
            "int run_tadah_hpo(const std::string& mode) {\n"
            '  std::ofstream("config.hpo") << "PC_ISOLATION " << mode << "\\n";\n'
            "  return 0;\n}\n"
            "void drive(PcWorkerPool& pool) { (void)pool.width(); }\n"
            'bool process_mode(const std::string& m) { return m == "PROCESS"; }\n'
            + case("  CHECK(true);")), [])

    def test_code_the_mpi_build_does_not_compile_is_not_checked(self):
        # Before the fix: passes as well (a pin).
        self.assertEqual(problems(
            "#ifndef TADAH_HPO_MPI\n"
            'static const std::string kProc = "PC_ISOLATION PROCESS\\n";\n'
            "#define POOL(w) PcWorkerPool pool(w, s, 0.0)\n"
            "#endif\n" + case("  CHECK(true);")), [])

    def test_a_test_case_in_a_macro_definition_is_no_case_and_its_pool_counts(self):
        # It counted as no test case and nothing in it was checked
        # (([], 0)): a macro's pool is outside every case the check sees.
        text = ("#define POOL_CASE(n) TEST_CASE(n) { \\\n"
                "    PcWorkerPool pool(1, specs, 0.0); \\\n  }\n")
        p, n = check.check_text(text, "t.cpp")
        self.assertEqual(n, 0)
        self.assertEqual(len(p), 1, p)
        self.assertIn("t.cpp:2: builds a worker pool", p[0])


class MpiView(unittest.TestCase):
    POOL = case("  PcWorkerPool pool(1, specs, 0.0);")

    def test_a_case_the_mpi_build_does_not_compile_is_not_checked(self):
        for head, tail in (("#ifndef TADAH_HPO_MPI\n", "#endif\n"),
                           ("#if !defined(TADAH_HPO_MPI)\n", "#endif\n"),
                           ("#ifdef TADAH_HPO_MPI\n#else\n", "#endif\n"),
                           ("#if defined TADAH_HPO_MPI\nint x;\n#else\n",
                            "#endif\n")):
            with self.subTest(head=head):
                self.assertEqual(problems(head + self.POOL + tail), [])

    def test_a_case_the_mpi_build_compiles_is_checked(self):
        for head, tail in (("#ifdef TADAH_HPO_MPI\n", "#endif\n"),
                           ("#if defined(TADAH_HPO_MPI)\n", "#endif\n"),
                           ("#ifndef TADAH_HPO_MPI\nint x;\n#else\n",
                            "#endif\n"),
                           ("#ifdef OTHER\n", "#endif\n"),
                           ("#ifdef OTHER\nint x;\n#else\n", "#endif\n"),
                           ("#ifndef TADAH_HPO_MPI\nint x;\n#elif OTHER\n",
                            "#endif\n"),
                           ("#ifndef TADAH_HPO_MPI\nint x;\n#elif OTHER\n"
                            "int y;\n#else\n", "#endif\n"),
                           ("#ifdef OTHER\n#ifndef TADAH_HPO_MPI\nint x;\n"
                            "#endif\n", "#endif\n")):
            with self.subTest(head=head):
                self.assertEqual(len(problems(head + self.POOL + tail)), 1)

    def test_an_mpi_branch_followed_by_elif_drops_the_elif(self):
        self.assertEqual(problems("#ifdef TADAH_HPO_MPI\nint x;\n#elif OTHER\n"
                                  + self.POOL + "#endif\n"), [])

    def test_only_the_skipped_section_of_a_case_is_dropped(self):
        p = problems(case("#ifndef TADAH_HPO_MPI\n  " + SKIP +
                          "\n#endif\n  PcWorkerPool pool(1, specs, 0.0);"))
        self.assertEqual(len(p), 1, p)

    def test_braces_and_quotes_in_literals_do_not_end_a_body(self):
        p = problems(case("  const std::string s = \"}\";\n"
                          "  const char c = '}';\n"
                          "  const long n = 1'000'000;\n"
                          "  const char* r = R\"x(})x\";\n"
                          "  PcWorkerPool pool(1, specs, 0.0);"))
        self.assertEqual(len(p), 1, p)

    def test_the_case_name_joins_adjacent_literals(self):
        p = problems('TEST_CASE("first part, "\n          "second part", '
                     '"[t]") {\n  PcWorkerPool pool(1, specs, 0.0);\n}\n')
        self.assertIn('TEST_CASE "first part, second part"', p[0])


class LammpsRule(unittest.TestCase):
    def test_a_direct_lammps_instance_is_refused_in_every_view(self):
        # The message names the rule (it said "constructs a LAMMPS
        # instance directly" for the first six). The forms after the sixth
        # passed the check while it matched parenthesised spellings of
        # LAMMPS_NS::LAMMPS and new LMP; the object, brace and temporary
        # forms compile in both builds with a literal 0, a null MPI_Comm
        # under Open MPI.
        for stmt in ("auto* l = new LAMMPS_NS::LAMMPS(args, 0);",
                     "auto* l = new LAMMPS(args, MPI_COMM_SELF);",
                     "LAMMPS_NS::LAMMPS lmp(args, 0);",
                     "auto l = std::make_unique<LAMMPS_NS::LAMMPS>(args, 0);",
                     "LMPPtr l(new LMP(args, 0));",
                     "lammps_open_no_mpi(argc, argv, &h);",
                     "LammpsRunner::LMP lmp(args, 0);",
                     "tadah::hpo::LammpsRunner::LMP lmp{args, 0};",
                     "using namespace LAMMPS_NS; LAMMPS lmp(args, 0);",
                     "LMP lmp(args, 0);",
                     "LMPPtr l{new LMP{args, 0}};",
                     "auto* l = new LAMMPS_NS::LAMMPS{args, 0};",
                     "auto l = std::make_unique<LMP>(args, MPI_Comm(0));",
                     "auto l = std::make_shared<LammpsRunner::LMP>(args, c);",
                     "auto l = LAMMPS_NS::LAMMPS(args, 0);",
                     "std::optional<LAMMPS_NS::LAMMPS> l; l.emplace(args, 0);",
                     "using Instance = LAMMPS_NS::LAMMPS;",
                     "typedef LammpsRunner::LMP Instance;"):
            for head, tail in (("", ""), ("#ifndef TADAH_HPO_MPI\n",
                                          "#endif\n")):
                with self.subTest(stmt=stmt, head=head):
                    p = problems(head + "void f() {\n  " + stmt + "\n}\n" +
                                 tail)
                    self.assertEqual(len(p), 1, p)
                    self.assertIn("names the LAMMPS class other than as a "
                                  "pointer, a reference or a smart-pointer "
                                  "holder", p[0])
                    self.assertIn("LammpsRunner::create_instance(tokens)", p[0])

    def test_the_runner_and_pointers_are_fine(self):
        # Before the fix: passes as well (a pin: no instance is built).
        self.assertEqual(problems(
            "void f(LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::LAMMPS &ref,\n"
            "       const LammpsRunner::LMP& alias, LMP const* p) {\n"
            "  auto l = LammpsRunner().create_instance({\"lmp\"});\n"
            "  LammpsRunner::LMPPtr kept = std::move(l);\n"
            "  std::unique_ptr<LAMMPS_NS::LAMMPS> held;\n"
            "  auto* pair = dynamic_cast<LAMMPS_NS::PairTadah*>(lmp->force->pair);\n"
            "  auto* same = static_cast<LAMMPS_NS::LAMMPS*>(lmp);\n"
            "  Engine e = Engine::LAMMPS; (void)obj.LMP;\n"
            "  // new LAMMPS_NS::LAMMPS(args, 0) in a comment\n}\n"), [])


class CommandLine(unittest.TestCase):
    def run_tool(self, *dirs):
        return subprocess.run([sys.executable, str(TOOL), *map(str, dirs)],
                              capture_output=True, text=True)

    def test_exit_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            r = self.run_tool(d)
            self.assertEqual(r.returncode, 2, r.stdout + r.stderr)
            self.assertIn("no TEST_CASE", r.stderr)
            (d / "good.cpp").write_text(case("  " + SKIP + "\n"
                                             "  PcWorkerPool p(1, s, 0.0);"))
            r = self.run_tool(d)
            self.assertEqual(r.returncode, 0, r.stdout + r.stderr)
            self.assertIn("1 files, 1 test cases in the MPI build's view, "
                          "0 violation(s)", r.stdout)
            (d / "bad.cpp").write_text(case("  PcWorkerPool p(1, s, 0.0);",
                                            "bad"))
            r = self.run_tool(d)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn(f'{d / "bad.cpp"}:1: TEST_CASE "bad"', r.stdout)
            self.assertIn("2 files, 2 test cases", r.stdout)
            self.assertEqual(self.run_tool(d / "missing").returncode, 2)
            self.assertEqual(self.run_tool().returncode, 2)

    def test_headers_and_subdirectories_are_read(self):
        # A test helper in a header or a subdirectory is compiled into the
        # tests as well. Only *.cpp directly under the directory was read.
        with tempfile.TemporaryDirectory() as tmp:
            d = Path(tmp)
            (d / "good.cpp").write_text(case("  CHECK(true);"))
            (d / "helper.h").write_text(
                "inline void start() {\n  LammpsRunner::LMP lmp(args, 0);\n}\n")
            (d / "sub").mkdir()
            (d / "sub" / "more.hpp").write_text(
                'inline const char* kMode = "PROCESS";\n')
            (d / "sub" / "deeper.cpp").write_text(
                case("  PcWorkerPool p(1, s, 0.0);", "deep"))
            (d / "notes.txt").write_text("PcWorkerPool p(1, s, 0.0);\n")
            r = self.run_tool(d)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn(f"{d / 'helper.h'}:2: names the LAMMPS class",
                          r.stdout)
            self.assertIn(f"{d / 'sub' / 'more.hpp'}:1: requests "
                          "PC_ISOLATION PROCESS", r.stdout)
            self.assertIn(f'{d / "sub" / "deeper.cpp"}:1: TEST_CASE "deep"',
                          r.stdout)
            self.assertIn("4 files, 2 test cases in the MPI build's view, "
                          "3 violation(s)", r.stdout)


if __name__ == "__main__":
    unittest.main()
