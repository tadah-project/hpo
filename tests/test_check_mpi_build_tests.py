#!/usr/bin/env python3
"""Unit tests of check_mpi_build_tests.py (CTest hpo_mpi_build_tests_lint_tool).

The check reads HPO's test sources as the MPI build (TADAH_HPO_MPI) compiles
them and fails a TEST_CASE that starts PC_ISOLATION PROCESS workers there
without SKIP_WITHOUT_PC_PROCESS_WORKERS() as its first statement, and any
LAMMPS instance a test source constructs without LammpsRunner. Synthetic
sources, checked as text and as a directory through the command line.
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

_spec = importlib.util.spec_from_file_location("check_mpi_build_tests", TOOL)
check = importlib.util.module_from_spec(_spec)
sys.modules[_spec.name] = check   # dataclasses look their module up
_spec.loader.exec_module(check)

SKIP = "SKIP_WITHOUT_PC_PROCESS_WORKERS();"


def case(body: str, name: str = "a case") -> str:
    return f'TEST_CASE("{name}", "[t]") {{\n{body}\n}}\n'


def problems(text: str) -> list:
    return check.check_text(text, "t.cpp")[0]


class ProcessRule(unittest.TestCase):
    def test_a_named_pool_without_the_skip_is_refused(self):
        p = problems(case("  PcWorkerPool pool(1, specs, 0.0);\n"
                          "  CHECK(pool.worker_pid(0) > 0);", "pool case"))
        self.assertEqual(len(p), 1, p)
        self.assertIn('t.cpp:1: TEST_CASE "pool case" builds a worker pool '
                      "('PcWorkerPool pool(', line 2)", p[0])
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
        for stmt in ("auto p = std::make_unique<PcWorkerPool>(1, s, 0.0);",
                     "auto p = std::make_shared<tadah::hpo::PcWorkerPool>(1, s, 0.0);",
                     "auto* p = new PcWorkerPool(1, s, 0.0);",
                     "PcWorkerPool healthy{1, s, 0.0};",
                     "pc_worker_pool_init(2, std::move(per_slot), 0.0);"):
            with self.subTest(stmt=stmt):
                self.assertEqual(len(problems(case("  " + stmt))), 1)

    def test_a_temporary_pool_is_not_counted(self):
        self.assertEqual(problems(case(
            "  CHECK_THROWS_AS(PcWorkerPool(0, {}, 0.0), "
            "std::invalid_argument);")), [])

    def test_a_pool_in_a_comment_or_a_string_is_not_counted(self):
        self.assertEqual(problems(case(
            '  // PcWorkerPool pool(1, specs, 0.0);\n'
            '  /* PcWorkerPool pool(1, specs, 0.0); */\n'
            '  INFO("PcWorkerPool pool(1, specs, 0.0);");')), [])

    def test_a_process_request_for_pc_lammps_lines_is_refused(self):
        p = problems(case(
            '  const std::string root = "PC_LAMMPS --script pe.in "\n'
            '      "--varloss E 0 1\\nPC_ISOLATION PROCESS\\n";\n'
            "  REQUIRE(hpo_run(ctx) == 0);"))
        self.assertEqual(len(p), 1, p)
        self.assertIn("requests PC_ISOLATION PROCESS for its PC_LAMMPS lines "
                      "(line 2)", p[0])

    def test_an_environment_request_counts(self):
        p = problems(case(
            '  ::setenv("TADAH_PC_ISOLATION", "PROCESS", 1);\n'
            '  TaRun run(opt, "PC_LAMMPS --script pe.in --varloss E 0 1\\n");'))
        self.assertEqual(len(p), 1, p)
        self.assertEqual(problems(case(
            '  ::setenv("TADAH_PC_ISOLATION", "THREAD", 1);\n'
            '  TaRun run(opt, "PC_LAMMPS --script pe.in --varloss E 0 1\\n");')),
            [])

    def test_a_request_without_pc_lammps_lines_is_not_counted(self):
        self.assertEqual(problems(case(
            '  TaRun run(opt + "PC_ISOLATION PROCESS\\n");\n'
            '  CHECK_THAT(err, Catch::Contains("cannot run"));')), [])

    def test_a_case_asserting_the_refusal_needs_no_skip(self):
        self.assertEqual(problems(case(
            '  TaRun run(opt, "PC_LAMMPS --script pe.in\\nPC_ISOLATION '
            'PROCESS\\n");\n'
            '  CHECK_THAT(err, Catch::Contains("PC_ISOLATION PROCESS is not '
            'supported in "\n                                  "the MPI build '
            '(TADAH_HPO_MPI)"));')), [])

    def test_a_mode_compared_with_process_is_no_request(self):
        self.assertEqual(problems(case(
            '  run(bin, "PC_LAMMPS --script pe.in\\n", mode);\n'
            '  if (mode == "PROCESS") CHECK(workers >= 1);')), [])


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

    def test_a_test_case_in_a_macro_definition_is_not_a_test_case(self):
        text = ("#define POOL_CASE(n) TEST_CASE(n) { \\\n"
                "    PcWorkerPool pool(1, specs, 0.0); \\\n  }\n")
        self.assertEqual(check.check_text(text, "t.cpp"), ([], 0))

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
        for stmt in ("auto* l = new LAMMPS_NS::LAMMPS(args, 0);",
                     "auto* l = new LAMMPS(args, MPI_COMM_SELF);",
                     "LAMMPS_NS::LAMMPS lmp(args, 0);",
                     "auto l = std::make_unique<LAMMPS_NS::LAMMPS>(args, 0);",
                     "LMPPtr l(new LMP(args, 0));",
                     "lammps_open_no_mpi(argc, argv, &h);"):
            for head, tail in (("", ""), ("#ifndef TADAH_HPO_MPI\n",
                                          "#endif\n")):
                with self.subTest(stmt=stmt, head=head):
                    p = problems(head + "void f() {\n  " + stmt + "\n}\n" +
                                 tail)
                    self.assertEqual(len(p), 1, p)
                    self.assertIn("constructs a LAMMPS instance directly", p[0])
                    self.assertIn("LammpsRunner::create_instance(tokens)", p[0])

    def test_the_runner_and_pointers_are_fine(self):
        self.assertEqual(problems(
            "void f(LAMMPS_NS::LAMMPS* lmp, LAMMPS_NS::LAMMPS &ref) {\n"
            "  auto l = LammpsRunner().create_instance({\"lmp\"});\n"
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
            (d / "not_a_test.h").write_text(case("  PcWorkerPool p(1, s, 0.0);"))
            r = self.run_tool(d)
            self.assertEqual(r.returncode, 1, r.stdout + r.stderr)
            self.assertIn(f'{d / "bad.cpp"}:1: TEST_CASE "bad"', r.stdout)
            self.assertIn("2 files, 2 test cases", r.stdout)
            self.assertEqual(self.run_tool(d / "missing").returncode, 2)
            self.assertEqual(self.run_tool().returncode, 2)


if __name__ == "__main__":
    unittest.main()
