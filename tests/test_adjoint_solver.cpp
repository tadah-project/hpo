// Phase 3.4 — DenseAdjointSolver tests.
//
// Pure Eigen — no LAMMPS, no pair_tadah. The solver is the
// LDLT-driven half of the IFT chain, fed by 3.2 (Hessian) and 3.3
// (observable rhs); verifying it in isolation lets Phase 3.5 trust
// the result.

#include "catch2/catch.hpp"

#include <tadah/hpo/adjoint_solver.h>

#include <Eigen/Dense>

#include <algorithm>
#include <cmath>

TEST_CASE("adjoint_solver — diagonal-plus-noise H: λ matches H.inverse() · rhs "
          "to 1e-12",
          "[adjoint_solver][phase3]")
{
    const int n = 12;
    Eigen::MatrixXd D = Eigen::MatrixXd::Identity(n, n) * 100.0;
    /* Deterministic small symmetric perturbation — Random() would
     * make the parity tolerance non-portable across builds. */
    Eigen::MatrixXd noise(n, n);
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < n; ++j)
            noise(i, j) = 0.01 * std::sin(0.5 * i + 0.3 * j);
    Eigen::MatrixXd H = D + 0.5 * (noise + noise.transpose());

    Eigen::VectorXd rhs(n);
    for (int i = 0; i < n; ++i) rhs(i) = 0.7 + 0.13 * i;

    /* n_atom_dofs=0 disables acoustic projection; we want a clean
     * matrix-inverse comparison without rhs being clipped. */
    tadah::hpo::DenseAdjointSolver solver(H, /*n_atom_dofs=*/0);
    Eigen::VectorXd lambda;
    solver.solve(rhs, lambda);
    CHECK_FALSE(solver.regularised());

    const Eigen::VectorXd expected = H.inverse() * rhs;
    const double rel = (lambda - expected).cwiseAbs().maxCoeff()
                       / std::max(1.0, expected.cwiseAbs().maxCoeff());
    CHECK(rel < 1e-12);
}

TEST_CASE("adjoint_solver — zero rhs returns zero λ via fast-path "
          "(no factorisation use)",
          "[adjoint_solver][phase3]")
{
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(6, 6) * 2.0;
    tadah::hpo::DenseAdjointSolver solver(H);
    Eigen::VectorXd zero = Eigen::VectorXd::Zero(6);
    Eigen::VectorXd lambda;
    solver.solve(zero, lambda);
    REQUIRE(lambda.size() == 6);
    CHECK(lambda.cwiseAbs().maxCoeff() == 0.0);
}

TEST_CASE("adjoint_solver — acoustic translation in rhs is projected out: "
          "λ identical with/without the constant shift",
          "[adjoint_solver][phase3]")
{
    const int Natoms = 3;
    const int n_atom = 3 * Natoms;
    const int n_box  = 1;
    const int n      = n_atom + n_box;

    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n) * 10.0;
    /* Tri-diagonal coupling so the solve isn't trivial. */
    for (int i = 0; i < n - 1; ++i) {
        H(i, i + 1) = 0.1;
        H(i + 1, i) = 0.1;
    }

    Eigen::VectorXd rhs1(n);
    for (int i = 0; i < n; ++i) rhs1(i) = 0.5 * (i + 1) - 1.0;

    /* rhs2 = rhs1 + uniform x-translation on the atom block. After
     * acoustic projection, both rhs vectors project to the same point
     * in row-space → λ must be identical. */
    Eigen::VectorXd rhs2 = rhs1;
    for (int i = 0; i < Natoms; ++i) rhs2(3 * i + 0) += 1.23;

    tadah::hpo::DenseAdjointSolver solver(H, n_atom);
    Eigen::VectorXd lam1, lam2;
    solver.solve(rhs1, lam1);
    solver.solve(rhs2, lam2);

    CHECK((lam1 - lam2).cwiseAbs().maxCoeff() < 1e-10);
}

TEST_CASE("adjoint_solver — Tikhonov bump kicks in on near-singular H",
          "[adjoint_solver][phase3]")
{
    const int n = 4;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n);
    /* Drive the smallest pivot below kPivotThreshold (1e-8). */
    H(0, 0) = 1e-15;

    tadah::hpo::DenseAdjointSolver solver(H, 0);
    CHECK(solver.regularised());

    /* After regularisation, the system is still solvable and gives a
     * sensible answer on a well-conditioned rhs (the rhs entry on the
     * tiny diagonal is effectively absorbed by the Tikhonov bump). */
    Eigen::VectorXd rhs = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd lambda;
    solver.solve(rhs, lambda);
    CHECK(std::isfinite(lambda.norm()));
}

TEST_CASE("adjoint_solver — round-trip H·λ ≈ rhs_projected within 1e-8",
          "[adjoint_solver][phase3]")
{
    const int Natoms = 4;
    const int n_atom = 3 * Natoms;
    const int n_box  = 1;
    const int n      = n_atom + n_box;

    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n) * 10.0;
    for (int i = 0; i < n - 1; ++i) {
        H(i, i + 1) = 0.05;
        H(i + 1, i) = 0.05;
    }

    Eigen::VectorXd rhs(n);
    for (int i = 0; i < n; ++i)
        rhs(i) = (i % 2 == 0 ? 1.0 : -1.0) * 0.3 * (i + 1);

    tadah::hpo::DenseAdjointSolver solver(H, n_atom);
    Eigen::VectorXd lambda;
    solver.solve(rhs, lambda);

    /* Manually project rhs the same way the solver does (per-axis
     * centroid subtraction over the atom block) to verify the round
     * trip on the projected system, not the raw rhs. */
    Eigen::VectorXd rhs_proj = rhs;
    for (int axis = 0; axis < 3; ++axis) {
        double sum = 0.0;
        for (int i = 0; i < Natoms; ++i) sum += rhs_proj(3 * i + axis);
        const double mean = sum / static_cast<double>(Natoms);
        for (int i = 0; i < Natoms; ++i) rhs_proj(3 * i + axis) -= mean;
    }

    const double diff = (H * lambda - rhs_proj).cwiseAbs().maxCoeff();
    CHECK(diff < 1e-8);
}

TEST_CASE("adjoint_solver — rejects size mismatches and non-square H",
          "[adjoint_solver][phase3]")
{
    Eigen::MatrixXd Hsq = Eigen::MatrixXd::Identity(5, 5);
    Eigen::MatrixXd Hnon(5, 3);
    Hnon.setIdentity();

    CHECK_THROWS_AS(tadah::hpo::DenseAdjointSolver(Hnon, 0),
                    std::runtime_error);

    /* n_atom_dofs > rows is rejected. */
    CHECK_THROWS_AS(tadah::hpo::DenseAdjointSolver(Hsq, 6),
                    std::runtime_error);

    /* Once constructed, solve() rejects rhs-size mismatch. */
    tadah::hpo::DenseAdjointSolver s(Hsq, 0);
    Eigen::VectorXd bad_rhs(3);
    bad_rhs.setOnes();
    Eigen::VectorXd lambda;
    CHECK_THROWS_AS(s.solve(bad_rhs, lambda), std::runtime_error);
}

TEST_CASE("adjoint_solver — multiple solves on the same factorisation "
          "stay consistent (factorisation isn't invalidated between calls)",
          "[adjoint_solver][phase3]")
{
    const int n = 8;
    Eigen::MatrixXd H = Eigen::MatrixXd::Identity(n, n) * 5.0;
    for (int i = 0; i < n - 1; ++i) {
        H(i, i + 1) = 0.2;
        H(i + 1, i) = 0.2;
    }

    tadah::hpo::DenseAdjointSolver solver(H, 0);

    Eigen::VectorXd rhs1 = Eigen::VectorXd::Ones(n);
    Eigen::VectorXd rhs2(n);
    for (int i = 0; i < n; ++i) rhs2(i) = std::cos(0.5 * i);

    Eigen::VectorXd lam1, lam2, lam1_again;
    solver.solve(rhs1, lam1);
    solver.solve(rhs2, lam2);
    solver.solve(rhs1, lam1_again);

    CHECK((lam1 - lam1_again).cwiseAbs().maxCoeff() < 1e-14);

    /* Linearity: H · (lam1 + lam2) = rhs1 + rhs2. */
    const Eigen::VectorXd lam_sum = lam1 + lam2;
    const Eigen::VectorXd rhs_sum = rhs1 + rhs2;
    const double diff = (H * lam_sum - rhs_sum).cwiseAbs().maxCoeff();
    CHECK(diff < 1e-10);
}
