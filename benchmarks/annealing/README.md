# Annealing benchmarks (opt-in)

Cross-revision and cross-oracle comparison tools for the TADAH annealing
family. Nothing here runs in CI; the CI-grade quality record is the
in-suite performance matrix (`tests/test_anneal_perf_matrix.cpp` against
`tests/data/anneal_perf_baseline.tsv`, statistical tag) and the
equilibrium suite (`ctest -L statistical`).

## Cross-revision A/B

Build the benchmark against any TADAH snapshot without editing its build
files, using project injection:

    cmake -S HPO -B build_bench \
      -DCMAKE_PROJECT_INCLUDE=$PWD/HPO/benchmarks/annealing/inject.cmake
    cmake --build build_bench --target hpo_annealing_benchmark -j
    ./build_bench/hpo_annealing_benchmark 50 > rev_A.tsv

Repeat on the other revision, then:

    ./summarize.py rev_A.tsv rev_B.tsv

`summarize.py` prints every cell whose success count moved with Wilson
95% intervals and flags differences resolved beyond CI overlap. Wall
times exclude output I/O and, on these microsecond objectives, measure
driver overhead -- not a substitute for real-workload timings.

## Baseline maintenance

`regen_perf_baseline.py <build_dir>` reruns the in-suite matrix in
record mode and prints the drift against the previous baseline (kept as
`.bak`). The review criterion for accepting a new baseline is
non-inferiority of the cells your change did not intend to move.

## External oracles (context, never gates)

* `run_asa_headtohead.sh` -- builds the pristine upstream ASA-30.53 drop
  and compares its canonical test run against Ingber's shipped reference
  outputs. (The bitwise vendored-vs-pristine golden gate runs in ctest.)
* `compare_scipy.py` -- scipy `dual_annealing` on the shared functions
  at the matrix budgets (independent algorithm family; context only).
