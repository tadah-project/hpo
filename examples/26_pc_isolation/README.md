# 26 -- PC_LAMMPS worker processes and a hard deadline

Example 01 with its LAMMPS execution spelled out.

What this shows:
- `PC_ISOLATION PROCESS` (the default): each worker slot's LAMMPS
  instances run in a persistent child process, so concurrent instances
  never share LAMMPS' process globals; training and prediction stay in
  the main process. The journal says `PC_ISOLATION: PROCESS -- ...`.
- `PC_WORKER_TIMEOUT 600`: a worker whose script runs longer than 600 s
  (a hung run) is killed and the evaluation fails. Without it the
  deadline derives from each script's own `PC_SCRIPT_TIMEOUT`, and a
  script without one (as here) has none. The journal prints the soft
  and hard limits.
- The isolation mode does not change results: the reference fingerprint
  is example 01's, and `TADAH_PC_ISOLATION=THREAD` (the in-process
  fallback, for environments that block the worker socket) gives the
  same output files. The fingerprint also requires the journal's
  `PC_ISOLATION: PROCESS` line and the 600 s hard deadline. Under
  `TADAH_PC_ISOLATION=THREAD` the reference suite does not apply those
  two checks and prints a `NOTE` for each, so it verifies PROCESS mode
  only when it runs without that override.

Run:

    bash ../common/run.sh
