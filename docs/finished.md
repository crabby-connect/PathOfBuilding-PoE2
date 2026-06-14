# Finished — Tree-Optimizer wiring

Completed work, newest first. Counterpart to `docs/todo.md`.

## 2026-06-14

- **Promoted the beam search into the crate (production search loop).** Extracted the
  full beam + Pareto + elimination-diet search out of `examples/beam_ab.rs` into a real
  module `rust/pob-optimizer/src/beam.rs` (`Graph`, `BeamParams`, `optimize()`), behind a
  new FFI entry **`pob_opt_run_beam`** that runs the whole search inside the pool and
  returns the winning node-id set + (score, dps, ehp). Added `crate-type=["cdylib","rlib"]`
  so the search is unit-testable; added two `cargo test --lib` tests (param parsing +
  synthetic-graph growth). Re-pointed `beam_ab.rs` to drive `pob_opt_run_beam` via the
  shipped DLL (no duplicated search), and fixed the size-then-fetch idiom to a single call
  sized from `cap_points + 1` (the old code re-ran the whole minutes-long search just to
  count). Verified end-to-end against `mymonk.xml`: pool boots, graph exports (4478 nodes,
  P=127), microbench ~2.3× speedup with par==seq match, beam+diet converges, save succeeds,
  `BEAM A/B HARNESS PASSED`.

- **Exposed `OptimizerPool:runBeam(params)`** in `src/Modules/OptimizerPool.lua` (camelCase
  tunables → the cdylib's key=value), sizing the output buffer from the budget so the search
  runs once. Added the `pob_opt_run_beam` cdef.

- **Deploy step** `rust/deploy-optimizer.ps1`: `cargo build --release` + copy
  `pob_optimizer.dll` into `runtime/` (with `-SkipBuild`). Gitignored the deployed DLL
  (`runtime/pob_optimizer.dll`). Verified the copy works.

- **Extended `examples/smoke.rs`** to cover the newer FFI surface: a second clean-slate pool
  asserts `score_batch3` returns finite (score, dps, ehp), the graph export parses with a
  positive budget/node count, the `w_dps`/`w_ehp` weights actually move the base score
  (88.7 vs 150.1), and `call_save` writes a valid non-empty XML. `SMOKE TEST PASSED`.

- **Refreshed `docs/rust-optimizer-integration.md`** (was stale from 2026-06-12): documented
  `score_batch3`, `run_beam`, `call_save`, the config weights, the clean-slate /
  graph-export / save worker fns, `beam_ab.rs`, the deploy script, and narrowed "Still open"
  to the UI + an optional packaging-CI hook.

- **Set up progress trackers** `docs/todo.md` + `docs/finished.md`.

(Prior work — the Rust cdylib, pool, FFI, and the score3/weights/graph-export/save additions
landed in commit 82d71ca46 — is recorded in git history and the design/integration docs.)
