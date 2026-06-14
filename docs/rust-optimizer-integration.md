# Rust Tree-Optimizer Integration (production cdylib)

**Status:** integrated (2026-06-12); search loop promoted into the crate (2026-06-14);
growth/penalty score split protecting the DPS/EHP floor (2026-06-14).
Code in `rust/pob-optimizer/`.
**Builds on:** `docs/rust-offload-poc.md` (the PoC that proved the worker model)
and `docs/tree-optimizer-design.md` (the search this feeds).
**Progress trackers:** `docs/todo.md` / `docs/finished.md`.

This is the production form of the parallel tree-optimizer search driver. The PoC
proved N independent LuaJIT states can each boot the headless calc engine and run
`calcs.perform` on N threads (~3.4× ceiling, memory-bandwidth bound). This turns
that into a shippable **cdylib** + a **persistent worker pool** + a **LuaJIT FFI**
boundary the host PoB state calls — and, as of 2026-06-14, **the full beam +
Pareto + elimination-diet search itself runs inside the crate** (`src/beam.rs`,
behind `pob_opt_run_beam`), so the host kicks off one call and reads back a result
rather than orchestrating the loop on its single state.

The calc engine stays canonical upstream Lua. Rust is the search *driver* and now
the search *loop*; the per-candidate scoring is still canonical Lua in the workers.

## Pieces

| File | Role |
| --- | --- |
| `rust/pob-optimizer/src/lib.rs` | crate root; re-exports the FFI surface |
| `rust/pob-optimizer/src/ffi.rs` | the `extern "C"` boundary (the only public surface) |
| `rust/pob-optimizer/src/pool.rs` | persistent worker pool (N threads, one LuaJIT state each, booted once and reused) |
| `rust/pob-optimizer/src/lua.rs` | minimal LuaJIT C-API binding, resolved from the shipped `runtime/lua51.dll` |
| `rust/pob-optimizer/src/config.rs` | parses the `key=value` config string from the host (incl. `w_dps`/`w_ehp` weights, `penalty_k`) |
| `rust/pob-optimizer/src/beam.rs` | the production search: beam + Pareto + elimination diet (graph parse, `optimize()`, `penalized_score()`) |
| `rust/pob-optimizer/worker_bootstrap.lua` | per-worker boot: engine + build + the score/candidate/graph-export/save fns |
| `rust/pob-optimizer/examples/smoke.rs` | end-to-end test, calling the C ABI exactly as Lua does |
| `rust/pob-optimizer/examples/beam_ab.rs` | native A/B harness: microbench (par vs 1-worker) + full search via `pob_opt_run_beam` + save |
| `rust/deploy-optimizer.ps1` | build the cdylib and copy `pob_optimizer.dll` into `runtime/` |
| `src/Modules/OptimizerPool.lua` | host-side FFI wrapper (first FFI use in the codebase); `:scoreBatch3`, `:runBeam` |

## Build

```
cargo build --release --manifest-path rust/pob-optimizer/Cargo.toml
```

Produces `rust/pob-optimizer/target/release/pob_optimizer.dll`. For the host to
load it, `pob_optimizer.dll` must be on the OS loader path — in production, copy
it next to the other runtime DLLs (`runtime/`), which is already on PATH when the
GUI runs. **`rust/deploy-optimizer.ps1` does the build + copy** in one step:

```
pwsh rust/deploy-optimizer.ps1              # cargo build --release + copy to runtime/
pwsh rust/deploy-optimizer.ps1 -SkipBuild   # copy an already-built DLL only
```

(The DLL is NOT checked in; `rust/.gitignore` excludes `target/`, and the repo
`.gitignore` excludes the deployed `runtime/pob_optimizer.dll`.)

## Test

```
# PATH must include runtime/ so transitive DLLs (lua-utf8.dll, ...) resolve.
cargo run --release --manifest-path rust/pob-optimizer/Cargo.toml --example smoke -- 4
```

The smoke test boots a 4-worker pool against `src/Builds/mymonk.xml`, enumerates
the 3231 eligible candidates, scores the base tree + 8 real single-node additions,
and asserts: workers agree on the base score (isolation), node additions actually
change the score (additions reach the engine), and a second batch reuses the same
pool (persistence). Base score 14746.721 matches the PoC exactly. It also exercises
the newer surface — `score_batch3` agrees with the scalar path, the weights change
the score, the graph export parses, and `call_save` writes a valid XML.

The **beam A/B harness** drives the full production search through the shipped DLL
(microbenchmark par-vs-1-worker, then the `pob_opt_run_beam` search, then save):

```
cargo run --release --manifest-path rust/pob-optimizer/Cargo.toml \
    --example beam_ab -- [workers] [capPoints] [beamWidth] [outFile] [wDps] [wEhp] [penaltyK]
```

`capPoints=0` (default) uses the build's real budget; a small cap (e.g. 12) is a
quick end-to-end check. `penaltyK` (default 5) tunes the regression penalty (see
"Scoring: growth vs. penalty"). The crate also has `cargo test --lib` unit tests for
the beam (param parsing, a synthetic-graph growth test, and the penalized-score
axis-collapse guard — no engine needed).

## FFI contract

```c
typedef struct PobOptPool PobOptPool;
PobOptPool* pob_opt_create(const char* config);                  // NULL on failure
int  pob_opt_score_batch(PobOptPool*, const int32_t* ids,
                         const int32_t* lengths, int32_t n, double* out); // scalar; 0 ok
int  pob_opt_score_batch3(PobOptPool*, const int32_t* ids,
                          const int32_t* lengths, int32_t n, double* out); // 3 doubles/cand
int  pob_opt_run_beam(PobOptPool*, const char* params,
                      int32_t* out_ids, int32_t cap, double* out_stats);  // winning id count; -1 err
double pob_opt_call_save(PobOptPool*, const int32_t* packed, int32_t len);// 1.0 ok / 0.0 fail / NaN
int  pob_opt_candidate_ids(PobOptPool*, int32_t* out, int32_t cap); // count; -1 if NULL
int  pob_opt_worker_count(PobOptPool*);                          // -1 if NULL
void pob_opt_destroy(PobOptPool*);                               // NULL-safe
const char* pob_opt_last_error(void);                            // thread-local
```

- **`config`** is newline-delimited `key=value` (NOT JSON, to keep the cdylib
  dependency-free). Keys: `lua_dll`, `src_dir`, `runtime_dir`, `runtime_lua`,
  `build_xml`, `bootstrap`, `score_fn`, `candidate_fn` (optional), `workers`
  (optional; 0 = available cores), `w_dps` / `w_ehp` (optional upside weights,
  default 1.0 each), `penalty_k` (optional quadratic regression-penalty strength,
  default 5.0). All four are injected into the worker bootstrap and exported back
  to the beam via the graph trailer (see "Scoring: growth vs. penalty"). See
  `config.rs`.
- **`score_batch`** flattens candidates: `ids` is every node id concatenated,
  `lengths[k]` is candidate k's id count. Writes `n` doubles to `out` in order;
  a candidate that fails to score gets `NaN` (the search treats NaN as reject).
- **`score_batch3`** is `score_batch` but writes **three** doubles per candidate —
  `(growthScore, dps, ehp)` interleaved (so `out` holds `3*n`). Drives the
  clean-slate score fn; `growthScore` is the smooth absolute score the beam grows
  on, and the beam re-derives a penalized final score from `dps`/`ehp` (see
  "Scoring: growth vs. penalty"). NaN triple on failure.
- **`run_beam`** runs the WHOLE search (`src/beam.rs::optimize`) inside the pool
  and writes the winning node-id set (incl. class start, sorted) to `out_ids`,
  returning the total count. The pool must be created with
  `candidate_fn=__pob_graph_export` + `score_fn=__pob_score_cleanslate`. `params`
  is a newline/comma `key=value` override (or NULL/empty for defaults): keys
  `cap_points` (0 = build budget), `beam_width`, `max_jump`, `pareto_extra`,
  `detour`, `patience_max`, `max_rounds`, `verbose`. `out_stats` (3 doubles) gets
  `[score, dps, ehp]`. Returns -1 on error. **Size `out_ids` to `cap_points + 1`
  and call ONCE** — the size-then-fetch idiom would re-run the (minutes-long)
  search just to count.
- **`call_save`** applies a winning id set to a worker's spec and writes an
  importable PoB XML. `packed` = `[ n_ids, id1..idn, pathBytes... ]` (the trailing
  bytes are the UTF-8 output path). **Mutates a worker's spec**, so call it ONCE,
  after all scoring is done — any later score on the pool is then relative to the
  mutated spec.
- **Errors** never panic across the boundary (every entry point is
  `catch_unwind`). `create` returns NULL / the int fns return non-zero or -1, and
  the reason is in `pob_opt_last_error()` (thread-local; copy it immediately).
- **NOT re-entrant.** Results are tagged by within-batch index only; the single
  host PoB state must call these serially (it holds the `*mut` handle).

## Worker contract (`worker_bootstrap.lua`)

The worker exposes several globals the FFI dispatches to:
- **`__pob_score(ids)`** — takes a 1-based array of node ids to ADD to the tree
  (each node implies its precomputed `node.path`; connectivity is never repaired —
  design doc §5) and returns a scalar `W_DPS·FullDPS + W_EHP·TotalEHP`. The misc
  calculator is built **once** at boot (the #1 perf rule). Unknown ids are skipped.
- **`__pob_score_cleanslate(ids)`** — the search's real scorer. `ids` is the FULL
  candidate node-id set (incl. class start + travel nodes); it is scored ABSOLUTELY
  via the clean-slate override (`addNodes = candidate`, `removeNodes = current
  alloc − start`), reproducing the spike's numbers. Returns **three** values
  `(growthScore, dps, ehp)` where `growthScore = 100·(W_DPS·dps/refDps +
  W_EHP·ehp/refEhp)` (refs = the original build's values, so DPS and EHP are on the
  same scale). This is the **growth signal** — smooth and monotonic in both axes so
  the beam can climb from the start-only tree. It is NOT the final objective; the
  beam applies the regression penalty (see below). Always finite (no NaN gate),
  because a per-candidate reject would stall growth (every partial tree is below the
  full original build on both axes).
- **`__pob_graph_export()`** — returns the flat tree topology + budget the Rust
  beam needs, plus a trailer of penalty constants:
  `[ startId, budgetP, nNodes, (id, typeCode, nLinks, link...)·nNodes,
  refDps·1000, refEhp·1000, W_DPS·1000, W_EHP·1000, PENALTY_K·1000 ]`
  (typeCode 0=other/1=Normal/2=Notable/3=Keystone; trailer ints are ×1000 to keep
  a few decimals). Main-tree, non-ascendancy, non-mastery nodes only. The trailer is
  appended AFTER the node data, so older parsers that read exactly `nNodes` nodes
  ignore it (backward-compatible; missing trailer ⇒ penalty disabled). Wired in as
  `candidate_fn` for the beam.
- **`__pob_save_optimized(packed)`** — applies the winner to the live spec
  (keeping the build's ascendancy) and `build:SaveDB`s it to the packed path.
- Score weights `W_DPS`/`W_EHP` and the penalty strength `PENALTY_K` are injected by
  Rust from the config (`w_dps`/`w_ehp`/`penalty_k`), defaulting to 1/1/5.

### Scoring: growth vs. penalty

The objective is split across the worker/beam boundary to reconcile two conflicting
needs — a smooth gradient to **grow** the tree, and a DPS/EHP floor to **select** the
final tree:

- **Growth** (worker, `__pob_score_cleanslate`): the smooth absolute score
  `100·(W_DPS·dps/refDps + W_EHP·ehp/refEhp)`, monotonic in both axes, drives beam
  expansion. Every node that raises dps or ehp raises the score, so the beam always
  has a gradient — even for partial trees that are below the original build.
- **Selection** (beam, `Graph::penalized_score` in `src/beam.rs`): the elimination
  diet only ever compares **full-budget** trees, and judges them by a penalized
  score — each axis's gain ABOVE the original build counts linearly, any shortfall
  BELOW it is docked **quadratically and uncapped** (`−K·delta²` per axis). The
  quadratic, uncapped penalty means no finite gain on one axis can buy back a
  collapse on the other (the hole in a linear sum: `−50% dps / +250% ehp` used to net
  positive — now the `K·0.5²` term dominates). `PENALTY_K = 0` (no trailer) reduces
  it to the smooth growth score, preserving old behavior. Raising `W_DPS` relative to
  `W_EHP` biases the upside toward damage; raising `K` makes regressions less
  forgiving (5 lenient, 25 strict). Unit-tested in `penalized_score_blocks_axis_collapse`.

**Why the penalty must NOT live in the worker:** an earlier design rejected (NaN'd)
any candidate that regressed both axes vs. the original. Because the beam grows from
0 nodes and every partial tree is below the full original build on both axes, that
rejected every step and the beam never grew past 0 points. The growth/penalty split
above is the fix.

## Known limits (carried from the PoC)

- **~3.4× ceiling**, memory-bandwidth bound. Provision ~4–6 workers; more cores
  yield little and each worker holds a full engine (~tens of MB).
- Boot is 1.5–5.8 s **per worker**, paid once (pool is persistent). Don't respawn
  pools per diet round — create once, reuse, `:destroy()` at the end.

## Still open (not done here)

- The **UI** (`OptimizerTab` / modal, design doc §5): a dialog that boots the pool,
  calls `OptimizerPool:runBeam`, previews the winning tree, and applies it via an
  undoable spec edit. This is the only remaining piece before the optimizer is
  user-reachable — see `docs/todo.md`.
- **Packaging-CI hook** (optional): `rust/deploy-optimizer.ps1` does the build +
  copy locally; wiring it into the release/installer workflow so shipped builds
  include `pob_optimizer.dll` is not done yet.

**Done since 2026-06-12** (see `docs/finished.md`): `score_batch3` + the
clean-slate scorer, config `w_dps`/`w_ehp` weights, `__pob_graph_export`, the
in-crate beam search (`src/beam.rs` + `pob_opt_run_beam`), `__pob_save_optimized` +
`call_save`, the `beam_ab` harness, `OptimizerPool:runBeam`, and the deploy script.
The **growth/penalty score split** (2026-06-14): `Graph::penalized_score` with a
quadratic, uncapped regression floor vs. the original build (config `penalty_k`,
exported via the graph trailer), so the diet stops trading away DPS for EHP; plus
NaN-safe beam selection/marginals. Validated on mymonk at 127 pts: DPS 784→1217
(+35% vs. the old linear 1/1 result) while EHP held.
