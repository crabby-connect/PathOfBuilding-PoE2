# Rust Tree-Optimizer Integration (production cdylib)

**Status:** integrated (2026-06-12). Code in `rust/pob-optimizer/`.
**Builds on:** `docs/rust-offload-poc.md` (the PoC that proved the worker model)
and `docs/tree-optimizer-design.md` (the search this feeds).

This is the production form of the parallel tree-optimizer search driver. The PoC
proved N independent LuaJIT states can each boot the headless calc engine and run
`calcs.perform` on N threads (~3.4× ceiling, memory-bandwidth bound). This turns
that into a shippable **cdylib** + a **persistent worker pool** + a **LuaJIT FFI**
boundary the host PoB state calls.

The calc engine stays canonical upstream Lua. Rust is only the search *driver*.

## Pieces

| File | Role |
| --- | --- |
| `rust/pob-optimizer/src/lib.rs` | crate root; re-exports the FFI surface |
| `rust/pob-optimizer/src/ffi.rs` | the `extern "C"` boundary (the only public surface) |
| `rust/pob-optimizer/src/pool.rs` | persistent worker pool (N threads, one LuaJIT state each, booted once and reused) |
| `rust/pob-optimizer/src/lua.rs` | minimal LuaJIT C-API binding, resolved from the shipped `runtime/lua51.dll` |
| `rust/pob-optimizer/src/config.rs` | parses the `key=value` config string from the host |
| `rust/pob-optimizer/worker_bootstrap.lua` | per-worker boot: engine + build + `__pob_score(ids)` + `__pob_candidate_ids()` |
| `rust/pob-optimizer/examples/smoke.rs` | end-to-end test, calling the C ABI exactly as Lua does |
| `src/Modules/OptimizerPool.lua` | host-side FFI wrapper (first FFI use in the codebase) |

## Build

```
cargo build --release --manifest-path rust/pob-optimizer/Cargo.toml
```

Produces `rust/pob-optimizer/target/release/pob_optimizer.dll`. For the host to
load it, `pob_optimizer.dll` must be on the OS loader path — in production, copy
it next to the other runtime DLLs (`runtime/`), which is already on PATH when the
GUI runs. (The DLL is NOT checked in; `rust/.gitignore` excludes `target/`.)

## Test

```
# PATH must include runtime/ so transitive DLLs (lua-utf8.dll, ...) resolve.
cargo run --release --manifest-path rust/pob-optimizer/Cargo.toml --example smoke -- 4
```

The smoke test boots a 4-worker pool against `src/Builds/mymonk.xml`, enumerates
the 3231 eligible candidates, scores the base tree + 8 real single-node additions,
and asserts: workers agree on the base score (isolation), node additions actually
change the score (additions reach the engine), and a second batch reuses the same
pool (persistence). Base score 14746.721 matches the PoC exactly.

## FFI contract

```c
typedef struct PobOptPool PobOptPool;
PobOptPool* pob_opt_create(const char* config);                  // NULL on failure
int  pob_opt_score_batch(PobOptPool*, const int32_t* ids,
                         const int32_t* lengths, int32_t n, double* out); // 0 ok
int  pob_opt_candidate_ids(PobOptPool*, int32_t* out, int32_t cap); // count; -1 if NULL
int  pob_opt_worker_count(PobOptPool*);                          // -1 if NULL
void pob_opt_destroy(PobOptPool*);                               // NULL-safe
const char* pob_opt_last_error(void);                            // thread-local
```

- **`config`** is newline-delimited `key=value` (NOT JSON, to keep the cdylib
  dependency-free). Keys: `lua_dll`, `src_dir`, `runtime_dir`, `runtime_lua`,
  `build_xml`, `bootstrap`, `score_fn`, `candidate_fn` (optional), `workers`
  (optional; 0 = available cores). See `config.rs`.
- **`score_batch`** flattens candidates: `ids` is every node id concatenated,
  `lengths[k]` is candidate k's id count. Writes `n` doubles to `out` in order;
  a candidate that fails to score gets `NaN` (the search treats NaN as reject).
- **Errors** never panic across the boundary (every entry point is
  `catch_unwind`). `create` returns NULL / `score_batch` returns non-zero, and
  the reason is in `pob_opt_last_error()` (thread-local; copy it immediately).
- **NOT re-entrant.** Results are tagged by within-batch index only; the single
  host PoB state must call `score_batch` serially (it holds the `*mut` handle).

## Worker contract (`worker_bootstrap.lua`)

`__pob_score(ids)` takes a 1-based Lua array of node ids to ADD to the tree (each
node implies its precomputed `node.path`; connectivity is never repaired — design
doc §5) and returns a scalar `W_DPS·FullDPS + W_EHP·TotalEHP`. Weights default to
1/1, overridable via `__pob_w_dps` / `__pob_w_ehp` globals. The misc calculator is
built **once** at boot (the #1 perf rule). Unknown ids are skipped (score as base
tree), so the search may probe freely.

## Known limits (carried from the PoC)

- **~3.4× ceiling**, memory-bandwidth bound. Provision ~4–6 workers; more cores
  yield little and each worker holds a full engine (~tens of MB).
- Boot is 1.5–5.8 s **per worker**, paid once (pool is persistent). Don't respawn
  pools per diet round — create once, reuse, `:destroy()` at the end.

## Still open (not done here)

- The Lua **search loop** (beam + Pareto) and the **UI** (`OptimizerTab`/modal,
  §5) — out of scope for this integration, which is the Rust offload only.
- Deploy step that copies `pob_optimizer.dll` into `runtime/` as part of the
  build/packaging pipeline.
