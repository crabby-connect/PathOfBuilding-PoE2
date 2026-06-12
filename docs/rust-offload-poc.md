# Offloading the Tree Optimizer to Rust — PoC Findings

**Status:** proof-of-concept complete (2026-06-12). Code in `rust/pob-optimizer-poc/`.
**Companion to:** `docs/tree-optimizer-design.md` (esp. the THROUGHPUT memos, §4).

## The question this answers

The design doc's settled conclusion: `calcs.perform` is a ~8–10 ms **irreducible** floor
(must run every candidate because node mods changed); cheaper-call tricks buy only ~1.2×;
the real levers reduce *call count*. The user asked whether **Rust** can help. The only
framing that pays off without forking the ~19k-line calc engine is **parallelism**: run the
same Lua `perform` calls across many cores. The calc engine stays canonical Lua; Rust is the
**search driver** that drives a pool of independent LuaJIT worker states.

This PoC de-risks the single fact that makes or breaks that plan:

> Can we stand up N independent LuaJIT states inside Rust, each booting PoB's headless calc
> engine, and run `perform` on N threads with useful speedup?

**Answer: yes, it runs and is correct — but it does NOT scale linearly. It plateaus ~3.4×.**

## What was built

`rust/pob-optimizer-poc/` — a stand-alone Rust binary (the production form is a cdylib loaded
via LuaJIT FFI; the worker model is identical):

- `src/lua.rs` — minimal Lua 5.1 / LuaJIT C-API binding, resolved **at runtime from the
  shipped `runtime/lua51.dll`** via `libloading` (no import lib needed; also the correct
  production model — workers must use the exact LuaJIT the host ships).
- `worker_bootstrap.lua` — per-worker boot: sets `package.path`/`cpath` to the runtime libs,
  `dofile("HeadlessWrapper.lua")` (which boots the full engine via `Launch.lua`), loads
  `src/Builds/mymonk.xml`, grabs `GetMiscCalculator` **once**, exposes `__poc_eval()` =
  one real `calcFunc({addNodes=…}, true)` pass (one `perform` + FullDPS roll-up).
- `src/main.rs` — spawns N OS threads, each creating its own `lua_State`, booting the engine,
  and running `reps` evals. Times boot separately from the steady-state eval loop.

**Run:** `cargo run --release --manifest-path rust/pob-optimizer-poc/Cargo.toml -- [reps] [threads]`
(PATH must include `runtime/` so transitive DLLs like `lua-utf8.dll` resolve).

### Bootstrap gotchas discovered (all solved, documented in the bootstrap)
1. `Common.lua` needs native `lua-utf8.dll` → `package.cpath = runtime/?.dll`.
2. `Main.lua:64` indexes global `arg` (set by the lua CLI, not an embedded state) → define `arg = {}`.
3. The engine assumes `cwd == src/` (as `.busted` sets) → Rust `chdir`s there before booting.

## Results (16-core machine, release build, mymonk.xml)

**Correctness:** every worker independently reports the identical base score (14746.7) and
3231 candidates — confirming fully isolated states, no cross-talk. Steady-state **5.0–5.5
ms/call**, matching the design doc's ~10 ms FullDPS / ~8 ms NoFull baseline (this PoC's score
is a lighter DPS+EHP sum, hence the lower end). The engine runs *identically* inside a
Rust-hosted state.

**Scaling (eval loop only, one-time boot excluded — a real search amortizes boot over tens of
thousands of calls):**

| threads | speedup | efficiency |
| --- | --- | --- |
| 2 | 1.46× | 73% |
| 3 | 1.93× | 64% |
| 4 | 2.26× | 56% |
| 5 | 2.49× | 50% |
| 8 | 2.88× | 36% |
| 16 | 3.36× | 21% |

**The wall:** speedup asymptotes near **3.4×** and per-worker eval time *inflates* with thread
count (1000 evals: 5 s at t=1 → 26 s at t=16). That signature — work-per-thread growing as
threads grow — is **memory-bandwidth / cache contention**, not lock contention or GC (states
share nothing). PoB's `perform` is allocation-heavy: each call churns large mod tables, so N
workers saturate the memory subsystem well before they saturate the cores.

## What this means for the project

1. **Rust parallelism is real and worth it — at the LOW end.** ~2.2× at 4 workers, ~2.9× at
   8, stacking *multiplicatively* on the existing Lua levers (accel+memo ≈ 1.7–1.9×, Pareto
   ≈ 1.2×). A 30-min diet run → realistically **~5–8 min** at 4–8 workers. That is the single
   biggest lever found, and it does not touch the calc engine (stays upstream-compatible).
2. **Do NOT expect linear scaling.** 16 cores ≠ 16×; the ceiling is ~3.4× on this machine.
   Provision ~4–6 workers as the default; more cores yield little and waste memory (each
   worker holds a full engine + tree, ~tens of MB).
3. **Call-count reduction still matters and is cheaper.** The Lua-only levers in the design
   doc (tighter beam, power-per-point jump seeding, fewer diet rounds) are days of work and
   compose with this. Land those first; they may make the dialog usable before the Rust cost
   is paid, and they shrink the call count the workers then chew through.

## Open items before productionizing (cdylib + FFI)

- **Boot cost amortization.** Per-worker boot is 1.5–5.8 s (tree load + mod parse). A real
  search must reuse a **persistent worker pool** across all diet rounds, not respawn per
  round. (The PoC respawns; production must not.)
- **Bandwidth mitigation.** Investigate whether `accelerate`-reuse (design doc, fewer
  allocations/call) raises the parallel ceiling, not just the serial speed — fewer allocations
  ⇒ less memory traffic ⇒ better scaling. This is the one cheaper-call lever that might also
  buy *parallel* headroom.
- **FFI boundary + host integration.** Production form is a cdylib called from the main PoB
  state via LuaJIT FFI. The main state generates candidate node-sets and collects scores; the
  cdylib owns the worker threads + their `lua_State`s. Each worker thread must own the state it
  created (LuaJIT's one-state-per-thread rule). No SimpleGraphic host change is required for the
  workers themselves — they boot headlessly via `HeadlessWrapper.lua`, exactly as this PoC proves.
- **Serialization protocol.** Candidate sets in / scores out across the FFI boundary as flat
  id arrays (the memo signature is already an order-independent sorted-id list — reuse it).
