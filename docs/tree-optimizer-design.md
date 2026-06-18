# Passive Tree Optimizer — Design Doc

**Goal:** Given the current build (skills, items, config) and a skill-point budget, search the
passive tree for an allocation that maximizes a weighted combination of **real DPS and EHP**,
using PoB's own calc engine as the objective — not a surrogate stat-weight sum.

**Status:** implemented as a parallel Rust cdylib (`rust/pob-optimizer`): beam search + Pareto
frontier + elimination diet behind `pob_opt_run_beam`. Host integration in
`src/Modules/OptimizerPool.lua` (LuaJIT FFI). The search runs across a persistent pool of worker
LuaJIT states, each a full headless calc engine. **UI not yet built.**

This doc describes the *current* design. Measured numbers are dated because they depend on the
reference build, which changes; the latest are in §4 and §8.

---

## 0. What already exists in the repo

The optimizer is mostly *orchestration* of existing machinery, not new algorithms.

| Need | Already implemented | Where |
| --- | --- | --- |
| Score a node set in real DPS/EHP without rebuilding env | `GetMiscCalculator` → `calcFunc(override)` closure | `Modules/Calcs.lua:123` |
| `calcFunc({ addNodes=… , removeNodes=… })` delta eval | node-hover tooltip diffs | `Classes/PassiveTreeView.lua:1875-1899` |
| Per-node offence/defence power, per-point scoring loop | `PowerBuilder` | `Classes/CalcsTab.lua:521-599` |
| Shortest path from allocated frontier to any node (BFS) | `BuildPathFromNode` → `node.pathDist`, `node.path` | `Classes/PassiveSpec.lua:1061` |
| Connectivity rules (start nodes, ascendancy boundaries, masteries, unlock constraints) | pathing predicate | `Classes/PassiveSpec.lua:1083-1105` |
| Point budget accounting | `CountAllocNodes` | `Classes/PassiveSpec.lua:990` |
| Node typing (Notable/Keystone/Socket/Normal) | tree load | `Classes/PassiveTree.lua:202-263` |
| Adjacency | `node.linked` / `node.linkedId` | `Classes/PassiveTree.lua:300-301` |

**Key consequence:** the objective is `calcFunc(override)`, which returns the full
`env.player.output` table (`.FullDPS`, `.TotalEHP`, every stat). A node's value is **not**
additive — `PowerBuilder` already proves the per-candidate re-run pattern.

---

## 1. Scope

The optimizer does a **clean-slate reallocation**, not "what to add next":

- Gear/skills/config **fixed**; class + ascendancy **fixed** (one search from
  `spec.curClass.startNodeId`).
- Budget **P = the normal passive points currently used** (`CountAllocNodes`, captured before
  reset). No level/quest math.
- Discard the current tree allocation and find the best connected P-node subtree rooted at the
  class start. This is the prize-collecting Steiner tree problem.

**In scope (v1):** objective `score = wDPS·FullDPS + wEHP·TotalEHP` with user weights, plus a
Pareto frontier; all of PoB's pathing/connectivity rules.

**Out of scope (v1):** provable optimality (the objective is non-monotone/non-additive — no
admissible A\* heuristic exists; v1 is approximate beam search, say so in the UI); re-slotting
skills/items/config; jewel/timeless-jewel/mastery/cluster optimization; ascendancy points (v2).

---

## 2. Objective function

```lua
local calcFunc, calcBase = build.calcsTab:GetMiscCalculator() -- ONCE, outside the loop
-- clean-slate eval: pass a reset frontier + the candidate tree as addNodes
local out = calcFunc({ addNodes = candidateSet, removeNodes = currentAlloc }, useFullDPS)
local score = wDPS * (out.FullDPS or out.TotalDPS or 0) + wEHP * (out.TotalEHP or 0)
```

**Clean-slate via override (no per-candidate rebuild):** `calcSetup` builds the eval node set as
`override.addNodes ∪ (spec.allocNodes − override.removeNodes)` (`CalcSetup.lua:718-749`). So
scoring a candidate = `addNodes` = full candidate tree (incl. start + travel), `removeNodes` =
the original build's non-ascendancy allocation. The evaluated set is then exactly the candidate.
**Verified:** re-scoring the live build's own node set through this path reproduces calcBase
exactly (§4).

**Constraints (do not violate):**
- `GetMiscCalculator` takes **no arguments** and must be called **once** — it runs a full
  `initEnv` + `perform` (`Calcs.lua:123-150`). Per-candidate calls are the #1 perf trap.
- **FullDPS cache caveat** (`Calcs.lua:139-147`): without forcing a fresh FullDPS roll-up per
  override, A-vs-A cache reuse makes deltas read 0. Pass overrides through the closure (don't
  mutate the spec); respect `useFullDPS` as `PowerBuilder` does.
- `addNodes`/`removeNodes` use node-object keys (`{ [node]=true }`); stay consistent.

---

## 3. Algorithm

Compressed graph + beam search + Pareto frontier + elimination diet. (A\* was dropped — no
admissible heuristic against the real calc.)

**3.1 Compression.** Targets are `Notable | Keystone | Socket` on the main tree. Normal travel
nodes enter a solution as path cost via `node.path`, not as targets. ~hundreds of targets, not
thousands of nodes.

**3.2 Travel precompute.** Heap-based Dijkstra from the allocated frontier fills point cost + the
traversed path for every node, obeying all connectivity rules. Re-run when the frontier changes.
Soft-banned nodes (diet) get a high detour weight so a ban never walls off a better keystone
behind it — routes detour, traversing a ban only as last resort. This runs once per beam *state*
per *step* (thousands of times per `run_beam`), so the frontier selection uses a `BinaryHeap` with
lazy deletion (O(E log V)), not the old O(V²) linear min-scan — a free, behavior-preserving win the
original cost model (§4) overlooked because it assumed `calcs.perform` dominated everything.

Three further pure-Rust wins make this hot path allocation-light (all behavior-preserving — same
pdist, same path, same deterministic tie-break; gated by the crate unit tests):
- **Dense node index.** The `HashMap`-keyed `Graph` is projected once into a `NodeIndex`: an
  ascending-id-sorted dense numbering with CSR adjacency (`adj` + `adj_off`) and per-index type
  flags. The Dijkstra inner edge-relaxation loop indexes contiguous arrays instead of hashing an
  `i32` per edge. Numbering by ascending id makes the heap's `(weight, index)` tie-break *identical*
  to the old `(weight, id)` one, so the basin selection the §8 result credits is preserved exactly.
- **Reusable scratch.** The per-call `wdist/pdist/prev/visited` `HashMap`s are gone; a `Pathfinder`
  owns `Vec`-backed scratch sized to the node count once and clears only the *dirtied* indices each
  call (tracked in a `dirty` list), so a Dijkstra is zero heap-allocation in steady state.
- **Path memo (the §1a lever).** `Pathfinder` caches `(frontier-signature, excluded-signature) →
  result`. `excluded` only changes once per diet round, so within a beam pass the key is effectively
  the frontier; recurring frontiers across diet rounds / restarts skip the Dijkstra entirely — the
  same call-count lever the score memo applies to the calc, now applied to the pathing the original
  "perform dominates" model ignored. Signatures are size-folded XORs of a splitmix64 per index; a
  collision can only ever cost a re-derived path, never corrupt a score (scores memo separately by
  node-set). All hot integer-keyed maps/sets switched from SipHash to `FxHashMap`/`FxHashSet`.

**3.3 Power seed (one-time, before round 0).** Score every notable/keystone as a single-add from
the start (one parallel batch) and rank by penalized power-per-point. Two uses:
- `max_jump_cand` (opt-in): cap per-state jump candidates by **real power** (keystones exempt).
  Unlike a distance cap — which starves distant *damage* — power ranking keeps far damage and
  drops junk. There is no calc-free heuristic that distinguishes a distant damage notable from
  distant junk, so this is the only safe way to cap. Default uncapped.
- `seed_anchors` (default 6): seed round-0 with the top-power distant clusters so the beam
  commits to good far routes from step 1 instead of greedily climbing near EHP (the documented
  local-optimum trap). Each anchor = start tree + shortest path to one high-power target.

**3.4 Beam search.** State = connected set of nodes + points spent + last output. Expand each
state by (a) adjacent single nodes and (b) path-jumps to reachable notables/keystones within
`max_jump`. Reject over-budget expansions. Keep top-N by scalar score **plus** Pareto-non-dominated
states (high on one axis even if lower-scoring), deduped by node-set signature. Connectivity is
automatic: a target is only ever added together with its path back to the frontier.

**3.5 Pareto frontier** in `(FullDPS, TotalEHP)` output space (node-stat space is meaningless
here).

**3.6 Elimination diet (local-optimum escape by perturbation).** Round 0 = unconstrained beam
best. Each round bans one droppable notable/keystone, re-optimizes, and keeps the ban only if the
penalized score improved (else revert + tick patience). Stop after `patience_max` consecutive
failures. Two phases: ban the lowest-marginal node, then the deepest (longest-path) droppable
node. A **build-enabler guard** marks any node whose removal collapses an axis (leave-one-out dps
or ehp drops below 1% of full) as MANDATORY — never banned (e.g. Hollow Palm Technique, which
enables this Monk's unarmed playstyle; banning it gives dps=0).

Now that throughput is solved, `patience_max` defaults to **8** and `max_rounds` to **40** (was
2/12) — the diet runs long and bans hard, since a few low-marginal misses no longer justify
stopping. Both are overridable in `beam_ab` via `PATIENCE_MAX` / `MAX_ROUNDS` env for sweeps.

**Dead-leaf reclaim (batch, free).** A notable *or keystone* that is a *leaf* of the allocation
(≤1 allocated neighbour, so it carries no travel for any other node) **and** contributes ≤0
marginal is a wasted point — e.g. a notable like "Splinters" hanging off the edge whose stats do
nothing for the build. At the top of each diet round, **all** such leaves are collected and banned
*together*, then the full beam re-runs **once** with the whole leaf set excluded — one full-algo
run per round, never one run per dead leaf. Accept iff the re-grown tree does not regress (≥ best,
not strictly >), since shedding dead leaves and re-spending their points is a structural win even
at a tie; no patience tick either way. The bans are permanent (those leaves won't return as
targets), so the set strictly shrinks and the pass terminates. Only when a round finds *no* dead
leaves does it fall through to the normal single lowest-marginal ban. The leaf test
(`is_alloc_leaf`) is what protects the main path: a load-bearing node has ≥2 allocated neighbours
and never qualifies — type is irrelevant to safety. A non-leaf worthless node is still removable by
the normal diet, just not unconditionally. (The normal diet requires *strict* improvement, so a
dead leaf that's only a *tie* to drop — exactly the Splinters case — slips past it; the relaxed
accept-on-tie reclaim is what catches it.)

**3.7 Multi-seed restart (local-optimum escape by re-seeding).** The diet is greedy single-node
removal — it hill-climbs in ban-space and cannot cross a valley where the *initial basin* (which
far cluster round 0 committed to) is itself wrong. Round 0 picks its anchors from the top
`seed_anchors` power-ranked targets; a single bad commitment there is something no sequence of
single bans can undo (the [smoke run](../smoke_fullrun.log) showed exactly this: round 0 landed
in an all-EHP basin at penalized −84.8 and the 16-round diet clawed back only to +16.8, stopping
on patience, not exhaustion). So after the primary search converges, the *whole* search (round 0 +
diet) re-runs `restarts` more times, each seeded from the **next band** of anchors (offset
`k·seed_anchors`), committing round 0 to structurally-different far routes. The best converged tree
across all bands wins (`optimize`'s `run_full` closure, looped). The power seed + the cross-round
memo are **shared** across restarts, so a restart that re-explores overlapping sets is much cheaper
than the first. `restarts` defaults to **2** (overridable via `RESTARTS` env in `beam_ab`); 0
restores the old single-search behavior. Skipped entirely when `seed_anchors == 0` (no perturbation
to apply).

**Score shape.** The worker returns a *growth* score `100·(wDPS·dps/refDps + wEHP·ehp/refEhp)`,
monotonic in both axes so the beam can climb from the start-only tree. The DPS/EHP **regression
penalty** vs. the original build is applied by the Rust beam to full-budget trees only
(`beam.rs penalized_score`) — not per partial tree, which would stall growth at 0 points.

---

## 4. Throughput (the main risk) — measured

The per-call cost is dominated by `calcs.perform`, which is **irreducible** (the candidate's
mods changed, so it must run). Cheaper-call tricks barely move it; the win must come from **fewer
calls, not cheaper calls.**

- `perform` floor ≈ 8 ms/call. Accel env-reuse + `useFullDPS=false` is a flat ~1.2× — not the
  5–10× needed. Apply it anyway (free), but treat call-count reduction as the real work.
- **Call-count levers, all implemented:** set-signature **memoization** (~40% hit rate, ~1.67×
  fewer calls); **Pareto-augmented beam** (same optimum at half the base width, −22% evals);
  narrower beam; opt-in jump-candidate cap; power-seeded anchors (better routes sooner, fewer
  wasted rounds).
- **Parallelism:** the worker pool gives ~3.4× over single-thread on a 16-core box (scores match
  single-thread within 1e-6).
- **Power seed cost:** ~1.2 s for all ~1,017 targets — under 1% of a multi-minute search.
- **Pathing cost (added):** the per-state Dijkstra (§3.2) is no longer O(V²) — a heap with lazy
  deletion makes it O(E log V). With memoization lifting the calc hit-rate to ~40%, this pure-Rust
  pathing was a non-trivial slice of wall time the original "perform dominates" model ignored. It is
  now also **allocation-light and memoized** (§3.2): a dense node index removes id-hashing from the
  inner loop, a reused scratch buffer removes the four-`HashMap`-per-call allocation, and a
  frontier-keyed path memo skips the Dijkstra when a frontier recurs across diet rounds / restarts.

A full-P search is still multi-minute (and the §3.7 restarts multiply the diet cost, partly offset
by the shared seed + memo), so the dialog must run async with progress + Cancel.

**4.1 Persistent score cache (cross-run memo).** The in-run memo dies with `optimize`. Setting
`POB_OPT_CACHE=<path>` (read by `pob_opt_run_beam`) persists it to a small binary file and warm-
starts the next run on the **same build** from it — the common tuning loop (run, tweak, run again).
Three design points make it safe and useful:
- **Only raw `dps`/`ehp` are stored**, never the growth score. Col 0 is re-derived on load via
  `Graph::growth_score` (a Rust mirror of the worker's `scoreNorm`, using the trailer weights). So a
  re-run with **different DPS/EHP weights reuses every previously-evaluated set for free** — a fixed
  set's raw dps/ehp don't depend on the weights. (It's not a *whole-search* hit, because different
  weights steer the beam to explore some *different* sets; those, and only those, are calced. Measured
  on a reduced probe: same-weights re-run = 0 new evals (7.1s → 0.1s); weights 1/1 → 3/0.2 = only
  2056 of 4026 evals re-run.) This is why the worker's col 0 is now *recomputed* in Rust
  (`optimize_with_memo`) instead of trusted: it decouples the cached value from the weighting.
- The file is tagged with a **build signature** = a hash of the graph topology + budget + the
  build's `refDps`/`refEhp` (the export trailer's first two ints), **excluding** the weight/penalty
  ints. Any gear/skill/tree change moves the topology or the refs → signature mismatch → the stale
  cache is silently discarded (cold start), never reused incorrectly. A corrupt/short file likewise
  degrades to a cold start; a failed write is non-fatal. Cache code is in `src/cache.rs` (no deps,
  little-endian); it activates only when a real trailer is present. Default (env unset) = no cache,
  behavior identical to before.
- **The signature MUST be order-independent.** `__pob_graph_export` emits nodes in `pairs(nodeById)`
  order, which Lua does not keep stable across processes — a sequential hash of the flat export gives
  a *different* signature every run, so the cache would always be discarded (the first cut hit exactly
  this: warm-loaded 0). `build_signature` therefore parses the structure and folds each node's
  `(id, type, sorted-links)` commutatively (XOR), so emission order can't change the result. Verified
  identical across two separate processes (`build_sig=0x1ca4…`), and guarded by
  `signature_is_node_order_independent`.

This is **exact-set reuse only** — the "these IDs already exist, copy the score" idea. Fuzzy reuse
(estimating an un-evaluated set from a near neighbour) is deliberately *not* done: the objective is
non-additive (§0), so any approximated score breaks trust. Only sets that were actually calced are
ever returned from the cache.

**4.2 Memory.** The score memo is the only run-length-dependent heap consumer (the 16 worker LuaJIT
engines are large but fixed). `pob_opt_run_beam` reports its entry count + estimated heap MB under
`verbose=1` (`BeamResult::memo_entries` / `memo_bytes_est`) so memory/paging can be checked against
a number rather than assumed.

---

## 5. UI / integration (not yet built)

- Modal launched from the Tree tab (lower-risk than a new tab for v1).
- Controls: DPS weight, EHP weight, FullDPS toggle, budget (default = current points), beam width
  (advanced), Run / Cancel + progress bar.
- Results: sortable Pareto solutions (ΔDPS, ΔEHP, points). Selecting one **previews** added nodes
  on the tree; **Apply** allocates them via an **undoable** spec edit.
- Search runs on a copy of the spec; commit only on Apply. The save path already exists
  (`__pob_save_optimized` → `spec:ImportFromNodeList` + `build:SaveDB`), confirmed to produce a
  file that opens correctly in real PoB.

---

## 6. Correctness gates (before merge)

- [ ] `GetMiscCalculator` called exactly once per Run.
- [ ] FullDPS deltas non-zero for a known-good node (vs. hover tooltip).
- [ ] Budget never exceeded vs. `CountAllocNodes`; ascendancy/socket pools not mixed in.
- [ ] Every emitted solution connected to class start.
- [ ] Search on a spec copy; live build unchanged unless Apply pressed; Apply undoable.
- [ ] Pathing edge cases: masteries, ascendancy boundaries, `unlockConstraint`, intuitive-leap
      jewels (`node.intuitiveLeapLikesAffecting`).
- [ ] Cancel stops the loop and frees the spec copy.

---

## 7. Running it

Crate unit tests:
```powershell
cargo test --release --manifest-path rust/pob-optimizer/Cargo.toml
```

Full end-to-end run + save (native A/B harness, drives the shipped cdylib exactly as the host
LuaJIT does — the only place the Windows cdylib can be exercised on a dev box):
```powershell
$env:PATH = "$PWD\runtime;$env:PATH"
cargo run --release --manifest-path rust/pob-optimizer/Cargo.toml --example beam_ab `
    -- [workers] [capPoints] [beamWidth] [outFile] [wDps] [wEhp] [penaltyK]
# defaults: workers=auto, capPoints=0 (real budget), beamWidth=8, out=mymonk_optimized.xml,
#           wDps=1, wEhp=1, penaltyK=5.  env: SEED_ANCHORS (default 6), MAX_JUMP_CAND (0=uncapped),
#           PATIENCE_MAX (default 8), MAX_ROUNDS (default 40), RESTARTS (default 2)
```
The harness prints the LIVE BASELINE (the build's own dps/ehp), runs a microbenchmark + power-seed
cost pass, then the full beam + diet, and saves the winner to `src/Builds/<outFile>`.
Set `LIVE_IDS=<comma-separated nodes= ids>` to run the clean-slate self-consistency probe.
Set `POB_OPT_CACHE=<path.bin>` to enable the persistent score cache (§4.1): the first run writes it,
later runs on the same build warm-start from it (a weight-only change still hits fully).

The Lua-side beam also runs in the CI container via `spec/System/SpikeCleanSlateBeam_spec.lua`
(busted, LuaJIT) — run from the repo root with `--filter` (no explicit spec path).

---

## 8. Current reference build & latest result

**Reference build: `src/Builds/mymonk.xml`** — a Monk. The file is edited between sessions, so P
and the baseline shift; always re-measure the LIVE BASELINE before comparing.

**Live baseline (2026-06-15, P=110): dps=1942.6, ehp=30,943.1.** (Earlier sessions saw
2110.5 / 36,642.2 — the file changed; always re-measure.) Verified historically: clean-slate
re-score of the live node set reproduces the baseline exactly (delta −0.0 / −0.0 on both axes), so
the `addNodes + removeCurrent` override is correct and trustworthy on both axes.

**Latest result (2026-06-15, P=110, beam=8, w=1/1, K=5, restarts=2, allocation-light pathing):**
penalized **score=37.9, dps=1636.2, ehp=46,514.6** in **947s** (`smoke_fullrun_v3.log`).
**Behavior-identical to the prior `smoke_fullrun_v2.log` run** — same final, same per-round
trajectory (round 0 −68.4 → 7.0 → 37.9), same restart outcomes (band 0 wins; bands −8.4, −10.6) —
which is exactly the point: the §3.2 dense-index / scratch-reuse / path-memo rewrite is a pure
representation swap and reproduces the live-calc result bit-for-bit on both axes.

**Honest throughput note.** The allocation-light pathing did **not** move total wall time on this
build (947s vs 937s — flat, within noise). Once the heap-Dijkstra (O(E log V)) was already in place,
pathing was no longer a meaningful slice of *this* search's wall clock: the ~836K calc evals across
all three searches dominate, and the calc floor (≈8 ms/call, ~3.6× effective on 16 workers) is
irreducible. So the §3.2 wins are best understood as **insurance, not a speedup** here — they cap
pathing's allocation pressure and make the path memo available, which matters most on builds with a
*higher* memo hit-rate or a *narrower* calc (where pathing's share of wall time is larger) than
mymonk. The remaining throughput lever is still **fewer calc calls**, not faster pathing.

Two search findings worth keeping (unchanged from v2, re-confirmed here):
- The **heap-Dijkstra (§3.2)** deterministic tie-break handed the primary search a good round-0
  basin (−68.4), which the diet grew across two accepted bans (−68.4 → 7.0 → 37.9) before stopping
  on patience (8/8).
- The **restarts (§3.7)** did *not* win this build — band 0 already found the best basin (37.9 vs
  the restart bands' −8.4 and −10.6). They're insurance, and the `restarts_never_worsen` guarantee
  held. The bands genuinely diverge, though: a restart band landed round 0 in a 4317-dps *all-damage*
  basin — the mirror image of the all-EHP trap band 0 used to fall into — so the perturbation is
  doing real structural exploration even when it doesn't change the winner here.

**Beam tunables** (`OptimizerPool:runBeam` / `BeamParams`): `capPoints` (0 = real budget),
`beamWidth`, `maxJump`, `paretoExtra`, `detour`, `patienceMax`, `maxRounds`, `maxJumpCand`
(0 = uncapped), `seedAnchors` (default 6), `restarts` (default 2, §3.7), `verbose`.

**Open work:**
1. Build the UI (§5) — the search and save path are done; the dialog is the remaining piece.
2. Per-build safety re-check for `accelerate.skills` and `useFullDPS=false` on tree-granted-skill
   or multi-skill/FullDPS builds (safe for mymonk; not universally).
3. *Algorithm (next):* pair/swap diet moves (the remaining greedy-single-ban limitation §3.7 only
   partly addresses), and a convergence signal in `BeamResult` (patience-stop vs. exhaustion, plus
   round-0-to-final delta) so the UI can tell the user whether widening the beam is worth it.
