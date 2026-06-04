# Passive Tree Optimizer — Design Doc

**Goal:** A new dialog in PoB-PoE2 that, given the current build (skills, items, config)
and a skill-point budget, searches the passive tree for an allocation that maximizes a
weighted combination of **real DPS and EHP** — using PoB's own calc engine as the
objective, not a surrogate stat-weight sum.

**Status:** design. No code written yet.

---

## 0. Key realization: most of this already exists in the repo

Before designing anything new, note what PoB already implements. The optimizer is mostly
*orchestration* of existing machinery, not new algorithms.

| Need | Already implemented | Where |
| --- | --- | --- |
| Score a node set in real DPS/EHP without rebuilding env | `GetMiscCalculator` → `calcFunc(override)` closure | `Modules/Calcs.lua:123` |
| `calcFunc({ addNodes=… }) / { removeNodes=… }` delta eval | node-hover tooltip diffs | `Classes/PassiveTreeView.lua:1875-1899` |
| Per-node offence/defence power **and per-point** scoring loop | `PowerBuilder` | `Classes/CalcsTab.lua:521-599` |
| Shortest path from allocated frontier to any node (BFS) | `BuildPathFromNode` → `node.pathDist`, `node.path` | `Classes/PassiveSpec.lua:1061` |
| Dependency set (what falls off if you remove a node) | `node.depends`, `BuildAllDependsAndPaths` | `Classes/PassiveSpec.lua:1220` |
| Connectivity rules (no pathing through start nodes, ascendancy boundaries, masteries, unlock constraints) | pathing predicate | `Classes/PassiveSpec.lua:1083-1105` |
| Point budget accounting | `CountAllocNodes` (used, asc, sockets, weapon sets) | `Classes/PassiveSpec.lua:990` |
| Node typing (Notable/Keystone/Socket/Normal) | tree load | `Classes/PassiveTree.lua:202-263` |
| Adjacency | `node.linked` / `node.linkedId` | `Classes/PassiveTree.lua:300-301` |
| Tab registration | `self.calcsTab = new("CalcsTab", self)` etc. | `Modules/Build.lua:517-536` |

**The single most important consequence:** the objective function is `calcFunc(override)`,
which returns the full `env.player.output` table (`.FullDPS`, `.TotalEHP`, every stat).
A node's value is **not** additive — `PowerBuilder` already proves the intended pattern of
re-running the calc per candidate.

---

## 1. Scope & non-goals

**PROBLEM RESTATED (2026-06-03, supersedes the incremental framing below).**
The optimizer does a **clean-slate reallocation**, not "what to add next":
- Take gear/skills/config as **fixed**, and the build's **current class + ascendancy as
  fixed** (one search from `spec.curClass.startNodeId`).
- Budget **P = exactly the normal passive points currently used** (`CountAllocNodes`,
  captured before reset). No level/quest-point math — P is just the current count.
- **Discard the current tree allocation** (`spec:ResetNodes()` keeps only the start node)
  and find the best connected P-node subtree rooted at the class start.
- This is the full **prize-collecting Steiner tree** problem: grow an entire tree from the
  root, depth = P (often 40-100+), NOT a shallow depth-10-15 incremental add. **The §3.3
  21s throughput projection does NOT apply here** — that was for adding ~6 nodes.

> **RESOLVED — clean slate IS cheaply expressible.** `calcSetup` builds the eval node set
> as `override.addNodes ∪ (override.spec.allocNodes − override.removeNodes)`
> (`CalcSetup.lua:718-749`), and the calc honors `override.spec` (`CalcSetup.lua:519`). So:
> **pass a reset spec (only the class start allocated) as `override.spec` and the candidate
> P-node tree as `addNodes`** → the final node set is exactly the candidate, evaluated via
> the fast `calcFunc` override path. **No per-candidate `BuildOutput()` rebuild.** Recipe:
> 1. `P = build.spec:CountAllocNodes()` — capture budget *before* reset.
> 2. Build a reset spec (clone + `ResetNodes()`, so `allocNodes` = {start}).
> 3. `calcFunc({ spec = resetSpec, addNodes = candidateSet }, true)`.

> **VERIFIED in `SpikeCleanSlateBeam_spec.lua` (2026-06-03).** Re-scoring the build's exact
> current node set via the clean-slate override path reproduced its real numbers exactly
> (dps 956.6 / ehp 2323.7 on `mymonk.xml`). The eval mechanism is correct — confirmed, not
> assumed. (The spike uses `addNodes`+`removeNodes` on the live spec rather than a clone;
> equivalent result, no clone needed.)
>
> **OPEN — the SEARCH is myopic (the real current problem).** Greedy beam growth from the
> bare class start chases nearby EHP/attribute notables and never invests in distant damage
> clusters. On `mymonk.xml` capped at 25 pts it picked Hollow Palm Technique + defensive
> notables near start (dps stuck at 166 vs hand-made 956). Path-jump expansion (jump to a
> notable via shortest path) helped only marginally because `maxJump=8` can't reach the
> build's damage cluster, which sits in 74-point territory. Fixes to try: (a) full P run, not
> capped — the damage is simply farther than the cap; (b) larger `maxJump`; (c) seed the
> search toward high-DPS targets found by a depth-1 sweep; (d) diversity/Pareto beam so
> damage-seeking branches survive. NOTE: capped-P comparisons are inherently unfair (fewer
> points than the hand-made tree) — only the full-P run is apples-to-apples.

> **PROVEN (2026-06-03) — full P=74, beam=12, maxJump=12 on `mymonk.xml`:**
> from-scratch **score 249.7 vs hand-made 200.0 (+24.9%)** at equal 74 points. Held DPS
> within ~5% (910.7 vs 956.6), added +1267 EHP (+55%), kept real damage notables (Glaciation,
> Snowpiercer) + Mind Over Matter. Score climbed monotonically 138->192->246->249.
> **Core hypothesis confirmed: a beam optimizer on PoB's real calc engine can match/beat a
> hand-built tree at equal budget.** THREE new open items:
> 1. **Throughput regressed: 40,473 calls / 191s (~3 min)** — far over the old 21s projection
>    (that was the easier incremental problem). Too slow for a dialog. Needs smaller beam,
>    jump-candidate caps, modKey-cache reuse (2.28x measured), or frontier pruning. #1 problem.
> 2. **"+25% by stacking EHP" is weight-dependent**, not objectively better (wDPS=wEHP=1
>    rebalanced a DPS-heavy build toward defense). Reinforces the Pareto-frontier need (3.5).
> 3. Hollow Palm Technique persisted in the picks. RESOLVED (2026-06-03): it is a legit
>    **main-tree Keystone** (tree 0_5 id 64601, `ascendancyName=nil`, `isKeystone=true`),
>    NOT an ascendancy node, and it is *correctly* valued high — it enables the
>    unarmed-as-quarterstaff playstyle this Monk is built around. It is NOT a snapshot
>    artifact like Mind Over Matter. This exposes a real flaw in the crutch detector (below).

> **CRUTCH-DETECTOR FLAW + CASCADE REDESIGN (2026-06-03).** The Pass-2 heuristic
> ("marginal > mean+2sd ⇒ crutch, exclude, re-optimize") conflates two different things:
> a *snapshot artifact* the calc OVER-values (Mind Over Matter's hidden "50% less mana
> recovery" — exclude is right) vs. a *build-defining keystone* the calc CORRECTLY values
> high (Hollow Palm — excluding it just yields a worse build). High marginal ≠ crutch. On the
> beam=10 run, hard-excluding Hollow Palm collapsed Pass 2 to a degenerate **dps=0** full-EHP
> build (narrow beam couldn't rebuild a damage path). So auto-exclusion-by-marginal is not a
> safe default.
>
> Instead of a single Pass 2, the spike now runs a **6-run exclusion cascade** to fan out
> diverse builds rather than chase one "true" exclusion:
> ```
> RUN 1: exclude {}          -> detects O1
> RUN 2: exclude {O1}        -> detects O2
> RUN 3: exclude {O1,O2}     -> detects O3
> RUN 4: exclude {O2}        (isolate run-2's pick)
> RUN 5: exclude {O3}        (isolate run-3's pick)
> RUN 6: exclude {O1,O2,O3}  (all three chain outliers together)
> ```
> Each run re-detects its own top-marginal node; the report pools all six bests and shows the
> distinct builds (<70% node overlap). This is a DIAGNOSTIC to see how much real diversity the
> landscape holds, not the final dialog behavior. Cost: 6 full beams (~10-15 min at beam=10).

> **CASCADE RESULTS — FIRST (pre-enabler-guard) run (2026-06-03, beam=10, maxJump=12,
> mymonk.xml).** This run excluded Hollow Palm in the chain and exposed the problem; it is
> kept for the lesson, but the numbers were SUPERSEDED by the enabler-guarded run below.
> | run | excluded | score | dps | ehp |
> | --- | --- | --- | --- | --- |
> | 1 | — | 249.7 | 910.7 | 3590.7 |
> | 2 | O1=HollowPalm | 223.4 | **0.0** | 5190.4 |
> | 3 | O1,O2 | 248.3 | **0.0** | 5769.4 |
> | 4 | O2=CarefulConsid. | 250.8 | 897.2 | 3647.8 |
> | 5 | O3=Lucidity | 279.7 | 1090.1 | 3852.1 |
> | 6 | O1,O2,O3 | 193.6 | **0.0** | 4498.2 |
>
> Lessons that drove the enabler guard:
> 1. The real value of exclusion is **local-optimum escape by perturbation**, NOT "crutch
>    removal." Evicting a *mediocre* occupied node and re-optimizing can find a better build.
> 2. Every run that excludes Hollow Palm (2,3,6) collapsed to **dps=0** — proving Hollow Palm
>    is a load-bearing **build enabler** (no weapon → no damage), not a crutch.

> **CASCADE RESULTS — enabler-guarded run (2026-06-03, same params) — CURRENT BEHAVIOR.**
> Hollow Palm auto-detected as a build enabler, marked MANDATORY, kept in every run. The chain
> now excludes only droppable nodes; all six runs produce playable builds (no dps=0):
> | run | excluded | score | dps | ehp | pick |
> | --- | --- | --- | --- | --- | --- |
> | 1 | — | 249.7 | 910.7 | 3590.7 | Mind Over Matter |
> | 2 | O1=MoM | 257.1 | 1196.9 | 3066.0 | Glaciation |
> | **3** | **O1,O2** | **263.8** | 1185.0 | 3251.3 | Chakra of Thought |
> | 4 | O2=Glaciation | 263.8 | 1185.0 | 3251.3 | Chakra of Thought |
> | 5 | O3=Chakra | 249.7 | 910.7 | 3590.7 | Mind Over Matter |
> | 6 | O1,O2,O3 | 247.4 | **1210.9** | 2806.9 | Giantslayer |
>
> **Best = RUN 3/4: score 263.8, STRICTLY DOMINATES hand-made** (dps 1185>956 AND ehp
> 3251>2324 at equal 74 pts, +31.9%). The chain correctly targets the genuine crutch first
> (O1 = Mind Over Matter, the snapshot artifact), and excluding it lifts dps 910→1197. All
> sanity checks pass: no run excluded a mandatory enabler; clean-slate re-score reproduces the
> live build exactly (956.6/2323.7). NOTE: which exclusion path yields the global best is
> landscape-sensitive (the pre-guard run's 279.7 came from a path this run doesn't take the
> same way) — a tuning point, not a regression; every guarded run is healthy.

> **BUILD-ENABLER DETECTION (2026-06-03, implemented in the spike).** A node is a *build
> enabler* if removing it from the best build collapses an axis: leave-one-out `dps` (or
> `ehp`) drops below 1% of the full build's value on that axis. Such a node is registered
> MANDATORY and is **never excluded** by any cascade run; the chain's outlier picks (O1/O2/O3)
> are the top-marginal *non-enabler* nodes instead. This replaces the wasted dps=0 collapse
> runs (2,3,6 above) with runs that exclude only genuinely droppable nodes — same 6 runs, all
> productive. Detection is free: the marginal loop already computes each leave-one-out score,
> so it just inspects the dps/ehp those evals already return. Sanity checks assert (a) no run
> excluded a mandatory enabler, (b) the clean-slate re-score still reproduces the live build.
> **For v1 this generalizes to: classify high-marginal nodes by *whether their removal
> collapses an axis* (enabler, keep) vs. *leaves both axes healthy* (candidate crutch).**

> **ELIMINATION DIET — CURRENT ALGORITHM (2026-06-03, supersedes the fixed 6-run cascade).**
> Decided with the user: the cascade's real value is **local-optimum escape by perturbation**,
> not crutch removal — so replace the fixed cascade with an *accumulating-greedy elimination
> diet*. Round 0 = unconstrained beam best. Each round bans the **lowest-marginal droppable**
> notable/keystone (weakest link — its point is better spent elsewhere), re-optimizes, and
> keeps the ban only if the score improved; else reverts and ticks `patience`. Stop after
> `SPIKE_PATIENCE` (2) consecutive failed rounds. A `tried` set skips already-reverted victims
> within a streak and is cleared on every accepted ban (the build changed, so a previously-bad
> ban may now help — this actually fired: Step Like Mist failed R2, succeeded R4).
>
> Two mechanics the user specifically required:
> - **Build-enabler guard** (above): enablers never eligible to ban.
> - **Soft-ban + prefer-detours pathing**: a ban means "don't *target* this node," NOT "can't
>   *path through* it." Pathfinding is Dijkstra with weight `SPIKE_DETOUR` (50) on banned
>   nodes, so a ban never walls off a better keystone behind it — routes detour, traversing a
>   ban only as a last resort. Split `traversable()` (structural) vs `targetable()` (+excludeSet).
>
> **TWO-PHASE DIET (2026-06-03, user request).** The single-marginal diet became `runDiet(phase,
> selector, …)` run twice, bans accumulating across phases:
> - **PHASE 1 — lowest-marginal**: ban the weakest-contribution droppable node.
> - **PHASE 2 — longest-path**: ban the droppable node sitting DEEPEST in the build (longest
>   path from the class start via `buildDepths` BFS over the allocated set; tie-break lowest
>   marginal). Removing a deep, low-value node frees the most points to reinvest elsewhere.
> Each phase is its own accumulating-greedy loop with independent patience.
>
> **SAVE-TO-POB (2026-06-03, user request, CONFIRMED in real PoB).** The spike applies the diet
> winner to the live spec (`spec:ImportFromNodeList(nil, curClassId, curAscendClassId,
> curSecondaryAscendClassId, hashList, …)`, hashList = winner main-tree ids + kept ascendancy
> ids) and writes `build:SaveDB()` XML to `SPIKE_SAVE_XML` (default
> `src/Builds/mymonk_optimized.xml`). Runs LAST, after all sanity checks, since it mutates the
> live spec. **Caveat:** the headless `loadBuildFromXML` reload reads dps=0 — the headless
> wrapper doesn't drive the GUI skill-selection dropdowns, so a *tree-granted* active skill
> isn't auto-selected. This is a headless artifact, NOT a save bug: the XML is complete (Skills/
> Config/main-skill present), the clean-slate re-score matches the diet best, every optimized id
> is asserted present in the saved `nodes` attr, and the file opens correctly in real PoB
> (user-verified). Don't trust the headless reload number; trust the re-score + node-presence checks.
>
> **FULL P=74 RESULT (beam=10, maxJump=12, mymonk.xml, 8 rounds ~30min):**
> | round | banned | score | dps | ehp | kept? |
> | --- | --- | --- | --- | --- | --- |
> | 0 | — | 251.4 | 814.6 | 3864.1 | base |
> | 1 | Immaterial (+0.0) | 271.2 | 950.8 | 3991.5 | ✅ |
> | 3 | Lucidity | 281.4 | 1022.3 | 4054.5 | ✅ |
> | 4 | Step Like Mist | 287.3 | 1224.6 | 3700.4 | ✅ |
> | **6** | **Mindful Awareness** | **288.4** | **1260.7** | **3639.5** | ✅ |
> | 2,5,7,8 | (various) | <best | — | — | reverted |
>
> **Final: score 288.4 — STRICTLY DOMINATES hand-made** (dps 1260.7>956.6 AND ehp 3639.5>2323.7
> at equal 74 pts, **+44.2%**), banning 4 dead/low-value nodes. Monotonic climb, clean
> convergence, no dps=0 collapse, all sanity checks pass. This is the project's strongest,
> most defensible result and the algorithm is now stable. Remaining blockers: **throughput**
> (~190s/round × 8 = the 30-min cost is too slow for a dialog) and the still-unbuilt UI.

> **THROUGHPUT — MEASURED, REDIRECTS THE STRATEGY (2026-06-04).** The per-call cost is
> dominated by `calcs.perform`, which is **irreducible**: it must run every call because the
> candidate's node mods changed. Cheaper-call optimizations barely move it — the win must come
> from **fewer calls, not cheaper calls.** Measured on `mymonk.xml`, N=200, MIN ms/call (via a
> `SPIKE_ACCEL=1` benchmark added to `SpikeCleanSlateBeam_spec.lua`):
> | calculator path | ms/call | note |
> | --- | --- | --- |
> | default `getMiscCalculator` (FullDPS) | 10.0 | baseline |
> | **accelerated env-reuse (FullDPS)** | **8.8** | only **1.14×** |
> | default (useFullDPS=false) | 8.2 | FullDPS roll-up ≈ 1.8 ms |
> | accel + NoFull | ~8.3 | **~1.2× combined — NOT the 5-10× needed** |
>
> The **accelerated calculator** holds ONE persistent env and per call re-inits with
> `accelerate = { requirementsItems = true, requirementsGems = true, skills = true }` (keeping
> `nodeAlloc = false`, since node alloc is exactly what changes). This skips item/gem/skill
> **re-parse** — which turned out to be a *small* slice. The engine already uses this internally
> (`Calcs.lua:308` `accelerationTbl`; honored by `initEnv`'s `accelerate` table,
> `CalcSetup.lua:343+`). **Correctness: the accel path reproduces 956.6/2323.7 exactly**, so it
> is safe for this build (no candidate node grants a skill that `accelerate.skills` would drop —
> but that risk is real for tree-granted-skill builds and must be re-checked per build).
> `useFullDPS=false` is safe for mymonk because its main skill *is* the FullDPS skill
> (`TotalDPS ≈ FullDPS`); multi-skill/FullDPS builds still need the roll-up or DPS deltas read 0
> (cache caveat, `Calcs.lua:139`).
>
> **CONCLUSION — the §4 lever ranking below (accel > skip-FullDPS > caching > beam) was wrong.**
> `perform` is the ~8 ms floor; accel+NoFull is a flat ~1.2× and that's it. The only levers with
> 5-10× headroom **reduce the call count**: smaller beam width, jump-candidate caps, frontier /
> Pareto pruning, a cross-step set cache, and fewer diet rounds. Apply accel+NoFull anyway
> (it's a free ~1.2× and the accel calculator is the right production primitive), but treat
> **call-count reduction as the real throughput work.**

> **CALL-COUNT MEMO — IMPLEMENTED + VALIDATED (2026-06-04, `SPIKE_MEMO`, default on).** First
> call-count lever, landed in the spike: `scoreSet` memoizes the score by an **order-independent
> sorted-id signature** of the candidate node-set. Overlapping beam frontiers and successive diet
> rounds keep re-producing the *same* resulting set; the cache collapses those to one `perform`.
> Memoizing by the concrete set is independent of the active `excludeSet` (exclusions gate move
> *generation*, not the calc of a fixed set), so it's valid across diet rounds. **Validated:**
> memo on vs off give byte-identical results (depth-15 beam-8: 130.5 / 196.9 / 2553.0 both ways,
> all sanity asserts still pass). Measured **40 % hit rate** (1521 logical evals → 916 performs).
> **Stacked: accel+NoFull (~1.2× cheaper-call) × memo (~1.67× fewer-call) ≈ 1.7–1.9× overall.**
> Still short of a snappy dialog; remaining call-count levers (tighter beam, jump caps, Pareto
> pruning, fewer rounds) are the next work. Spike flags added: `SPIKE_ACCEL`, `SPIKE_MEMO`,
> `SPIKE_BENCH_N`.

> **PARETO-AUGMENTED BEAM — IMPLEMENTED + VALIDATED (2026-06-04, `SPIKE_PARETO`, default on).**
> Second call-count lever. The beam was pure top-N by scalar score, which clusters all N slots in
> one corner of (DPS, EHP) and culls the damage-seeking branch (the §3.5 problem). `selectBeam`
> now keeps top-`beamWidth` by score **plus** up to `SPIKE_PARETO_EXTRA` Pareto-non-dominated
> states (high on one axis even if lower-scoring), and dedups states by node-set signature. Keeping
> the diverse trade-offs alive lets the beam run **narrower** without losing the optimum — fewer
> states expanded ⇒ fewer `perform` calls. **Measured (depth 20, mymonk):** plain beam=12 →
> score 139.5 / **1963 distinct evals**; Pareto beam=6 (+4) → **score 139.5 (identical) / 1534
> evals (−22 %)**. Same optimum at half the base width. Quality-preserving and stacks on accel+memo;
> all sanity asserts still pass (clean-slate re-score = 956.6/2323.7). Flags: `SPIKE_PARETO` (0=off),
> `SPIKE_PARETO_EXTRA` (default max(4, beamWidth/2)).

**In scope (v1):**
- Objective: `score = wDPS * FullDPS + wEHP * TotalEHP` with user-set weights, plus a
  Pareto frontier so hybrid solutions aren't discarded.
- Respect all of PoB's existing pathing/connectivity rules.

**Out of scope (v1) — call out explicitly so we don't over-promise:**
- Provable optimality. The real objective is non-monotone and non-additive (e.g. "more
  projectiles" is worth 0 until a projectile skill is slotted), so no admissible A*
  heuristic exists. v1 is **approximate** (beam search). Say this in the UI.
- Re-slotting skills, swapping items, or changing config to suit the tree.
- Jewel/timeless-jewel optimization, mastery-effect choice optimization, cluster jewels.
- Ascendancy point optimization (different budget pool; can be a v2 toggle).

---

## 2. Objective function (the heart of it)

> **Note:** `GetMiscCalculator()` takes **no arguments** — it returns the cached
> `{func, baseOutput}` built during `CalcsTab:BuildOutput()` (`CalcsTab.lua:693`). Earlier
> drafts wrote `GetMiscCalculator(build)`; that was wrong. Call `BuildOutput()` first if you
> need to guarantee the cache is fresh.

```lua
local calcFunc, calcBase = build.calcsTab:GetMiscCalculator() -- ONCE, outside the loop
-- evaluate a candidate delta:
local out = calcFunc({ addNodes = candidateNodeSet }, useFullDPS) -- returns env.player.output
local score = wDPS * (out.FullDPS or out.TotalDPS or 0) + wEHP * (out.TotalEHP or 0)
```

**Critical constraints (from the code, do not violate):**
- **Build the calculator exactly once.** `GetMiscCalculator` runs a full `initEnv` +
  `perform` pass (`Calcs.lua:123-150`). Calling it per candidate is the #1 perf trap.
- **FullDPS cache caveat.** `Calcs.lua:139-147` documents that without forcing a fresh
  FullDPS roll-up per override, A-vs-A cache reuse makes deltas read as 0. Pass overrides
  through the closure (don't mutate the spec) and respect `useFullDPS` exactly as
  `PowerBuilder` does (`CalcsTab.lua:523`, `589`).
- `addNodes` keys: the tooltip code uses both node objects (`{ [node]=true }`) and ids
  (`{ [node.id]=true }`) depending on context (`PassiveTreeView.lua:1888-1896`).
  Pick the node-object form and stay consistent; verify against `buildModListForNodeList`.

---

## 3. Pipeline

```
build.calcsTab:GetMiscCalculator()         -- env built ONCE → calcFunc, calcBase
        │
        ▼
Compress tree → candidate set {notables, keystones, sockets reachable within budget}
        │   (skip travel/Normal nodes as *targets*; they're path cost, not goals)
        ▼
Precompute travel: BuildPathFromNode from current allocated frontier
        │   → node.pathDist (point cost) + node.path (nodes traversed) for every node
        ▼
Pre-filter: rank candidates by PowerBuilder-style power-per-point (cheap-ish)
        │   → keep top K as expansion fuel
        ▼
Beam search (width N) over connected partial builds, budget-constrained
        │   each beam member scored by calcFunc(real DPS/EHP); connectivity guaranteed
        ▼
Pareto frontier over (FullDPS, TotalEHP)
        │
        ▼
Result list: each = {addedNodes, pointCost, ΔDPS, ΔEHP}; preview + Apply-to-spec
```

**Search algorithm: compressed graph + beam search + Pareto frontier.** A\* was the
original plan but is dropped — see §7 / §8. No A\*, no branch-and-bound "provably optimal"
claim, because the real-calc objective has no admissible heuristic.

### 3.1 Compression (your step 1 — correct)
Targets are `type == "Notable" | "Keystone" | "Socket"` on the main tree. Normal travel
nodes enter a solution as path cost via `node.path`. This shrinks the search from
thousands of nodes to ~hundreds of meaningful targets.

> **OPEN DECISION — travel nodes as expandable targets.** User leaned toward letting the
> beam expand *one node at a time* (incl. travel nodes) so routing emerges from the search
> and travel-node stats are valued naturally — instead of precomputing one route per
> target. This is the most correct option but multiplies the branching factor ~10–50×,
> colliding with the throughput risk in §4. Viable **only** if the §3.3 pre-filter is
> aggressive. Not finalized; see §8.

### 3.2 Travel precompute (your step 2 — correct, already built)
`PassiveSpec:BuildPathFromNode(root)` is BFS that fills `node.pathDist` (skill-point cost
to reach from the allocated frontier) and `node.path` (the actual nodes traversed),
already obeying every connectivity rule (`PassiveSpec.lua:1083-1105`). **Reuse it.** Re-run
it whenever the allocated frontier changes during search (after committing a beam step),
since pathDist is relative to what's currently allocated.

### 3.3 Pre-filter — TESTED, then REPLACED by dedup + narrow beam
**A cheap modlist-sum pre-filter does NOT work** (measured 2026-06-03, `mymonk.xml`):
recall of the true top-20 was only 10% at M=100, 40% at M=500. The raw modlist sum rewards
big-literal nodes (+attributes/+life) and badly under-ranks the actual winners — keystones
like Mind Over Matter (tiny literal, huge calc impact) and small interaction nodes
(Snowpiercer, Chakra of Thought). This is the multiplicative/conditional-calc point from §7,
now proven: a linear surrogate is a bad *ranking*, not just a bad objective.

**Chosen approach instead: no surrogate — dedup by `modKey` + a narrower beam.**
- Dedup: 3,228 candidate nodes → **1,415 unique `modKey`s (2.28×)**. `calcFunc` is cached by
  `modKey` (as `PowerBuilder` does, `CalcsTab.lua:575`), so duplicate-effect nodes are free
  after the first eval. This also caps per-step cost at `uniqueCount`.
- Beam time (projected from 5.67 ms/call, dedup cap applied):

  | beam N | topK | depth | unique calcs | time |
  | --- | --- | --- | --- | --- |
  | 150 | 30 | 15 | 21,225 | 120s |
  | 40 | 20 | 12 | 9,600 | 54s |
  | **25** | **15** | **10** | **3,750** | **~21s** |

- **N=25 → ~21s is a usable dialog** (coroutine + progress). No pre-filter needed.

**Caveats (do not treat the table as final):**
1. Projection is best-case: assumes the `modKey` cache survives a whole beam step. The
   allocated frontier shifts each step and can invalidate cached deltas (`Calcs.lua:139`).
   Real time may be higher — only a real beam measures it.
2. Narrow beam = lower quality. N=25 can miss "long travel to a great keystone" paths that
   PoE2 trees are full of. Validate N=25 results vs. N=150 on a known build before locking
   the width.
3. One build / one tree. Heavier skills (minion, DoT) cost more per call.

### 3.4 Beam search (your step 8 — promoted to primary, correctly)
- State: a connected set of added nodes + points spent + last full output.
- Expand each beam member by each top-K reachable target (add target + its `node.path`).
- Reject expansions exceeding budget (`CountAllocNodes` accounting).
- Score survivors with `calcFunc`. Keep top N by scalarized score, **plus** anything on
  the Pareto frontier (§3.5).
- Suggested defaults: beam width N≈100–200, top-K≈30 candidates/step, depth = budget.
- Connectivity is automatic: we only ever add a target *together with* its `node.path`
  back to the allocated frontier (your step 5 — never repair disconnection afterward).

### 3.5 Pareto frontier (your step 7 — correct, in output space)
Keep non-dominated `(FullDPS, TotalEHP)` solutions, not just argmax of the scalar score.
Frontier lives in **calc-output space**, since node-stat space is meaningless here.

---

## 4. Performance budget (the project's main risk) — MEASURED

**Spike results (2026-06-03, tree 0_5, in the CI container).** Two builds — note the real
build with a slotted skill costs ~2.7× more per call, so design against the REAL number:

| Build | Depth-1 ms/call | Eligible nodes | Projected beam (150×30×15 = 67,500) |
| --- | --- | --- | --- |
| Empty default (no skill) | 2.09 ms | 3,276 | ~141s (~2.4 min) |
| **Real Monk (`src/Builds/mymonk.xml`)** | **5.74 ms** | 3,228 | **~387s (~6.5 min)** |

Real-build run base = 956.6 DPS / 2323.7 EHP; top suggestions were sensible hybrids
(Mind Over Matter +866 EHP; Glaciation/Snowpiercer/Breath of Ice = cold DPS; depth-2 found
MoM + Stormcharged +1416). **The objective function is validated** — it reads the real
build and produces believable offense/defense trade-offs.

**This settles two things:**
1. **The §3.3 pre-filter is mandatory, not optional.** At **5.74 ms/call (real build)** a
   full-width beam is **~6.5 minutes**. The candidate set must be cut ~10× (power-per-point)
   before full `calcFunc` evals, and the dialog must run as a **coroutine with progress +
   Cancel** (like `PowerBuilder`, `CalcsTab.lua:538`), never synchronously.
2. **"Travel nodes as expandable targets" (§8) is dead for v1.** Multiplying a 6.5-min
   baseline by 10–50× is absurd. Precompute one route per target via `BuildPathFromNode`.

Also observed: duplicate same-name nodes are evaluated separately (three "Strength"
entries). Production must **dedup by `modKey`** as `PowerBuilder` already does
(`cache[node.modKey]`, `CalcsTab.lua:575`) to avoid wasted identical evals.

Conditions still required regardless:
1. `GetMiscCalculator` is called once;
2. the FullDPS cache caveat (§2) is handled so deltas aren't silently 0;
3. the pre-filter keeps full evals off the long tail of useless candidates.

> The empty-build run shows only EHP deltas (DPS ≈ 0) because no skill is slotted — the
> objective is genuinely reading the calc engine. Run with `SPIKE_BUILD_XML` pointing at a
> real exported build to see DPS deltas.

**De-risk before building the full thing:** a throwaway spike that calls
`GetMiscCalculator` once and brute-forces all single-node and 2-node additions, timing it
on real hardware, confirms throughput. (This is exactly the loop already at
`PassiveTreeView.lua:1875` and `CalcsTab.lua:589`.)

**The spike is written and has been run:** `spec/System/SpikeTreeOptimizer_spec.lua`. It is
a busted spec; run it via the project's CI container (LuaJIT + busted live there, not on a
typical dev machine). From the repo root:

```powershell
# empty build (plumbing + throughput only)
docker run --rm -v "${PWD}:/work" -w /work `
  ghcr.io/pathofbuildingcommunity/pathofbuilding-tests:latest `
  busted --lua=luajit --filter Spike

# real build (meaningful DPS deltas) — XML must be under the repo root so the mount sees it
docker run --rm -v "${PWD}:/work" -w /work -e SPIKE_BUILD_XML=/work/mybuild.xml `
  ghcr.io/pathofbuildingcommunity/pathofbuilding-tests:latest `
  busted --lua=luajit --filter Spike
```

Notes that cost an hour to discover: run from the **repo root** (where `.busted` lives),
**no explicit spec path** (it double-resolves against `directory=src`), select via
`--filter Spike`. Results are in §4.

---

## 5. UI / integration

- New tab/dialog registered alongside the others in `Modules/Build.lua:517-536`
  (e.g. `self.optimizerTab = new("OptimizerTab", self)`), or a modal launched from the
  Tree tab toolbar. A modal is lower-risk for v1.
- Controls: DPS weight, EHP weight, "include FullDPS" toggle, budget (default = remaining
  points), beam width (advanced), Run / Cancel + progress bar.
- Results: sortable list of frontier solutions (ΔDPS, ΔEHP, points). Selecting one
  **previews** the added nodes on the tree (reuse `PassiveTreeView` highlight path);
  **Apply** allocates them via `PassiveSpec:AllocNode` per node (which itself calls
  `BuildAllDependsAndPaths`). Apply must go through an undo-able spec edit so the user can
  revert — check how `treeTab` wraps spec mutations for undo.
- Run on a **copy of the current spec** so the live build isn't mutated mid-search; only
  commit on Apply.

---

## 6. Correctness checklist (review gates before merge)

- [ ] `GetMiscCalculator` called exactly once per Run (assert/guard it).
- [ ] FullDPS deltas are non-zero for a known-good node (regression vs. hover tooltip).
- [ ] Budget never exceeded vs. `CountAllocNodes`; ascendancy/socket pools not mixed in.
- [ ] Every emitted solution is connected to class start (validate via `FindStartFromNode`).
- [ ] Search runs on a spec copy; live build unchanged unless Apply pressed.
- [ ] Apply is undoable.
- [ ] Pathing edge cases honored: masteries, ascendancy boundaries, `unlockConstraint`,
      intuitive-leap-like jewels (`node.intuitiveLeapLikesAffecting`, seen at
      `PassiveTreeView.lua:1902` and `PassiveSpec.lua:927`).
- [ ] Cancel actually stops the loop and frees the spec copy.

---

## 7. What changed from the original plan, and why

| Original step | Verdict | Change |
| --- | --- | --- |
| 1 Compress tree | ✅ keep | targets = notables/keystones/sockets only |
| 2 Precompute travel | ✅ keep | already built: `BuildPathFromNode` |
| 3 Weighted stat score | ⚠️ demote | becomes a **pre-filter**, not the objective |
| 4 A*/B&B "provably optimal" | ❌ dropped | no admissible heuristic vs. real calc → A\* collapses to heuristic best-first; use beam instead |
| 5 Maintain connectivity | ✅ keep | add target + its `node.path` together; never repair |
| 6 Aggressive pruning | ✅ keep | as pre-filter pruning, output-bound not stat-bound |
| 7 Pareto frontier | ✅ keep | frontier in (DPS, EHP) **output** space |
| 8 Beam search | ✅ promote | this is the primary search, not a fallback |
| 9 Avoid simulated annealing | ➖ neutral | once objective = real calc, the B&B-vs-SA split mostly dissolves; both are heuristic |
| 10 Architecture | ✅ revised | see §3 — objective is `calcFunc`, not a stat sum |

**The one correction that drives all the others:** in PoB, a passive node parses into a
`modList` (`PassiveTree.lua:425-477`) whose value is realized only through the calc engine
(multiplicative increases, conditional flags, "more" multipliers). So you must score
candidates with `calcFunc`, the engine already sitting in this repo — which also means
"provably optimal" is off the table and beam search + Pareto is the honest, working design.

---

## 8. Open decisions (not yet finalized)

1. **A\* — RESOLVED: dropped.** Algorithm is compressed graph + beam + Pareto. (§3, §7 row 4.)
2. **Travel nodes as expandable targets — LEANING NO (per §4 measurement).** At 2.09 ms/call
   a full-width beam is already ~2.4 min; the ~10–50× branching from expandable travel
   nodes pushes it to tens of minutes. Recommend v1 precomputes one route per target via
   `BuildPathFromNode` and revisits this only if the §3.3 pre-filter proves it bounds evals.
   (§3.1, §4.)
3. **Build order — OPEN.** Brute-force depth-1/2 spike first (validates `expand()` +
   throughput, recommended) vs. straight to beam vs. brute-force-only product. The
   brute-forcer's `expand()` primitive is the beam's inner step regardless, so it is built
   once either way.
