//! The clean-slate beam search + Pareto selection + elimination diet — the
//! production tree-optimizer search, run entirely inside the worker pool.
//!
//! This is the search the design doc (docs/tree-optimizer-design.md) converged
//! on, ported out of `examples/beam_ab.rs` so it ships as a real crate module
//! the FFI (`pob_opt_run_beam`) and the A/B example both call. It does NOT score
//! anything itself: it takes a `score` closure (driven by `Pool::score_batch3`)
//! and only decides which candidate node-sets to evaluate.
//!
//! Algorithm (see the design doc for the why):
//!   * Grow a connected tree from the class start, expanding each beam state by
//!     (a) single adjacent targetable nodes and (b) path-JUMPS to a distant
//!     notable/keystone via shortest path (the jump escapes the all-defense
//!     local optimum by investing in far damage clusters).
//!   * Select top-N by scalar score PLUS up to `pareto_extra` Pareto-non-dominated
//!     (dps, ehp) states, so the damage-seeking branch survives a narrow beam.
//!   * Elimination diet: ban the lowest-marginal droppable node, re-optimize, keep
//!     the ban iff it improved; enablers (removal collapses an axis) are never
//!     banned; bans are SOFT (pathing may still detour through them).

use std::collections::{HashMap, HashSet};

pub const T_NORMAL: i32 = 1;
pub const T_NOTABLE: i32 = 2;
pub const T_KEYSTONE: i32 = 3;

/// The tree as the worker exports it (via `__pob_graph_export`): adjacency + node
/// types + the class start + the build's real point budget.
pub struct Graph {
    pub start: i32,
    pub budget: usize,
    pub links: HashMap<i32, Vec<i32>>,
    pub ty: HashMap<i32, i32>,
    /// Penalty constants the worker appends after the node data (see `parse`):
    /// the ORIGINAL build's dps/ehp (the floor to protect) and the score weights /
    /// quadratic-penalty strength. Used by `penalized_score` to dock full-budget
    /// trees that regress an axis below the original. Defaults if the trailer is
    /// absent (older export): refs=1.0 (=> no meaningful floor), weights=1, k=0.
    pub ref_dps: f64,
    pub ref_ehp: f64,
    pub w_dps: f64,
    pub w_ehp: f64,
    pub penalty_k: f64,
}

impl Graph {
    /// Parse the flat int array `__pob_graph_export` emits:
    ///   [ startId, budgetP, nNodes, (id, typeCode, nLinks, link...) * nNodes ]
    pub fn parse(flat: &[i32]) -> Graph {
        let start = flat[0];
        let budget = flat[1].max(0) as usize;
        let n = flat[2] as usize;
        let mut links = HashMap::new();
        let mut ty = HashMap::new();
        let mut p = 3usize;
        for _ in 0..n {
            let id = flat[p];
            let tc = flat[p + 1];
            let k = flat[p + 2] as usize;
            p += 3;
            let mut adj = Vec::with_capacity(k);
            for _ in 0..k {
                adj.push(flat[p]);
                p += 1;
            }
            links.insert(id, adj);
            ty.insert(id, tc);
        }
        // TRAILER (optional, appended after the node data): 5 ints, each ×1000:
        //   [ refDps, refEhp, w_dps, w_ehp, penalty_k ]. Absent on older exports —
        //   fall back to refs=1.0 (no floor), weights=1, k=0 (penalty disabled).
        let (ref_dps, ref_ehp, w_dps, w_ehp, penalty_k) = if flat.len() >= p + 5 {
            (
                (flat[p] as f64 / 1000.0).max(1.0),
                (flat[p + 1] as f64 / 1000.0).max(1.0),
                flat[p + 2] as f64 / 1000.0,
                flat[p + 3] as f64 / 1000.0,
                flat[p + 4] as f64 / 1000.0,
            )
        } else {
            (1.0, 1.0, 1.0, 1.0, 0.0)
        };
        Graph { start, budget, links, ty, ref_dps, ref_ehp, w_dps, w_ehp, penalty_k }
    }

    /// Final-tree score with the DPS/EHP regression penalty. `growth` is the
    /// worker's smooth absolute score (used to GROW the beam); this re-derives a
    /// comparable score for FULL-BUDGET trees that rewards each axis's gain above
    /// the original build and docks any shortfall quadratically (so no finite gain
    /// on one axis buys back a collapse on the other). With `penalty_k == 0` (no
    /// trailer) it reduces to the smooth growth score, preserving old behavior.
    pub fn penalized_score(&self, growth: f64, dps: f64, ehp: f64) -> f64 {
        if self.penalty_k <= 0.0 {
            return growth;
        }
        let d_dps = dps / self.ref_dps - 1.0;
        let d_ehp = ehp / self.ref_ehp - 1.0;
        let term = |d: f64, w: f64| {
            if d >= 0.0 {
                w * d
            } else {
                -self.penalty_k * d * d
            }
        };
        100.0 * (term(d_dps, self.w_dps) + term(d_ehp, self.w_ehp))
    }

    pub fn is_target(&self, id: i32) -> bool {
        matches!(self.ty.get(&id), Some(&T_NORMAL | &T_NOTABLE | &T_KEYSTONE))
    }
}

/// Tunables for one full search (beam + diet). All have sane defaults; the host
/// passes a key=value override string (see `BeamParams::parse`). `cap_points` of 0
/// means "use the build's real budget the graph export carries".
#[derive(Clone, Copy)]
pub struct BeamParams {
    pub cap_points: usize,
    pub beam_width: usize,
    pub max_jump: usize,
    pub pareto_extra: usize,
    pub detour: i64,
    pub patience_max: usize,
    pub max_rounds: usize,
    /// Per-state jump-candidate cap, ranked by the power seed (power-per-point).
    /// 0 => uncapped (every notable/keystone within max_jump becomes a candidate,
    /// the old behavior). When >0, keep the top-K jump targets by seed power-per-
    /// point (keystones always kept — their value is structural, not power-ranked).
    /// Unlike the doc's distance-cap (which starved distant DAMAGE), ranking by real
    /// per-node power lets a tight cap keep the FAR damage notables and drop junk.
    pub max_jump_cand: usize,
    /// Number of top-power distant clusters to seed as extra round-0 beam anchors
    /// (besides the bare start). 0 => start-only (old behavior). Each anchor is the
    /// start tree + the shortest path to one high-power target, so round 0 explores
    /// genuinely-good far routes from step 1 instead of greedily wandering toward
    /// near EHP (the local-optimum trap the design doc hit repeatedly).
    pub seed_anchors: usize,
    /// Number of ADDITIONAL perturbed-seed restarts of the whole search (round 0 +
    /// diet). 0 => single search (old behavior). Each restart re-runs from a different
    /// band of seed anchors (offset by `restart_index * seed_anchors`), so it commits
    /// round 0 to structurally-different far routes and can escape a local optimum the
    /// primary search settled into. The best final tree across all restarts wins. The
    /// power seed + memo are shared, so restart N is much cheaper than the first.
    pub restarts: usize,
    /// Print step/round progress to stdout (the example wants it; the FFI path
    /// leaves it off so a GUI host isn't spammed).
    pub verbose: bool,
}

impl Default for BeamParams {
    fn default() -> Self {
        BeamParams {
            cap_points: 0, // 0 => graph.budget
            beam_width: 8,
            max_jump: 12,
            pareto_extra: 0, // 0 => max(4, beam_width/2)
            detour: 50,
            // Throughput is no longer the bottleneck (worker pool + memoization), so
            // the diet runs long and bans hard: a handful of consecutive low-marginal
            // misses no longer means "stop" — keep probing deeper droppable nodes.
            patience_max: 8,
            max_rounds: 40,
            max_jump_cand: 0, // 0 => uncapped
            seed_anchors: 6,
            restarts: 2,
            verbose: false,
        }
    }
}

impl BeamParams {
    /// Parse a newline- or comma-delimited `key=value` override string. Unknown
    /// keys error (catch typos early); absent keys keep the default. Keys:
    /// cap_points, beam_width, max_jump, pareto_extra, detour, patience_max,
    /// max_rounds, max_jump_cand, seed_anchors, restarts, verbose (0/1). Empty string =>
    /// all defaults.
    pub fn parse(s: &str) -> Result<BeamParams, String> {
        let mut p = BeamParams::default();
        for raw in s.split(['\n', ',']) {
            let line = raw.trim();
            if line.is_empty() {
                continue;
            }
            let (key, val) = line
                .split_once('=')
                .ok_or_else(|| format!("beam params: '{line}' is not key=value"))?;
            let key = key.trim();
            let val = val.trim();
            let num = |v: &str| v.parse::<i64>().map_err(|_| format!("beam params: {key}='{v}' is not a number"));
            match key {
                "cap_points" => p.cap_points = num(val)?.max(0) as usize,
                "beam_width" => p.beam_width = num(val)?.max(1) as usize,
                "max_jump" => p.max_jump = num(val)?.max(0) as usize,
                "pareto_extra" => p.pareto_extra = num(val)?.max(0) as usize,
                "detour" => p.detour = num(val)?.max(1),
                "patience_max" => p.patience_max = num(val)?.max(0) as usize,
                "max_rounds" => p.max_rounds = num(val)?.max(0) as usize,
                "max_jump_cand" => p.max_jump_cand = num(val)?.max(0) as usize,
                "seed_anchors" => p.seed_anchors = num(val)?.max(0) as usize,
                "restarts" => p.restarts = num(val)?.max(0) as usize,
                "verbose" => p.verbose = num(val)? != 0,
                other => return Err(format!("beam params: unknown key '{other}'")),
            }
        }
        Ok(p)
    }

    /// Resolve the 0-means-default fields against the graph.
    fn resolved(&self, graph: &Graph) -> (usize, usize) {
        let cap = if self.cap_points > 0 { self.cap_points } else { graph.budget };
        let pareto = if self.pareto_extra > 0 { self.pareto_extra } else { (self.beam_width / 2).max(4) };
        (cap, pareto)
    }
}

/// The result of a full search: the winning connected id-set (incl. start, sorted)
/// and its score/dps/ehp.
pub struct BeamResult {
    pub ids: Vec<i32>,
    pub score: f64,
    pub dps: f64,
    pub ehp: f64,
    /// Total scoring calls dispatched (across all beam + diet rounds).
    pub evals: usize,
}

/// A beam state: the connected allocated id-set (incl. start) plus its score.
#[derive(Clone)]
struct State {
    ids: Vec<i32>, // sorted, includes start
    set: HashSet<i32>,
    score: f64,
    dps: f64,
    ehp: f64,
}

/// The one-time PowerBuilder-style seed: for each notable/keystone target, its real
/// power-per-point — the penalized-score gain of adding ONLY that target (+ its
/// shortest path) to the start tree, divided by the point cost. Computed once before
/// round 0 in a single batch (~1s parallel, measured), independent of the diet bans.
///
/// This is the calc-based ranking the design doc names as "the only way to cap jump
/// candidates without the quality hit" — a distant DAMAGE notable scores high here
/// (real dps gain), distant junk scores low, so a tight `max_jump_cand` keeps the
/// far damage the blind distance-cap starved.
struct PowerSeed {
    /// target id -> power-per-point (higher = better value). Absent => unreachable
    /// or non-positive (treated as lowest rank).
    ppp: HashMap<i32, f64>,
}

impl PowerSeed {
    fn power(&self, id: i32) -> f64 {
        self.ppp.get(&id).copied().unwrap_or(f64::NEG_INFINITY)
    }
}

/// Score every notable/keystone target as a single-add from the start, in one batch,
/// and rank by penalized power-per-point. `base` is the start-only [score, dps, ehp].
fn compute_seed(
    g: &Graph,
    base: [f64; 3],
    detour: i64,
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    total_evals: &mut usize,
    score: &mut dyn FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
) -> PowerSeed {
    let start_set: HashSet<i32> = HashSet::from([g.start]);
    // Shortest path (point cost) from the start to every reachable node, so a seed
    // candidate is the start tree + the real travel path to the target (same as a
    // beam jump), and power-per-point divides by that real cost.
    let (pdist, prev) = paths_from_set(g, &start_set, &HashSet::new(), detour);
    let base_pen = g.penalized_score(base[0], base[1], base[2]);

    let mut targets: Vec<(i32, Vec<i32>, usize)> = Vec::new();
    for (&tid, &d) in &pdist {
        if d >= 1
            && tid != g.start
            && matches!(g.ty.get(&tid), Some(&T_NOTABLE | &T_KEYSTONE))
        {
            let path = path_to(&prev, &start_set, tid);
            if path.is_empty() {
                continue;
            }
            let mut ids: Vec<i32> = start_set.iter().copied().chain(path.iter().copied()).collect();
            ids.sort_unstable();
            targets.push((tid, ids, d));
        }
    }

    let to_score: Vec<Vec<i32>> =
        targets.iter().map(|(_, ids, _)| ids.clone()).filter(|s| !memo.contains_key(s)).collect();
    if !to_score.is_empty() {
        let scores = score(&to_score);
        *total_evals += to_score.len();
        for (s, sc) in to_score.iter().zip(scores.iter()) {
            memo.insert(s.clone(), *sc);
        }
    }

    let mut ppp: HashMap<i32, f64> = HashMap::new();
    for (tid, ids, d) in &targets {
        let sc = memo[ids];
        let pen = g.penalized_score(sc[0], sc[1], sc[2]);
        if pen.is_finite() {
            // Gain over the start tree, per point spent reaching the target. Keystones
            // can have a structural value the single-add misses, so they are kept
            // regardless of rank (see run_beam); ppp only orders the cap.
            ppp.insert(*tid, (pen - base_pen) / (*d as f64));
        }
    }
    PowerSeed { ppp }
}

/// Run the full beam + elimination diet. `score(&[set])` returns [score, dps, ehp]
/// per set (NAN on failure) — the only contact with the calc engine. Memoizes by
/// id-set across all rounds (overlapping frontiers re-produce the same sets).
pub fn optimize(
    graph: &Graph,
    params: &BeamParams,
    mut score: impl FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
) -> BeamResult {
    let (cap_points, pareto_extra) = params.resolved(graph);
    let mut memo: HashMap<Vec<i32>, [f64; 3]> = HashMap::new();
    let mut total_evals = 0usize;

    // One-time power seed (before round 0): ranks distant jump targets by REAL power
    // so the beam can escape the near-EHP local optimum. Reused across all diet rounds.
    let seed_base = score_or_memo(&[graph.start], &mut memo, &mut total_evals, &mut score);
    let seed = compute_seed(graph, seed_base, params.detour, &mut memo, &mut total_evals, &mut score);

    macro_rules! log {
        ($($a:tt)*) => { if params.verbose { println!($($a)*); } };
    }

    // The beam GROWS on the worker's smooth absolute score (monotonic in both axes,
    // so a partial tree below the original build still has a meaningful gradient to
    // climb). The DIET, which compares FULL-BUDGET trees, judges them by the
    // penalized score instead — that's where the DPS/EHP regression floor lives. A
    // tree that wins on smooth score by gutting DPS for EHP loses on penalized
    // score, so the diet bans the offending nodes. (penalty_k==0 => identity.)
    let repenalize = |st: &mut State| { st.score = graph.penalized_score(st.score, st.dps, st.ehp); };

    // One full search (round 0 + the elimination diet) seeded from anchor band
    // `anchor_offset`. Factored out so the multi-seed restart loop below can run it
    // repeatedly from DIFFERENT anchor bands, sharing the power seed + memo, and keep
    // the best converged tree. Returns the converged (penalized-scored) State.
    let run_full = |anchor_offset: usize,
                        memo: &mut HashMap<Vec<i32>, [f64; 3]>,
                        total_evals: &mut usize,
                        score: &mut dyn FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>|
     -> State {
        // Round 0: unconstrained.
        log!("ROUND 0 (unconstrained, anchor_offset={anchor_offset})");
        let mut best = run_beam(
            graph, cap_points, params.beam_width, params.max_jump, pareto_extra,
            params.max_jump_cand, params.seed_anchors, anchor_offset, &seed,
            &HashSet::new(), params.detour, memo, total_evals, params.verbose, score,
        );
        repenalize(&mut best);
        log!(
            "  round 0 best: penalized score={:.1} dps={:.1} ehp={:.1} pts={}",
            best.score, best.dps, best.ehp, best.ids.len() - 1
        );

        let mut banned: HashSet<i32> = HashSet::new();
        let mut tried: HashSet<i32> = HashSet::new();
        let mut patience = 0usize;
        let mut round = 0usize;

        // Elimination diet. Each round, in order:
        //
        //   (a) DEAD-LEAF reclaim (batch, no patience cost). Every notable/keystone that
        //       is a LEAF of the allocation (≤1 allocated neighbour, so it carries no
        //       travel for any other node) AND contributes ≤0 marginal is a wasted point.
        //       Ban ALL of them at once and re-run the full beam ONCE with the whole leaf
        //       set excluded — one full-algo run per round, not one per dead leaf. The
        //       leaf test is what protects the main path: a load-bearing node has ≥2
        //       allocated neighbours and never qualifies. Accept iff the re-grown tree
        //       does not regress (≥ best, not strictly >) — shedding dead leaves and
        //       re-spending their points is a structural win even at a tie. No patience
        //       tick either way; this is opportunistic cleanup, not a real probe.
        //
        //   (b) NORMAL ban. If no dead leaves remained, ban the single lowest-marginal
        //       droppable node, re-optimize, keep the ban iff it strictly improved; else
        //       revert + tick patience. Enablers (removal collapses an axis) are never
        //       banned. Bans are SOFT: pathing may still detour through a banned node, it
        //       just can't be targeted.
        while round < params.max_rounds && patience < params.patience_max {
            round += 1;
            let marg = analyze_marginals(graph, &best, memo, total_evals, score);
            let droppable = |m: &&Marginal| {
                !m.is_enabler && !banned.contains(&m.id) && !tried.contains(&m.id)
            };

            // (a) Collect EVERY dead-leaf notable/keystone and shed them together. A
            // worthless LEAF tip (notable or keystone) is a wasted point regardless of
            // type — e.g. a notable like "Splinters" hanging off the edge whose stats do
            // nothing for this build. The leaf test (≤1 allocated neighbour) is what keeps
            // this safe: a load-bearing node has ≥2 allocated neighbours and never
            // qualifies, so the main travel route is never corrupted.
            let dead_leaves: Vec<i32> = marg
                .iter()
                .filter(droppable)
                .filter(|m| {
                    matches!(graph.ty.get(&m.id), Some(&T_NOTABLE | &T_KEYSTONE))
                        && m.marginal <= 1e-6
                        && is_alloc_leaf(graph, &best.set, m.id)
                })
                .map(|m| m.id)
                .collect();
            if !dead_leaves.is_empty() {
                let mut trial = banned.clone();
                trial.extend(dead_leaves.iter().copied());
                log!("  round {round}: reclaim {} DEAD LEAF node(s) {:?}", dead_leaves.len(), dead_leaves);
                let mut r = run_beam(
                    graph, cap_points, params.beam_width, params.max_jump, pareto_extra,
                    params.max_jump_cand, params.seed_anchors, anchor_offset, &seed,
                    &trial, params.detour, memo, total_evals, params.verbose, score,
                );
                repenalize(&mut r);
                if r.score >= best.score - 1e-6 {
                    log!("    RECLAIMED {:.1} -> {:.1} dps={:.1} ehp={:.1}. Keeping bans.", best.score, r.score, r.dps, r.ehp);
                    best = r;
                    banned = trial; // permanent: these dead leaves won't return as targets
                    tried.clear();
                } else {
                    // Regressed (rare — re-fill couldn't recover): keep the bans anyway so
                    // we don't re-detect the same dead leaves next round, but don't adopt
                    // the worse tree. The normal diet below will work from the old best.
                    log!("    leaf reclaim regressed ({:.1} vs {:.1}). Banning targets but keeping tree.", r.score, best.score);
                    banned = trial;
                }
                continue; // one full run this round; re-analyze marginals fresh next round
            }

            // (b) No dead leaves left — normal single lowest-marginal ban.
            let victim = marg
                .iter()
                .filter(droppable)
                .min_by(|a, b| a.marginal.partial_cmp(&b.marginal).unwrap());
            let victim = match victim {
                Some(v) => v.clone(),
                None => {
                    log!("  round {round}: no untried droppable node left. Stop.");
                    break;
                }
            };
            tried.insert(victim.id);
            let mut trial = banned.clone();
            trial.insert(victim.id);
            log!("  round {round}: ban {} (marginal {:+.1})", victim.id, victim.marginal);
            let mut r = run_beam(
                graph, cap_points, params.beam_width, params.max_jump, pareto_extra,
                params.max_jump_cand, params.seed_anchors, anchor_offset, &seed,
                &trial, params.detour, memo, total_evals, params.verbose, score,
            );
            repenalize(&mut r);
            if r.score > best.score + 1e-6 {
                log!("    IMPROVED {:.1} -> {:.1} dps={:.1} ehp={:.1}. Keeping ban.", best.score, r.score, r.dps, r.ehp);
                best = r;
                banned = trial;
                patience = 0;
                tried.clear();
            } else {
                log!(
                    "    no improvement ({:.1} vs {:.1}). Revert. patience {}/{}",
                    r.score, best.score, patience + 1, params.patience_max
                );
                patience += 1;
            }
        }
        best
    };

    // Primary search (anchor band 0) plus `restarts` perturbed restarts. Each restart
    // takes the NEXT band of `seed_anchors` targets (offset by k·seed_anchors), so it
    // commits round 0 to genuinely-different far routes and can land in a different
    // basin than the primary search. Keep the best converged tree across all of them.
    // The power seed + memo are shared, so a restart that re-explores overlapping sets
    // is much cheaper than the first run. Restarts are skipped when seeding is off
    // (seed_anchors==0) — there'd be no perturbation to apply.
    let mut best = run_full(0, &mut memo, &mut total_evals, &mut score);
    if params.seed_anchors > 0 {
        for k in 1..=params.restarts {
            let offset = k * params.seed_anchors;
            log!("=== RESTART {k}/{} (anchor band offset {offset}) ===", params.restarts);
            let cand = run_full(offset, &mut memo, &mut total_evals, &mut score);
            if cand.score > best.score + 1e-6 {
                log!(
                    "  restart {k} WON: {:.1} -> {:.1} dps={:.1} ehp={:.1}",
                    best.score, cand.score, cand.dps, cand.ehp
                );
                best = cand;
            } else {
                log!("  restart {k} no better ({:.1} vs {:.1})", cand.score, best.score);
            }
        }
    }

    BeamResult {
        ids: best.ids,
        score: best.score,
        dps: best.dps,
        ehp: best.ehp,
        evals: total_evals,
    }
}

/// One full beam pass (no diet): grow a connected tree from the start to
/// `cap_points`, returning the top state. `excluded` is the soft-ban set the diet
/// fills (empty for round 0).
#[allow(clippy::too_many_arguments)]
fn run_beam(
    g: &Graph,
    cap_points: usize,
    beam_width: usize,
    max_jump: usize,
    pareto_extra: usize,
    max_jump_cand: usize,
    seed_anchors: usize,
    // How many top-ranked anchor targets to SKIP before taking `seed_anchors`. 0 for
    // the primary search; on a multi-seed restart this is `restart_index *
    // seed_anchors`, so each restart commits round 0 to a DIFFERENT band of far
    // routes (the local-optimum escape — see `optimize`).
    anchor_offset: usize,
    seed: &PowerSeed,
    excluded: &HashSet<i32>,
    detour: i64,
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    total_evals: &mut usize,
    verbose: bool,
    score: &mut dyn FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
) -> State {
    let start_set: HashSet<i32> = HashSet::from([g.start]);
    let base = score_or_memo(&[g.start], memo, total_evals, score);
    let mut beam = vec![State {
        ids: vec![g.start],
        set: start_set.clone(),
        score: base[0],
        dps: base[1],
        ehp: base[2],
    }];

    let targetable = |v: i32| g.is_target(v) && !excluded.contains(&v);

    // Multi-anchor seeding: besides the bare start, seed the beam with states anchored
    // on the top-power distant targets (by seed power-per-point), so round 0 commits to
    // genuinely-good far routes from step 1 instead of greedily climbing near EHP. Each
    // anchor = start tree + the shortest path to one high-power target (within budget,
    // not soft-banned). Scored in one batch with the base.
    if seed_anchors > 0 {
        let (pdist, prev) = paths_from_set(g, &start_set, excluded, detour);
        let mut ranked: Vec<(i32, usize)> = pdist
            .iter()
            .filter(|(&tid, &d)| {
                d >= 1 && d <= cap_points && tid != g.start && targetable(tid) && seed.power(tid).is_finite()
            })
            .map(|(&tid, &d)| (tid, d))
            .collect();
        ranked.sort_by(|a, b| seed.power(b.0).partial_cmp(&seed.power(a.0)).unwrap());

        let mut anchor_sets: Vec<(Vec<i32>, HashSet<i32>)> = Vec::new();
        for (tid, _) in ranked.into_iter().skip(anchor_offset).take(seed_anchors) {
            let path = path_to(&prev, &start_set, tid);
            if path.is_empty() {
                continue;
            }
            let mut set = start_set.clone();
            for &a in &path {
                set.insert(a);
            }
            let mut ids: Vec<i32> = set.iter().copied().collect();
            ids.sort_unstable();
            anchor_sets.push((ids, set));
        }
        let to_score: Vec<Vec<i32>> =
            anchor_sets.iter().map(|(ids, _)| ids.clone()).filter(|s| !memo.contains_key(s)).collect();
        if !to_score.is_empty() {
            let scores = score(&to_score);
            *total_evals += to_score.len();
            for (s, sc) in to_score.iter().zip(scores.iter()) {
                memo.insert(s.clone(), *sc);
            }
        }
        for (ids, set) in anchor_sets {
            let sc = memo[&ids];
            if sc[0].is_finite() {
                beam.push(State { ids, set, score: sc[0], dps: sc[1], ehp: sc[2] });
            }
        }
    }

    for step in 1..=cap_points {
        let mut cand_sets: Vec<Vec<i32>> = Vec::new();
        let mut seen: HashSet<Vec<i32>> = HashSet::new();
        let mut origin: Vec<(Vec<i32>, HashSet<i32>)> = Vec::new();

        for st in &beam {
            let pts_used = st.ids.len() - 1;
            if pts_used >= cap_points {
                // full: carry forward unchanged (will re-enter selection)
                if seen.insert(st.ids.clone()) {
                    origin.push((st.ids.clone(), st.set.clone()));
                    cand_sets.push(st.ids.clone());
                }
                continue;
            }
            let points_left = cap_points - pts_used;
            let (pdist, prev) = paths_from_set(g, &st.set, excluded, detour);

            // (a) single adjacent targetable nodes
            let mut moves: Vec<Vec<i32>> = Vec::new();
            for &id in &st.ids {
                if let Some(adj) = g.links.get(&id) {
                    for &v in adj {
                        if !st.set.contains(&v) && targetable(v) {
                            moves.push(vec![v]);
                        }
                    }
                }
            }
            // (b) path-jumps to a notable/keystone within maxJump (and within budget).
            // Collect (target, isKeystone) first so we can power-rank/cap them before
            // materializing paths. With max_jump_cand>0 we keep the top-K notables by
            // seed power-per-point (keystones always kept — structural value the single-
            // add seed can miss), which lets a tight cap retain FAR damage and drop junk.
            let mut jump_targets: Vec<i32> = Vec::new();
            for (&tid, &d) in &pdist {
                if !st.set.contains(&tid)
                    && d >= 2
                    && d <= max_jump.min(points_left)
                    && targetable(tid)
                    && matches!(g.ty.get(&tid), Some(&T_NOTABLE | &T_KEYSTONE))
                {
                    jump_targets.push(tid);
                }
            }
            if max_jump_cand > 0 && jump_targets.len() > max_jump_cand {
                let is_keystone = |id: i32| matches!(g.ty.get(&id), Some(&T_KEYSTONE));
                // Keep all keystones; rank the rest by seed power-per-point, keep top-K.
                let mut notables: Vec<i32> =
                    jump_targets.iter().copied().filter(|&id| !is_keystone(id)).collect();
                notables.sort_by(|a, b| seed.power(*b).partial_cmp(&seed.power(*a)).unwrap());
                notables.truncate(max_jump_cand);
                jump_targets.retain(|&id| is_keystone(id));
                jump_targets.extend(notables);
            }
            for tid in jump_targets {
                moves.push(path_to(&prev, &st.set, tid));
            }

            for add in moves {
                if add.is_empty() || st.ids.len() - 1 + add.len() > cap_points {
                    continue;
                }
                let mut new_set = st.set.clone();
                for &a in &add {
                    new_set.insert(a);
                }
                let mut new_ids: Vec<i32> = new_set.iter().copied().collect();
                new_ids.sort_unstable();
                if seen.insert(new_ids.clone()) {
                    origin.push((new_ids.clone(), new_set));
                    cand_sets.push(new_ids);
                }
            }
        }
        if cand_sets.is_empty() {
            break;
        }

        let to_score: Vec<Vec<i32>> =
            cand_sets.iter().filter(|s| !memo.contains_key(*s)).cloned().collect();
        if !to_score.is_empty() {
            let scores = score(&to_score);
            *total_evals += to_score.len();
            for (s, sc) in to_score.iter().zip(scores.iter()) {
                memo.insert(s.clone(), *sc);
            }
        }

        let mut next: Vec<State> = origin
            .into_iter()
            .map(|(ids, set)| {
                let sc = memo[&ids];
                State { ids, set, score: sc[0], dps: sc[1], ehp: sc[2] }
            })
            // A NaN score is the scorer's REJECT signal (e.g. the quadratic-shortfall
            // model rejects a candidate that regresses BOTH axes vs. the original
            // build; the calc engine also returns NaN on failure). Drop rejects so
            // they never enter the beam — and so the sort below never compares NaN
            // (partial_cmp -> None -> unwrap panic).
            .filter(|st| st.score.is_finite())
            .collect();
        // If every extension this step was rejected, keep the current beam rather
        // than panicking on an empty `next` (beam[0] is read below and downstream).
        if next.is_empty() {
            if verbose && (step % 5 == 0 || step == cap_points) {
                let b = &beam[0];
                println!(
                    "  step {step:2}: best score={:.1} (pts={}, dps={:.1}, ehp={:.1})  [{} evals]  (no valid extension)",
                    b.score, b.ids.len() - 1, b.dps, b.ehp, *total_evals
                );
            }
            continue;
        }
        next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beam = select_beam(next, beam_width, pareto_extra);

        if verbose && (step % 5 == 0 || step == cap_points) {
            let b = &beam[0];
            println!(
                "  step {step:2}: best score={:.1} (pts={}, dps={:.1}, ehp={:.1})  [{} evals]",
                b.score, b.ids.len() - 1, b.dps, b.ehp, *total_evals
            );
        }
    }
    beam.into_iter().next().unwrap()
}

/// One node's marginal contribution to the best build, plus whether it's a build
/// ENABLER (leaving it out collapses an axis to ~0 — e.g. Hollow Palm removing all
/// weapon damage). Enablers are never banned by the diet.
#[derive(Clone)]
struct Marginal {
    id: i32,
    marginal: f64,
    is_enabler: bool,
}

/// Is `id` a LEAF of the allocated subtree `set` — i.e. exactly one of its tree
/// neighbours is allocated? A leaf carries no travel for any other allocated node,
/// so dropping it strands nothing; a node with ≥2 allocated neighbours may be on
/// the main path and is structurally load-bearing. The start is never a leaf.
fn is_alloc_leaf(g: &Graph, set: &HashSet<i32>, id: i32) -> bool {
    if id == g.start {
        return false;
    }
    let allocated_neighbours = g
        .links
        .get(&id)
        .map(|adj| adj.iter().filter(|n| set.contains(n)).count())
        .unwrap_or(0);
    allocated_neighbours <= 1
}

/// For each notable/keystone in `best`, measure marginal = best.score − score(best
/// without that node), and flag enablers (leave-one-out dps or ehp < 1% of best's).
/// Scores the leave-one-out sets in ONE batch (the throughput win paying off).
fn analyze_marginals(
    g: &Graph,
    best: &State,
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    total_evals: &mut usize,
    score: &mut dyn FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
) -> Vec<Marginal> {
    const COLLAPSE: f64 = 0.01;
    let nodes: Vec<i32> = best
        .ids
        .iter()
        .copied()
        .filter(|&id| matches!(g.ty.get(&id), Some(&T_NOTABLE | &T_KEYSTONE)))
        .collect();
    let loo_sets: Vec<Vec<i32>> = nodes
        .iter()
        .map(|&drop| {
            let mut v: Vec<i32> = best.ids.iter().copied().filter(|&x| x != drop).collect();
            v.sort_unstable();
            v
        })
        .collect();
    let to_score: Vec<Vec<i32>> = loo_sets.iter().filter(|s| !memo.contains_key(*s)).cloned().collect();
    if !to_score.is_empty() {
        let scores = score(&to_score);
        *total_evals += to_score.len();
        for (s, sc) in to_score.iter().zip(scores.iter()) {
            memo.insert(s.clone(), *sc);
        }
    }
    let mut marg = Vec::with_capacity(nodes.len());
    for (i, &id) in nodes.iter().enumerate() {
        let loo = memo[&loo_sets[i]];
        // Marginal = how much penalized score this node contributes (best vs. the
        // tree without it). `best.score` is ALREADY penalized (optimize re-scores
        // it); penalize the leave-one-out raw growth score the same way so both
        // sides are on the same scale. The diet bans the lowest-marginal node, so a
        // node that only adds EHP at the cost of the DPS floor scores LOW here and
        // gets dieted out first — exactly the DPS protection we want.
        let loo_pen = g.penalized_score(loo[0], loo[1], loo[2]);
        let marginal = if loo_pen.is_finite() {
            best.score - loo_pen
        } else {
            f64::INFINITY
        };
        let is_enabler = !loo_pen.is_finite()
            || (best.dps > 0.0 && loo[1] < COLLAPSE * best.dps)
            || (best.ehp > 0.0 && loo[2] < COLLAPSE * best.ehp);
        marg.push(Marginal { id, marginal, is_enabler });
    }
    marg
}

/// Pareto-augmented beam selection: top-`width` by score, then up to `extra`
/// Pareto-non-dominated states (high on one axis even if lower-scoring) so the
/// damage-seeking branch isn't culled by a pure scalar top-N. `sorted` is
/// score-descending. Dedups by id-set.
fn select_beam(sorted: Vec<State>, width: usize, extra: usize) -> Vec<State> {
    let mut out: Vec<State> = Vec::new();
    let mut seen: HashSet<Vec<i32>> = HashSet::new();
    for st in &sorted {
        if out.len() >= width {
            break;
        }
        if seen.insert(st.ids.clone()) {
            out.push(st.clone());
        }
    }
    let mut added = 0;
    for st in &sorted {
        if added >= extra {
            break;
        }
        if seen.contains(&st.ids) {
            continue;
        }
        let dominated = out.iter().any(|k| {
            k.dps >= st.dps && k.ehp >= st.ehp && (k.dps > st.dps || k.ehp > st.ehp)
        });
        if !dominated {
            seen.insert(st.ids.clone());
            out.push(st.clone());
            added += 1;
        }
    }
    out
}

/// BFS shortest paths from a candidate id-set over the graph, honoring an optional
/// `excluded` soft-ban (detour-weighted). Returns, for every reachable node, its
/// point-distance (count of NEW nodes to reach it) and a `prev` map to reconstruct
/// the path. Heap-based Dijkstra with detour weight (mirrors the spike's
/// pathsFromSet). This is called once per beam STATE per STEP, thousands of times
/// per `run_beam`, so the frontier selection must not be the old O(V²) linear scan
/// over `wdist` — a binary heap with lazy deletion keeps it O(E log V).
fn paths_from_set(
    g: &Graph,
    id_set: &HashSet<i32>,
    excluded: &HashSet<i32>,
    detour: i64,
) -> (HashMap<i32, usize>, HashMap<i32, i32>) {
    use std::cmp::Reverse;
    use std::collections::BinaryHeap;

    let mut wdist: HashMap<i32, i64> = HashMap::new();
    let mut pdist: HashMap<i32, usize> = HashMap::new();
    let mut prev: HashMap<i32, i32> = HashMap::new();
    let mut visited: HashSet<i32> = HashSet::new();
    // Min-heap keyed by weighted distance. Entries are (Reverse(weight), id); we
    // never decrease-key, just push a fresh entry on relaxation and skip stale pops
    // (the `visited` guard + the `w > wdist[&u]` check below). The start frontier
    // all sits at weight 0.
    let mut heap: BinaryHeap<Reverse<(i64, i32)>> = BinaryHeap::new();
    for &id in id_set {
        wdist.insert(id, 0);
        pdist.insert(id, 0);
        heap.push(Reverse((0, id)));
    }
    while let Some(Reverse((w, u))) = heap.pop() {
        // Stale entry (a better path to u was found after this was queued), or u is
        // already finalized: skip. This is the lazy-deletion replacement for the old
        // linear min-scan; the first time we pop a node it carries its final weight.
        if !visited.insert(u) {
            continue;
        }
        if wdist.get(&u).is_some_and(|&best| w > best) {
            continue;
        }
        if let Some(adj) = g.links.get(&u) {
            for &v in adj {
                if !visited.contains(&v) && (id_set.contains(&v) || g.is_target(v) || v == g.start) {
                    let nw = w + if excluded.contains(&v) { detour } else { 1 };
                    if wdist.get(&v).is_none_or(|&old| nw < old) {
                        wdist.insert(v, nw);
                        let pc = pdist[&u] + if id_set.contains(&v) { 0 } else { 1 };
                        pdist.insert(v, pc);
                        prev.insert(v, u);
                        heap.push(Reverse((nw, v)));
                    }
                }
            }
        }
    }
    (pdist, prev)
}

/// Reconstruct the list of NEW node ids on the path to `target` (target first),
/// stopping at the first node already in `id_set`.
fn path_to(prev: &HashMap<i32, i32>, id_set: &HashSet<i32>, target: i32) -> Vec<i32> {
    let mut out = Vec::new();
    let mut cur = target;
    while !id_set.contains(&cur) {
        out.push(cur);
        match prev.get(&cur) {
            Some(&p) => cur = p,
            None => break,
        }
    }
    out
}

/// Score one id-set, caching by the (sorted) set. Helper for the start-only base.
fn score_or_memo(
    ids: &[i32],
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    total_evals: &mut usize,
    score: &mut dyn FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
) -> [f64; 3] {
    let key = ids.to_vec();
    if let Some(&m) = memo.get(&key) {
        return m;
    }
    let s = score(&[key.clone()])[0];
    *total_evals += 1;
    memo.insert(key, s);
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A tiny synthetic line graph: start(0) - A(1) - B(2) - C(3), all notables,
    /// with a fake score that rewards reaching the far end. No calc engine — this
    /// only exercises the search's plumbing (growth, budget, selection, result).
    fn line_graph() -> Graph {
        let mut links = HashMap::new();
        let mut ty = HashMap::new();
        for id in 0..=3 {
            ty.insert(id, T_NOTABLE);
        }
        links.insert(0, vec![1]);
        links.insert(1, vec![0, 2]);
        links.insert(2, vec![1, 3]);
        links.insert(3, vec![2]);
        // penalty_k=0 => penalized_score is identity, so the node-count fake score
        // below is compared directly (no regression floor in this plumbing test).
        Graph {
            start: 0,
            budget: 3,
            links,
            ty,
            ref_dps: 1.0,
            ref_ehp: 1.0,
            w_dps: 1.0,
            w_ehp: 1.0,
            penalty_k: 0.0,
        }
    }

    #[test]
    fn alloc_leaf_protects_the_main_path() {
        // Line 0-1-2-3, all allocated. Only the two ends are leaves; the interior
        // nodes carry travel for the far end and must NOT read as leaves (dropping
        // one would strand everything past it — "corrupt the main path").
        let g = line_graph();
        let set: HashSet<i32> = [0, 1, 2, 3].into_iter().collect();
        assert!(!is_alloc_leaf(&g, &set, 0), "start is never a leaf");
        assert!(!is_alloc_leaf(&g, &set, 1), "interior node is load-bearing");
        assert!(!is_alloc_leaf(&g, &set, 2), "interior node is load-bearing");
        assert!(is_alloc_leaf(&g, &set, 3), "the far end IS a droppable leaf");
        // Drop the far end: now node 2 becomes the new leaf.
        let set2: HashSet<i32> = [0, 1, 2].into_iter().collect();
        assert!(is_alloc_leaf(&g, &set2, 2));
        assert!(!is_alloc_leaf(&g, &set2, 1));
    }

    #[test]
    fn params_parse_defaults_and_overrides() {
        let d = BeamParams::parse("").unwrap();
        assert_eq!(d.beam_width, 8);
        assert_eq!(d.max_jump, 12);
        assert_eq!(d.restarts, 2);
        let o = BeamParams::parse("beam_width=4,max_jump=6\nverbose=1\nrestarts=0").unwrap();
        assert_eq!(o.beam_width, 4);
        assert_eq!(o.max_jump, 6);
        assert!(o.verbose);
        assert_eq!(o.restarts, 0);
        assert!(BeamParams::parse("nope=1").is_err());
    }

    #[test]
    fn beam_grows_a_connected_tree_within_budget() {
        let g = line_graph();
        // Score = number of allocated nodes (so the optimum spends the whole budget),
        // with a small dps/ehp so nothing is flagged an enabler.
        let score = |cands: &[Vec<i32>]| -> Vec<[f64; 3]> {
            cands
                .iter()
                .map(|c| {
                    let n = c.len() as f64;
                    [n, n, n]
                })
                .collect()
        };
        let params = BeamParams { cap_points: 3, beam_width: 4, max_jump: 4, ..Default::default() };
        let res = optimize(&g, &params, score);
        // start + 3 allocated = 4 ids; connected line 0-1-2-3.
        assert_eq!(res.ids, vec![0, 1, 2, 3]);
        assert_eq!(res.score, 4.0);
        assert!(res.evals > 0);
    }

    #[test]
    fn penalized_score_blocks_axis_collapse() {
        // ref = original build's dps/ehp; w=1/1; K=10. The growth arg is ignored
        // when penalty_k>0, so pass 0.0.
        let g = Graph {
            start: 0, budget: 0, links: HashMap::new(), ty: HashMap::new(),
            ref_dps: 100.0, ref_ehp: 1000.0, w_dps: 1.0, w_ehp: 1.0, penalty_k: 10.0,
        };
        // Pure improvement on both axes: positive.
        assert!(g.penalized_score(0.0, 110.0, 1080.0) > 0.0);
        // The motivating bad trade: -13% dps for +3.5% ehp. Quadratic penalty on
        // the dps drop dominates the small ehp gain => negative (rejected by diet).
        assert!(g.penalized_score(0.0, 87.0, 1035.0) < 0.0);
        // The hole in a LINEAR penalty: -50% dps but +250% ehp. A linear sum would
        // net positive; the quadratic dps penalty (10*0.5^2=2.5) cancels the +2.5
        // ehp gain => <= 0. No finite ehp blow-up rescues a collapsed dps axis.
        assert!(g.penalized_score(0.0, 50.0, 3500.0) <= 0.0);
        // penalty_k=0 => identity (old behavior): returns the growth arg verbatim.
        let g0 = Graph { penalty_k: 0.0, ..g };
        assert_eq!(g0.penalized_score(42.0, 1.0, 1.0), 42.0);
    }

    /// A "Y" graph: start(0) branches to a NEAR junk notable (1, low score) and,
    /// through a travel chain (2->3), to a FAR high-value notable (4). With a 1-wide
    /// beam and a 1-jump-candidate cap, a blind (distance-first) selection would keep
    /// the near junk and never reach the far prize; the power seed ranks node 4 highest
    /// (real gain per point), so the cap keeps the FAR damage. budget=3 reaches it.
    fn y_graph() -> Graph {
        let mut links = HashMap::new();
        let mut ty = HashMap::new();
        // 0 start; 1 near junk notable; 2,3 travel-ish notables on the way to 4; 4 prize.
        for id in 0..=4 {
            ty.insert(id, T_NOTABLE);
        }
        links.insert(0, vec![1, 2]);
        links.insert(1, vec![0]);
        links.insert(2, vec![0, 3]);
        links.insert(3, vec![2, 4]);
        links.insert(4, vec![3]);
        Graph {
            start: 0, budget: 3, links, ty,
            ref_dps: 1.0, ref_ehp: 1.0, w_dps: 1.0, w_ehp: 1.0, penalty_k: 0.0,
        }
    }

    #[test]
    fn seed_cap_keeps_far_high_power_target() {
        let g = y_graph();
        // Score: node 4 is worth a lot; everything else is worth little. (penalty_k=0
        // so growth score is returned verbatim; dps/ehp mirror the score so nothing is
        // flagged an enabler.) The set's value = max single-node value present.
        let score = |cands: &[Vec<i32>]| -> Vec<[f64; 3]> {
            cands
                .iter()
                .map(|c| {
                    let v = if c.contains(&4) { 100.0 } else { c.len() as f64 };
                    [v, v, v]
                })
                .collect()
        };
        // Narrowest possible beam + tightest jump cap: only the power seed steers it.
        let params = BeamParams {
            cap_points: 3, beam_width: 1, max_jump: 4, pareto_extra: 0,
            max_jump_cand: 1, seed_anchors: 2, ..Default::default()
        };
        let res = optimize(&g, &params, score);
        assert!(res.ids.contains(&4), "seed-capped beam failed to reach the far prize: {:?}", res.ids);
        assert_eq!(res.score, 100.0);
    }

    #[test]
    fn restarts_never_worsen_the_result() {
        // The restart loop keeps the BEST converged tree across all anchor bands, so
        // adding restarts must never produce a worse final score than restarts=0 on
        // the same graph/scorer. (It can only swap in a strictly-better basin.) This
        // is the core correctness guarantee of the multi-seed restart.
        let g = y_graph();
        let score = |cands: &[Vec<i32>]| -> Vec<[f64; 3]> {
            cands
                .iter()
                .map(|c| {
                    let v = if c.contains(&4) { 100.0 } else { c.len() as f64 };
                    [v, v, v]
                })
                .collect()
        };
        let base = BeamParams {
            cap_points: 3, beam_width: 1, max_jump: 4, pareto_extra: 0,
            max_jump_cand: 1, seed_anchors: 1, ..Default::default()
        };
        let no_restart = optimize(&g, &BeamParams { restarts: 0, ..base }, score);
        let with_restart = optimize(&g, &BeamParams { restarts: 3, ..base }, score);
        assert!(
            with_restart.score >= no_restart.score - 1e-9,
            "restarts regressed the result: {} < {}",
            with_restart.score, no_restart.score
        );
    }
}
