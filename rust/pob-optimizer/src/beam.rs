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
            patience_max: 2,
            max_rounds: 12,
            verbose: false,
        }
    }
}

impl BeamParams {
    /// Parse a newline- or comma-delimited `key=value` override string. Unknown
    /// keys error (catch typos early); absent keys keep the default. Keys:
    /// cap_points, beam_width, max_jump, pareto_extra, detour, patience_max,
    /// max_rounds, verbose (0/1). Empty string => all defaults.
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

    // Round 0: unconstrained.
    log!("ROUND 0 (unconstrained)");
    let mut best = run_beam(
        graph, cap_points, params.beam_width, params.max_jump, pareto_extra,
        &HashSet::new(), params.detour, &mut memo, &mut total_evals, params.verbose, &mut score,
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

    // Elimination diet: ban the lowest-marginal droppable node, re-optimize, keep
    // the ban iff it improved; else revert + tick patience. Enablers (removal
    // collapses an axis) are never banned. Soft-ban: pathing may still detour
    // through a banned node, it just can't be targeted.
    while round < params.max_rounds && patience < params.patience_max {
        round += 1;
        let marg = analyze_marginals(graph, &best, &mut memo, &mut total_evals, &mut score);
        let victim = marg
            .iter()
            .filter(|m| !m.is_enabler && !banned.contains(&m.id) && !tried.contains(&m.id))
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
            &trial, params.detour, &mut memo, &mut total_evals, params.verbose, &mut score,
        );
        repenalize(&mut r);
        if r.score > best.score + 1e-6 {
            log!(
                "    IMPROVED {:.1} -> {:.1} dps={:.1} ehp={:.1}. Keeping ban.",
                best.score, r.score, r.dps, r.ehp
            );
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
    excluded: &HashSet<i32>,
    detour: i64,
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    total_evals: &mut usize,
    verbose: bool,
    score: &mut impl FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
) -> State {
    let start_set: HashSet<i32> = HashSet::from([g.start]);
    let base = score_or_memo(&[g.start], memo, total_evals, score);
    let mut beam = vec![State {
        ids: vec![g.start],
        set: start_set,
        score: base[0],
        dps: base[1],
        ehp: base[2],
    }];

    let targetable = |v: i32| g.is_target(v) && !excluded.contains(&v);

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
            // (b) path-jumps to a notable/keystone within maxJump (and within budget)
            for (&tid, &d) in &pdist {
                if !st.set.contains(&tid)
                    && d >= 2
                    && d <= max_jump.min(points_left)
                    && targetable(tid)
                    && matches!(g.ty.get(&tid), Some(&T_NOTABLE | &T_KEYSTONE))
                {
                    moves.push(path_to(&prev, &st.set, tid));
                }
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

/// For each notable/keystone in `best`, measure marginal = best.score − score(best
/// without that node), and flag enablers (leave-one-out dps or ehp < 1% of best's).
/// Scores the leave-one-out sets in ONE batch (the throughput win paying off).
fn analyze_marginals(
    g: &Graph,
    best: &State,
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    total_evals: &mut usize,
    score: &mut impl FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
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
/// the path. Dijkstra with detour weight (mirrors the spike's pathsFromSet).
fn paths_from_set(
    g: &Graph,
    id_set: &HashSet<i32>,
    excluded: &HashSet<i32>,
    detour: i64,
) -> (HashMap<i32, usize>, HashMap<i32, i32>) {
    let mut wdist: HashMap<i32, i64> = HashMap::new();
    let mut pdist: HashMap<i32, usize> = HashMap::new();
    let mut prev: HashMap<i32, i32> = HashMap::new();
    let mut visited: HashSet<i32> = HashSet::new();
    for &id in id_set {
        wdist.insert(id, 0);
        pdist.insert(id, 0);
    }
    loop {
        let mut u = None;
        let mut best = i64::MAX;
        for (&id, &w) in &wdist {
            if !visited.contains(&id) && w < best {
                u = Some(id);
                best = w;
            }
        }
        let u = match u {
            Some(u) => u,
            None => break,
        };
        visited.insert(u);
        if let Some(adj) = g.links.get(&u) {
            for &v in adj {
                if !visited.contains(&v) && (id_set.contains(&v) || g.is_target(v) || v == g.start) {
                    let w = wdist[&u] + if excluded.contains(&v) { detour } else { 1 };
                    if wdist.get(&v).is_none_or(|&old| w < old) {
                        wdist.insert(v, w);
                        let pc = pdist[&u] + if id_set.contains(&v) { 0 } else { 1 };
                        pdist.insert(v, pc);
                        prev.insert(v, u);
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
    score: &mut impl FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
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
    fn params_parse_defaults_and_overrides() {
        let d = BeamParams::parse("").unwrap();
        assert_eq!(d.beam_width, 8);
        assert_eq!(d.max_jump, 12);
        let o = BeamParams::parse("beam_width=4,max_jump=6\nverbose=1").unwrap();
        assert_eq!(o.beam_width, 4);
        assert_eq!(o.max_jump, 6);
        assert!(o.verbose);
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
}
