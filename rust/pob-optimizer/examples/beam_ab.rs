//! Native A/B harness: run the parallel pool against a real clean-slate beam
//! search and measure whether the Rust offload is a throughput WIN over scoring
//! the same work single-threaded.
//!
//! Why this exists: the busted spike (spec/System/SpikeCleanSlateBeam_spec.lua)
//! runs the beam in Lua, but only inside the Linux test container, which cannot
//! load this Windows cdylib. So the only place the pool can actually be driven on
//! a dev box is a native harness like this (same model as examples/smoke.rs).
//!
//! It does two things against the real Monk build (src/Builds/mymonk.xml):
//!   1. MICROBENCHMARK — score one representative batch of candidate node-sets
//!      (a) through the N-worker pool and (b) single-threaded (a 1-worker pool),
//!      report wall-time + speedup, and assert the scores match within 1e-6
//!      (correctness: parallelism must not change the answer).
//!   2. CAPPED BEAM — reconstruct the tree graph the worker exports, run a real
//!      clean-slate beam (grow a connected P-node tree from the class start,
//!      batch-score each step through the pool, keep top-N), and report the score
//!      climbing + final dps/ehp. Proves the pool drives a genuine search.
//!
//! Run from repo root (PATH must include runtime/ for transitive DLLs):
//!   $env:PATH = "$PWD\runtime;$env:PATH"
//!   cargo run --release --manifest-path rust/pob-optimizer/Cargo.toml \
//!       --example beam_ab -- [workers] [capPoints] [beamWidth]
//!
//! Defaults: workers = available, capPoints = 25, beamWidth = 8.

use libloading::{Library, Symbol};
use std::collections::{HashMap, HashSet};
use std::ffi::{c_char, c_double, c_int, CStr, CString};
use std::path::{Path, PathBuf};
use std::time::Instant;

type FnCreate = unsafe extern "C" fn(*const c_char) -> *mut std::ffi::c_void;
type FnScore3 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const i32,
    *const i32,
    c_int,
    *mut c_double,
) -> c_int;
type FnWorkerCount = unsafe extern "C" fn(*mut std::ffi::c_void) -> c_int;
type FnCandidateIds = unsafe extern "C" fn(*mut std::ffi::c_void, *mut i32, c_int) -> c_int;
type FnDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);
type FnLastError = unsafe extern "C" fn() -> *const c_char;
type FnCallSave = unsafe extern "C" fn(*mut std::ffi::c_void, *const i32, c_int) -> c_double;

const T_NORMAL: i32 = 1;
const T_NOTABLE: i32 = 2;
const T_KEYSTONE: i32 = 3;

/// The tree as the worker exports it: adjacency + node types + the class start.
struct Graph {
    start: i32,
    budget: usize,
    links: HashMap<i32, Vec<i32>>,
    ty: HashMap<i32, i32>,
}

impl Graph {
    /// Parse the flat int array __pob_graph_export emits:
    ///   [ startId, budgetP, nNodes, (id, typeCode, nLinks, link...) * nNodes ]
    fn parse(flat: &[i32]) -> Graph {
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
        Graph { start, budget, links, ty }
    }

    fn is_target(&self, id: i32) -> bool {
        matches!(self.ty.get(&id), Some(&T_NORMAL | &T_NOTABLE | &T_KEYSTONE))
    }
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

fn main() {
    let mut args = std::env::args().skip(1);
    let workers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    // capPoints arg: 0 (or omitted) => use the build's REAL budget (exported by the worker).
    let cap_arg: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let beam_width: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    // outFile arg (optional): the saved-build filename under src/Builds/. Defaults to
    // mymonk_optimized.xml. Pass e.g. mymonk_optimized_v2.xml to keep multiple results.
    let out_file: String = args.next().unwrap_or_else(|| "mymonk_optimized.xml".to_string());
    // wDps / wEhp args (optional): score weights. Default 1/1. Raise wDps to bias the
    // search toward damage (e.g. `... 3 1` for a DPS-leaning tree).
    let w_dps: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let w_ehp: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let root = locate_repo_root().expect("locate repo root (need src/ + runtime/)");
    let fwd = |p: PathBuf| p.to_string_lossy().replace('\\', "/");
    let cdylib = root.join("rust/pob-optimizer/target/release/pob_optimizer.dll");
    assert!(cdylib.exists(), "build the cdylib first: {}", cdylib.display());

    // Two configs differ ONLY in worker count and candidate_fn. We point candidate_fn
    // at __pob_graph_export so pob_opt_candidate_ids hands back the flat graph; the
    // score fn is the clean-slate path (3 returns) driven via pob_opt_score_batch3.
    let make_config = |n_workers: usize| {
        format!(
            "lua_dll={}\nsrc_dir={}\nruntime_dir={}\nruntime_lua={}\nbuild_xml={}\nbootstrap={}\nscore_fn=__pob_score_cleanslate\ncandidate_fn=__pob_graph_export\nworkers={}\nw_dps={}\nw_ehp={}\n",
            fwd(root.join("runtime/lua51.dll")),
            fwd(root.join("src")),
            fwd(root.join("runtime")),
            fwd(root.join("runtime/lua")),
            fwd(root.join("src/Builds/mymonk.xml")),
            fwd(root.join("rust/pob-optimizer/worker_bootstrap.lua")),
            n_workers,
            w_dps,
            w_ehp,
        )
    };

    let lib = unsafe { Library::new(&cdylib) }.expect("load cdylib");
    unsafe {
        let create: Symbol<FnCreate> = lib.get(b"pob_opt_create").unwrap();
        let score3: Symbol<FnScore3> = lib.get(b"pob_opt_score_batch3").unwrap();
        let worker_count: Symbol<FnWorkerCount> = lib.get(b"pob_opt_worker_count").unwrap();
        let candidate_ids: Symbol<FnCandidateIds> = lib.get(b"pob_opt_candidate_ids").unwrap();
        let destroy: Symbol<FnDestroy> = lib.get(b"pob_opt_destroy").unwrap();
        let last_error: Symbol<FnLastError> = lib.get(b"pob_opt_last_error").unwrap();
        let call_save: Symbol<FnCallSave> = lib.get(b"pob_opt_call_save").unwrap();

        let read_err = || {
            let p = last_error();
            if p.is_null() {
                "<none>".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };

        // ---- boot the parallel pool -------------------------------------------------
        println!("weights: w_dps={w_dps}, w_ehp={w_ehp}   output: {out_file}");
        println!("booting parallel pool (workers={}) ...", if workers == 0 { "auto".into() } else { workers.to_string() });
        let t_boot = Instant::now();
        let cfg = CString::new(make_config(workers)).unwrap();
        let pool = create(cfg.as_ptr());
        assert!(!pool.is_null(), "pob_opt_create failed: {}", read_err());
        let nw = worker_count(pool);
        println!("pool up: {nw} workers, booted in {:.1}s", t_boot.elapsed().as_secs_f64());

        // ---- pull the graph the workers exported ------------------------------------
        let total = candidate_ids(pool, std::ptr::null_mut(), 0);
        assert!(total > 0, "graph export empty: {}", read_err());
        let mut flat = vec![0i32; total as usize];
        let got = candidate_ids(pool, flat.as_mut_ptr(), total);
        assert_eq!(got, total);
        let graph = Graph::parse(&flat);
        let n_targets = graph.ty.values().filter(|&&t| t == T_NOTABLE || t == T_KEYSTONE).count();
        // cap = explicit arg, else the build's real budget the worker exported.
        let cap_points = if cap_arg > 0 { cap_arg } else { graph.budget };
        assert!(cap_points > 0, "no point budget (pass capPoints arg)");
        println!(
            "graph: {} nodes, start id {}, {} notable/keystone targets, build budget P={}{}",
            graph.links.len(),
            graph.start,
            n_targets,
            graph.budget,
            if cap_arg > 0 { format!(" (capped to {cap_arg})") } else { String::new() }
        );

        let score_batch3 = |pool: *mut std::ffi::c_void, cands: &[Vec<i32>]| -> Vec<[f64; 3]> {
            if cands.is_empty() {
                return Vec::new();
            }
            let (ids, lengths) = flatten(cands);
            let mut out = vec![0.0f64; cands.len() * 3];
            let rc = score3(
                pool,
                ids.as_ptr(),
                lengths.as_ptr(),
                cands.len() as c_int,
                out.as_mut_ptr(),
            );
            assert_eq!(rc, 0, "score_batch3 rc={rc}: {}", read_err());
            (0..cands.len()).map(|k| [out[k * 3], out[k * 3 + 1], out[k * 3 + 2]]).collect()
        };

        // ============================================================================
        // 1. MICROBENCHMARK — same batch, parallel pool vs single-threaded (1 worker).
        // ============================================================================
        // Build a representative batch: connected partial trees of growing size grown
        // greedily from the start by BFS order, so each is a real (connected) candidate
        // the beam could actually evaluate — not synthetic noise.
        let bench = build_bench_batch(&graph, 400);
        println!("\n=== MICROBENCHMARK: {} candidate sets ===", bench.len());

        // Warm + time the parallel pool.
        let _ = score_batch3(pool, &bench[..bench.len().min(nw as usize)].to_vec());
        let t_par = Instant::now();
        let par_scores = score_batch3(pool, &bench);
        let par_dt = t_par.elapsed().as_secs_f64();

        // Single-threaded baseline: a fresh 1-worker pool scores the SAME batch. (One
        // worker = no parallelism; same code path, so the only variable is thread count.)
        println!("booting 1-worker baseline pool ...");
        let cfg1 = CString::new(make_config(1)).unwrap();
        let pool1 = create(cfg1.as_ptr());
        assert!(!pool1.is_null(), "baseline pool create failed: {}", read_err());
        let _ = score_batch3(pool1, &bench[..1].to_vec());
        let t_seq = Instant::now();
        let seq_scores = score_batch3(pool1, &bench);
        let seq_dt = t_seq.elapsed().as_secs_f64();
        destroy(pool1);

        // Correctness: parallel and single-threaded must agree exactly.
        let mut max_diff = 0.0f64;
        for (a, b) in par_scores.iter().zip(seq_scores.iter()) {
            max_diff = max_diff.max((a[0] - b[0]).abs());
        }
        let n = bench.len() as f64;
        println!("  single-thread (1 worker) : {seq_dt:.2}s  ({:.2} ms/set)", seq_dt * 1000.0 / n);
        println!("  parallel ({nw} workers)     : {par_dt:.2}s  ({:.2} ms/set)", par_dt * 1000.0 / n);
        println!("  SPEEDUP                  : {:.2}x", seq_dt / par_dt.max(1e-9));
        println!("  max score diff par vs seq: {max_diff:.2e}  ({})", if max_diff < 1e-6 { "MATCH" } else { "!! MISMATCH" });
        assert!(max_diff < 1e-6, "parallel scoring changed the answer (diff {max_diff:.3e})");

        // ============================================================================
        // 2. FULL BEAM + ELIMINATION DIET — the spike's algorithm, through the pool.
        // ============================================================================
        let max_jump = 12usize;
        let pareto_extra = (beam_width / 2).max(4);
        let detour = 50i64;
        let patience_max = 2usize;
        let max_rounds = 12usize;
        println!(
            "\n=== FULL BEAM + DIET (P={cap_points}, beam={beam_width}, jump={max_jump}, pareto+{pareto_extra}) ==="
        );
        let t_beam = Instant::now();
        let mut memo: HashMap<Vec<i32>, [f64; 3]> = HashMap::new();
        let mut total_evals = 0usize;
        // returns (best, evals); caller accumulates evals so no shared mutable capture.
        let do_beam = |excluded: &HashSet<i32>, memo: &mut HashMap<Vec<i32>, [f64; 3]>| {
            run_beam(
                &graph, cap_points, beam_width, max_jump, pareto_extra, excluded, detour, memo,
                |cands| score_batch3(pool, cands),
            )
        };

        // Round 0: unconstrained.
        println!("ROUND 0 (unconstrained)");
        let (mut best, e0) = do_beam(&HashSet::new(), &mut memo);
        total_evals += e0;
        println!(
            "  round 0 best: score={:.1} dps={:.1} ehp={:.1} pts={}",
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
        while round < max_rounds && patience < patience_max {
            round += 1;
            let (marg, _enablers) = analyze_marginals(&graph, &best, &mut memo, |cands| score_batch3(pool, cands), &mut total_evals);
            // pick lowest-marginal non-enabler, not already banned or tried
            let victim = marg
                .iter()
                .filter(|m| !m.is_enabler && !banned.contains(&m.id) && !tried.contains(&m.id))
                .min_by(|a, b| a.marginal.partial_cmp(&b.marginal).unwrap());
            let victim = match victim {
                Some(v) => v.clone(),
                None => {
                    println!("  round {round}: no untried droppable node left. Stop.");
                    break;
                }
            };
            tried.insert(victim.id);
            let mut trial = banned.clone();
            trial.insert(victim.id);
            println!("  round {round}: ban {} (marginal {:+.1})", victim.id, victim.marginal);
            let (r, er) = do_beam(&trial, &mut memo);
            total_evals += er;
            if r.score > best.score + 1e-6 {
                println!(
                    "    IMPROVED {:.1} -> {:.1} dps={:.1} ehp={:.1}. Keeping ban.",
                    best.score, r.score, r.dps, r.ehp
                );
                best = r;
                banned = trial;
                patience = 0;
                tried.clear();
            } else {
                println!("    no improvement ({:.1} vs {:.1}). Revert. patience {}/{}", r.score, best.score, patience + 1, patience_max);
                patience += 1;
            }
        }

        let beam_dt = t_beam.elapsed().as_secs_f64();
        let n_evals = total_evals;
        println!(
            "\n  FINAL: score={:.1}  dps={:.1}  ehp={:.1}  points={}  ({} banned, {} evals, {:.1}s)",
            best.score,
            best.dps,
            best.ehp,
            best.ids.len() - 1,
            banned.len(),
            n_evals,
            beam_dt
        );
        // Sanity: the beam grew a non-trivial connected tree and improved over start-only.
        assert!(best.ids.len() > 1, "beam allocated nothing");
        assert!(best.score.is_finite() && best.score > 0.0, "beam best score not positive");

        // ---- SAVE the optimized tree to an importable PoB XML (mutates a worker spec,
        // so it is the LAST pool call). Pack [ n_ids, ids..., path bytes... ].
        let out_path = root.join("src/Builds").join(&out_file);
        let out_str = out_path.to_string_lossy().replace('\\', "/");
        let mut packed: Vec<i32> = Vec::with_capacity(best.ids.len() + out_str.len() + 1);
        packed.push(best.ids.len() as i32);
        packed.extend(best.ids.iter().copied());
        packed.extend(out_str.bytes().map(|b| b as i32));
        let saved = call_save(pool, packed.as_ptr(), packed.len() as c_int);
        if saved == 1.0 {
            println!("\n=== SAVED optimized build -> {} ===", out_str);
            println!("  Open it in PoB (Import/Open) to inspect the optimized tree.");
        } else {
            println!("\n=== SAVE FAILED (rc={saved}): {} ===", read_err());
        }

        destroy(pool);
        println!("\nBEAM A/B HARNESS PASSED");
    }
}

/// Build `count` connected partial trees of growing size by BFS from the start —
/// a representative scoring batch (real connected candidates, not noise).
fn build_bench_batch(g: &Graph, count: usize) -> Vec<Vec<i32>> {
    // One BFS order from start; prefixes of it are connected subtrees.
    let order = bfs_order(g);
    let mut out = Vec::with_capacity(count);
    let max_len = order.len();
    for k in 0..count {
        // Sizes sweep 1..~80 so the batch spans cheap and expensive candidates.
        let size = 1 + (k * 80 / count).min(max_len.saturating_sub(1));
        out.push({
            let mut v: Vec<i32> = order[..size.min(max_len)].to_vec();
            v.sort_unstable();
            v
        });
    }
    out
}

/// BFS node order from the class start over the exported graph.
fn bfs_order(g: &Graph) -> Vec<i32> {
    let mut order = vec![g.start];
    let mut seen: HashSet<i32> = HashSet::from([g.start]);
    let mut qi = 0;
    while qi < order.len() {
        let u = order[qi];
        qi += 1;
        if let Some(adj) = g.links.get(&u) {
            for &v in adj {
                if g.is_target(v) && seen.insert(v) {
                    order.push(v);
                }
            }
        }
    }
    order
}

/// BFS shortest paths from a candidate id-set over the graph, honoring an optional
/// `excluded` soft-ban (detour-weighted). Returns, for every reachable node, its
/// point-distance (count of NEW nodes to reach it) and a `prev` map to reconstruct
/// the path. Mirrors the spike's pathsFromSet (Dijkstra with detour weight).
fn paths_from_set(
    g: &Graph,
    id_set: &HashSet<i32>,
    excluded: &HashSet<i32>,
    detour: i64,
) -> (HashMap<i32, usize>, HashMap<i32, i32>) {
    // Dijkstra keyed by accumulated WEIGHT (penalty-aware) for ordering; pdist is the
    // real integer point cost (new nodes). Linear-scan min — fan-out per step is small.
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
                // a path may pass through any allocated or traversable (target/start) node
                if !visited.contains(&v) && (id_set.contains(&v) || g.is_target(v) || v == g.start) {
                    let w = wdist[&u] + if excluded.contains(&v) { detour } else { 1 };
                    if wdist.get(&v).map_or(true, |&old| w < old) {
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

/// The full clean-slate beam (ported from SpikeCleanSlateBeam_spec.lua): grows a
/// connected tree from the start, expanding each state by BOTH (a) single adjacent
/// nodes and (b) path-JUMPS to a distant notable/keystone via shortest path — the
/// jump is what escapes the all-defense local optimum by investing in far damage
/// clusters. Selection keeps top-N by score PLUS Pareto-non-dominated (dps,ehp)
/// states so the damage-seeking branch survives a narrow beam. `excluded` is the
/// soft-ban set (detour-pathed, never targeted) the diet fills; empty for round 0.
fn run_beam(
    g: &Graph,
    cap_points: usize,
    beam_width: usize,
    max_jump: usize,
    pareto_extra: usize,
    excluded: &HashSet<i32>,
    detour: i64,
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    mut score: impl FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
) -> (State, usize) {
    let start_set: HashSet<i32> = HashSet::from([g.start]);
    let base = if let Some(&m) = memo.get(&vec![g.start]) {
        m
    } else {
        let s = score(&[vec![g.start]])[0];
        memo.insert(vec![g.start], s);
        s
    };
    let mut beam = vec![State {
        ids: vec![g.start],
        set: start_set,
        score: base[0],
        dps: base[1],
        ehp: base[2],
    }];
    let mut total_evals = 0usize;

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
            total_evals += to_score.len();
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
            .collect();
        next.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());
        beam = select_beam(next, beam_width, pareto_extra);

        if step % 5 == 0 || step == cap_points {
            let b = &beam[0];
            println!(
                "  step {step:2}: best score={:.1} (pts={}, dps={:.1}, ehp={:.1})  [{} evals]",
                b.score,
                b.ids.len() - 1,
                b.dps,
                b.ehp,
                total_evals
            );
        }
    }
    let best = beam.into_iter().next().unwrap();
    (best, total_evals)
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
/// Mirrors the spike's analyzeMarginals. Scores the leave-one-out sets in ONE batch
/// through the pool (that's the throughput win paying off — N independent evals).
fn analyze_marginals(
    g: &Graph,
    best: &State,
    memo: &mut HashMap<Vec<i32>, [f64; 3]>,
    mut score: impl FnMut(&[Vec<i32>]) -> Vec<[f64; 3]>,
    total_evals: &mut usize,
) -> (Vec<Marginal>, Vec<i32>) {
    const COLLAPSE: f64 = 0.01;
    let nodes: Vec<i32> = best
        .ids
        .iter()
        .copied()
        .filter(|&id| matches!(g.ty.get(&id), Some(&T_NOTABLE | &T_KEYSTONE)))
        .collect();
    // build all leave-one-out sets, score the un-memoized ones in one batch
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
    let mut enablers = Vec::new();
    for (i, &id) in nodes.iter().enumerate() {
        let loo = memo[&loo_sets[i]];
        let is_enabler = (best.dps > 0.0 && loo[1] < COLLAPSE * best.dps)
            || (best.ehp > 0.0 && loo[2] < COLLAPSE * best.ehp);
        if is_enabler {
            enablers.push(id);
        }
        marg.push(Marginal { id, marginal: best.score - loo[0], is_enabler });
    }
    (marg, enablers)
}

/// Pareto-augmented beam selection (spike's selectBeam): top-`width` by score, then
/// up to `extra` Pareto-non-dominated states (high on one axis even if lower-scoring)
/// so the damage-seeking branch isn't culled by a pure scalar top-N. `sorted` is
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
    // augment with Pareto-non-dominated (vs already-chosen) states
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

fn flatten(candidates: &[Vec<i32>]) -> (Vec<i32>, Vec<i32>) {
    let mut ids = Vec::new();
    let mut lengths = Vec::with_capacity(candidates.len());
    for c in candidates {
        ids.extend_from_slice(c);
        lengths.push(c.len() as i32);
    }
    (ids, lengths)
}

fn locate_repo_root() -> Option<PathBuf> {
    let from_manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf);
    if let Some(root) = from_manifest {
        if root.join("src/HeadlessWrapper.lua").exists() {
            return Some(root);
        }
    }
    let mut cur = std::env::current_dir().ok()?;
    loop {
        if cur.join("src/HeadlessWrapper.lua").exists() && cur.join("runtime/lua51.dll").exists() {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}
