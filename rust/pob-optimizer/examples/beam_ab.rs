//! Native A/B harness: drive the SHIPPED cdylib's C ABI exactly as the host
//! LuaJIT does, to (1) measure whether the parallel pool is a throughput WIN over
//! single-threaded scoring, and (2) run the full production beam search end-to-end
//! and save the optimized build.
//!
//! Why this exists: the busted spike (spec/System/SpikeCleanSlateBeam_spec.lua)
//! runs the beam in Lua, but only inside the Linux test container, which cannot
//! load this Windows cdylib. So the only place the pool can actually be driven on
//! a dev box is a native harness like this (same model as examples/smoke.rs).
//! Like smoke.rs, it loads the DLL via libloading and calls the C ABI — it does
//! NOT link the crate, so it exercises exactly the surface the host uses.
//!
//! The search itself now lives in the crate (src/beam.rs) behind the
//! `pob_opt_run_beam` FFI entry; this harness no longer reimplements it. It does:
//!   1. MICROBENCHMARK — score one representative batch of candidate node-sets
//!      (a) through the N-worker pool and (b) single-threaded (a 1-worker pool),
//!      report wall-time + speedup, and assert the scores match within 1e-6.
//!   2. FULL SEARCH — call pob_opt_run_beam (the production beam + Pareto + diet),
//!      report the winning score/dps/ehp, then save the optimized build via
//!      pob_opt_call_save.
//!
//! Run from repo root (PATH must include runtime/ for transitive DLLs):
//!   $env:PATH = "$PWD\runtime;$env:PATH"
//!   cargo run --release --manifest-path rust/pob-optimizer/Cargo.toml \
//!       --example beam_ab -- [workers] [capPoints] [beamWidth] [outFile] [wDps] [wEhp]
//!
//! Defaults: workers = available, capPoints = 0 (build's real budget), beamWidth = 8,
//! outFile = mymonk_optimized.xml, wDps = 1, wEhp = 1.

use libloading::{Library, Symbol};
use std::collections::HashSet;
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
type FnRunBeam = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const c_char,
    *mut i32,
    c_int,
    *mut c_double,
) -> c_int;
type FnDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);
type FnLastError = unsafe extern "C" fn() -> *const c_char;
type FnCallSave = unsafe extern "C" fn(*mut std::ffi::c_void, *const i32, c_int) -> c_double;

const T_NOTABLE: i32 = 2;
const T_KEYSTONE: i32 = 3;

fn main() {
    let mut args = std::env::args().skip(1);
    let workers: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    // capPoints arg: 0 (or omitted) => use the build's REAL budget (exported by the worker).
    let cap_arg: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(0);
    let beam_width: usize = args.next().and_then(|s| s.parse().ok()).unwrap_or(8);
    // outFile arg (optional): the saved-build filename under src/Builds/.
    let out_file: String = args.next().unwrap_or_else(|| "mymonk_optimized.xml".to_string());
    // wDps / wEhp args (optional): score weights. Default 1/1. Raise wDps to bias toward damage.
    let w_dps: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);
    let w_ehp: f64 = args.next().and_then(|s| s.parse().ok()).unwrap_or(1.0);

    let root = locate_repo_root().expect("locate repo root (need src/ + runtime/)");
    let fwd = |p: PathBuf| p.to_string_lossy().replace('\\', "/");
    let cdylib = root.join("rust/pob-optimizer/target/release/pob_optimizer.dll");
    assert!(cdylib.exists(), "build the cdylib first: {}", cdylib.display());

    // Two configs differ ONLY in worker count. candidate_fn => __pob_graph_export so the
    // pool hands back the flat graph the beam needs; score_fn => the clean-slate 3-return path.
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
        let run_beam: Symbol<FnRunBeam> = lib.get(b"pob_opt_run_beam").unwrap();
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

        // ---- pull the graph the workers exported (for the benchmark batch + reporting) ----
        let total = candidate_ids(pool, std::ptr::null_mut(), 0);
        assert!(total > 0, "graph export empty: {}", read_err());
        let mut flat = vec![0i32; total as usize];
        let got = candidate_ids(pool, flat.as_mut_ptr(), total);
        assert_eq!(got, total);
        let graph = parse_graph(&flat);
        let n_targets = graph.ty.values().filter(|&&t| t == T_NOTABLE || t == T_KEYSTONE).count();
        let budget = graph.budget;
        println!(
            "graph: {} nodes, start id {}, {} notable/keystone targets, build budget P={}",
            graph.links.len(), graph.start, n_targets, budget
        );

        let score_batch3 = |pool: *mut std::ffi::c_void, cands: &[Vec<i32>]| -> Vec<[f64; 3]> {
            if cands.is_empty() {
                return Vec::new();
            }
            let (ids, lengths) = flatten(cands);
            let mut out = vec![0.0f64; cands.len() * 3];
            let rc = score3(pool, ids.as_ptr(), lengths.as_ptr(), cands.len() as c_int, out.as_mut_ptr());
            assert_eq!(rc, 0, "score_batch3 rc={rc}: {}", read_err());
            (0..cands.len()).map(|k| [out[k * 3], out[k * 3 + 1], out[k * 3 + 2]]).collect()
        };

        // ============================================================================
        // 1. MICROBENCHMARK — same batch, parallel pool vs single-threaded (1 worker).
        // ============================================================================
        let bench = build_bench_batch(&graph, 400);
        println!("\n=== MICROBENCHMARK: {} candidate sets ===", bench.len());

        let _ = score_batch3(pool, &bench[..bench.len().min(nw as usize)]);
        let t_par = Instant::now();
        let par_scores = score_batch3(pool, &bench);
        let par_dt = t_par.elapsed().as_secs_f64();

        println!("booting 1-worker baseline pool ...");
        let cfg1 = CString::new(make_config(1)).unwrap();
        let pool1 = create(cfg1.as_ptr());
        assert!(!pool1.is_null(), "baseline pool create failed: {}", read_err());
        let _ = score_batch3(pool1, &bench[..1]);
        let t_seq = Instant::now();
        let seq_scores = score_batch3(pool1, &bench);
        let seq_dt = t_seq.elapsed().as_secs_f64();
        destroy(pool1);

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
        // 2. FULL SEARCH — the production beam + diet, via the pob_opt_run_beam FFI.
        // ============================================================================
        let cap_points = if cap_arg > 0 { cap_arg } else { budget };
        let params = format!(
            "cap_points={cap_arg},beam_width={beam_width},max_jump=12,patience_max=2,max_rounds=12,verbose=1"
        );
        println!("\n=== FULL BEAM + DIET via pob_opt_run_beam (P={cap_points}, beam={beam_width}) ===");
        let t_beam = Instant::now();
        let params_c = CString::new(params).unwrap();
        // The winning set is at most cap_points + 1 ids (start + cap_points allocated),
        // so size the buffer from that known bound and call ONCE — the size-then-fetch
        // idiom would re-run the whole (minutes-long) search just to count.
        let mut win_ids = vec![0i32; cap_points + 1];
        let mut stats = [0.0f64; 3];
        let count = run_beam(pool, params_c.as_ptr(), win_ids.as_mut_ptr(), win_ids.len() as c_int, stats.as_mut_ptr());
        assert!(count > 0, "run_beam returned {count}: {}", read_err());
        assert!(count as usize <= win_ids.len(), "run_beam count {count} exceeds budget bound {}", win_ids.len());
        win_ids.truncate(count as usize);
        let beam_dt = t_beam.elapsed().as_secs_f64();
        println!(
            "\n  FINAL: score={:.1}  dps={:.1}  ehp={:.1}  points={}  ({:.1}s)",
            stats[0], stats[1], stats[2], win_ids.len() - 1, beam_dt
        );
        assert!(win_ids.len() > 1, "beam allocated nothing");
        assert!(stats[0].is_finite() && stats[0] > 0.0, "beam best score not positive");

        // ---- SAVE the optimized tree (mutates a worker spec, so it is the LAST pool call). ----
        let out_path = root.join("src/Builds").join(&out_file);
        let out_str = out_path.to_string_lossy().replace('\\', "/");
        let mut packed: Vec<i32> = Vec::with_capacity(win_ids.len() + out_str.len() + 1);
        packed.push(win_ids.len() as i32);
        packed.extend(win_ids.iter().copied());
        packed.extend(out_str.bytes().map(|b| b as i32));
        let saved = call_save(pool, packed.as_ptr(), packed.len() as c_int);
        if saved == 1.0 {
            println!("\n=== SAVED optimized build -> {out_str} ===");
            println!("  Open it in PoB (Import/Open) to inspect the optimized tree.");
        } else {
            println!("\n=== SAVE FAILED (rc={saved}): {} ===", read_err());
        }

        destroy(pool);
        println!("\nBEAM A/B HARNESS PASSED");
    }
}

/// Minimal graph view for the benchmark batch (the search uses the crate's own
/// parser behind the FFI; this is only for building a representative batch here).
struct MiniGraph {
    start: i32,
    budget: usize,
    links: std::collections::HashMap<i32, Vec<i32>>,
    ty: std::collections::HashMap<i32, i32>,
}

fn parse_graph(flat: &[i32]) -> MiniGraph {
    let start = flat[0];
    let budget = flat[1].max(0) as usize;
    let n = flat[2] as usize;
    let mut links = std::collections::HashMap::new();
    let mut ty = std::collections::HashMap::new();
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
    MiniGraph { start, budget, links, ty }
}

impl MiniGraph {
    fn is_target(&self, id: i32) -> bool {
        matches!(self.ty.get(&id), Some(&1 | &2 | &3))
    }
}

/// Build `count` connected partial trees of growing size by BFS from the start —
/// a representative scoring batch (real connected candidates, not noise).
fn build_bench_batch(g: &MiniGraph, count: usize) -> Vec<Vec<i32>> {
    let order = bfs_order(g);
    let mut out = Vec::with_capacity(count);
    let max_len = order.len();
    for k in 0..count {
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
fn bfs_order(g: &MiniGraph) -> Vec<i32> {
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
