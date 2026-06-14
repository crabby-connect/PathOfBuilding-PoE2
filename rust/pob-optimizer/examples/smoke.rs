//! End-to-end smoke test for the cdylib, calling the extern "C" surface exactly
//! as the host LuaJIT would (same C ABI, same call sequence). This exercises the
//! WHOLE stack against the real engine — config parse, DLL load, worker boot,
//! HeadlessWrapper engine boot, batch scoring, result reassembly, teardown —
//! without needing a standalone luajit CLI (the repo has none; tests normally run
//! in the pathofbuilding-tests container).
//!
//! Run from repo root:
//!   cargo run --release --manifest-path rust/pob-optimizer/Cargo.toml \
//!       --example smoke -- [workers]
//!
//! PATH must include `runtime/` so transitive DLLs (lua-utf8.dll, ...) resolve.
//!
//! It calls the FFI fns through their raw symbols (resolved from the just-built
//! cdylib) so the test and the shipped library are byte-for-byte the same path.

use libloading::{Library, Symbol};
use std::ffi::{c_char, c_double, c_int, CStr, CString};
use std::path::{Path, PathBuf};

type FnCreate = unsafe extern "C" fn(*const c_char) -> *mut std::ffi::c_void;
type FnScore = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const i32,
    *const i32,
    c_int,
    *mut c_double,
) -> c_int;
type FnScore3 = unsafe extern "C" fn(
    *mut std::ffi::c_void,
    *const i32,
    *const i32,
    c_int,
    *mut c_double,
) -> c_int;
type FnWorkerCount = unsafe extern "C" fn(*mut std::ffi::c_void) -> c_int;
type FnCandidateIds = unsafe extern "C" fn(*mut std::ffi::c_void, *mut i32, c_int) -> c_int;
type FnCallSave = unsafe extern "C" fn(*mut std::ffi::c_void, *const i32, c_int) -> c_double;
type FnDestroy = unsafe extern "C" fn(*mut std::ffi::c_void);
type FnLastError = unsafe extern "C" fn() -> *const c_char;

fn main() {
    let workers: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(4);

    let root = locate_repo_root().expect("locate repo root (need src/ + runtime/)");
    let fwd = |p: PathBuf| p.to_string_lossy().replace('\\', "/");
    let cdylib = root
        .join("rust/pob-optimizer/target/release/pob_optimizer.dll");
    assert!(cdylib.exists(), "build the cdylib first: {}", cdylib.display());

    // Config builder: the original path uses the scalar score fn + candidate enum;
    // the clean-slate path (score3 / graph export / save, exercised below) swaps
    // score_fn/candidate_fn and adds weights. `extra` appends those lines.
    let make_config = |score_fn: &str, candidate_fn: &str, extra: &str| {
        format!(
            "lua_dll={}\nsrc_dir={}\nruntime_dir={}\nruntime_lua={}\nbuild_xml={}\nbootstrap={}\nscore_fn={}\ncandidate_fn={}\nworkers={}\n{}",
            fwd(root.join("runtime/lua51.dll")),
            fwd(root.join("src")),
            fwd(root.join("runtime")),
            fwd(root.join("runtime/lua")),
            fwd(root.join("src/Builds/mymonk.xml")),
            fwd(root.join("rust/pob-optimizer/worker_bootstrap.lua")),
            score_fn,
            candidate_fn,
            workers,
            extra,
        )
    };
    let config = make_config("__pob_score", "__pob_candidate_ids", "");

    let lib = unsafe { Library::new(&cdylib) }.expect("load cdylib");
    unsafe {
        let create: Symbol<FnCreate> = lib.get(b"pob_opt_create").unwrap();
        let score: Symbol<FnScore> = lib.get(b"pob_opt_score_batch").unwrap();
        let score3: Symbol<FnScore3> = lib.get(b"pob_opt_score_batch3").unwrap();
        let worker_count: Symbol<FnWorkerCount> = lib.get(b"pob_opt_worker_count").unwrap();
        let candidate_ids: Symbol<FnCandidateIds> = lib.get(b"pob_opt_candidate_ids").unwrap();
        let call_save: Symbol<FnCallSave> = lib.get(b"pob_opt_call_save").unwrap();
        let destroy: Symbol<FnDestroy> = lib.get(b"pob_opt_destroy").unwrap();
        let last_error: Symbol<FnLastError> = lib.get(b"pob_opt_last_error").unwrap();

        let read_err = || {
            let p = last_error();
            if p.is_null() {
                "<none>".to_string()
            } else {
                CStr::from_ptr(p).to_string_lossy().into_owned()
            }
        };

        println!("booting pool ({workers} workers) ...");
        let cfg = CString::new(config).unwrap();
        let pool = create(cfg.as_ptr());
        assert!(!pool.is_null(), "pob_opt_create failed: {}", read_err());
        println!("pool up, worker_count = {}", worker_count(pool));

        // Discover real candidate ids from the engine (sized call, then fetch).
        let total = candidate_ids(pool, std::ptr::null_mut(), 0);
        assert!(total > 0, "no candidate ids: {}", read_err());
        let mut all_ids = vec![0i32; total as usize];
        let got = candidate_ids(pool, all_ids.as_mut_ptr(), total);
        assert_eq!(got, total);
        println!("engine reports {total} eligible candidate nodes");

        // Batch: the base tree (empty id-set) twice — must be byte-identical and
        // equal across workers (the PoC's isolation check, through the FFI path) —
        // plus several real single-node additions. Each addition should change the
        // score (the engine actually evaluated the node), and at least one should
        // differ, proving non-empty candidates flow end-to-end.
        let mut candidates: Vec<Vec<i32>> = vec![vec![], vec![]];
        for &id in all_ids.iter().take(8) {
            candidates.push(vec![id]);
        }
        let (ids, lengths) = flatten(&candidates);
        let mut out = vec![0.0f64; candidates.len()];
        let rc = score(
            pool,
            ids.as_ptr(),
            lengths.as_ptr(),
            candidates.len() as c_int,
            out.as_mut_ptr(),
        );
        assert_eq!(rc, 0, "score_batch rc={rc}: {}", read_err());

        println!("scores: {out:?}");
        for (i, s) in out.iter().enumerate() {
            assert!(s.is_finite(), "candidate {i} scored non-finite {s}");
        }
        let base = out[0];
        assert!(
            (out[1] - base).abs() < 1e-6,
            "two base-tree queries disagree: {} vs {base} (workers not isolated?)",
            out[1]
        );
        println!("OK: base score {base:.3}, two empty queries agree exactly");

        let changed = out[2..].iter().filter(|&&s| (s - base).abs() > 1e-6).count();
        assert!(
            changed > 0,
            "no single-node addition changed the score — additions not reaching the engine?"
        );
        println!("OK: {changed}/{} node additions changed the score", out.len() - 2);

        // Second batch to prove the pool is PERSISTENT (reused, not respawned).
        let rc2 = score(
            pool,
            ids.as_ptr(),
            lengths.as_ptr(),
            candidates.len() as c_int,
            out.as_mut_ptr(),
        );
        assert_eq!(rc2, 0, "second score_batch rc={rc2}: {}", read_err());
        assert!((out[0] - base).abs() < 1e-6, "score changed across batches");
        println!("OK: second batch reused the same pool, base score stable");

        destroy(pool);
        println!("OK: pool destroyed cleanly");

        // ====================================================================
        // CLEAN-SLATE SURFACE: score_batch3 + graph export + weights + save.
        // A second pool wired the way the beam uses it (clean-slate scorer +
        // graph export), so the newer FFI is regression-covered like the above.
        // ====================================================================
        println!("\n--- clean-slate surface (score3 / graph / weights / save) ---");
        let cfg_cs = CString::new(make_config(
            "__pob_score_cleanslate",
            "__pob_graph_export",
            "w_dps=1\nw_ehp=1\n",
        ))
        .unwrap();
        let pool = create(cfg_cs.as_ptr());
        assert!(!pool.is_null(), "clean-slate pool create failed: {}", read_err());

        // Graph export parses: candidate_ids now returns the flat graph array
        // [ startId, budgetP, nNodes, ... ]. Assert a positive node count + budget.
        let glen = candidate_ids(pool, std::ptr::null_mut(), 0);
        assert!(glen >= 3, "graph export too short ({glen}): {}", read_err());
        let mut flat = vec![0i32; glen as usize];
        candidate_ids(pool, flat.as_mut_ptr(), glen);
        let (start_id, budget, n_nodes) = (flat[0], flat[1], flat[2]);
        assert!(budget > 0 && n_nodes > 0, "graph export budget/nodes not positive: budget={budget} nodes={n_nodes}");
        println!("OK: graph export parsed (start {start_id}, budget {budget}, {n_nodes} nodes)");

        // score_batch3 on the base tree (start-only) returns finite (score,dps,ehp).
        let base3 = {
            let cand = [start_id];
            let mut out = vec![0.0f64; 3];
            let rc = score3(pool, cand.as_ptr(), [1i32].as_ptr(), 1, out.as_mut_ptr());
            assert_eq!(rc, 0, "score_batch3 rc={rc}: {}", read_err());
            assert!(out.iter().all(|v| v.is_finite()), "score3 returned non-finite {out:?}");
            println!("OK: score3 base (score={:.3}, dps={:.1}, ehp={:.1})", out[0], out[1], out[2]);
            out
        };
        destroy(pool);

        // Weights actually move the score: a dps-heavy pool scores the same base
        // tree differently from the ehp-heavy one (unless dps==ehp, which mymonk's
        // base is not). Boot two extra single-worker pools and compare.
        let score_base_with = |w_dps: f64, w_ehp: f64| -> f64 {
            let cfg = CString::new(make_config(
                "__pob_score_cleanslate",
                "__pob_graph_export",
                &format!("w_dps={w_dps}\nw_ehp={w_ehp}\nworkers=1\n"),
            ))
            .unwrap();
            let p = create(cfg.as_ptr());
            assert!(!p.is_null(), "weighted pool create failed: {}", read_err());
            let g = candidate_ids(p, std::ptr::null_mut(), 0);
            let mut f = vec![0i32; g as usize];
            candidate_ids(p, f.as_mut_ptr(), g);
            let cand = [f[0]];
            let mut out = vec![0.0f64; 3];
            let rc = score3(p, cand.as_ptr(), [1i32].as_ptr(), 1, out.as_mut_ptr());
            assert_eq!(rc, 0, "weighted score3 rc={rc}: {}", read_err());
            destroy(p);
            out[0]
        };
        let dps_heavy = score_base_with(3.0, 1.0);
        let ehp_heavy = score_base_with(1.0, 3.0);
        assert!(
            (dps_heavy - ehp_heavy).abs() > 1e-6,
            "weights had no effect: dps-heavy {dps_heavy} == ehp-heavy {ehp_heavy}"
        );
        println!("OK: weights move the score (dps-heavy {dps_heavy:.3} vs ehp-heavy {ehp_heavy:.3})");

        // call_save writes a valid (non-empty) XML for the start-only tree. Pack
        // [ n_ids, ids..., path bytes... ] as the worker contract requires.
        let pool = create(cfg_cs.as_ptr());
        assert!(!pool.is_null(), "save pool create failed: {}", read_err());
        let out_path = root.join("src/Builds/smoke_optimized.xml");
        let out_str = out_path.to_string_lossy().replace('\\', "/");
        let _ = std::fs::remove_file(&out_path);
        let mut packed: Vec<i32> = vec![1, start_id];
        packed.extend(out_str.bytes().map(|b| b as i32));
        let saved = call_save(pool, packed.as_ptr(), packed.len() as c_int);
        assert_eq!(saved, 1.0, "call_save returned {saved}: {}", read_err());
        let written = std::fs::metadata(&out_path).map(|m| m.len()).unwrap_or(0);
        assert!(written > 0, "call_save wrote an empty file");
        println!("OK: call_save wrote {written} bytes -> {out_str}");
        let _ = std::fs::remove_file(&out_path); // clean up the smoke artifact
        destroy(pool);
        let _ = base3; // (kept for readability; asserted finite above)
        println!("OK: clean-slate pool destroyed cleanly");
    }
    println!("\nSMOKE TEST PASSED");
}

/// Flatten candidate id-sets into the (ids, lengths) layout score_batch expects.
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
        if cur.join("src/HeadlessWrapper.lua").exists()
            && cur.join("runtime/lua51.dll").exists()
        {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}
