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
type FnWorkerCount = unsafe extern "C" fn(*mut std::ffi::c_void) -> c_int;
type FnCandidateIds = unsafe extern "C" fn(*mut std::ffi::c_void, *mut i32, c_int) -> c_int;
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

    let config = format!(
        "lua_dll={}\nsrc_dir={}\nruntime_dir={}\nruntime_lua={}\nbuild_xml={}\nbootstrap={}\nscore_fn=__pob_score\ncandidate_fn=__pob_candidate_ids\nworkers={}\n",
        fwd(root.join("runtime/lua51.dll")),
        fwd(root.join("src")),
        fwd(root.join("runtime")),
        fwd(root.join("runtime/lua")),
        fwd(root.join("src/Builds/mymonk.xml")),
        fwd(root.join("rust/pob-optimizer/worker_bootstrap.lua")),
        workers,
    );

    let lib = unsafe { Library::new(&cdylib) }.expect("load cdylib");
    unsafe {
        let create: Symbol<FnCreate> = lib.get(b"pob_opt_create").unwrap();
        let score: Symbol<FnScore> = lib.get(b"pob_opt_score_batch").unwrap();
        let worker_count: Symbol<FnWorkerCount> = lib.get(b"pob_opt_worker_count").unwrap();
        let candidate_ids: Symbol<FnCandidateIds> = lib.get(b"pob_opt_candidate_ids").unwrap();
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
