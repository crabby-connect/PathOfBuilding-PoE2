//! C-ABI surface, called from the host LuaJIT via ffi.cdef/ffi.load. This is
//! the ONLY public boundary of the cdylib; everything else is internal.
//!
//! Lua-side contract (see worker_bootstrap.lua header and the design doc):
//!
//!   local C = ffi.load("pob_optimizer")       -- pob_optimizer.dll on PATH
//!   ffi.cdef[[
//!     typedef struct PobOptPool PobOptPool;
//!     PobOptPool* pob_opt_create(const char* config);
//!     int  pob_opt_score_batch(PobOptPool*, const int32_t* ids,
//!                              const int32_t* lengths, int32_t n, double* out);
//!     int  pob_opt_candidate_ids(PobOptPool*, int32_t* out, int32_t cap);
//!     int  pob_opt_worker_count(PobOptPool*);
//!     void pob_opt_destroy(PobOptPool*);
//!     const char* pob_opt_last_error(void);
//!   ]]
//!
//! The `config` string is a tiny newline-delimited `key=value` format (NOT JSON,
//! to keep the cdylib dependency-free) — see config.rs for the keys. Lua emits it
//! with a few string concatenations; Rust parses it without a JSON crate.
//!
//! score_batch flattens candidates: `ids` is every node id concatenated, and
//! `lengths[k]` is how many ids candidate k owns (so the worker slices them out).
//! `out` must point to `n` doubles the caller owns; scores are written in order,
//! NAN for any candidate that failed to score.
//!
//! Error handling: create returns NULL on failure; the reason is retrievable via
//! pob_opt_last_error() (thread-local, valid until the next FFI call on that
//! thread). No Rust panic is allowed to cross the FFI boundary — every entry
//! point is wrapped in catch_unwind.

use crate::config::Config;
use crate::lua::LuaLib;
use crate::pool::Pool;
use std::cell::RefCell;
use std::ffi::{c_char, c_double, c_int, CStr, CString};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::ptr;
use std::sync::Arc;

thread_local! {
    /// Last error message for THIS thread, as a NUL-terminated C string kept
    /// alive until the next call overwrites it. Lua copies it immediately.
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_last_error(msg: impl Into<String>) {
    let c = CString::new(msg.into()).unwrap_or_else(|_| CString::new("error (contained NUL)").unwrap());
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

/// Opaque handle the host holds. Internally it owns the worker pool plus the
/// LuaLib that keeps the DLL symbols alive for the pool's lifetime.
pub struct PobOptPool {
    pool: Pool,
    _lib: Arc<LuaLib>,
}

/// Boot a worker pool from a JSON config string. Returns NULL on failure (see
/// pob_opt_last_error). The returned pointer must be freed with pob_opt_destroy.
///
/// # Safety
/// `config_json` must be a valid NUL-terminated UTF-8 C string, or NULL.
#[no_mangle]
pub unsafe extern "C" fn pob_opt_create(config_json: *const c_char) -> *mut PobOptPool {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<*mut PobOptPool, String> {
        if config_json.is_null() {
            return Err("pob_opt_create: config_json is NULL".into());
        }
        let cfg_str = CStr::from_ptr(config_json)
            .to_str()
            .map_err(|_| "config_json is not valid UTF-8".to_string())?;
        let cfg = Config::parse(cfg_str)?;

        let lib = Arc::new(LuaLib::load(&cfg.lua_dll)?);
        // The engine assumes cwd == src/. All worker states share the one process
        // cwd, so set it once here, before any worker boots. (Same as the PoC.)
        std::env::set_current_dir(&cfg.src_dir)
            .map_err(|e| format!("chdir to src dir '{}': {e}", cfg.src_dir))?;

        let bootstrap = cfg.render_bootstrap()?;
        let pool = Pool::new(
            Arc::clone(&lib),
            bootstrap,
            cfg.score_fn.clone(),
            cfg.candidate_fn.clone(),
            cfg.workers,
        )?;
        Ok(Box::into_raw(Box::new(PobOptPool { pool, _lib: lib })))
    }));
    match result {
        Ok(Ok(p)) => p,
        Ok(Err(e)) => {
            set_last_error(e);
            ptr::null_mut()
        }
        Err(_) => {
            set_last_error("pob_opt_create panicked");
            ptr::null_mut()
        }
    }
}

/// Score `n` candidate node-id sets. `ids` is all candidate ids concatenated;
/// `lengths[k]` is the id count of candidate k. Writes `n` doubles to `out` in
/// candidate order (NAN for any that failed to score). Returns 0 on success,
/// non-zero on error (see pob_opt_last_error). On error, `out` is untouched.
///
/// # Safety
/// `pool` must be a live handle from pob_opt_create. `ids` must point to
/// (sum of lengths) int32s; `lengths` and `out` to `n` elements each. All must
/// stay valid for the duration of the call.
#[no_mangle]
pub unsafe extern "C" fn pob_opt_score_batch(
    pool: *mut PobOptPool,
    ids: *const i32,
    lengths: *const i32,
    n: c_int,
    out: *mut c_double,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if pool.is_null() {
            return Err("score_batch: pool is NULL".into());
        }
        if n < 0 {
            return Err("score_batch: n is negative".into());
        }
        let n = n as usize;
        if n == 0 {
            return Ok(());
        }
        if lengths.is_null() || out.is_null() {
            return Err("score_batch: lengths or out is NULL".into());
        }
        if ids.is_null() {
            // ids may legitimately be empty only if every length is 0; but a NULL
            // ids with n>0 candidates is a caller bug.
            return Err("score_batch: ids is NULL".into());
        }
        let pool = &(*pool).pool;
        let lengths = std::slice::from_raw_parts(lengths, n);

        // Reconstruct the candidate id-sets from the flat layout, validating that
        // the lengths don't run past the ids buffer the caller promised.
        let mut candidates: Vec<Vec<i32>> = Vec::with_capacity(n);
        let mut offset: usize = 0;
        for (k, &len) in lengths.iter().enumerate() {
            if len < 0 {
                return Err(format!("score_batch: candidate {k} has negative length"));
            }
            let len = len as usize;
            let slice = std::slice::from_raw_parts(ids.add(offset), len);
            candidates.push(slice.to_vec());
            offset += len;
        }

        let scores = pool.score_batch(&candidates);
        debug_assert_eq!(scores.len(), n);
        let out = std::slice::from_raw_parts_mut(out, n);
        out.copy_from_slice(&scores);
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            1
        }
        Err(_) => {
            set_last_error("pob_opt_score_batch panicked");
            2
        }
    }
}

/// Like `pob_opt_score_batch` but writes THREE doubles per candidate —
/// (score, dps, ehp) interleaved — so `out` must hold `3*n` doubles. Drives the
/// clean-slate score fn (the host's beam needs dps/ehp for Pareto pruning). A
/// candidate that fails scores (NAN, NAN, NAN). Returns 0 on success.
///
/// # Safety
/// As `pob_opt_score_batch`, but `out` must point to `3*n` doubles.
#[no_mangle]
pub unsafe extern "C" fn pob_opt_score_batch3(
    pool: *mut PobOptPool,
    ids: *const i32,
    lengths: *const i32,
    n: c_int,
    out: *mut c_double,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<(), String> {
        if pool.is_null() {
            return Err("score_batch3: pool is NULL".into());
        }
        if n < 0 {
            return Err("score_batch3: n is negative".into());
        }
        let n = n as usize;
        if n == 0 {
            return Ok(());
        }
        if lengths.is_null() || out.is_null() {
            return Err("score_batch3: lengths or out is NULL".into());
        }
        if ids.is_null() {
            return Err("score_batch3: ids is NULL".into());
        }
        let pool = &(*pool).pool;
        let lengths = std::slice::from_raw_parts(lengths, n);

        let mut candidates: Vec<Vec<i32>> = Vec::with_capacity(n);
        let mut offset: usize = 0;
        for (k, &len) in lengths.iter().enumerate() {
            if len < 0 {
                return Err(format!("score_batch3: candidate {k} has negative length"));
            }
            let len = len as usize;
            let slice = std::slice::from_raw_parts(ids.add(offset), len);
            candidates.push(slice.to_vec());
            offset += len;
        }

        let triples = pool.score_batch3(&candidates);
        debug_assert_eq!(triples.len(), n);
        let out = std::slice::from_raw_parts_mut(out, n * 3);
        for (k, t) in triples.iter().enumerate() {
            out[k * 3] = t[0];
            out[k * 3 + 1] = t[1];
            out[k * 3 + 2] = t[2];
        }
        Ok(())
    }));
    match result {
        Ok(Ok(())) => 0,
        Ok(Err(e)) => {
            set_last_error(e);
            1
        }
        Err(_) => {
            set_last_error("pob_opt_score_batch3 panicked");
            2
        }
    }
}

/// Save the optimized tree to an XML file by calling the worker's
/// `__pob_save_optimized(packed)`. `packed` is the int array the worker unpacks:
/// `[ n_ids, id1..idn, pathByte1..pathByteM ]` (path bytes are the UTF-8 of the
/// output file path). `len` is the packed length. Returns 1.0 on success, 0.0 on
/// failure, NAN if the call could not be dispatched. MUTATES a worker's spec — call
/// once, after all scoring is done.
///
/// # Safety
/// `pool` must be a live handle; `packed` must point to `len` valid int32s.
#[no_mangle]
pub unsafe extern "C" fn pob_opt_call_save(
    pool: *mut PobOptPool,
    packed: *const i32,
    len: c_int,
) -> c_double {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<f64, String> {
        if pool.is_null() {
            return Err("call_save: pool is NULL".into());
        }
        if len < 0 {
            return Err("call_save: len is negative".into());
        }
        let len = len as usize;
        if len > 0 && packed.is_null() {
            return Err("call_save: packed is NULL".into());
        }
        let args = std::slice::from_raw_parts(packed, len);
        Ok((*pool).pool.call_named("__pob_save_optimized", args))
    }));
    match result {
        Ok(Ok(v)) => v,
        Ok(Err(e)) => {
            set_last_error(e);
            f64::NAN
        }
        Err(_) => {
            set_last_error("pob_opt_call_save panicked");
            f64::NAN
        }
    }
}

/// Run the FULL clean-slate beam + Pareto + elimination-diet search inside the
/// pool and return the winning passive-tree node-id set. This is the production
/// search (the same one `examples/beam_ab.rs` drives); the host just kicks it off
/// and reads back a result instead of orchestrating the loop on its single state.
///
/// The pool MUST have been created with `candidate_fn=__pob_graph_export` and
/// `score_fn=__pob_score_cleanslate` (so candidate_ids hands back the flat graph
/// and scoring returns score/dps/ehp); otherwise the search has no topology and
/// this returns -1.
///
/// `params` is a newline/comma `key=value` override string (or NULL/empty for all
/// defaults) — keys: cap_points, beam_width, max_jump, pareto_extra, detour,
/// patience_max, max_rounds, verbose. See `beam::BeamParams::parse`.
///
/// `out_ids` is sized by the caller to `cap` int32s; the winning ids (incl. the
/// class start, sorted) are written there. Returns the TOTAL winning id count
/// (call once with cap=0/out_ids=NULL to size, then again to fetch), or -1 on
/// error (see pob_opt_last_error). `out_stats`, if non-NULL, receives 3 doubles:
/// [score, dps, ehp] of the winning build.
///
/// # Safety
/// `pool` must be a live handle. `params` is a NUL-terminated C string or NULL.
/// `out_ids` must point to `cap` int32s (or be NULL iff cap==0); `out_stats` must
/// point to 3 doubles or be NULL.
#[no_mangle]
pub unsafe extern "C" fn pob_opt_run_beam(
    pool: *mut PobOptPool,
    params: *const c_char,
    out_ids: *mut i32,
    cap: c_int,
    out_stats: *mut c_double,
) -> c_int {
    let result = catch_unwind(AssertUnwindSafe(|| -> Result<crate::beam::BeamResult, String> {
        if pool.is_null() {
            return Err("run_beam: pool is NULL".into());
        }
        let params_str = if params.is_null() {
            ""
        } else {
            CStr::from_ptr(params)
                .to_str()
                .map_err(|_| "run_beam: params is not valid UTF-8".to_string())?
        };
        let beam_params = crate::beam::BeamParams::parse(params_str)?;

        let pool = &(*pool).pool;
        let flat = pool.candidate_ids();
        if flat.len() < 3 {
            return Err(
                "run_beam: no graph exported (create the pool with candidate_fn=__pob_graph_export)"
                    .into(),
            );
        }
        let graph = crate::beam::Graph::parse(flat);

        // Optional persistent score cache (opt-in via POB_OPT_CACHE=<path>). Warm-starts
        // the memo from a prior run of the SAME build and writes it back on completion;
        // a weight-only change still hits fully (col 0 is re-derived). Unset => no cache,
        // identical to the old behavior. The cache only activates with a real trailer
        // (refs present), since the signature + growth-score recompute need it.
        let cache_path = std::env::var_os("POB_OPT_CACHE").map(std::path::PathBuf::from);
        let build_sig = crate::cache::build_signature(flat);
        let mut memo = match (&cache_path, graph.has_trailer()) {
            (Some(path), true) => crate::cache::load_with_sig(path, build_sig, &graph),
            _ => Default::default(),
        };
        if beam_params.verbose {
            if let Some(path) = &cache_path {
                println!(
                    "  cache: warm-loaded {} entries from {} (build_sig={:#x})",
                    memo.len(), path.display(), build_sig
                );
            }
        }

        // The search's only contact with the engine: batch-score via the pool.
        let res = crate::beam::optimize_with_memo(&graph, &beam_params, &mut memo, |cands| {
            pool.score_batch3(cands)
        });

        if let (Some(path), true) = (&cache_path, graph.has_trailer()) {
            // A failed cache write must never fail the search — log to last_error slot
            // for diagnostics but return the result regardless.
            match crate::cache::save(path, build_sig, &memo) {
                Ok(n) if beam_params.verbose => {
                    println!("  cache: wrote {n} entries -> {}", path.display());
                }
                Ok(_) => {}
                Err(e) => set_last_error(format!("run_beam: cache save failed (non-fatal): {e}")),
            }
        }
        if beam_params.verbose {
            // Memory signal: the score memo is the dominant run-length-dependent heap
            // user (worker engines are large but fixed). Print it so the paging
            // hypothesis is checkable against a number.
            println!(
                "  memo: {} entries, ~{:.1} MB heap; {} total calc evals",
                res.memo_entries,
                res.memo_bytes_est as f64 / (1024.0 * 1024.0),
                res.evals
            );
        }
        Ok(res)
    }));
    match result {
        Ok(Ok(res)) => {
            if !out_stats.is_null() {
                let stats = std::slice::from_raw_parts_mut(out_stats, 3);
                stats[0] = res.score;
                stats[1] = res.dps;
                stats[2] = res.ehp;
            }
            let cap = cap.max(0) as usize;
            if !out_ids.is_null() && cap > 0 {
                let n = res.ids.len().min(cap);
                std::ptr::copy_nonoverlapping(res.ids.as_ptr(), out_ids, n);
            }
            res.ids.len() as c_int
        }
        Ok(Err(e)) => {
            set_last_error(e);
            -1
        }
        Err(_) => {
            set_last_error("pob_opt_run_beam panicked");
            -1
        }
    }
}

/// Copy the eligible candidate node ids (computed once at boot) into `out`,
/// which the caller sizes to `cap` int32s. Returns the TOTAL candidate count
/// (which may exceed `cap` — call once with cap=0/out=NULL to size, then again).
/// Returns -1 if `pool` is NULL. Writes min(count, cap) ids.
///
/// # Safety
/// `pool` must be a live handle, or NULL. `out` must point to `cap` int32s (or be
/// NULL iff cap==0).
#[no_mangle]
pub unsafe extern "C" fn pob_opt_candidate_ids(
    pool: *mut PobOptPool,
    out: *mut i32,
    cap: c_int,
) -> c_int {
    if pool.is_null() {
        return -1;
    }
    let ids = (*pool).pool.candidate_ids();
    let cap = cap.max(0) as usize;
    if !out.is_null() && cap > 0 {
        let n = ids.len().min(cap);
        std::ptr::copy_nonoverlapping(ids.as_ptr(), out, n);
    }
    ids.len() as c_int
}

/// Number of worker threads in the pool, or -1 if `pool` is NULL. Useful for the
/// host to size batches (≈ a few × worker_count keeps all workers fed).
///
/// # Safety
/// `pool` must be a live handle from pob_opt_create, or NULL.
#[no_mangle]
pub unsafe extern "C" fn pob_opt_worker_count(pool: *mut PobOptPool) -> c_int {
    if pool.is_null() {
        return -1;
    }
    (*pool).pool.worker_count() as c_int
}

/// Destroy a pool created by pob_opt_create, joining all worker threads. Safe to
/// call with NULL (no-op). After this returns the handle is dangling.
///
/// # Safety
/// `pool` must be a handle from pob_opt_create that has not already been
/// destroyed, or NULL.
#[no_mangle]
pub unsafe extern "C" fn pob_opt_destroy(pool: *mut PobOptPool) {
    if pool.is_null() {
        return;
    }
    // Reconstitute the Box and drop it (Pool::drop joins the worker threads).
    // catch_unwind so a worker panic during join can't unwind across FFI.
    let _ = catch_unwind(AssertUnwindSafe(|| {
        drop(Box::from_raw(pool));
    }));
}

/// The last error message on the calling thread, or NULL if none. The pointer is
/// valid until the next FFI call on this thread; copy it immediately.
#[no_mangle]
pub extern "C" fn pob_opt_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => ptr::null(),
    })
}
