//! pob-optimizer: the parallel tree-optimizer search driver, production cdylib.
//!
//! The host PoB LuaJIT state generates candidate passive-tree node-id sets and
//! wants each scored by the real calc engine (`calcs.perform` via the misc
//! calculator). Scoring is the ~8ms bottleneck and dominates any search. This
//! library drives a PERSISTENT pool of worker LuaJIT states — each a full,
//! independent headless engine — to score candidates across CPU cores.
//!
//! See docs/rust-offload-poc.md for the PoC findings (the worker model and its
//! ~3.4x scaling ceiling) and docs/tree-optimizer-design.md for the search it
//! feeds. The public surface is in `ffi` (extern "C", called via LuaJIT FFI);
//! everything else is internal.

pub mod beam;
mod config;
mod ffi;
mod lua;
mod pool;

// Re-export the FFI entry points so they are exported from the cdylib.
pub use ffi::{
    pob_opt_call_save, pob_opt_candidate_ids, pob_opt_create, pob_opt_destroy,
    pob_opt_last_error, pob_opt_run_beam, pob_opt_score_batch, pob_opt_score_batch3,
    pob_opt_worker_count, PobOptPool,
};
