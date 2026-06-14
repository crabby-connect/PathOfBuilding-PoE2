//! Persistent worker pool. N OS threads, each owning one LuaJIT state for its
//! whole life: booted once (engine + tree + build), then reused across every
//! search round. This is the production correction to the PoC, which respawned
//! a fresh state per round and paid the 1.5–5.8s boot every time.
//!
//! Threading model (matches the PoC's proven facts):
//!   * Each `lua_State` is created on, used by, and destroyed on its own thread.
//!     States share nothing, so there is no lock around `perform`.
//!   * The shared `LuaLib` (DLL symbols) is Send+Sync — only code addresses.
//!   * Work is handed in as batches of candidate node-id sets; results come back
//!     as a parallel Vec<f64> of scores, one per candidate, in input order.

use crate::lua::{Lua, LuaLib};
use std::sync::mpsc::{Receiver, Sender};
use std::sync::{mpsc, Arc, Mutex};
use std::thread::JoinHandle;

/// One candidate to score: a node-id set, tagged with its index in the caller's
/// batch so results can be reassembled in order regardless of which worker took
/// it. `triple` selects the score arity: false => scalar score fn (one return,
/// dps/ehp come back NAN); true => clean-slate fn returning (score, dps, ehp).
struct Job {
    index: usize,
    ids: Vec<i32>,
    triple: bool,
    /// When set, call THIS named global (scalar return) instead of the configured
    /// score fn. Used for one-off worker calls like saving the optimized build.
    call_fn: Option<String>,
}

/// A scored result flowing back from a worker. Always carries [score, dps, ehp];
/// the scalar path fills dps/ehp with NAN (the host ignores them there).
struct Done {
    index: usize,
    out: [f64; 3],
}

enum Msg {
    Job(Job),
    /// Sent once per worker to unblock and terminate it on shutdown.
    Shutdown,
}

pub struct Pool {
    workers: Vec<JoinHandle<()>>,
    // The receiving end of the work queue lives in the workers (each holds an
    // Arc<Mutex<Receiver>> clone), so the channel stays alive as long as any
    // worker does. The Pool only keeps the sender.
    job_tx: Sender<Msg>,
    done_rx: Receiver<Done>,
    n_workers: usize,
    /// Eligible candidate node ids, computed once at boot. All workers compute
    /// the same list (identical tree); we keep the first worker's. Empty if the
    /// bootstrap installs no candidate function.
    candidate_ids: Vec<i32>,
}

impl Pool {
    /// Boot `n_workers` states, each running `bootstrap` (the worker boot chunk,
    /// with paths already substituted) and then exposing the score function the
    /// hot path calls. Returns once every worker has finished booting, or an
    /// error from the first worker that failed to boot.
    pub fn new(
        lib: Arc<LuaLib>,
        bootstrap: String,
        score_fn: String,
        candidate_fn: String,
        n_workers: usize,
    ) -> Result<Self, String> {
        let (job_tx, job_rx) = mpsc::channel::<Msg>();
        let (done_tx, done_rx) = mpsc::channel::<Done>();
        let job_rx = Arc::new(Mutex::new(job_rx));

        // Each worker reports boot success/failure once (with its candidate-id
        // list on success), so we can fail fast with a real error message instead
        // of a half-booted pool.
        let (boot_tx, boot_rx) = mpsc::channel::<Result<Vec<i32>, String>>();

        let mut workers = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let lib = Arc::clone(&lib);
            let bootstrap = bootstrap.clone();
            let score_fn = score_fn.clone();
            let candidate_fn = candidate_fn.clone();
            let job_rx = Arc::clone(&job_rx);
            let done_tx = done_tx.clone();
            let boot_tx = boot_tx.clone();
            workers.push(std::thread::spawn(move || {
                worker_main(lib, bootstrap, score_fn, candidate_fn, job_rx, done_tx, boot_tx);
            }));
        }
        drop(boot_tx); // only the workers hold senders now

        // Collect one boot result per worker. Keep the first worker's candidate
        // list (all are identical — same tree).
        let mut first_err: Option<String> = None;
        let mut candidate_ids: Vec<i32> = Vec::new();
        for _ in 0..n_workers {
            match boot_rx.recv() {
                Ok(Ok(ids)) => {
                    if candidate_ids.is_empty() {
                        candidate_ids = ids;
                    }
                }
                Ok(Err(e)) => {
                    first_err.get_or_insert(e);
                }
                Err(_) => {
                    first_err.get_or_insert_with(|| "worker panicked during boot".into());
                }
            };
        }

        let pool = Pool {
            workers,
            job_tx,
            done_rx,
            n_workers,
            candidate_ids,
        };

        if let Some(e) = first_err {
            // Tear the (partial) pool down cleanly before surfacing the error.
            pool.shutdown_internal();
            return Err(e);
        }
        Ok(pool)
    }

    pub fn worker_count(&self) -> usize {
        self.n_workers
    }

    /// Eligible candidate node ids (computed once at boot). Empty if the
    /// bootstrap installs no candidate function.
    pub fn candidate_ids(&self) -> &[i32] {
        &self.candidate_ids
    }

    /// Score a batch of candidate node-id sets. Returns one f64 per candidate,
    /// in the same order as `candidates`. Blocks until all are scored.
    ///
    /// A candidate that errors during scoring yields `f64::NAN` for that slot
    /// (the search treats NAN as "reject"); one bad candidate never poisons the
    /// batch or the worker — the worker's state is reset and it keeps serving.
    ///
    /// NOT re-entrant: results are tagged only by within-batch index, so two
    /// overlapping `score_batch` calls would steal each other's results. This is
    /// fine by construction — the single host PoB state drives the search and
    /// calls this serially (the FFI surface takes `*mut`, so the host already
    /// holds an exclusive handle). Don't call it from two host threads at once.
    pub fn score_batch(&self, candidates: &[Vec<i32>]) -> Vec<f64> {
        self.score_batch_inner(candidates, false)
            .into_iter()
            .map(|t| t[0])
            .collect()
    }

    /// Like `score_batch` but returns [score, dps, ehp] per candidate. Drives the
    /// clean-slate score fn (3 returns), so the host's Pareto beam can prune by
    /// both axes. Same ordering/NAN contract as `score_batch`.
    pub fn score_batch3(&self, candidates: &[Vec<i32>]) -> Vec<[f64; 3]> {
        self.score_batch_inner(candidates, true)
    }

    fn score_batch_inner(&self, candidates: &[Vec<i32>], triple: bool) -> Vec<[f64; 3]> {
        let n = candidates.len();
        if n == 0 {
            return Vec::new();
        }
        for (index, ids) in candidates.iter().enumerate() {
            // Send can only fail if all workers are gone; in that case we bail to
            // NAN-filled results rather than hang.
            if self
                .job_tx
                .send(Msg::Job(Job {
                    index,
                    ids: ids.clone(),
                    triple,
                    call_fn: None,
                }))
                .is_err()
            {
                return vec![[f64::NAN; 3]; n];
            }
        }
        let mut scores = vec![[f64::NAN; 3]; n];
        for _ in 0..n {
            match self.done_rx.recv() {
                Ok(d) => scores[d.index] = d.out,
                Err(_) => break, // all workers gone; leftover slots stay NAN
            }
        }
        scores
    }

    /// Run a one-off call to a named global on SOME worker, passing `args` as the
    /// id-array and returning its scalar result (NAN on failure). Used for the save
    /// step (`__pob_save_optimized`), which mutates the worker's spec — so the host
    /// must do no further scoring on the pool afterward. Blocks for the one result.
    pub fn call_named(&self, name: &str, args: &[i32]) -> f64 {
        if self
            .job_tx
            .send(Msg::Job(Job {
                index: 0,
                ids: args.to_vec(),
                triple: false,
                call_fn: Some(name.to_string()),
            }))
            .is_err()
        {
            return f64::NAN;
        }
        match self.done_rx.recv() {
            Ok(d) => d.out[0],
            Err(_) => f64::NAN,
        }
    }

    fn shutdown_internal(&self) {
        for _ in 0..self.n_workers {
            let _ = self.job_tx.send(Msg::Shutdown);
        }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.shutdown_internal();
        for h in self.workers.drain(..) {
            let _ = h.join();
        }
    }
}

/// A worker's whole life: boot its state once, report boot status, then loop
/// pulling jobs until a Shutdown arrives or the channel closes.
fn worker_main(
    lib: Arc<LuaLib>,
    bootstrap: String,
    score_fn: String,
    candidate_fn: String,
    job_rx: Arc<Mutex<Receiver<Msg>>>,
    done_tx: Sender<Done>,
    boot_tx: Sender<Result<Vec<i32>, String>>,
) {
    // SAFETY: this state is created here and never leaves this thread.
    let lua = match unsafe { Lua::new(&lib) } {
        Ok(l) => l,
        Err(e) => {
            let _ = boot_tx.send(Err(format!("create state: {e}")));
            return;
        }
    };
    if let Err(e) = unsafe { lua.run(&bootstrap) } {
        let _ = boot_tx.send(Err(format!("boot: {e}")));
        return;
    }
    // Collect the eligible candidate ids while booting (cheap, one-time). An
    // empty candidate_fn means the host doesn't need them; report an empty list.
    let candidate_ids = if candidate_fn.is_empty() {
        Vec::new()
    } else {
        match unsafe { lua.call_global_int_array(&candidate_fn) } {
            Ok(ids) => ids,
            Err(e) => {
                let _ = boot_tx.send(Err(format!("candidate enumeration: {e}")));
                return;
            }
        }
    };
    let _ = boot_tx.send(Ok(candidate_ids));
    drop(boot_tx);

    loop {
        // Hold the lock only long enough to pull one message, so workers don't
        // serialize on the queue while actually scoring.
        let msg = {
            let rx = job_rx.lock().expect("job queue mutex poisoned");
            rx.recv()
        };
        let job = match msg {
            Ok(Msg::Job(j)) => j,
            Ok(Msg::Shutdown) | Err(_) => break,
        };
        // One bad candidate must not kill the worker: the Lua side already reset
        // its stack in pop_error, so the next call starts clean (=> NAN slot).
        let out = if let Some(fn_name) = &job.call_fn {
            // One-off named call (e.g. save), scalar return in slot 0.
            match unsafe { lua.call_global_score(fn_name, &job.ids) } {
                Ok(s) => [s, f64::NAN, f64::NAN],
                Err(_) => [f64::NAN; 3],
            }
        } else if job.triple {
            unsafe { lua.call_global_score3(&score_fn, &job.ids) }.unwrap_or([f64::NAN; 3])
        } else {
            match unsafe { lua.call_global_score(&score_fn, &job.ids) } {
                Ok(s) => [s, f64::NAN, f64::NAN],
                Err(_) => [f64::NAN; 3],
            }
        };
        if done_tx
            .send(Done {
                index: job.index,
                out,
            })
            .is_err()
        {
            break; // pool gone
        }
    }
}
