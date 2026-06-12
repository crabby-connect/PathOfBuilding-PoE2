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
/// it.
struct Job {
    index: usize,
    ids: Vec<i32>,
}

/// A scored result flowing back from a worker.
struct Done {
    index: usize,
    score: f64,
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
                }))
                .is_err()
            {
                return vec![f64::NAN; n];
            }
        }
        let mut scores = vec![f64::NAN; n];
        for _ in 0..n {
            match self.done_rx.recv() {
                Ok(d) => scores[d.index] = d.score,
                Err(_) => break, // all workers gone; leftover slots stay NAN
            }
        }
        scores
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
        let score = match unsafe { lua.call_global_score(&score_fn, &job.ids) } {
            Ok(s) => s,
            // One bad candidate must not kill the worker: the Lua side already
            // reset its stack in pop_error, so the next call starts clean.
            Err(_) => f64::NAN,
        };
        if done_tx
            .send(Done {
                index: job.index,
                score,
            })
            .is_err()
        {
            break; // pool gone
        }
    }
}
