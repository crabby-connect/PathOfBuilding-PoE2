//! Parallel tree-optimizer PoC driver.
//!
//! Proves the single load-bearing fact for "offload the search to Rust": N
//! independent LuaJIT states, each booting PoB's headless calc engine, run
//! `calcs.perform` on N threads with near-linear speedup on the ~8ms-per-call
//! bottleneck.
//!
//! Usage (from repo root):
//!   cargo run --release --manifest-path rust/pob-optimizer-poc/Cargo.toml -- [reps] [threads]
//!
//! reps    = perform() evaluations PER worker        (default 200)
//! threads = number of parallel worker states/cores  (default = available cores)
//!
//! It runs a 1-worker baseline, then an N-worker parallel batch of the SAME
//! total work, and prints wall time + speedup. Each worker chdir's into `src/`
//! (the engine assumes that cwd, as `.busted` does) and dofiles HeadlessWrapper.

mod lua;

use lua::{Lua, LuaLib};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::Instant;

const DEFAULT_REPS: usize = 200;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let reps: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(DEFAULT_REPS);
    let threads: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4));

    // Resolve repo root from this crate's location: rust/pob-optimizer-poc -> ../..
    let repo_root = locate_repo_root().unwrap_or_else(|| {
        eprintln!("could not locate repo root (expected a `src/` and `runtime/` next to it)");
        std::process::exit(2);
    });
    let src_dir = repo_root.join("src");
    let runtime_lua = repo_root.join("runtime").join("lua");
    let dll = repo_root.join("runtime").join("lua51.dll");
    let build_xml = src_dir.join("Builds").join("mymonk.xml");
    let bootstrap = repo_root
        .join("rust")
        .join("pob-optimizer-poc")
        .join("worker_bootstrap.lua");

    for (label, p) in [
        ("lua51.dll", &dll),
        ("src dir", &src_dir),
        ("build xml", &build_xml),
        ("bootstrap", &bootstrap),
    ] {
        if !p.exists() {
            eprintln!("missing {label}: {}", p.display());
            std::process::exit(2);
        }
    }

    let bootstrap_src = std::fs::read_to_string(&bootstrap).expect("read bootstrap");
    let runtime_dir = repo_root.join("runtime");
    let chunk = bootstrap_src
        .replace("@@RUNTIME_LUA@@", &lua_path(&runtime_lua))
        .replace("@@RUNTIME@@", &lua_path(&runtime_dir))
        .replace("@@BUILD_XML@@", &lua_path(&build_xml));

    println!("PoC: parallel LuaJIT calc workers");
    println!("  repo root : {}", repo_root.display());
    println!("  reps/work : {reps}");
    println!("  threads   : {threads}");
    println!();

    let lib = Arc::new(unsafe {
        LuaLib::load(dll.to_str().expect("dll path utf8")).unwrap_or_else(|e| {
            eprintln!("failed to load LuaJIT: {e}");
            std::process::exit(1);
        })
    });

    // The engine assumes cwd == src/. All worker states share one process cwd,
    // so set it once up front; bootstrapping does only relative dofile/io.open.
    std::env::set_current_dir(&src_dir).expect("chdir src");

    // ---- Baseline: a single worker doing `reps` evals ---------------------
    println!("[baseline] booting 1 worker + {reps} evals ...");
    let base = run_worker_timed(&lib, &chunk, reps).unwrap_or_else(|e| {
        eprintln!("baseline worker failed: {e}");
        std::process::exit(1);
    });
    let serial_eval = base.evals;
    let per_call_ms = serial_eval.as_secs_f64() * 1000.0 / reps as f64;
    println!(
        "[baseline] base score {:.1}, {} candidates, checksum {:.1}",
        base.base, base.ncand, base.checksum
    );
    println!("[baseline] boot {:.3}s (one-time, amortized in production)", base.boot.as_secs_f64());
    println!(
        "[baseline] {reps} evals in {:.3}s  ({per_call_ms:.2} ms/call steady-state)\n",
        serial_eval.as_secs_f64()
    );

    // ---- Parallel: N workers, each `reps` evals, concurrently -------------
    println!("[parallel] booting {threads} workers x {reps} evals ...");
    let mut handles = Vec::new();
    for _ in 0..threads {
        let lib = Arc::clone(&lib);
        let chunk = chunk.clone();
        handles.push(std::thread::spawn(move || run_worker_timed(&lib, &chunk, reps)));
    }
    let mut ok = 0usize;
    let mut max_eval = std::time::Duration::ZERO;
    let mut max_boot = std::time::Duration::ZERO;
    for h in handles {
        match h.join().expect("worker thread panicked") {
            Ok(w) => {
                ok += 1;
                max_eval = max_eval.max(w.evals);
                max_boot = max_boot.max(w.boot);
            }
            Err(e) => eprintln!("[parallel] worker failed: {e}"),
        }
    }
    let total_calls = reps * threads;

    println!(
        "[parallel] {ok}/{threads} workers ok, {total_calls} total evals",

    );
    println!(
        "[parallel] slowest worker: boot {:.3}s, eval loop {:.3}s",
        max_boot.as_secs_f64(),
        max_eval.as_secs_f64()
    );
    println!(
        "[parallel] effective {:.2} ms/call (eval loop)\n",
        max_eval.as_secs_f64() * 1000.0 / reps as f64
    );

    // Production-relevant speedup: the EVAL LOOP only (boot is one-time and
    // amortized over a real search's tens of thousands of calls). Compare the
    // serial eval time x threads against the parallel eval wall time (= the
    // slowest worker's eval loop, since they run concurrently).
    let serial_equiv = serial_eval.as_secs_f64() * threads as f64;
    let speedup = serial_equiv / max_eval.as_secs_f64();
    println!("=== RESULT (eval loop, boot excluded) ===");
    println!("  serial-equivalent {threads}x baseline : {:.2}s", serial_equiv);
    println!("  parallel eval wall time             : {:.2}s", max_eval.as_secs_f64());
    println!(
        "  SPEEDUP                              : {speedup:.2}x  (ideal {threads}.00x, {:.0}% efficiency)",
        speedup / threads as f64 * 100.0
    );
}

/// Boot a fresh worker state, run the bootstrap, then `reps` evals. Returns
/// (base score, candidate count, summed score checksum) for sanity output.
struct WorkerResult {
    base: f64,
    ncand: i64,
    checksum: f64,
    boot: std::time::Duration,
    evals: std::time::Duration,
}

/// Boot a worker and run `reps` evals, separately timing the one-time boot
/// (state create + engine bootstrap + tree/mod parse) from the steady-state
/// eval loop. A real search amortizes boot over tens of thousands of evals, so
/// the eval-loop speedup is the number that predicts production scaling.
fn run_worker_timed(lib: &LuaLib, chunk: &str, reps: usize) -> Result<WorkerResult, String> {
    unsafe {
        let t_boot = Instant::now();
        let lua = Lua::new(lib)?;
        lua.run(chunk)?;
        let base = lua.call_global_number("__poc_base_score")?;
        let ncand = lua.call_global_number("__poc_candidate_count")? as i64;
        let boot = t_boot.elapsed();

        let t_eval = Instant::now();
        let mut checksum = 0.0f64;
        for _ in 0..reps {
            checksum += lua.call_global_number("__poc_eval")?;
        }
        let evals = t_eval.elapsed();
        Ok(WorkerResult { base, ncand, checksum, boot, evals })
    }
}

/// Format a filesystem path for embedding in a Lua string literal (forward
/// slashes; Lua on Windows accepts them in io.open/package.path).
fn lua_path(p: &Path) -> String {
    p.to_string_lossy().replace('\\', "/")
}

fn locate_repo_root() -> Option<PathBuf> {
    // Prefer compile-time crate dir: .../rust/pob-optimizer-poc -> up two.
    let from_manifest = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(|p| p.parent())
        .map(Path::to_path_buf);
    if let Some(root) = from_manifest {
        if root.join("src").join("HeadlessWrapper.lua").exists() {
            return Some(root);
        }
    }
    // Fallback: walk up from cwd.
    let mut cur = std::env::current_dir().ok()?;
    loop {
        if cur.join("src").join("HeadlessWrapper.lua").exists()
            && cur.join("runtime").join("lua51.dll").exists()
        {
            return Some(cur);
        }
        if !cur.pop() {
            return None;
        }
    }
}
