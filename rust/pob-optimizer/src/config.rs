//! Pool configuration, parsed from the tiny `key=value` string the host passes
//! to pob_opt_create. No JSON crate — the surface is small and Lua emits it with
//! plain concatenation.
//!
//! Keys (all required unless noted):
//!   lua_dll       absolute path to runtime/lua51.dll
//!   src_dir       absolute path to the engine's src/ (cwd workers run from)
//!   runtime_dir   absolute path to runtime/ (native .dll modules: lua-utf8, ...)
//!   runtime_lua   absolute path to runtime/lua/ (pure-lua modules: dkjson, ...)
//!   build_xml     absolute path to the build .xml every worker loads
//!   bootstrap     absolute path to worker_bootstrap.lua (the boot template)
//!   score_fn      name of the global score function the bootstrap installs
//!   candidate_fn  name of the global candidate-id enumerator (optional; empty
//!                 disables candidate enumeration)
//!   workers       worker-thread count (optional; default = available cores)
//!
//! Path values are used verbatim. They may contain '=' (Windows paths don't, but
//! be safe): we split on the FIRST '=' only. Forward or back slashes both work;
//! the bootstrap template normalizes to forward slashes for Lua string literals.

pub struct Config {
    pub lua_dll: String,
    pub src_dir: String,
    pub runtime_dir: String,
    pub runtime_lua: String,
    pub build_xml: String,
    pub bootstrap_path: String,
    pub score_fn: String,
    pub candidate_fn: String,
    pub workers: usize,
    /// Score weights injected into the worker (default 1.0 each): the worker's
    /// normalized score is 100*(w_dps*dps/refDps + w_ehp*ehp/refEhp). Raising
    /// w_dps relative to w_ehp pulls the search toward damage.
    pub w_dps: f64,
    pub w_ehp: f64,
}

impl Config {
    pub fn parse(s: &str) -> Result<Self, String> {
        let mut lua_dll = None;
        let mut src_dir = None;
        let mut runtime_dir = None;
        let mut runtime_lua = None;
        let mut build_xml = None;
        let mut bootstrap_path = None;
        let mut score_fn = None;
        let mut candidate_fn: Option<String> = None;
        let mut workers: Option<usize> = None;
        let mut w_dps: f64 = 1.0;
        let mut w_ehp: f64 = 1.0;

        for (lineno, raw) in s.lines().enumerate() {
            let line = raw.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let (key, val) = line
                .split_once('=')
                .ok_or_else(|| format!("config line {}: no '=' in '{}'", lineno + 1, line))?;
            let key = key.trim();
            let val = val.trim().to_string();
            match key {
                "lua_dll" => lua_dll = Some(val),
                "src_dir" => src_dir = Some(val),
                "runtime_dir" => runtime_dir = Some(val),
                "runtime_lua" => runtime_lua = Some(val),
                "build_xml" => build_xml = Some(val),
                "bootstrap" => bootstrap_path = Some(val),
                "score_fn" => score_fn = Some(val),
                "candidate_fn" => candidate_fn = Some(val),
                "workers" => {
                    let w: usize = val
                        .parse()
                        .map_err(|_| format!("config: workers='{val}' is not a number"))?;
                    workers = Some(w);
                }
                "w_dps" => {
                    w_dps = val
                        .parse()
                        .map_err(|_| format!("config: w_dps='{val}' is not a number"))?;
                }
                "w_ehp" => {
                    w_ehp = val
                        .parse()
                        .map_err(|_| format!("config: w_ehp='{val}' is not a number"))?;
                }
                other => return Err(format!("config: unknown key '{other}'")),
            }
        }

        let req = |o: Option<String>, name: &str| {
            o.ok_or_else(|| format!("config: missing required key '{name}'"))
        };

        let workers = match workers {
            Some(0) | None => std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(4),
            Some(w) => w,
        };

        Ok(Config {
            lua_dll: req(lua_dll, "lua_dll")?,
            src_dir: req(src_dir, "src_dir")?,
            runtime_dir: req(runtime_dir, "runtime_dir")?,
            runtime_lua: req(runtime_lua, "runtime_lua")?,
            build_xml: req(build_xml, "build_xml")?,
            bootstrap_path: req(bootstrap_path, "bootstrap")?,
            score_fn: req(score_fn, "score_fn")?,
            candidate_fn: candidate_fn.unwrap_or_default(),
            workers,
            w_dps,
            w_ehp,
        })
    }

    /// Read the bootstrap template and substitute the path placeholders, exactly
    /// as the PoC driver did (@@RUNTIME_LUA@@, @@RUNTIME@@, @@BUILD_XML@@), so the
    /// same worker_bootstrap.lua works for both. Slashes are normalized to '/'.
    pub fn render_bootstrap(&self) -> Result<String, String> {
        let template = std::fs::read_to_string(&self.bootstrap_path)
            .map_err(|e| format!("read bootstrap '{}': {e}", self.bootstrap_path))?;
        let fwd = |p: &str| p.replace('\\', "/");
        Ok(template
            .replace("@@RUNTIME_LUA@@", &fwd(&self.runtime_lua))
            .replace("@@RUNTIME@@", &fwd(&self.runtime_dir))
            .replace("@@BUILD_XML@@", &fwd(&self.build_xml))
            // Score weights as Lua number literals (e.g. "3" or "1.5").
            .replace("@@W_DPS@@", &format!("{}", self.w_dps))
            .replace("@@W_EHP@@", &format!("{}", self.w_ehp)))
    }
}
