-- Path of Building
--
-- Module: OptimizerPool
-- Lua-side wrapper over the Rust parallel tree-optimizer cdylib (pob_optimizer).
--
-- This is the host integration of rust/pob-optimizer: it loads the cdylib via
-- LuaJIT FFI, boots a PERSISTENT pool of worker LuaJIT states (each a full
-- headless calc engine), and exposes `scoreBatch(candidates)` so the search loop
-- can score many node-id sets across CPU cores. See docs/rust-offload-poc.md and
-- docs/tree-optimizer-design.md.
--
-- The calc engine stays canonical Lua; this only drives parallel `perform` calls.
-- NOTE: this is the FIRST use of LuaJIT FFI in the codebase. The cdylib must be
-- built (cargo build --release in rust/pob-optimizer) and pob_optimizer.dll must
-- be reachable on the loader path (alongside the other runtime DLLs).

local ffi = require("ffi")

ffi.cdef[[
	typedef struct PobOptPool PobOptPool;
	PobOptPool* pob_opt_create(const char* config);
	int  pob_opt_score_batch(PobOptPool*, const int32_t* ids,
	                         const int32_t* lengths, int32_t n, double* out);
	int  pob_opt_score_batch3(PobOptPool*, const int32_t* ids,
	                          const int32_t* lengths, int32_t n, double* out);
	int  pob_opt_candidate_ids(PobOptPool*, int32_t* out, int32_t cap);
	int  pob_opt_worker_count(PobOptPool*);
	void pob_opt_destroy(PobOptPool*);
	const char* pob_opt_last_error(void);
]]

local OptimizerPool = { }
OptimizerPool.__index = OptimizerPool

-- Load the cdylib once per process. ffi.load resolves "pob_optimizer" to
-- pob_optimizer.dll via the OS loader path (Windows) / lib search path.
local C
local function lib()
	if not C then
		C = ffi.load("pob_optimizer")
	end
	return C
end

local function lastError()
	local p = lib().pob_opt_last_error()
	if p == nil then return "(no error)" end
	return ffi.string(p)
end

-- Build the newline-delimited key=value config the cdylib parses (NOT JSON; see
-- rust/pob-optimizer/src/config.rs). All paths are absolute. `opts` overrides:
--   runtimeDir  (default GetRuntimePath()), srcDir, buildXml (required),
--   bootstrap, scoreFn, candidateFn, workers.
local function buildConfig(opts)
	assert(opts and opts.buildXml, "OptimizerPool: opts.buildXml is required")
	local runtimeDir = opts.runtimeDir or GetRuntimePath()
	assert(runtimeDir and runtimeDir ~= "", "OptimizerPool: runtimeDir unknown (set opts.runtimeDir)")
	local srcDir = opts.srcDir or (runtimeDir .. "/../src")
	local bootstrap = opts.bootstrap or (runtimeDir .. "/../rust/pob-optimizer/worker_bootstrap.lua")
	local lines = {
		"lua_dll=" .. runtimeDir .. "/lua51.dll",
		"src_dir=" .. srcDir,
		"runtime_dir=" .. runtimeDir,
		"runtime_lua=" .. runtimeDir .. "/lua",
		"build_xml=" .. opts.buildXml,
		"bootstrap=" .. bootstrap,
		"score_fn=" .. (opts.scoreFn or "__pob_score"),
		"candidate_fn=" .. (opts.candidateFn or "__pob_candidate_ids"),
		"workers=" .. tostring(opts.workers or 0), -- 0 => library default (cores)
	}
	return table.concat(lines, "\n") .. "\n"
end

-- Boot a pool. Returns the OptimizerPool instance, or nil + error string.
function OptimizerPool.new(opts)
	local ok, config = pcall(buildConfig, opts)
	if not ok then return nil, config end
	local handle = lib().pob_opt_create(config)
	if handle == nil then
		return nil, "pob_opt_create failed: " .. lastError()
	end
	local self = setmetatable({ handle = handle }, OptimizerPool)
	-- Free the native pool when this object is collected (joins worker threads).
	-- An explicit :destroy() is preferred; this is the safety net.
	ffi.gc(handle, function(h) lib().pob_opt_destroy(h) end)
	return self
end

function OptimizerPool:workerCount()
	return lib().pob_opt_worker_count(self.handle)
end

-- The eligible candidate node ids the workers computed at boot, as a Lua array.
function OptimizerPool:candidateIds()
	local total = lib().pob_opt_candidate_ids(self.handle, nil, 0)
	if total <= 0 then return { } end
	local buf = ffi.new("int32_t[?]", total)
	lib().pob_opt_candidate_ids(self.handle, buf, total)
	local ids = { }
	for i = 0, total - 1 do ids[i + 1] = buf[i] end
	return ids
end

-- Score a batch of candidate node-id sets. `candidates` is an array of arrays of
-- integer node ids (each inner array is one candidate; empty = base tree).
-- Returns a parallel array of scores (number), NaN for any that failed to score.
function OptimizerPool:scoreBatch(candidates)
	local n = #candidates
	if n == 0 then return { } end

	-- Flatten to the (ids, lengths) layout the cdylib expects.
	local total = 0
	for i = 1, n do total = total + #candidates[i] end
	local idsBuf = ffi.new("int32_t[?]", math.max(total, 1))
	local lenBuf = ffi.new("int32_t[?]", n)
	local cursor = 0
	for i = 1, n do
		local c = candidates[i]
		lenBuf[i - 1] = #c
		for j = 1, #c do
			idsBuf[cursor] = c[j]
			cursor = cursor + 1
		end
	end

	local outBuf = ffi.new("double[?]", n)
	local rc = lib().pob_opt_score_batch(self.handle, idsBuf, lenBuf, n, outBuf)
	if rc ~= 0 then
		error("OptimizerPool:scoreBatch failed (rc=" .. rc .. "): " .. lastError())
	end
	local scores = { }
	for i = 1, n do scores[i] = outBuf[i - 1] end
	return scores
end

-- Score a batch of candidate node-id sets, returning (score, dps, ehp) per
-- candidate. `candidates` is an array of arrays of integer node ids. Returns a
-- parallel array of { score, dps, ehp } tables; NaN fields for any that failed.
-- Drives the clean-slate worker fn (__pob_score_cleanslate), so the host's beam
-- can prune by both axes (Pareto). Same flat (ids,lengths) layout as scoreBatch.
function OptimizerPool:scoreBatch3(candidates)
	local n = #candidates
	if n == 0 then return { } end

	local total = 0
	for i = 1, n do total = total + #candidates[i] end
	local idsBuf = ffi.new("int32_t[?]", math.max(total, 1))
	local lenBuf = ffi.new("int32_t[?]", n)
	local cursor = 0
	for i = 1, n do
		local c = candidates[i]
		lenBuf[i - 1] = #c
		for j = 1, #c do
			idsBuf[cursor] = c[j]
			cursor = cursor + 1
		end
	end

	-- 3 doubles per candidate: (score, dps, ehp) interleaved.
	local outBuf = ffi.new("double[?]", n * 3)
	local rc = lib().pob_opt_score_batch3(self.handle, idsBuf, lenBuf, n, outBuf)
	if rc ~= 0 then
		error("OptimizerPool:scoreBatch3 failed (rc=" .. rc .. "): " .. lastError())
	end
	local out = { }
	for i = 1, n do
		local b = (i - 1) * 3
		out[i] = { score = outBuf[b], dps = outBuf[b + 1], ehp = outBuf[b + 2] }
	end
	return out
end

-- Explicitly tear down the pool (joins all worker threads). Idempotent.
function OptimizerPool:destroy()
	if self.handle ~= nil then
		ffi.gc(self.handle, nil) -- cancel the finalizer; we free now
		lib().pob_opt_destroy(self.handle)
		self.handle = nil
	end
end

return OptimizerPool
