-- Production worker bootstrap. Run ONCE per LuaJIT state the Rust pool creates;
-- the state is then reused for the whole search (no per-round respawn — that was
-- the PoC's cost the design doc flagged to remove).
--
-- It reproduces what the busted spikes get for free via `.busted` (helper =
-- HeadlessWrapper.lua, directory = src, lpath = ../runtime/lua/...), but without
-- busted: set package.path/cpath, dofile HeadlessWrapper (boots the full engine
-- via Launch.lua), load the build, grab the misc calculator ONCE, and install
-- the score function the Rust hot path calls.
--
-- Rust substitutes the @@...@@ path placeholders before running this chunk
-- (config.rs render_bootstrap). The chdir to src/ is done by Rust first.
--
-- Score function contract (must match Config.score_fn, default `__pob_score`):
--   __pob_score(ids)  where ids is a 1-based array of passive node ids to ADD to
--   the current tree. Returns a scalar score. Adding a node implies adding its
--   precomputed travel path too (connectivity is never repaired — design doc §5).

-- runtime lua libs (socket, dkjson, xml, sha, ...) live one level up from src/
package.path = "@@RUNTIME_LUA@@/?.lua;@@RUNTIME_LUA@@/?/init.lua;" .. package.path
-- native C modules (lua-utf8.dll, socket.dll, ...) live in runtime/ next to lua51.dll
package.cpath = "@@RUNTIME@@/?.dll;" .. package.cpath

-- A normal lua/luajit CLI sets the global `arg` table; our embedded state does
-- not, and Main.lua indexes arg[1]. Provide an empty one.
arg = arg or {}

-- Boot the engine headlessly. HeadlessWrapper defines host callbacks as stubs,
-- dofile("Launch.lua"), runs OnInit + one OnFrame, leaving a usable `build`
-- global plus newBuild/loadBuildFromXML helpers.
dofile("HeadlessWrapper.lua")

-- Load the reference build.
do
	local f = assert(io.open("@@BUILD_XML@@", "r"), "cannot open build xml")
	local xml = f:read("*a")
	f:close()
	loadBuildFromXML(xml, "PobOptWorker")
end

local build = assert(_G.build, "no build after load")

-- Build the calculator ONCE (the #1 perf rule). calcFunc evaluates an override
-- against the prebuilt env via calcs.perform.
build.calcsTab:BuildOutput()
local calcFunc, calcBase = build.calcsTab:GetMiscCalculator()
assert(type(calcFunc) == "function", "no calcFunc")

local spec = build.spec
local mainEnv = build.calcsTab.mainEnv

-- Index nodes by id so the Rust-provided id array maps to node objects. spec.nodes
-- is keyed by id already in this engine, but build a defensive id->node map that
-- also tolerates table-keyed specs.
local nodeById = {}
for k, node in pairs(spec.nodes) do
	if type(node) == "table" then
		local id = node.id or (type(k) == "number" and k) or nil
		if id ~= nil then nodeById[id] = node end
	end
end

-- Build the addNodes set for a candidate: each requested node PLUS its precomputed
-- travel path (node.path). Mirrors the spike's pathSetFor and the design doc's
-- "add target + its node.path together; never repair" rule.
local function addSetFor(ids)
	local set = {}
	for i = 1, #ids do
		local node = nodeById[ids[i]]
		if node then
			if node.path then
				for _, n in ipairs(node.path) do set[n] = true end
			end
			set[node] = true
		end
		-- An unknown id is skipped rather than fatal: the search may probe ids the
		-- worker's (copy of the) tree can't reach; such a candidate just scores as
		-- the base tree, which the search will not prefer.
	end
	return set
end

-- Weighted scalar over REAL calc outputs. Weights come from the host via globals
-- the search can set before the run; default 1/1 (DPS + EHP), matching the spike.
local W_DPS = tonumber(_G.__pob_w_dps) or 1.0
local W_EHP = tonumber(_G.__pob_w_ehp) or 1.0
local function scoreOf(out)
	local dps = out.FullDPS or out.TotalDPS or 0
	local ehp = out.TotalEHP or 0
	return W_DPS * dps + W_EHP * ehp
end

-- THE HOT PATH. One real calcFunc({addNodes=...}, true) pass = one calcs.perform
-- + FullDPS roll-up. Empty ids => score the base tree (a valid query).
function __pob_score(ids)
	if #ids == 0 then
		return scoreOf(calcBase)
	end
	local out = calcFunc({ addNodes = addSetFor(ids) }, true)
	return scoreOf(out)
end

-- Eligibility mirrors PowerBuilder (src/Classes/CalcsTab.lua) and the spike:
-- reachable, un-allocated, has mods, not item-granted, on the main tree. The
-- host can call this once to seed its candidate set; every worker computes the
-- same list (identical tree), so any worker's answer is authoritative.
local function isEligible(node)
	return node
		and not node.alloc
		and node.modKey ~= nil and node.modKey ~= ""
		and not (mainEnv.grantedPassives and mainEnv.grantedPassives[node.id])
		and node.pathDist ~= nil and node.pathDist < 1000
		and not node.ascendancyName
end

-- Returns a 1-based array of eligible candidate node ids. Used by the host to
-- enumerate what to score, and by the Rust smoke test to exercise real additions.
function __pob_candidate_ids()
	local ids = {}
	for id, node in pairs(nodeById) do
		if isEligible(node) then ids[#ids + 1] = id end
	end
	return ids
end

-- Optional sanity hooks the Rust side / tests can call.
__pob_base = scoreOf(calcBase)
function __pob_base_score() return __pob_base end
