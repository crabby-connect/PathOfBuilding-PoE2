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

-- Weighted scalar over REAL calc outputs. Weights are injected by Rust from the
-- pool config (config.rs render_bootstrap, keys w_dps/w_ehp), defaulting to 1/1.
-- Raising W_DPS relative to W_EHP pulls the search toward damage.
local W_DPS = tonumber("@@W_DPS@@") or 1.0
local W_EHP = tonumber("@@W_EHP@@") or 1.0
-- Quadratic regression-penalty strength (config key penalty_k). Used only by the
-- Rust beam (exported via __pob_graph_export), NOT by the per-candidate scorer.
local PENALTY_K = tonumber("@@PENALTY_K@@") or 5.0
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

-- ---- CLEAN-SLATE score path (mirrors SpikeCleanSlateBeam_spec.lua) ----------------
-- The beam optimizer does NOT score "current tree + added nodes"; it scores an ABSOLUTE
-- candidate P-node tree grown from the class start, via the clean-slate override:
--     addNodes = candidateTree(full id set incl. start),  removeNodes = currentAlloc-start
-- so the evaluated node set == the candidate exactly (CalcSetup.lua:718-749). To make the
-- pool reproduce the spike's numbers (not the add-semantics above), __pob_score_cleanslate
-- takes the FULL candidate id set, expands to node objects (NO node.path — the host already
-- includes the travel nodes in the id set the same way the spike's beam does), and scores it
-- with removeNodes pinned to the build's original allocation.
local startId = spec.curClass and spec.curClass.startNodeId
-- removeCurrent: everything currently allocated EXCEPT the class start and ascendancy nodes,
-- built ONCE (the original build's allocation never changes during the search).
local removeCurrent = {}
for id, node in pairs(spec.allocNodes or {}) do
	if id ~= startId and not node.ascendancyName then
		removeCurrent[node] = true
	end
end

-- SMOOTH GROWTH score: 100*(W_DPS*dps/refDps + W_EHP*ehp/refEhp), monotonic in
-- both axes so the beam can climb from the start-only tree (every added node that
-- raises dps or ehp raises the score). refs = the ORIGINAL build's values
-- (calcBase = the build as currently allocated, no overrides). This is the GROWTH
-- signal ONLY; the DPS/EHP regression penalty vs. the original is applied by the
-- Rust beam to FULL-BUDGET trees (see beam.rs penalized_score) — it must NOT live
-- here, because every partial tree is below the full original on both axes and a
-- penalty/reject here would stall growth at 0 points.
local refDps = math.max(calcBase.FullDPS or calcBase.TotalDPS or 0, 1)
local refEhp = math.max(calcBase.TotalEHP or 0, 1)
-- Returns (growthScore, dps, ehp): the beam needs dps/ehp both to grow and to
-- compute the final penalty. Always finite (no NaN gate) so the beam never stalls.
local function scoreNorm(out)
	local dps = out.FullDPS or out.TotalDPS or 0
	local ehp = out.TotalEHP or 0
	return 100 * (W_DPS * dps / refDps + W_EHP * ehp / refEhp), dps, ehp
end

-- Clean-slate hot path. ids = full candidate node-id set (incl. class start + travel nodes).
-- Empty ids => the start-only tree's score, which the beam uses as its from-scratch baseline.
-- Returns three numbers (score, dps, ehp) consumed by call_global_score3.
function __pob_score_cleanslate(ids)
	local addSet = {}
	for i = 1, #ids do
		local node = nodeById[ids[i]]
		if node then addSet[node] = true end
	end
	local out = calcFunc({ addNodes = addSet, removeNodes = removeCurrent }, true)
	return scoreNorm(out)
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

-- ---- GRAPH EXPORT (for the native Rust beam harness) -----------------------------
-- The beam search must run where the tree topology lives. The busted spike runs it in
-- Lua; the native Rust harness (examples/beam_ab.rs) has no graph, so we export it once.
-- Returns a FLAT array of integers the Rust side parses (call_global_int_array reads a
-- numeric array). Layout:
--   [ startId, budgetP, nNodes,
--     repeated nNodes times: id, typeCode, nLinks, link1, link2, ... linkK ]
-- budgetP = the normal passive points the CURRENT build spends (spec:CountAllocNodes),
-- so the native harness can budget the beam to the real build instead of a guessed cap.
-- typeCode: 0=other, 1=Normal, 2=Notable, 3=Keystone (matches the beam's targeting).
-- Only MAIN-tree, non-ascendancy, non-mastery nodes are exported (the beam's universe);
-- links are filtered to other exported nodes so the Rust adjacency is self-contained.
local function typeCode(node)
	local t = node.type
	if t == "Keystone" then return 3
	elseif t == "Notable" then return 2
	elseif t == "Normal" then return 1
	else return 0 end
end
local function exportable(node)
	return node and node.id
		and node.type ~= "ClassStart" and node.type ~= "AscendClassStart"
		and not node.ascendancyName
		and node.type ~= "Mastery"
end
function __pob_graph_export()
	local budgetP = (spec.CountAllocNodes and spec:CountAllocNodes()) or 0
	local flat = { startId or 0, budgetP, 0 }
	-- mark exportable ids so we can filter links to within the exported set
	local inSet = {}
	for id, node in pairs(nodeById) do
		if exportable(node) then inSet[id] = true end
	end
	-- the class start must be present as a node even if exportable() excludes it (it's a
	-- ClassStart) so the beam has a root with its real adjacency.
	local startNode = startId and nodeById[startId]
	local function emitNode(id, node)
		flat[#flat+1] = id
		flat[#flat+1] = typeCode(node)
		local links = {}
		for _, other in ipairs(node.linked or {}) do
			if other.id and (inSet[other.id] or other.id == startId) and other.id ~= id then
				links[#links+1] = other.id
			end
		end
		flat[#flat+1] = #links
		for _, lid in ipairs(links) do flat[#flat+1] = lid end
	end
	local n = 0
	if startNode then emitNode(startId, startNode); n = n + 1 end
	for id, node in pairs(nodeById) do
		if inSet[id] and id ~= startId then emitNode(id, node); n = n + 1 end
	end
	flat[3] = n
	-- TRAILER (appended after the node data; backward-compatible — older parsers
	-- read exactly `n` nodes and stop): the constants the Rust beam needs to apply
	-- the DPS/EHP regression penalty to FULL-BUDGET trees. refDps/refEhp are the
	-- ORIGINAL build's values (the floor to protect); W_DPS/W_EHP/PENALTY_K mirror
	-- the score weights. Sent as ints (×1000 to keep a few decimals of the small
	-- weight/K values; refs are large so truncation is negligible). Order:
	--   [ refDps*1000, refEhp*1000, W_DPS*1000, W_EHP*1000, PENALTY_K*1000 ]
	flat[#flat+1] = math.floor(refDps * 1000 + 0.5)
	flat[#flat+1] = math.floor(refEhp * 1000 + 0.5)
	flat[#flat+1] = math.floor(W_DPS * 1000 + 0.5)
	flat[#flat+1] = math.floor(W_EHP * 1000 + 0.5)
	flat[#flat+1] = math.floor(PENALTY_K * 1000 + 0.5)
	return flat
end

-- ---- SAVE the optimized tree to an importable PoB XML ----------------------------
-- Mirrors SpikeCleanSlateBeam_spec.lua's save block: apply the winning main-tree id
-- set to the live spec (keeping the build's existing ascendancy), then build:SaveDB so
-- the file opens directly in PoB. This MUTATES the worker's spec, so the host must call
-- it exactly once, on a dedicated throwaway query AFTER all scoring is done (any further
-- __pob_score_cleanslate call on this worker would now be relative to the mutated spec).
--
-- Contract: the host packs [ n_ids, id1..idn, pathChars... ] into the id array, where the
-- trailing chars after the ids are the BYTES of the output path (so we avoid a second FFI
-- entry). __pob_save_optimized(packed) unpacks, saves, and returns 1.0 on success / 0.0 on
-- failure (a number, so it flows back through call_global_score3's first return slot).
function __pob_save_optimized(packed)
	local n = packed[1] or 0
	local hashList = {}
	for i = 1, n do
		local id = packed[1 + i]
		local node = nodeById[id]
		if node and not node.ascendancyName then table.insert(hashList, id) end
	end
	-- keep the build's existing ascendancy allocation (ascendancy is fixed, not searched)
	for id, node in pairs(spec.allocNodes or {}) do
		if node.ascendancyName then table.insert(hashList, id) end
	end
	-- decode the trailing path bytes
	local pathBytes = {}
	for i = 2 + n, #packed do pathBytes[#pathBytes+1] = string.char(packed[i]) end
	local outPath = table.concat(pathBytes)
	if outPath == "" then return 0.0 end

	local ok = pcall(function()
		spec:ImportFromNodeList(nil, spec.curClassId, spec.curAscendClassId,
			spec.curSecondaryAscendClassId or 0, hashList, {}, {}, {})
		spec:BuildAllDependsAndPaths()
		local xmlText = build:SaveDB("optimized")
		assert(xmlText, "SaveDB returned nil")
		local f = assert(io.open(outPath, "w+"), "cannot open " .. outPath)
		f:write(xmlText); f:close()
	end)
	return ok and 1.0 or 0.0
end

-- Optional sanity hooks the Rust side / tests can call.
__pob_base = scoreOf(calcBase)
function __pob_base_score() return __pob_base end
