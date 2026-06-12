-- PoC worker bootstrap, run once per LuaJIT state created by the Rust driver.
--
-- It reproduces exactly what the busted spikes get for free via `.busted`
-- (helper = HeadlessWrapper.lua, directory = src, lpath = ../runtime/lua/...),
-- but without busted: set package.path, chdir is done by Rust before this runs,
-- dofile HeadlessWrapper (which boots the full engine via Launch.lua), load the
-- reference build, grab the misc calculator ONCE, and expose __poc_eval().
--
-- __poc_eval() runs ONE real `calcFunc(override)` pass — i.e. one `calcs.perform`
-- plus FullDPS roll-up, the ~8ms bottleneck the whole effort is about. It returns
-- a scalar score so the Rust hot loop can call it with a tiny FFI surface.
--
-- The Rust side substitutes @@PATHS@@ before running this chunk.

-- runtime lua libs (socket, dkjson, xml, sha, ...) live one level up from src/
package.path = "@@RUNTIME_LUA@@/?.lua;@@RUNTIME_LUA@@/?/init.lua;" .. package.path
-- native C modules (lua-utf8.dll, socket.dll, ...) live in runtime/ next to
-- lua51.dll; the require C-loader resolves them via cpath ?.dll.
package.cpath = "@@RUNTIME@@/?.dll;" .. package.cpath

-- A normal lua/luajit CLI sets the global `arg` table; our embedded state does
-- not, and Main.lua:64 indexes arg[1]. Provide an empty one (no import link).
arg = arg or {}

-- Boot the engine headlessly. HeadlessWrapper defines all host callbacks as
-- stubs, dofile("Launch.lua"), runs OnInit + one OnFrame, and leaves a usable
-- `build` global plus newBuild/loadBuildFromXML helpers.
dofile("HeadlessWrapper.lua")

-- Load the reference build (same one the design doc/spikes use).
do
	local f = assert(io.open("@@BUILD_XML@@", "r"), "cannot open build xml")
	local xml = f:read("*a")
	f:close()
	loadBuildFromXML(xml, "PoCWorker")
end

local build = _G.build
assert(build, "no build after load")

-- Build the calculator ONCE (the #1 perf rule). calcFunc is the closure that
-- evaluates an override against the prebuilt env via calcs.perform.
build.calcsTab:BuildOutput()
local calcFunc, calcBase = build.calcsTab:GetMiscCalculator()
assert(type(calcFunc) == "function", "no calcFunc")

local spec = build.spec

-- Precompute an eligibility list of single reachable un-allocated nodes so each
-- eval perturbs the tree (mirrors the depth-1 sweep in SpikeTreeOptimizer_spec).
-- We need real per-call work, not a no-op, so the parallel timing is honest.
local mainEnv = build.calcsTab.mainEnv
local function pathSetFor(node)
	local set = {}
	if node.path then for _, n in ipairs(node.path) do set[n] = true end end
	set[node] = true
	return set
end
local candidates = {}
for _, node in pairs(spec.nodes) do
	if node and not node.alloc and node.modKey ~= nil and node.modKey ~= ""
		and not (mainEnv.grantedPassives and mainEnv.grantedPassives[node.id])
		and node.pathDist ~= nil and node.pathDist < 1000
		and not node.ascendancyName then
		candidates[#candidates + 1] = node
	end
end
assert(#candidates > 0, "no eligible candidate nodes")

local function scoreOf(out)
	local dps = out.FullDPS or out.TotalDPS or 0
	local ehp = out.TotalEHP or 0
	return dps + ehp
end

-- Round-robin cursor so successive calls hit different candidates -> different
-- node mods -> a real, non-cached perform each time (the realistic worst case).
local cursor = 0
function __poc_eval()
	cursor = cursor + 1
	if cursor > #candidates then cursor = 1 end
	local out = calcFunc({ addNodes = pathSetFor(candidates[cursor]) }, true)
	return scoreOf(out)
end

-- Expose a baseline score + candidate count for the Rust side to sanity-print.
__poc_base = scoreOf(calcBase)
__poc_ncand = #candidates
function __poc_base_score() return __poc_base end
function __poc_candidate_count() return __poc_ncand end
