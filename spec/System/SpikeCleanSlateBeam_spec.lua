-- Spike #2: clean-slate beam reallocation (see docs/tree-optimizer-design.md §1, §3).
--
-- Problem (decided 2026-06-03): fixed class+ascendancy, fixed gear/skills/config. Discard
-- the current passive allocation and grow the best connected P-node tree from the class
-- start, where P = the number of normal passive points the build currently uses.
--
-- Clean-slate eval trick (verified in CalcSetup.lua:718-749): the calc's final node set is
--   override.addNodes  UNION  (override.spec.allocNodes  MINUS  override.removeNodes)
-- so passing addNodes = candidateTree and removeNodes = currentAlloc (minus start) makes
-- the evaluated set == candidateTree exactly, WITHOUT mutating the real build or cloning.
--
-- Search: beam search that grows the tree from the class start (connectivity automatic),
-- wrapped in an ACCUMULATING-GREEDY ELIMINATION DIET.
--   Beam:  state = allocated id-set (incl class start); each step expands by single adjacent
--          nodes AND notable/keystone path-jumps; score = real calcFunc; keep top-N. Depth = P.
--   Diet:  round 0 = unconstrained best, then TWO phases (each accumulating-greedy, stop on
--          `patience` failed rounds; bans carry from phase 1 into phase 2):
--            PHASE 1 (marginal): ban the LOWEST-marginal droppable node (weakest link).
--            PHASE 2 (longpath): ban the DEEPEST droppable node (longest path from start,
--                     tie-break lowest marginal) to free the most points to reinvest.
--          Each round keeps the ban only if the re-optimized score improved, else reverts.
--   Guards:
--     - BUILD ENABLERS (nodes whose removal collapses dps or ehp to ~0, e.g. Hollow Palm =
--       "no weapon") are auto-detected, marked MANDATORY, and never banned.
--     - SOFT-BAN + DETOUR pathing: a ban means "don't TARGET this node," not "can't path
--       through it." Pathfinding is detour-weighted (excluded node costs DETOUR_PENALTY), so
--       banning a node never walls off a better keystone behind it — routes detour, and only
--       traverse a banned node as a last resort.
--
-- This measures the TWO things projections can't:
--   1. Real wall-time for a from-scratch depth-P search (P often 40-100+).
--   2. Whether the from-scratch tree BEATS the build's current hand-made tree at equal P.
--
-- Run (from repo root, PowerShell — Git Bash mangles -w /work):
--   docker run --rm -v "${PWD}:/work" -w /work -e SPIKE_BUILD_XML=/work/src/Builds/mymonk.xml `
--     ghcr.io/pathofbuildingcommunity/pathofbuilding-tests:latest `
--     busted --lua=luajit --filter CleanSlate
--
-- Tunables via env:
--   SPIKE_BEAM (beam width, default 20)      SPIKE_MAXJUMP (max path-jump length, default 8)
--   SPIKE_MAXDEPTH (cap P for a quick run)   SPIKE_WDPS / SPIKE_WEHP (axis weights, default 1)
--   SPIKE_DETOUR (excluded-node traversal penalty, default 50)
--   SPIKE_ROUNDS (max rounds PER PHASE, default 12)  SPIKE_PATIENCE (failed rounds/phase, 2)
--   SPIKE_SAVE_XML (output path for the optimized build, default
--                   /work/src/Builds/mymonk_optimized.xml — open it in PoB to inspect)

describe("CleanSlate: beam reallocation", function()
	local build

	before_each(function()
		newBuild()
		build = _G.build
		local xmlPath = os.getenv("SPIKE_BUILD_XML")
		if xmlPath then
			local f = io.open(xmlPath, "r")
			if f then
				local xml = f:read("*a")
				f:close()
				loadBuildFromXML(xml, "CleanSlate")
			end
		end
	end)

	it("grows a P-node tree from the class start and compares to the current tree", function()
		local spec = build.spec
		build.calcsTab:BuildOutput()
		local calcFunc, calcBase = build.calcsTab:GetMiscCalculator()

		local beamWidth = tonumber(os.getenv("SPIKE_BEAM")) or 20
		local maxDepthCap = tonumber(os.getenv("SPIKE_MAXDEPTH")) or 0

		-- ---- Budget P and the current tree's baseline score ------------------------------
		local P = (spec:CountAllocNodes()) -- normal passive points used by the current build
		assert.is_true(P and P > 0, "build has no allocated normal passives - load a real build via SPIKE_BUILD_XML")

		-- Raw current-build numbers (calcBase = current tree). Used both as the comparison
		-- target AND to NORMALIZE the score so DPS (hundreds) and EHP (thousands) are on the
		-- same scale. Without this, EHP magnitude dominates and the beam chases pure defense
		-- (observed: 20-pt tree had ehp UP, dps cratered). Each axis is scaled to ~1.0 at the
		-- current build's value; weights then express the real DPS:EHP preference.
		local rawCurDps = calcBase.FullDPS or calcBase.TotalDPS or 0
		local rawCurEhp = calcBase.TotalEHP or 0
		local refDps = math.max(rawCurDps, 1)
		local refEhp = math.max(rawCurEhp, 1)
		local W_DPS = tonumber(os.getenv("SPIKE_WDPS")) or 1.0
		local W_EHP = tonumber(os.getenv("SPIKE_WEHP")) or 1.0
		local function scoreOf(out)
			local dps = out.FullDPS or out.TotalDPS or 0
			local ehp = out.TotalEHP or 0
			-- normalized: 100 * weighted sum of (axis / current-build-axis)
			return 100 * (W_DPS * dps / refDps + W_EHP * ehp / refEhp), dps, ehp
		end

		local curScore, curDps, curEhp = scoreOf(calcBase) -- normalized current = ~100*(W_DPS+W_EHP)

		local startId = spec.curClass.startNodeId
		print(string.format("\nClass: %s   Budget P = %d normal points", spec.curClassName, P))
		print(string.format("Current tree score=%.1f  dps=%.1f  ehp=%.1f", curScore, curDps, curEhp))

		-- removeNodes = everything currently allocated EXCEPT the class start, so clean-slate
		-- evals start from just the start node.
		local removeCurrent = {}
		for id, node in pairs(spec.allocNodes) do
			if id ~= startId and not node.ascendancyName then
				removeCurrent[node] = true
			end
		end

		-- nodes table by id; helper to fetch node objects
		local nodes = spec.nodes

		-- Score a candidate set (table of node objects) from clean slate.
		local calls = 0
		local function scoreSet(nodeObjSet)
			calls = calls + 1
			local out = calcFunc({ addNodes = nodeObjSet, removeNodes = removeCurrent }, true)
			return scoreOf(out)
		end

		-- excludeSet: node ids the search must never allocate. This is the CURRENTLY ACTIVE
		-- exclusion set; runRun() swaps it in/out per cascade run so each run is isolated.
		-- Used to forbid auto-detected "crutch" outliers (e.g. Mind Over Matter).
		local excludeSet = {}
		local function setExclusions(ids)
			excludeSet = {}
			for _, id in ipairs(ids or {}) do excludeSet[id] = true end
		end
		-- Eligibility for a node to be added: main tree, has mods or is a passable travel node,
		-- not a class/ascendancy start, not an ascendancy node, not excluded.
		-- traversable: may a PATH pass through this node? Structural rules only — NOT excludeSet.
		-- This is the "soft ban" half: excluding a node must never wall off what's behind it.
		local function traversable(node)
			return node
				and node.type ~= "ClassStart" and node.type ~= "AscendClassStart"
				and not node.ascendancyName
				and node.type ~= "Mastery"        -- masteries need an effect choice; skip in spike
		end
		-- targetable: may this node be CHOSEN as a move endpoint / kept destination? This is
		-- where excludeSet bites — the diet stops *targeting* an excluded node, but (via
		-- traversable + detour-weighted pathing below) a route may still pass through it as a
		-- last resort, allocating it only as unavoidable travel cost.
		local function targetable(node)
			return traversable(node) and not excludeSet[node.id]
		end
		-- Per-step traversal weight: 1 for a normal node, DETOUR_PENALTY for an excluded one,
		-- so weighted-shortest-path prefers a detour and goes through an excluded node only when
		-- no cheaper route exists. Must stay an INTEGER point cost for budget accounting, so the
		-- penalty is applied as ordering weight in BFS, while real point cost = #path nodes.
		local DETOUR_PENALTY = tonumber(os.getenv("SPIKE_DETOUR")) or 50
		local function traverseWeight(node)
			return excludeSet[node.id] and DETOUR_PENALTY or 1
		end

		-- ---- Beam state -----------------------------------------------------------------
		-- A state = { ids = set of allocated ids (incl start), set = node-object set, score }
		local startNode = nodes[startId]
		local function newState(ids, objSet, score)
			return { ids = ids, set = objSet, score = score }
		end

		local function cloneIdSet(t) local c = {} for k in pairs(t) do c[k] = true end return c end

		-- BUDGET-DRIVEN: targetPoints is the point budget, NOT an iteration count. Moves spend
		-- a variable number of points (1 for a single node, up to maxJump for a path-jump), so
		-- the loop runs until states reach targetPoints, not for a fixed number of steps.
		-- Hoisted to outer scope: used both inside runBeam and in the post-run analysis.
		local targetPoints = (maxDepthCap > 0) and math.min(maxDepthCap, P) or P
		local function pointsIn(idSet)
			local n = 0
			for id in pairs(idSet) do if id ~= startId then n = n + 1 end end
			return n
		end
		print(string.format("Beam width = %d, target points = %d (P=%d%s)", beamWidth, targetPoints,
			P, maxDepthCap > 0 and (", capped at " .. maxDepthCap) or ""))

		-- runBeam: full beam search from the bare class start, honoring the current excludeSet.
		-- Returns the final beam (sorted, best first) and the call count for this run.
		local function runBeam()
		local startCalls = calls
		local initIds = { [startId] = true }
		local initSet = { [startNode] = true }
		local beam = { newState(initIds, initSet, 0) }
		-- Score of the empty (start-only) tree as the true baseline-from-scratch:
		do
			local s = scoreSet(initSet)
			beam[1].score = s
		end

		-- Weighted shortest path (Dijkstra) from a candidate id-set. Returns, for each reachable
		-- node: `pdist` = real POINT cost (count of new nodes, for budget accounting) and a
		-- pathTo reconstructor. Edge weight = traverseWeight(dest): an excluded node costs
		-- DETOUR_PENALTY so routes prefer detours and traverse an excluded node only when no
		-- cheaper path exists (soft-ban + prefer-detours). Ordering is by accumulated WEIGHT;
		-- ties and budget use the integer point count. Self-contained so it works on an
		-- in-flight candidate set, not the real allocNodes.
		local function pathsFromSet(idSet)
			local wdist = {}   -- accumulated traversal weight (penalty-aware), for ordering
			local pdist = {}   -- real point cost = number of new nodes on the path
			local prev = {}
			local visited = {}
			-- frontier nodes start at weight/points 0
			for id in pairs(idSet) do wdist[id] = 0; pdist[id] = 0 end
			-- Dijkstra with a linear-scan min (node fan-out per step is small; beam dominates).
			while true do
				local u, best = nil, math.huge
				for id, w in pairs(wdist) do
					if not visited[id] and w < best then u, best = id, w end
				end
				if not u then break end
				visited[u] = true
				local node = nodes[u]
				for _, other in ipairs(node.linked or {}) do
					local oid = other.id
					if not visited[oid] and (idSet[oid] or traversable(other)) then
						local w = wdist[u] + traverseWeight(other)
						if wdist[oid] == nil or w < wdist[oid] then
							wdist[oid] = w
							pdist[oid] = pdist[u] + (idSet[oid] and 0 or 1)
							prev[oid] = u
						end
					end
				end
			end
			-- reconstruct the new-node path (excluding nodes already in idSet) for a target
			local function pathTo(targetId)
				local out = {}
				local cur = targetId
				while cur ~= nil and not idSet[cur] do
					out[#out + 1] = cur
					cur = prev[cur]
				end
				return out -- list of ids to add (target first)
			end
			return pdist, pathTo
		end

		-- Expansion now offers TWO move types per state:
		--   (a) single adjacent node  (cost 1) — cheap, local refinement
		--   (b) jump to a NOTABLE/KEYSTONE via shortest path (cost = path length) — escapes
		--       the all-EHP local optimum by investing in distant damage clusters.
		-- Path-jumps are deduped by target modKey; we also cap jump length so we don't blow
		-- the budget in one move.
		local maxJump = tonumber(os.getenv("SPIKE_MAXJUMP")) or 8
		local t0 = os.clock()
		local step = 0
		local maxSteps = targetPoints + 5 -- safety: at minimum 1 pt/step, so this always terminates
		while step < maxSteps do
			step = step + 1
			-- stop once every beam state is full
			local allFull = true
			for _, st in ipairs(beam) do if pointsIn(st.ids) < targetPoints then allFull = false break end end
			if allFull then break end

			local nextStates = {}
			for _, st in ipairs(beam) do
				local pointsLeft = targetPoints - pointsIn(st.ids)
				if pointsLeft <= 0 then
					nextStates[#nextStates + 1] = st -- full; carry forward unchanged
				else
					local dist, pathTo = pathsFromSet(st.ids)
					local moves = {}             -- list of { ids = {idsToAdd}, key = dedupKey }
					local seenKey = {}
					-- (a) single adjacent nodes (must be targetable: not excluded as a destination)
					for id in pairs(st.ids) do
						for _, other in ipairs(nodes[id].linked or {}) do
							if not st.ids[other.id] and targetable(other) then
								local k = "s:" .. (other.modKey ~= "" and other.modKey or other.id)
								if not seenKey[k] then seenKey[k] = true; moves[#moves+1] = { ids = { other.id }, key = k } end
							end
						end
					end
					-- (b) notable/keystone path-jumps (target must be targetable; the PATH may pass
					-- through excluded nodes via the detour-weighted pdist, but only as last resort)
					for tid, d in pairs(dist) do
						local tnode = nodes[tid]
						if not st.ids[tid] and d >= 2 and d <= math.min(maxJump, pointsLeft)
							and (tnode.type == "Notable" or tnode.type == "Keystone") and targetable(tnode) then
							local k = "j:" .. (tnode.modKey ~= "" and tnode.modKey or tid)
							if not seenKey[k] then
								seenKey[k] = true
								moves[#moves+1] = { ids = pathTo(tid), key = k }
							end
						end
					end
					-- evaluate each move
					for _, mv in ipairs(moves) do
						local newSet = {}
						for n in pairs(st.set) do newSet[n] = true end
						local newIds = cloneIdSet(st.ids)
						for _, addId in ipairs(mv.ids) do
							newSet[nodes[addId]] = true
							newIds[addId] = true
						end
						local sc = scoreSet(newSet)
						nextStates[#nextStates + 1] = newState(newIds, newSet, sc)
					end
				end
			end
			if #nextStates == 0 then
				print(string.format("Step %d: no frontier left, stopping early.", step))
				break
			end
			table.sort(nextStates, function(a, b) return a.score > b.score end)
			beam = {}
			for i = 1, math.min(beamWidth, #nextStates) do beam[i] = nextStates[i] end
			local bp = pointsIn(beam[1].ids)
			if step % 3 == 0 or bp >= targetPoints then
				print(string.format("  step %2d: best score=%.1f (pts=%d/%d)  (%d calls, %.1fs)",
					step, beam[1].score, bp, targetPoints, calls, os.clock() - t0))
			end
		end
		return beam, calls - startCalls, os.clock() - t0
		end  -- runBeam

		-- mandatory: ids of detected BUILD ENABLERS — nodes whose removal collapses an axis
		-- (dps or ehp) to ~0. These are load-bearing keystones (e.g. Hollow Palm Technique,
		-- which removes weapon damage entirely), NOT crutches. Excluding one only yields a
		-- degenerate dps=0 (or ehp=0) build, so the cascade must NEVER exclude them and must
		-- skip them when picking a chain outlier. Persists across all runs.
		local mandatory = {}
		-- An axis "collapses" if leaving the node out drops that axis below 1% of the full
		-- build's value on the same axis (and the full build had a non-zero value there).
		local COLLAPSE_FRAC = 0.01
		local function collapses(fullV, leaveOutV)
			return fullV > 0 and leaveOutV < COLLAPSE_FRAC * fullV
		end

		-- analyzeMarginals: given a finished beam, measure each notable/keystone's marginal
		-- contribution (marginal = bestScore - score(best WITHOUT node)) AND its leave-one-out
		-- dps/ehp. Prints the ranking, registers any BUILD ENABLER (axis-collapsing node) into
		-- `mandatory`, and returns the FULL marginal list sorted high→low (each entry tagged
		-- isEnabler), so the caller can pick the lowest-marginal droppable (elimination diet) or
		-- the highest. Returns marg, mean, sd.
		local function analyzeMarginals(rbeam, label)
			local rbest = rbeam[1]
			local rScore, rDps, rEhp = scoreSet(rbest.set)
			local marg = {}
			for id in pairs(rbest.ids) do
				local n = nodes[id]
				if n and (n.type == "Notable" or n.type == "Keystone") then
					local without = {}
					for nd in pairs(rbest.set) do if nd.id ~= id then without[nd] = true end end
					local s, d, e = scoreSet(without)
					local isEnabler = collapses(rDps, d) or collapses(rEhp, e)
					marg[#marg+1] = { id = id, name = n.dn or ("#"..id), marginal = rScore - s,
						loDps = d, loEhp = e, isEnabler = isEnabler,
						enablerAxis = collapses(rDps, d) and "dps" or (collapses(rEhp, e) and "ehp" or nil) }
				end
			end
			table.sort(marg, function(a, b) return a.marginal > b.marginal end)
			local sum, n = 0, #marg
			for _, m in ipairs(marg) do sum = sum + m.marginal end
			local mean = sum / math.max(n, 1)
			local var = 0
			for _, m in ipairs(marg) do var = var + (m.marginal - mean)^2 end
			local sd = math.sqrt(var / math.max(n, 1))
			print(string.format("\n--- MARGINAL contribution (%s) ---", label))
			for i = 1, math.min(8, #marg) do
				local m = marg[i]
				local tag = m.isEnabler and string.format("   <-- ENABLER (%s->0)", m.enablerAxis)
					or ((m.marginal > mean + 2*sd) and "   <-- outlier" or "")
				print(string.format("  %-28s marginal=%+.1f%s", m.name:sub(1,28), m.marginal, tag))
			end
			print(string.format("  (mean=%.1f, sd=%.1f, outlier threshold = mean+2sd = %.1f)", mean, sd, mean + 2*sd))
			-- Register newly-found enablers as mandatory.
			for _, m in ipairs(marg) do
				if m.isEnabler and not mandatory[m.id] then
					mandatory[m.id] = true
					print(string.format("  ** BUILD ENABLER detected: %s (removing it -> %s=0). Marked MANDATORY; never excluded.",
						m.name, m.enablerAxis))
				end
			end
			for _, m in ipairs(marg) do m.isOutlier = m.marginal > mean + 2*sd end
			return marg, mean, sd
		end

		-- pathLenInBuild: distance (in points) from the class start to a node WITHIN the build's
		-- own allocated subgraph — i.e. how deep into the tree it sits. Longer = more isolated =
		-- more points potentially freed when it (and its now-unused tail) is removed. Used as the
		-- Phase-2 "longest path removal" victim metric. BFS over only the allocated id-set.
		local function buildDepths(idSet)
			local dist = { [startId] = 0 }
			local queue = { startId }
			local qi = 1
			while qi <= #queue do
				local id = queue[qi]; qi = qi + 1
				for _, other in ipairs(nodes[id].linked or {}) do
					if idSet[other.id] and dist[other.id] == nil then
						dist[other.id] = dist[id] + 1
						queue[#queue + 1] = other.id
					end
				end
			end
			return dist
		end

		-- lowestDroppable: from an analyzed marginal list, the NON-enabler with the smallest
		-- marginal that is neither already-banned (`banned`) nor already tried-and-reverted
		-- (`tried`) — the Phase-1 victim (weakest remaining link). Skipping tried victims stops
		-- the diet from re-testing the same node every round after a revert.
		local function lowestDroppable(marg, banned, tried, bestRun)
			local pick = nil
			for _, m in ipairs(marg) do
				if not m.isEnabler and not banned[m.id] and not tried[m.id] then
					if pick == nil or m.marginal < pick.marginal then pick = m end
				end
			end
			return pick
		end

		-- longestPathDroppable: the Phase-2 victim — the droppable NON-enabler keystone/notable
		-- sitting DEEPEST in the build (longest path from start), tie-broken by LOWEST marginal
		-- ("longest path, low value"). Removing a deep, low-value node frees the most points to
		-- reinvest. Needs the build's id-set for depth, so it takes bestRun.
		local function longestPathDroppable(marg, banned, tried, bestRun)
			local depths = buildDepths(bestRun.best.ids)
			local pick, pickDepth = nil, -1
			for _, m in ipairs(marg) do
				if not m.isEnabler and not banned[m.id] and not tried[m.id] then
					local d = depths[m.id] or 0
					if d > pickDepth or (d == pickDepth and pick and m.marginal < pick.marginal) then
						pick, pickDepth = m, d
					end
				end
			end
			if pick then pick.pathLen = pickDepth end
			return pick
		end

		-- runRun: set the exclusion set (list of ids), run a full beam, report, and return
		-- { beam, best, score, dps, ehp, points, outlier }. outlier is run's top-marginal pick.
		-- SAFETY: any id in `mandatory` (a detected build enabler) is stripped from the
		-- exclusion list before running, so no run can ever disable a load-bearing keystone.
		local function runRun(label, excludeIds)
			local effective, skipped = {}, {}
			for _, id in ipairs(excludeIds or {}) do
				if mandatory[id] then skipped[#skipped+1] = id else effective[#effective+1] = id end
			end
			setExclusions(effective)
			local exNames = {}
			for _, id in ipairs(effective) do
				local nn = nodes[id]; exNames[#exNames+1] = (nn and nn.dn) or ("#"..id)
			end
			print(string.format("\n========== %s (excluding: %s) ==========",
				label, #exNames > 0 and table.concat(exNames, ", ") or "nothing"))
			if #skipped > 0 then
				local sn = {}
				for _, id in ipairs(skipped) do local nn = nodes[id]; sn[#sn+1] = (nn and nn.dn) or ("#"..id) end
				print(string.format("  (kept MANDATORY build enabler%s in despite request: %s)",
					#sn == 1 and "" or "s", table.concat(sn, ", ")))
			end
			local rbeam, rCalls, rTime = runBeam()
			local rbest = rbeam[1]
			local s, d, e = scoreSet(rbest.set)
			print(string.format("%s best: score=%.1f dps=%.1f ehp=%.1f points=%d (%d calls, %.1fs)",
				label, s, d, e, pointsIn(rbest.ids), rCalls, rTime))
			local marg = analyzeMarginals(rbeam, label)
			return { beam = rbeam, best = rbest, score = s, dps = d, ehp = e,
				points = pointsIn(rbest.ids), marg = marg, effectiveExclude = effective }
		end

		-- ===================== ELIMINATION-DIET LOOP =====================
		-- Accumulating greedy elimination. Round 0 establishes the unconstrained best. Each
		-- round eliminates the LOWEST-marginal droppable notable/keystone (the weakest link —
		-- its points are best spent elsewhere), bans it, and re-optimizes. A ban is kept only
		-- if the re-optimized build IMPROVES on the current best; otherwise it's reverted and a
		-- "patience" counter ticks. Stop on no-improvement after `patienceMax` consecutive
		-- failed rounds (natural convergence). Enablers are never eligible (the marginal
		-- analysis tags them; lowestDroppable skips them) and pathing is soft-ban + detour, so
		-- banning a node never walls off a better keystone behind it.
		local runs = {}
		local excludedList = {}  -- accumulated kept bans (ids), for reporting/assertions
		local patienceMax = tonumber(os.getenv("SPIKE_PATIENCE")) or 2
		local maxRounds = tonumber(os.getenv("SPIKE_ROUNDS")) or 12

		local r0 = runRun("ROUND 0 (unconstrained)", {})
		runs[#runs+1] = r0
		local best = r0.best
		local pointsUsed = r0.points
		local bestRun = r0
		local bestExcluded = {}        -- ordered ban list that produced bestRun
		local bannedSet = {}           -- id -> true mirror of bestExcluded (fast lookup)
		local tried = {}               -- victims tried since the last accepted ban (skip on revert)

		-- runDiet: accumulating-greedy elimination loop. `selector(marg, banned, tried, bestRun)`
		-- picks the round's victim; `phase` labels output. Mutates the shared `runs`/`bannedSet`
		-- tracking via the passed-in state and RETURNS the updated {bestRun, bestExcluded,
		-- bannedSet}. Starts from the given bestRun/bestExcluded so phases chain.
		local function runDiet(phase, selector, startBestRun, startExcluded, startBanned)
			local bestRun = startBestRun
			local bestExcluded = startExcluded
			local bannedSet = startBanned
			local tried = {}
			local patience = 0
			local round = 0
			print(string.format("\n##### DIET PHASE: %s (start score=%.1f) #####", phase, bestRun.score))
			while round < maxRounds and patience < patienceMax do
				round = round + 1
				-- Restore excludeSet to the accepted bans before inspecting the best build, so
				-- the selector sees the TRUE ban state (the last runRun may have left a reverted
				-- trial's exclusions active).
				setExclusions(bestExcluded)
				local victim = selector(bestRun.marg, bannedSet, tried, bestRun)
				if not victim then
					print(string.format("[%s] round %d: no untried droppable node left. Stop.", phase, round))
					break
				end
				tried[victim.id] = true
				local trialExclude = {}
				for _, id in ipairs(bestExcluded) do trialExclude[#trialExclude+1] = id end
				trialExclude[#trialExclude+1] = victim.id
				local extra = victim.pathLen and string.format(", pathLen %d", victim.pathLen) or ""
				local label = string.format("%s R%d (drop %s, marginal %+.1f%s)",
					phase, round, victim.name, victim.marginal, extra)
				local r = runRun(label, trialExclude)
				runs[#runs+1] = r
				if r.score > bestRun.score + 1e-6 then
					print(string.format("  [%s] IMPROVED %.1f -> %.1f (banned %s). Keeping ban.",
						phase, bestRun.score, r.score, victim.name))
					bestRun = r
					bestExcluded = trialExclude
					bannedSet[victim.id] = true
					best = r.best
					pointsUsed = r.points
					patience = 0
					tried = {}   -- build changed; previously-tried victims may now help — revisit
				else
					print(string.format("  [%s] no improvement (%.1f vs best %.1f). Reverting %s. patience %d/%d.",
						phase, r.score, bestRun.score, victim.name, patience + 1, patienceMax))
					patience = patience + 1
				end
			end
			return bestRun, bestExcluded, bannedSet
		end

		-- PHASE 1: lowest-marginal elimination (validated). PHASE 2: longest-path elimination,
		-- chained from Phase 1's result — ban the deepest droppable node to free the most points.
		bestRun, bestExcluded, bannedSet = runDiet("P1-marginal", lowestDroppable, bestRun, bestExcluded, bannedSet)
		bestRun, bestExcluded, bannedSet = runDiet("P2-longpath", longestPathDroppable, bestRun, bestExcluded, bannedSet)
		excludedList = bestExcluded

		-- keep excludeSet matching the accepted best build for any downstream eval
		setExclusions(bestExcluded)
		local round = #runs - 1  -- total diet rounds across both phases (excl. round 0), for reporting

		-- ===================== DIET RESULT =====================
		print(string.format("\n=== ELIMINATION-DIET BEST (after %d round%s) ===", round, round == 1 and "" or "s"))
		do
			local st = bestRun.best
			local s, d, e = scoreSet(st.set)
			local names = {}
			for id in pairs(st.ids) do
				local nn = nodes[id]
				if nn and (nn.type == "Notable" or nn.type == "Keystone") then names[#names+1] = nn.dn or ("#"..id) end
			end
			table.sort(names)
			local banNames = {}
			for _, id in ipairs(bestExcluded) do local nn = nodes[id]; banNames[#banNames+1] = (nn and nn.dn) or ("#"..id) end
			print(string.format("score=%.1f dps=%.1f ehp=%.1f points=%d  (%d node%s banned: %s)",
				s, d, e, pointsIn(st.ids), #bestExcluded, #bestExcluded == 1 and "" or "s",
				#banNames > 0 and table.concat(banNames, ", ") or "none"))
			print("  notables/keystones: " .. table.concat(names, ", "))
		end
		print(string.format("For reference - hand-made: score=%.1f dps=%.1f ehp=%.1f points=%d", curScore, curDps, curEhp, P))

		-- ---- per-round progress table ----------------------------------------------------
		print("\n=== DIET PROGRESS ===")
		print("  round | score  | dps    | ehp    | banned this round / result")
		for ri, r in ipairs(runs) do
			print(string.format("  %5d | %6.1f | %6.1f | %6.1f | %s",
				ri - 1, r.score, r.dps, r.ehp, r.effectiveExclude and #r.effectiveExclude > 0
					and ("excl: " .. tostring(#r.effectiveExclude)) or "(unconstrained)"))
		end

		-- ---- mandatory build-enabler report ---------------------------------------------
		local mandNames = {}
		for id in pairs(mandatory) do local nn = nodes[id]; mandNames[#mandNames+1] = (nn and nn.dn) or ("#"..id) end
		table.sort(mandNames)
		print(string.format("\n=== MANDATORY BUILD ENABLERS (never excluded) ===\n  %s",
			#mandNames > 0 and table.concat(mandNames, ", ") or "(none detected)"))

		-- ---- SANITY CHECKS ---------------------------------------------------------------
		-- (a) No run's EFFECTIVE exclusion set ever contained a mandatory enabler.
		for ri, r in ipairs(runs) do
			for _, id in ipairs(r.effectiveExclude or {}) do
				assert.is_falsy(mandatory[id],
					string.format("ROUND %d excluded mandatory enabler %s — enabler guard failed",
						ri - 1, (nodes[id] and nodes[id].dn) or ("#"..id)))
			end
		end
		-- (b) Any run that kept all enablers should NOT have collapsed dps to ~0 just from the
		-- exclusion (a real collapse means the enabler set is incomplete). Report, don't hard
		-- fail — a legit all-defense build can read dps 0 — but flag it loudly for inspection.
		for ri, r in ipairs(runs) do
			if collapses(curDps, r.dps) then
				print(string.format("  WARNING: ROUND %d has dps=%.1f (~0) despite mandatory guard — possible missed enabler.", ri - 1, r.dps))
			end
		end

		-- (2) DECISIVE: score the CURRENT build's exact node set through the SAME clean-slate
		-- path (addNodes = current alloc, removeNodes = current alloc minus start). If this
		-- reproduces ~956 DPS, the eval is correct and the SEARCH is the problem. If it shows
		-- ~166, the clean-slate baseline is broken (e.g. skill not coming online from scratch).
		local curNodeSet = { [startNode] = true }
		for id, node in pairs(spec.allocNodes) do
			if not node.ascendancyName then curNodeSet[node] = true end
		end
		local reproScore, reproDps, reproEhp = scoreSet(curNodeSet)
		print(string.format("\nDIAG re-score current node set via clean-slate path: dps=%.1f ehp=%.1f", reproDps, reproEhp))
		print(string.format("  (should match current dps=%.1f ehp=%.1f. If dps is way off, the", curDps, curEhp))
		print("   clean-slate eval is the bug, NOT the search.)")

		-- SANITY (c): the clean-slate eval must reproduce the live build's real numbers.
		-- If this drifts, the override path is broken and every score above is meaningless.
		assert.is_true(calls > 0)
		assert.is_true(pointsUsed > 0)
		local function approx(a, b) return math.abs(a - b) <= math.max(1, 0.01 * math.max(math.abs(a), math.abs(b))) end
		assert.is_true(approx(reproDps, curDps),
			string.format("clean-slate re-score dps %.1f != live %.1f — eval path broken", reproDps, curDps))
		assert.is_true(approx(reproEhp, curEhp),
			string.format("clean-slate re-score ehp %.1f != live %.1f — eval path broken", reproEhp, curEhp))

		-- ---- SAVE the winning build as an importable PoB file (LAST — it mutates the live spec) --
		-- Apply the diet's best node set to the LIVE spec, then serialize via build:SaveDB so the
		-- result opens directly in PoB. Hash list = winner's main-tree ids + the build's existing
		-- ascendancy ids (ascendancy is fixed, not searched). ImportFromNodeList resets and
		-- re-allocates the live spec (PassiveSpec.lua:319), exactly as a normal load does. Done
		-- AFTER all sanity checks so the mutation can't corrupt the re-score. Guarded: a save
		-- failure prints but does not fail the spike.
		do
			local outPath = os.getenv("SPIKE_SAVE_XML") or "/work/src/Builds/mymonk_optimized.xml"
			-- VERIFY FIRST, while the live spec is still the original: re-score the diet winner's
			-- exact node set through the trusted clean-slate path. This is the meaningful check
			-- (the post-apply mainOutput read is unreliable headless — a tree-granted active skill
			-- needs a full build rebuild to come back online, which BuildOutput alone doesn't do).
			local verSet = { [startNode] = true }
			for id in pairs(bestRun.best.ids) do verSet[nodes[id]] = true end
			local vScore, vDps, vEhp = scoreSet(verSet)
			local matches = math.abs(vScore - bestRun.score) <= math.max(1, 0.01 * bestRun.score)

			local ok, err = pcall(function()
				-- Hash list = winner's main-tree ids + the build's kept ascendancy ids.
				local hashList = {}
				for id in pairs(bestRun.best.ids) do
					if not nodes[id].ascendancyName then table.insert(hashList, id) end
				end
				for id, node in pairs(spec.allocNodes) do
					if node.ascendancyName then table.insert(hashList, id) end
				end
				spec:ImportFromNodeList(nil, spec.curClassId, spec.curAscendClassId,
					spec.curSecondaryAscendClassId or 0, hashList, {}, {}, {})
				spec:BuildAllDependsAndPaths()
				local xmlText = build:SaveDB("optimized")
				assert(xmlText, "SaveDB returned nil")
				local f = assert(io.open(outPath, "w+"), "cannot open " .. outPath)
				f:write(xmlText); f:close()
				-- confirm every optimized main-tree id is present in the saved nodes attribute
				local nodesAttr = xmlText:match('nodes="([^"]*)"') or ""
				local present = {}
				for idStr in nodesAttr:gmatch("%d+") do present[tonumber(idStr)] = true end
				for _, id in ipairs(hashList) do
					assert(present[id], "saved XML missing optimized node " .. id)
				end
			end)
			if ok then
				print(string.format("\n=== SAVED optimized build -> %s ===", outPath))
				print("  Open it in PoB (Import/Open) to inspect the optimized tree.")
				print(string.format("  verified clean-slate re-score: score=%.1f dps=%.1f ehp=%.1f %s",
					vScore, vDps, vEhp, matches and "(matches diet best)" or "(!! MISMATCH vs diet best)"))
				-- FULL ROUND-TRIP CHECK: reload the saved XML through the same path PoB uses on
				-- open (SetMode BUILD + OnFrame = full rebuild, so a tree-granted active skill
				-- comes back online — unlike a bare BuildOutput). This proves the file works in PoB.
				local rok, rerr = pcall(function()
					local sf = assert(io.open(outPath, "r"))
					local savedXml = sf:read("*a"); sf:close()
					loadBuildFromXML(savedXml, "OptimizedReload")
					_G.build.calcsTab:BuildOutput()
					local ro = _G.build.calcsTab.mainOutput or {}
					local rd = ro.FullDPS or ro.TotalDPS or 0
					print(string.format("  RELOADED build (headless full rebuild): dps=%.1f ehp=%.1f", rd, ro.TotalEHP or 0))
					if collapses(bestRun.dps, rd) then
						print("  NOTE: dps reads ~0 on HEADLESS reload. This is a known headless limitation —")
						print("  a tree-granted active skill is not auto-selected as the main skill without the")
						print("  GUI's skill setup. The saved XML is COMPLETE (Skills/Config/main-skill all")
						print("  present; clean-slate re-score above matches). Open it in real PoB to confirm DPS.")
					end
				end)
				if not rok then print("  (reload check skipped: " .. tostring(rerr) .. ")") end
			else
				print(string.format("\n=== SAVE FAILED (non-fatal): %s ===", tostring(err)))
			end
		end
	end)
end)
