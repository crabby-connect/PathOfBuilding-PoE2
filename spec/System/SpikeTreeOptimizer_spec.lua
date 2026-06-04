-- Throwaway spike for the passive tree optimizer (see docs/tree-optimizer-design.md §4).
--
-- Purpose: validate calc throughput and the expand() primitive against a real build,
-- BEFORE committing to beam-search machinery. This is NOT production code and is not
-- wired into the GUI. It is written as a busted spec so it runs under the project's
-- existing harness:
--
--   cd src
--   busted --lua=luajit ../spec/System/SpikeTreeOptimizer_spec.lua
--
-- (Same invocation .github/workflows/test.yml uses, inside the pathofbuilding-tests
-- container.) Set SPIKE_BUILD_XML to an exported build path for meaningful numbers;
-- otherwise it runs against the fresh default build (~0 DPS) just to prove the loop works.
--
-- What it measures:
--   1. Builds the misc calculator ONCE (the #1 perf trap if done per-candidate).
--   2. Depth-1: brute-forces every reachable un-allocated node as a single addition -
--      this is expand() from the current frontier with N = infinity.
--   3. Depth-2: top-K depth-1 nodes, all pairs, to show the combinatorial blow-up the
--      beam is meant to tame.
--   4. Reports ms/call and extrapolates whether the design-doc beam fits in a dialog.

describe("Spike: tree optimizer throughput", function()
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
				loadBuildFromXML(xml, "Spike")
			else
				print("SPIKE_BUILD_XML set but unreadable: " .. xmlPath)
			end
		end
	end)

	it("brute-forces depth-1 and depth-2 node additions and reports timings", function()
		local spec = build.spec

		-- Force a fresh calc pass so self.miscCalculator is valid, then grab the closure ONCE.
		-- GetMiscCalculator() takes NO args (src/Classes/CalcsTab.lua:693).
		build.calcsTab:BuildOutput()
		local calcFunc, calcBase = build.calcsTab:GetMiscCalculator()
		local mainEnv = build.calcsTab.mainEnv

		assert.is_function(calcFunc)
		assert.is_table(calcBase)

		-- Weighted scalar over REAL outputs. Tuning is not the point of the spike.
		local W_DPS, W_EHP = 1.0, 1.0
		local function scoreOf(output)
			local dps = output.FullDPS or output.TotalDPS or 0
			local ehp = output.TotalEHP or 0
			return W_DPS * dps + W_EHP * ehp, dps, ehp
		end

		local baseScore, baseDps, baseEhp = scoreOf(calcBase)
		print(string.format("\nBase: score=%.1f  dps=%.1f  ehp=%.1f", baseScore, baseDps, baseEhp))

		local calls = 0
		local function evalAdd(addNodesSet)
			calls = calls + 1
			-- useFullDPS=true matches PowerBuilder so FullDPS deltas aren't cache-zeroed
			-- (cache caveat: src/Modules/Calcs.lua:139-147).
			return scoreOf(calcFunc({ addNodes = addNodesSet }, true))
		end

		-- Eligibility mirrors PowerBuilder (src/Classes/CalcsTab.lua:548): reachable,
		-- un-allocated, has mods, not item-granted. pathDist/path were filled by
		-- PassiveSpec:BuildPathFromNode during load. Main tree only for the spike.
		local function isEligible(node)
			return node
				and not node.alloc
				and node.modKey ~= nil and node.modKey ~= ""
				and not mainEnv.grantedPassives[node.id]
				and node.pathDist ~= nil and node.pathDist < 1000
				and not node.ascendancyName
		end

		local candidates = {}
		for _, node in pairs(spec.nodes) do
			if isEligible(node) then
				candidates[#candidates + 1] = node
			end
		end
		print(string.format("Eligible candidate nodes: %d", #candidates))
		assert.is_true(#candidates > 0, "no eligible nodes - is a class/tree loaded?")

		-- Add a node WITH its path so every candidate is connected (design doc §3.4 / step 5).
		local function pathSetFor(node)
			local set = {}
			if node.path then
				for _, n in ipairs(node.path) do set[n] = true end
			end
			set[node] = true
			return set
		end

		-- Depth-1: expand() with N = infinity.
		local t0 = os.clock()
		local d1 = {}
		for _, node in ipairs(candidates) do
			local score, dps, ehp = evalAdd(pathSetFor(node))
			d1[#d1 + 1] = {
				node = node,
				delta = score - baseScore,
				dDps = dps - baseDps,
				dEhp = ehp - baseEhp,
				cost = node.pathDist or 1,
			}
		end
		local t1 = os.clock()
		local d1Calls = calls
		table.sort(d1, function(a, b) return a.delta > b.delta end)

		print(string.format("\nDepth-1: %d evals in %.3fs  (%.2f ms/call)",
			d1Calls, t1 - t0, (t1 - t0) * 1000 / math.max(d1Calls, 1)))
		print("Top 10 single-node additions (by weighted score delta):")
		for i = 1, math.min(10, #d1) do
			local r = d1[i]
			print(string.format("  %2d. %-32s  dScore=%+.1f  dDps=%+.1f  dEhp=%+.1f  cost=%d",
				i, (r.node.dn or "?"):sub(1, 32), r.delta, r.dDps, r.dEhp, r.cost))
		end

		-- Depth-2: top-K, all pairs. Shows why full brute force past depth ~2 is hopeless.
		local K = math.min(30, #d1)
		local t2 = os.clock()
		local d2 = {}
		for i = 1, K do
			for j = i + 1, K do
				local a, b = d1[i].node, d1[j].node
				local set = pathSetFor(a)
				for k in pairs(pathSetFor(b)) do set[k] = true end
				d2[#d2 + 1] = { a = a, b = b, delta = (evalAdd(set)) - baseScore }
			end
		end
		local t3 = os.clock()
		local d2Calls = calls - d1Calls
		table.sort(d2, function(a, b) return a.delta > b.delta end)

		print(string.format("\nDepth-2 (top-%d, all pairs): %d evals in %.3fs  (%.2f ms/call)",
			K, d2Calls, t3 - t2, (t3 - t2) * 1000 / math.max(d2Calls, 1)))
		print("Top 5 node pairs (by weighted score delta):")
		for i = 1, math.min(5, #d2) do
			local r = d2[i]
			print(string.format("  %2d. %-22s + %-22s  dScore=%+.1f",
				i, (r.a.dn or "?"):sub(1, 22), (r.b.dn or "?"):sub(1, 22), r.delta))
		end

		-- ----------------------------------------------------------------------------
		-- PRE-FILTER CUT-RATE PROTOTYPE (design doc §3.3).
		-- Question: can a CHEAP score (no calcFunc call) rank candidates well enough that
		-- a top-M cut keeps the TRUE top-K (by real calcFunc delta we measured in d1)?
		-- If recall@M is high for M ~100-300, the pre-filter works and the beam is feasible.
		--
		-- Cheap score = weighted sum over the node's OWN finalModList values, no engine call.
		-- This is the "linear surrogate" we rejected as an OBJECTIVE; here we only test
		-- whether it's good enough as a RANKING. Per-point variant divides by pathDist.
		-- ----------------------------------------------------------------------------
		local function cheapScore(node)
			local mods = node.finalModList or node.modList
			if not mods then return 0 end
			local s = 0
			for _, mod in ipairs(mods) do
				local v = mod.value
				if type(v) == "number" then
					-- crude magnitude weighting; abs so penalties don't cancel bonuses
					s = s + math.abs(v)
				elseif type(v) == "table" and type(v.mod) == "table" then
					-- conditional/nested mods: count a flat proxy so they aren't free
					s = s + 1
				end
			end
			return s
		end

		-- Ground-truth rank: d1 is already sorted by real delta desc. Tag each with its
		-- true rank and attach the cheap scores.
		local byNode = {}
		for rank, r in ipairs(d1) do
			r.trueRank = rank
			r.cheap = cheapScore(r.node)
			r.cheapPP = r.cheap / math.max(r.cost, 1)
			byNode[r.node] = r
		end

		local function recallAtM(scoreKey, M, K)
			-- top-M candidates by the cheap key
			local order = {}
			for _, r in ipairs(d1) do order[#order + 1] = r end
			table.sort(order, function(a, b) return a[scoreKey] > b[scoreKey] end)
			local kept = {}
			for i = 1, math.min(M, #order) do kept[order[i].node] = true end
			-- how many of the TRUE top-K survived the cut?
			local hit = 0
			for i = 1, math.min(K, #d1) do
				if kept[d1[i].node] then hit = hit + 1 end
			end
			return hit, math.min(K, #d1)
		end

		print(string.format("\nPre-filter recall (cheap modlist score vs. true top-K by calcFunc):"))
		print("  cut M | recall@K=20 (raw) | recall@K=20 (per-point) | recall@K=50 (per-point)")
		for _, M in ipairs({ 100, 200, 300, 500 }) do
			local hRaw20 = recallAtM("cheap", M, 20)
			local hPP20 = recallAtM("cheapPP", M, 20)
			local hPP50 = recallAtM("cheapPP", M, 50)
			print(string.format("  %5d | %2d/20 (%3.0f%%)        | %2d/20 (%3.0f%%)            | %2d/50 (%3.0f%%)",
				M, hRaw20, hRaw20 / 20 * 100, hPP20, hPP20 / 20 * 100, hPP50, hPP50 / 50 * 100))
		end
		print("  NOTE: low recall = cheap modlist score is a bad ranking (expected; the calc is")
		print("  multiplicative/conditional). Pre-filter via modlist sum is not viable.")

		-- ----------------------------------------------------------------------------
		-- DEDUP + CACHE measurement (chosen direction; design doc §3.3 alt).
		-- No surrogate. Two questions:
		--   1. How many UNIQUE modKeys among the candidates? PowerBuilder caches calcFunc by
		--      modKey (CalcsTab.lua:575), so duplicate-effect nodes cost only one eval.
		--   2. Given the real ms/call, how long is a beam at N=40 vs N=150 if each step only
		--      pays for unique modKeys, not raw node count?
		-- ----------------------------------------------------------------------------
		local uniqueKeys = {}
		local uniqueCount = 0
		for _, node in ipairs(candidates) do
			local key = node.modKey
			if not uniqueKeys[key] then
				uniqueKeys[key] = true
				uniqueCount = uniqueCount + 1
			end
		end
		local dupFactor = #candidates / math.max(uniqueCount, 1)
		print(string.format("\nDedup: %d candidate nodes -> %d unique modKeys (%.2fx duplication).",
			#candidates, uniqueCount, dupFactor))

		-- Beam projection: per step, a beam of width N expands each member against the
		-- frontier's reachable candidates, but with modKey caching the WORST case unique
		-- evals per step is bounded by uniqueCount (you never eval the same modKey twice
		-- within a calcFunc cache lifetime). Use a conservative model: per depth step the
		-- beam evaluates min(N * topK, uniqueCount) unique calcs.
		local ms = (t1 - t0) * 1000 / math.max(d1Calls, 1)
		local function beamSeconds(N, topK, depth)
			local perStep = math.min(N * topK, uniqueCount)
			return perStep * depth * ms / 1000, perStep * depth
		end
		print("Projected beam time WITH dedup cap (per-step unique evals capped at uniqueCount):")
		print("  beam N | topK | depth | unique calcs | seconds")
		for _, cfg in ipairs({ {150,30,15}, {40,30,15}, {40,20,12}, {25,15,10} }) do
			local secs, ncalls = beamSeconds(cfg[1], cfg[2], cfg[3])
			print(string.format("  %5d | %4d | %5d | %12d | %6.1fs",
				cfg[1], cfg[2], cfg[3], ncalls, secs))
		end
		print("  (Cap reflects modKey caching: a step can't exceed uniqueCount distinct evals.)")

		-- Extrapolate: does the design-doc beam fit in a dialog?
		local msPerCall = (t1 - t0) * 1000 / math.max(d1Calls, 1)
		local beamWidth, topK, depth = 150, 30, 15
		local projected = beamWidth * topK * depth
		print(string.format("\nThroughput: ~%.2f ms/call.", msPerCall))
		print(string.format("Projected beam (N=%d, top-K=%d, depth=%d) = %d calls = ~%.1fs.",
			beamWidth, topK, depth, projected, projected * msPerCall / 1000))
		print("If that's minutes, the pre-filter (design doc §3.3) must cut full evals hard.\n")

		-- The spike's only hard assertion: the loop ran and produced deltas. On the default
		-- build (no skill) deltas may legitimately be ~0, so we don't assert on magnitude.
		assert.is_true(d1Calls == #candidates)
		assert.is_true(d2Calls == K * (K - 1) / 2)
	end)
end)
