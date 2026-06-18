//! Persistent score cache — the cross-run form of the in-search memo.
//!
//! The search's `memo` (`set -> [score, dps, ehp]`) already does exact-set reuse
//! WITHIN one run: any candidate whose precise id-set was scored before skips the
//! ~8ms calc. This module persists that map to disk so a re-run on the SAME build
//! starts warm — the common tuning loop (run, tweak weights, run again).
//!
//! What's cached, and why it's safe:
//!   * Only the RAW, weight-independent calc outputs `dps`/`ehp` per id-set are
//!     stored. The growth score (col 0) is RE-DERIVED on load via `Graph::growth_score`
//!     from the CURRENT weights — so a cache built under one DPS/EHP weighting stays
//!     valid under any other (only the weights change; dps/ehp for a fixed set don't).
//!   * The file is tagged with a `build_signature`: a hash of the graph topology +
//!     budget + the build's reference dps/ehp (`refDps`/`refEhp` from the export
//!     trailer). Any gear/skill/tree change moves the topology or the refs, so the
//!     signature mismatches and the stale cache is silently discarded. The signature
//!     deliberately EXCLUDES the weight/penalty trailer ints, which is exactly what
//!     lets a weight-only change reuse the file.
//!
//! Format (little-endian, no deps):
//!   magic "POBC" | u32 version | u64 build_sig | u64 count
//!   then `count` entries: u32 nids | i32×nids ids (sorted) | f64 dps | f64 ehp

use crate::beam::Graph;
use rustc_hash::FxHashMap;
use std::io::{Read, Write};
use std::path::Path;

const MAGIC: &[u8; 4] = b"POBC";
const VERSION: u32 = 1;

/// Hash the build's identity from the flat graph export, EXCLUDING the weight/penalty
/// trailer so a weight-only re-run reuses the cache. Covers everything that changes a
/// SET's raw dps/ehp: topology, budget, and the build's reference stats (refDps/refEhp,
/// the first two trailer ints, which move whenever gear/skills change). If the trailer
/// is absent (older export) we hash the whole array — those runs aren't cached anyway.
pub fn build_signature(flat: &[i32]) -> u64 {
    // CRITICAL: the export (`__pob_graph_export`) emits nodes in `pairs(nodeById)` order,
    // which Lua does NOT guarantee stable across processes. So the signature must be
    // ORDER-INDEPENDENT — a sequential hash of `flat` would differ every run and the
    // cache would never be reused. We parse the structure and fold each node's identity
    // commutatively (XOR), so node emission order can't change the result.
    //
    // Layout: [ startId, budgetP, nNodes, (id, typeCode, nLinks, link*) * nNodes,
    //           refDps, refEhp, wDps, wEhp, penaltyK ]. We hash the header + every node's
    //           (id, type, SORTED links) + refDps/refEhp, but NOT the weight/penalty ints
    //           (so a weight-only change reuses the cache).
    if flat.len() < 3 {
        return splitmix64(flat.len() as u64);
    }
    let start = flat[0];
    let budget = flat[1];
    let n = flat[2].max(0) as usize;

    // splitmix64 finalizer — a strong integer mix so XOR-combining doesn't cancel.
    let mut acc: u64 = splitmix64(start as u32 as u64)
        ^ splitmix64((budget as u32 as u64).wrapping_add(0x1234))
        ^ splitmix64((n as u64).wrapping_add(0x5678));

    let mut p = 3usize;
    for _ in 0..n {
        if p + 3 > flat.len() {
            break; // malformed; hash what we have
        }
        let id = flat[p];
        let tc = flat[p + 1];
        let k = flat[p + 2].max(0) as usize;
        p += 3;
        if p + k > flat.len() {
            break;
        }
        let mut links: Vec<i32> = flat[p..p + k].to_vec();
        p += k;
        // Sort links so adjacency order (also `ipairs` over `node.linked`, stable but
        // belt-and-suspenders) can't perturb a node's contribution.
        links.sort_unstable();
        // Per-node hash: mix id + type, then fold each link in commutatively-but-positionally
        // via a running splitmix (links are sorted, so this is deterministic).
        let mut nh = splitmix64((id as u32 as u64) << 8 ^ tc as u32 as u64);
        for &l in &links {
            nh = splitmix64(nh ^ splitmix64(l as u32 as u64));
        }
        // XOR each node into the accumulator: order of nodes no longer matters.
        acc ^= nh;
    }

    // Fold in the build's reference stats (the trailer's first two ints), which move
    // whenever gear/skills change. Trailer present iff there are >=5 ints after the nodes.
    if p + 5 <= flat.len() {
        acc ^= splitmix64((flat[p] as u32 as u64).wrapping_add(0xA5A5));
        acc ^= splitmix64((flat[p + 1] as u32 as u64).wrapping_add(0x5A5A));
    }
    acc
}

/// splitmix64 finalizer — a high-quality integer hash.
fn splitmix64(x: u64) -> u64 {
    let mut z = x.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

/// Load a cache file into a fresh memo, validating the file's build signature against
/// `expect_sig` and re-deriving col 0 (growth score) from the stored raw dps/ehp under
/// the CURRENT graph weights. Returns an empty map (never an error) if the file is
/// missing, malformed, or tagged for a different build — a stale or unreadable cache
/// must never block or corrupt a run, only fail to warm it.
pub fn load_with_sig(path: &Path, expect_sig: u64, graph: &Graph) -> FxHashMap<Vec<i32>, [f64; 3]> {
    let mut memo: FxHashMap<Vec<i32>, [f64; 3]> = FxHashMap::default();
    let mut f = match std::fs::File::open(path) {
        Ok(f) => f,
        Err(_) => return memo, // no cache yet — cold start
    };
    let mut buf = Vec::new();
    if f.read_to_end(&mut buf).is_err() {
        return memo;
    }
    let mut p = 0usize;
    // Each reader bounds-checks against `buf` and bails the whole load to the (possibly
    // partial) memo on a short/truncated file — a corrupt cache degrades to a cold start.
    macro_rules! rd_u32 {
        () => {{
            if p + 4 > buf.len() { return memo; }
            let v = u32::from_le_bytes([buf[p], buf[p + 1], buf[p + 2], buf[p + 3]]);
            p += 4;
            v
        }};
    }
    macro_rules! rd_u64 {
        () => {{
            if p + 8 > buf.len() { return memo; }
            let v = u64::from_le_bytes([
                buf[p], buf[p + 1], buf[p + 2], buf[p + 3],
                buf[p + 4], buf[p + 5], buf[p + 6], buf[p + 7],
            ]);
            p += 8;
            v
        }};
    }
    macro_rules! rd_f64 {
        () => {{ f64::from_bits(rd_u64!()) }};
    }
    // Header.
    if p + 4 > buf.len() || &buf[p..p + 4] != MAGIC {
        return memo; // not our file
    }
    p += 4;
    if rd_u32!() != VERSION {
        return memo;
    }
    if rd_u64!() != expect_sig {
        return memo; // cache is for a different build — discard
    }
    let count = rd_u64!() as usize;
    memo.reserve(count);
    for _ in 0..count {
        let nids = rd_u32!() as usize;
        // Guard against a corrupt huge count blowing memory.
        if nids > 1_000_000 {
            return memo;
        }
        let mut ids = Vec::with_capacity(nids);
        for _ in 0..nids {
            ids.push(rd_u32!() as i32);
        }
        let dps = rd_f64!();
        let ehp = rd_f64!();
        let score = graph.growth_score(dps, ehp);
        memo.insert(ids, [score, dps, ehp]);
    }
    memo
}

/// Atomically write the memo to `path`, tagged with `build_sig`. Only entries with
/// finite dps/ehp are stored (a NaN/reject slot carries no reusable information).
/// Writes to a temp file then renames, so a crash mid-write can't corrupt an existing
/// good cache. Errors are returned but callers may choose to ignore them (a failed
/// cache write must never fail a search).
pub fn save(path: &Path, build_sig: u64, memo: &FxHashMap<Vec<i32>, [f64; 3]>) -> std::io::Result<usize> {
    let finite: Vec<(&Vec<i32>, &[f64; 3])> =
        memo.iter().filter(|(_, v)| v[1].is_finite() && v[2].is_finite()).collect();

    let mut out: Vec<u8> = Vec::with_capacity(24 + finite.len() * 24);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&VERSION.to_le_bytes());
    out.extend_from_slice(&build_sig.to_le_bytes());
    out.extend_from_slice(&(finite.len() as u64).to_le_bytes());
    for (ids, v) in &finite {
        out.extend_from_slice(&(ids.len() as u32).to_le_bytes());
        for &id in ids.iter() {
            out.extend_from_slice(&(id as u32).to_le_bytes());
        }
        out.extend_from_slice(&v[1].to_bits().to_le_bytes()); // dps
        out.extend_from_slice(&v[2].to_bits().to_le_bytes()); // ehp
    }

    // PID-suffixed temp so two concurrent writers to the same path don't clobber each
    // other's partial file before the atomic rename. (The host drives serially, but the
    // cost is nil and it makes the write self-contained.)
    let tmp = path.with_extension(format!("tmp{}", std::process::id()));
    {
        let mut f = std::fs::File::create(&tmp)?;
        f.write_all(&out)?;
        f.sync_all()?;
    }
    std::fs::rename(&tmp, path)?;
    Ok(finite.len())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn graph_with_refs(ref_dps: f64, ref_ehp: f64, w_dps: f64, w_ehp: f64) -> Graph {
        Graph {
            start: 0,
            budget: 0,
            links: HashMap::new(),
            ty: HashMap::new(),
            ref_dps,
            ref_ehp,
            w_dps,
            w_ehp,
            penalty_k: 5.0,
        }
    }

    #[test]
    fn signature_ignores_weight_trailer_but_not_refs() {
        // flat = [start, budget, nNodes=0, refDps, refEhp, wDps, wEhp, penaltyK]
        let base = vec![0, 110, 0, 1942_600, 30943_100, 1000, 1000, 5000];
        let weights_changed = vec![0, 110, 0, 1942_600, 30943_100, 2000, 500, 5000];
        let refs_changed = vec![0, 110, 0, 1800_000, 30943_100, 1000, 1000, 5000];
        assert_eq!(
            build_signature(&base),
            build_signature(&weights_changed),
            "changing only weights/penalty must NOT change the signature"
        );
        assert_ne!(
            build_signature(&base),
            build_signature(&refs_changed),
            "changing the build's ref dps MUST change the signature"
        );
    }

    #[test]
    fn signature_is_node_order_independent() {
        // THE bug this guards: `__pob_graph_export` emits nodes in nondeterministic
        // `pairs()` order, so the signature must not depend on node emission order.
        // Two exports of the SAME graph with nodes A(id=5,type=2,links[7,9]) and
        // B(id=7,type=1,links[5]) emitted in opposite order must hash identically.
        // Layout per node: id, typeCode, nLinks, links...
        // header [start=5, budget=3, nNodes=2], trailer [refDps,refEhp,w,w,k].
        let a = [5, 2, 2, 7, 9];
        let b = [7, 1, 1, 5];
        let trailer = [1000_000, 2000_000, 1000, 1000, 5000];

        let mut order_ab = vec![5, 3, 2];
        order_ab.extend_from_slice(&a);
        order_ab.extend_from_slice(&b);
        order_ab.extend_from_slice(&trailer);

        let mut order_ba = vec![5, 3, 2];
        order_ba.extend_from_slice(&b);
        order_ba.extend_from_slice(&a);
        order_ba.extend_from_slice(&trailer);

        assert_eq!(
            build_signature(&order_ab),
            build_signature(&order_ba),
            "node emission order must not change the signature"
        );

        // But a genuinely different topology (drop a link) MUST differ.
        let mut diff = vec![5, 3, 2];
        diff.extend_from_slice(&[5, 2, 1, 7]); // A now links only [7]
        diff.extend_from_slice(&b);
        diff.extend_from_slice(&trailer);
        assert_ne!(
            build_signature(&order_ab),
            build_signature(&diff),
            "a real topology change MUST change the signature"
        );
    }

    #[test]
    fn roundtrip_recomputes_score_under_current_weights() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("pobc_test_{}.bin", std::process::id()));

        let mut memo: FxHashMap<Vec<i32>, [f64; 3]> = FxHashMap::default();
        // score col is whatever; only dps/ehp are persisted.
        memo.insert(vec![0, 5, 9], [999.0, 1000.0, 2000.0]);
        memo.insert(vec![0, 7], [f64::NAN, f64::NAN, f64::NAN]); // reject — must be dropped

        let sig = 0xABCD_1234;
        let n = save(&path, sig, &memo).unwrap();
        assert_eq!(n, 1, "only the finite entry is persisted");

        // Load under DIFFERENT weights than were ever used to write: col 0 is derived
        // fresh from the stored raw dps/ehp.
        let g = graph_with_refs(1000.0, 1000.0, 2.0, 0.5);
        let loaded = load_with_sig(&path, sig, &g);
        assert_eq!(loaded.len(), 1);
        let e = loaded[&vec![0, 5, 9]];
        assert_eq!(e[1], 1000.0);
        assert_eq!(e[2], 2000.0);
        // growth = 100*(2.0*1000/1000 + 0.5*2000/1000) = 100*(2 + 1) = 300
        assert!((e[0] - 300.0).abs() < 1e-9, "col 0 recomputed under load-time weights, got {}", e[0]);

        // Wrong signature => empty (stale build).
        assert!(load_with_sig(&path, sig + 1, &g).is_empty());

        let _ = std::fs::remove_file(&path);
    }
}
