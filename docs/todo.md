# TODO — Tree-Optimizer wiring

Living tracker for the remaining work from `docs/rust-optimizer-integration.md`
("Still open"). Completed items move to `docs/finished.md`.

## Remaining

### UI (`OptimizerTab` / modal) — OUT OF SCOPE for now, the last piece
Design doc §5. A dialog that:
- [ ] Boots the pool (clean-slate + graph-export config) on a worker, with a
      progress/coroutine model so the GUI doesn't freeze during the search.
- [ ] Calls `OptimizerPool:runBeam(params)` (DPS/EHP weights, budget, beam width).
- [ ] Previews the winning tree (highlight `result.ids` via `PassiveTreeView`).
- [ ] Applies via an UNDOABLE spec edit (allocate the winning node set), and runs
      the search on a spec copy so the live build isn't mutated mid-search.
- [ ] Cancel that actually stops the search and frees the pool.

### Packaging-CI hook (optional, low priority)
- [ ] Wire `rust/deploy-optimizer.ps1` into the release/installer workflow so
      shipped builds include `pob_optimizer.dll` in `runtime/`. (Local build+copy
      already works via the script.)

## Notes / risks to revisit
- **"missing node …" boot chatter**: the headless engine prints these while loading
  tree 0_5; harmless (scores still match), but worth a glance if a real build ever
  scores wrong.
- **call_save mutates a worker spec** — the host must call it exactly once, after all
  scoring, on a pool it won't score again. The UI must respect this ordering.
- **Re-validate accel/NoFull per build** (design doc §throughput): tree-granted-skill
  builds can read dps=0 with the wrong calculator flags; the clean-slate path is
  validated on mymonk but not universally.

## Out of scope (this session)
- The UI above (explicitly deferred by the user).
