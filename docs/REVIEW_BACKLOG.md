# Code-Review Backlog

A whole-project critical review (2026-07-14: architecture/SOLID, antipatterns,
resource leaks, concurrency, test edge cases, comment hygiene, correctness)
produced 26 confirmed findings. As of 2026-07-14 all but two are **fixed**
(the two remainders below are a deliberately-deferred low-value extraction and
one minor consistency cleanup). Every fix must clear the
gates in [`TDD_RULES.md`](TDD_RULES.md): `ctest` (all pass) + `sesolver_vkcheck`
(all PASS) + the `--selftest-*` arcs. Prefer RED-first for any testable logic.

## Already fixed (for context — do not redo)

- **State-synthesis leak** — `release_state()` recycles slots + sets via
  `free_full_states_` (was leaking 4 descriptor sets + a slot per synth/release).
- **Laser + B-field** — now mutually exclusive (`toggle_laser`/`set_bfield_b`);
  was an inconsistent Hamiltonian (paramagnetic rotation dropped) + a false
  "psi evolved" title.
- **Harmonic coherent state** — `sigma = 1/sqrt(2 w)` (was `1/sqrt(w)`, sqrt(2)x
  too wide → breathing instead of rigid oscillation).
- **Qt comment rot** — 24 source sites swept to present-tense contracts.
- **HydrogenDirector → BaseDirector reparent** — retires the sibling
  duplication + god-object findings (removes the duplicated member block and the
  plumbing methods; hydrogen keeps only its specialized overrides).
- **GPU marching-cubes oracle** — cyclic-hue colour metric + valid sort key
  (a discontinuous-wheel abs-RGB compare false-failed on the RTX 5090). *Still
  needs a Linux/5090 `sesolver_vkcheck` re-run to confirm — could not reproduce
  on the RTX 4060.*

### Batch A/B/C (2026-07-14, ee0cfa5..HEAD; ctest 264/264, vkcheck all PASS)

- **[medium] Shell god-object / OCP-LSP leak** — the ~28 down-cast forwarders
  and the concrete `HydrogenDirector*`/`TunnelingDirector*` members are GONE.
  `scenario.hpp` now declares `HydrogenApi` / `TunnelApi` capability
  interfaces + `ScenarioDirector::hydrogen()`/`tunnel()` (null unless the
  scene implements them); the panel takes `HydrogenApi&`, the arcs go through
  `shell->hy()`/`tn()`. (Deviates from the literal `draw_panel(shell)`: that
  would pull ImGui into the framework-neutral director layer.)
- **All 7 latent-correctness guards** — `normalize()` zero-norm; observables
  `obs_ratio`/`obs_sigma` (den==0 + variance clamp, bitwise no-op when den>0);
  `marching_cubes_at_fraction` empty/non-positive-peak; `fft()` power-of-two
  runtime throw (assert survived NDEBUG); `mean_potential_gradient` norm==1
  precondition DOCUMENTED (normalizing would desync the GPU mean_force oracle);
  `upload_field_tables` checks both bool returns + moves the memo after a
  successful upload; `kFlashTicks` unifies the coupled 25 / 25.0f literals.
  `tests/degenerate_guards_test.cpp` (RED-verified 5/6 against unfixed code).
- **3 of 4 latent concurrency items** — mc vbuf/indirect `CONCURRENT`;
  `norm_and_peak`/`project_psi`/`scale` leading `barrier_compute_to_compute`
  (self-contained vs an unwaited async batch); `Engine::destroy()`
  `reset_lazy_state()` nulls the lazy handles + `*_ok_` flags + cache sizes
  (vkcheck `check_engine_reinit`: init->destroy->init->step parity).

## Remaining — architecture (1 low, deliberately deferred)

- **[low] Extract a `MeasurementEngine`** from `HydrogenDirector`
  (`run_partial_measure` / `rebuild_psi_from` / `project_manifold_out`, ~180
  lines). DEFERRED (2026-07-14): per this item's own note the shared
  `cpu_is_truth_`/display-bridge/engine coupling makes the extraction
  low-cohesion — it would need a fat back-reference into HydrogenDirector, so
  the churn on the (just-refactored, manually-verified) shell is not worth the
  negligible cohesion gain. Revisit only if that coupling is broken first.

## Remaining — latent correctness

ALL FIXED (Batch B, 2026-07-14) -- see the "Already fixed" section above.

## Remaining — latent resource / concurrency (1 low)

- **Compute `Kernel` keeps its `VkShaderModule` for the object lifetime**
  (`app/src/vk_compute.hpp`) — the graphics pipelines free theirs right after
  pipeline creation. Retained memory (~29 small modules for the session), not a
  leak. Minor consistency fix: free `module_` at the end of `Kernel::create`.

## Verified CLEAN (do not re-investigate)

Async compute↔render is correctly fence-synchronized (one-in-flight anchor
holds); OpenMP is bitwise-deterministic (fixed serial per-z-slab combine); there
are ZERO `std::mutex`/`std::thread` anywhere; `g_validation_errors` is a correct
`std::atomic`; the test `static const Relaxed` caches are pure memoization, not
order-coupled.

## Test-robustness observation (new, 2026-07-14)

- **`--selftest-cascade` is timing-marginal** — its 90 s wall window at
  time-scale 4x packs only ~2-3 display lifetimes of the 3d state at the
  CURRENT au/s (the 1-step-per-render policy lowered throughput), so it has a
  real Poisson false-fail probability (observed: one `photons = 0` run, then
  `photons = 2` on the immediate re-run). Not a correctness bug; the window /
  time-scale constant wants raising to restore the intended ~11-lifetime
  margin. (decay/rabi/manifold, which share the same photon-count path, are
  comfortably above threshold.)

## See also

- [`TDD_RULES.md`](TDD_RULES.md) — the verification gates every fix must clear.
- Physics-audit open item: no `⟨L_z⟩` / probability-current *diagnostic* (the
  ±m ring states ARE preparable now via the L_z partial measurement); and the
  m-sign's absolute handedness vs Larmor rotation under B is untested — a
  B-on ring-rotation check would pin it.
