# Architecture

## Goal & scope

Solve and visualize the **single-electron** time-dependent Schrödinger equation
(TDSE) in atomic units:

```
i ∂ψ/∂t = H ψ,    H = -½ ∇² + V(r, t)
```

First deliverable: a **Gaussian wavepacket** evolving in real time (free space,
then a potential), with `|ψ(r,t)|²` rendered as an electron cloud.

**Out of scope (by design):** multiple electrons, Hartree/HF/DFT, the many-body
wavefunction. The exact `N`-electron wavefunction lives in `3N` dimensions, so a
grid costs `M^(3N)` — at `M=50` that is ~250 GB for `N=2` and ~31 PB for `N=3`.
Direct FDM dies at ~2 electrons; "arbitrary electrons" would mean a different,
research-grade mean-field (DFT) project. We stay single-electron on purpose.

## The Humble Object seam

```
            +-------------------+        +------------------------+
 tests/ --->|   sesolver_core   |<-------|        app/            |
 (gtest)    |  pure, no Qt/GL   |        |  Qt + hand-written GL  |
            |  fully testable   |        |  "Humble Object"       |
            +-------------------+        +------------------------+
```

- **`core` depends on nothing** (no Qt, no OpenGL, no windowing). Every behavior
  has an analytic or golden oracle and is unit-tested, test-first.
- **`app` is a thin shell**: it owns the window + GL context (Qt) and the
  hand-written OpenGL draw calls, and holds **no domain logic**. It is verified
  manually, not by unit tests. Anything in `app` worth testing must be pushed
  down into `core` as pure data/geometry first (e.g. marching-cubes vertex
  generation, transfer-function math, camera matrices).
- Dependency direction points **inward**: `app → core`, `tests → core`. `core`
  never points outward.

## Reuse boundary (purist reinvention)

The boundary applies to THIRD-PARTY libraries only. The **C++ standard
library is always fair game** (user decision): `ses::Complex` is an alias of
`std::complex` (built with `-fcx-limited-range` so multiply/divide use the
naive formulas the exact-value test oracles pin), and `std::vector`,
`<cmath>`, `<numbers>`, `<random>` etc. are used freely.

Hand-written (the learning lives here):
- vector/matrix/camera math (no stdlib equivalent until C++26 `<linalg>`)
- FFT (1D radix-2 → N-D, CPU and GPU compute-shader variants)
- grid, complex field, finite-difference / spectral operators
- time propagation (split-operator Fourier method, real and imaginary time)
- potentials (harmonic, softened Coulomb)
- visualization geometry/color math (marching cubes, transfer functions)
- all OpenGL rendering/compute logic (shaders, buffers, camera, volume cloud)

Reused (pure plumbing, not the learning target):
- **Qt 6** — window, OpenGL context + function loading, widgets/UI only
- **GoogleTest** — test harness

Explicitly *not* reused: GLM (we hand-roll math), GLFW/SDL (Qt is the shell),
FFTW (we hand-roll the FFT), Eigen/BLAS/LAPACK.

## Numerical decisions

- **Atomic units** (ℏ = mₑ = e = 1) everywhere; SI only at the I/O boundary.
- **Complex fields** are first-class (TDSE is inherently complex).
- **Propagator: split-operator (Fourier).** One step of `exp(-iHΔt)` is split as
  `exp(-iVΔt/2) · exp(-iTΔt) · exp(-iVΔt/2)`, with the kinetic factor applied in
  k-space (diagonal there) via the hand-written FFT. Unitary, norm-preserving,
  and it makes the hand-rolled FFT the centerpiece.
- **Never** `-ffast-math` / `/fp:fast` — it breaks NaN handling and bitwise
  reproducibility that the tests rely on.
- Soften the Coulomb `1/r` singularity (`1/√(r²+a²)`) so a grid point on the
  nucleus does not produce NaNs.

## Validation strategy

Validation is the spine, not a phase. Each numerical layer ships with an
analytic oracle as its red test:
- FFT: linearity, `IFFT(FFT(x)) = x`, Parseval, known transforms of δ and cosines.
- Free propagation: a 1D Gaussian wavepacket disperses analytically —
  `σ(t) = σ₀·√(1 + (t/(2σ₀²))²)` (atomic units, mₑ=1), center moves at `k₀`.
  Norm stays 1.
- Bound dynamics: harmonic-oscillator coherent state oscillates rigidly;
  stationary states (later, via imaginary time) match `E` and shape.
- Every discretization carries a **convergence-order** test (error ~ `h^p`).

## Build topology

`CMakeLists.txt` (root) → `core/` (always) → `tests/` (if `SES_BUILD_TESTS`) →
`app/` (if `SES_BUILD_APP` **and** Qt6 found). The app is optional so the TDD
loop never requires a Qt toolchain.
