module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <vector>
export module ses.spinexact;
export import ses.spin;
export import ses.spinlattice;


// EXACT 4x4 Heisenberg lattice: the FULL 2^16 = 65536-amplitude
// wavefunction (basis bit i = 1 means site i spin-DOWN), evolved by a
// Strang split of H = (1/2) sum_i B.sigma_i - J sum_bonds sigma_i.sigma_j:
// half single-site field rotations, then the 24 open-boundary bond gates
// exp(+i J dt sigma.sigma) EXACTLY via sigma.sigma = 2 SWAP - 1, then the
// other half. Entanglement is REAL here: the per-site Bloch arrows
// SHRINK (|<sigma_i>| < 1) as the reduced states mix -- the visible
// difference against the mean-field product ansatz.
// CONTRACT: tests/spinexact_test.cpp.


export namespace ses {

inline constexpr int kExactSide = 4;
inline constexpr int kExactSites = 16;
inline constexpr std::size_t kExactDim = 1u << kExactSites;

struct SpinState16 {
    std::vector<std::complex<double>> c;
};

// Tensor product of the lattice's per-site spinors (bit 0 of the index
// is site 0; bit set = down).
inline SpinState16 exact_from_product(const SpinLattice& /*l*/) {
    return SpinState16{};  // RED stub
}

// Reduced per-site Bloch vector <sigma_i>.
inline void exact_site_bloch(const SpinState16& /*s*/, int /*site*/,
                             double* x, double* y, double* z) {
    *x = 0.0;
    *y = 0.0;
    *z = 0.0;  // RED stub
}

// One Strang step of dt.
inline void exact_step(SpinState16& /*s*/, double /*bx*/, double /*by*/,
                       double /*bz*/, double /*j*/, double /*dt*/) {
    // RED stub
}

// <H> (for the conservation contract).
inline double exact_energy(const SpinState16& /*s*/, double /*bx*/,
                           double /*by*/, double /*bz*/, double /*j*/) {
    return 0.0;  // RED stub
}

// Born-measure site i along +-z: collapse + renormalize, return +-1.
inline int exact_measure_z(SpinState16& /*s*/, int /*site*/,
                           double /*u*/) {
    return 0;  // RED stub
}

// Single-site basis rotation U_i = exp(-i angle/2 n.sigma) applied to
// site i only (the axis-measurement helper).
inline void exact_site_rotate(SpinState16& /*s*/, int /*site*/,
                              double /*nx*/, double /*ny*/, double /*nz*/,
                              double /*angle*/) {
    // RED stub
}

}  // namespace ses
