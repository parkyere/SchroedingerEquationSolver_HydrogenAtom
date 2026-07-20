module;
#include <cmath>
#include <complex>
export module ses.spin;


// A single electron spin, pinned in space: the 2-spinor under
// H = (1/2) B . sigma (atomic units, gamma folded in -- Larmor omega =
// |B|). Every step is the EXACT SU(2) rotation exp(-i |B| dt/2 B_hat .
// sigma) (Rodrigues on 2x2: cos - i sin n.sigma), so the norm is exact
// to round-off and there is no Trotter anywhere. A static E field adds
// only a global phase to a pinned spin (no electric dipole): the scene
// displays it honestly as flux, never as Bloch motion.
// CONTRACT: tests/spin_test.cpp (Larmor rate, Rabi flip, collapse, echo).


export namespace ses {

struct Spinor {
    std::complex<double> up{1.0, 0.0};
    std::complex<double> dn{0.0, 0.0};
};

// Exact rotation by `angle` about the UNIT axis n.
inline void spin_rotate(Spinor& /*s*/, double /*nx*/, double /*ny*/,
                        double /*nz*/, double /*angle*/) {
    // RED stub
}

// One dt under H = (1/2) B . sigma: rotation by |B| dt about B_hat.
inline void spin_step(Spinor& /*s*/, double /*bx*/, double /*by*/,
                      double /*bz*/, double /*dt*/) {
    // RED stub
}

// The Bloch vector <sigma>.
inline void bloch_vector(const Spinor& /*s*/, double* x, double* y,
                         double* z) {
    *x = 0.0;
    *y = 0.0;
    *z = 0.0;  // RED stub
}

// Born measurement along the UNIT axis n: collapse to |+n> (return +1)
// when u < p_plus, else |-n> (return -1).
inline int spin_measure(Spinor& /*s*/, double /*nx*/, double /*ny*/,
                        double /*nz*/, double /*u*/) {
    return 0;  // RED stub
}

}  // namespace ses
