module;
#include <algorithm>
#include <cmath>
#include <complex>
export module ses.spin;


// Pinned single-electron spin: 2-spinor under H = (1/2) B . sigma
// (atomic units, gamma folded, Larmor omega = |B|); exact SU(2) step, no Trotter.
// Static E on a pinned spin = global phase only (no electric dipole).
// CONTRACT: tests/spin_test.cpp (Larmor rate, Rabi flip, collapse, echo).


export namespace ses {

struct Spinor {
    std::complex<double> up{1.0, 0.0};
    std::complex<double> dn{0.0, 0.0};
};

// Rotate by `angle` about UNIT axis n: exp(-i angle/2 n.sigma) = cos - i sin n.sigma.
inline void spin_rotate(Spinor& s, double nx, double ny, double nz,
                        double angle) {
    const double c = std::cos(0.5 * angle);
    const double sn = std::sin(0.5 * angle);
    const std::complex<double> i{0.0, 1.0};
    const std::complex<double> up =
        (c - i * sn * nz) * s.up - i * sn * (nx - i * ny) * s.dn;
    const std::complex<double> dn =
        -i * sn * (nx + i * ny) * s.up + (c + i * sn * nz) * s.dn;
    s.up = up;
    s.dn = dn;
}

inline void spin_step(Spinor& s, double bx, double by, double bz,
                      double dt) {
    const double b = std::sqrt(bx * bx + by * by + bz * bz);
    if (b <= 0.0) {
        return;
    }
    spin_rotate(s, bx / b, by / b, bz / b, b * dt);
}

inline void bloch_vector(const Spinor& s, double* x, double* y,
                         double* z) {
    const std::complex<double> cr = std::conj(s.up) * s.dn;
    *x = 2.0 * cr.real();
    *y = 2.0 * cr.imag();
    *z = std::norm(s.up) - std::norm(s.dn);
}

// Born measurement along UNIT axis n: return +1 (|+n>) if u<p_plus else -1;
// collapsed state = |up> rotated onto the outcome axis.
inline int spin_measure(Spinor& s, double nx, double ny, double nz,
                        double u) {
    double x = 0.0;
    double y = 0.0;
    double z = 0.0;
    bloch_vector(s, &x, &y, &z);
    const double p_plus = 0.5 * (1.0 + x * nx + y * ny + z * nz);
    const int outcome = u < p_plus ? +1 : -1;
    const double tx = outcome * nx;
    const double ty = outcome * ny;
    const double tz = outcome * nz;
    Spinor eig;
    const double th = std::acos(std::clamp(tz, -1.0, 1.0));
    const double axn = std::hypot(-ty, tx);  // z x t
    if (axn > 1e-12) {
        spin_rotate(eig, -ty / axn, tx / axn, 0.0, th);
    } else if (tz < 0.0) {
        spin_rotate(eig, 1.0, 0.0, 0.0, 3.14159265358979323846);
    }
    s = eig;
    return outcome;
}

}  // namespace ses
