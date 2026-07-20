module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.spinlattice;
export import ses.spin;


// MEAN-FIELD Heisenberg lattice: B_eff = B_ext - 2 J sum_nn <sigma_j>
// (J>0 ferro, J<0 Neel), H = +1/2 B.sigma; exact SU(2) step + Gilbert alpha.
// CONTRACT: tests/spinlattice_test.cpp (ferro, Neel, energy conservation, Larmor).


export namespace ses {

struct SpinLattice {
    int nx = 0;
    int ny = 0;
    std::vector<Spinor> s;
};

inline Spinor spinor_from_bloch(double x, double y, double z) {
    Spinor s;
    const double th = std::acos(std::clamp(z, -1.0, 1.0));
    const double axn = std::hypot(-y, x);
    if (axn > 1e-12) {
        spin_rotate(s, -y / axn, x / axn, 0.0, th);
    } else if (z < 0.0) {
        spin_rotate(s, 1.0, 0.0, 0.0, 3.14159265358979323846);
    }
    return s;
}

// Snapshot all Bloch vectors first: neighbor fields carry no sweep-order bias.
inline void spinlattice_step(SpinLattice& l, double bx, double by,
                             double bz, double j, double alpha,
                             double dt) {
    const std::size_t count = l.s.size();
    std::vector<double> n(3 * count);
    for (std::size_t i = 0; i < count; ++i) {
        bloch_vector(l.s[i], &n[3 * i], &n[3 * i + 1], &n[3 * i + 2]);
    }
    // Strang-symmetrized checkerboard (bipartite): 2nd order, cancels leapfrog skew.
    auto sweep = [&](int parity, double h) {
        for (int yy = 0; yy < l.ny; ++yy) {
            for (int xx = 0; xx < l.nx; ++xx) {
                if (((xx + yy) & 1) != parity) {
                    continue;
                }
                const std::size_t i =
                    static_cast<std::size_t>(yy * l.nx + xx);
                double ex = bx;
                double ey = by;
                double ez = bz;
                const int dx[4] = {1, -1, 0, 0};
                const int dy[4] = {0, 0, 1, -1};
                for (int d = 0; d < 4; ++d) {
                    const int qx = xx + dx[d];
                    const int qy = yy + dy[d];
                    if (qx < 0 || qx >= l.nx || qy < 0 || qy >= l.ny) {
                        continue;
                    }
                    const std::size_t q =
                        static_cast<std::size_t>(qy * l.nx + qx);
                    ex -= 2.0 * j * n[3 * q];
                    ey -= 2.0 * j * n[3 * q + 1];
                    ez -= 2.0 * j * n[3 * q + 2];
                }
                // Project out the n-parallel field: no torque but would tilt the axis.
                const double nb = n[3 * i] * ex + n[3 * i + 1] * ey +
                                  n[3 * i + 2] * ez;
                double px = 0.0;
                double py = 0.0;
                double pz = 0.0;
                spin_step(l.s[i], ex - nb * n[3 * i],
                          ey - nb * n[3 * i + 1],
                          ez - nb * n[3 * i + 2], h);
                bloch_vector(l.s[i], &px, &py, &pz);
                if (alpha > 0.0) {
                    // Gilbert drift off the post-rotation vector: n(n.B) - B = -B_perp.
                    const double pb = px * ex + py * ey + pz * ez;
                    px += alpha * h * (px * pb - ex);
                    py += alpha * h * (py * pb - ey);
                    pz += alpha * h * (pz * pb - ez);
                    const double nn =
                        std::sqrt(px * px + py * py + pz * pz);
                    if (nn > 1e-12) {
                        px /= nn;
                        py /= nn;
                        pz /= nn;
                        l.s[i] = spinor_from_bloch(px, py, pz);
                    }
                }
                n[3 * i] = px;
                n[3 * i + 1] = py;
                n[3 * i + 2] = pz;
            }
        }
    };
    sweep(0, 0.5 * dt);
    sweep(1, dt);
    sweep(0, 0.5 * dt);
}

inline void lattice_magnetization(const SpinLattice& l, double* x,
                                  double* y, double* z) {
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    for (const Spinor& s : l.s) {
        double px = 0.0;
        double py = 0.0;
        double pz = 0.0;
        bloch_vector(s, &px, &py, &pz);
        sx += px;
        sy += py;
        sz += pz;
    }
    const double inv = l.s.empty() ? 0.0 : 1.0 / l.s.size();
    *x = sx * inv;
    *y = sy * inv;
    *z = sz * inv;
}

inline double lattice_staggered(const SpinLattice& l) {
    double sx = 0.0;
    double sy = 0.0;
    double sz = 0.0;
    for (int yy = 0; yy < l.ny; ++yy) {
        for (int xx = 0; xx < l.nx; ++xx) {
            const std::size_t i =
                static_cast<std::size_t>(yy * l.nx + xx);
            const double sgn = ((xx + yy) & 1) != 0 ? -1.0 : 1.0;
            double px = 0.0;
            double py = 0.0;
            double pz = 0.0;
            bloch_vector(l.s[i], &px, &py, &pz);
            sx += sgn * px;
            sy += sgn * py;
            sz += sgn * pz;
        }
    }
    const double inv = l.s.empty() ? 0.0 : 1.0 / l.s.size();
    return std::sqrt(sx * sx + sy * sy + sz * sz) * inv;
}

// Mean-field energy E = -J sum_bonds n_i.n_j + 1/2 B . sum_i n_i.
inline double lattice_energy(const SpinLattice& l, double bx, double by,
                             double bz, double j) {
    std::vector<double> n(3 * l.s.size());
    for (std::size_t i = 0; i < l.s.size(); ++i) {
        bloch_vector(l.s[i], &n[3 * i], &n[3 * i + 1], &n[3 * i + 2]);
    }
    double e = 0.0;
    for (int yy = 0; yy < l.ny; ++yy) {
        for (int xx = 0; xx < l.nx; ++xx) {
            const std::size_t i =
                static_cast<std::size_t>(yy * l.nx + xx);
            e += 0.5 * (bx * n[3 * i] + by * n[3 * i + 1] +
                        bz * n[3 * i + 2]);
            if (xx + 1 < l.nx) {
                const std::size_t q = i + 1;
                e -= j * (n[3 * i] * n[3 * q] +
                          n[3 * i + 1] * n[3 * q + 1] +
                          n[3 * i + 2] * n[3 * q + 2]);
            }
            if (yy + 1 < l.ny) {
                const std::size_t q =
                    i + static_cast<std::size_t>(l.nx);
                e -= j * (n[3 * i] * n[3 * q] +
                          n[3 * i + 1] * n[3 * q + 1] +
                          n[3 * i + 2] * n[3 * q + 2]);
            }
        }
    }
    return e;
}

}  // namespace ses
