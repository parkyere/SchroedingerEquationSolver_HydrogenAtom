module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.spinlattice;
export import ses.spin;


// MEAN-FIELD Heisenberg lattice: nx x ny pinned spins, each a pure
// per-site spinor (a PRODUCT ansatz -- no entanglement, honestly a
// quantum-dressed classical Heisenberg/LLG lattice). Site i sees
// B_eff = B_ext - 2 J sum_nn <sigma_j> (J > 0 aligns with neighbors:
// ferromagnet; J < 0 staggers: Neel), steps by the EXACT SU(2) rotation,
// and Gilbert damping alpha bleeds energy toward the local ground
// (n -> -B_eff_hat for H = +1/2 B.sigma). Neighbor fields are read from
// a simultaneous SNAPSHOT (no sweep-order bias). Open boundaries.
// CONTRACT: tests/spinlattice_test.cpp (ferro order, Neel stagger,
// undamped energy conservation, rigid Larmor of the aligned lattice).


export namespace ses {

struct SpinLattice {
    int nx = 0;
    int ny = 0;
    std::vector<Spinor> s;
};

// Pure spinor pointing along the UNIT Bloch vector n: rotate |up> onto n
// about the axis z x n.
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

// One dt of mean-field dynamics: SNAPSHOT every Bloch vector, then per
// site the exact SU(2) rotation in B_eff = B_ext - 2 J sum_nn n_j,
// followed by the Gilbert drift n += alpha dt (n (n.B_eff) - B_eff)
// toward the local ground (n = -B_eff_hat for H = +1/2 B.sigma).
inline void spinlattice_step(SpinLattice& l, double bx, double by,
                             double bz, double j, double alpha,
                             double dt) {
    const std::size_t count = l.s.size();
    std::vector<double> n(3 * count);
    for (std::size_t i = 0; i < count; ++i) {
        bloch_vector(l.s[i], &n[3 * i], &n[3 * i + 1], &n[3 * i + 2]);
    }
    // STRANG-SYMMETRIZED checkerboard (the square lattice is bipartite):
    // black half-step, white full step, black half-step -- second order,
    // and the intra-step skew of the plain leapfrog largely cancels.
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
                        continue;  // open boundary
                    }
                    const std::size_t q =
                        static_cast<std::size_t>(qy * l.nx + qx);
                    ex -= 2.0 * j * n[3 * q];
                    ey -= 2.0 * j * n[3 * q + 1];
                    ez -= 2.0 * j * n[3 * q + 2];
                }
                // Project out the n-parallel field component BEFORE the
                // finite rotation: it exerts no torque (a pure per-site
                // phase) but would tilt the discrete rotation axis.
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
                    // Gilbert drift off the POST-rotation vector (the
                    // full field self-projects: n(n.B) - B = -B_perp).
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

// Mean magnetization <sigma> over the lattice.
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

// Checkerboard-signed (staggered) magnetization magnitude.
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
