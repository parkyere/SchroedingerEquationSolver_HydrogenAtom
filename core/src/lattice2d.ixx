module;
#include <cassert>
#include <cmath>
#include <complex>
#include <cstddef>
#include <utility>
#include <vector>
export module ses.lattice2d;
export import ses.field;
export import ses.grid;
import ses.parallel;


// 2D finite-difference propagator with Peierls link phases: the engine for a
// LOCALIZED magnetic flux (double-slit solenoid), where FFT split-operator
// fails -- (p-A)^2/2 has no FFT-diagonal factor for a flux line. Kinetic
// bonds split into 4 disjoint parity groups, each an exact 2x2 rotation;
// symmetric-Strang ordering is the only approximation, O(dt^2). Onsite
// carries +2tx+2ty (FD Laplacian diagonal, t = 1/(2h^2) per axis) so energies
// stay honest. Solenoid = string gauge: straddling x-links carry e^{-+i Phi},
// so every plaquette product is 1 except the solenoid's -- B=0 on the
// electron's domain. Boundaries OPEN (no wrap bonds); scenes add absorbers.


export namespace ses {

class PeierlsLattice2D {
public:
    PeierlsLattice2D(const Grid3D& g, const std::vector<double>& potential,
                     double dt)
        : g_(&g), nx_(g.x.n), ny_(g.y.n), dt_(dt), v_(potential) {
        assert(g.z.n == 1);
        assert(static_cast<int>(potential.size()) == g.size());
        tx_ = 0.5 / (g.x.spacing() * g.x.spacing());
        ty_ = 0.5 / (g.y.spacing() * g.y.spacing());
        // half-dt for the palindrome; y-odd (cy2) is the full-dt center.
        cx_ = std::cos(tx_ * 0.5 * dt);
        sx_ = std::sin(tx_ * 0.5 * dt);
        cy_ = std::cos(ty_ * 0.5 * dt);
        sy_ = std::sin(ty_ * 0.5 * dt);
        cy2_ = std::cos(ty_ * dt);
        sy2_ = std::sin(ty_ * dt);
        // imaginary-time twins: cosh/sinh mixing.
        rcx_ = std::cosh(tx_ * 0.5 * dt);
        rsx_ = std::sinh(tx_ * 0.5 * dt);
        rcy_ = std::cosh(ty_ * 0.5 * dt);
        rsy_ = std::sinh(ty_ * 0.5 * dt);
        rcy2_ = std::cosh(ty_ * dt);
        rsy2_ = std::sinh(ty_ * dt);
        half_v_.resize(potential.size());
        relax_half_v_.resize(potential.size());
        for (std::size_t i = 0; i < potential.size(); ++i) {
            const double e0 = potential[i] + 2.0 * tx_ + 2.0 * ty_;
            const double th = -0.5 * e0 * dt;
            half_v_[i] = std::complex<double>{std::cos(th), std::sin(th)};
            relax_half_v_[i] = std::exp(-0.5 * e0 * dt);
        }
        link_x_.assign(static_cast<std::size_t>(nx_ * ny_), 1.0);
        link_y_.assign(static_cast<std::size_t>(nx_ * ny_), 1.0);
    }

    double dt() const noexcept { return dt_; }

    // String gauge: cut_up puts e^{-i phi} on the x-links above the solenoid,
    // cut_down e^{+i phi} below -- gauge-equivalent (tested as such).
    void set_solenoid(double phi, double xs, double ys, bool cut_up = true) {
        link_x_.assign(link_x_.size(), 1.0);
        int is = -1;
        for (int i = 0; i + 1 < nx_; ++i) {
            if (g_->x.coord(i) <= xs && xs < g_->x.coord(i + 1)) {
                is = i;
            }
        }
        int js = -1;
        for (int j = 0; j + 1 < ny_; ++j) {
            if (g_->y.coord(j) <= ys && ys < g_->y.coord(j + 1)) {
                js = j;
            }
        }
        if (is < 0 || js < 0) {
            return;  // solenoid outside the lattice
        }
        const std::complex<double> up{std::cos(phi), -std::sin(phi)};
        const std::complex<double> dn{std::cos(phi), std::sin(phi)};
        if (cut_up) {
            for (int j = js + 1; j < ny_; ++j) {
                link_x_[static_cast<std::size_t>(j * nx_ + is)] = up;
            }
        } else {
            for (int j = 0; j <= js; ++j) {
                link_x_[static_cast<std::size_t>(j * nx_ + is)] = dn;
            }
        }
    }

    // Uniform field B along z, Landau gauge A_x = +B y, A_y = 0: x-links of
    // row j carry e^{-i B hx y_j}. Anchor at y = 0, NOT ymin, or the packet
    // gets a spurious v_x = -B*ymin. Replaces any solenoid.
    void set_uniform_field(double b) {
        const double bh = b * g_->x.spacing();
        for (int j = 0; j < ny_; ++j) {
            const double th = -bh * g_->y.coord(j);
            const std::complex<double> u{std::cos(th), std::sin(th)};
            for (int i = 0; i < nx_; ++i) {
                link_x_[static_cast<std::size_t>(j * nx_ + i)] = u;
            }
        }
    }

    // Directed link U on edge (i,j)->(i+1,j) [x] resp. (i,j)->(i,j+1) [y];
    // exposed for the plaquette-topology contract.
    std::complex<double> link_x(int i, int j) const {
        return link_x_[static_cast<std::size_t>(j * nx_ + i)];
    }
    std::complex<double> link_y(int i, int j) const {
        return link_y_[static_cast<std::size_t>(j * nx_ + i)];
    }

    void step(Field3D& psi, int nsteps = 1) const;

    // Imaginary-time relaxation: same bond splitting with cosh/sinh mixing
    // plus real onsite decay, renormalized each step. Reaches the ground of
    // a dot IN a B field (Fock-Darwin), out of reach for B=0 imaginary time.
    void relax(Field3D& psi, int nsteps = 1) const;

    // <H> = hops + onsite, normalized; live readout + relax convergence check.
    double energy(const Field3D& psi) const;

private:
    void phase(std::vector<std::complex<double>>& a,
               const std::vector<std::complex<double>>& table) const;

    // One x-bond parity group, all rows (rows independent, bonds disjoint by
    // parity -> parallel-safe). mix = i*sin (real time) or sinh (imaginary).
    void sweep_x(std::vector<std::complex<double>>& a, int parity, double c,
                 std::complex<double> mix) const;

    // One y-bond parity group (bond-rows disjoint by parity -> parallel-safe).
    void sweep_y(std::vector<std::complex<double>>& a, int parity, double c,
                 std::complex<double> mix) const;

    const Grid3D* g_;
    int nx_;
    int ny_;
    double dt_;
    double tx_ = 0.0;
    double ty_ = 0.0;
    double cx_, sx_;    // x bonds, dt/2
    double cy_, sy_;    // y-even bonds, dt/2
    double cy2_, sy2_;  // y-odd bonds, full dt (palindrome center)
    double rcx_, rsx_;  // imaginary-time twins (cosh/sinh)
    double rcy_, rsy_;
    double rcy2_, rsy2_;
    std::vector<double> v_;
    std::vector<std::complex<double>> half_v_;
    std::vector<std::complex<double>> relax_half_v_;
    std::vector<std::complex<double>> link_x_;
    std::vector<std::complex<double>> link_y_;
};


// Out-of-class ON PURPOSE: members defined outside the class in a module
// interface are NOT implicitly inline, so they compile once in this TU.
// In-class, an importer's parallel_for instantiation crashed a pool worker
// (MSVC modules codegen). Keep one canonical copy. See ses.parallel.

void PeierlsLattice2D::phase(
    std::vector<std::complex<double>>& a,
    const std::vector<std::complex<double>>& table) const {
    parallel_for(static_cast<int>(a.size()), [&](int i) {
        a[static_cast<std::size_t>(i)] *= table[static_cast<std::size_t>(i)];
    });
}

void PeierlsLattice2D::sweep_x(std::vector<std::complex<double>>& a,
                               int parity, double c,
                               std::complex<double> mix) const {
    parallel_for(ny_, [&](int j) {
        const std::size_t row = static_cast<std::size_t>(j * nx_);
        for (int i = parity; i + 1 < nx_; i += 2) {
            const std::complex<double> u =
                link_x_[row + static_cast<std::size_t>(i)];
            const std::complex<double> pa =
                a[row + static_cast<std::size_t>(i)];
            const std::complex<double> pb =
                a[row + static_cast<std::size_t>(i + 1)];
            a[row + static_cast<std::size_t>(i)] = c * pa + mix * u * pb;
            a[row + static_cast<std::size_t>(i + 1)] =
                mix * std::conj(u) * pa + c * pb;
        }
    });
}

void PeierlsLattice2D::sweep_y(std::vector<std::complex<double>>& a,
                               int parity, double c,
                               std::complex<double> mix) const {
    const int rows = (ny_ - 1 - parity + 1) / 2;  // j = parity, +2, ...
    parallel_for(rows, [&](int r) {
        const int j = parity + 2 * r;
        if (j + 1 >= ny_) {
            return;
        }
        const std::size_t lo = static_cast<std::size_t>(j * nx_);
        const std::size_t hi = static_cast<std::size_t>((j + 1) * nx_);
        for (int i = 0; i < nx_; ++i) {
            const std::complex<double> u =
                link_y_[lo + static_cast<std::size_t>(i)];
            const std::complex<double> pa =
                a[lo + static_cast<std::size_t>(i)];
            const std::complex<double> pb =
                a[hi + static_cast<std::size_t>(i)];
            a[lo + static_cast<std::size_t>(i)] = c * pa + mix * u * pb;
            a[hi + static_cast<std::size_t>(i)] =
                mix * std::conj(u) * pa + c * pb;
        }
    });
}

void PeierlsLattice2D::step(Field3D& psi, int nsteps) const {
    assert(psi.data().size() == half_v_.size());
    std::vector<std::complex<double>>& a = psi.data();
    for (int s = 0; s < nsteps; ++s) {
        phase(a, half_v_);
        sweep_x(a, 0, cx_, {0.0, sx_});
        sweep_x(a, 1, cx_, {0.0, sx_});
        sweep_y(a, 0, cy_, {0.0, sy_});
        sweep_y(a, 1, cy2_, {0.0, sy2_});
        sweep_y(a, 0, cy_, {0.0, sy_});
        sweep_x(a, 1, cx_, {0.0, sx_});
        sweep_x(a, 0, cx_, {0.0, sx_});
        phase(a, half_v_);
    }
}

void PeierlsLattice2D::relax(Field3D& psi, int nsteps) const {
    assert(psi.data().size() == half_v_.size());
    std::vector<std::complex<double>>& a = psi.data();
    for (int s = 0; s < nsteps; ++s) {
        phase(a, relax_half_v_);
        sweep_x(a, 0, rcx_, {rsx_, 0.0});
        sweep_x(a, 1, rcx_, {rsx_, 0.0});
        sweep_y(a, 0, rcy_, {rsy_, 0.0});
        sweep_y(a, 1, rcy2_, {rsy2_, 0.0});
        sweep_y(a, 0, rcy_, {rsy_, 0.0});
        sweep_x(a, 1, rcx_, {rsx_, 0.0});
        sweep_x(a, 0, rcx_, {rsx_, 0.0});
        phase(a, relax_half_v_);
        normalize(psi);
    }
}

double PeierlsLattice2D::energy(const Field3D& psi) const {
    double e = 0.0;
    double den = 0.0;
    for (int j = 0; j < ny_; ++j) {
        for (int i = 0; i < nx_; ++i) {
            const std::complex<double> z = psi(i, j, 0);
            const double w = std::norm(z);
            den += w;
            e += (v_[static_cast<std::size_t>(g_->flat(i, j, 0))] +
                  2.0 * tx_ + 2.0 * ty_) *
                 w;
            if (i + 1 < nx_) {
                e += -tx_ * 2.0 *
                     (std::conj(z) * link_x(i, j) * psi(i + 1, j, 0)).real();
            }
            if (j + 1 < ny_) {
                e += -ty_ * 2.0 *
                     (std::conj(z) * link_y(i, j) * psi(i, j + 1, 0)).real();
            }
        }
    }
    return e / den;
}

// Landau-level ladder in the SAME gauge as set_uniform_field. pi = -i grad
// - A, central differences with periodic wrap; the gauge is not wrap-
// consistent, so states must stay off the boundary (magnetic length
// 1/sqrt(B) << box, as the scene guarantees). UNNORMALIZED; callers renormalize.
// CONTRACT: tests/lattice2d_test.cpp LandauLadderClimbsOneCyclotronQuantum.
inline Field3D landau_ladder(const Field3D& psi, double b, bool up) {
    const Grid3D& g = psi.grid();
    const double h = g.x.spacing();
    const double inv2h = 1.0 / (2.0 * h);
    const double s = 1.0 / std::sqrt(2.0 * b);
    Field3D out{g};
    parallel_for(g.y.n, [&](int j) {
        const int ny = g.y.n;
        const int nx = g.x.n;
        const double y = g.y.coord(j);
        const int jp = (j + 1) % ny;
        const int jm = (j - 1 + ny) % ny;
        const std::complex<double> kI{0.0, 1.0};
        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int im = (i - 1 + nx) % nx;
            const std::complex<double> ddx =
                (psi(ip, j, 0) - psi(im, j, 0)) * inv2h;
            const std::complex<double> ddy =
                (psi(i, jp, 0) - psi(i, jm, 0)) * inv2h;
            // A_x = +B y, sign pinned by the contract (guiding-center pair,
            // <H> unchanged).
            const std::complex<double> pix = -kI * ddx - b * y * psi(i, j, 0);
            const std::complex<double> piy = -kI * ddy;
            out(i, j, 0) = s * (up ? (pix + kI * piy) : (pix - kI * piy));
        }
    });
    return out;
}

// 2D isotropic-HO circular ladder at B=0: a_R = (a_x - i a_y)/sqrt(2)
// (dagger adds omega to <H>, +1 to <L_z>). UNNORMALIZED; callers renormalize.
// CONTRACT: tests/ho2d_test.cpp.
// Fock-Darwin spectrum of a lattice-gauge state: rotate to symmetric gauge,
// project onto the circular HO basis at Omega, ladder the energies. Returns
// (E, |c|^2) up to e_max, largest-E last.
// CONTRACT: tests/ho_spectrum_test.cpp (delta lines + <H> reconstruction).
inline Field3D ho2d_ladder(const Field3D& psi, double omega, bool up,
                           bool left = false);

inline std::vector<std::pair<double, double>> fock_darwin_spectrum(
    const Field3D& psi, double omega0, double b, double e_max) {
    const Grid3D& g = psi.grid();
    const double om = std::sqrt(omega0 * omega0 + 0.25 * b * b);
    const double w_r = om - 0.5 * b;  // right = CCW cyclotron (slow)
    const double w_l = om + 0.5 * b;
    const double cell = g.x.spacing() * g.y.spacing() * g.z.spacing();
    // Rotate the lattice-gauge state into the symmetric gauge: chi = -B x y/2.
    Field3D rot{g};
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double th = -0.5 * b * g.x.coord(i) * y;
            rot(i, j, 0) = psi(i, j, 0) *
                           std::complex<double>{std::cos(th), std::sin(th)};
        }
    }
    // Circular basis at Omega: two ladder chains from the analytic ground,
    // each column normalized before the overlap.
    Field3D ground{g};
    for (int j = 0; j < g.y.n; ++j) {
        const double y = g.y.coord(j);
        for (int i = 0; i < g.x.n; ++i) {
            const double x = g.x.coord(i);
            ground(i, j, 0) = std::exp(-0.5 * om * (x * x + y * y));
        }
    }
    normalize(ground);
    std::vector<std::pair<double, double>> lines;
    Field3D right = ground;
    for (int nr = 0;; ++nr) {
        const double e_r = om + nr * w_r;
        if (e_r > e_max) {
            break;
        }
        Field3D cur = right;
        for (int nl = 0;; ++nl) {
            const double e = e_r + nl * w_l;
            if (e > e_max) {
                break;
            }
            std::complex<double> ov{};
            for (std::size_t c = 0; c < cur.data().size(); ++c) {
                ov += std::conj(cur.data()[c]) * rot.data()[c];
            }
            ov *= cell;
            lines.emplace_back(e, std::norm(ov));
            Field3D next = ho2d_ladder(cur, om, true, true);
            if (norm_sq(next) < 1e-12) {
                break;
            }
            normalize(next);
            cur = std::move(next);
        }
        Field3D nxt = ho2d_ladder(right, om, true, false);
        if (norm_sq(nxt) < 1e-12) {
            break;
        }
        normalize(nxt);
        right = std::move(nxt);
    }
    return lines;
}

inline Field3D ho2d_ladder(const Field3D& psi, double omega, bool up,
                           bool left) {
    const Grid3D& g = psi.grid();
    const double inv2h = 1.0 / (2.0 * g.x.spacing());
    const double cx = std::sqrt(omega / 2.0);
    const double cd = 1.0 / std::sqrt(2.0 * omega);
    const double s = 1.0 / std::sqrt(2.0);
    Field3D out{g};
    parallel_for(g.y.n, [&](int j) {
        const int ny = g.y.n;
        const int nx = g.x.n;
        const double y = g.y.coord(j);
        const int jp = (j + 1) % ny;
        const int jm = (j - 1 + ny) % ny;
        const std::complex<double> kI{0.0, 1.0};
        for (int i = 0; i < nx; ++i) {
            const int ip = (i + 1) % nx;
            const int im = (i - 1 + nx) % nx;
            const double x = g.x.coord(i);
            const std::complex<double> ddx =
                (psi(ip, j, 0) - psi(im, j, 0)) * inv2h;
            const std::complex<double> ddy =
                (psi(i, jp, 0) - psi(i, jm, 0)) * inv2h;
            // a_R = (a_x - i a_y)/sqrt(2); LEFT chirality (+i a_y) flips q,
            // dagger flips the derivative sign.
            const std::complex<double> qI = left ? -kI : kI;
            out(i, j, 0) =
                up ? s * (cx * (x + qI * y) * psi(i, j, 0) -
                          cd * (ddx + qI * ddy))
                   : s * (cx * (x - qI * y) * psi(i, j, 0) +
                          cd * (ddx - qI * ddy));
        }
    });
    return out;
}

}  // namespace ses

