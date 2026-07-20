module;
#include <complex>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>
export module ses.projection;
export import ses.grid;
export import ses.radial;
export import ses.field;
export import ses.harmonics;


// Orbital-free projection: <n|psi> with no resident 3-D atlas. Central V factorizes
// |n> = (u_nl/r) Y_lm, so ONE grid pass (state-count independent) fills g_lm, then a
// 1-D radial dot per state gives each amplitude. CPU-double oracle for the GPU deposit.
// CONTRACT: deposit weights W_j(r) must mirror fill_orbital (ses.harmonics) exactly.


export namespace ses {

// level indexes the caller's u tables; (l,m) are angular quantum numbers.
struct ProjectorState {
    int level;
    int l;
    int m;
};

inline constexpr int lm_index(int l, int m) noexcept { return l * l + (l + m); }
inline constexpr int lm_count(int l_max) noexcept { return (l_max + 1) * (l_max + 1); }

struct RadialAngularProjection {
    std::vector<std::complex<double>> amp;               // <n|psi>, unit-normalized
    std::vector<double> norm2;                      // N_n = sum_j u[j]^2 h (=1 for eigen-u)
    std::vector<std::vector<std::complex<double>>> g_lm;  // [lm_count][n_radial]
    // raw grid inner product <n|psi> = amp[n] * sqrt(N_n).
};

// Static GPU-deposit geometry (grid-only, built once): cells CSR-sorted by bin i0
// so the GPU runs one workgroup/bin -- deterministic gather, no atomics. i0 uses
// fp32 IDENTICAL to the shader so both agree on every cell (fp32-eps straddlers).
struct RadialBinIndex {
    std::vector<std::uint32_t> sorted_cell;  // in-sphere cell flat indices, grouped by i0
    std::vector<std::uint32_t> bin_off;      // CSR offsets, length n_radial+1
};

// fp32 bin key (-1 = outside sphere). Free function so the Slang deposit kernel
// (project_deposit.comp.slang) and the CPU sort provably share the arithmetic.
inline int radial_bin_key(const Grid3D& g, const RadialGrid& rgrid, int i, int j,
                          int k) noexcept {
    const float hf = static_cast<float>(rgrid.h());
    const float rmaxf = static_cast<float>(rgrid.rmax);
    const float x = static_cast<float>(g.x.xmin) +
                    static_cast<float>(i) * static_cast<float>(g.x.spacing());
    const float y = static_cast<float>(g.y.xmin) +
                    static_cast<float>(j) * static_cast<float>(g.y.spacing());
    const float z = static_cast<float>(g.z.xmin) +
                    static_cast<float>(k) * static_cast<float>(g.z.spacing());
    const float r = std::sqrt(x * x + y * y + z * z);
    if (r >= rmaxf) {
        return -1;
    }
    if (r < hf) {
        return 0;
    }
    int i0 = static_cast<int>(r / hf - 1.0f);
    if (i0 >= rgrid.n) {
        i0 = rgrid.n - 1;  // r<rmax should already bound i0<=n-1
    }
    return i0;
}

inline RadialBinIndex build_radial_bin_index(const Grid3D& g, const RadialGrid& rgrid) {
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const int nr = rgrid.n;
    std::vector<std::uint32_t> counts(static_cast<std::size_t>(nr), 0);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int key = radial_bin_key(g, rgrid, i, j, k);
                if (key >= 0) {
                    ++counts[static_cast<std::size_t>(key)];
                }
            }
        }
    }
    RadialBinIndex out;
    out.bin_off.assign(static_cast<std::size_t>(nr + 1), 0);
    for (int b = 0; b < nr; ++b) {
        out.bin_off[static_cast<std::size_t>(b + 1)] =
            out.bin_off[static_cast<std::size_t>(b)] + counts[static_cast<std::size_t>(b)];
    }
    out.sorted_cell.resize(out.bin_off[static_cast<std::size_t>(nr)]);
    std::vector<std::uint32_t> pos(out.bin_off.begin(), out.bin_off.end() - 1);
    for (int k = 0; k < nz; ++k) {
        for (int j = 0; j < ny; ++j) {
            for (int i = 0; i < nx; ++i) {
                const int key = radial_bin_key(g, rgrid, i, j, k);
                if (key >= 0) {
                    const std::uint32_t idx =
                        static_cast<std::uint32_t>(i + nx * (j + ny * k));
                    out.sorted_cell[pos[static_cast<std::size_t>(key)]++] = idx;
                }
            }
        }
    }
    return out;
}

inline RadialAngularProjection project_radial_angular(
    const Field3D& psi, const RadialGrid& rgrid,
    const std::vector<std::vector<double>>& u_by_level,
    const std::vector<ProjectorState>& states, int l_max = 5) {
    const Grid3D& g = psi.grid();
    const int nx = g.x.n;
    const int ny = g.y.n;
    const int nz = g.z.n;
    const double h = rgrid.h();
    const double rmax = rgrid.rmax;
    const int nr = rgrid.n;
    const double dV = g.cell_volume();
    const int ncomp = lm_count(l_max);

    RadialAngularProjection out;
    out.g_lm.assign(static_cast<std::size_t>(ncomp),
                    std::vector<std::complex<double>>(static_cast<std::size_t>(nr),
                                                 std::complex<double>{}));

    for (int k = 0; k < nz; ++k) {
        const double z = g.z.coord(k);
        for (int j = 0; j < ny; ++j) {
            const double y = g.y.coord(j);
            for (int i = 0; i < nx; ++i) {
                const double x = g.x.coord(i);
                const double r = std::sqrt(x * x + y * y + z * z);
                if (r >= rmax) {
                    continue;  // outside radial box: no deposit (norm deficit)
                }
                int b0 = 0;
                int b1 = -1;
                double w0 = 0.0;
                double w1 = 0.0;
                if (r < h) {
                    b0 = 0;
                    w0 = 1.0 / h;
                } else {
                    const double t = r / h - 1.0;  // r_i = (i+1) h
                    const int i0 = static_cast<int>(t);
                    const double frac = t - static_cast<double>(i0);
                    b0 = i0;
                    w0 = (1.0 - frac) / r;
                    if (i0 + 1 < nr) {
                        b1 = i0 + 1;
                        w1 = frac / r;  // outermost node = pinned u(rmax)=0: dropped
                    }
                }
                const std::complex<double> pdV = psi(i, j, k) * dV;
                for (int l = 0; l <= l_max; ++l) {
                    for (int m = -l; m <= l; ++m) {
                        const double Y = real_spherical_harmonic(l, m, x, y, z);
                        const std::complex<double> contrib = pdV * Y;
                        const std::size_t c = static_cast<std::size_t>(lm_index(l, m));
                        out.g_lm[c][static_cast<std::size_t>(b0)] += contrib * w0;
                        if (b1 >= 0) {
                            out.g_lm[c][static_cast<std::size_t>(b1)] += contrib * w1;
                        }
                    }
                }
            }
        }
    }

    out.amp.assign(states.size(), std::complex<double>{});
    out.norm2.assign(states.size(), 0.0);
    for (std::size_t s = 0; s < states.size(); ++s) {
        const ProjectorState& st = states[s];
        const std::vector<double>& u =
            u_by_level[static_cast<std::size_t>(st.level)];
        const std::vector<std::complex<double>>& gc =
            out.g_lm[static_cast<std::size_t>(lm_index(st.l, st.m))];
        std::complex<double> raw{};
        double n2 = 0.0;
        for (int jr = 0; jr < nr; ++jr) {
            raw += u[static_cast<std::size_t>(jr)] * gc[static_cast<std::size_t>(jr)];
            n2 += u[static_cast<std::size_t>(jr)] * u[static_cast<std::size_t>(jr)] * h;
        }
        out.norm2[s] = n2;
        out.amp[s] = (n2 > 0.0) ? raw * (1.0 / std::sqrt(n2)) : std::complex<double>{};
    }
    return out;
}

}  // namespace ses
