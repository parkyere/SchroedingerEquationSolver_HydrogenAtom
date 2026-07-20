module;
#include <utility>
#include <cstddef>
#include <vector>
export module ses.spectrum1d;
export import ses.field;
export import ses.grid;
import ses.radial;


// 1D box [xmin,xmax] with Dirichlet ends is the l=0 radial problem; thin
// adapter over the radial FD solver. Mapping is exact: m = g.n-1 interior
// points give radial spacing L/(m+1) = scene spacing, interior point j ->
// scene point j+1, scene point 0 carries the Dirichlet zero. Dirichlet-vs-
// periodic error is exponentially small for confined states. Eigenfunctions
// real, discretely normalized, sign convention positive near xmin. Verified
// in tests vs the HO ladder and the closed-form Morse spectrum.


export namespace ses {

struct Bound1D {
    double energy{};
    Field1D psi;

    explicit Bound1D(const Grid1D& g) : psi(g) {}
};

inline std::vector<Bound1D> bound_states_1d(const Grid1D& g,
                                            const std::vector<double>& v,
                                            int count) {
    const RadialGrid rg{g.xmax - g.xmin, g.n - 1};
    std::vector<double> vr(static_cast<std::size_t>(g.n - 1));
    for (int j = 0; j + 1 < g.n; ++j) {
        vr[static_cast<std::size_t>(j)] = v[static_cast<std::size_t>(j + 1)];
    }
    const RadialHamiltonian ham = radial_hamiltonian(rg, vr, 0);
    std::vector<Bound1D> out;
    out.reserve(static_cast<std::size_t>(count));
    for (int k = 0; k < count; ++k) {
        const RadialState s = radial_eigenstate(rg, ham, k);
        Bound1D b{g};
        b.energy = s.energy;
        b.psi[0] = 0.0;  // Dirichlet end (periodic wrap point)
        for (int j = 0; j + 1 < g.n; ++j) {
            b.psi[j + 1] = s.u[static_cast<std::size_t>(j)];
        }
        out.push_back(std::move(b));
    }
    return out;
}

}  // namespace ses
