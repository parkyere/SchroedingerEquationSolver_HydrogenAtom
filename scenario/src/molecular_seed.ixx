module;
#include <cstddef>
#include <cstdint>
#include <complex>
#include <vector>
export module ses.scenario.molecular_seed;
export import ses.field;
export import ses.grid;


// State seeds for the one-electron molecule scenes (H2+): the known
// molecular orbitals resolved by SYMMETRY sector, and an arbitrary
// random-shaped wavefunction.
//
// A D-infinity-h molecule commutes with inversion (parity g/u), reflection
// through planes containing the bond axis (x), and Lz. Seeding the deflated
// ITP in an irrep sector lands it on the LOWEST orbital of that irrep --
// symmetry keeps it orthogonal to the other sectors; deflation handles the
// same-irrep tower (2sigma_g above 1sigma_g). Seed shapes (bond axis = x):
//   SigmaG  : even everywhere            -> 1sigma_g (bonding, no node)
//   SigmaU  : x-odd (node ⊥ axis)        -> 1sigma_u* (antibonding)
//   PiUy    : y-odd (node plane ∋ axis)  -> 1pi_u
//   PiUz    : z-odd (degenerate partner) -> 1pi_u'
//   SigmaG2 : even, one radial node      -> 2sigma_g (deflate vs 1sigma_g)


export namespace ses_shell {

enum class MolOrbital { SigmaG, SigmaU, PiUy, PiUz, SigmaG2 };

// STUB (red): a normalized uniform field -- even, seed-independent.
ses::Field3D molecular_orbital_seed(const ses::Grid3D& g, MolOrbital /*sym*/) {
    ses::Field3D f{g};
    for (auto& c : f.data()) {
        c = std::complex<double>{1.0, 0.0};
    }
    ses::normalize(f);
    return f;
}

ses::Field3D random_molecular_seed(const ses::Grid3D& g,
                                   std::uint64_t /*seed*/) {
    ses::Field3D f{g};
    for (auto& c : f.data()) {
        c = std::complex<double>{1.0, 0.0};
    }
    ses::normalize(f);
    return f;
}

}  // namespace ses_shell
