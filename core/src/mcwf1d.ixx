module;
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <vector>
export module ses.mcwf1d;
export import ses.field;
export import ses.grid;
export import ses.imaginary_time;
import ses.ladder;
import ses.observables;


// 1D MCWF unraveling of cavity photon loss (rate kappa, jump op a).
// No-jump exactness (harmonic V): e^{-H tau} = const x e^{-n omega tau},
// so relax at dtau = kappa dt/(2 omega) reproduces e^{-kappa n dt/2}.
// CONTRACT: tests/mcwf1d_test.cpp.


export namespace ses {

// u = caller's uniform draw in [0, 1). `damp` must be built at
// dtau = kappa dt / (2 omega) on the SAME grid/potential.
inline bool photon_loss_step(Field1D& psi, double omega,
                             const std::vector<double>& v, double kappa,
                             double dt, double u,
                             const ImaginaryTimePropagator1D& damp) {
    const double n_bar =
        std::max(0.0, mean_energy(psi, v) / omega - 0.5);
    if (u < kappa * n_bar * dt) {
        ladder_lower(psi, omega);  // a psi / ||.||
        return true;
    }
    damp.relax(psi, 1);  // no-jump e^{-kappa n dt/2}/norm, EXACT (harmonic V)
    return false;
}

}  // namespace ses
