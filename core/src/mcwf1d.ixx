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


// 1D grid MCWF unraveling of CAVITY PHOTON LOSS (rate kappa, jump op a):
// p_jump = kappa <n> dt with <n> = <H>/omega - 1/2; a JUMP applies the
// lowering operator (renormalized) -- on a cat |a> + |-a> this flips the
// PARITY to |a> - |-a| (a is diagonal on each branch with opposite
// eigenvalue signs): the interference fringes invert per lost photon.
// NO-JUMP applies the conditional damping e^{-kappa n_hat dt/2}/norm,
// realized EXACTLY (harmonic V) as ONE renormalized imaginary-time step
// at dtau = kappa dt / (2 omega): e^{-H tau} = const x e^{-n_hat omega tau}.
// CONTRACT: tests/mcwf1d_test.cpp.


export namespace ses {

// One unraveling step; u in [0, 1) is the caller's uniform draw; `damp`
// must be built at dtau = kappa dt / (2 omega) on the SAME grid/potential.
// Returns true when the jump fired.
inline bool photon_loss_step(Field1D& /*psi*/, double /*omega*/,
                             const std::vector<double>& /*v*/,
                             double /*kappa*/, double /*dt*/, double /*u*/,
                             const ImaginaryTimePropagator1D& /*damp*/) {
    return false;  // RED stub
}

}  // namespace ses
