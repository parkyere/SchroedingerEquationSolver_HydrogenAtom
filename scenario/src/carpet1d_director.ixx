module;
#include <cstddef>
#include <algorithm>
#include <cmath>
#include <complex>
#include <memory>
#include <string>
#include <utility>
#include <vector>
export module ses.scenario.carpet1d_director;
export import ses.scenario.lattice2d_director;
import ses.heightfield;
import ses.propagator;
import ses.wavepacket;
import ses.parallel;


// Quantum carpet: a free packet on a PERIODIC RING (the FFT box itself --
// no walls, no wall artifacts) weaves the temporal Talbot carpet. With
// k_n = 2 pi n / L and E_n = k_n^2 / 2, every phase realigns at
// T_rev = L^2 / pi (EXACT on the spectral grid): the packet fully
// revives; at T/2 it reappears displaced by L/2 (the half-Talbot clone);
// rational fractions of T weave the fractional-revival lattice. The
// display stacks |psi(x)|^2 rows into an (x, t) height carpet, one full
// revival per carpet height -- after the first pass the wrap lands on
// the SAME pattern and the carpet stands still.
// CONTRACT: tests/carpet1d_test.cpp + --selftest-carpet.


export namespace ses_shell {

// Full revival time of the ring of circumference L.
inline double carpet_revival_time(double /*l*/) {
    return 0.0;  // RED stub
}

}  // namespace ses_shell
