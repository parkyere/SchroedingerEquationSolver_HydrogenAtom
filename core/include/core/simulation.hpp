#pragma once

// WavepacketSimulation: the tested orchestration layer the GL shell drives.
// Owns grid + potential + propagator + psi; advance() steps real time.
// Exactly equivalent to gaussian_wavepacket followed by SplitOperator3D
// steps (pinned by tests) -- no hidden physics lives here.

#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/propagator.hpp>
#include <core/vec.hpp>
#include <core/wavepacket.hpp>

#include <utility>
#include <vector>

namespace ses {

class WavepacketSimulation {
public:
    struct Config {
        Grid3D grid;
        std::vector<double> potential;
        Vec3d r0;
        Vec3d sigma;
        Vec3d k0;
        double dt;
    };

    explicit WavepacketSimulation(Config cfg)
        : grid_(cfg.grid),
          dt_(cfg.dt),
          propagator_(cfg.grid, cfg.potential, cfg.dt),
          psi_(gaussian_wavepacket(cfg.grid, cfg.r0, cfg.sigma, cfg.k0)) {}

    void advance(int nsteps) {
        propagator_.step(psi_, nsteps);
        steps_ += nsteps;
    }

    double time() const { return steps_ * dt_; }
    const Grid3D& grid() const { return grid_; }
    const Field3D& psi() const { return psi_; }
    std::vector<double> density() const { return probability_density(psi_); }

private:
    Grid3D grid_;
    double dt_;
    SplitOperator3D propagator_;
    Field3D psi_;
    long long steps_ = 0;
};

}  // namespace ses
