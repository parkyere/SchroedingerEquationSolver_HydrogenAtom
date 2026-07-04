#pragma once

// WavepacketSimulation: the tested orchestration layer the GL shell drives.
// Owns grid + potential + propagator + psi; advance() steps real time.
// Exactly equivalent to gaussian_wavepacket followed by SplitOperator3D
// steps (pinned by tests) -- no hidden physics lives here.

#include <core/field.hpp>
#include <core/grid.hpp>
#include <core/imaginary_time.hpp>
#include <core/propagator.hpp>
#include <core/vec.hpp>
#include <core/wavepacket.hpp>

#include <optional>
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
          potential_(std::move(cfg.potential)),
          propagator_(cfg.grid, potential_, cfg.dt),
          psi_(gaussian_wavepacket(cfg.grid, cfg.r0, cfg.sigma, cfg.k0)) {}

    void advance(int nsteps) {
        propagator_.step(psi_, nsteps);
        steps_ += nsteps;
    }

    // Imaginary-time relaxation toward the ground state. Does NOT advance
    // real time (tau is not t). The propagator's decay tables are cached and
    // rebuilt only when dtau changes.
    void relax(int nsteps, double dtau) {
        if (!relaxer_ || relaxer_dtau_ != dtau) {
            relaxer_.emplace(grid_, potential_, dtau);
            relaxer_dtau_ = dtau;
        }
        relaxer_->relax(psi_, nsteps);
    }

    double time() const { return steps_ * dt_; }
    const Grid3D& grid() const { return grid_; }
    const Field3D& psi() const { return psi_; }
    const std::vector<double>& potential() const { return potential_; }
    std::vector<double> density() const { return probability_density(psi_); }

private:
    Grid3D grid_;
    double dt_;
    std::vector<double> potential_;
    SplitOperator3D propagator_;
    Field3D psi_;
    long long steps_ = 0;
    std::optional<ImaginaryTimePropagator3D> relaxer_;
    double relaxer_dtau_ = 0.0;
};

}  // namespace ses
