module;
#include <algorithm>
#include <string>
export module ses.scenario.tunneling1d_director;
export import ses.scenario.line1d_director;
import ses.wavepacket;


// Slab is HALF the 3D tunneling scene's width so kappa*w = 1.25 keeps the
// transmitted packet visible, not a ~1% wisp. T is vs initial unit norm
// (probability_in_range is absolute).


export namespace ses_shell {

constexpr double kTun1dBox = 80.0;   // Bohr
// oversampled (1D is cheap): phasor curve resolves ripples + evanescent decay
constexpr int kTun1dPoints = 65536;
constexpr double kTun1dV0 = 0.25;    // Ha
constexpr double kTun1dXLo = 0.0;
constexpr double kTun1dXHi = 2.5;
constexpr double kTun1dK0 = 0.5;
constexpr double kTun1dLaunchX = -30.0;
constexpr double kTun1dSigma = 5.0;
constexpr double kTun1dDt = 0.04;
constexpr double kTun1dAbsorb = 10.0;  // Bohr
constexpr double kTun1dRScale = 60.0;  // radius ~ |psi|^2
constexpr double kTun1dEScale = 40.0;  // V display scale
constexpr double kTun1dYClamp = 12.0;

class Tunneling1DDirector final : public Line1DDirector, public TunnelApi {
public:
    Tunneling1DDirector()
        : Line1DDirector(ses::Grid1D{-kTun1dBox, kTun1dBox, kTun1dPoints},
                         ses::barrier_potential(
                             ses::Grid1D{-kTun1dBox, kTun1dBox, kTun1dPoints},
                             kTun1dV0, kTun1dXLo, kTun1dXHi),
                         kTun1dDt, kTun1dRScale, kTun1dEScale, kTun1dYClamp) {
        set_mask(ses::absorbing_mask(grid1d_, kTun1dAbsorb));
        set_state(ses::gaussian_wavepacket(grid1d_, kTun1dLaunchX, kTun1dSigma,
                                           kTun1dK0));
    }

    TunnelApi* tunnel() override { return this; }
    double transmitted_max() const override { return t_max_; }

    // slight angle keeps the phasor twist readable
    double default_camera_azimuth() const override { return 0.22; }
    double default_camera_elevation() const override { return 0.24; }
    double default_camera_distance() const override { return 185.0; }

protected:
    const char* scene_name() const override { return "1D quantum tunneling"; }
    // paces the approach to ~12 s at scale 1
    int steps_per_tick() const override { return 2; }

    std::string title_suffix() override {
        return strf("  V0 = %.2f Ha, E = %.3f Ha (forbidden)  P(x<%.0f) %.3f | "
                    "P(x>%.0f) %.3f (max T %.3f)",
                    kTun1dV0, 0.5 * kTun1dK0 * kTun1dK0, kTun1dXLo, p_left_,
                    kTun1dXHi, p_right_, t_max_);
    }

    // probe is microseconds: safe to track T every batch
    void after_batch() override {
        p_left_ = ses::probability_in_range(psi_, grid1d_.xmin, kTun1dXLo);
        p_right_ = ses::probability_in_range(psi_, kTun1dXHi, grid1d_.xmax);
        t_max_ = std::max(t_max_, p_right_);
    }

    void after_reset() override {
        p_left_ = 0.0;
        p_right_ = 0.0;
        t_max_ = 0.0;
    }

private:
    double p_left_ = 0.0;
    double p_right_ = 0.0;
    double t_max_ = 0.0;
};

}  // namespace ses_shell
