module;
#include <complex>
#include <cstddef>
#include <utility>
#include <algorithm>
#include <cmath>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>
#include <vector>
#include <volk.h>
export module ses.scenario.base_director;
export import ses.grid;
export import ses.vk.engine_blobs;
export import ses.scenario;
export import ses.simulation;
export import ses.sampling;
export import ses.imaginary_time;
export import ses.observables;
export import ses.marching_cubes;
export import ses.field;
export import ses.potential;
export import ses.colormap;
import ses.emission;


// Shared machinery for potential-swap scenarios (trap, tunneling).
// HydrogenDirector reuses the members but overrides the whole frame flow.
// volk.h textually first: VK_* macros never cross module boundaries.


export namespace ses_shell {

enum class BaseViewMode { Cloud, Surface };
// RelaxingExcited: only HydrogenDirector (deflated relax) sets it.
enum class BaseStepping { RealTime, Relaxing, RelaxingExcited };

constexpr int kBaseStepsPerTick = 1;
constexpr int kBaseRelaxStepsPerTick = 1;
constexpr double kBaseRelaxDtau = 0.05;
constexpr double kBaseIsoFraction = 0.25;
constexpr double kBaseMeasureSigma = 1.25;  // Bohr
constexpr double kBaseHaToEv = 27.211386;
constexpr double kBaseAuToFs = 2.4188843e-2;

class BaseDirector : public ScenarioDirector {
public:
    explicit BaseDirector(ses::WavepacketSimulation sim) : sim_(std::move(sim)) {
        remesh();
        stage_volume();
    }

    const ses::Grid3D& grid() const override { return sim_.grid(); }

    // Engine/gradient failure demotes to CPU; absorber failure only drops the mask.
    void init_compute(ses_vk::DeviceContext& ctx, bool device_ok,
                      std::int64_t /*free_vram_bytes*/) override {
        compute_attempted_ = true;
        gpu_ok_ = device_ok &&
                  engine_.initialize(ctx, sim_.grid(),
                                     ses_vk::engine_blobs(sim_.grid().x.n),
                                     sim_.potential(), sim_.dt(),
                                     sim_.psi().data());
        if (!gpu_ok_) {
            return;
        }
        // Relax tables are transient (uploaded on relax entry); only gradient is fatal here.
        if (!engine_.set_potential_gradient(sim_.potential())) {
            std::fprintf(stderr, "engine: gradient setup failed -- "
                                 "falling back to CPU stepping\n");
            gpu_ok_ = false;
            return;
        }
        if (absorber_width() > 0.0) {
            absorber_on_ = engine_.set_absorber(
                ses::absorbing_mask(sim_.grid(), absorber_width()));
        }
        on_gpu_ready();
    }

    void release_gpu() override {
        engine_.destroy();
        gpu_ok_ = false;
    }

    bool use_gpu_path() const { return gpu_ok_; }

    void run_frame() override {
        if (!use_gpu_path()) {
            return;
        }
        // Reclaim last frame's async batch FIRST (display flip + cb reuse).
        engine_.wait_async();
        if (cpu_is_truth_) {
            double pk = 0.0;
            for (const std::complex<double>& z : sim_.psi().data()) {
                pk = std::max(pk, std::norm(z));
            }
            if (pk > 0.0) {
                peak_ = pk;
            }
            engine_.upload_state(sim_.psi().data());
            cpu_is_truth_ = false;
            volume_dirty_ = false;
            write_display_texture();
        }
        service_requests();
        if (pending_gpu_steps_ > 0) {
            if (stepping_ == BaseStepping::RealTime) {
                run_real_time_batch();
                if (mode_ == BaseViewMode::Cloud) {
                    volume_written_ = true;
                } else {
                    mc_dirty_ = true;
                }
            } else {
                run_relax_batch();
                write_display_texture();
            }
            pending_gpu_steps_ = 0;
            volume_dirty_ = false;
            if (gpu_title_due_) {
                gpu_title_due_ = false;
                title_dirty_ = true;
            }
        }
        // kBaseIsoFraction * peak mirrors marching_cubes_at_fraction (CPU/GPU iso parity).
        if (mode_ == BaseViewMode::Surface && engine_.mc_ready() &&
            mc_dirty_) {
            engine_.mc_extract(kBaseIsoFraction * peak_);
            mc_dirty_ = false;
            volume_written_ = true;  // display changed: accumulation resets
        }
    }

    void tick() override {
        if (use_gpu_path()) {
            // Clamp drops catch-up ticks: at most one tick's steps per rendered frame.
            const int per_tick = steps_per_tick() * time_scale_;
            pending_gpu_steps_ =
                std::min(pending_gpu_steps_ + per_tick, per_tick);
            if (++ticks_ % 10 == 0) {
                gpu_title_due_ = true;
            }
            return;
        }
        ensure_cpu_current();
        // CPU fallback: not time-scaled (sync steps would stall the UI).
        if (stepping_ == BaseStepping::RealTime) {
            sim_.advance(kBaseStepsPerTick);
        } else {
            sim_.relax(kBaseRelaxStepsPerTick, kBaseRelaxDtau);
        }
        stage_active_view();
        if (++ticks_ % 10 == 0) {
            norm_display_ = ses::norm_sq(sim_.psi());
            title_dirty_ = true;
        }
    }

    // steps per tick, not dt.
    void set_time_scale(int scale) override {
        time_scale_ = std::clamp(scale, 1, 16);
    }
    int time_scale() const override { return time_scale_; }

    double sim_time() const override { return sim_.time() + gpu_time_; }
    double sim_dt() const override { return sim_.dt(); }

    // ---- generic controls ----

    void do_set_real_time() override {
        stepping_ = BaseStepping::RealTime;
        engine_.release_relax_tables();
    }

    void reset_simulation() override {
        sim_ = remake_simulation();
        stepping_ = BaseStepping::RealTime;
        cpu_is_truth_ = true;
        gpu_time_ = 0.0;
        pending_gpu_steps_ = 0;
        after_reset();
        stage_active_view();
    }

    void measure_now() override {
        ensure_cpu_current();
        std::uniform_real_distribution<double> uniform(0.0, 1.0);
        sim_.measure(uniform(rng_), kBaseMeasureSigma);
        stepping_ = BaseStepping::RealTime;
        stage_active_view();
    }

    void toggle_view_mode() override {
        mode_ = (mode_ == BaseViewMode::Cloud) ? BaseViewMode::Surface
                                               : BaseViewMode::Cloud;
        if (mode_ == BaseViewMode::Surface) {
            if (gpu_ok_ && engine_.mc_prepare(kMcMaxTris)) {
                mc_dirty_ = true;
            } else {
                ensure_cpu_current();
            }
        } else {
            engine_.release_mc();
            if (gpu_ok_ && !cpu_is_truth_) {
                write_display_texture();
            }
        }
        stage_active_view();
    }

    void set_relaxing() {
        if (!relax_allowed()) {
            return;
        }
        if (use_gpu_path() && !ensure_relax_tables()) {
            return;
        }
        stepping_ = BaseStepping::Relaxing;
        relax_plateau_ = 0;
        relax_prev_energy_ = 0.0;
        if (!use_gpu_path()) {
            ensure_cpu_current();
        }
    }

    bool handle_key(char key) override {
        if (key == '2') {
            set_relaxing();
            return true;
        }
        return false;
    }

    bool solving() const override { return false; }
    bool scene_ready() const override { return compute_attempted_; }

    // ---- display accessors ----

    bool cloud() const override { return mode_ == BaseViewMode::Cloud; }
    VkBuffer surface_vbuf() const override {
        return (mode_ == BaseViewMode::Surface && engine_.mc_ready())
                   ? engine_.mc_vertex_buffer()
                   : VK_NULL_HANDLE;
    }
    VkBuffer surface_indirect() const override {
        return (mode_ == BaseViewMode::Surface && engine_.mc_ready())
                   ? engine_.mc_indirect_buffer()
                   : VK_NULL_HANDLE;
    }
    double peak() const override { return peak_; }
    bool compute_attempted() const override { return compute_attempted_; }
    bool gpu_ok() const override { return gpu_ok_; }
    VkImageView psi_volume_view() override {
        return gpu_ok_ ? engine_.volume_view() : VK_NULL_HANDLE;
    }
    VkImageView flow_velocity_view() override {
        return gpu_ok_ ? engine_.flow_velocity_view() : VK_NULL_HANDLE;
    }
    float next_flash_intensity() override { return 0.0f; }
    bool take_volume_written() override {
        const bool w = volume_written_;
        volume_written_ = false;
        return w;
    }
    bool take_volume_dirty() override {
        const bool d = volume_dirty_;
        volume_dirty_ = false;
        return d;
    }
    bool take_mesh_dirty() override {
        const bool d = mesh_dirty_;
        mesh_dirty_ = false;
        return d;
    }
    void mark_display_dirty() override {
        mesh_dirty_ = true;
        volume_dirty_ = true;
    }
    bool take_title_dirty() override {
        const bool t = title_dirty_;
        title_dirty_ = false;
        return t;
    }
    const std::vector<float>& psi_staging() const override { return psi_staging_; }
    const ses::Mesh& mesh() const override { return mesh_; }
    const std::vector<ses::Rgb>& colors() const override { return colors_; }

    std::string title_text() override {
        const double t_au = sim_.time() + gpu_time_;
        std::string s = scene_name() + std::string("   t = ") +
                        strf("%.2f au (%.3f fs)", t_au, t_au * kBaseAuToFs) + "   ";
        if (stepping_ != BaseStepping::RealTime) {
            s += cpu_is_truth_
                     ? strf("E = %.3f eV   ",
                            ses::mean_energy(sim_.psi(), sim_.potential()) *
                                kBaseHaToEv)
                     : strf("E ~ %.3f eV   ", relax_energy_display_ * kBaseHaToEv);
        }
        if (stepping_ == BaseStepping::RealTime && use_gpu_path()) {
            s += strf("emit P = %.2e au   ", radiated_power_);
        }
        s += strf("norm = %.6f   [%s, %s, %s]  1=real 2=relax R=reset tab=view "
                  "[ ]=density M=pos",
                  norm_display_,
                  mode_ == BaseViewMode::Cloud ? "cloud" : "surface",
                  stepping_ == BaseStepping::RealTime ? "real-time"
                                                      : "relaxing->ground",
                  use_gpu_path() ? "gpu 256^3" : "cpu 256^3");
        s += title_suffix();
        return s;
    }

protected:
    // ---- scenario hooks ----
    virtual ses::WavepacketSimulation remake_simulation() const = 0;
    virtual const char* scene_name() const = 0;
    virtual std::string title_suffix() { return std::string(); }
    virtual double absorber_width() const { return 0.0; }
    virtual void on_gpu_ready() {}
    virtual void after_reset() {}
    virtual void service_requests() {}   // run_frame, before stepping
    virtual void after_step_batch() {}   // run_frame, after a real-time batch
    virtual int steps_per_tick() const { return kBaseStepsPerTick; }
    virtual bool relax_allowed() const { return true; }

    static std::string strf(const char* fmt, ...) {
        char buf[192];
        va_list args;
        va_start(args, fmt);
        std::vsnprintf(buf, sizeof buf, fmt, args);
        va_end(args);
        return std::string{buf};
    }

    void run_real_time_batch() {
        if (gpu_title_due_) {
            const ses_vk::Engine::NormPeak np = engine_.norm_and_peak();
            norm_display_ = np.sum;
            if (np.peak > 0.0) {
                peak_ = np.peak;
            }
            // fp32 drift renormalization (split-operator is unitary exactly).
            if (np.sum > 0.0 && std::abs(np.sum - 1.0) > 1e-4 &&
                absorber_width() == 0.0) {
                engine_.scale(static_cast<float>(1.0 / std::sqrt(np.sum)));
            }
            radiated_power_ = ses::larmor_power(engine_.mean_force());
        }
        // ASYNC: overlaps this frame's render; next run_frame waits and flips.
        // after_step_batch hooks reading psi serialize on the same queue.
        engine_.step_async(pending_gpu_steps_, absorber_on_, true);
        gpu_time_ += pending_gpu_steps_ * sim_.dt();
        after_step_batch();
    }

    // Virtual: molecule scenes substitute a deflated-excited batch, reusing the rest.
    virtual void run_relax_batch() {
        const ses_vk::Engine::RelaxStats stats =
            engine_.relax_step(pending_gpu_steps_);
        relax_energy_display_ = stats.energy;
        if (stats.peak > 0.0) {
            peak_ = stats.peak;
        }
        norm_display_ = 1.0;  // pinned by per-step renormalization
        // Auto-complete on the ITP energy plateau.
        if (gpu_title_due_) {
            if (std::abs(stats.energy - relax_prev_energy_) < 5e-5) {
                ++relax_plateau_;
            } else {
                relax_plateau_ = 0;
            }
            relax_prev_energy_ = stats.energy;
            if (relax_plateau_ >= 12) {
                relax_plateau_ = 0;
                stepping_ = BaseStepping::RealTime;
                engine_.release_relax_tables();
            }
        }
    }

    bool ensure_relax_tables() {
        if (engine_.relax_tables_ready()) {
            return true;
        }
        const double dtau = relax_dtau();
        const ses::ImaginaryTimePropagator3D relaxer{sim_.grid(), sim_.potential(),
                                                     dtau};
        return engine_.set_relax_tables(relaxer.half_potential_weight(),
                                        relaxer.kinetic_weight(), dtau,
                                        sim_.grid().cell_volume());
    }

    // Deep wells override: V*dtau must stay moderate or the ITP fixed point is a
    // Trotter artifact, not the grid-H eigenstate.
    virtual double relax_dtau() const { return kBaseRelaxDtau; }

    void ensure_cpu_current() {
        pending_gpu_steps_ = 0;  // uncredited steps must not fire later
        if (cpu_is_truth_ || !gpu_ok_) {
            return;
        }
        if (!engine_.readback(readback_buf_)) {
            return;  // readback failed: keep the CPU state
        }
        ses::Field3D f{sim_.grid()};
        for (std::size_t i = 0; i < f.data().size(); ++i) {
            f.data()[i] =
                std::complex<double>{readback_buf_[2 * i], readback_buf_[2 * i + 1]};
        }
        sim_.set_psi(f);
        cpu_is_truth_ = true;
    }

    void stage_active_view() {
        if (mode_ == BaseViewMode::Cloud) {
            if (use_gpu_path()) {
                return;  // run_frame uploads and bridges
            }
            stage_volume();
            volume_dirty_ = true;
        } else {
            if (gpu_ok_) {
                mc_dirty_ = true;
                return;
            }
            remesh();
            mesh_dirty_ = true;
        }
    }

    void remesh() {
        mesh_ = ses::marching_cubes_at_fraction(sim_.density(), sim_.grid(),
                                                kBaseIsoFraction);
        colors_ = ses::phase_colors(mesh_, sim_.psi());
    }

    void stage_volume() {
        const auto& field = sim_.psi().data();
        psi_staging_.resize(field.size() * 2);
        double peak = 0.0;
        for (std::size_t i = 0; i < field.size(); ++i) {
            psi_staging_[2 * i] = static_cast<float>(field[i].real());
            psi_staging_[2 * i + 1] = static_cast<float>(field[i].imag());
            peak = std::max(peak, std::norm(field[i]));
        }
        peak_ = peak;
    }

    void write_display_texture() {
        if (mode_ == BaseViewMode::Surface && engine_.mc_ready()) {
            mc_dirty_ = true;
            return;
        }
        engine_.write_psi_to_volume();
        volume_written_ = true;  // resets the temporal accumulation
    }

    ses::WavepacketSimulation sim_;
    ses_vk::Engine engine_;
    BaseViewMode mode_ = BaseViewMode::Cloud;
    BaseStepping stepping_ = BaseStepping::RealTime;
    bool compute_attempted_ = false;
    bool gpu_ok_ = false;
    bool cpu_is_truth_ = true;
    int pending_gpu_steps_ = 0;
    int time_scale_ = 1;
    bool gpu_title_due_ = false;
    bool title_dirty_ = false;
    double gpu_time_ = 0.0;
    double norm_display_ = 1.0;
    double relax_energy_display_ = 0.0;
    double radiated_power_ = 0.0;  // au
    double relax_prev_energy_ = 0.0;
    int relax_plateau_ = 0;
    std::vector<float> readback_buf_;

    bool mc_dirty_ = false;  // Surface: re-extract mesh
    ses::Mesh mesh_;
    std::vector<ses::Rgb> colors_;
    std::vector<float> psi_staging_;
    double peak_ = 0.0;
    bool mesh_dirty_ = false;
    bool volume_dirty_ = false;
    bool volume_written_ = false;
    long long ticks_ = 0;

    bool absorber_on_ = false;
    std::mt19937 rng_{std::random_device{}()};
};

}  // namespace ses_shell
