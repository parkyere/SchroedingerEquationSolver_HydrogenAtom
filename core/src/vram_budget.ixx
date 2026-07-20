module;
#include <cstdint>
export module ses.vram_budget;


// An overflowing fp32 atlas makes WDDM page across PCIe and thrash every frame.
// Free-VRAM probe lives in ses.vk.vram_probe (solver/).


export namespace ses {

enum class GpuPrecision { Fp32, Fp16 };

// Sentinel: VK_EXT_memory_budget unavailable.
inline constexpr std::int64_t kVramUnknown = -1;

// headroom_bytes: VRAM held back for textures/framebuffers/driver.
// Unmeasurable budget keeps fp32 (never silently degrade fidelity).
inline constexpr GpuPrecision choose_state_precision(std::int64_t free_vram_bytes,
                                                     int num_states,
                                                     std::int64_t bytes_per_state_fp32,
                                                     std::int64_t headroom_bytes,
                                                     bool* out_fits = nullptr) noexcept {
    const auto set_fits = [&](bool v) {
        if (out_fits != nullptr) {
            *out_fits = v;
        }
    };
    if (free_vram_bytes == kVramUnknown || free_vram_bytes < 0) {
        set_fits(true);
        return GpuPrecision::Fp32;
    }
    const std::int64_t budget = free_vram_bytes - headroom_bytes;  // may be < 0
    const std::int64_t need32 =
        static_cast<std::int64_t>(num_states) * bytes_per_state_fp32;
    const std::int64_t need16 =
        static_cast<std::int64_t>(num_states) * (bytes_per_state_fp32 / 2);
    if (need32 <= budget) {
        set_fits(true);
        return GpuPrecision::Fp32;
    }
    if (need16 <= budget) {
        set_fits(true);
        return GpuPrecision::Fp16;
    }
    set_fits(false);  // best effort: even fp16 overflows.
    return GpuPrecision::Fp16;
}

}  // namespace ses
