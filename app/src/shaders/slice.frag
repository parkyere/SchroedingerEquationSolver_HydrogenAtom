#version 450

// Cross-section sheet: sample psi on the plane and paint it. Map modes
// (slice.w): 0 = |psi|^2 density (magnitude ramp), 1 = Re(psi) diverging
// (blue<-0->orange, the orbital's signed lobes), 2 = phase arg(psi) (cyclic
// HSV LUT, brightness by |psi|). The card carries a base opacity so nodal
// planes read as dark bands ON a visible sheet. The colour/alpha and the
// clip-side discard MIRROR ses::slice_shade / ses::clip_ray_interval
// (core/cross_section.hpp) -- the tested single source of truth.
layout(location = 0) in vec3 v_world;

layout(std140, binding = 0) uniform Ubo {
    mat4 mvp;
    vec4 eye;
    vec4 box_min;
    vec4 box_max;
    vec4 proton_center;
    vec4 proton_color;
    float inv_peak;
    float absorbance;
    float proton_radius;
    float jitter_frame;
    vec4 clip;
    vec4 slice;  // .x enable, .y axis, .z offset, .w map mode
};

layout(binding = 1) uniform sampler3D psi_tex;
layout(binding = 2) uniform sampler1D phase_tex;

layout(location = 0) out vec4 frag;

float comp(vec3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

void main() {
    // Respect the clip plane: the sheet is hidden on the cut-away half so it
    // does not float in the emptied space when the two planes differ.
    if (clip.x > 0.5 &&
        clip.z * (comp(v_world, int(clip.y + 0.5)) - clip.w) > 0.0) {
        discard;
    }
    vec3 ext = box_max.xyz - box_min.xyz;
    vec3 half_texel = 0.5 / vec3(textureSize(psi_tex, 0));
    vec3 uvw = (v_world - box_min.xyz) / ext + half_texel;
    vec2 s = texture(psi_tex, uvw).rg;
    float dens = dot(s, s) * inv_peak;         // normalized |psi|^2 in [0,1]
    float amp = sqrt(max(inv_peak, 0.0));      // shared sign/Re scale

    int mode = int(slice.w + 0.5);
    vec3 col;
    float bright;
    if (mode == 1) {  // Re(psi), diverging
        float r = clamp(s.r * amp, -1.0, 1.0);
        vec3 pos = vec3(1.0, 0.55, 0.15);   // + lobe (warm)
        vec3 neg = vec3(0.15, 0.45, 1.0);   // - lobe (cool)
        col = (r >= 0.0) ? mix(vec3(0.03), pos, r) : mix(vec3(0.03), neg, -r);
        bright = abs(r);
    } else if (mode == 2) {  // phase
        float ph = (atan(s.g, s.r) + 3.14159265358979) / 6.28318530717959;
        col = texture(phase_tex, ph).rgb;
        bright = sqrt(clamp(dens, 0.0, 1.0));
        col *= 0.25 + 0.75 * bright;
    } else {  // density (mode 0)
        float d = clamp(dens, 0.0, 1.0);
        col = vec3(pow(d, 1.6), pow(d, 0.9), 0.25 + 0.75 * d);  // navy->white
        bright = d;
    }
    // Base opacity keeps the sheet (and its nodes) visible; denser regions
    // are more solid so the cross-section still reads as the cloud.
    float alpha = clamp(0.45 + 0.5 * bright, 0.0, 0.95);
    frag = vec4(col * alpha, alpha);  // premultiplied
}
