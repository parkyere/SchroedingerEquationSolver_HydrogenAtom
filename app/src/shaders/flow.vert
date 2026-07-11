#version 450

// Probability-flow particle sprites: vertex-pulled from the particle SSBO
// (one point per gl_VertexIndex), colored by the LOCAL phase of psi at the
// particle (same LUT as the cloud), sized by distance, faded in/out by age.
layout(std430, binding = 0) readonly buffer Particles { vec4 part[]; };
layout(std140, binding = 1) uniform Ubo {
    mat4 mvp;
    vec4 eye;             // .xyz
    vec4 box_min;         // .xyz
    vec4 box_max;         // .xyz
    vec4 proton_center;
    vec4 proton_color;
    float inv_peak;
    float absorbance;
    float proton_radius;
    float jitter_frame;
};
layout(binding = 2) uniform sampler3D psi_tex;
layout(binding = 3) uniform sampler1D phase_tex;

layout(location = 0) out vec4 v_color;

void main() {
    vec4 st = part[gl_VertexIndex];
    vec3 p = st.xyz;
    float age = st.w;

    // Half-texel alignment (grid point i = texel center (i+0.5)/n).
    vec3 uvw = (p - box_min.xyz) / (box_max.xyz - box_min.xyz) +
               0.5 / vec3(textureSize(psi_tex, 0));
    vec2 s = texture(psi_tex, uvw).xy;
    float rho = dot(s, s) * inv_peak;
    float phase01 = (atan(s.y, s.x) + 3.14159265358979) / 6.28318530717959;
    vec3 col = texture(phase_tex, phase01).rgb;

    // Fade in over the first ~30 frames; invisible in empty space (respawn
    // candidates parked outside the cloud must not glow).
    float fade = clamp(age / 30.0, 0.0, 1.0) * clamp(rho * 40.0, 0.0, 1.0);
    v_color = vec4(col * fade * 0.55, 1.0);

    gl_Position = mvp * vec4(p, 1.0);
    float dist = length(p - eye.xyz);
    gl_PointSize = clamp(240.0 / max(dist, 1.0), 1.5, 5.0);
}
