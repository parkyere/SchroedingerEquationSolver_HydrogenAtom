#version 450

// Cross-section quad: a plane through the nucleus, normal along slice.y
// (0/1/2), at offset slice.z, spanning the box in the other two axes. Two
// triangles from gl_VertexIndex -- no vertex buffer. Shares the volume UBO.
// Mirrors ses::slice_quad_corner (core/cross_section.hpp).
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

layout(location = 0) out vec3 v_world;

void set_comp(inout vec3 v, int i, float x) {
    if (i == 0) v.x = x;
    else if (i == 1) v.y = x;
    else v.z = x;
}
float comp(vec3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

// CCW quad corners as (s, t) in {0,1}^2.
const vec2 kST[6] = vec2[6](vec2(0, 0), vec2(1, 0), vec2(1, 1),
                            vec2(0, 0), vec2(1, 1), vec2(0, 1));

void main() {
    int a = int(slice.y + 0.5);
    int u = (a + 1) % 3;
    int w = (a + 2) % 3;
    vec2 st = kST[gl_VertexIndex];
    vec3 p = vec3(0.0);
    set_comp(p, a, slice.z);
    set_comp(p, u, mix(comp(box_min.xyz, u), comp(box_max.xyz, u), st.s));
    set_comp(p, w, mix(comp(box_min.xyz, w), comp(box_max.xyz, w), st.t));
    v_world = p;
    gl_Position = mvp * vec4(p, 1.0);
}
