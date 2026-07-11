#version 450

// Soft circular sprite, ADDITIVELY blended into the HDR scene: the bloom
// pass then makes dense streams of particles sparkle.
layout(location = 0) in vec4 v_color;
layout(location = 0) out vec4 frag;

void main() {
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d) * 4.0;  // 0 center .. 1 edge
    float w = max(1.0 - r2, 0.0);
    frag = vec4(v_color.rgb * (w * w), 0.0);  // additive: alpha unused
}
