// instance = streak, vertex = trail slot (0=oldest tail .. len-1=newest head); fade matches ses.flow::trail_fade.
// gl_VertexIndex/gl_InstanceIndex avoid the shaderDrawParameters capability SV_VertexID/SV_InstanceID need.
import glsl;

[[vk::binding(0, 0)]] StructuredBuffer<float4, Std430DataLayout> trail;

struct Ubo {
    float4x4 mvp;
    float4 eye;             // unused; matches shared UBO layout
    float4 box_min;
    float4 box_max;
    float4 proton_center;
    float4 proton_color;
    float inv_peak;
    float absorbance;
    float proton_radius;
    float jitter_frame;
};
[[vk::binding(1, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

struct Push {
    int trail_len;
};
[[vk::push_constant]] ConstantBuffer<Push> push;

struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 v_color : TEXCOORD0;
};

[shader("vertex")]
VSOut main() {
    VSOut o;
    uint vid = uint(gl_VertexIndex);
    uint iid = uint(gl_InstanceIndex);
    int slot = int(iid) * push.trail_len + int(vid);
    float3 p = trail[slot].xyz;

    float fade = (push.trail_len > 1)
                     ? float(vid) / float(push.trail_len - 1)
                     : 1.0;
    float a = fade * 0.85;
    o.v_color = float4(a, a, a, a);     // premultiplied white: rgb = a

    o.pos = mul(ubo.mvp, float4(p, 1.0));
    return o;
}
