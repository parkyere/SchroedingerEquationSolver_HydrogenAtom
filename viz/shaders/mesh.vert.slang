// Ubo must byte-match the fragment stage's UBO; eye is vec4-padded (.xyz payload).
// MVP bakes the GL->Vulkan clip correction (y-flip, depth [-1,1]->[0,1]) host-side;
// plain mul(mvp,p) -- column-major matches GLSL mvp*p, no clip fix here.
struct VSIn {
    [[vk::location(0)]] float3 pos : POSITION;
    [[vk::location(1)]] float3 normal : NORMAL;
    [[vk::location(2)]] float3 color : COLOR;
};
struct VSOut {
    float4 pos : SV_Position;
    [[vk::location(0)]] float3 v_normal : TEXCOORD0;
    [[vk::location(1)]] float3 v_pos : TEXCOORD1;
    [[vk::location(2)]] float3 v_color : TEXCOORD2;
};

struct Ubo {
    float4x4 mvp;
    float4 eye;
};
[[vk::binding(0, 0)]] ConstantBuffer<Ubo, Std140DataLayout> ubo;

[shader("vertex")]
VSOut main(VSIn input) {
    VSOut o;
    o.pos = mul(ubo.mvp, float4(input.pos, 1.0));
    o.v_normal = input.normal;
    o.v_pos = input.pos;
    o.v_color = input.color;
    return o;
}
