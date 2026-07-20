// Output PREMULTIPLIED: pipeline blends One/OneMinusSrcAlpha, depth-write off, drawn after overlays.
struct FSIn {
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

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    float3 n = normalize(input.v_normal);
    float3 vdir = normalize(ubo.eye.xyz - input.v_pos);
    float facing = abs(dot(n, vdir));
    float fres = pow(1.0 - facing, 3.0);
    float spec = pow(facing, 48.0);
    float alpha = 0.06 + 0.50 * fres;
    float3 c = input.v_color * (0.35 + 0.65 * fres) +
               float3(0.9, 0.9, 0.95) * spec * 0.6;
    return float4(c * alpha + float3(spec, spec, spec) * 0.25, alpha);
}
