// scene image: SHADER_READ_ONLY_OPTIMAL at hand-off.
struct FSIn {
    float4 pos : SV_Position;
    [[vk::location(0)]] float2 v_uv : TEXCOORD0;
};
[[vk::binding(0, 0)]] Sampler2D<float4> scene;

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    return scene.Sample(input.v_uv);
}
