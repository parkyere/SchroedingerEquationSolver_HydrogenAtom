// Kawase dual-filter downsample: half-texel corner offsets make each bilinear tap average 4 texels; center weighted x4 + 4 unit corner taps, /8 normalizes.
[[vk::binding(0, 0)]] Sampler2D<float4> src;
[[vk::binding(1, 0)]] [[vk::image_format("rgba16f")]] RWTexture2D<float4> dst;

struct Params {
    int bright_pass;
    float threshold;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(16, 16, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    int2 p = int2(gid.xy);
    uint w, h;
    dst.GetDimensions(w, h);
    int2 size = int2(w, h);
    if (p.x >= size.x || p.y >= size.y) {
        return;
    }
    float2 uv = (float2(p) + 0.5) / float2(size);
    uint sw, sh;
    src.GetDimensions(sw, sh);
    float2 half_texel = 0.5 / float2(sw, sh);

    float3 sum = src.SampleLevel(uv, 0.0).rgb * 4.0;
    sum += src.SampleLevel(uv + float2(-half_texel.x, -half_texel.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(half_texel.x, -half_texel.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(-half_texel.x, half_texel.y), 0.0).rgb;
    sum += src.SampleLevel(uv + float2(half_texel.x, half_texel.y), 0.0).rgb;
    float3 c = sum / 8.0;

    if (params.bright_pass != 0) {
        float lum = max(c.r, max(c.g, c.b));
        float wgt = max(lum - params.threshold, 0.0);
        wgt = wgt / max(lum, 1e-4);
        c *= wgt;
    }
    dst[p] = float4(c, 1.0);
}
