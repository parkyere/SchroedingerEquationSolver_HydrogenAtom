// Single-site SU(2) gate on the exact 2^N spin state; one thread per amplitude
// pair, half_n = 2^(N-1) via insert-zero-bit index (no idle threads).
// data: fp32 Complex (float2). CONTRACT: vkcheck check_spin_step.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    uint half_n;
    uint site;
    uint pad0;
    uint pad1;
    float4 row0;
    float4 row1;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= params.half_n) {
        return;
    }
    // Insert 0 bit at `site` to form the pair index lo (bit clear).
    uint bit = 1u << params.site;
    uint low = t & (bit - 1u);
    uint high = (t & ~(bit - 1u)) << 1u;
    uint lo = high | low;
    uint hi = lo | bit;
    Complex up = data[lo];
    Complex dn = data[hi];
    Complex a00 = Complex(params.row0.xy);
    Complex a01 = Complex(params.row0.zw);
    Complex a10 = Complex(params.row1.xy);
    Complex a11 = Complex(params.row1.zw);
    data[lo] = a00 * up + a01 * dn;
    data[hi] = a10 * up + a11 * dn;
}
