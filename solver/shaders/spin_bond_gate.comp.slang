// Heisenberg bond gate exp(+i theta sigma_i.sigma_j); sigma.sigma = 2 SWAP - 1
// -> parallel amplitudes phase, antiparallel pair mixes. Thread per disjoint
// 4-amplitude block (quarter_n = 2^(N-2)).
// CONTRACT: vkcheck check_spin_step.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> data;

struct Params {
    uint quarter_n;
    uint site_i;
    uint site_j;
    uint pad;
    float4 gate;     // phase.re,im, diag.re,im
    float4 off4;     // off.re,im, pad
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint t = tid.x;
    if (t >= params.quarter_n) {
        return;
    }
    // base = t with 0 bits inserted at p0<p1, clearing both sites.
    uint p0 = min(params.site_i, params.site_j);
    uint p1 = max(params.site_i, params.site_j);
    uint low = t & ((1u << p0) - 1u);
    uint mid = (t >> p0) & ((1u << (p1 - p0 - 1u)) - 1u);
    uint high = t >> (p1 - 1u);
    uint base = low | (mid << (p0 + 1u)) | (high << (p1 + 1u));
    uint bi = 1u << params.site_i;
    uint bj = 1u << params.site_j;
    Complex phase = Complex(params.gate.xy);
    Complex diag = Complex(params.gate.zw);
    Complex off = Complex(params.off4.xy);
    data[base] = phase * data[base];
    data[base | bi | bj] = phase * data[base | bi | bj];
    Complex a = data[base | bj];
    Complex b = data[base | bi];
    data[base | bj] = diag * a + off * b;
    data[base | bi] = off * a + diag * b;
}
