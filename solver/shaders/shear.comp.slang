// One shear of the three-shear exact rotation (paramagnetic): in mixed rep
// (FFT freq_axis, real coord_axis) a coordinate shift becomes a phase multiply.
// std140 Params must match host struct field order (packing contract).
import complex;
import grid;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;

struct Params {
    uint n;
    int nx;
    int ny;
    int nz;
    int freq_axis;   // 0=x,1=y,2=z
    int coord_axis;
    int nf;
    float kscale;    // 2*pi / L_freq
    float cmin;
    float ch;
    float coeff;
};
[[vk::binding(1, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint idx = tid.x;
    if (idx >= params.n) {
        return;
    }
    int3 v = unflatten_moddiv(idx, params.nx, params.ny);
    int e[3] = { v.x, v.y, v.z };
    int f = e[params.freq_axis];
    int p = e[params.coord_axis];
    // 2f<nf (not f<nf/2): keeps DC=0 at nf=1; matches CPU fftfreq contract.
    int sf = (2 * f < params.nf) ? f : f - params.nf;
    float kf = params.kscale * float(sf);
    float d = params.coeff * (params.cmin + float(p) * params.ch);
    psi[idx] = psi[idx] * cis(-kf * d);
}
