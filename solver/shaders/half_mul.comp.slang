// Strang half-kick; coef = -dt/2, psi *= exp(i*coef*V), V = R32 potential.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<float, Std430DataLayout> v;

struct Params {
    uint n;
    float coef;
};
[[vk::binding(2, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    psi[i] = psi[i] * cis(params.coef * v[i]);
}
