// Fused kick+absorber: diagonal factors commute -> exact single pass.
// coef = -dt full kick (merged half kicks) or -dt/2 trailing half kick.
import complex;

[[vk::binding(0, 0)]] RWStructuredBuffer<Complex, Std430DataLayout> psi;
[[vk::binding(1, 0)]] StructuredBuffer<float, Std430DataLayout> v;
[[vk::binding(2, 0)]] StructuredBuffer<float, Std430DataLayout> m;

struct Params {
    uint n;
    float coef;
};
[[vk::binding(3, 0)]] ConstantBuffer<Params, Std140DataLayout> params;

[shader("compute")]
[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x;
    if (i >= params.n) {
        return;
    }
    psi[i] = (psi[i] * cis(params.coef * v[i])) * m[i];
}
