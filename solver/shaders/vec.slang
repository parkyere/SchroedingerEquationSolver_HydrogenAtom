// Ternary not v[i]: dynamic index spills float3 to addressable Function storage.
// Shared by the clip/slice render shaders.
module vec;

public float comp(float3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

public void set_comp(inout float3 v, int i, float x) {
    if (i == 0) v.x = x;
    else if (i == 1) v.y = x;
    else v.z = x;
}
