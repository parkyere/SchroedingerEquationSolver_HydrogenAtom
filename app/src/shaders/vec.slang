// vec.slang -- dynamic vector-component access by a runtime axis index (0/1/2)
// into a float3, shared by the clip/slice render shaders. The ternary stays in
// registers; native v[i] with a dynamic index would spill the vector to
// addressable Function storage. Import with `import vec;`.
module vec;

public float comp(float3 v, int i) { return i == 0 ? v.x : (i == 1 ? v.y : v.z); }

public void set_comp(inout float3 v, int i, float x) {
    if (i == 0) v.x = x;
    else if (i == 1) v.y = x;
    else v.z = x;
}
