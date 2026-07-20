module grid;

// idx -> (i,j,k), x-fastest; nx,ny MUST be powers of two (mask/shift, no div/mod).
public int3 unflatten_pow2(uint idx, int nx, int ny) {
    int lx = int(firstbithigh(uint(nx)));
    int ly = int(firstbithigh(uint(ny)));
    return int3(int(idx & (uint(nx) - 1u)),
                int((idx >> lx) & (uint(ny) - 1u)),
                int(idx >> (lx + ly)));
}

// idx -> (i,j,k), x-fastest; general nx,ny (% and /).
public int3 unflatten_moddiv(uint idx, int nx, int ny) {
    int fi = int(idx);
    return int3(fi % nx, (fi / nx) % ny, fi / (nx * ny));
}

public float3 cell_pos(int3 ijk, float3 box_min, float3 cell_h) {
    return box_min + float3(ijk) * cell_h;
}
