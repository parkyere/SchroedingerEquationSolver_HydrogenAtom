// Complex == bare float2 (re, im) under Std430; buffers bind as Complex or float2 interchangeably (see pack_half / physics kernels).
module complex;

public struct Complex {
    public float2 v;

    public __init(float2 f) { v = f; }
    public __init(float re, float im) { v = float2(re, im); }

    public property float re {
        get { return v.x; }
        set { v.x = newValue; }
    }
    public property float im {
        get { return v.y; }
        set { v.y = newValue; }
    }
}

public Complex operator+(Complex a, Complex b) { return Complex(a.v + b.v); }
public Complex operator-(Complex a, Complex b) { return Complex(a.v - b.v); }

public Complex operator*(Complex a, Complex b) {
    return Complex(a.v.x * b.v.x - a.v.y * b.v.y,
                   a.v.x * b.v.y + a.v.y * b.v.x);
}

public Complex operator*(Complex a, float s) { return Complex(a.v * s); }
public Complex operator*(float s, Complex a) { return Complex(a.v * s); }
public Complex operator/(Complex a, float s) { return Complex(a.v / s); }

public Complex conj(Complex a) { return Complex(a.v.x, -a.v.y); }
public float norm2(Complex a) { return dot(a.v, a.v); }
public float arg(Complex a) { return atan2(a.v.y, a.v.x); }

public Complex mix(Complex a, Complex b, float t) { return Complex(lerp(a.v, b.v, t)); }

// mul_conj: term order fixed for bit-reproducible FMA contraction.
public Complex mul_conj(Complex a, Complex b) {
    return Complex(a.v.x * b.v.x + a.v.y * b.v.y,
                   a.v.x * b.v.y - a.v.y * b.v.x);
}
public Complex cis(float theta) { return Complex(cos(theta), sin(theta)); }
