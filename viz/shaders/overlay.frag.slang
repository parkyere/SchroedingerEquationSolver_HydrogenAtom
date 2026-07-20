// v_color is premultiplied for the premultiplied-alpha blend.
struct FSIn {
    float4 pos : SV_Position;
    [[vk::location(0)]] float4 v_color : TEXCOORD0;
};

[shader("fragment")]
float4 main(FSIn input) : SV_Target {
    return input.v_color;
}
