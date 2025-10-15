#version 460 core
#extension GL_GOOGLE_include_directive : enable

layout(location = 0) in vec3 vColor;
layout(location = 1) in float vDensity;
layout(location = 2) in float vFeature;
layout(location = 3) in float vRadius;
layout(location = 4) in float vEmissive;
layout(location = 5) in vec3 vWorldPos;
layout(location = 6) in float vViewDepth;

layout(location = 0) out vec4 outColor;

layout(set = 0, binding = 0) uniform CameraUniform {
    mat4 view;
    mat4 proj;
    mat4 viewProj;
    vec4 cameraPos;
    vec4 viewport;   // width, height, near, far
    vec4 boundsMin;
    vec4 boundsMax;
    vec4 extras;     // time, dt, pointCount, padding
} uCamera;

layout(push_constant) uniform PushConstants {
    float pointScale;
    float intensityScale;
    float opacityScale;
    float densityMin;
    float densityMax;
    float gamma;
    float splatSharpness;
    float depthFade;
    uint  colorMode;
    uint  animate;
    float hueShift;
    float pad0;
} pc;

float saturate(float v) {
    return clamp(v, 0.0, 1.0);
}

vec3 viridis(float t) {
    const vec3 c0 = vec3(0.267004, 0.004874, 0.329415);
    const vec3 c1 = vec3(0.282327, 0.094955, 0.417331);
    const vec3 c2 = vec3(0.253935, 0.265254, 0.529983);
    const vec3 c3 = vec3(0.206756, 0.371758, 0.553117);
    const vec3 c4 = vec3(0.163625, 0.471133, 0.558148);
    const vec3 c5 = vec3(0.134692, 0.658636, 0.517649);
    const vec3 c6 = vec3(0.477504, 0.821444, 0.318195);
    const vec3 c7 = vec3(0.993248, 0.906157, 0.143936);
    vec3 stops[8] = vec3[8](c0, c1, c2, c3, c4, c5, c6, c7);
    t = saturate(t);
    float scaled = t * 7.0;
    int idx = int(floor(scaled));
    int next = min(idx + 1, 7);
    float f = scaled - float(idx);
    return mix(stops[idx], stops[next], f);
}

vec3 rgb2hsv(vec3 c) {
    float maxc = max(c.r, max(c.g, c.b));
    float minc = min(c.r, min(c.g, c.b));
    float delta = maxc - minc;
    float h = 0.0;
    if (delta > 1e-6) {
        if (maxc == c.r) {
            h = mod((c.g - c.b) / delta, 6.0);
        } else if (maxc == c.g) {
            h = (c.b - c.r) / delta + 2.0;
        } else {
            h = (c.r - c.g) / delta + 4.0;
        }
        h /= 6.0;
    }
    float s = maxc <= 0.0 ? 0.0 : delta / maxc;
    return vec3(h, s, maxc);
}

vec3 hsv2rgb(vec3 c) {
    float h = c.x * 6.0;
    float s = clamp(c.y, 0.0, 1.0);
    float v = clamp(c.z, 0.0, 20.0);
    int i = int(floor(h));
    float f = h - float(i);
    float p = v * (1.0 - s);
    float q = v * (1.0 - s * f);
    float t = v * (1.0 - s * (1.0 - f));
    vec3 result;
    if (i == 0) result = vec3(v, t, p);
    else if (i == 1) result = vec3(q, v, p);
    else if (i == 2) result = vec3(p, v, t);
    else if (i == 3) result = vec3(p, q, v);
    else if (i == 4) result = vec3(t, p, v);
    else result = vec3(v, p, q);
    return result;
}

vec3 apply_hue_shift(vec3 color, float hueShift) {
    vec3 hsv = rgb2hsv(color);
    hsv.x = fract(hsv.x + hueShift);
    return hsv2rgb(hsv);
}

float radial_falloff(vec2 coord, float sharpness) {
    vec2 p = coord * 2.0 - 1.0;
    float r2 = dot(p, p);
    return pow(max(0.0, 1.0 - r2), max(sharpness, 1.0));
}

vec3 apply_color_mode(vec3 baseColor, float densityNorm, float heightNorm, float featureNorm, float emissive) {
    if (pc.colorMode == 1u) {
        return viridis(densityNorm);
    }
    if (pc.colorMode == 2u) {
        vec3 low = vec3(0.12, 0.24, 0.42);
        vec3 mid = vec3(0.75, 0.45, 0.35);
        vec3 high = vec3(0.95, 0.86, 0.48);
        vec3 warm = mix(low, mid, heightNorm);
        vec3 highlight = mix(mid, high, smoothstep(0.65, 1.0, heightNorm));
        return mix(warm, highlight, densityNorm);
    }
    if (pc.colorMode == 3u) {
        vec3 cold = vec3(0.1, 0.25, 0.6);
        vec3 hot = vec3(0.95, 0.35, 0.3);
        return mix(cold, hot, featureNorm);
    }
    return baseColor;
}

void main() {
    float densityRange = max(pc.densityMax - pc.densityMin, 1e-6);
    float densityNorm = saturate((vDensity - pc.densityMin) / densityRange);
    float heightNorm = saturate((vWorldPos.y - uCamera.boundsMin.y) / max(1e-4, uCamera.boundsMax.y - uCamera.boundsMin.y));
    float featureNorm = saturate(vFeature);

    vec3 base = vColor;
    base = apply_color_mode(base, densityNorm, heightNorm, featureNorm, vEmissive);
    base = max(base, vec3(0.0));
    if (abs(pc.hueShift) > 1e-4) {
        base = apply_hue_shift(base, pc.hueShift);
    }
    if (pc.animate != 0u) {
        float time = uCamera.extras.x;
        base *= 0.9 + 0.15 * sin(time * 1.35 + featureNorm * 6.28318);
    }

    float falloff = radial_falloff(gl_PointCoord, pc.splatSharpness);
    float opacity = falloff * (0.25 + 0.75 * densityNorm);
    opacity *= pc.opacityScale;
    float depthFade = exp(-pc.depthFade * max(0.0, vViewDepth));
    opacity *= depthFade;
    opacity = saturate(opacity);

    vec3 emissiveColor = vec3(1.0, 0.86, 0.55) * vEmissive * 0.8;
    vec3 color = (base + emissiveColor) * pc.intensityScale;
    color = pow(color, vec3(pc.gamma));

    outColor = vec4(color, opacity);
    if (outColor.a <= 0.001) {
        discard;
    }
}
