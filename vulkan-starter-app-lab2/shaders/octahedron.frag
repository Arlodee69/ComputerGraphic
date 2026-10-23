#version 450
layout(location = 0) in vec3 color;
layout(location = 1) in vec3 world_position;
layout(location = 2) in vec3 world_normal;
layout(location = 0) out vec4 fragment_color;
layout(set = 0, binding = 0, std140) uniform SceneData {
    mat4 view_projection;
    vec4 camera;
    vec4 ambient;
    vec4 direction;
    vec4 directional_color;
    uvec4 counts;
} scene;
layout(set = 1, binding = 0, std140) uniform ObjectData {
    mat4 model;
    vec4 albedo;
    vec4 specular;
    vec4 parameters;
} object_data;
struct PointLight { vec4 position_intensity; vec4 color; };
struct SpotLight { vec4 position_intensity; vec4 direction_inner; vec4 color_outer; };
layout(set = 0, binding = 1, std430) readonly buffer PointLights { PointLight lights[]; } points;
layout(set = 0, binding = 2, std430) readonly buffer SpotLights { SpotLight lights[]; } spots;
layout(push_constant) uniform DrawData { mat4 model; vec4 color; } draw_data;

vec3 safe_normalize(vec3 v) { return v * inversesqrt(max(dot(v,v), 1e-10)); }
vec3 blinn_phong(vec3 n, vec3 v, vec3 l, vec3 radiance) {
    float diffuse = max(dot(n,l), 0.0);
    vec3 halfway = safe_normalize(l + v);
    float specular = diffuse > 0.0 ? pow(max(dot(n,halfway), 0.0), object_data.parameters.x) : 0.0;
    return (color * diffuse + object_data.specular.rgb * specular) * radiance;
}
float cone_weight(float cosine, float inner, float outer) {
    if (inner - outer < 1e-5) return step(inner, cosine);
    float t = clamp((cosine - outer) / (inner - outer), 0.0, 1.0);
    return t*t*(3.0 - 2.0*t);
}
void main() {
    if (draw_data.color.a > 0.5) { fragment_color = vec4(color,1.0); return; }
    vec3 n = safe_normalize(world_normal);
    vec3 v = safe_normalize(scene.camera.xyz - world_position);
    vec3 result = color * scene.ambient.rgb * scene.ambient.a;
    result += blinn_phong(n,v,-scene.direction.xyz,scene.directional_color.rgb * scene.direction.a);
    for (uint i = 0; i < scene.counts.x; ++i) {
        PointLight light = points.lights[i];
        vec3 delta = light.position_intensity.xyz - world_position;
        float distance_squared = max(dot(delta,delta), 0.01);
        result += blinn_phong(n,v,safe_normalize(delta),light.color.rgb * light.position_intensity.w / distance_squared);
    }
    for (uint i = 0; i < scene.counts.y; ++i) {
        SpotLight light = spots.lights[i];
        vec3 delta = light.position_intensity.xyz - world_position;
        vec3 l = safe_normalize(delta);
        float cone = cone_weight(dot(-l,light.direction_inner.xyz),light.direction_inner.w,light.color_outer.w);
        float distance_squared = max(dot(delta,delta), 0.01);
        result += blinn_phong(n,v,l,light.color_outer.rgb * light.position_intensity.w * cone / distance_squared);
    }
    // Reinhard mapping preserves bright highlights without hard clipping.
    fragment_color = vec4(result / (vec3(1.0) + result), 1.0);
}
