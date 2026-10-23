#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 vertex_color;
layout(location = 2) in vec3 vertex_normal;
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
layout(push_constant) uniform DrawData { mat4 model; vec4 color; } draw_data;
layout(location = 0) out vec3 color;
layout(location = 1) out vec3 world_position;
layout(location = 2) out vec3 world_normal;
void main() {
    bool marker = draw_data.color.a > 0.5;
    mat4 model = marker ? draw_data.model : object_data.model;
    world_position = (model * vec4(position, 1.0)).xyz;
    gl_Position = scene.view_projection * vec4(world_position, 1.0);
    world_normal = transpose(inverse(mat3(model))) * vertex_normal;
    color = marker ? draw_data.color.rgb : mix(vec3(1.0), vertex_color, object_data.parameters.y) * object_data.albedo.rgb;
}
