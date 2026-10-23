#version 450
layout(location = 0) in vec3 position;
layout(location = 1) in vec3 vertex_color;
layout(set = 0, binding = 0, std140) uniform ObjectData {
    mat4 mvp;
    mat4 model;
    vec4 tint;
    vec4 options;
} object_data;
layout(location = 0) out vec3 color;
layout(location = 1) out vec3 world_position;
void main() {
    gl_Position = object_data.mvp * vec4(position, 1.0);
    world_position = (object_data.model * vec4(position, 1.0)).xyz;
    color = mix(vec3(1.0), vertex_color, object_data.options.x) * object_data.tint.rgb;
}
