#version 450
layout(location = 0) in vec3 color;
layout(location = 1) in vec3 world_position;
layout(location = 0) out vec4 fragment_color;
void main() {
    vec3 normal = normalize(cross(dFdy(world_position), dFdx(world_position)));
    float light = 0.35 + 0.65 * max(dot(normal, normalize(vec3(0.4, 0.8, 0.6))), 0.0);
    fragment_color = vec4(color * light, 1.0);
}
