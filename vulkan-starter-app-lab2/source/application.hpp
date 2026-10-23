#pragma once

#include "graphics_internal.hpp"

namespace application {

bool initialize();
void shutdown();

void update(double time);
void render(const graphics::internal::FrameData& fd);

bool self_test();
void verification_step(uint32_t frame);

} // namespace application
