// SDK includes
#include <bento_base/log.h>
#include <bento_collection/vector.h>
#include <bento_base/security.h>
#include <bento_math/types.h>
#include <bento_memory/safe_system_allocator.h>
#include <bento_graphics/vulkan_backend.h>

int main()
{
	// Allocator used for this program
	bento::SafeSystemAllocator currentAllocator;

	// Create the render system
	bento::vulkan::render_system::init_render_system();

	// Create the render environment
	bento::RenderEnvironment renderEnv = bento::vulkan::render_system::create_render_environment(1280, 720, "vulkan_tester", currentAllocator);

	// Create the command buffer
	bento::CommandBuffer commandBuffer = bento::vulkan::command_buffer::create(renderEnv, currentAllocator);

	// Start to record the command buffer
	bento::vulkan::command_buffer::start_record(commandBuffer);

	// Wait 5000 milliseconds
	SLEEP_FUNCTION(5000);
	
	// Stop to record the command buffer
	bento::vulkan::command_buffer::stop_record(commandBuffer);

	// Destroy the command buffer
	bento::vulkan::command_buffer::destroy(commandBuffer);

	// Destroy the render environment
	bento::vulkan::render_system::destroy_render_environment(renderEnv);

	// Destroy the render system
	bento::vulkan::render_system::shutdown_render_system();

	// If we get here everything is fine
	return 0;
}