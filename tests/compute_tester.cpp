// bento includes
#include <bento_base/log.h>
#include <bento_base/security.h>
#include <bento_memory/safe_system_allocator.h>
#include <bento_collection/vector.h>
#include <bento_collection/dynamic_string.h>
#include <bento_compute/compute_api.h>

// external includes
#include <algorithm>
#include <string>

const char* TestKernel1D = "\
__kernel void process(__global float* A, __global float* B, const uint elements_per_tile, const uint num_elements)\
{\
	int tileIndex = get_group_id(0);\
	for (int i = 0; i < elements_per_tile; ++i)\
	{\
		int globalIndex = tileIndex * elements_per_tile + i;\
		if (globalIndex >= num_elements)\
			break;\
		B[globalIndex] = pow(A[globalIndex], 2.4f);\
	}\
}";

void test_kernel_1d(bento::ComputeContext context, bento::ComputeCommandList commandList, bento::ComputeKernel kernel, uint32_t numElements, bento::IAllocator& allocator)
{
	// Evaluate the dispatch parameters (one thread per tile)
	uint32_t tileSize = std::min(bento::dispatch_tile_size(context), (uint32_t)1024);
	uint32_t tileCount = (numElements + (tileSize - 1)) / tileSize;

	// Create init values
	bento::Vector<float> input_values(allocator, numElements);
	for (uint32_t data_idx = 0; data_idx < numElements; ++data_idx)
	{
		input_values[data_idx] = (float)data_idx;
	}

	// Create and write init value buffer
	bento::ComputeBuffer input_buffer = bento::create_buffer(context, numElements * sizeof(float), bento::ComputeBufferType::READ_ONLY, allocator);
	bento::write_buffer(commandList, input_buffer, (unsigned char*)&input_values[0]);

	// Create an output buffer
	bento::ComputeBuffer output_buffer = bento::create_buffer(context, numElements * sizeof(float), bento::ComputeBufferType::WRITE_ONLY, allocator);

	// Sent the arguments
	bento::kernel_argument(kernel, 0, input_buffer);
	bento::kernel_argument(kernel, 1, output_buffer);
	bento::kernel_argument(kernel, 2, sizeof(unsigned int), (unsigned char*)&tileSize);
	bento::kernel_argument(kernel, 3, sizeof(unsigned int), (unsigned char*)&numElements);

	// Dispatch the task on the GPU
	bento::IVector3 numTilesParam = { tileCount, 1, 1 };
	bento::IVector3 tileSizes = { tileSize, 1, 1 };
	dispatch_kernel(commandList, kernel, numTilesParam, tileSizes);

	// Wait for the kernel to finish
	bento::flush_command_list(commandList);

	// Read the output values
	bento::Vector<float> output_values(allocator, numElements);
	bento::read_buffer(commandList, output_buffer, (unsigned char*)&output_values[0]);

	// Check the results
	// Check the results
	bool failed = false;
	for (uint32_t element_idx = 0; element_idx < numElements; ++element_idx)
	{
		float expectedValue = std::powf(input_values[element_idx], 2.4f);
		float gpuValue = output_values[element_idx];
		failed |= (abs(expectedValue - gpuValue) > expectedValue * 0.00001);
		if (failed)
			break;
	}
	assert(!failed);

	// Release all the compute resources
	bento::destroy_buffer(output_buffer);
	bento::destroy_buffer(input_buffer);
}

const char* TestKernel2D = "\
__kernel void process(__global float* A, __global float* B, const uint elements_per_tile, const uint elements_per_thread, const uint num_elements)\
{\
	/*Tile data*/\
	int tileIndex = get_group_id(1) * get_num_groups(0) + get_group_id(0);\
\
	/*If this tile has nothing to compute, just skip*/\
	int tileOffset = tileIndex * elements_per_tile;\
	if (tileOffset >= num_elements)\
		return;\
\
	/*Thread data*/\
	int threadIndex = get_local_linear_id();\
\
	/*If this thread has nothing to compute, just skip*/\
	int threadOffset = threadIndex * elements_per_thread;\
	if (threadOffset >= elements_per_tile)\
		return;\
\
	/*Loop through the elements this thread has to compute*/\
	for (int i = 0; i < elements_per_thread; ++i)\
	{\
		int localIndex = threadOffset + i;\
		if (localIndex >= elements_per_tile)\
			break;\
\
		/*Compute the global*/\
		int globalIndex = tileOffset + localIndex;\
\
		/*Run the calculus*/\
		B[globalIndex] = pow(A[globalIndex], 2.7f);\
	}\
}";

// For this test, I am going with a 4x4 tile and 4 elements per thread to process
// this means each tile can process up to 64 elements
void test_kernel_2d(bento::ComputeContext context, bento::ComputeCommandList commandList, bento::ComputeKernel kernel, bento::IAllocator& allocator)
{
	// This implementation is for 16m elements
	uint32_t numElements = 1000000;
	uint32_t numElementsPerTileRow = 512;
	uint32_t elementsPerTile = 61;
	uint32_t elementsPerThread = 4;

	// Create init values
	bento::Vector<float> input_values(allocator, numElements);
	for (uint32_t data_idx = 0; data_idx < numElements; ++data_idx)
	{
		input_values[data_idx] = (float)data_idx;
	}
	// Create and write init value buffer
	bento::ComputeBuffer input_buffer = bento::create_buffer(context, numElements * sizeof(float), bento::ComputeBufferType::READ_ONLY, allocator);
	bento::write_buffer(commandList, input_buffer, (unsigned char*)&input_values[0]);

	// Create an output buffer
	bento::ComputeBuffer output_buffer = bento::create_buffer(context, numElements * sizeof(float), bento::ComputeBufferType::WRITE_ONLY, allocator);

	// Sent the arguments
	bento::kernel_argument(kernel, 0, input_buffer);
	bento::kernel_argument(kernel, 1, output_buffer);
	bento::kernel_argument(kernel, 2, sizeof(unsigned int), (unsigned char*)&elementsPerTile);
	bento::kernel_argument(kernel, 3, sizeof(unsigned int), (unsigned char*)&elementsPerThread);
	bento::kernel_argument(kernel, 4, sizeof(unsigned int), (unsigned char*)&numElements);

	// Dispatch the task on the GPU
	bento::IVector3 numTilesParam = { numElementsPerTileRow, numElementsPerTileRow, 1 };
	bento::IVector3 tileSize = { 4, 4, 1 };
	dispatch_kernel(commandList, kernel, numTilesParam, tileSize);

	// Wait for the kernel to finish
	bento::flush_command_list(commandList);

	// Read the output values
	bento::Vector<float> output_values(allocator, numElements);
	bento::read_buffer(commandList, output_buffer, (unsigned char*)&output_values[0]);

	// Check the results
	bool failed = false;
	for (uint32_t element_idx = 0; element_idx < numElements; ++element_idx)
	{
		float expectedValue = std::powf(input_values[element_idx], 2.7f);
		float gpuValue = output_values[element_idx];
		failed |= (abs(expectedValue - gpuValue) > expectedValue * 0.00001);
		if (failed)
			break;
	}
	assert(!failed);

	// Release all the compute resources
	bento::destroy_buffer(output_buffer);
	bento::destroy_buffer(input_buffer);
}

int main()
{
	// Allocator used for this program
	bento::SafeSystemAllocator allocator;

	// Create a compute context
	bento::ComputeContext context = bento::create_compute_context(allocator);

	// Create a command list
	bento::ComputeCommandList commandList = bento::create_command_list(context, allocator);

	// Run all the variations of the test
	bento::ComputeProgram program1D = bento::create_program_source(context, TestKernel1D);
	bento::ComputeKernel kernel1D = bento::create_kernel(program1D, "process");
	test_kernel_1d(context, commandList, kernel1D, 1, allocator);
	test_kernel_1d(context, commandList, kernel1D, 100, allocator);
	test_kernel_1d(context, commandList, kernel1D, 1024, allocator);
	test_kernel_1d(context, commandList, kernel1D, 1025, allocator);
	test_kernel_1d(context, commandList, kernel1D, 100000, allocator);
	bento::destroy_kernel(kernel1D);
	bento::destroy_program(program1D);

	// run the 2d test
	bento::ComputeProgram program2D = bento::create_program_source(context, TestKernel2D);
	bento::ComputeKernel kernel2D = bento::create_kernel(program2D, "process");
	test_kernel_2d(context, commandList, kernel2D, allocator);
	bento::destroy_kernel(kernel2D);
	bento::destroy_program(program2D);

	// Destroy all the compute structures
	bento::destroy_command_list(commandList);
	bento::destroy_compute_context(context);

	// Display the total allocated memory
	bento::default_logger()->log(bento::LogLevel::info, "Total Allocated memory", std::to_string(allocator.total_memory_allocated()).c_str());

	// Display the success of the test
	bento::default_logger()->log(bento::LogLevel::info, "TESTS", "Compute string succeeded.");

	// AAAAND we're done
	return 0;
}