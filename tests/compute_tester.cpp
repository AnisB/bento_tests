// SDK includes
#include <bento_base/log.h>
#include <bento_collection/vector.h>
#include <bento_base/security.h>
#include <bento_compute/compute_api.h>
#include <bento_math/types.h>
#include <bento_memory/safe_system_allocator.h>

void square_number_test(bento::ComputeContext context, bento::ComputeCommandList commandList, bento::IAllocator& currentAllocator)
{
	const char* squareKernel = "\
	__kernel void process(__global uint* A, __global uint* B, const uint num_elements)\
	{\
		int id = get_global_id(0);\
		int num_workers = get_global_size(0);\
	\
		for (int i = id; i < num_elements; i += num_workers)\
		{\
			B[i] = A[i] * A[i];\
		}\
	}";

	// Define a number of element to process
	const uint32_t num_elements = 1000000;

	// Create and write init value buffer
	bento::ComputeBuffer input_buffer = create_buffer(context, num_elements * sizeof(float), bento::ComputeBufferType::READ_ONLY, currentAllocator);

	// Create init values
	bento::Vector<uint32_t> input_values(currentAllocator, num_elements);
	for (uint32_t data_idx = 0; data_idx < num_elements; ++data_idx)
		input_values[data_idx] = data_idx;

	bento::write_buffer(commandList, input_buffer, (unsigned char*)&input_values[0]);

	// Create an output buffer
	bento::ComputeBuffer output_buffer = create_buffer(context, num_elements * sizeof(float), bento::ComputeBufferType::WRITE_ONLY, currentAllocator);

	// Create a program and a kernel
	bento::ComputeProgram program = bento::create_program_source(context, squareKernel);
	bento::ComputeKernel kernel = bento::create_kernel(program, "process");

	// Sent the arguments
	bento::kernel_argument(kernel, 0, input_buffer);
	bento::kernel_argument(kernel, 1, output_buffer);
	bento::kernel_argument(kernel, 2, sizeof(unsigned int), (unsigned char*)&num_elements);

	// Launch the kernel
	bento::dispatch_kernel_1D(commandList, kernel, num_elements);

	// Wait for the kernel to finish
	bento::flush_command_list(commandList);

	// Read the output values
	bento::Vector<uint32_t> output_values(currentAllocator, num_elements);
	bento::read_buffer(commandList, output_buffer, (unsigned char*)&output_values[0]);

	// Check the results
	for (uint32_t element_idx = 0; element_idx < num_elements; ++element_idx)
		assert((input_values[element_idx] * input_values[element_idx]) == output_values[element_idx]);

	// Release the kernel and program
	bento::destroy_kernel(kernel);
	bento::destroy_program(program);

	// Release all the resources
	bento::destroy_buffer(output_buffer);
	bento::destroy_buffer(input_buffer);
}

void divide_number_test(bento::ComputeContext context, bento::ComputeCommandList commandList, bento::IAllocator& currentAllocator)
{
	const char* divideKernel = "\
	__kernel void process(__global double* A, __global double* B, __global double* C, const uint num_elements)\
	{\
		int id = get_global_id(0);\
		int num_workers = get_global_size(0);\
	\
		for (int i = id; i < num_elements; i += num_workers)\
		{\
			C[i] = A[i] / B[i];\
		}\
	}";

	// Define a number of element to process
	const uint32_t num_elements = 4000000;

	// Create and write init value buffers
	bento::ComputeBuffer input_buffer_0 = create_buffer(context, num_elements * sizeof(double), bento::ComputeBufferType::READ_ONLY, currentAllocator);
	bento::ComputeBuffer input_buffer_1 = create_buffer(context, num_elements * sizeof(double), bento::ComputeBufferType::READ_ONLY, currentAllocator);

	// Create init values
	bento::Vector<double> input_values_0(currentAllocator, num_elements);
	bento::Vector<double> input_values_1(currentAllocator, num_elements);
	for (uint32_t data_idx = 0; data_idx < num_elements; ++data_idx)
	{
		input_values_0[data_idx] = (double)data_idx;
		input_values_1[data_idx] = (double)data_idx * data_idx;
	}

	// Write them to the gpu
	bento::write_buffer(commandList, input_buffer_0, (void*)&input_values_0[0]);
	bento::write_buffer(commandList, input_buffer_1, (void*)&input_values_1[0]);

	// Create an output buffer
	bento::ComputeBuffer output_buffer = create_buffer(context, num_elements * sizeof(double), bento::ComputeBufferType::WRITE_ONLY, currentAllocator);

	// Create a program and a kernel
	bento::ComputeProgram program = bento::create_program_source(context, divideKernel);
	bento::ComputeKernel kernel = bento::create_kernel(program, "process");

	// Sent the arguments
	bento::kernel_argument(kernel, 0, input_buffer_0);
	bento::kernel_argument(kernel, 1, input_buffer_1);
	bento::kernel_argument(kernel, 2, output_buffer);
	bento::kernel_argument(kernel, 3, sizeof(unsigned int), (void*)&num_elements);

	// Launch the kernel
	bento::dispatch_kernel_1D(commandList, kernel, num_elements);

	// Wait for the kernel to finish
	bento::flush_command_list(commandList);

	// Read the output values
	bento::Vector<double> output_values(currentAllocator, num_elements);
	bento::read_buffer(commandList, output_buffer, (unsigned char*)&output_values[0]);

	// Check the results
	for (uint32_t element_idx = 0; element_idx < num_elements; ++element_idx)
		assert((input_values_0[element_idx] / input_values_1[element_idx]) == output_values[element_idx]);

	// Release the kernel and program
	bento::destroy_kernel(kernel);
	bento::destroy_program(program);

	// Release all the resources
	bento::destroy_buffer(output_buffer);
	bento::destroy_buffer(input_buffer_0);
	bento::destroy_buffer(input_buffer_1);
}

float rand_01()
{
	return (float)(rand() % RAND_MAX) / RAND_MAX;
}

void particle_test(bento::ComputeContext context, bento::ComputeCommandList commandList, bento::IAllocator& currentAllocator)
{
	const char* particleKernel = "\
		typedef struct __attribute__((packed))\
		{\
			float4 pos;\
			float4 velocity;\
		} TParticleDescriptor;\
		\
		__kernel void update(__global TParticleDescriptor* particleArray, const float deltaTime, const int numParticles)\
		{\
			int id = get_global_id(0);\
			int num_workers = get_global_size(0);\
			for (int i = id; i < numParticles; i += num_workers)\
			{\
				particleArray[i].pos = particleArray[i].pos + particleArray[i].velocity * deltaTime; \
			}\
		}";

	// Define a number of element to process
	const uint32_t numParticles = 1000000;

	// Internal particle descriptor
	struct TParticleDescriptor
	{
		bento::Vector4 pos;
		bento::Vector4 velocity;
	};

	// Initialize the random number generation
	srand(666);

	// Create and write init value buffer
	bento::ComputeBuffer particleBuffer = create_buffer(context, numParticles * sizeof(TParticleDescriptor), bento::ComputeBufferType::READ_WRITE, currentAllocator);

	// Create init values
	bento::Vector<TParticleDescriptor> inputParticles(currentAllocator, numParticles);
	for (uint32_t particleIdx = 0; particleIdx < numParticles; ++particleIdx)
	{
		inputParticles[particleIdx].pos.x = rand_01() * 2.0f - 1.0f;
		inputParticles[particleIdx].pos.y = rand_01() * 2.0f - 1.0f;
		inputParticles[particleIdx].pos.z = rand_01() * 2.0f - 1.0f;
		inputParticles[particleIdx].velocity.x = rand_01() * 2.0f - 1.0f;
		inputParticles[particleIdx].velocity.y = rand_01() * 2.0f - 1.0f;
		inputParticles[particleIdx].velocity.z = rand_01() * 2.0f - 1.0f;
	}

	bento::write_buffer(commandList, particleBuffer, (void*)&inputParticles[0]);

	// Create a program and a kernel
	bento::ComputeProgram program = bento::create_program_source(context, particleKernel);
	bento::ComputeKernel kernel = bento::create_kernel(program, "update");

	// Update duration
	const float deltaTime = 1.0;

	// Sent the arguments
	bento::kernel_argument(kernel, 0, particleBuffer);
	bento::kernel_argument(kernel, 1, sizeof(deltaTime), (void*)&deltaTime);
	bento::kernel_argument(kernel, 2, sizeof(unsigned int), (unsigned char*)&numParticles);

	// Launch the kernel
	bento::dispatch_kernel_1D(commandList, kernel, numParticles);

	// Wait for the kernel to finish
	bento::flush_command_list(commandList);

	// Read the output values
	bento::Vector<TParticleDescriptor> outputParticles(currentAllocator, numParticles);
	bento::read_buffer(commandList, particleBuffer, (unsigned char*)&outputParticles[0]);

	// Check the results
	for (uint32_t element_idx = 0; element_idx < numParticles; ++element_idx)
	{
		assert((inputParticles[element_idx].pos.x + inputParticles[element_idx].velocity.x * deltaTime) == outputParticles[element_idx].pos.x);
		assert((inputParticles[element_idx].pos.y + inputParticles[element_idx].velocity.y * deltaTime) == outputParticles[element_idx].pos.y);
		assert((inputParticles[element_idx].pos.z + inputParticles[element_idx].velocity.z * deltaTime) == outputParticles[element_idx].pos.z);
	}

	// Release the kernel and program
	bento::destroy_kernel(kernel);
	bento::destroy_program(program);

	// Release all the resources
	bento::destroy_buffer(particleBuffer);
}

int main()
{
	// Allocator used for this program
	bento::SafeSystemAllocator currentAllocator;

	// Create a compute context
	bento::ComputeContext context = create_compute_context(currentAllocator);

	// Create a command list
	bento::ComputeCommandList commandList = create_command_list(context, currentAllocator);

	// Run the tests
	square_number_test(context, commandList, currentAllocator);
	divide_number_test(context, commandList, currentAllocator);
	particle_test(context, commandList, currentAllocator);

	// Destroy the command list
	bento::destroy_command_list(commandList);

	// Destroy the compute context
	bento::destroy_compute_context(context);

	// If we get here everything is fine
	return 0;
}