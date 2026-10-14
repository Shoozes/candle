#include <cuda_runtime.h>

namespace {

__device__ __forceinline__ float fp4_value(unsigned int nibble) {
    switch (nibble & 0x0f) {
    case 0: return 0.0f;
    case 1: return 0.5f;
    case 2: return 1.0f;
    case 3: return 1.5f;
    case 4: return 2.0f;
    case 5: return 3.0f;
    case 6: return 4.0f;
    case 7: return 6.0f;
    case 8: return -0.0f;
    case 9: return -0.5f;
    case 10: return -1.0f;
    case 11: return -1.5f;
    case 12: return -2.0f;
    case 13: return -3.0f;
    case 14: return -4.0f;
    default: return -6.0f;
    }
}

__global__ void gpt_oss_mxfp4_matmul_kernel(
    const float *input,
    const unsigned char *blocks,
    const unsigned char *scales,
    const unsigned int *experts,
    float *output,
    int route_count,
    int output_width,
    int input_width,
    int expert_count) {
    const int output_index = blockIdx.x * blockDim.x + threadIdx.x;
    const int route_index = blockIdx.y;
    if (route_index >= route_count || output_index >= output_width) {
        return;
    }

    const unsigned int expert = experts[route_index];
    const int blocks_per_row = input_width / 32;
    float sum = 0.0f;
    if (expert >= static_cast<unsigned int>(expert_count)) {
        output[route_index * output_width + output_index] = nanf("");
        return;
    }
    for (int input_index = 0; input_index < input_width; ++input_index) {
        const int block_index =
            ((static_cast<int>(expert) * output_width + output_index) * blocks_per_row) +
            (input_index / 32);
        const unsigned char packed = blocks[block_index * 16 + (input_index % 32) / 2];
        const unsigned int nibble = (input_index % 2 == 0) ? (packed & 0x0f) : (packed >> 4);
        const float scale = exp2f(static_cast<float>(scales[block_index]) - 127.0f);
        sum += input[route_index * input_width + input_index] * fp4_value(nibble) * scale;
    }
    output[route_index * output_width + output_index] = sum;
}

} // namespace

namespace {
__global__ void gpt_oss_bf16_f32_matmul_kernel(
    const float *input, const unsigned short *weight, float *output,
    int batch, int rows, int width) {
    const unsigned int lane = threadIdx.x % 32;
    const unsigned long long row = static_cast<unsigned long long>(blockIdx.x) * 4 + threadIdx.x / 32;
    const unsigned int b = blockIdx.y;
    if (row >= static_cast<unsigned int>(rows) || b >= static_cast<unsigned int>(batch)) return;
    float sum = 0.0f;
    for (unsigned long long col = lane; col < static_cast<unsigned int>(width); col += 32) {
        const float w = __uint_as_float(static_cast<unsigned int>(weight[row * width + col]) << 16);
        sum = __fadd_rn(sum, __fmul_rn(input[static_cast<unsigned long long>(b) * width + col], w));
    }
    for (int delta = 16; delta > 0; delta /= 2) sum = __fadd_rn(sum, __shfl_down_sync(0xffffffff, sum, delta));
    if (lane == 0) output[static_cast<unsigned long long>(b) * rows + row] = sum;
}
} // namespace

extern "C" int launch_gpt_oss_bf16_f32_matmul(
    const float *input, const unsigned short *weight, float *output,
    int batch, int rows, int width, long long stream) {
    if (batch <= 0 || batch > 65535 || rows <= 0 || width <= 0) return static_cast<int>(cudaErrorInvalidValue);
    const dim3 block(128, 1, 1);
    const dim3 grid((static_cast<unsigned int>(rows) + 3) / 4, static_cast<unsigned int>(batch), 1);
    gpt_oss_bf16_f32_matmul_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        input, weight, output, batch, rows, width);
    return static_cast<int>(cudaGetLastError());
}

extern "C" int launch_gpt_oss_mxfp4_matmul(
    const float *input,
    const unsigned char *blocks,
    const unsigned char *scales,
    const unsigned int *experts,
    float *output,
    int route_count,
    int output_width,
    int input_width,
    int expert_count,
    long long stream) {
    const dim3 block(128, 1, 1);
    const dim3 grid(
        (static_cast<unsigned int>(output_width) + block.x - 1) / block.x,
        static_cast<unsigned int>(route_count),
        1);
    gpt_oss_mxfp4_matmul_kernel<<<grid, block, 0, reinterpret_cast<cudaStream_t>(stream)>>>(
        input, blocks, scales, experts, output, route_count, output_width, input_width,
        expert_count);
    return static_cast<int>(cudaGetLastError());
}
