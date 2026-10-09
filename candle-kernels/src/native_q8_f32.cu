#include <cuda_fp16.h>
#include <stdint.h>

// Q8_0 GGUF blocks: one F16 scale followed by 32 signed byte coefficients.
// Activation values stay F32; no activation quantization or dense weight buffer.
extern "C" __global__ void native_q8_f32_matmul(
    const unsigned char *weights, const float *input, float *output,
    int64_t rows, int64_t width, int64_t columns) {
    const int lane = threadIdx.x;
    const int64_t cell = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    if (cell >= rows * columns) return;
    const int64_t row = cell / columns;
    const int64_t column = cell % columns;
    const int64_t blocks = width / 32;
    float sum = 0.0f;
    for (int64_t block = 0; block < blocks; ++block) {
        const unsigned char *packed = weights + (column * blocks + block) * 34;
        const float scale = __half2float(*reinterpret_cast<const __half *>(packed));
        const float weight = (float)reinterpret_cast<const signed char *>(packed + 2)[lane] * scale;
        sum = __fmaf_rn(weight, input[row * width + block * 32 + lane], sum);
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        sum += __shfl_down_sync(0xffffffff, sum, offset);
    if (lane == 0) output[cell] = sum;
}

// One warp owns a column and four input rows. Decode each block once; keep
// each row's original lane-wise FMA order and warp reduction unchanged.
extern "C" __global__ void native_q8_f32_matmul_rows4(
    const unsigned char *weights, const float *input, float *output,
    int64_t rows, int64_t width, int64_t columns) {
    const int lane = threadIdx.x;
    const int64_t tile = (int64_t)blockIdx.x * blockDim.y + threadIdx.y;
    const int64_t row = (tile / columns) * 4;
    const int64_t column = tile % columns;
    if (row >= rows) return;
    const int64_t blocks = width / 32;
    float sums[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int64_t block = 0; block < blocks; ++block) {
        const unsigned char *packed = weights + (column * blocks + block) * 34;
        const float scale = __half2float(*reinterpret_cast<const __half *>(packed));
        const float weight = (float)reinterpret_cast<const signed char *>(packed + 2)[lane] * scale;
        #pragma unroll
        for (int r = 0; r < 4; ++r) {
            if (row + r < rows)
                sums[r] = __fmaf_rn(weight, input[(row + r) * width + block * 32 + lane], sums[r]);
        }
    }
    #pragma unroll
    for (int r = 0; r < 4; ++r) {
        if (row + r < rows) {
            for (int offset = 16; offset > 0; offset >>= 1)
                sums[r] += __shfl_down_sync(0xffffffff, sums[r], offset);
            if (lane == 0) output[(row + r) * columns + column] = sums[r];
        }
    }
}
