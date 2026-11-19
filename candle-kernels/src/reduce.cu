#include "cuda_utils.cuh"
#include <cmath>
#include <stdint.h>
#include <cuda/std/limits>

#define WARP_SIZE 32
const int BLOCK_SIZE = 1024;

// Helpers to initialize reduction identities for both floating-point and
// integer types. For floats we keep using +/-INFINITY, while for integers
// we use well-defined numeric_limits values instead of relying on casting
// +/-INFINITY to an integer type (which is undefined behaviour and has been
// observed to break on newer GPU architectures such as Blackwell).
template <typename T>
__device__ __forceinline__ T reduce_init_lowest() {
  // Default implementation is used for floating-point types (__half,
  // __nv_bfloat16, float, double). The conversion from -INFINITY (double)
  // to these types is well-defined and produces -inf.
  return -INFINITY;
}

template <typename T>
__device__ __forceinline__ T reduce_init_highest() {
  // Default implementation is used for floating-point types (__half,
  // __nv_bfloat16, float, double). The conversion from INFINITY (double)
  // to these types is well-defined and produces +inf.
  return INFINITY;
}

// Integer specializations – use numeric_limits instead of +/-INFINITY.
template <>
__device__ __forceinline__ int64_t reduce_init_lowest<int64_t>() {
  return ::cuda::std::numeric_limits<int64_t>::lowest();
}

template <>
__device__ __forceinline__ uint32_t reduce_init_lowest<uint32_t>() {
  return ::cuda::std::numeric_limits<uint32_t>::lowest();
}

template <>
__device__ __forceinline__ uint8_t reduce_init_lowest<uint8_t>() {
  return ::cuda::std::numeric_limits<uint8_t>::lowest();
}

template <>
__device__ __forceinline__ int64_t reduce_init_highest<int64_t>() {
  return ::cuda::std::numeric_limits<int64_t>::max();
}

template <>
__device__ __forceinline__ uint32_t reduce_init_highest<uint32_t>() {
  return ::cuda::std::numeric_limits<uint32_t>::max();
}

template <>
__device__ __forceinline__ uint8_t reduce_init_highest<uint8_t>() {
  return ::cuda::std::numeric_limits<uint8_t>::max();
}

// TODO: Maybe add some fast_sum_f16_f32 variant that not only accumulate in f32
// but also expect a f32 output so that this can be used for normalization e.g.
// in softmax.

// Fast reduce sum kernel, this assumes that the dimensions to loop over are at
// the end, each block is responsible for populating one value in the output
// array. There are at most 1024 threads per block.
template <typename T>
__device__ void
fast_sum(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  shr[tid] = 0;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] += src[strided_i];
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] += shr[tid + s];
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

// Specialized vectorized fast_sum for bf16: 8 elements per float4 load
#if __CUDA_ARCH__ >= 800
__device__ void
fast_sum_bf16_vec(const size_t src_numel, const size_t el_to_sum_per_block,
                  const size_t num_dims, const size_t *info,
                  const __nv_bfloat16 *src, __nv_bfloat16 *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);

  float local_sum = 0.0f;

  if (is_contiguous(num_dims, dims, strides)) {
    // Round start up and stop down to float4-aligned boundaries
    const size_t vec_start = (start_idx + 7) / 8;
    const size_t vec_stop = stop_idx / 8;

    // Scalar head: elements before the first aligned float4
    size_t head_end = min(vec_start * 8, stop_idx);
    for (size_t i = start_idx + tid; i < head_end; i += blockDim.x) {
      local_sum += __bfloat162float(src[i]);
    }

    if (vec_start < vec_stop && is_aligned_16(src)) {
      const float4 *src4 = reinterpret_cast<const float4*>(src);
      for (size_t vi = vec_start + tid; vi < vec_stop; vi += blockDim.x) {
        float4 v = src4[vi];
        const __nv_bfloat16 *bp = reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; j++) {
          local_sum += __bfloat162float(bp[j]);
        }
      }
    } else {
      for (size_t i = head_end + tid; i < vec_stop * 8; i += blockDim.x) {
        local_sum += __bfloat162float(src[i]);
      }
    }

    // Scalar tail: elements after the last aligned float4
    size_t tail_start = vec_stop * 8;
    for (size_t i = tail_start + tid; i < stop_idx; i += blockDim.x) {
      local_sum += __bfloat162float(src[i]);
    }
  } else {
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
      size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
      local_sum += __bfloat162float(src[strided_i]);
      idx += blockDim.x;
    }
  }

  // Warp-level reduction
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);

  __shared__ float shr[32];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) shr[warp_id] = local_sum;
  __syncthreads();

  int num_warps = blockDim.x / WARP_SIZE;
  if (tid < WARP_SIZE) {
    local_sum = (tid < num_warps) ? shr[tid] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid == 0) dst[dst_id] = __float2bfloat16(local_sum);
  }
}
#endif

static __device__ __forceinline__ float2 warp_reduce_sum(float2 a) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        a.x += __shfl_xor_sync(0xffffffff, a.x, mask, 32);
        a.y += __shfl_xor_sync(0xffffffff, a.y, mask, 32);
    }
    return a;
}

static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, mask, 32);
    }
    return x;
}

// LayerNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L477
template <typename T>
__device__ void layernorm(const T * x, T * dst, const T * alpha, const T * beta, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float2 mean_var = make_float2(0.f, 0.f);

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = x[row*ncols + col];
        mean_var.x += xi;
        mean_var.y += xi * xi;
    }

    // sum up partial sums
    mean_var = warp_reduce_sum(mean_var);
    if (block_size > WARP_SIZE) {
        __shared__ float2 s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = mean_var;
        }
        __syncthreads();
        mean_var = s_sum[lane_id];
        mean_var = warp_reduce_sum(mean_var);
    }

    const float mean = mean_var.x / ncols;
    const float var = mean_var.y / ncols - mean * mean;
    const float inv_std = rsqrtf(var + eps);

    if (alpha == nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs);
      }
    }
    else if (alpha == nullptr && beta != nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float b = static_cast<float>(beta[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs + b);
      }
    }
    else if (alpha != nullptr && beta == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs * a);
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          float b = static_cast<float>(beta[col]);
          float lhs = (static_cast<float>(x[row*ncols + col]) - mean) * inv_std;
          dst[row*ncols + col] = static_cast<T>(lhs * a + b);
      }
    }
}

// RmsNorm implementation adapted from ggml, accumulation is made using f32.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L523
template <typename T>
__device__ void rmsnorm(const T * x, T * dst, const T * alpha, const int ncols, const int block_size, const float eps) {
    const int row = blockIdx.x*blockDim.y + threadIdx.y;
    const int tid = threadIdx.x;

    float tmp = 0.0f; // partial sum for thread in warp

    for (int col = tid; col < ncols; col += block_size) {
        const float xi = static_cast<float>(x[row*ncols + col]);
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __shared__ float s_sum[32];
        int warp_id = threadIdx.x / WARP_SIZE;
        int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / ncols;
    const float scale = rsqrtf(mean + eps);

    if (alpha == nullptr) {
      for (int col = tid; col < ncols; col += block_size) {
          dst[row*ncols + col] = static_cast<T>(scale * static_cast<float>(x[row*ncols + col]));
      }
    }
    else {
      for (int col = tid; col < ncols; col += block_size) {
          float a = static_cast<float>(alpha[col]);
          dst[row*ncols + col] = static_cast<T>(scale * static_cast<float>(x[row*ncols + col]) * a);
      }
    }
}

// Softmax implementation adapted from ggml.
// https://github.com/ggerganov/llama.cpp/blob/d59bd97065cd7ded6c4ecab54b1d5e0b1b11e318/ggml-cuda.cu#L4159
template <typename T, typename ACC>
__device__ void softmax(const T * x, T * dst, const int ncols) {
    const int row = blockDim.x*blockIdx.x + threadIdx.x;
    const int block_size = blockDim.y;
    const int tid = threadIdx.y;

    T max_val = -INFINITY;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        max_val = maxg(max_val, x[i]);
    }

    // find the max value in the block
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        max_val = maxg(max_val, __shfl_xor_sync(0xffffffff, max_val, mask, 32));
    }

    ACC tmp = 0.;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        const T val = expg(x[i] - max_val);
        tmp += static_cast<ACC>(val);
        dst[i] = val;
    }

    // sum up partial sums
#pragma unroll
    for (int mask = 16; mask > 0; mask >>= 1) {
        tmp += __shfl_xor_sync(0xffffffff, tmp, mask, 32);
    }

    const ACC inv_tmp = 1. / tmp;

    for (int col = tid; col < ncols; col += block_size) {
        const int i = row*ncols + col;
        dst[i] *= inv_tmp;
    }
}

template <typename T>
__device__ void ropei(const T * src, const T * cos, const T * sin, T * dst, const uint32_t bh, const uint32_t td, const uint32_t stride_b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t rope_idx = idx % (td / 2);
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      rope_idx += b_idx * (td / 2);
    }
    T c = cos[rope_idx];
    T s = sin[rope_idx];

    dst[2 * idx] = src[2 * idx] * c - src[2 * idx + 1] * s;
    dst[2 * idx + 1] = src[2 * idx] * s + src[2 * idx + 1] * c;
}

template <typename T>
__device__ void rope(const T * src, const T * cos, const T * sin, T * dst, const uint32_t bh, const uint32_t td, const uint32_t d, const uint32_t stride_b) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= bh * td) return;

    uint32_t i_bh = idx / (td / 2);
    uint32_t i_td = idx - (td / 2) * i_bh;
    uint32_t i_t = i_td / (d / 2);
    uint32_t i_d = i_td - (d / 2) * i_t;
    uint32_t i1 = i_bh * td + i_t * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * (td / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__device__ void rope_thd(
    const T * src,
    const T * cos,
    const T * sin,
    T * dst,
    const uint32_t b,
    const uint32_t t,
    const uint32_t h,
    const uint32_t d,
    const uint32_t stride_b
) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (2 * idx >= b * t * h * d) return;

    uint32_t i_bth = idx / (d / 2);
    uint32_t i_d = idx - (d / 2) * i_bth;
    uint32_t i_t = (i_bth / h) % t;
    uint32_t i1 = i_bth * d + i_d;
    uint32_t i2 = i1 + d / 2;
    uint32_t i_cs = i_t * (d / 2) + i_d;
    if (stride_b > 0) {
      uint32_t b_idx = (2 * idx) / stride_b;
      i_cs += b_idx * ((t * d) / 2);
    }
    T c = cos[i_cs];
    T s = sin[i_cs];

    dst[i1] = src[i1] * c - src[i2] * s;
    dst[i2] = src[i1] * s + src[i2] * c;
}

template <typename T>
__device__ void
fast_max(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // Initialize with the lowest representable value for T so that the first
  // comparison in the reduction always picks a real element.
  shr[tid] = reduce_init_lowest<T>();
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] = maxg(shr[tid], src[strided_i]);
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] = maxg(shr[tid], shr[tid + s]);
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

template <typename T>
__device__ void
fast_min(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // Initialize with the highest representable value for T so that the first
  // comparison in the reduction always picks a real element.
  shr[tid] = reduce_init_highest<T>();
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    shr[tid] = ming(shr[tid], src[strided_i]);
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s)
      shr[tid] = ming(shr[tid], shr[tid + s]);
  }

  if (tid == 0)
    dst[dst_id] = shr[0];
}

template <typename T>
__device__ void
fast_argmin(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, uint32_t *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  __shared__ uint32_t shr_index[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // For floating types this uses +inf; for integer types we use the largest
  // representable value instead of casting INFINITY to an integer.
  shr[tid] = reduce_init_highest<T>();
  shr_index[tid] = 0xFFFFFFFF;
  bool not_set = true;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    if (not_set || src[strided_i] < shr[tid]) {
      shr[tid] = src[strided_i];
      // Assume that the reduction takes place over the last dimension which is contiguous.
      shr_index[tid] = idx % dims[num_dims - 1];
      not_set = false;
    }
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s && shr[tid + s] < shr[tid]) {
      shr[tid] = shr[tid + s];
      shr_index[tid] = shr_index[tid + s];
    }
  }

  if (tid == 0)
    dst[dst_id] = shr_index[0];
}

template <typename T>
__device__ void
fast_argmax(const size_t src_numel, const size_t el_to_sum_per_block,
         const size_t num_dims, const size_t *info, const T *src, uint32_t *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;

  __shared__ T shr[BLOCK_SIZE];
  __shared__ uint32_t shr_index[BLOCK_SIZE];
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;

  // For floating types this uses -inf; for integer types we use the lowest
  // representable value instead of casting -INFINITY to an integer.
  shr[tid] = reduce_init_lowest<T>();
  shr_index[tid] = 0xFFFFFFFF;
  bool not_set = true;
  // Elements summed in this block range from dst_id * el_to_sum_per_block
  // to (dst_id + 1) * el_to_sum_per_block.
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  size_t idx = start_idx + tid;

  while (idx < stop_idx) {
    // TODO: Fast version for the contiguous case.
    size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
    if (not_set || src[strided_i] > shr[tid]) {
      shr[tid] = src[strided_i];
      // Assume that the reduction takes place over the last dimension which is contiguous.
      shr_index[tid] = idx % dims[num_dims - 1];
      not_set = false;
    }
    idx += blockDim.x;
  }

  // Parallel reduction, see the slides:
  // https://www.olcf.ornl.gov/wp-content/uploads/2019/12/05_Atomics_Reductions_Warp_Shuffle.pdf
  // https://stackoverflow.com/questions/66078814/is-cuda-atomicadd-operation-faster-than-launch-another-kernel-when-we-do-reduce
  for (int s = blockDim.x / 2; s > 0; s >>= 1) {
    __syncthreads();
    if (tid < s && shr[tid + s] > shr[tid]) {
      shr[tid] = shr[tid + s];
      shr_index[tid] = shr_index[tid + s];
    }
  }

  if (tid == 0)
    dst[dst_id] = shr_index[0];
}

#define FAST_OP(TYPENAME, MIN_NAME, MAX_NAME, ARGMIN_NAME, ARGMAX_NAME, SUM_NAME) \
  extern "C" __global__ void ARGMIN_NAME(                                      \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void ARGMAX_NAME(                                     \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      uint32_t *dst) {                                                         \
    fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst);     \
  }                                                                            \
  extern "C" __global__ void MIN_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void MAX_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }                                                                            \
  extern "C" __global__ void SUM_NAME(                                         \
      const size_t src_numel, const size_t el_to_sum_per_block,                \
      const size_t num_dims, const size_t *info, const TYPENAME *src,          \
      TYPENAME *dst) {                                                         \
    fast_sum(src_numel, el_to_sum_per_block, num_dims, info, src, dst);        \
  }

#define SUM_OP(TYPENAME, FN_NAME)                                              \
  extern "C" __global__ void FN_NAME(                                          \
      const size_t numel, const size_t num_dims, const size_t num_sum_dims,    \
      const size_t *info, const TYPENAME *inp, TYPENAME *out) {                \
    const size_t *dims = info;                                                 \
    const size_t *strides = info + num_dims;                                   \
    const size_t *sum_dims_l = info + 2 * num_dims;                            \
    const size_t *sum_dims_s = info + 2 * num_dims + num_sum_dims;             \
    if (is_contiguous(num_dims, dims, strides)) {                              \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[i]);                                    \
      }                                                                        \
    } else {                                                                   \
      for (unsigned int i = blockIdx.x * blockDim.x + threadIdx.x; i < numel;  \
           i += blockDim.x * gridDim.x) {                                      \
        unsigned strided_i = get_strided_index(i, num_dims, dims, strides);    \
        size_t dst_index = i;                                                  \
        for (unsigned int nd = 0; nd < num_sum_dims; ++nd) {                   \
          size_t stride = sum_dims_s[nd];                                      \
          size_t pre = dst_index / stride;                                     \
          size_t post = dst_index % stride;                                    \
          dst_index = (pre / sum_dims_l[nd]) * stride + post;                  \
        }                                                                      \
        atomicAdd(out + dst_index, inp[strided_i]);                            \
      }                                                                        \
    }                                                                          \
  }

#define SOFTMAX_OP(TYPENAME, ACC_TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst,                                      \
      const int n_cols) {                                                      \
    softmax<TYPENAME, ACC_TYPENAME>(src, dst, n_cols);                         \
  }                                                                            \

#define RMSNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const int n_cols, const int block_size, const float eps) {               \
    rmsnorm<TYPENAME>(src, dst, alpha, n_cols, block_size, eps);               \
  }                                                                            \

#define LAYERNORM_OP(TYPENAME, FN_NAME) \
  extern "C" __global__ void FN_NAME(                                          \
      const TYPENAME *src, TYPENAME *dst, const TYPENAME *alpha,               \
      const TYPENAME *beta, const int n_cols, const int block_size, const float eps) { \
    layernorm<TYPENAME>(src, dst, alpha, beta, n_cols, block_size, eps);       \
  }                                                                            \

#define ROPE_OP(TYPENAME, FN_NAME, FN_NAME_I, FN_NAME_THD) \
  extern "C" __global__ void FN_NAME_I( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t stride_b) { \
    ropei<TYPENAME>(src, cos, sin, dst, bh, td, stride_b); \
  } \
  extern "C" __global__ void FN_NAME( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t bh, \
      const uint32_t td, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope<TYPENAME>(src, cos, sin, dst, bh, td, d, stride_b); \
  } \
  extern "C" __global__ void FN_NAME_THD( \
      const TYPENAME *src, \
      const TYPENAME *cos, \
      const TYPENAME *sin, \
      TYPENAME *dst, \
      const uint32_t b, \
      const uint32_t t, \
      const uint32_t h, \
      const uint32_t d, \
      const uint32_t stride_b) { \
    rope_thd<TYPENAME>(src, cos, sin, dst, b, t, h, d, stride_b); \
  } \

// Small-reduce kernel: one thread per output element, for el_to_sum <= 32.
// Eliminates massive block-count overhead when reducing over tiny dimensions (e.g., MoE topk=8).
template <typename T>
__device__ void
fast_sum_small_impl(const size_t dst_el, const size_t el_to_sum_per_block,
                    const size_t num_dims, const size_t *info, const T *src, T *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= dst_el) return;

  T sum = 0;
  if (el_to_sum_per_block == 0) {
    dst[gid] = sum;
    return;
  }

  if (is_contiguous(num_dims, dims, strides)) {
    size_t start = gid * el_to_sum_per_block;
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      sum += src[start + i];
    }
  } else {
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      size_t strided_i = get_strided_index(gid * el_to_sum_per_block + i, num_dims, dims, strides);
      sum += src[strided_i];
    }
  }
  dst[gid] = sum;
}

#if __CUDA_ARCH__ >= 800
extern "C" __global__ void fast_sum_small_bf16(
    const size_t dst_el, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= dst_el) return;

  float sum = 0.0f;
  if (el_to_sum_per_block == 0) {
    dst[gid] = __float2bfloat16(sum);
    return;
  }

  if (is_contiguous(num_dims, dims, strides)) {
    size_t start = gid * el_to_sum_per_block;
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      sum += __bfloat162float(src[start + i]);
    }
  } else {
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      size_t strided_i = get_strided_index(gid * el_to_sum_per_block + i, num_dims, dims, strides);
      sum += __bfloat162float(src[strided_i]);
    }
  }

  dst[gid] = __float2bfloat16(sum);
}
#endif

extern "C" __global__ void fast_sum_small_f32(
    const size_t dst_el, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const float *src,
    float *dst) {
  fast_sum_small_impl(dst_el, el_to_sum_per_block, num_dims, info, src, dst);
}

extern "C" __global__ void fast_sum_small_f64(
    const size_t dst_el, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const double *src,
    double *dst) {
  fast_sum_small_impl(dst_el, el_to_sum_per_block, num_dims, info, src, dst);
}

extern "C" __global__ void fast_sum_small_u32(
    const size_t dst_el, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const uint32_t *src,
    uint32_t *dst) {
  fast_sum_small_impl(dst_el, el_to_sum_per_block, num_dims, info, src, dst);
}

extern "C" __global__ void fast_sum_small_i64(
    const size_t dst_el, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const int64_t *src,
    int64_t *dst) {
  fast_sum_small_impl(dst_el, el_to_sum_per_block, num_dims, info, src, dst);
}

extern "C" __global__ void fast_sum_small_u8(
    const size_t dst_el, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const uint8_t *src,
    uint8_t *dst) {
  fast_sum_small_impl(dst_el, el_to_sum_per_block, num_dims, info, src, dst);
}

#if __CUDA_ARCH__ >= 530
extern "C" __global__ void fast_sum_small_f16(
    const size_t dst_el, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __half *src,
    __half *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  size_t gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= dst_el) return;

  float sum = 0.0f;
  if (el_to_sum_per_block == 0) {
    dst[gid] = __float2half(sum);
    return;
  }

  if (is_contiguous(num_dims, dims, strides)) {
    size_t start = gid * el_to_sum_per_block;
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      sum += __half2float(src[start + i]);
    }
  } else {
    for (size_t i = 0; i < el_to_sum_per_block; ++i) {
      size_t strided_i = get_strided_index(gid * el_to_sum_per_block + i, num_dims, dims, strides);
      sum += __half2float(src[strided_i]);
    }
  }
  dst[gid] = __float2half(sum);
}
#endif

#if __CUDA_ARCH__ >= 800
SOFTMAX_OP(__nv_bfloat16, float, softmax_bf16)
RMSNORM_OP(__nv_bfloat16, rmsnorm_bf16)
LAYERNORM_OP(__nv_bfloat16, layernorm_bf16)
ROPE_OP(__nv_bfloat16, rope_bf16, rope_i_bf16, rope_thd_bf16)
SUM_OP(__nv_bfloat16, sum_bf16)

// Use vectorized fast_sum for bf16, original for other ops
extern "C" __global__ void fast_sum_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  fast_sum_bf16_vec(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_min_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_max_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    __nv_bfloat16 *dst) {
  fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_argmin_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    uint32_t *dst) {
  fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}
extern "C" __global__ void fast_argmax_bf16(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const __nv_bfloat16 *src,
    uint32_t *dst) {
  fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst);
}

// NOTE: No reduce ops for f8
// SUM_OP(__nv_fp8_e4m3, sum_fp8_e4m3)
// SOFTMAX_OP(__nv_fp8_e4m3, float, softmax_fp8_e4m3)
// RMSNORM_OP(__nv_fp8_e4m3, rmsnorm_fp8_e4m3)
// LAYERNORM_OP(__nv_fp8_e4m3, layernorm_fp8_e4m3)
// ROPE_OP(__nv_fp8_e4m3, rope_fp8_e4m3, rope_i_fp8_e4m3, rope_thd_fp8_e4m3)
// FAST_OP(__nv_fp8_e4m3, fast_min_fp8_e4m3, fast_max_fp8_e4m3, fast_argmin_fp8_e4m3, fast_argmax_fp8_e4m3, fast_sum_fp8_e4m3)
#endif

#if __CUDA_ARCH__ >= 530
SOFTMAX_OP(__half, float, softmax_f16)
RMSNORM_OP(__half, rmsnorm_f16)
LAYERNORM_OP(__half, layernorm_f16)
ROPE_OP(__half, rope_f16, rope_i_f16, rope_thd_f16)
SUM_OP(__half, sum_f16)
FAST_OP(__half, fast_min_f16, fast_max_f16, fast_argmin_f16, fast_argmax_f16, fast_sum_f16)
#endif

SUM_OP(float, sum_f32)
SUM_OP(double, sum_f64)
SUM_OP(uint32_t, sum_u32)
SOFTMAX_OP(float, float, softmax_f32)
SOFTMAX_OP(double, double, softmax_f64)
RMSNORM_OP(float, rmsnorm_f32)
RMSNORM_OP(double, rmsnorm_f64)

static __device__ __forceinline__ float lfm2_moe_plain_dot_fold(
    const float (&accum)[4], unsigned int mask
) {
    const unsigned int group = 0xffu << ((threadIdx.y % 4) * 8);
    float sum = __fadd_rn(__fadd_rn(accum[0], accum[2]), __fadd_rn(accum[1], accum[3]));
    sum = __fadd_rn(sum, __shfl_xor_sync(mask & group, sum, 4, 8));
    sum = __fadd_rn(sum, __shfl_xor_sync(mask & group, sum, 1, 8));
    return __fadd_rn(sum, __shfl_xor_sync(mask & group, sum, 2, 8));
}

static __device__ __forceinline__ float lfm2_moe_plain_dot(
    const float *left, const float *right, unsigned int width, unsigned int mask
) {
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int col = threadIdx.x; col < width; col += 32) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const unsigned int index = col + i * 8;
            accum[i] = __fmaf_rn(left[index], right[index], accum[i]);
        }
    }
    return lfm2_moe_plain_dot_fold(accum, mask);
}

extern "C" __global__ void lfm2_moe_router_f32(
    const float *input, const float *weights, float *output,
    unsigned int cells, unsigned int experts, unsigned int width
) {
    const unsigned int cell = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int mask = __ballot_sync(0xffffffff, cell < cells);
    if (cell >= cells) return;

    const float *x = input + size_t(cell / experts) * width;
    const float *w = weights + size_t(cell % experts) * width;
    const unsigned int lane = threadIdx.x;
    float sum = 0.0f;
    for (unsigned int i = lane; i < width; i += 8) {
        sum = __fmaf_rn(w[i], x[i], sum);
    }

    // The CPU router folds eight FMA lanes in this order.
    sum = __fadd_rn(sum, __shfl_xor_sync(mask, sum, 4, 8));
    sum = __fadd_rn(sum, __shfl_xor_sync(mask, sum, 2, 8));
    sum = __fadd_rn(sum, __shfl_xor_sync(mask, sum, 1, 8));
    if (lane == 0) output[cell] = sum;
}

extern "C" __global__ void lfm2_moe_router_decode_f32(
    const float *input, const float *weights, float *output,
    unsigned int cells, unsigned int experts, unsigned int width
) {
    const unsigned int cell = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int mask = __ballot_sync(0xffffffff, cell < cells);
    if (cell >= cells) return;
    const float *x = input + size_t(cell / experts) * width;
    const float *w = weights + size_t(cell % experts) * width;
    const float sum = lfm2_moe_plain_dot(w, x, width, mask);
    if (threadIdx.x == 0) output[cell] = sum;
}

extern "C" __global__ void lfm2_moe_rope_f32(
    const float *input, const float *cos, const float *sin, float *output,
    unsigned int pairs, unsigned int tokens, unsigned int half
) {
    const unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= pairs) return;

    const unsigned int row = i / half;
    const unsigned int lane = i % half;
    const unsigned int coeff = (row % tokens) * half + lane;
    const unsigned int first = row * (2 * half) + lane;
    const unsigned int second = first + half;
    const float a = input[first];
    const float b = input[second];
    const float c = cos[coeff];
    const float s = sin[coeff];
    output[first] = __fsub_rn(__fmul_rn(a, c), __fmul_rn(b, s));
    output[second] = __fadd_rn(__fmul_rn(a, s), __fmul_rn(b, c));
}

extern "C" __global__ void lfm2_moe_attention_scores_f32(
    const float *q, const float *k, const uint8_t *mask, float *scores,
    unsigned int cells, unsigned int queries, unsigned int keys,
    unsigned int width, unsigned int heads_per_kv, unsigned int masked
) {
    const unsigned int cell = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int active = __ballot_sync(0xffffffff, cell < cells);
    if (cell >= cells) return;

    const unsigned int row = cell / keys;
    const unsigned int key = cell % keys;
    const unsigned int lane = threadIdx.x;
    if (masked && mask[size_t(row % queries) * keys + key] != 0) {
        if (lane == 0) scores[cell] = -INFINITY;
        return;
    }
    const unsigned int kv_head = (row / queries) / heads_per_kv;
    const float *query = q + size_t(row) * width;
    const float *key_row = k + (size_t(kv_head) * keys + key) * width;
    const float sum = lfm2_moe_plain_dot(query, key_row, width, active);
    if (lane == 0) scores[cell] = sum;
}

// Port of the MIT-licensed ggml exponent and Arm polynomial recorded in
// quantized_lfm2_moe/cpu_attention.rs. Keep the exceptional exponent split.
static __device__ __forceinline__ float lfm2_moe_attention_exp(float x) {
    const float shift = __uint_as_float(0x4b400000);
    const float z = __fmaf_rn(x, __uint_as_float(0x3fb8aa3b), shift);
    const float n = __fsub_rn(z, shift);
    const float b = __fmaf_rn(-n, __uint_as_float(0x35bfbe8e),
                              __fmaf_rn(-n, __uint_as_float(0x3f317200), x));
    const float u = __fmul_rn(b, b);
    const float high = __fmaf_rn(__uint_as_float(0x3c072010), b, __uint_as_float(0x3d2b9f17));
    const float low = __fmaf_rn(__uint_as_float(0x3e2aaf33), b, __uint_as_float(0x3efffedb));
    const float j = __fmaf_rn(__fmaf_rn(high, u, low), u, __fmul_rn(__uint_as_float(0x3f7ffff6), b));
    const unsigned int e = __float_as_uint(z) << 23;
    if (fabsf(n) > 126.0f) {
        const unsigned int g = n <= 0.0f ? 0x82000000u : 0;
        const float s1 = __uint_as_float(g + 0x7f000000u);
        const float s2 = __uint_as_float(e - g);
        return fabsf(n) > 192.0f ? __fmul_rn(s1, s1) : __fmul_rn(__fmaf_rn(s2, j, s2), s1);
    }
    const float scale = __uint_as_float(e + 0x3f800000u);
    return __fmaf_rn(j, scale, scale);
}

extern "C" __global__ void lfm2_moe_attention_softmax_f32(
    float *scores, unsigned int keys, float scale
) {
    float *row = scores + size_t(blockIdx.x) * keys;
    const unsigned int lane = threadIdx.x;
    __shared__ float maxima[128];
    __shared__ float inverse;

    float max_score = -INFINITY;
    for (unsigned int key = lane; key < keys; key += blockDim.x) {
        const float score = __fadd_rn(__fmul_rn(row[key], scale), 0.0f);
        row[key] = score;
        max_score = fmaxf(max_score, score);
    }
    maxima[lane] = max_score;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (lane < stride) maxima[lane] = fmaxf(maxima[lane], maxima[lane + stride]);
        __syncthreads();
    }
    for (unsigned int key = lane; key < keys; key += blockDim.x) {
        row[key] = lfm2_moe_attention_exp(__fsub_rn(row[key], maxima[0]));
    }
    __syncthreads();
    if (lane == 0) {
        double sum = 0.0;
        // Four pair sums and two cross-lane adds round before the F64 row fold.
        for (unsigned int key = 0; key < keys; key += 8) {
            float pair[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                const float a = key + i < keys ? row[key + i] : 0.0f;
                const float b = key + i + 4 < keys ? row[key + i + 4] : 0.0f;
                pair[i] = __fadd_rn(a, b);
            }
            const float fold = __fadd_rn(__fadd_rn(pair[0], pair[2]), __fadd_rn(pair[1], pair[3]));
            sum = __dadd_rn(sum, double(fold));
        }
        inverse = __double2float_rn(__ddiv_rn(1.0, sum));
    }
    __syncthreads();
    for (unsigned int key = lane; key < keys; key += blockDim.x) {
        row[key] = __fmul_rn(row[key], inverse);
    }
}

extern "C" __global__ void lfm2_moe_attention_values_f32(
    const float *probabilities, const float *v, float *output,
    unsigned int elements, unsigned int queries, unsigned int keys,
    unsigned int width, unsigned int heads_per_kv
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= elements) return;
    const unsigned int row = index / width;
    const unsigned int kv_head = (row / queries) / heads_per_kv;
    const float *prob = probabilities + size_t(row) * keys;
    const float *val = v + size_t(kv_head) * keys * width + index % width;
    float accum[8] = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
    const unsigned int full = keys - keys % 8;
    for (unsigned int key = 0; key < full; key += 8) {
#pragma unroll
        for (int lane = 0; lane < 8; ++lane) {
            const unsigned int at = key + lane;
            accum[lane] = __fmaf_rn(prob[at], val[size_t(at) * width], accum[lane]);
        }
    }
    for (unsigned int key = full; key < keys; ++key) {
        accum[key % 8] = __fmaf_rn(prob[key], val[size_t(key) * width], accum[key % 8]);
    }
    // Pinned llamafile sees a zero-padded 256-key block, including signed-zero effects.
    const unsigned int padding = min((256u - keys % 256u) % 256u, 8u);
    for (unsigned int lane = 0; lane < padding; ++lane) {
        const unsigned int i = (keys + lane) % 8;
        accum[i] = __fadd_rn(accum[i], 0.0f);
    }
    const float a = __fadd_rn(accum[4], accum[0]);
    const float b = __fadd_rn(accum[5], accum[1]);
    const float c = __fadd_rn(accum[6], accum[2]);
    const float d = __fadd_rn(accum[7], accum[3]);
    output[index] = __fadd_rn(__fadd_rn(a, c), __fadd_rn(b, d));
}

extern "C" __global__ void lfm2_moe_attention_decode_values_f32(
    const float *probabilities, const float *v, float *output,
    unsigned int elements, unsigned int queries, unsigned int keys,
    unsigned int width, unsigned int heads_per_kv
) {
    const unsigned int index = blockIdx.x * blockDim.y + threadIdx.y;
    const unsigned int active = __ballot_sync(0xffffffff, index < elements);
    if (index >= elements) return;
    const unsigned int row = index / width;
    const unsigned int lane = threadIdx.x;
    const unsigned int kv_head = (row / queries) / heads_per_kv;
    const float *prob = probabilities + size_t(row) * keys;
    const float *val = v + size_t(kv_head) * keys * width + index % width;
    float accum[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (unsigned int base = lane; base < keys; base += 32) {
#pragma unroll
        for (unsigned int group = 0; group < 4; ++group) {
            const unsigned int key = base + group * 8;
            if (key < keys) accum[group] = __fmaf_rn(prob[key], val[size_t(key) * width], accum[group]);
        }
    }
    const unsigned int padding = min((256u - keys % 256u) % 256u, 32u);
#pragma unroll
    for (unsigned int group = 0; group < 4; ++group) {
        const unsigned int slot = group * 8 + lane;
        if ((slot + 32 - keys % 32) % 32 < padding) accum[group] = __fadd_rn(accum[group], 0.0f);
    }
    // ggml_vec_dot_f32 combines the vector halves, then adds adjacent pairs.
    const float sum = lfm2_moe_plain_dot_fold(accum, active);
    if (lane == 0) output[index] = sum;
}

// Port of the MIT-licensed ggml SiLU schedule recorded in quantized_lfm2_moe/cpu.rs.
extern "C" __global__ void lfm2_moe_silu_f32(
    const float *input, float *output, unsigned int count
) {
    const unsigned int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;

    const float x = input[index];
    const float negative = __fsub_rn(0.0f, x);
    const float shift = __uint_as_float(0x4b400000);
    const float reduced = __fmaf_rn(negative, __uint_as_float(0x3fb8aa3b), shift);
    const float n = __fsub_rn(reduced, shift);
    if (!isfinite(n) || fabsf(n) > 126.0f) {
        output[index] = __fdiv_rn(x, __fadd_rn(1.0f, expf(negative)));
        return;
    }
    const float high = __fmaf_rn(-n, __uint_as_float(0x3f317200), negative);
    const float b = __fmaf_rn(-n, __uint_as_float(0x35bfbe8e), high);
    const float u = __fmul_rn(b, b);
    const float lower = __fmaf_rn(__uint_as_float(0x3e2aaf33), b, __uint_as_float(0x3efffedb));
    const float upper = __fmaf_rn(__uint_as_float(0x3c072010), b, __uint_as_float(0x3d2b9f17));
    const float j = __fmaf_rn(
        __fmaf_rn(upper, u, lower), u, __fmul_rn(__uint_as_float(0x3f7ffff6), b)
    );
    const float scale = __uint_as_float((__float_as_uint(reduced) << 23) + 0x3f800000U);
    output[index] = __fdiv_rn(x, __fadd_rn(1.0f, __fmaf_rn(j, scale, scale)));
}

extern "C" __global__ void rmsnorm_f32_ggml_cpu_order(
        const float * input, float * output, const float * weight, unsigned int width, float eps) {
    const size_t offset = size_t(blockIdx.x) * width;
    __shared__ float reciprocal;
    if (threadIdx.x == 0) {
        // Keep the CPU fold order and round each square to F32 before accumulating.
        double sum = 0.0;
        for (unsigned int col = 0; col < width; ++col) {
            const float value = input[offset + col];
            sum = __dadd_rn(sum, double(__fmul_rn(value, value)));
        }
        const float mean = __double2float_rn(__ddiv_rn(sum, double(width)));
        reciprocal = __fdiv_rn(1.0f, __fsqrt_rn(__fadd_rn(mean, eps)));
    }
    __syncthreads();
    for (size_t col = threadIdx.x; col < width; col += blockDim.x) {
        const float normed = __fmul_rn(input[offset + col], reciprocal);
        output[offset + col] = __fmul_rn(normed, weight[col]);
    }
}
LAYERNORM_OP(float, layernorm_f32)
LAYERNORM_OP(double, layernorm_f64)
ROPE_OP(float, rope_f32, rope_i_f32, rope_thd_f32)
ROPE_OP(double, rope_f64, rope_i_f64, rope_thd_f64)

// Vectorized fast_sum for f32: 4 elements per float4 load
extern "C" __global__ void fast_sum_f32(
    const size_t src_numel, const size_t el_to_sum_per_block,
    const size_t num_dims, const size_t *info, const float *src, float *dst) {
  const size_t *dims = info;
  const size_t *strides = info + num_dims;
  size_t tid = threadIdx.x;
  size_t dst_id = blockIdx.x;
  size_t start_idx = dst_id * el_to_sum_per_block;
  size_t stop_idx = min(start_idx + el_to_sum_per_block, src_numel);
  float local_sum = 0.0f;
  if (is_contiguous(num_dims, dims, strides)) {
    const size_t vec_start = (start_idx + 3) / 4;
    const size_t vec_stop = stop_idx / 4;

    // Scalar head
    size_t head_end = min(vec_start * 4, stop_idx);
    for (size_t i = start_idx + tid; i < head_end; i += blockDim.x)
      local_sum += src[i];

    if (vec_start < vec_stop && is_aligned_16(src)) {
      const float4 *src4 = reinterpret_cast<const float4*>(src);
      for (size_t vi = vec_start + tid; vi < vec_stop; vi += blockDim.x) {
        float4 v = src4[vi];
        local_sum += v.x + v.y + v.z + v.w;
      }
    } else {
      for (size_t i = head_end + tid; i < vec_stop * 4; i += blockDim.x)
        local_sum += src[i];
    }

    // Scalar tail
    size_t tail_start = vec_stop * 4;
    for (size_t i = tail_start + tid; i < stop_idx; i += blockDim.x)
      local_sum += src[i];
  } else {
    size_t idx = start_idx + tid;
    while (idx < stop_idx) {
      size_t strided_i = get_strided_index(idx, num_dims, dims, strides);
      local_sum += src[strided_i];
      idx += blockDim.x;
    }
  }
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
    local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
  __shared__ float shr[32];
  int warp_id = tid / WARP_SIZE;
  int lane_id = tid % WARP_SIZE;
  if (lane_id == 0) shr[warp_id] = local_sum;
  __syncthreads();
  int num_warps = blockDim.x / WARP_SIZE;
  if (tid < WARP_SIZE) {
    local_sum = (tid < num_warps) ? shr[tid] : 0.0f;
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
      local_sum += __shfl_xor_sync(0xffffffff, local_sum, offset);
    if (tid == 0) dst[dst_id] = local_sum;
  }
}
extern "C" __global__ void fast_min_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, float *dst) { fast_min(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
extern "C" __global__ void fast_max_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, float *dst) { fast_max(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
extern "C" __global__ void fast_argmin_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, uint32_t *dst) { fast_argmin(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
extern "C" __global__ void fast_argmax_f32(const size_t src_numel, const size_t el_to_sum_per_block, const size_t num_dims, const size_t *info, const float *src, uint32_t *dst) { fast_argmax(src_numel, el_to_sum_per_block, num_dims, info, src, dst); }
FAST_OP(double, fast_min_f64, fast_max_f64, fast_argmin_f64, fast_argmax_f64, fast_sum_f64)
FAST_OP(uint32_t, fast_min_u32, fast_max_u32, fast_argmin_u32, fast_argmax_u32, fast_sum_u32)
FAST_OP(int64_t, fast_min_i64, fast_max_i64, fast_argmin_i64, fast_argmax_i64, fast_sum_i64)
FAST_OP(uint8_t, fast_min_u8, fast_max_u8, fast_argmin_u8, fast_argmax_u8, fast_sum_u8)
