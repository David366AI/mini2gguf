#include "reduce-max.cuh"

#include <algorithm>
#include <cfloat>
#include <cstdint>

template <typename T>
static __device__ __forceinline__ float reduce_max_to_float(T v);

template <>
__device__ __forceinline__ float reduce_max_to_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ float reduce_max_to_float<half>(half v) {
    return __half2float(v);
}

template <typename T>
static __device__ __forceinline__ T reduce_max_from_float(float v);

template <>
__device__ __forceinline__ float reduce_max_from_float<float>(float v) {
    return v;
}

template <>
__device__ __forceinline__ half reduce_max_from_float<half>(float v) {
    return __float2half(v);
}

template <typename T>
static __global__ void reduce_max_axis_kernel(
        const T * __restrict__ src,
        T * __restrict__ dst,
        const int64_t src_nb0,
        const int64_t src_nb1,
        const int64_t src_nb2,
        const int64_t src_nb3,
        const int64_t dst_ne0,
        const int64_t dst_ne1,
        const int64_t dst_ne2,
        const int64_t dst_ne3,
        const int64_t dst_nb0,
        const int64_t dst_nb1,
        const int64_t dst_nb2,
        const int64_t dst_nb3,
        const int32_t axis,
        const int64_t reduce_ne) {
    const int64_t out_idx = blockIdx.x;
    const int64_t out_total = dst_ne0 * dst_ne1 * dst_ne2 * dst_ne3;
    if (out_idx >= out_total) {
        return;
    }

    int64_t rem = out_idx;
    const int64_t i0 = rem % dst_ne0;
    rem /= dst_ne0;
    const int64_t i1 = rem % dst_ne1;
    rem /= dst_ne1;
    const int64_t i2 = rem % dst_ne2;
    rem /= dst_ne2;
    const int64_t i3 = rem % dst_ne3;

    const int64_t src_axis_stride =
        axis == 0 ? src_nb0 :
        axis == 1 ? src_nb1 :
        axis == 2 ? src_nb2 : src_nb3;

    const int64_t src_base =
        i0 * src_nb0 +
        i1 * src_nb1 +
        i2 * src_nb2 +
        i3 * src_nb3;

    float maxval = -FLT_MAX;
    for (int64_t r = threadIdx.x; r < reduce_ne; r += blockDim.x) {
        maxval = fmaxf(maxval, reduce_max_to_float<T>(src[src_base + r * src_axis_stride]));
    }

    for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
        maxval = fmaxf(maxval, __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE));
    }

    const int n_warps = blockDim.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    const int warp_id = threadIdx.x / WARP_SIZE;

    if (n_warps > 1) {
        constexpr int max_warps = 1024 / WARP_SIZE;
        __shared__ float shared_max[max_warps];

        if (lane_id == 0) {
            shared_max[warp_id] = maxval;
        }

        __syncthreads();

        if (warp_id == 0) {
            maxval = lane_id < n_warps ? shared_max[lane_id] : -FLT_MAX;
            for (int offset = WARP_SIZE/2; offset > 0; offset >>= 1) {
                maxval = fmaxf(maxval, __shfl_xor_sync(0xFFFFFFFF, maxval, offset, WARP_SIZE));
            }
        }
    }

    if (warp_id == 0 && lane_id == 0) {
        const int64_t dst_index =
            i0 * dst_nb0 +
            i1 * dst_nb1 +
            i2 * dst_nb2 +
            i3 * dst_nb3;
        dst[dst_index] = reduce_max_from_float<T>(maxval);
    }
}

template <typename T>
static void ggml_cuda_op_reduce_max_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    const int32_t axis = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(axis >= 0 && axis < GGML_MAX_DIMS);

    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        if (d == axis) {
            GGML_ASSERT(dst->ne[d] == 1);
        } else {
            GGML_ASSERT(dst->ne[d] == src0->ne[d]);
        }
    }

    const int64_t reduce_ne = src0->ne[axis];
    const int64_t out_total = ggml_nelements(dst);

    const int64_t src_nb0 = src0->nb[0] / (int64_t)sizeof(T);
    const int64_t src_nb1 = src0->nb[1] / (int64_t)sizeof(T);
    const int64_t src_nb2 = src0->nb[2] / (int64_t)sizeof(T);
    const int64_t src_nb3 = src0->nb[3] / (int64_t)sizeof(T);

    const int64_t dst_nb0 = dst->nb[0] / (int64_t)sizeof(T);
    const int64_t dst_nb1 = dst->nb[1] / (int64_t)sizeof(T);
    const int64_t dst_nb2 = dst->nb[2] / (int64_t)sizeof(T);
    const int64_t dst_nb3 = dst->nb[3] / (int64_t)sizeof(T);

    const T * src_d = (const T *) src0->data;
    T * dst_d = (T *) dst->data;

    const int64_t num_threads = std::min<int64_t>(1024, ((reduce_ne + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE);
    const dim3 blocks_num((uint32_t) out_total, 1, 1);
    const dim3 blocks_dim(num_threads, 1, 1);

    reduce_max_axis_kernel<T><<<blocks_num, blocks_dim, 0, ctx.stream()>>>(
        src_d, dst_d,
        src_nb0, src_nb1, src_nb2, src_nb3,
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        dst_nb0, dst_nb1, dst_nb2, dst_nb3,
        axis, reduce_ne);
}

void ggml_cuda_op_reduce_max(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == dst->type);

    switch (src0->type) {
        case GGML_TYPE_F32:
            ggml_cuda_op_reduce_max_impl<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_op_reduce_max_impl<half>(ctx, dst);
            break;
        default:
            GGML_ABORT("%s: unsupported type: %s", __func__, ggml_type_name(src0->type));
    }
}
