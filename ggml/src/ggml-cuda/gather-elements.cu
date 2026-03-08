#include "gather-elements.cuh"

#include <cstdint>

template <typename TData, typename TIndex>
static __global__ void k_gather_elements(
        const TData * __restrict__ data,
        const TIndex * __restrict__ indices,
        TData * __restrict__ dst,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2,
        const int64_t ne3,
        const int64_t data_s0,
        const int64_t data_s1,
        const int64_t data_s2,
        const int64_t data_s3,
        const int64_t idx_s0,
        const int64_t idx_s1,
        const int64_t idx_s2,
        const int64_t idx_s3,
        const int64_t dst_s0,
        const int64_t dst_s1,
        const int64_t dst_s2,
        const int64_t dst_s3,
        const int32_t axis,
        const int64_t axis_size) {
    const int64_t idx = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t ne = ne0 * ne1 * ne2 * ne3;
    if (idx >= ne) {
        return;
    }

    int64_t t = idx;
    const int64_t i0 = t % ne0; t /= ne0;
    const int64_t i1 = t % ne1; t /= ne1;
    const int64_t i2 = t % ne2; t /= ne2;
    const int64_t i3 = t;

    int64_t gather_index = (int64_t) indices[i0*idx_s0 + i1*idx_s1 + i2*idx_s2 + i3*idx_s3];
    if (gather_index < 0) {
        gather_index += axis_size;
    }

    int64_t d0 = i0;
    int64_t d1 = i1;
    int64_t d2 = i2;
    int64_t d3 = i3;
    switch (axis) {
        case 0: d0 = gather_index; break;
        case 1: d1 = gather_index; break;
        case 2: d2 = gather_index; break;
        case 3: d3 = gather_index; break;
        default: return;
    }

    dst[i0*dst_s0 + i1*dst_s1 + i2*dst_s2 + i3*dst_s3] =
        data[d0*data_s0 + d1*data_s1 + d2*data_s2 + d3*data_s3];
}

template <typename TData, typename TIndex>
static void ggml_cuda_op_gather_elements_impl(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * data = dst->src[0];
    const ggml_tensor * indices = dst->src[1];

    const int32_t axis = ggml_get_op_params_i32(dst, 0);
    GGML_ASSERT(axis >= 0 && axis < GGML_MAX_DIMS);

    GGML_ASSERT(data->nb[0] % sizeof(TData) == 0);
    GGML_ASSERT(data->nb[1] % sizeof(TData) == 0);
    GGML_ASSERT(data->nb[2] % sizeof(TData) == 0);
    GGML_ASSERT(data->nb[3] % sizeof(TData) == 0);

    GGML_ASSERT(indices->nb[0] % sizeof(TIndex) == 0);
    GGML_ASSERT(indices->nb[1] % sizeof(TIndex) == 0);
    GGML_ASSERT(indices->nb[2] % sizeof(TIndex) == 0);
    GGML_ASSERT(indices->nb[3] % sizeof(TIndex) == 0);

    GGML_ASSERT(dst->nb[0] % sizeof(TData) == 0);
    GGML_ASSERT(dst->nb[1] % sizeof(TData) == 0);
    GGML_ASSERT(dst->nb[2] % sizeof(TData) == 0);
    GGML_ASSERT(dst->nb[3] % sizeof(TData) == 0);

    const int64_t ne = ggml_nelements(dst);

    const int64_t data_s0 = data->nb[0] / (int64_t) sizeof(TData);
    const int64_t data_s1 = data->nb[1] / (int64_t) sizeof(TData);
    const int64_t data_s2 = data->nb[2] / (int64_t) sizeof(TData);
    const int64_t data_s3 = data->nb[3] / (int64_t) sizeof(TData);

    const int64_t idx_s0 = indices->nb[0] / (int64_t) sizeof(TIndex);
    const int64_t idx_s1 = indices->nb[1] / (int64_t) sizeof(TIndex);
    const int64_t idx_s2 = indices->nb[2] / (int64_t) sizeof(TIndex);
    const int64_t idx_s3 = indices->nb[3] / (int64_t) sizeof(TIndex);

    const int64_t dst_s0 = dst->nb[0] / (int64_t) sizeof(TData);
    const int64_t dst_s1 = dst->nb[1] / (int64_t) sizeof(TData);
    const int64_t dst_s2 = dst->nb[2] / (int64_t) sizeof(TData);
    const int64_t dst_s3 = dst->nb[3] / (int64_t) sizeof(TData);

    const int block_size = 256;
    const int blocks = (int) ((ne + block_size - 1) / block_size);

    k_gather_elements<TData, TIndex><<<blocks, block_size, 0, ctx.stream()>>>(
        (const TData *) data->data,
        (const TIndex *) indices->data,
        (TData *) dst->data,
        dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3],
        data_s0, data_s1, data_s2, data_s3,
        idx_s0, idx_s1, idx_s2, idx_s3,
        dst_s0, dst_s1, dst_s2, dst_s3,
        axis, data->ne[axis]);
}

template <typename TData>
static void ggml_cuda_op_gather_elements_dispatch_index(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * indices = dst->src[1];

    switch (indices->type) {
        case GGML_TYPE_I32:
            ggml_cuda_op_gather_elements_impl<TData, int32_t>(ctx, dst);
            break;
        case GGML_TYPE_I64:
            ggml_cuda_op_gather_elements_impl<TData, int64_t>(ctx, dst);
            break;
        default:
            GGML_ABORT("%s: unsupported indices type for gather_elements: %s",
                    __func__, ggml_type_name(indices->type));
    }
}

void ggml_cuda_op_gather_elements(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * data = dst->src[0];
    const ggml_tensor * indices = dst->src[1];

    GGML_ASSERT(data != nullptr && indices != nullptr);
    GGML_ASSERT(data->type == dst->type);
    GGML_ASSERT(ggml_are_same_shape(dst, indices));

    switch (data->type) {
        case GGML_TYPE_F32:
            ggml_cuda_op_gather_elements_dispatch_index<float>(ctx, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_op_gather_elements_dispatch_index<half>(ctx, dst);
            break;
        default:
            GGML_ABORT("%s: unsupported data type for gather_elements: %s",
                    __func__, ggml_type_name(data->type));
    }
}
