#include "gru.cuh"

#include <cstdint>

static __device__ __forceinline__ float ggml_cuda_gru_sigmoid(float x) {
    return 1.0f / (1.0f + expf(-x));
}

static __device__ __forceinline__ float ggml_cuda_gru_load_typed(const void * base, int type, int64_t idx) {
    switch (type) {
        case GGML_TYPE_F32:
            return ((const float *) base)[idx];
        case GGML_TYPE_F16:
            return __half2float(((const half *) base)[idx]);
        default:
            return 0.0f;
    }
}

static __global__ void ggml_cuda_gru_kernel(
        const void * __restrict__ x,
        const void * __restrict__ w,
        const void * __restrict__ r,
        const void * __restrict__ b,
        const void * __restrict__ h0,
        float * __restrict__ y,
        const int x_type,
        const int w_type,
        const int r_type,
        const int b_type,
        const int h0_type,
        const int64_t x_nb0,
        const int64_t x_nb1,
        const int64_t x_nb2,
        const int64_t w_nb0,
        const int64_t w_nb1,
        const int64_t w_nb2,
        const int64_t r_nb0,
        const int64_t r_nb1,
        const int64_t r_nb2,
        const int64_t b_nb0,
        const int64_t b_nb1,
        const int64_t h0_nb0,
        const int64_t h0_nb1,
        const int64_t h0_nb2,
        const int64_t y_nb0,
        const int64_t y_nb1,
        const int64_t y_nb2,
        const int64_t y_nb3,
        const int32_t hidden,
        const int32_t input_size,
        const int32_t seq_len,
        const int32_t batch,
        const int32_t num_dir,
        const int32_t linear_before_reset,
        const int32_t direction,
        const int32_t output_last,
        const int32_t has_bias,
        const int32_t has_h0) {
    if (blockIdx.x >= (uint32_t) batch || blockIdx.y >= (uint32_t) num_dir || threadIdx.x != 0) {
        return;
    }

    extern __shared__ float shared[];
    float * h_prev = shared;
    float * h_cur  = h_prev + hidden;
    float * z_gate = h_cur + hidden;
    float * r_gate = z_gate + hidden;

    const int64_t b_idx = (int64_t) blockIdx.x;
    const int64_t d = (int64_t) blockIdx.y;

    const bool reverse_dir =
        direction == GGML_GRU_REVERSE ||
        (direction == GGML_GRU_BIDIRECTIONAL && d == 1);

    if (has_h0) {
        for (int64_t h = 0; h < hidden; ++h) {
            h_prev[h] = ggml_cuda_gru_load_typed(h0, h0_type, h * h0_nb0 + b_idx * h0_nb1 + d * h0_nb2);
        }
    } else {
        for (int64_t h = 0; h < hidden; ++h) {
            h_prev[h] = 0.0f;
        }
    }

    for (int64_t step = 0; step < seq_len; ++step) {
        const int64_t t_in = reverse_dir ? (seq_len - 1 - step) : step;
        const int64_t t_out = reverse_dir ? (seq_len - 1 - step) : step;

        for (int64_t h = 0; h < hidden; ++h) {
            const int64_t z_off = h;
            const int64_t r_off = hidden + h;

            float z_pre = 0.0f;
            float r_pre = 0.0f;

            for (int64_t i = 0; i < input_size; ++i) {
                const float x_i = ggml_cuda_gru_load_typed(x, x_type, i * x_nb0 + b_idx * x_nb1 + t_in * x_nb2);
                z_pre += ggml_cuda_gru_load_typed(w, w_type, i * w_nb0 + z_off * w_nb1 + d * w_nb2) * x_i;
                r_pre += ggml_cuda_gru_load_typed(w, w_type, i * w_nb0 + r_off * w_nb1 + d * w_nb2) * x_i;
            }

            for (int64_t j = 0; j < hidden; ++j) {
                const float h_j = h_prev[j];
                z_pre += ggml_cuda_gru_load_typed(r, r_type, j * r_nb0 + z_off * r_nb1 + d * r_nb2) * h_j;
                r_pre += ggml_cuda_gru_load_typed(r, r_type, j * r_nb0 + r_off * r_nb1 + d * r_nb2) * h_j;
            }

            if (has_bias) {
                z_pre += ggml_cuda_gru_load_typed(b, b_type, h * b_nb0 + d * b_nb1) +
                         ggml_cuda_gru_load_typed(b, b_type, (3 * hidden + h) * b_nb0 + d * b_nb1);
                r_pre += ggml_cuda_gru_load_typed(b, b_type, (hidden + h) * b_nb0 + d * b_nb1) +
                         ggml_cuda_gru_load_typed(b, b_type, (4 * hidden + h) * b_nb0 + d * b_nb1);
            }

            z_gate[h] = ggml_cuda_gru_sigmoid(z_pre);
            r_gate[h] = ggml_cuda_gru_sigmoid(r_pre);
        }

        for (int64_t h = 0; h < hidden; ++h) {
            const int64_t h_off = 2 * hidden + h;
            float n_pre = 0.0f;

            for (int64_t i = 0; i < input_size; ++i) {
                const float x_i = ggml_cuda_gru_load_typed(x, x_type, i * x_nb0 + b_idx * x_nb1 + t_in * x_nb2);
                n_pre += ggml_cuda_gru_load_typed(w, w_type, i * w_nb0 + h_off * w_nb1 + d * w_nb2) * x_i;
            }
            if (has_bias) {
                n_pre += ggml_cuda_gru_load_typed(b, b_type, (2 * hidden + h) * b_nb0 + d * b_nb1);
            }

            if (linear_before_reset == 1) {
                float rec = 0.0f;
                for (int64_t j = 0; j < hidden; ++j) {
                    rec += ggml_cuda_gru_load_typed(r, r_type, j * r_nb0 + h_off * r_nb1 + d * r_nb2) * h_prev[j];
                }
                if (has_bias) {
                    rec += ggml_cuda_gru_load_typed(b, b_type, (5 * hidden + h) * b_nb0 + d * b_nb1);
                }
                n_pre += r_gate[h] * rec;
            } else {
                float rec = 0.0f;
                for (int64_t j = 0; j < hidden; ++j) {
                    rec += ggml_cuda_gru_load_typed(r, r_type, j * r_nb0 + h_off * r_nb1 + d * r_nb2) * (r_gate[j] * h_prev[j]);
                }
                if (has_bias) {
                    rec += ggml_cuda_gru_load_typed(b, b_type, (5 * hidden + h) * b_nb0 + d * b_nb1);
                }
                n_pre += rec;
            }

            const float n_t = tanhf(n_pre);
            const float h_t = n_t + z_gate[h] * (h_prev[h] - n_t);
            h_cur[h] = h_t;

            if (output_last == 0) {
                y[h * y_nb0 + b_idx * y_nb1 + d * y_nb2 + t_out * y_nb3] = h_t;
            }
        }

        for (int64_t h = 0; h < hidden; ++h) {
            h_prev[h] = h_cur[h];
        }
    }

    if (output_last == 1) {
        for (int64_t h = 0; h < hidden; ++h) {
            y[h * y_nb0 + b_idx * y_nb1 + d * y_nb2] = h_prev[h];
        }
    }
}

void ggml_cuda_op_gru(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * x = dst->src[0];
    const ggml_tensor * w = dst->src[1];
    const ggml_tensor * r = dst->src[2];
    const ggml_tensor * b = dst->src[3];
    const ggml_tensor * h0 = dst->src[4];

    GGML_ASSERT(x != nullptr && w != nullptr && r != nullptr);

    const int32_t hidden_size = ggml_get_op_params_i32(dst, 0);
    const int32_t linear_before_reset = ggml_get_op_params_i32(dst, 1);
    const int32_t direction = ggml_get_op_params_i32(dst, 2);
    const int32_t output_last = ggml_get_op_params_i32(dst, 3);

    const int64_t seq_len = x->ne[2];
    const int64_t batch = x->ne[1];
    const int64_t input_size = x->ne[0];
    const int64_t num_dir = w->ne[2];
    const int64_t hidden = hidden_size > 0 ? (int64_t) hidden_size : (w->ne[1] / 3);

    GGML_ASSERT(hidden > 0 && seq_len > 0 && batch > 0 && input_size > 0 && num_dir > 0);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_ASSERT((size_t) hidden <= (48 * 1024) / (4 * sizeof(float)));

    const int64_t x_nb0 = x->nb[0] / (int64_t) ggml_type_size(x->type);
    const int64_t x_nb1 = x->nb[1] / (int64_t) ggml_type_size(x->type);
    const int64_t x_nb2 = x->nb[2] / (int64_t) ggml_type_size(x->type);

    const int64_t w_nb0 = w->nb[0] / (int64_t) ggml_type_size(w->type);
    const int64_t w_nb1 = w->nb[1] / (int64_t) ggml_type_size(w->type);
    const int64_t w_nb2 = w->nb[2] / (int64_t) ggml_type_size(w->type);

    const int64_t r_nb0 = r->nb[0] / (int64_t) ggml_type_size(r->type);
    const int64_t r_nb1 = r->nb[1] / (int64_t) ggml_type_size(r->type);
    const int64_t r_nb2 = r->nb[2] / (int64_t) ggml_type_size(r->type);

    const int64_t b_nb0 = b ? b->nb[0] / (int64_t) ggml_type_size(b->type) : 0;
    const int64_t b_nb1 = b ? b->nb[1] / (int64_t) ggml_type_size(b->type) : 0;

    const int64_t h0_nb0 = h0 ? h0->nb[0] / (int64_t) ggml_type_size(h0->type) : 0;
    const int64_t h0_nb1 = h0 ? h0->nb[1] / (int64_t) ggml_type_size(h0->type) : 0;
    const int64_t h0_nb2 = h0 ? h0->nb[2] / (int64_t) ggml_type_size(h0->type) : 0;

    const int64_t y_nb0 = dst->nb[0] / (int64_t) sizeof(float);
    const int64_t y_nb1 = dst->nb[1] / (int64_t) sizeof(float);
    const int64_t y_nb2 = dst->nb[2] / (int64_t) sizeof(float);
    const int64_t y_nb3 = dst->nb[3] / (int64_t) sizeof(float);

    const dim3 blocks((uint32_t) batch, (uint32_t) num_dir, 1);
    const dim3 threads(1, 1, 1);
    const size_t shared_mem = (size_t) hidden * 4 * sizeof(float);

    ggml_cuda_gru_kernel<<<blocks, threads, shared_mem, ctx.stream()>>>(
        x->data, w->data, r->data, b ? b->data : nullptr, h0 ? h0->data : nullptr, (float *) dst->data,
        x->type, w->type, r->type, b ? b->type : GGML_TYPE_COUNT, h0 ? h0->type : GGML_TYPE_COUNT,
        x_nb0, x_nb1, x_nb2,
        w_nb0, w_nb1, w_nb2,
        r_nb0, r_nb1, r_nb2,
        b_nb0, b_nb1,
        h0_nb0, h0_nb1, h0_nb2,
        y_nb0, y_nb1, y_nb2, y_nb3,
        (int32_t) hidden,
        (int32_t) input_size,
        (int32_t) seq_len,
        (int32_t) batch,
        (int32_t) num_dir,
        linear_before_reset,
        direction,
        output_last,
        b ? 1 : 0,
        h0 ? 1 : 0);
}
