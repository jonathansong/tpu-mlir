/* Tiny LLM Ada300 rx-ops bridge — full implementation.
 *
 * Each rxops_bridge_* function is emitted by convert-tpu-to-rxops for one TPU
 * op.  The implementation converts flat-pointer + dimension arguments into
 * rxops_tensor descriptors and calls the rx-ops public API.
 *
 * Ops covered:
 *   Add, Mul, MatMul, RMSNorm, A16MatMul (int4 AWQ dequant), Reshape (memcpy),
 *   Slice ND, Gather ND, Transpose ND, Concat 2/3/4, Rope (contiguous_halves),
 *   FAttention (full-sequence static prefill), SiLU, Sigmoid, Relu, Tanh,
 *   Gelu, TopK (configurable-k values+indices, also used for Arg/argmax k=1).
 */

#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "rx_ops.h"
#include "rx_ops_data_structure.h"
#include "rx_ops_data_type.h"
#include "internal/rx_ops_register.h"

extern void rxnn_ada300_init(void);
extern void *rxnn_mem_alloc(int64_t size);
extern void rxnn_mem_free(void *ptr);

static int g_inited;
static void ensure_ada300(void) {
  if (!g_inited) {
    rxnn_ada300_init();
    g_inited = 1;
  }
}

/* ── Tensor fill helpers ──────────────────────────────────────────────────── */

#define MAX_DIM_RXBRIDGE 8

static void fill_tensor_nd(struct rxops_tensor *t, void *data,
                            enum rxops_dtype_enum dtype,
                            const int64_t *dims, int rank, int layout) {
  memset(t, 0, sizeof(*t));
  t->data = data;
  t->dtype = dtype;
  t->mtype = RXOPS_MEM_TYPE_CPU;
  t->dim_count = rank;
  t->layout = layout;
  for (int i = 0; i < rank && i < MAX_DIM_RXBRIDGE; ++i)
    t->dim[i] = (int32_t)dims[i];
}

static void fill_tensor_1d(struct rxops_tensor *t, void *data,
                           enum rxops_dtype_enum dtype, int64_t n) {
  int64_t d = n;
  fill_tensor_nd(t, data, dtype, &d, 1, RXOPS_LAYOUT_N);
}

static void fill_tensor_2d(struct rxops_tensor *t, void *data,
                           enum rxops_dtype_enum dtype,
                           int64_t rows, int64_t cols) {
  int64_t d[2] = {rows, cols};
  fill_tensor_nd(t, data, dtype, d, 2, RXOPS_LAYOUT_NC);
}

static void fill_tensor_4d(struct rxops_tensor *t, void *data,
                           enum rxops_dtype_enum dtype,
                           int64_t d0, int64_t d1, int64_t d2, int64_t d3) {
  int64_t d[4] = {d0, d1, d2, d3};
  fill_tensor_nd(t, data, dtype, d, 4, RXOPS_LAYOUT_NCHW);
}

/* Make a params_base with Ada300 api and empty callback. */
static void make_params(struct rxops_params_base *p, struct rxops_callback *cb) {
  memset(p, 0, sizeof(*p));
  memset(cb, 0, sizeof(*cb));
  p->api = RXOPS_ADA300;
  p->cb = cb;
}

/* ── fp16 conversion helpers ─────────────────────────────────────────────── */

static inline void f32_to_f16(__fp16_t *dst, const float *src, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    __fp16 h = (__fp16)src[i];
    __builtin_memcpy(&dst[i], &h, sizeof(dst[i]));
  }
}

static inline void f16_to_f32(float *dst, const __fp16_t *src, int64_t n) {
  for (int64_t i = 0; i < n; ++i) {
    __fp16 h;
    __builtin_memcpy(&h, &src[i], sizeof(h));
    dst[i] = (float)h;
  }
}

/* Decode a double bit-cast packed into int64 (from the pass). */
static inline double i64_to_f64(int64_t bits) {
  double v;
  memcpy(&v, &bits, sizeof(v));
  return v;
}

static inline int64_t product(const int64_t *dims, int n) {
  int64_t r = 1;
  for (int i = 0; i < n; ++i) r *= dims[i];
  return r;
}

/* ── Add f32 ─────────────────────────────────────────────────────────────── */

int rxops_bridge_add_f32(float *out, const float *in0, const float *in1,
                         int64_t n) {
  struct rxops_tensor t_in0, t_in1, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT32, n);
  fill_tensor_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT32, n);
  fill_tensor_1d(&t_out, (void *)out,  RXOPS_DTYPE_FLOAT32, n);
  make_params(&params, &cb);
  rxops_add_init(&t_in0, &t_in1, &t_out, &params, NULL);
  return rxops_add(&t_in0, &t_in1, &t_out, &params, NULL);
}

/* ── Mul f32 ─────────────────────────────────────────────────────────────── */

int rxops_bridge_mul_f32(float *out, const float *in0, const float *in1,
                         int64_t n) {
  struct rxops_tensor t_in0, t_in1, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT32, n);
  fill_tensor_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT32, n);
  fill_tensor_1d(&t_out, (void *)out,  RXOPS_DTYPE_FLOAT32, n);
  make_params(&params, &cb);
  rxops_mul_init(&t_in0, &t_in1, &t_out, &params, NULL);
  return rxops_mul(&t_in0, &t_in1, &t_out, &params, NULL);
}

/* ── MatMul f32 (input f32 → convert to f16, run matmul, convert back) ───── */

int rxops_bridge_matmul_f32(float *C, const float *A, const float *B,
                            int64_t M, int64_t N, int64_t K) {
  int64_t szA = M * K, szB = K * N, szC = M * N;
  __fp16_t *A16 = (__fp16_t *)rxnn_mem_alloc(szA * sizeof(__fp16_t));
  __fp16_t *B16 = (__fp16_t *)rxnn_mem_alloc(szB * sizeof(__fp16_t));
  __fp16_t *C16 = (__fp16_t *)rxnn_mem_alloc(szC * sizeof(__fp16_t));
  if (!A16 || !B16 || !C16) {
    rxnn_mem_free(A16); rxnn_mem_free(B16); rxnn_mem_free(C16);
    return RXOPS_FALSE;
  }
  ensure_ada300();
  f32_to_f16(A16, A, szA);
  f32_to_f16(B16, B, szB);

  struct rxops_tensor t_a, t_b, t_c;
  struct rxops_params_base params;
  struct rxops_ada300_matmul_params ada_p;
  struct rxops_callback cb;
  fill_tensor_2d(&t_a, A16, RXOPS_DTYPE_FLOAT16, M, K);
  fill_tensor_2d(&t_b, B16, RXOPS_DTYPE_FLOAT16, K, N);
  fill_tensor_2d(&t_c, C16, RXOPS_DTYPE_FLOAT16, M, N);
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.trans_a = 0; ada_p.trans_b = 0;
  rxops_matmul_init(&t_a, &t_b, &t_c, &params, &ada_p);
  int rc = rxops_matmul(&t_a, &t_b, &t_c, &params, &ada_p);
  f16_to_f32(C, C16, szC);
  rxnn_mem_free(A16); rxnn_mem_free(B16); rxnn_mem_free(C16);
  return rc;
}

/* ── RMSNorm f32 ─────────────────────────────────────────────────────────── */

int rxops_bridge_rms_norm_f32(float *out, const float *in, const float *gamma,
                               int64_t batch, int64_t hidden,
                               int64_t eps_i64) {
  double eps_d = i64_to_f64(eps_i64);
  float eps = (float)eps_d;
  struct rxops_tensor t_in, t_w, t_out;
  struct rxops_params_base params;
  struct rxops_ada300_rms_norm_params ada_p;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_2d(&t_in, (void *)in,    RXOPS_DTYPE_FLOAT32, batch, hidden);
  fill_tensor_1d(&t_w,  (void *)gamma, RXOPS_DTYPE_FLOAT32, hidden);
  fill_tensor_2d(&t_out,(void *)out,   RXOPS_DTYPE_FLOAT32, batch, hidden);
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.epsilon = eps;
  ada_p.axis = -1;
  rxops_rms_norm_init(&t_in, &t_w, &t_out, &params, &ada_p);
  return rxops_rms_norm(&t_in, &t_w, &t_out, &params, &ada_p);
}

/* ── A16MatMul: int4 AWQ dequantize then fullyconnected ─────────────────── */
/*
 * qweight: [N, K_packed] ui8 where K_packed = K/2 for 4-bit
 * scales:  [N, num_groups] f32
 * qzeros:  [N, num_groups] ui8
 * input:   [M, K] f32
 * output:  [M, N] f32
 */

static void dequant_int4_awq(float *w_f32,
                              const uint8_t *qw, const float *scales,
                              const uint8_t *qzeros,
                              int64_t N, int64_t K, int64_t gs) {
  int64_t num_groups = K / gs;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      int64_t g = k / gs;
      float scale = scales[n * num_groups + g];
      uint8_t zp   = qzeros[n * num_groups + g];
      /* packed int4: even k → low nibble, odd k → high nibble */
      int64_t byte_idx = n * (K / 2) + k / 2;
      uint8_t byte = qw[byte_idx];
      int w = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
      w_f32[n * K + k] = (float)(w - (int)zp) * scale;
    }
  }
}

int rxops_bridge_a16matmul_f32(float *out, const float *in,
                                const uint8_t *qweight, const float *scales,
                                const uint8_t *qzeros,
                                int64_t M, int64_t K, int64_t N,
                                int64_t q_group_size, int64_t weight_bits) {
  (void)weight_bits; /* currently assumes 4-bit */
  ensure_ada300();

  /* Dequantize weights to f32 [N, K] */
  float *w_f32 = (float *)rxnn_mem_alloc(N * K * sizeof(float));
  if (!w_f32) return RXOPS_FALSE;
  dequant_int4_awq(w_f32, qweight, scales, qzeros, N, K, q_group_size);

  /* Convert input and weight to f16 for the matmul kernel */
  __fp16_t *in16 = (__fp16_t *)rxnn_mem_alloc(M * K * sizeof(__fp16_t));
  __fp16_t *w16  = (__fp16_t *)rxnn_mem_alloc(N * K * sizeof(__fp16_t));
  __fp16_t *out16 = (__fp16_t *)rxnn_mem_alloc(M * N * sizeof(__fp16_t));
  if (!in16 || !w16 || !out16) {
    rxnn_mem_free(w_f32); rxnn_mem_free(in16);
    rxnn_mem_free(w16); rxnn_mem_free(out16);
    return RXOPS_FALSE;
  }
  f32_to_f16(in16, in, M * K);
  f32_to_f16(w16, w_f32, N * K);
  rxnn_mem_free(w_f32);

  /* fullyconnected: input [M, K], weights [N, K] (transposed applied by fc),
   * output [M, N] */
  struct rxops_tensor t_in, t_w, t_out, t_bias;
  struct rxops_params_base params;
  struct rxops_ada300_fc_params ada_p;
  struct rxops_callback cb;
  fill_tensor_2d(&t_in,  in16,  RXOPS_DTYPE_FLOAT16, M, K);
  fill_tensor_2d(&t_w,   w16,   RXOPS_DTYPE_FLOAT16, N, K);
  fill_tensor_2d(&t_out, out16, RXOPS_DTYPE_FLOAT16, M, N);
  memset(&t_bias, 0, sizeof(t_bias));
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.units = (int32_t)N;
  rxops_fullyconnected_init(&t_in, &t_out, &t_w, NULL, &params, &ada_p);
  int rc = rxops_fullyconnected(&t_in, &t_out, &t_w, NULL, &params, &ada_p);
  f16_to_f32(out, out16, M * N);
  rxnn_mem_free(in16); rxnn_mem_free(w16); rxnn_mem_free(out16);
  return rc;
}

/* ── Reshape (memcpy) ────────────────────────────────────────────────────── */

int rxops_bridge_reshape_bytes(void *out, const void *in, int64_t nbytes) {
  memcpy(out, in, (size_t)nbytes);
  return RXOPS_TRUE;
}

/* ── Slice ND (strided copy, max kMaxRxOpsBridgeRank=6 dims) ─────────────── */
/*
 * Args after elem_bytes, rank:
 *   out_shape[6], in_shape[6], offsets[6], steps[6]
 * Performs a C-nested loop copy from the strided subregion of `in` to `out`.
 */

int rxops_bridge_slice_nd(void *out, const void *in, int64_t elem_bytes,
                          int64_t rank,
                          int64_t os0, int64_t os1, int64_t os2,
                          int64_t os3, int64_t os4, int64_t os5,
                          int64_t is0, int64_t is1, int64_t is2,
                          int64_t is3, int64_t is4, int64_t is5,
                          int64_t of0, int64_t of1, int64_t of2,
                          int64_t of3, int64_t of4, int64_t of5,
                          int64_t st0, int64_t st1, int64_t st2,
                          int64_t st3, int64_t st4, int64_t st5) {
  int64_t out_shape[6] = {os0, os1, os2, os3, os4, os5};
  int64_t in_shape[6]  = {is0, is1, is2, is3, is4, is5};
  int64_t offsets[6]   = {of0, of1, of2, of3, of4, of5};
  int64_t steps[6]     = {st0, st1, st2, st3, st4, st5};

  /* Compute input strides (row-major / C-contiguous) */
  int64_t in_stride[6];
  in_stride[rank - 1] = 1;
  for (int i = (int)rank - 2; i >= 0; --i)
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];

  /* Total elements to copy */
  int64_t total = 1;
  for (int i = 0; i < (int)rank; ++i) total *= out_shape[i];

  /* Iterate over all output indices using a flat counter */
  for (int64_t flat = 0; flat < total; ++flat) {
    /* Decompose flat index into per-dim output indices */
    int64_t tmp = flat;
    int64_t out_idx[6], in_offset = 0;
    for (int d = (int)rank - 1; d >= 0; --d) {
      out_idx[d] = tmp % out_shape[d];
      tmp /= out_shape[d];
      in_offset += (offsets[d] + out_idx[d] * steps[d]) * in_stride[d];
    }
    char *dst = (char *)out + flat * elem_bytes;
    const char *src = (const char *)in + in_offset * elem_bytes;
    memcpy(dst, src, (size_t)elem_bytes);
  }
  return RXOPS_TRUE;
}

/* ── Gather ND (axis gather, up to 6 dims) ───────────────────────────────── */
/*
 * data:    [ds0..5] (rank given by leading non-1 dims)
 * indices: [is0..5]
 * out:     [os0..5]
 * Semantics: out[i0..iR] = data[ replace axis dim with floor(indices[...]) ]
 */

int rxops_bridge_gather_nd(void *out, const void *data, const void *indices,
                           int64_t axis, int64_t elem_bytes,
                           int64_t ds0, int64_t ds1, int64_t ds2,
                           int64_t ds3, int64_t ds4, int64_t ds5,
                           int64_t is0, int64_t is1, int64_t is2,
                           int64_t is3, int64_t is4, int64_t is5,
                           int64_t os0, int64_t os1, int64_t os2,
                           int64_t os3, int64_t os4, int64_t os5) {
  int64_t data_shape[6]  = {ds0, ds1, ds2, ds3, ds4, ds5};
  int64_t idx_shape[6]   = {is0, is1, is2, is3, is4, is5};
  int64_t out_shape[6]   = {os0, os1, os2, os3, os4, os5};

  /* Determine actual ranks from non-trailing-1 dims (heuristic: use ds5≠1) */
  /* For correctness, infer data_rank from ds values */
  int data_rank = 6;
  while (data_rank > 1 && data_shape[data_rank - 1] == 1) --data_rank;
  int idx_rank = 6;
  while (idx_rank > 1 && idx_shape[idx_rank - 1] == 1) --idx_rank;

  /* Data strides */
  int64_t data_stride[6];
  data_stride[data_rank - 1] = 1;
  for (int i = data_rank - 2; i >= 0; --i)
    data_stride[i] = data_stride[i + 1] * data_shape[i + 1];

  /* Output rank = data_rank - 1 + idx_rank (axis gather semantics) */
  /* out_shape already provided by the pass */
  int out_rank = data_rank - 1 + idx_rank;
  if (out_rank > 6) out_rank = 6; /* clamp */

  /* Compute total output elements */
  int64_t total_out = 1;
  for (int i = 0; i < out_rank; ++i) total_out *= out_shape[i];
  int64_t total_idx = 1;
  for (int i = 0; i < idx_rank; ++i) total_idx *= idx_shape[i];

  /* Iterate over output */
  for (int64_t flat_out = 0; flat_out < total_out; ++flat_out) {
    /* Decompose output index */
    int64_t tmp = flat_out;
    int64_t oi[6] = {0};
    for (int d = out_rank - 1; d >= 0; --d) {
      oi[d] = tmp % out_shape[d];
      tmp /= out_shape[d];
    }
    /* Map output dims to data dims:
     *  - dims [0, axis) → data dims [0, axis)
     *  - dims [axis, axis+idx_rank) → index flat index → gather dim
     *  - dims [axis+idx_rank, out_rank) → data dims [axis+1, data_rank)  */
    int64_t data_idx[6] = {0};
    for (int i = 0; i < axis; ++i) data_idx[i] = oi[i];
    /* Compute flat index into indices tensor from out dims [axis..axis+idx_rank) */
    int64_t flat_idx = 0;
    int64_t idx_stride_tmp = 1;
    for (int i = idx_rank - 1; i >= 0; --i) {
      flat_idx += oi[(int)axis + i] * idx_stride_tmp;
      idx_stride_tmp *= idx_shape[i];
    }
    /* Read the gather index value */
    int64_t gather_dim_idx = 0;
    if (elem_bytes == 4) {
      gather_dim_idx = (int64_t)((const float *)indices)[flat_idx];
    } else if (elem_bytes == 8) {
      gather_dim_idx = (int64_t)((const double *)indices)[flat_idx];
    } else {
      /* integer index tensors (si32) — treat as int32 */
      gather_dim_idx = (int64_t)((const int32_t *)indices)[flat_idx];
    }
    data_idx[axis] = gather_dim_idx;
    for (int i = (int)axis + 1; i < data_rank; ++i)
      data_idx[i] = oi[(int)axis + idx_rank + (i - (int)axis - 1)];
    /* Compute flat src offset */
    int64_t src_off = 0;
    for (int d = 0; d < data_rank; ++d)
      src_off += data_idx[d] * data_stride[d];
    memcpy((char *)out + flat_out * elem_bytes,
           (const char *)data + src_off * elem_bytes,
           (size_t)elem_bytes);
  }
  return RXOPS_TRUE;
}

/* ── Transpose ND ────────────────────────────────────────────────────────── */

int rxops_bridge_transpose_nd(void *out, const void *in, int64_t elem_bytes,
                               int64_t rank,
                               int64_t p0, int64_t p1, int64_t p2,
                               int64_t p3, int64_t p4, int64_t p5,
                               int64_t is0, int64_t is1, int64_t is2,
                               int64_t is3, int64_t is4, int64_t is5,
                               int64_t os0, int64_t os1, int64_t os2,
                               int64_t os3, int64_t os4, int64_t os5) {
  int64_t perm[6]      = {p0, p1, p2, p3, p4, p5};
  int64_t in_shape[6]  = {is0, is1, is2, is3, is4, is5};
  int64_t out_shape[6] = {os0, os1, os2, os3, os4, os5};

  /* Input strides */
  int64_t in_stride[6];
  in_stride[rank - 1] = 1;
  for (int i = (int)rank - 2; i >= 0; --i)
    in_stride[i] = in_stride[i + 1] * in_shape[i + 1];

  int64_t total = 1;
  for (int i = 0; i < (int)rank; ++i) total *= out_shape[i];

  for (int64_t flat = 0; flat < total; ++flat) {
    int64_t tmp = flat;
    int64_t out_idx[6];
    for (int d = (int)rank - 1; d >= 0; --d) {
      out_idx[d] = tmp % out_shape[d];
      tmp /= out_shape[d];
    }
    /* out_idx[d] corresponds to in_idx[perm[d]] */
    int64_t src_off = 0;
    for (int d = 0; d < (int)rank; ++d)
      src_off += out_idx[d] * in_stride[perm[d]];
    memcpy((char *)out + flat * elem_bytes,
           (const char *)in + src_off * elem_bytes,
           (size_t)elem_bytes);
  }
  return RXOPS_TRUE;
}

/* ── Concat ND helpers ────────────────────────────────────────────────────── */
/*
 * Concatenate along `axis`. All inputs/output are C-contiguous f32 (or any
 * dtype since we use elem_bytes).
 * Passes: out*, in0*, in1*, elem_bytes, rank, axis,
 *         out_shape[6], in0_shape[6], in1_shape[6]
 */

static int concat_nd_impl(void *out, int n_inputs, const void **inputs,
                          int64_t elem_bytes, int64_t rank, int64_t axis,
                          const int64_t *out_shape,
                          const int64_t in_shapes[][6]) {
  /* Compute output strides */
  int64_t out_stride[6];
  out_stride[rank - 1] = 1;
  for (int i = (int)rank - 2; i >= 0; --i)
    out_stride[i] = out_stride[i + 1] * out_shape[i + 1];

  /* For each input, compute offset in the output along axis */
  int64_t axis_offset = 0;
  for (int inp = 0; inp < n_inputs; ++inp) {
    const int64_t *is = in_shapes[inp];
    /* Input strides */
    int64_t in_stride[6];
    in_stride[rank - 1] = 1;
    for (int i = (int)rank - 2; i >= 0; --i)
      in_stride[i] = in_stride[i + 1] * is[i + 1];

    int64_t total_in = 1;
    for (int i = 0; i < (int)rank; ++i) total_in *= is[i];

    for (int64_t flat = 0; flat < total_in; ++flat) {
      int64_t tmp = flat;
      int64_t in_idx[6];
      for (int d = (int)rank - 1; d >= 0; --d) {
        in_idx[d] = tmp % is[d];
        tmp /= is[d];
      }
      /* Output index: same as in_idx except axis dim shifted */
      int64_t out_flat = 0;
      for (int d = 0; d < (int)rank; ++d) {
        int64_t oi = (d == (int)axis) ? (in_idx[d] + axis_offset) : in_idx[d];
        out_flat += oi * out_stride[d];
      }
      memcpy((char *)out + out_flat * elem_bytes,
             (const char *)inputs[inp] + flat * elem_bytes,
             (size_t)elem_bytes);
    }
    axis_offset += is[axis];
  }
  return RXOPS_TRUE;
}

int rxops_bridge_concat2_nd(void *out, const void *in0, const void *in1,
                             int64_t eb, int64_t rank, int64_t axis,
                             int64_t os0, int64_t os1, int64_t os2,
                             int64_t os3, int64_t os4, int64_t os5,
                             int64_t i0s0, int64_t i0s1, int64_t i0s2,
                             int64_t i0s3, int64_t i0s4, int64_t i0s5,
                             int64_t i1s0, int64_t i1s1, int64_t i1s2,
                             int64_t i1s3, int64_t i1s4, int64_t i1s5) {
  const void *inputs[2] = {in0, in1};
  int64_t out_shape[6] = {os0, os1, os2, os3, os4, os5};
  int64_t in_shapes[2][6] = {{i0s0, i0s1, i0s2, i0s3, i0s4, i0s5},
                              {i1s0, i1s1, i1s2, i1s3, i1s4, i1s5}};
  return concat_nd_impl(out, 2, inputs, eb, rank, axis, out_shape, in_shapes);
}

int rxops_bridge_concat3_nd(void *out, const void *in0, const void *in1,
                             const void *in2,
                             int64_t eb, int64_t rank, int64_t axis,
                             int64_t os0, int64_t os1, int64_t os2,
                             int64_t os3, int64_t os4, int64_t os5,
                             int64_t i0s0, int64_t i0s1, int64_t i0s2,
                             int64_t i0s3, int64_t i0s4, int64_t i0s5,
                             int64_t i1s0, int64_t i1s1, int64_t i1s2,
                             int64_t i1s3, int64_t i1s4, int64_t i1s5,
                             int64_t i2s0, int64_t i2s1, int64_t i2s2,
                             int64_t i2s3, int64_t i2s4, int64_t i2s5) {
  const void *inputs[3] = {in0, in1, in2};
  int64_t out_shape[6] = {os0, os1, os2, os3, os4, os5};
  int64_t in_shapes[3][6] = {{i0s0, i0s1, i0s2, i0s3, i0s4, i0s5},
                              {i1s0, i1s1, i1s2, i1s3, i1s4, i1s5},
                              {i2s0, i2s1, i2s2, i2s3, i2s4, i2s5}};
  return concat_nd_impl(out, 3, inputs, eb, rank, axis, out_shape, in_shapes);
}

int rxops_bridge_concat4_nd(void *out, const void *in0, const void *in1,
                             const void *in2, const void *in3,
                             int64_t eb, int64_t rank, int64_t axis,
                             int64_t os0, int64_t os1, int64_t os2,
                             int64_t os3, int64_t os4, int64_t os5,
                             int64_t i0s0, int64_t i0s1, int64_t i0s2,
                             int64_t i0s3, int64_t i0s4, int64_t i0s5,
                             int64_t i1s0, int64_t i1s1, int64_t i1s2,
                             int64_t i1s3, int64_t i1s4, int64_t i1s5,
                             int64_t i2s0, int64_t i2s1, int64_t i2s2,
                             int64_t i2s3, int64_t i2s4, int64_t i2s5,
                             int64_t i3s0, int64_t i3s1, int64_t i3s2,
                             int64_t i3s3, int64_t i3s4, int64_t i3s5) {
  const void *inputs[4] = {in0, in1, in2, in3};
  int64_t out_shape[6] = {os0, os1, os2, os3, os4, os5};
  int64_t in_shapes[4][6] = {{i0s0, i0s1, i0s2, i0s3, i0s4, i0s5},
                              {i1s0, i1s1, i1s2, i1s3, i1s4, i1s5},
                              {i2s0, i2s1, i2s2, i2s3, i2s4, i2s5},
                              {i3s0, i3s1, i3s2, i3s3, i3s4, i3s5}};
  return concat_nd_impl(out, 4, inputs, eb, rank, axis, out_shape, in_shapes);
}

/* ── Rope (contiguous_halves, f32) ──────────────────────────────────────── */
/*
 * q:   [batch, seq, n_heads, head_dim] f32
 * cos: [batch, seq, cos_heads, head_dim] f32  (cos_heads may be 1 for GQA)
 * sin: [batch, seq, cos_heads, head_dim] f32
 * out: same shape as q
 *
 * contiguous_halves mode:
 *   half = head_dim / 2
 *   out[..., :half]  = q[..., :half]*cos[..., :half]  - q[..., half:]*sin[..., :half]
 *   out[..., half:]  = q[..., :half]*sin[..., :half]  + q[..., half:]*cos[..., :half]
 *
 * Note: input2 in tpu::RopeOp is sin, input3 is cos (as in block_7.mlir).
 */

int rxops_bridge_rope_contiguous_f32(float *out, const float *q,
                                      const float *cos_in, const float *sin_in,
                                      int64_t seq, int64_t n_heads,
                                      int64_t head_dim, int64_t cos_heads) {
  int64_t half = head_dim / 2;
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t h = 0; h < n_heads; ++h) {
      const float *qrow = q  + (s * n_heads + h) * head_dim;
      float *orow       = out + (s * n_heads + h) * head_dim;
      /* cos/sin may broadcast over heads (cos_heads=1 for GQA) */
      int64_t ch = h % cos_heads;
      const float *crow = cos_in + (s * cos_heads + ch) * head_dim;
      const float *srow = sin_in + (s * cos_heads + ch) * head_dim;
      for (int64_t d = 0; d < half; ++d) {
        float q0 = qrow[d];
        float q1 = qrow[d + half];
        float c  = crow[d];
        float si = srow[d];
        orow[d]        = q0 * c - q1 * si;
        orow[d + half] = q0 * si + q1 * c;
      }
    }
  }
  return RXOPS_TRUE;
}

/* ── FAttention (static prefill, full sequence) ──────────────────────────── */
/*
 * Uses rxops_full_attention with a trivial paged KV cache:
 *   cache_k, cache_v: one block of size = mk (all keys fit in one block)
 *   block_table: [0]
 *   seq_lens: [0]  (pre-fill starting from 0)
 *
 * Output: [1, mq, np_q*d_head] — the flattened attention result.
 */

int rxops_bridge_fattention_f32(float *out, const float *q,
                                 const float *k, const float *v,
                                 const float *mask,
                                 int64_t batch, int64_t mq, int64_t mk,
                                 int64_t np_q, int64_t np_kv, int64_t d_head,
                                 int64_t scale_i64, int64_t has_mask) {
  (void)batch;
  double scale_d = i64_to_f64(scale_i64);
  float norm_factor = (float)scale_d;
  ensure_ada300();

  /* Allocate paged KV cache: 1 block, shape [1, np_kv, mk, d_head] */
  int64_t kv_elems = np_kv * mk * d_head;
  float *cache_k_buf = (float *)rxnn_mem_alloc(kv_elems * sizeof(float));
  float *cache_v_buf = (float *)rxnn_mem_alloc(kv_elems * sizeof(float));
  int32_t *block_table_buf = (int32_t *)rxnn_mem_alloc(sizeof(int32_t));
  int32_t *seq_lens_buf    = (int32_t *)rxnn_mem_alloc(sizeof(int32_t));
  float *attn_out_buf = (float *)rxnn_mem_alloc(mq * np_q * d_head * sizeof(float));
  if (!cache_k_buf || !cache_v_buf || !block_table_buf ||
      !seq_lens_buf || !attn_out_buf) {
    rxnn_mem_free(cache_k_buf); rxnn_mem_free(cache_v_buf);
    rxnn_mem_free(block_table_buf); rxnn_mem_free(seq_lens_buf);
    rxnn_mem_free(attn_out_buf);
    return RXOPS_FALSE;
  }
  block_table_buf[0] = 0;
  seq_lens_buf[0] = 0;

  /* Tensor descriptors:
   *   q      [1, mq, np_q, d_head]
   *   k_new  [1, mk, np_kv, d_head]
   *   v_new  [1, mk, np_kv, d_head]
   *   cache_k [num_blocks=1, np_kv, block_size=mk, d_head]
   *   cache_v [1, np_kv, mk, d_head]
   *   block_table [1, max_blocks_per_seq=1]
   *   seq_lens [1]
   *   output  [1, mq, np_q, d_head]
   */
  struct rxops_tensor t_q, t_k, t_v, t_ck, t_cv, t_bt, t_sl, t_mask, t_out;
  fill_tensor_4d(&t_q,  (void *)q,            RXOPS_DTYPE_FLOAT32, 1, mq, np_q, d_head);
  fill_tensor_4d(&t_k,  (void *)k,            RXOPS_DTYPE_FLOAT32, 1, mk, np_kv, d_head);
  fill_tensor_4d(&t_v,  (void *)v,            RXOPS_DTYPE_FLOAT32, 1, mk, np_kv, d_head);
  fill_tensor_4d(&t_ck, cache_k_buf,          RXOPS_DTYPE_FLOAT32, 1, np_kv, mk, d_head);
  fill_tensor_4d(&t_cv, cache_v_buf,          RXOPS_DTYPE_FLOAT32, 1, np_kv, mk, d_head);
  fill_tensor_2d(&t_bt, block_table_buf,      RXOPS_DTYPE_INT32, 1, 1);
  fill_tensor_1d(&t_sl, seq_lens_buf,         RXOPS_DTYPE_INT32, 1);
  fill_tensor_4d(&t_out, attn_out_buf,        RXOPS_DTYPE_FLOAT32, 1, mq, np_q, d_head);

  struct rxops_params_base params;
  struct rxops_ada300_full_attention_params ada_p;
  struct rxops_callback cb;
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.norm_factor = norm_factor;
  ada_p.np_q = (int32_t)np_q;
  ada_p.np_kv = (int32_t)np_kv;
  ada_p.d_head = (int32_t)d_head;
  ada_p.block_size = (int32_t)mk;
  ada_p.max_blocks_per_seq = 1;
  ada_p.causal = 0; /* prefill: no causal mask (or use attn_mask if provided) */
  ada_p.has_mask = (has_mask != 0);

  struct rxops_tensor *mask_ptr = NULL;
  if (has_mask && mask) {
    fill_tensor_4d(&t_mask, (void *)mask, RXOPS_DTYPE_FLOAT32, 1, 1, mq, mk);
    mask_ptr = &t_mask;
  }

  rxops_full_attention_init(&t_q, &t_k, &t_v, &t_ck, &t_cv,
                            &t_bt, &t_sl, mask_ptr, &t_out, &params, &ada_p);
  int rc = rxops_full_attention(&t_q, &t_k, &t_v, &t_ck, &t_cv,
                                &t_bt, &t_sl, mask_ptr, &t_out, &params, &ada_p);

  /* Copy attention result to output, flattening last two dims */
  memcpy(out, attn_out_buf, (size_t)(mq * np_q * d_head) * sizeof(float));

  rxnn_mem_free(cache_k_buf); rxnn_mem_free(cache_v_buf);
  rxnn_mem_free(block_table_buf); rxnn_mem_free(seq_lens_buf);
  rxnn_mem_free(attn_out_buf);
  return rc;
}

/* ── Unary activations f32 ───────────────────────────────────────────────── */

int rxops_bridge_silu_f32(float *out, const float *in, int64_t n) {
  struct rxops_tensor t_in, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
  fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
  make_params(&params, &cb);
  rxops_silu_init(&t_in, &t_out, &params, NULL);
  return rxops_silu(&t_in, &t_out, &params, NULL);
}

int rxops_bridge_sigmoid_f32(float *out, const float *in, int64_t n) {
  struct rxops_tensor t_in, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT32, n);
  fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT32, n);
  make_params(&params, &cb);
  rxops_sigmoid_init(&t_in, &t_out, &params, NULL);
  return rxops_sigmoid(&t_in, &t_out, &params, NULL);
}

int rxops_bridge_relu_f32(float *out, const float *in, int64_t n) {
  ensure_ada300();
  /* Fallback: C loop — Ada300 relu via rxops_relu would need the cb wired */
  for (int64_t i = 0; i < n; ++i)
    out[i] = in[i] > 0.0f ? in[i] : 0.0f;
  return RXOPS_TRUE;
}

int rxops_bridge_tanh_f32(float *out, const float *in, int64_t n) {
  for (int64_t i = 0; i < n; ++i)
    out[i] = tanhf(in[i]);
  return RXOPS_TRUE;
}

int rxops_bridge_gelu_f32(float *out, const float *in, int64_t n) {
  /* tanh-GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715*x^3))) */
  static const float k0 = 0.7978845608028654f; /* sqrt(2/pi) */
  for (int64_t i = 0; i < n; ++i) {
    float x = in[i];
    float inner = k0 * (x + 0.044715f * x * x * x);
    out[i] = 0.5f * x * (1.0f + tanhf(inner));
  }
  return RXOPS_TRUE;
}

/* ── TopK (used for ArgMax via k=1) ─────────────────────────────────────── */
/*
 * Finds top-k values and their indices along an axis.
 * n_before = product of dims before axis
 * axis_size = size of the axis dimension
 * n_after = product of dims after axis
 */

int rxops_bridge_topk_f32(float *vals_out, float *idx_out, const float *in,
                           int64_t k, int64_t n_before, int64_t axis_size,
                           int64_t n_after) {
  /* Simple partial sort: for each (before, after) slice, find top-k. */
  for (int64_t b = 0; b < n_before; ++b) {
    for (int64_t a = 0; a < n_after; ++a) {
      /* Gather the axis slice */
      for (int64_t ki = 0; ki < k; ++ki) {
        float best_val = -3.4028235e+38f; /* -FLT_MAX */
        int64_t best_idx = 0;
        for (int64_t j = 0; j < axis_size; ++j) {
          float v = in[b * axis_size * n_after + j * n_after + a];
          /* Exclude already-chosen indices for top-k > 1 */
          int already = 0;
          for (int64_t ki2 = 0; ki2 < ki; ++ki2) {
            if ((int64_t)idx_out[b * k * n_after + ki2 * n_after + a] == j) {
              already = 1; break;
            }
          }
          if (!already && v > best_val) {
            best_val = v; best_idx = j;
          }
        }
        int64_t out_flat = b * k * n_after + ki * n_after + a;
        vals_out[out_flat] = best_val;
        idx_out[out_flat]  = (float)best_idx;
      }
    }
  }
  return RXOPS_TRUE;
}

/* ── Unused stubs kept for linkage compatibility ─────────────────────────── */

int rxops_bridge_exp_f32(float *out, const float *in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = expf(in[i]);
  return RXOPS_TRUE;
}

int rxops_bridge_sqrt_f32(float *out, const float *in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = sqrtf(in[i]);
  return RXOPS_TRUE;
}

int rxops_bridge_log_f32(float *out, const float *in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = logf(in[i]);
  return RXOPS_TRUE;
}

int rxops_bridge_rsqrt_f32(float *out, const float *in, int64_t n) {
  for (int64_t i = 0; i < n; ++i) out[i] = 1.0f / sqrtf(in[i]);
  return RXOPS_TRUE;
}

/* ══════════════════════════════════════════════════════════════════════════
 * F16 bridge variants — use native Ada300 fp16 kernels via vfcvt hardware.
 * The pass emits these when tensor element type is f16 (post-TopToTpu lowering).
 * ══════════════════════════════════════════════════════════════════════════ */

/* ── Cast helpers via rxops_tensor_data_convert (uses vfcvt hw) ─────────── */

int rxops_bridge_cast_f16_to_f32(void *out_f32, const void *in_f16,
                                   int64_t n) {
  struct rxops_tensor t_src, t_dst;
  ensure_ada300();
  fill_tensor_1d(&t_src, (void *)in_f16,  RXOPS_DTYPE_FLOAT16, n);
  fill_tensor_1d(&t_dst, (void *)out_f32, RXOPS_DTYPE_FLOAT32, n);
  return rxops_tensor_data_convert(&t_dst, &t_src);
}

int rxops_bridge_cast_f32_to_f16(void *out_f16, const void *in_f32,
                                   int64_t n) {
  struct rxops_tensor t_src, t_dst;
  ensure_ada300();
  fill_tensor_1d(&t_src, (void *)in_f32,  RXOPS_DTYPE_FLOAT32, n);
  fill_tensor_1d(&t_dst, (void *)out_f16, RXOPS_DTYPE_FLOAT16, n);
  return rxops_tensor_data_convert(&t_dst, &t_src);
}

/* ── Add f16 ─────────────────────────────────────────────────────────────── */

int rxops_bridge_add_f16(void *out, const void *in0, const void *in1,
                          int64_t n) {
  struct rxops_tensor t_in0, t_in1, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT16, n);
  fill_tensor_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT16, n);
  fill_tensor_1d(&t_out, (void *)out,  RXOPS_DTYPE_FLOAT16, n);
  make_params(&params, &cb);
  rxops_add_init(&t_in0, &t_in1, &t_out, &params, NULL);
  return rxops_add(&t_in0, &t_in1, &t_out, &params, NULL);
}

/* ── Mul f16 ─────────────────────────────────────────────────────────────── */

int rxops_bridge_mul_f16(void *out, const void *in0, const void *in1,
                          int64_t n) {
  struct rxops_tensor t_in0, t_in1, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in0, (void *)in0, RXOPS_DTYPE_FLOAT16, n);
  fill_tensor_1d(&t_in1, (void *)in1, RXOPS_DTYPE_FLOAT16, n);
  fill_tensor_1d(&t_out, (void *)out,  RXOPS_DTYPE_FLOAT16, n);
  make_params(&params, &cb);
  rxops_mul_init(&t_in0, &t_in1, &t_out, &params, NULL);
  return rxops_mul(&t_in0, &t_in1, &t_out, &params, NULL);
}

/* ── MatMul f16 (input already f16 — no conversion needed) ──────────────── */

int rxops_bridge_matmul_f16(void *C, const void *A, const void *B,
                              int64_t M, int64_t N, int64_t K) {
  struct rxops_tensor t_a, t_b, t_c;
  struct rxops_params_base params;
  struct rxops_ada300_matmul_params ada_p;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_2d(&t_a, (void *)A, RXOPS_DTYPE_FLOAT16, M, K);
  fill_tensor_2d(&t_b, (void *)B, RXOPS_DTYPE_FLOAT16, K, N);
  fill_tensor_2d(&t_c, (void *)C, RXOPS_DTYPE_FLOAT16, M, N);
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.trans_a = 0; ada_p.trans_b = 0;
  rxops_matmul_init(&t_a, &t_b, &t_c, &params, &ada_p);
  return rxops_matmul(&t_a, &t_b, &t_c, &params, &ada_p);
}

/* ── RMSNorm f16 ─────────────────────────────────────────────────────────── */

int rxops_bridge_rms_norm_f16(void *out, const void *in, const void *gamma,
                               int64_t batch, int64_t hidden,
                               int64_t eps_i64) {
  double eps_d = i64_to_f64(eps_i64);
  float eps = (float)eps_d;
  struct rxops_tensor t_in, t_w, t_out;
  struct rxops_params_base params;
  struct rxops_ada300_rms_norm_params ada_p;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_2d(&t_in, (void *)in,    RXOPS_DTYPE_FLOAT16, batch, hidden);
  fill_tensor_1d(&t_w,  (void *)gamma, RXOPS_DTYPE_FLOAT16, hidden);
  fill_tensor_2d(&t_out,(void *)out,   RXOPS_DTYPE_FLOAT16, batch, hidden);
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.epsilon = eps;
  ada_p.axis = -1;
  rxops_rms_norm_init(&t_in, &t_w, &t_out, &params, &ada_p);
  return rxops_rms_norm(&t_in, &t_w, &t_out, &params, &ada_p);
}

/* ── A16MatMul f16 input (weight stays int4, output is f16) ─────────────── */

int rxops_bridge_a16matmul_f16(void *out, const void *in,
                                const uint8_t *qweight, const float *scales,
                                const uint8_t *qzeros,
                                int64_t M, int64_t K, int64_t N,
                                int64_t q_group_size, int64_t weight_bits) {
  /* Dequantize weights to f16 */
  __fp16_t *w16 = (__fp16_t *)rxnn_mem_alloc(N * K * sizeof(__fp16_t));
  __fp16_t *out16_tmp = (__fp16_t *)rxnn_mem_alloc(M * N * sizeof(__fp16_t));
  if (!w16 || !out16_tmp) {
    rxnn_mem_free(w16); rxnn_mem_free(out16_tmp);
    return RXOPS_FALSE;
  }
  /* Dequant int4 weights to f16 */
  int64_t num_groups = K / q_group_size;
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t k = 0; k < K; ++k) {
      int64_t g = k / q_group_size;
      float scale = scales[n * num_groups + g];
      uint8_t zp   = qzeros[n * num_groups + g];
      int64_t byte_idx = n * (K / 2) + k / 2;
      uint8_t byte = qweight[byte_idx];
      int w = (k & 1) ? ((byte >> 4) & 0xF) : (byte & 0xF);
      __fp16 h = (__fp16)((float)(w - (int)zp) * scale);
      __builtin_memcpy(&w16[n * K + k], &h, sizeof(h));
    }
  }

  struct rxops_tensor t_in, t_w, t_out;
  struct rxops_params_base params;
  struct rxops_ada300_fc_params ada_p;
  struct rxops_callback cb;
  fill_tensor_2d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT16, M, K);
  fill_tensor_2d(&t_w,   w16,          RXOPS_DTYPE_FLOAT16, N, K);
  fill_tensor_2d(&t_out, out16_tmp,    RXOPS_DTYPE_FLOAT16, M, N);
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.units = (int32_t)N;
  rxops_fullyconnected_init(&t_in, &t_out, &t_w, NULL, &params, &ada_p);
  int rc = rxops_fullyconnected(&t_in, &t_out, &t_w, NULL, &params, &ada_p);
  memcpy(out, out16_tmp, (size_t)(M * N) * sizeof(__fp16_t));
  rxnn_mem_free(w16); rxnn_mem_free(out16_tmp);
  return rc;
}

/* ── Activations f16 ─────────────────────────────────────────────────────── */

int rxops_bridge_silu_f16(void *out, const void *in, int64_t n) {
  struct rxops_tensor t_in, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT16, n);
  fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT16, n);
  make_params(&params, &cb);
  rxops_silu_init(&t_in, &t_out, &params, NULL);
  return rxops_silu(&t_in, &t_out, &params, NULL);
}

int rxops_bridge_sigmoid_f16(void *out, const void *in, int64_t n) {
  struct rxops_tensor t_in, t_out;
  struct rxops_params_base params;
  struct rxops_callback cb;
  ensure_ada300();
  fill_tensor_1d(&t_in,  (void *)in,  RXOPS_DTYPE_FLOAT16, n);
  fill_tensor_1d(&t_out, (void *)out, RXOPS_DTYPE_FLOAT16, n);
  make_params(&params, &cb);
  rxops_sigmoid_init(&t_in, &t_out, &params, NULL);
  return rxops_sigmoid(&t_in, &t_out, &params, NULL);
}

/* relu/tanh/gelu f16 — C scalar loops using __fp16 arithmetic */

int rxops_bridge_relu_f16(void *out, const void *in, int64_t n) {
  const __fp16 *i16 = (const __fp16 *)in;
  __fp16 *o16 = (__fp16 *)out;
  __fp16 zero = (__fp16)0.0f;
  for (int64_t i = 0; i < n; ++i)
    o16[i] = (i16[i] > zero) ? i16[i] : zero;
  return RXOPS_TRUE;
}

int rxops_bridge_tanh_f16(void *out, const void *in, int64_t n) {
  const __fp16 *i16 = (const __fp16 *)in;
  __fp16 *o16 = (__fp16 *)out;
  for (int64_t i = 0; i < n; ++i)
    o16[i] = (__fp16)tanhf((float)i16[i]);
  return RXOPS_TRUE;
}

int rxops_bridge_gelu_f16(void *out, const void *in, int64_t n) {
  static const float k0 = 0.7978845608028654f;
  const __fp16 *i16 = (const __fp16 *)in;
  __fp16 *o16 = (__fp16 *)out;
  for (int64_t i = 0; i < n; ++i) {
    float x = (float)i16[i];
    float inner = k0 * (x + 0.044715f * x * x * x);
    o16[i] = (__fp16)(0.5f * x * (1.0f + tanhf(inner)));
  }
  return RXOPS_TRUE;
}

/* ── Rope f16 (contiguous_halves, using __fp16 arithmetic) ──────────────── */

int rxops_bridge_rope_contiguous_f16(void *out, const void *q,
                                      const void *cos_in, const void *sin_in,
                                      int64_t seq, int64_t n_heads,
                                      int64_t head_dim, int64_t cos_heads) {
  const __fp16 *qp   = (const __fp16 *)q;
  const __fp16 *cp   = (const __fp16 *)cos_in;
  const __fp16 *sp   = (const __fp16 *)sin_in;
  __fp16 *op         = (__fp16 *)out;
  int64_t half = head_dim / 2;
  for (int64_t s = 0; s < seq; ++s) {
    for (int64_t h = 0; h < n_heads; ++h) {
      const __fp16 *qrow = qp + (s * n_heads + h) * head_dim;
      __fp16 *orow       = op + (s * n_heads + h) * head_dim;
      int64_t ch = h % cos_heads;
      const __fp16 *crow = cp + (s * cos_heads + ch) * head_dim;
      const __fp16 *srow = sp + (s * cos_heads + ch) * head_dim;
      for (int64_t d = 0; d < half; ++d) {
        float q0 = (float)qrow[d];
        float q1 = (float)qrow[d + half];
        float c  = (float)crow[d];
        float si = (float)srow[d];
        orow[d]        = (__fp16)(q0 * c - q1 * si);
        orow[d + half] = (__fp16)(q0 * si + q1 * c);
      }
    }
  }
  return RXOPS_TRUE;
}

/* ── FAttention f16 ──────────────────────────────────────────────────────── */

int rxops_bridge_fattention_f16(void *out, const void *q,
                                  const void *k, const void *v,
                                  const void *mask,
                                  int64_t batch, int64_t mq, int64_t mk,
                                  int64_t np_q, int64_t np_kv, int64_t d_head,
                                  int64_t scale_i64, int64_t has_mask) {
  (void)batch;
  double scale_d = i64_to_f64(scale_i64);
  float norm_factor = (float)scale_d;
  ensure_ada300();

  int64_t kv_elems = np_kv * mk * d_head;
  __fp16_t *cache_k_buf = (__fp16_t *)rxnn_mem_alloc(kv_elems * sizeof(__fp16_t));
  __fp16_t *cache_v_buf = (__fp16_t *)rxnn_mem_alloc(kv_elems * sizeof(__fp16_t));
  int32_t *block_table_buf = (int32_t *)rxnn_mem_alloc(sizeof(int32_t));
  int32_t *seq_lens_buf    = (int32_t *)rxnn_mem_alloc(sizeof(int32_t));
  __fp16_t *attn_out_buf   = (__fp16_t *)rxnn_mem_alloc(
                               mq * np_q * d_head * sizeof(__fp16_t));
  if (!cache_k_buf || !cache_v_buf || !block_table_buf ||
      !seq_lens_buf || !attn_out_buf) {
    rxnn_mem_free(cache_k_buf); rxnn_mem_free(cache_v_buf);
    rxnn_mem_free(block_table_buf); rxnn_mem_free(seq_lens_buf);
    rxnn_mem_free(attn_out_buf);
    return RXOPS_FALSE;
  }
  block_table_buf[0] = 0;
  seq_lens_buf[0] = 0;

  struct rxops_tensor t_q, t_k, t_v, t_ck, t_cv, t_bt, t_sl, t_mask, t_out;
  fill_tensor_4d(&t_q,  (void *)q,         RXOPS_DTYPE_FLOAT16, 1, mq, np_q, d_head);
  fill_tensor_4d(&t_k,  (void *)k,         RXOPS_DTYPE_FLOAT16, 1, mk, np_kv, d_head);
  fill_tensor_4d(&t_v,  (void *)v,         RXOPS_DTYPE_FLOAT16, 1, mk, np_kv, d_head);
  fill_tensor_4d(&t_ck, cache_k_buf,       RXOPS_DTYPE_FLOAT16, 1, np_kv, mk, d_head);
  fill_tensor_4d(&t_cv, cache_v_buf,       RXOPS_DTYPE_FLOAT16, 1, np_kv, mk, d_head);
  fill_tensor_2d(&t_bt, block_table_buf,   RXOPS_DTYPE_INT32, 1, 1);
  fill_tensor_1d(&t_sl, seq_lens_buf,      RXOPS_DTYPE_INT32, 1);
  fill_tensor_4d(&t_out, attn_out_buf,     RXOPS_DTYPE_FLOAT16, 1, mq, np_q, d_head);

  struct rxops_params_base params;
  struct rxops_ada300_full_attention_params ada_p;
  struct rxops_callback cb;
  make_params(&params, &cb);
  memset(&ada_p, 0, sizeof(ada_p));
  ada_p.norm_factor = norm_factor;
  ada_p.np_q = (int32_t)np_q;
  ada_p.np_kv = (int32_t)np_kv;
  ada_p.d_head = (int32_t)d_head;
  ada_p.block_size = (int32_t)mk;
  ada_p.max_blocks_per_seq = 1;
  ada_p.causal = 0;
  ada_p.has_mask = (has_mask != 0);

  struct rxops_tensor *mask_ptr = NULL;
  if (has_mask && mask) {
    fill_tensor_4d(&t_mask, (void *)mask, RXOPS_DTYPE_FLOAT16, 1, 1, mq, mk);
    mask_ptr = &t_mask;
  }

  rxops_full_attention_init(&t_q, &t_k, &t_v, &t_ck, &t_cv,
                            &t_bt, &t_sl, mask_ptr, &t_out, &params, &ada_p);
  int rc = rxops_full_attention(&t_q, &t_k, &t_v, &t_ck, &t_cv,
                                &t_bt, &t_sl, mask_ptr, &t_out, &params, &ada_p);

  memcpy(out, attn_out_buf,
         (size_t)(mq * np_q * d_head) * sizeof(__fp16_t));

  rxnn_mem_free(cache_k_buf); rxnn_mem_free(cache_v_buf);
  rxnn_mem_free(block_table_buf); rxnn_mem_free(seq_lens_buf);
  rxnn_mem_free(attn_out_buf);
  return rc;
}

/* ── TopK f16 ────────────────────────────────────────────────────────────── */

int rxops_bridge_topk_f16(void *vals_out, void *idx_out, const void *in,
                            int64_t k, int64_t n_before, int64_t axis_size,
                            int64_t n_after) {
  const __fp16 *inp = (const __fp16 *)in;
  __fp16 *vout = (__fp16 *)vals_out;
  __fp16 *iout = (__fp16 *)idx_out;
  for (int64_t b = 0; b < n_before; ++b) {
    for (int64_t a = 0; a < n_after; ++a) {
      for (int64_t ki = 0; ki < k; ++ki) {
        float best_val = -65504.0f; /* -FLT_MAX for f16 */
        int64_t best_idx = 0;
        for (int64_t j = 0; j < axis_size; ++j) {
          float v = (float)inp[b * axis_size * n_after + j * n_after + a];
          int already = 0;
          for (int64_t ki2 = 0; ki2 < ki; ++ki2) {
            if ((int64_t)(float)iout[b * k * n_after + ki2 * n_after + a] == j) {
              already = 1; break;
            }
          }
          if (!already && v > best_val) { best_val = v; best_idx = j; }
        }
        int64_t out_flat = b * k * n_after + ki * n_after + a;
        vout[out_flat] = (__fp16)best_val;
        iout[out_flat] = (__fp16)(float)best_idx;
      }
    }
  }
  return RXOPS_TRUE;
}

/* ── _mlir_ciface_ wrappers (called from MLIR C-interface ABI) ───────────── */

int _mlir_ciface_rxops_bridge_add_f32(float *out, const float *in0,
                                       const float *in1, int64_t n) {
  return rxops_bridge_add_f32(out, in0, in1, n);
}

int _mlir_ciface_rxops_bridge_matmul_f32(float *C, const float *A,
                                          const float *B, int64_t M,
                                          int64_t N, int64_t K) {
  return rxops_bridge_matmul_f32(C, A, B, M, N, K);
}

