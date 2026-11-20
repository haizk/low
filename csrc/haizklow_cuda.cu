// csrc/haizklow_cuda.cu
// HaizkLow v3 — Hardware-Native NHWC Morton 4x4 Brick SE(2) CUDA C Engine
//
// Kernels:
//   1. build_bmap_and_neighbors  : 1-shot int32 brick_map [B,G,G] + 8-neighbor table [8,N]
//   2. gather_bricks_nhwc        : Coalesced [B,C,H,W] -> [N,4,4,C]
//   3. scatter_bricks_nhwc_fwd/bwd: Coalesced [N,4,4,C] <-> [B,C,H,W] + mask
//   4. prolong_bricks_nhwc_fwd/bwd: Atomic-free exact bilinear 2x [M,4,4,C] <-> [4*M,4,4,C]
//   5. restrict_bricks_nhwc_fwd/bwd: Atomic-free exact 2x2 avg [4*M,4,4,C] <-> [M,4,4,C]
//   6. fused_se2_laplacian_fwd/bwd : Zero-bank-conflict SRAM SE(2) Collinear Transport + +-45 deg Turning
//                                    (Neumann zero-flux, self-adjoint, ZERO atomics on grad_u/grad_a/grad_c!)

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <algorithm>

template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

static constexpr int TILE_ROW_STRIDE = 8;
static constexpr int TILE_CH_STRIDE  = 52; // max index = 5*8+5 = 45 < 52; exact 0 bank conflicts!
static constexpr int MAX_FIBER_D     = 64;

// ============================================================
// 1. 1-Shot int32 Brick Map & 8-Neighbor Builder
// ============================================================

__global__ void fill_bmap_kernel(
    const int32_t* __restrict__ b_idx,
    const int32_t* __restrict__ by_idx,
    const int32_t* __restrict__ bx_idx,
    int32_t* __restrict__ bmap,
    int32_t N, int32_t G
) {
    int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int64_t b = b_idx[n], by = by_idx[n], bx = bx_idx[n];
    bmap[(b * G + by) * G + bx] = n;
}

__global__ void build_nb_kernel(
    const int32_t* __restrict__ b_idx,
    const int32_t* __restrict__ by_idx,
    const int32_t* __restrict__ bx_idx,
    const int32_t* __restrict__ bmap,
    int32_t* __restrict__ nb_idx,
    int32_t N, int32_t G
) {
    int32_t n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;
    int32_t b = b_idx[n], by = by_idx[n], bx = bx_idx[n];
    const int32_t* bmap_b = bmap + static_cast<int64_t>(b) * G * G;
    const int dy[8] = {-1,  1,  0, 0, -1, -1,  1, 1};
    const int dx[8] = { 0,  0, -1, 1, -1,  1, -1, 1};

    #pragma unroll
    for (int d = 0; d < 8; ++d) {
        int32_t ny = by + dy[d];
        int32_t nx = bx + dx[d];
        int32_t nb = N;
        if (ny >= 0 && ny < G && nx >= 0 && nx < G) {
            int32_t idx = bmap_b[ny * G + nx];
            if (idx >= 0) nb = idx;
        }
        nb_idx[static_cast<int64_t>(d) * N + n] = nb;
    }
}

// ============================================================
// 2 & 3. Coalesced NHWC Gather & Scatter ([B,C,H,W] <-> [N,4,4,C])
// ============================================================

template <typename scalar_t>
__global__ void gather_bricks_nhwc_kernel(
    const scalar_t* __restrict__ dense,
    const int32_t* __restrict__ b_idx,
    const int32_t* __restrict__ by_idx,
    const int32_t* __restrict__ bx_idx,
    scalar_t* __restrict__ bricks,
    int64_t N, int64_t C, int64_t G
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = N * 16 * C;
    if (tid >= total) return;

    int64_t c = tid % C;
    int64_t tmp = tid / C;
    int p = static_cast<int>(tmp & 15);
    int64_t n = tmp >> 4;

    int64_t b = b_idx[n];
    int64_t y = static_cast<int64_t>(by_idx[n]) * 4 + (p >> 2);
    int64_t x = static_cast<int64_t>(bx_idx[n]) * 4 + (p & 3);
    int64_t H = G * 4;
    bricks[tid] = dense[(b * C + c) * (H * H) + y * H + x];
}

template <typename scalar_t>
__global__ void scatter_bricks_nhwc_fwd_kernel(
    const scalar_t* __restrict__ bricks,
    const int32_t* __restrict__ b_idx,
    const int32_t* __restrict__ by_idx,
    const int32_t* __restrict__ bx_idx,
    scalar_t* __restrict__ dense,
    scalar_t* __restrict__ mask,
    int64_t N, int64_t C, int64_t G
) {
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t n = tid >> 4;
    if (n >= N) return;

    int p = static_cast<int>(tid & 15);
    int64_t b = b_idx[n];
    int64_t y = static_cast<int64_t>(by_idx[n]) * 4 + (p >> 2);
    int64_t x = static_cast<int64_t>(bx_idx[n]) * 4 + (p & 3);
    int64_t H = G * 4;
    int64_t hw = H * H;

    const scalar_t* bp = bricks + (n * 16 + p) * C;
    for (int64_t c = 0; c < C; ++c) {
        dense[(b * C + c) * hw + y * H + x] = bp[c];
    }
    mask[b * hw + y * H + x] = static_cast<scalar_t>(1);
}

// ============================================================
// 4 & 5. Atomic-Free Bilinear Prolongation & 2x2 Avg Restriction (NHWC)
// ============================================================

template <typename scalar_t>
__global__ void prolong_bricks_nhwc_fwd_kernel(
    const scalar_t* __restrict__ parent,
    scalar_t* __restrict__ children,
    int64_t M, int64_t C
) {
    using acc_t = typename AccType<scalar_t>::type;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * 64 * C;
    if (tid >= total) return;

    int64_t c = tid % C;
    int64_t tmp = tid / C;
    int p = static_cast<int>(tmp & 15);
    int64_t mc = tmp >> 4;
    int64_t m = mc >> 2;
    int q = static_cast<int>(mc & 3);

    int oy = (q >> 1) * 4 + (p >> 2);
    int ox = (q & 1) * 4 + (p & 3);

    const int i0_t[8] = {0, 0, 0, 1, 1, 2, 2, 3};
    const int i1_t[8] = {0, 1, 1, 2, 2, 3, 3, 3};
    const acc_t w1_t[8] = {acc_t(0), acc_t(0.25), acc_t(0.75), acc_t(0.25), acc_t(0.75), acc_t(0.25), acc_t(0.75), acc_t(0)};

    int y0 = i0_t[oy], y1 = i1_t[oy];
    int x0 = i0_t[ox], x1 = i1_t[ox];
    acc_t wy1 = w1_t[oy], wy0 = acc_t(1) - wy1;
    acc_t wx1 = w1_t[ox], wx0 = acc_t(1) - wx1;

    const scalar_t* pm = parent + m * 16 * C + c;
    acc_t v00 = static_cast<acc_t>(pm[(y0 * 4 + x0) * C]);
    acc_t v01 = static_cast<acc_t>(pm[(y0 * 4 + x1) * C]);
    acc_t v10 = static_cast<acc_t>(pm[(y1 * 4 + x0) * C]);
    acc_t v11 = static_cast<acc_t>(pm[(y1 * 4 + x1) * C]);

    acc_t res = wy0 * (wx0 * v00 + wx1 * v01) + wy1 * (wx0 * v10 + wx1 * v11);
    children[tid] = static_cast<scalar_t>(res);
}

template <typename scalar_t>
__global__ void prolong_bricks_nhwc_bwd_kernel(
    const scalar_t* __restrict__ grad_children,
    scalar_t* __restrict__ grad_parent,
    int64_t M, int64_t C
) {
    using acc_t = typename AccType<scalar_t>::type;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * 16 * C;
    if (tid >= total) return;

    int64_t c = tid % C;
    int64_t tmp = tid / C;
    int p = static_cast<int>(tmp & 15);
    int64_t m = tmp >> 4;
    int py = p >> 2;
    int px = p & 3;

    const int rev_len[4] = {3, 4, 4, 3};
    const int rev_o[4][4] = {{0, 1, 2, 0}, {1, 2, 3, 4}, {3, 4, 5, 6}, {5, 6, 7, 0}};
    const acc_t rev_w[4][4] = {
        {acc_t(1.0), acc_t(0.75), acc_t(0.25), acc_t(0.0)},
        {acc_t(0.25), acc_t(0.75), acc_t(0.75), acc_t(0.25)},
        {acc_t(0.25), acc_t(0.75), acc_t(0.75), acc_t(0.25)},
        {acc_t(0.25), acc_t(0.75), acc_t(1.0), acc_t(0.0)}
    };

    const scalar_t* gcm = grad_children + m * 64 * C + c;
    acc_t acc = acc_t(0);
    int ly = rev_len[py];
    int lx = rev_len[px];
    for (int iy = 0; iy < ly; ++iy) {
        int oy = rev_o[py][iy];
        acc_t wy = rev_w[py][iy];
        int qy = oy >> 2, cy = oy & 3;
        for (int ix = 0; ix < lx; ++ix) {
            int ox = rev_o[px][ix];
            acc_t wx = rev_w[px][ix];
            int qx = ox >> 2, cx = ox & 3;
            int q = (qy << 1) | qx;
            acc += (wy * wx) * static_cast<acc_t>(gcm[(q * 16 + cy * 4 + cx) * C]);
        }
    }
    grad_parent[tid] = static_cast<scalar_t>(acc);
}

template <typename scalar_t>
__global__ void restrict_bricks_nhwc_fwd_kernel(
    const scalar_t* __restrict__ children,
    scalar_t* __restrict__ parent,
    int64_t M, int64_t C
) {
    using acc_t = typename AccType<scalar_t>::type;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * 16 * C;
    if (tid >= total) return;

    int64_t c = tid % C;
    int64_t tmp = tid / C;
    int p = static_cast<int>(tmp & 15);
    int64_t m = tmp >> 4;
    int py = p >> 2;
    int px = p & 3;

    int q = ((py >> 1) << 1) | (px >> 1);
    int cy0 = (py & 1) * 2;
    int cx0 = (px & 1) * 2;

    const scalar_t* chq = children + (m * 4 + q) * 16 * C + c;
    acc_t s = static_cast<acc_t>(chq[(cy0 * 4 + cx0) * C])
            + static_cast<acc_t>(chq[(cy0 * 4 + cx0 + 1) * C])
            + static_cast<acc_t>(chq[((cy0 + 1) * 4 + cx0) * C])
            + static_cast<acc_t>(chq[((cy0 + 1) * 4 + cx0 + 1) * C]);
    parent[tid] = static_cast<scalar_t>(acc_t(0.25) * s);
}

template <typename scalar_t>
__global__ void restrict_bricks_nhwc_bwd_kernel(
    const scalar_t* __restrict__ grad_parent,
    scalar_t* __restrict__ grad_children,
    int64_t M, int64_t C
) {
    using acc_t = typename AccType<scalar_t>::type;
    int64_t tid = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = M * 64 * C;
    if (tid >= total) return;

    int64_t c = tid % C;
    int64_t tmp = tid / C;
    int p = static_cast<int>(tmp & 15);
    int64_t mc = tmp >> 4;
    int64_t m = mc >> 2;
    int q = static_cast<int>(mc & 3);

    int py = (q >> 1) * 2 + (p >> 3);
    int px = (q & 1) * 2 + ((p & 3) >> 1);
    acc_t gp = static_cast<acc_t>(grad_parent[(m * 16 + py * 4 + px) * C + c]);
    grad_children[tid] = static_cast<scalar_t>(acc_t(0.25) * gp);
}

// ============================================================
// 6. Fused SRAM SE(2) Graph Laplacian PDE (Zero Bank Conflicts, Zero-Atomic u/a/c Backward!)
// ============================================================

// Load 6x6 halo of scalar NHWC tensor x[N, 16, 4] for fiber tk (zero on missing neighbor -> Neumann zero-flux!)
template <typename scalar_t, typename acc_t>
__device__ __forceinline__ void load_6x6_nhwc_k(
    const scalar_t* __restrict__ x,
    const int32_t* __restrict__ nb_idx,
    int64_t n, int64_t N, int tk, int tx, acc_t* s
) {
    int py = tx >> 2;
    int px = tx & 3;
    s[(py + 1) * TILE_ROW_STRIDE + (px + 1)] = static_cast<acc_t>(x[(n * 16 + tx) * 4 + tk]);

    if (tx < 4) {
        int64_t nb = static_cast<int64_t>(nb_idx[0 * N + n]);
        s[0 * TILE_ROW_STRIDE + (tx + 1)] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + 12 + tx) * 4 + tk]) : acc_t(0);
    } else if (tx < 8) {
        int k = tx - 4;
        int64_t nb = static_cast<int64_t>(nb_idx[1 * N + n]);
        s[5 * TILE_ROW_STRIDE + (k + 1)] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + k) * 4 + tk]) : acc_t(0);
    } else if (tx < 12) {
        int k = tx - 8;
        int64_t nb = static_cast<int64_t>(nb_idx[2 * N + n]);
        s[(k + 1) * TILE_ROW_STRIDE + 0] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + k * 4 + 3) * 4 + tk]) : acc_t(0);
    } else {
        int k = tx - 12;
        int64_t nb = static_cast<int64_t>(nb_idx[3 * N + n]);
        s[(k + 1) * TILE_ROW_STRIDE + 5] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + k * 4 + 0) * 4 + tk]) : acc_t(0);
    }

    if (tx == 0) {
        int64_t nb = static_cast<int64_t>(nb_idx[4 * N + n]);
        s[0 * TILE_ROW_STRIDE + 0] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + 15) * 4 + tk]) : acc_t(0);
    } else if (tx == 1) {
        int64_t nb = static_cast<int64_t>(nb_idx[5 * N + n]);
        s[0 * TILE_ROW_STRIDE + 5] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + 12) * 4 + tk]) : acc_t(0);
    } else if (tx == 2) {
        int64_t nb = static_cast<int64_t>(nb_idx[6 * N + n]);
        s[5 * TILE_ROW_STRIDE + 0] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + 3) * 4 + tk]) : acc_t(0);
    } else if (tx == 3) {
        int64_t nb = static_cast<int64_t>(nb_idx[7 * N + n]);
        s[5 * TILE_ROW_STRIDE + 5] = (nb < N) ? static_cast<acc_t>(x[(nb * 16 + 0) * 4 + tk]) : acc_t(0);
    }
}

// Load 6x6 halo of NHWC tensor u[N, 16, 4, D] for fiber tk and channel d
template <typename scalar_t, typename acc_t>
__device__ __forceinline__ void load_6x6_nhwc_kd(
    const scalar_t* __restrict__ u,
    const int32_t* __restrict__ nb_idx,
    int64_t n, int64_t N, int tk, int64_t d, int64_t D, int tx, acc_t* s
) {
    int py = tx >> 2;
    int px = tx & 3;
    int64_t k_off = tk * D + d;
    int64_t stride_p = 4 * D;
    s[(py + 1) * TILE_ROW_STRIDE + (px + 1)] = static_cast<acc_t>(u[(n * 16 + tx) * stride_p + k_off]);

    if (tx < 4) {
        int64_t nb = static_cast<int64_t>(nb_idx[0 * N + n]);
        s[0 * TILE_ROW_STRIDE + (tx + 1)] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + 12 + tx) * stride_p + k_off]) : acc_t(0);
    } else if (tx < 8) {
        int k = tx - 4;
        int64_t nb = static_cast<int64_t>(nb_idx[1 * N + n]);
        s[5 * TILE_ROW_STRIDE + (k + 1)] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + k) * stride_p + k_off]) : acc_t(0);
    } else if (tx < 12) {
        int k = tx - 8;
        int64_t nb = static_cast<int64_t>(nb_idx[2 * N + n]);
        s[(k + 1) * TILE_ROW_STRIDE + 0] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + k * 4 + 3) * stride_p + k_off]) : acc_t(0);
    } else {
        int k = tx - 12;
        int64_t nb = static_cast<int64_t>(nb_idx[3 * N + n]);
        s[(k + 1) * TILE_ROW_STRIDE + 5] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + k * 4 + 0) * stride_p + k_off]) : acc_t(0);
    }

    if (tx == 0) {
        int64_t nb = static_cast<int64_t>(nb_idx[4 * N + n]);
        s[0 * TILE_ROW_STRIDE + 0] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + 15) * stride_p + k_off]) : acc_t(0);
    } else if (tx == 1) {
        int64_t nb = static_cast<int64_t>(nb_idx[5 * N + n]);
        s[0 * TILE_ROW_STRIDE + 5] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + 12) * stride_p + k_off]) : acc_t(0);
    } else if (tx == 2) {
        int64_t nb = static_cast<int64_t>(nb_idx[6 * N + n]);
        s[5 * TILE_ROW_STRIDE + 0] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + 3) * stride_p + k_off]) : acc_t(0);
    } else if (tx == 3) {
        int64_t nb = static_cast<int64_t>(nb_idx[7 * N + n]);
        s[5 * TILE_ROW_STRIDE + 5] = (nb < N) ? static_cast<acc_t>(u[(nb * 16 + 0) * stride_p + k_off]) : acc_t(0);
    }
}

template <typename scalar_t>
__global__ void fused_se2_laplacian_fwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ dt_s,
    const scalar_t* __restrict__ dt_c,
    const int32_t* __restrict__ nb_idx,
    scalar_t* __restrict__ out,
    int64_t N, int64_t D
) {
    using acc_t = typename AccType<scalar_t>::type;
    __shared__ acc_t smem_a[4][TILE_CH_STRIDE];
    __shared__ acc_t smem_u[4][TILE_CH_STRIDE];

    int64_t n = blockIdx.x;
    if (n >= N) return;

    int tx = threadIdx.x;
    int tk = threadIdx.y;
    int py = tx >> 2;
    int px = tx & 3;

    const int dy_arr[4] = {0, -1, 1, 1};
    const int dx_arr[4] = {1,  1, 0, 1};
    int dy = dy_arr[tk], dx = dx_arr[tk];
    int center  = (py + 1) * TILE_ROW_STRIDE + (px + 1);
    int idx_pos = (py + 1 + dy) * TILE_ROW_STRIDE + (px + 1 + dx);
    int idx_neg = (py + 1 - dy) * TILE_ROW_STRIDE + (px + 1 - dx);
    int kp = (tk + 1) & 3;
    int km = (tk + 3) & 3;

    // Load a halo ONCE and reuse across all D channels!
    load_6x6_nhwc_k<scalar_t, acc_t>(a, nb_idx, n, N, tk, tx, smem_a[tk]);
    __syncthreads();

    acc_t ap = smem_a[tk][center];
    acc_t g_pos = ap * smem_a[tk][idx_pos];
    acc_t g_neg = ap * smem_a[tk][idx_neg];
    acc_t c_fwd = static_cast<acc_t>(c[(n * 16 + tx) * 4 + tk]);
    acc_t c_bwd = static_cast<acc_t>(c[(n * 16 + tx) * 4 + km]);

    int64_t out_base = ((n * 16 + tx) * 4 + tk) * D;
    for (int64_t d = 0; d < D; ++d) {
        load_6x6_nhwc_kd<scalar_t, acc_t>(u, nb_idx, n, N, tk, d, D, tx, smem_u[tk]);
        __syncthreads();

        acc_t up = smem_u[tk][center];
        acc_t sp_flux  = g_pos * (smem_u[tk][idx_pos] - up) + g_neg * (smem_u[tk][idx_neg] - up);
        acc_t ang_flux = c_fwd * (smem_u[kp][center] - up)  + c_bwd * (smem_u[km][center] - up);
        acc_t val = up + static_cast<acc_t>(dt_s[d]) * sp_flux + static_cast<acc_t>(dt_c[d]) * ang_flux;
        out[out_base + d] = static_cast<scalar_t>(val);
        __syncthreads();
    }
}

template <typename scalar_t>
__global__ void fused_se2_laplacian_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ a,
    const scalar_t* __restrict__ c,
    const scalar_t* __restrict__ dt_s,
    const scalar_t* __restrict__ dt_c,
    const int32_t* __restrict__ nb_idx,
    scalar_t* __restrict__ grad_u,
    scalar_t* __restrict__ grad_a,
    scalar_t* __restrict__ grad_c,
    typename AccType<scalar_t>::type* __restrict__ grad_dt_s,
    typename AccType<scalar_t>::type* __restrict__ grad_dt_c,
    int64_t N, int64_t D
) {
    using acc_t = typename AccType<scalar_t>::type;
    __shared__ acc_t smem_a[4][TILE_CH_STRIDE];
    __shared__ acc_t smem_u[4][TILE_CH_STRIDE];
    __shared__ acc_t smem_go[4][TILE_CH_STRIDE];
    __shared__ acc_t s_gdts[MAX_FIBER_D];
    __shared__ acc_t s_gdtc[MAX_FIBER_D];
    __shared__ acc_t s_warp_s[2];
    __shared__ acc_t s_warp_c[2];

    int tx = threadIdx.x;
    int tk = threadIdx.y;
    int lane_id = tk * 16 + tx;
    int warp_id = lane_id >> 5;
    int lane    = lane_id & 31;
    int py = tx >> 2;
    int px = tx & 3;

    if (lane_id < D) {
        s_gdts[lane_id] = acc_t(0);
        s_gdtc[lane_id] = acc_t(0);
    }
    __syncthreads();

    const int dy_arr[4] = {0, -1, 1, 1};
    const int dx_arr[4] = {1,  1, 0, 1};
    int dy = dy_arr[tk], dx = dx_arr[tk];
    int center  = (py + 1) * TILE_ROW_STRIDE + (px + 1);
    int idx_pos = (py + 1 + dy) * TILE_ROW_STRIDE + (px + 1 + dx);
    int idx_neg = (py + 1 - dy) * TILE_ROW_STRIDE + (px + 1 - dx);
    int kp = (tk + 1) & 3;
    int km = (tk + 3) & 3;

    for (int64_t n = blockIdx.x; n < N; n += gridDim.x) {
        load_6x6_nhwc_k<scalar_t, acc_t>(a, nb_idx, n, N, tk, tx, smem_a[tk]);
        __syncthreads();

        acc_t ap    = smem_a[tk][center];
        acc_t a_pos = smem_a[tk][idx_pos];
        acc_t a_neg = smem_a[tk][idx_neg];
        acc_t g_pos = ap * a_pos;
        acc_t g_neg = ap * a_neg;
        acc_t c_fwd = static_cast<acc_t>(c[(n * 16 + tx) * 4 + tk]);
        acc_t c_bwd = static_cast<acc_t>(c[(n * 16 + tx) * 4 + km]);

        acc_t ga_acc = acc_t(0);
        acc_t gc_acc = acc_t(0);
        int64_t u_base = ((n * 16 + tx) * 4 + tk) * D;

        for (int64_t d = 0; d < D; ++d) {
            load_6x6_nhwc_kd<scalar_t, acc_t>(u, nb_idx, n, N, tk, d, D, tx, smem_u[tk]);
            load_6x6_nhwc_kd<scalar_t, acc_t>(grad_out, nb_idx, n, N, tk, d, D, tx, smem_go[tk]);
            __syncthreads();

            acc_t dts = static_cast<acc_t>(dt_s[d]);
            acc_t dtc = static_cast<acc_t>(dt_c[d]);

            acc_t up    = smem_u[tk][center];
            acc_t u_pos = smem_u[tk][idx_pos];
            acc_t u_neg = smem_u[tk][idx_neg];
            acc_t u_kp  = smem_u[kp][center];
            acc_t u_km  = smem_u[km][center];

            acc_t gop    = smem_go[tk][center];
            acc_t go_pos = smem_go[tk][idx_pos];
            acc_t go_neg = smem_go[tk][idx_neg];
            acc_t go_kp  = smem_go[kp][center];
            acc_t go_km  = smem_go[km][center];

            // 1. Self-adjoint grad_u (ZERO atomics!)
            acc_t sp_go  = g_pos * (go_pos - gop) + g_neg * (go_neg - gop);
            acc_t ang_go = c_fwd * (go_kp - gop)  + c_bwd * (go_km - gop);
            grad_u[u_base + d] = static_cast<scalar_t>(gop + dts * sp_go + dtc * ang_go);

            // 2. Exact grad_a & grad_c accumulated in registers across D (ZERO atomics!)
            ga_acc += dts * (a_pos * (gop - go_pos) * (u_pos - up) + a_neg * (gop - go_neg) * (u_neg - up));
            gc_acc += dtc * (gop - go_kp) * (u_kp - up);

            // 3. Exact grad_dt_s & grad_dt_c reduced within block
            acc_t v_s = gop * (g_pos * (u_pos - up) + g_neg * (u_neg - up));
            acc_t v_c = gop * (c_fwd * (u_kp - up)  + c_bwd * (u_km - up));
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                v_s += __shfl_down_sync(0xffffffffu, v_s, off);
                v_c += __shfl_down_sync(0xffffffffu, v_c, off);
            }
            if (lane == 0) {
                s_warp_s[warp_id] = v_s;
                s_warp_c[warp_id] = v_c;
            }
            __syncthreads();
            if (lane_id == 0) {
                s_gdts[d] += s_warp_s[0] + s_warp_s[1];
                s_gdtc[d] += s_warp_c[0] + s_warp_c[1];
            }
        }

        // Direct coalesced writes for grad_a & grad_c (ZERO atomics!)
        grad_a[(n * 16 + tx) * 4 + tk] = static_cast<scalar_t>(ga_acc);
        grad_c[(n * 16 + tx) * 4 + tk] = static_cast<scalar_t>(gc_acc);
        __syncthreads();
    }

    __syncthreads();
    if (lane_id < D) {
        atomicAdd(&grad_dt_s[lane_id], s_gdts[lane_id]);
        atomicAdd(&grad_dt_c[lane_id], s_gdtc[lane_id]);
    }
}

// ============================================================
// PyTorch C++ Entrypoints
// ============================================================

std::vector<torch::Tensor> build_bmap_and_neighbors(
    torch::Tensor b_idx, torch::Tensor by_idx, torch::Tensor bx_idx, int64_t B, int64_t G
) {
    b_idx = b_idx.contiguous(); by_idx = by_idx.contiguous(); bx_idx = bx_idx.contiguous();
    int32_t N = static_cast<int32_t>(b_idx.size(0));
    auto bmap = torch::full({B, G, G}, -1, b_idx.options());
    auto nb_idx = torch::empty({8, N}, b_idx.options());
    if (N == 0) return {bmap, nb_idx};

    dim3 block(256);
    dim3 grid((N + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    fill_bmap_kernel<<<grid, block, 0, stream>>>(
        b_idx.data_ptr<int32_t>(), by_idx.data_ptr<int32_t>(), bx_idx.data_ptr<int32_t>(),
        bmap.data_ptr<int32_t>(), N, static_cast<int32_t>(G)
    );
    build_nb_kernel<<<grid, block, 0, stream>>>(
        b_idx.data_ptr<int32_t>(), by_idx.data_ptr<int32_t>(), bx_idx.data_ptr<int32_t>(),
        bmap.data_ptr<int32_t>(), nb_idx.data_ptr<int32_t>(), N, static_cast<int32_t>(G)
    );
    return {bmap, nb_idx};
}

torch::Tensor gather_bricks_nhwc(
    torch::Tensor dense, torch::Tensor b_idx, torch::Tensor by_idx, torch::Tensor bx_idx, int64_t G
) {
    dense = dense.contiguous();
    b_idx = b_idx.contiguous(); by_idx = by_idx.contiguous(); bx_idx = bx_idx.contiguous();
    int64_t N = b_idx.size(0), C = dense.size(1);
    auto bricks = torch::empty({N, 4, 4, C}, dense.options());
    if (N == 0) return bricks;

    int64_t total = N * 16 * C;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(dense.scalar_type(), "gather_bricks_nhwc", ([&] {
        gather_bricks_nhwc_kernel<scalar_t><<<grid, block, 0, stream>>>(
            dense.data_ptr<scalar_t>(), b_idx.data_ptr<int32_t>(),
            by_idx.data_ptr<int32_t>(), bx_idx.data_ptr<int32_t>(),
            bricks.data_ptr<scalar_t>(), N, C, G
        );
    }));
    return bricks;
}

std::vector<torch::Tensor> scatter_bricks_nhwc_fwd(
    torch::Tensor bricks, torch::Tensor b_idx, torch::Tensor by_idx, torch::Tensor bx_idx,
    int64_t B, int64_t G
) {
    bricks = bricks.contiguous();
    b_idx = b_idx.contiguous(); by_idx = by_idx.contiguous(); bx_idx = bx_idx.contiguous();
    int64_t N = bricks.size(0), C = bricks.size(3), H = G * 4;
    auto dense = torch::zeros({B, C, H, H}, bricks.options());
    auto mask = torch::zeros({B, 1, H, H}, bricks.options());
    if (N == 0) return {dense, mask};

    int64_t total = N * 16;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bricks.scalar_type(), "scatter_bricks_nhwc_fwd", ([&] {
        scatter_bricks_nhwc_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            bricks.data_ptr<scalar_t>(), b_idx.data_ptr<int32_t>(),
            by_idx.data_ptr<int32_t>(), bx_idx.data_ptr<int32_t>(),
            dense.data_ptr<scalar_t>(), mask.data_ptr<scalar_t>(), N, C, G
        );
    }));
    return {dense, mask};
}

torch::Tensor scatter_bricks_nhwc_bwd(
    torch::Tensor grad_dense, torch::Tensor b_idx, torch::Tensor by_idx, torch::Tensor bx_idx, int64_t G
) {
    return gather_bricks_nhwc(grad_dense, b_idx, by_idx, bx_idx, G);
}

torch::Tensor prolong_bricks_nhwc_fwd(torch::Tensor parent) {
    parent = parent.contiguous();
    int64_t M = parent.size(0), C = parent.size(3);
    auto children = torch::empty({M * 4, 4, 4, C}, parent.options());
    if (M == 0) return children;

    int64_t total = M * 64 * C;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(parent.scalar_type(), "prolong_bricks_nhwc_fwd", ([&] {
        prolong_bricks_nhwc_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            parent.data_ptr<scalar_t>(), children.data_ptr<scalar_t>(), M, C
        );
    }));
    return children;
}

torch::Tensor prolong_bricks_nhwc_bwd(torch::Tensor grad_children) {
    grad_children = grad_children.contiguous();
    int64_t M = grad_children.size(0) / 4, C = grad_children.size(3);
    auto grad_parent = torch::empty({M, 4, 4, C}, grad_children.options());
    if (M == 0) return grad_parent;

    int64_t total = M * 16 * C;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_children.scalar_type(), "prolong_bricks_nhwc_bwd", ([&] {
        prolong_bricks_nhwc_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_children.data_ptr<scalar_t>(), grad_parent.data_ptr<scalar_t>(), M, C
        );
    }));
    return grad_parent;
}

torch::Tensor restrict_bricks_nhwc_fwd(torch::Tensor children) {
    children = children.contiguous();
    int64_t M = children.size(0) / 4, C = children.size(3);
    auto parent = torch::empty({M, 4, 4, C}, children.options());
    if (M == 0) return parent;

    int64_t total = M * 16 * C;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(children.scalar_type(), "restrict_bricks_nhwc_fwd", ([&] {
        restrict_bricks_nhwc_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            children.data_ptr<scalar_t>(), parent.data_ptr<scalar_t>(), M, C
        );
    }));
    return parent;
}

torch::Tensor restrict_bricks_nhwc_bwd(torch::Tensor grad_parent) {
    grad_parent = grad_parent.contiguous();
    int64_t M = grad_parent.size(0), C = grad_parent.size(3);
    auto grad_children = torch::empty({M * 4, 4, 4, C}, grad_parent.options());
    if (M == 0) return grad_children;

    int64_t total = M * 64 * C;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_parent.scalar_type(), "restrict_bricks_nhwc_bwd", ([&] {
        restrict_bricks_nhwc_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_parent.data_ptr<scalar_t>(), grad_children.data_ptr<scalar_t>(), M, C
        );
    }));
    return grad_children;
}

torch::Tensor fused_se2_laplacian_fwd(
    torch::Tensor u, torch::Tensor a, torch::Tensor c,
    torch::Tensor dt_s, torch::Tensor dt_c, torch::Tensor nb_idx
) {
    u = u.contiguous(); a = a.contiguous(); c = c.contiguous();
    dt_s = dt_s.contiguous(); dt_c = dt_c.contiguous(); nb_idx = nb_idx.contiguous();
    int64_t N = u.size(0), D = u.size(4);
    TORCH_CHECK(D <= MAX_FIBER_D, "fused_se2_laplacian supports fiber_dim D <= 64");
    auto out = torch::empty_like(u);
    if (N == 0) return out;

    dim3 block(16, 4);
    dim3 grid(N, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "fused_se2_laplacian_fwd", ([&] {
        fused_se2_laplacian_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            u.data_ptr<scalar_t>(), a.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(),
            dt_s.data_ptr<scalar_t>(), dt_c.data_ptr<scalar_t>(),
            nb_idx.data_ptr<int32_t>(), out.data_ptr<scalar_t>(), N, D
        );
    }));
    return out;
}

std::vector<torch::Tensor> fused_se2_laplacian_bwd(
    torch::Tensor grad_out, torch::Tensor u, torch::Tensor a, torch::Tensor c,
    torch::Tensor dt_s, torch::Tensor dt_c, torch::Tensor nb_idx
) {
    grad_out = grad_out.contiguous(); u = u.contiguous();
    a = a.contiguous(); c = c.contiguous();
    dt_s = dt_s.contiguous(); dt_c = dt_c.contiguous(); nb_idx = nb_idx.contiguous();
    int64_t N = u.size(0), D = u.size(4);
    TORCH_CHECK(D <= MAX_FIBER_D, "fused_se2_laplacian supports fiber_dim D <= 64");

    auto grad_u = torch::empty_like(u);
    auto grad_a = torch::empty_like(a);
    auto grad_c = torch::empty_like(c);
    auto acc_dtype = (u.scalar_type() == torch::kFloat64) ? torch::kFloat64 : torch::kFloat32;
    auto grad_dts = torch::zeros({D}, dt_s.options().dtype(acc_dtype));
    auto grad_dtc = torch::zeros({D}, dt_c.options().dtype(acc_dtype));
    if (N == 0) {
        return {grad_u, grad_a, grad_c, grad_dts.to(dt_s.scalar_type()), grad_dtc.to(dt_c.scalar_type())};
    }

    int64_t grid_x = std::min<int64_t>(N, 256);
    dim3 block(16, 4);
    dim3 grid(grid_x, 1);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "fused_se2_laplacian_bwd", ([&] {
        using acc_t = typename AccType<scalar_t>::type;
        fused_se2_laplacian_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(),
            a.data_ptr<scalar_t>(), c.data_ptr<scalar_t>(),
            dt_s.data_ptr<scalar_t>(), dt_c.data_ptr<scalar_t>(),
            nb_idx.data_ptr<int32_t>(),
            grad_u.data_ptr<scalar_t>(), grad_a.data_ptr<scalar_t>(), grad_c.data_ptr<scalar_t>(),
            grad_dts.data_ptr<acc_t>(), grad_dtc.data_ptr<acc_t>(), N, D
        );
    }));
    return {grad_u, grad_a, grad_c, grad_dts.to(dt_s.scalar_type()), grad_dtc.to(dt_c.scalar_type())};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("build_bmap_and_neighbors", &build_bmap_and_neighbors, "1-Shot int32 Brick Map & 8-Neighbor Builder");
    m.def("gather_bricks_nhwc", &gather_bricks_nhwc, "Coalesced NHWC 4x4 Brick Gather");
    m.def("scatter_bricks_nhwc_fwd", &scatter_bricks_nhwc_fwd, "Coalesced NHWC 4x4 Brick Scatter Forward");
    m.def("scatter_bricks_nhwc_bwd", &scatter_bricks_nhwc_bwd, "Coalesced NHWC 4x4 Brick Scatter Backward");
    m.def("prolong_bricks_nhwc_fwd", &prolong_bricks_nhwc_fwd, "Bilinear 2x NHWC Prolongation Forward");
    m.def("prolong_bricks_nhwc_bwd", &prolong_bricks_nhwc_bwd, "Bilinear 2x NHWC Prolongation Backward");
    m.def("restrict_bricks_nhwc_fwd", &restrict_bricks_nhwc_fwd, "2x2 Avg NHWC Restriction Forward");
    m.def("restrict_bricks_nhwc_bwd", &restrict_bricks_nhwc_bwd, "2x2 Avg NHWC Restriction Backward");
    m.def("fused_se2_laplacian_fwd", &fused_se2_laplacian_fwd, "Zero-Bank-Conflict SE(2) Graph Laplacian Forward");
    m.def("fused_se2_laplacian_bwd", &fused_se2_laplacian_bwd, "Zero-Atomic SE(2) Graph Laplacian Backward");
}
