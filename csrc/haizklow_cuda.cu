// csrc/haizklow_cuda.cu
// HaizkLow Phase 2 — Fused Morton 4x4 Brick Halo + Depthwise 3x3 & Dense Scatter CUDA Kernels
// Auto-compiled on Kaggle T4 via torch.utils.cpp_extension.load() (~4s cached).

#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <ATen/cuda/CUDAContext.h>
#include <type_traits>

template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Load 4x4 brick + 1px Morton halo (36 pixels) into 1D shared memory row s[36]
// Thread layout: tx in 0..15 (16 pixels of 4x4 brick), ty in 0..7 (8 channels/block)
template <typename scalar_t, typename acc_t>
__device__ __forceinline__ void load_6x6_tile(
    const scalar_t* __restrict__ x,
    const int64_t* __restrict__ nb_idx,
    int64_t n, int64_t N, int64_t c, int64_t C,
    int tx, acc_t* s
) {
    // 1. Interior 4x4 (16 coalesced loads)
    int py = tx >> 2;
    int px = tx & 3;
    s[(py + 1) * 6 + (px + 1)] = static_cast<acc_t>(x[(n * C + c) * 16 + tx]);

    // 2. Four 4-pixel halo edges (0:N, 1:S, 2:W, 3:E)
    if (tx < 4) {
        int64_t nb = nb_idx[0 * N + n];
        s[0 * 6 + (tx + 1)] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + 12 + tx]) : acc_t(0);
    } else if (tx < 8) {
        int k = tx - 4;
        int64_t nb = nb_idx[1 * N + n];
        s[5 * 6 + (k + 1)] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + k]) : acc_t(0);
    } else if (tx < 12) {
        int k = tx - 8;
        int64_t nb = nb_idx[2 * N + n];
        s[(k + 1) * 6 + 0] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + k * 4 + 3]) : acc_t(0);
    } else {
        int k = tx - 12;
        int64_t nb = nb_idx[3 * N + n];
        s[(k + 1) * 6 + 5] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + k * 4 + 0]) : acc_t(0);
    }

    // 3. Four 1-pixel halo corners (4:NW, 5:NE, 6:SW, 7:SE)
    if (tx == 0) {
        int64_t nb = nb_idx[4 * N + n];
        s[0] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + 15]) : acc_t(0);
    } else if (tx == 1) {
        int64_t nb = nb_idx[5 * N + n];
        s[5] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + 12]) : acc_t(0);
    } else if (tx == 2) {
        int64_t nb = nb_idx[6 * N + n];
        s[30] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + 3]) : acc_t(0);
    } else if (tx == 3) {
        int64_t nb = nb_idx[7 * N + n];
        s[35] = (nb < N) ? static_cast<acc_t>(x[(nb * C + c) * 16 + 0]) : acc_t(0);
    }
}

template <typename scalar_t>
__global__ void fused_halo_dw3x3_fwd_kernel(
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ w,
    const scalar_t* __restrict__ b,
    const int64_t* __restrict__ nb_idx,
    scalar_t* __restrict__ out,
    int64_t N, int64_t C
) {
    using acc_t = typename AccType<scalar_t>::type;
    __shared__ acc_t smem[8][36];

    int64_t n = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int64_t c = (int64_t)blockIdx.y * 8 + ty;
    bool valid = (n < N) && (c < C);

    if (valid) {
        load_6x6_tile<scalar_t, acc_t>(u, nb_idx, n, N, c, C, tx, smem[ty]);
    }
    __syncthreads();

    if (valid) {
        int py = tx >> 2;
        int px = tx & 3;
        const acc_t* s = smem[ty];
        const scalar_t* wc = w + c * 9;
        acc_t acc = static_cast<acc_t>(b[c]);
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                acc += s[(py + ky) * 6 + (px + kx)] * static_cast<acc_t>(wc[ky * 3 + kx]);
            }
        }
        out[(n * C + c) * 16 + tx] = static_cast<scalar_t>(acc);
    }
}

// Backward pass:
//   - grad_u: ZERO atomics! Uses exact symmetric nb_idx + 180-deg flipped 3x3 kernel!
//   - grad_w, grad_b: 16-lane sub-warp shuffle reduction -> only 1 atomic per block!
template <typename scalar_t>
__global__ void fused_halo_dw3x3_bwd_kernel(
    const scalar_t* __restrict__ grad_out,
    const scalar_t* __restrict__ u,
    const scalar_t* __restrict__ w,
    const int64_t* __restrict__ nb_idx,
    scalar_t* __restrict__ grad_u,
    typename AccType<scalar_t>::type* __restrict__ grad_w,
    typename AccType<scalar_t>::type* __restrict__ grad_b,
    int64_t N, int64_t C
) {
    using acc_t = typename AccType<scalar_t>::type;
    __shared__ acc_t smem_go[8][36];
    __shared__ acc_t smem_u[8][36];

    int64_t n = blockIdx.x;
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int64_t c = (int64_t)blockIdx.y * 8 + ty;
    bool valid = (n < N) && (c < C);

    if (valid) {
        load_6x6_tile<scalar_t, acc_t>(grad_out, nb_idx, n, N, c, C, tx, smem_go[ty]);
        load_6x6_tile<scalar_t, acc_t>(u, nb_idx, n, N, c, C, tx, smem_u[ty]);
    }
    __syncthreads();

    int py = tx >> 2;
    int px = tx & 3;

    // 1. Zero-atomic grad_u (convolve grad_out halo with 180-deg flipped w)
    if (valid) {
        const acc_t* sgo = smem_go[ty];
        const scalar_t* wc = w + c * 9;
        acc_t acc_gu = acc_t(0);
        #pragma unroll
        for (int ky = 0; ky < 3; ++ky) {
            #pragma unroll
            for (int kx = 0; kx < 3; ++kx) {
                acc_gu += sgo[(py + ky) * 6 + (px + kx)] * static_cast<acc_t>(wc[(2 - ky) * 3 + (2 - kx)]);
            }
        }
        grad_u[(n * C + c) * 16 + tx] = static_cast<scalar_t>(acc_gu);
    }

    // 2. Sub-warp (16-lane) shuffle reduction for grad_w & grad_b
    acc_t go_val = valid ? smem_go[ty][(py + 1) * 6 + (px + 1)] : acc_t(0);
    acc_t db = go_val;
    #pragma unroll
    for (int off = 8; off > 0; off >>= 1) {
        db += __shfl_down_sync(0xffffffffu, db, off, 16);
    }
    if (valid && tx == 0) {
        atomicAdd(&grad_b[c], db);
    }

    #pragma unroll
    for (int ky = 0; ky < 3; ++ky) {
        #pragma unroll
        for (int kx = 0; kx < 3; ++kx) {
            acc_t dw = valid ? (go_val * smem_u[ty][(py + ky) * 6 + (px + kx)]) : acc_t(0);
            #pragma unroll
            for (int off = 8; off > 0; off >>= 1) {
                dw += __shfl_down_sync(0xffffffffu, dw, off, 16);
            }
            if (valid && tx == 0) {
                atomicAdd(&grad_w[c * 9 + ky * 3 + kx], dw);
            }
        }
    }
}

// Fast coalesced 4x4 brick scatter (replaces slow 6D PyTorch index_put!)
template <typename scalar_t>
__global__ void scatter_bricks_fwd_kernel(
    const scalar_t* __restrict__ bricks,
    const int64_t* __restrict__ b_idx,
    const int64_t* __restrict__ by_idx,
    const int64_t* __restrict__ bx_idx,
    scalar_t* __restrict__ dense,
    scalar_t* __restrict__ mask,
    int64_t N, int64_t C, int64_t G
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = tid >> 4;
    if (n >= N) return;

    int p = tid & 15;
    int py = p >> 2;
    int px = p & 3;
    int64_t b = b_idx[n];
    int64_t y = by_idx[n] * 4 + py;
    int64_t x = bx_idx[n] * 4 + px;
    int64_t H = G * 4;
    int64_t hw = H * H;

    for (int64_t c = 0; c < C; ++c) {
        dense[(b * C + c) * hw + y * H + x] = bricks[(n * C + c) * 16 + p];
    }
    mask[b * hw + y * H + x] = static_cast<scalar_t>(1);
}

template <typename scalar_t>
__global__ void scatter_bricks_bwd_kernel(
    const scalar_t* __restrict__ grad_dense,
    const int64_t* __restrict__ b_idx,
    const int64_t* __restrict__ by_idx,
    const int64_t* __restrict__ bx_idx,
    scalar_t* __restrict__ grad_bricks,
    int64_t N, int64_t C, int64_t G
) {
    int64_t tid = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t n = tid >> 4;
    if (n >= N) return;

    int p = tid & 15;
    int py = p >> 2;
    int px = p & 3;
    int64_t b = b_idx[n];
    int64_t y = by_idx[n] * 4 + py;
    int64_t x = bx_idx[n] * 4 + px;
    int64_t H = G * 4;
    int64_t hw = H * H;

    for (int64_t c = 0; c < C; ++c) {
        grad_bricks[(n * C + c) * 16 + p] = grad_dense[(b * C + c) * hw + y * H + x];
    }
}

// ---- PyTorch C++ Entrypoints ----

torch::Tensor fused_halo_dw3x3_fwd(
    torch::Tensor u, torch::Tensor w, torch::Tensor b, torch::Tensor nb_idx
) {
    u = u.contiguous(); w = w.contiguous(); b = b.contiguous(); nb_idx = nb_idx.contiguous();
    int64_t N = u.size(0), C = u.size(1);
    auto out = torch::empty_like(u);
    if (N == 0) return out;

    dim3 block(16, 8);
    dim3 grid(N, (C + 7) / 8);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "fused_halo_dw3x3_fwd", ([&] {
        fused_halo_dw3x3_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            u.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(), b.data_ptr<scalar_t>(),
            nb_idx.data_ptr<int64_t>(), out.data_ptr<scalar_t>(), N, C
        );
    }));
    return out;
}

std::vector<torch::Tensor> fused_halo_dw3x3_bwd(
    torch::Tensor grad_out, torch::Tensor u, torch::Tensor w, torch::Tensor nb_idx
) {
    grad_out = grad_out.contiguous(); u = u.contiguous();
    w = w.contiguous(); nb_idx = nb_idx.contiguous();
    int64_t N = u.size(0), C = u.size(1);
    auto grad_u = torch::empty_like(u);
    auto acc_dtype = (u.scalar_type() == torch::kFloat64) ? torch::kFloat64 : torch::kFloat32;
    auto grad_w = torch::zeros({C, 1, 3, 3}, w.options().dtype(acc_dtype));
    auto grad_b = torch::zeros({C}, w.options().dtype(acc_dtype));
    if (N == 0) return {grad_u, grad_w.to(w.scalar_type()), grad_b.to(w.scalar_type())};

    dim3 block(16, 8);
    dim3 grid(N, (C + 7) / 8);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(u.scalar_type(), "fused_halo_dw3x3_bwd", ([&] {
        using acc_t = typename AccType<scalar_t>::type;
        fused_halo_dw3x3_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_out.data_ptr<scalar_t>(), u.data_ptr<scalar_t>(), w.data_ptr<scalar_t>(),
            nb_idx.data_ptr<int64_t>(), grad_u.data_ptr<scalar_t>(),
            grad_w.data_ptr<acc_t>(), grad_b.data_ptr<acc_t>(), N, C
        );
    }));
    return {grad_u, grad_w.to(w.scalar_type()), grad_b.to(w.scalar_type())};
}

std::vector<torch::Tensor> scatter_bricks_fwd(
    torch::Tensor bricks, torch::Tensor b_idx, torch::Tensor by_idx, torch::Tensor bx_idx,
    int64_t B, int64_t G
) {
    bricks = bricks.contiguous();
    b_idx = b_idx.contiguous(); by_idx = by_idx.contiguous(); bx_idx = bx_idx.contiguous();
    int64_t N = bricks.size(0), C = bricks.size(1), H = G * 4;
    auto dense = torch::zeros({B, C, H, H}, bricks.options());
    auto mask = torch::zeros({B, 1, H, H}, bricks.options());
    if (N == 0) return {dense, mask};

    int64_t total = N * 16;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(bricks.scalar_type(), "scatter_bricks_fwd", ([&] {
        scatter_bricks_fwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            bricks.data_ptr<scalar_t>(), b_idx.data_ptr<int64_t>(),
            by_idx.data_ptr<int64_t>(), bx_idx.data_ptr<int64_t>(),
            dense.data_ptr<scalar_t>(), mask.data_ptr<scalar_t>(), N, C, G
        );
    }));
    return {dense, mask};
}

torch::Tensor scatter_bricks_bwd(
    torch::Tensor grad_dense, torch::Tensor b_idx, torch::Tensor by_idx, torch::Tensor bx_idx,
    int64_t G
) {
    grad_dense = grad_dense.contiguous();
    b_idx = b_idx.contiguous(); by_idx = by_idx.contiguous(); bx_idx = bx_idx.contiguous();
    int64_t N = b_idx.size(0), C = grad_dense.size(1);
    auto grad_bricks = torch::empty({N, C, 4, 4}, grad_dense.options());
    if (N == 0) return grad_bricks;

    int64_t total = N * 16;
    dim3 block(256);
    dim3 grid((total + 255) / 256);
    auto stream = at::cuda::getCurrentCUDAStream();

    AT_DISPATCH_FLOATING_TYPES_AND_HALF(grad_dense.scalar_type(), "scatter_bricks_bwd", ([&] {
        scatter_bricks_bwd_kernel<scalar_t><<<grid, block, 0, stream>>>(
            grad_dense.data_ptr<scalar_t>(), b_idx.data_ptr<int64_t>(),
            by_idx.data_ptr<int64_t>(), bx_idx.data_ptr<int64_t>(),
            grad_bricks.data_ptr<scalar_t>(), N, C, G
        );
    }));
    return grad_bricks;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_halo_dw3x3_fwd", &fused_halo_dw3x3_fwd, "Fused 4x4 Morton Halo + DW3x3 Forward");
    m.def("fused_halo_dw3x3_bwd", &fused_halo_dw3x3_bwd, "Fused 4x4 Morton Halo + DW3x3 Backward");
    m.def("scatter_bricks_fwd", &scatter_bricks_fwd, "Coalesced 4x4 Brick Scatter Forward");
    m.def("scatter_bricks_bwd", &scatter_bricks_bwd, "Coalesced 4x4 Brick Scatter Backward");
}
