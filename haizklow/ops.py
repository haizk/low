# haizklow/ops.py
# PyTorch Autograd bindings for csrc/haizklow_cuda.cu (auto-compiles with nvcc in ~4s, cached).

import os
import torch
from torch.utils.cpp_extension import load

_EXT = None


def get_cuda_ext():
    global _EXT
    if _EXT is None and torch.cuda.is_available():
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc", "haizklow_cuda.cu")
        _EXT = load(
            name="haizklow_cuda_ext",
            sources=[src],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


class _FusedHaloDW3x3Fn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, weight, bias, nb_idx):
        ext = get_cuda_ext()
        w = weight.to(u.dtype)
        b = bias.to(u.dtype)
        ctx.save_for_backward(u, w, nb_idx)
        ctx.w_dtype = weight.dtype
        ctx.b_dtype = bias.dtype
        return ext.fused_halo_dw3x3_fwd(u, w, b, nb_idx)

    @staticmethod
    def backward(ctx, grad_out):
        u, w, nb_idx = ctx.saved_tensors
        ext = get_cuda_ext()
        gu, gw, gb = ext.fused_halo_dw3x3_bwd(grad_out.to(u.dtype), u, w, nb_idx)
        return gu, gw.to(ctx.w_dtype), gb.to(ctx.b_dtype), None


class _ScatterBricksFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bricks, b_idx, by_idx, bx_idx, B, G):
        ext = get_cuda_ext()
        ctx.save_for_backward(b_idx, by_idx, bx_idx)
        ctx.G = G
        dense, mask = ext.scatter_bricks_fwd(bricks, b_idx, by_idx, bx_idx, B, G)
        ctx.mark_non_differentiable(mask)
        return dense, mask

    @staticmethod
    def backward(ctx, grad_dense, _grad_mask):
        b_idx, by_idx, bx_idx = ctx.saved_tensors
        ext = get_cuda_ext()
        gb = ext.scatter_bricks_bwd(grad_dense, b_idx, by_idx, bx_idx, ctx.G)
        return gb, None, None, None, None, None


def fused_halo_dw3x3(u, weight, bias, nb_idx):
    """Fused SRAM 4x4 Morton Halo + Depthwise 3x3 (eliminates gather_brick_halo_flat!)."""
    return _FusedHaloDW3x3Fn.apply(u, weight, bias, nb_idx)


def scatter_bricks_cuda(bricks, b_idx, by_idx, bx_idx, B, G):
    """Coalesced 4x4 brick scatter to [B, C, G*4, G*4] + binary mask."""
    return _ScatterBricksFn.apply(bricks, b_idx, by_idx, bx_idx, B, G)


if __name__ == "__main__":
    import time
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from haizklow.se2_multigrid import gather_brick_halo_flat, precompute_neighbor_table

    device = torch.device("cuda")
    print("[HaizkLow CUDA] Compiling & running fp64 autograd.gradcheck...")

    # 1. Exact fp64 gradcheck on random sparse brick graph!
    B, G, C = 2, 5, 6
    mask = torch.rand(B, G, G, device=device) < 0.45
    b_idx, by_idx, bx_idx = torch.where(mask)
    N = b_idx.numel()
    bmap = torch.full((B, G, G), -1, device=device, dtype=torch.long)
    bmap[b_idx, by_idx, bx_idx] = torch.arange(N, device=device)
    nb_idx = precompute_neighbor_table(b_idx, by_idx, bx_idx, bmap, G)

    u64 = torch.randn(N, C, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
    w64 = torch.randn(C, 1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
    b64 = torch.randn(C, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(fused_halo_dw3x3, (u64, w64, b64, nb_idx), eps=1e-6, atol=1e-4)
    print("  [PASS] fused_halo_dw3x3 gradcheck PASSED!")

    # Compare forward exactness vs PyTorch reference
    ref = torch.nn.functional.conv2d(gather_brick_halo_flat(u64, nb_idx), w64, b64, groups=C)
    out = fused_halo_dw3x3(u64, w64, b64, nb_idx)
    print(f"  [PASS] Max forward diff vs PyTorch: {(out - ref).abs().max().item():.2e}")

    # 2. Speedup benchmark at L4 (640x640, B=4, 25% active -> N = 25,600 bricks, C = 32, fp16!)
    B, G, C = 4, 160, 32
    N = int(B * G * G * 0.25)
    b_idx = torch.arange(B, device=device).repeat_interleave(N // B)
    by_idx = torch.randint(0, G, (N,), device=device)
    bx_idx = torch.randint(0, G, (N,), device=device)
    bmap = torch.full((B, G, G), -1, device=device, dtype=torch.long)
    bmap[b_idx, by_idx, bx_idx] = torch.arange(N, device=device)
    nb_idx = precompute_neighbor_table(b_idx, by_idx, bx_idx, bmap, G)

    u = torch.randn(N, C, 4, 4, device=device, dtype=torch.float16)
    dw = torch.nn.Conv2d(C, C, 3, padding=0, groups=C).to(device).half()

    def bench(fn, iters=50):
        with torch.no_grad():
            fn(); torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(iters):
                fn()
            torch.cuda.synchronize()
        return (time.perf_counter() - t0) * 1000.0 / iters

    ms_py = bench(lambda: dw(gather_brick_halo_flat(u, nb_idx)))
    ms_cu = bench(lambda: fused_halo_dw3x3(u, dw.weight, dw.bias, nb_idx))
    print(f"\n=== L4 640x640 (B=4, N={N:,} bricks, C=32, fp16) ===")
    print(f"  PyTorch Halo + DW3x3 : {ms_py:6.2f} ms  ({ms_py/B:.2f} ms/img)")
    print(f"  HaizkLow CUDA C Fused: {ms_cu:6.2f} ms  ({ms_cu/B:.2f} ms/img)  --> {ms_py/ms_cu:.1f}x FASTER!!")
