# haizklow/ops.py
# PyTorch Autograd bindings for csrc/haizklow_cuda.cu (auto-compiles with nvcc, cached).

import os
import torch
from torch.utils.cpp_extension import load

_EXT = None


def get_cuda_ext():
    global _EXT
    if _EXT is None and torch.cuda.is_available():
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc", "haizklow_cuda.cu")
        _EXT = load(
            name="haizklow_cuda_ext_v2",
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
        nb = nb_idx.to(torch.int32)
        ctx.save_for_backward(u, w, nb)
        ctx.w_dtype = weight.dtype
        ctx.b_dtype = bias.dtype
        return ext.fused_halo_dw3x3_fwd(u, w, b, nb)

    @staticmethod
    def backward(ctx, grad_out):
        u, w, nb = ctx.saved_tensors
        ext = get_cuda_ext()
        gu, gw, gb = ext.fused_halo_dw3x3_bwd(grad_out.to(u.dtype), u, w, nb)
        return gu, gw.to(ctx.w_dtype), gb.to(ctx.b_dtype), None


class _ScatterBricksFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bricks, b_idx, by_idx, bx_idx, B, G):
        ext = get_cuda_ext()
        bi, byi, bxi = b_idx.to(torch.int32), by_idx.to(torch.int32), bx_idx.to(torch.int32)
        ctx.save_for_backward(bi, byi, bxi)
        ctx.G = G
        dense, mask = ext.scatter_bricks_fwd(bricks, bi, byi, bxi, B, G)
        ctx.mark_non_differentiable(mask)
        return dense, mask

    @staticmethod
    def backward(ctx, grad_dense, _grad_mask):
        bi, byi, bxi = ctx.saved_tensors
        ext = get_cuda_ext()
        gb = ext.scatter_bricks_bwd(grad_dense, bi, byi, bxi, ctx.G)
        return gb, None, None, None, None, None


def build_bmap_and_neighbors_cuda(b_idx, by_idx, bx_idx, B: int, G: int):
    """1-shot CUDA int32 brick_map [B, G, G] + 8-neighbor table [8, N]."""
    ext = get_cuda_ext()
    return ext.build_bmap_and_neighbors(
        b_idx.to(torch.int32), by_idx.to(torch.int32), bx_idx.to(torch.int32), int(B), int(G)
    )


def fused_halo_dw3x3(u, weight, bias, nb_idx):
    """Zero-bank-conflict SRAM 4x4 Morton Halo + Depthwise 3x3 (grid-stride backward)."""
    return _FusedHaloDW3x3Fn.apply(u, weight, bias, nb_idx)


def scatter_bricks_cuda(bricks, b_idx, by_idx, bx_idx, B: int, G: int):
    """Coalesced 4x4 brick scatter to [B, C, G*4, G*4] + binary mask."""
    return _ScatterBricksFn.apply(bricks, b_idx, by_idx, bx_idx, int(B), int(G))
