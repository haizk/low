# haizklow/ops.py
# HaizkLow v3 — PyTorch Autograd Bindings for NHWC SE(2) Brick CUDA Kernels

import os
import torch
from torch.utils.cpp_extension import load

_EXT = None


def get_cuda_ext():
    global _EXT
    if _EXT is None and torch.cuda.is_available():
        src = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "csrc", "haizklow_cuda.cu")
        _EXT = load(
            name="haizklow_cuda_ext_v3",
            sources=[src],
            extra_cuda_cflags=["-O3"],
            verbose=False,
        )
    return _EXT


class _FusedSE2LaplacianFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, u, a, c, dt_s, dt_c, nb_idx):
        ext = get_cuda_ext()
        a_t = a.to(u.dtype)
        c_t = c.to(u.dtype)
        dts_t = dt_s.to(u.dtype)
        dtc_t = dt_c.to(u.dtype)
        nb = nb_idx.to(torch.int32)
        ctx.save_for_backward(u, a_t, c_t, dts_t, dtc_t, nb)
        ctx.dtypes = (a.dtype, c.dtype, dt_s.dtype, dt_c.dtype)
        return ext.fused_se2_laplacian_fwd(u, a_t, c_t, dts_t, dtc_t, nb)

    @staticmethod
    def backward(ctx, grad_out):
        u, a_t, c_t, dts_t, dtc_t, nb = ctx.saved_tensors
        da, dc, ddts, ddtc = ctx.dtypes
        ext = get_cuda_ext()
        gu, ga, gc, gdts, gdtc = ext.fused_se2_laplacian_bwd(
            grad_out.to(u.dtype), u, a_t, c_t, dts_t, dtc_t, nb
        )
        return gu, ga.to(da), gc.to(dc), gdts.to(ddts), gdtc.to(ddtc), None


class _ProlongBricksNHWCFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, parent):
        ext = get_cuda_ext()
        return ext.prolong_bricks_nhwc_fwd(parent)

    @staticmethod
    def backward(ctx, grad_children):
        ext = get_cuda_ext()
        return ext.prolong_bricks_nhwc_bwd(grad_children)


class _RestrictBricksNHWCFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, children):
        ext = get_cuda_ext()
        return ext.restrict_bricks_nhwc_fwd(children)

    @staticmethod
    def backward(ctx, grad_parent):
        ext = get_cuda_ext()
        return ext.restrict_bricks_nhwc_bwd(grad_parent)


class _ScatterBricksNHWCFn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, bricks, b_idx, by_idx, bx_idx, B, G):
        ext = get_cuda_ext()
        bi, byi, bxi = b_idx.to(torch.int32), by_idx.to(torch.int32), bx_idx.to(torch.int32)
        ctx.save_for_backward(bi, byi, bxi)
        ctx.G = G
        dense, mask = ext.scatter_bricks_nhwc_fwd(bricks, bi, byi, bxi, B, G)
        ctx.mark_non_differentiable(mask)
        return dense, mask

    @staticmethod
    def backward(ctx, grad_dense, _grad_mask):
        bi, byi, bxi = ctx.saved_tensors
        ext = get_cuda_ext()
        gb = ext.scatter_bricks_nhwc_bwd(grad_dense, bi, byi, bxi, ctx.G)
        return gb, None, None, None, None, None


def build_bmap_and_neighbors_cuda(b_idx, by_idx, bx_idx, B: int, G: int):
    ext = get_cuda_ext()
    return ext.build_bmap_and_neighbors(
        b_idx.to(torch.int32), by_idx.to(torch.int32), bx_idx.to(torch.int32), int(B), int(G)
    )


def gather_bricks_nhwc_cuda(dense, b_idx, by_idx, bx_idx, G: int):
    ext = get_cuda_ext()
    return ext.gather_bricks_nhwc(
        dense, b_idx.to(torch.int32), by_idx.to(torch.int32), bx_idx.to(torch.int32), int(G)
    )


def scatter_bricks_nhwc_cuda(bricks, b_idx, by_idx, bx_idx, B: int, G: int):
    return _ScatterBricksNHWCFn.apply(bricks, b_idx, by_idx, bx_idx, int(B), int(G))


def fused_se2_laplacian(u, a, c, dt_s, dt_c, nb_idx):
    return _FusedSE2LaplacianFn.apply(u, a, c, dt_s, dt_c, nb_idx)


def prolong_bricks_nhwc(parent):
    return _ProlongBricksNHWCFn.apply(parent)


def restrict_bricks_nhwc(children):
    return _RestrictBricksNHWCFn.apply(children)
