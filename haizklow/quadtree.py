# haizklow/quadtree.py
# HaizkLow v3 — Pointerless Morton Z-Order 4x4 NHWC Brick Quadtree Primitives

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from .ops import (
    build_bmap_and_neighbors_cuda,
    gather_bricks_nhwc_cuda,
    prolong_bricks_nhwc,
    restrict_bricks_nhwc,
    scatter_bricks_nhwc_cuda,
)


def part1by1(n: int) -> int:
    n &= 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def morton2d_encode(bx: int, by: int) -> int:
    return part1by1(bx) | (part1by1(by) << 1)


def make_l0_morton_tables(root_bricks: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    G0 = root_bricks
    codes = [morton2d_encode(r % G0, r // G0) for r in range(G0 * G0)]
    morton_perm = torch.tensor(sorted(range(G0 * G0), key=lambda r: codes[r]), dtype=torch.long)
    inv_perm = torch.argsort(morton_perm)
    sorted_codes = torch.tensor([codes[r] for r in morton_perm.tolist()], dtype=torch.int32)
    l0_by = (morton_perm // G0).to(torch.int32)
    l0_bx = (morton_perm % G0).to(torch.int32)
    return morton_perm, inv_perm, l0_by, l0_bx, sorted_codes


def batch_gt_split_masks(target: torch.Tensor, tile_size: int = 4, root_bricks: int = 8) -> List[torch.Tensor]:
    """Returns GT active brick masks [L0, L1, ..., L_finest] for quadtree split supervision."""
    fg = (target.sum(dim=1, keepdim=True) > 0.5).float()
    cur = F.max_pool2d(fg, tile_size, tile_size)
    rev = [cur.squeeze(1)]
    while cur.shape[-1] > root_bricks:
        cur = F.max_pool2d(cur, 2, 2)
        rev.append(cur.squeeze(1))
    return list(reversed(rev))


@dataclass
class QuadtreeLevel:
    grid_size: int
    b_idx: torch.Tensor                 # [N] int32
    by_idx: torch.Tensor                # [N] int32
    bx_idx: torch.Tensor                # [N] int32
    morton: torch.Tensor                # [N] int32
    brick_map: torch.Tensor             # [B, G, G] int32 (-1 if unrefined)
    nb_idx: torch.Tensor                # [8, N] int32 (N if missing neighbor)
    parent_sel: Optional[torch.Tensor]  # [M] int64 indices into parent level
    f_raw: torch.Tensor                 # [N, 4, 4, 20]
    f_emb: torch.Tensor                 # [N, 4, 4, 4, D]
    f_gate_bias: torch.Tensor           # [N, 4, 4, 4, 2]
    state: torch.Tensor                 # [N, 4, 4, 4, D]


@torch.no_grad()
def build_bmap_and_neighbors(
    b_idx: torch.Tensor, by_idx: torch.Tensor, bx_idx: torch.Tensor, B: int, G: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    b_i32 = b_idx.to(torch.int32)
    by_i32 = by_idx.to(torch.int32)
    bx_i32 = bx_idx.to(torch.int32)
    if b_i32.is_cuda:
        return build_bmap_and_neighbors_cuda(b_i32, by_i32, bx_i32, B, G)

    N = b_i32.numel()
    b_l, by_l, bx_l = b_i32.long(), by_i32.long(), bx_i32.long()
    bmap = torch.full((B, G, G), -1, device=b_i32.device, dtype=torch.int32)
    bmap[b_l, by_l, bx_l] = torch.arange(N, device=b_i32.device, dtype=torch.int32)

    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    neg1 = torch.full((N,), -1, device=b_i32.device, dtype=torch.int32)
    sent = torch.full((N,), N, device=b_i32.device, dtype=torch.int32)
    nb_list = []
    for dy, dx in offsets:
        ny, nx = by_l + dy, bx_l + dx
        valid = (ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)
        idx = torch.where(valid, bmap[b_l, ny.clamp(0, G - 1), nx.clamp(0, G - 1)], neg1)
        nb_list.append(torch.where(idx >= 0, idx, sent))
    return bmap, torch.stack(nb_list, dim=0)


def expand_morton_children(
    cur: QuadtreeLevel, p_sel: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    M = p_sel.numel()
    device = p_sel.device
    c_parent = p_sel.repeat_interleave(4)
    c_q = torch.arange(4, device=device, dtype=torch.int32).repeat(M)
    c_b = cur.b_idx[c_parent]
    c_bx = cur.bx_idx[c_parent] * 2 + (c_q & 1)
    c_by = cur.by_idx[c_parent] * 2 + ((c_q >> 1) & 1)
    c_morton = (cur.morton[c_parent] << 2) | c_q
    return c_b, c_by, c_bx, c_morton


def gather_bricks_nhwc(
    dense: torch.Tensor, b_idx: torch.Tensor, by_idx: torch.Tensor, bx_idx: torch.Tensor, tile_size: int = 4
) -> torch.Tensor:
    B, C, H, W = dense.shape
    T = tile_size
    G = H // T
    if dense.is_cuda and T == 4:
        return gather_bricks_nhwc_cuda(dense, b_idx, by_idx, bx_idx, G)
    return dense.view(B, C, G, T, G, T).permute(0, 2, 4, 3, 5, 1)[b_idx.long(), by_idx.long(), bx_idx.long()].contiguous()


def scatter_bricks_nhwc(
    bricks: torch.Tensor, b_idx: torch.Tensor, by_idx: torch.Tensor, bx_idx: torch.Tensor,
    B: int, G: int, tile_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bricks.is_cuda and tile_size == 4:
        return scatter_bricks_nhwc_cuda(bricks, b_idx, by_idx, bx_idx, B, G)

    N, T, _, C = bricks.shape
    device = bricks.device
    grid = torch.zeros(B, G, G, T, T, C, device=device, dtype=bricks.dtype)
    mask = torch.zeros(B, G, G, T, T, 1, device=device, dtype=bricks.dtype)
    ones = torch.ones(N, T, T, 1, device=device, dtype=bricks.dtype)
    idx = (b_idx.long(), by_idx.long(), bx_idx.long())
    grid = grid.index_put(idx, bricks)
    mask = mask.index_put(idx, ones)
    lvl_dense = grid.permute(0, 5, 1, 3, 2, 4).reshape(B, C, G * T, G * T)
    lvl_mask = mask.permute(0, 5, 1, 3, 2, 4).reshape(B, 1, G * T, G * T)
    return lvl_dense, lvl_mask


def prolong_bricks_ref(parent: torch.Tensor) -> torch.Tensor:
    orig_tail = parent.shape[3:]
    M = parent.shape[0]
    C = parent.numel() // (M * 16)
    p = parent.reshape(M, 4, 4, C).permute(0, 3, 1, 2)
    up = F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False)
    return up.view(M, C, 2, 4, 2, 4).permute(0, 2, 4, 3, 5, 1).reshape(M * 4, 4, 4, *orig_tail)


def restrict_bricks_ref(children: torch.Tensor) -> torch.Tensor:
    orig_tail = children.shape[3:]
    M = children.shape[0] // 4
    C = children.numel() // (M * 4 * 16)
    ch = children.reshape(M, 2, 2, 4, 4, C).permute(0, 5, 1, 3, 2, 4).reshape(M, C, 8, 8)
    return F.avg_pool2d(ch, 2, 2).permute(0, 2, 3, 1).reshape(M, 4, 4, *orig_tail)


def prolong_bricks(parent: torch.Tensor) -> torch.Tensor:
    if parent.is_cuda and parent.shape[1] == 4 and parent.shape[2] == 4:
        tail = parent.shape[3:]
        M = parent.shape[0]
        C = parent.numel() // (M * 16)
        return prolong_bricks_nhwc(parent.reshape(M, 4, 4, C)).reshape(M * 4, 4, 4, *tail)
    return prolong_bricks_ref(parent)


def restrict_bricks(children: torch.Tensor) -> torch.Tensor:
    if children.is_cuda and children.shape[1] == 4 and children.shape[2] == 4:
        tail = children.shape[3:]
        M = children.shape[0] // 4
        C = children.numel() // (M * 4 * 16)
        return restrict_bricks_nhwc(children.reshape(M * 4, 4, 4, C)).reshape(M, 4, 4, *tail)
    return restrict_bricks_ref(children)


def _pad_nhwc_halo_ref(x: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
    N = x.shape[0]
    tail = x.shape[3:]
    nb = nb_idx.long()
    x_z = torch.cat([x, torch.zeros((1, 4, 4, *tail), device=x.device, dtype=x.dtype)], dim=0)
    pad = torch.zeros((N, 6, 6, *tail), device=x.device, dtype=x.dtype)
    pad[:, 1:5, 1:5] = x
    pad[:, 0, 1:5] = x_z[nb[0], 3, :]
    pad[:, 5, 1:5] = x_z[nb[1], 0, :]
    pad[:, 1:5, 0] = x_z[nb[2], :, 3]
    pad[:, 1:5, 5] = x_z[nb[3], :, 0]
    pad[:, 0, 0] = x_z[nb[4], 3, 3]
    pad[:, 0, 5] = x_z[nb[5], 3, 0]
    pad[:, 5, 0] = x_z[nb[6], 0, 3]
    pad[:, 5, 5] = x_z[nb[7], 0, 0]
    return pad


def fused_se2_laplacian_ref(
    u: torch.Tensor, a: torch.Tensor, c: torch.Tensor,
    dt_s: torch.Tensor, dt_c: torch.Tensor, nb_idx: torch.Tensor
) -> torch.Tensor:
    u_pad = _pad_nhwc_halo_ref(u, nb_idx)
    a_pad = _pad_nhwc_halo_ref(a, nb_idx)
    dirs = ((0, 1), (-1, 1), (1, 0), (1, 1))
    outs = []
    dts = dt_s.view(1, 1, 1, -1)
    dtc = dt_c.view(1, 1, 1, -1)
    for k, (dy, dx) in enumerate(dirs):
        ap = a_pad[:, 1:5, 1:5, k]
        a_pos = a_pad[:, 1 + dy:5 + dy, 1 + dx:5 + dx, k]
        a_neg = a_pad[:, 1 - dy:5 - dy, 1 - dx:5 - dx, k]
        g_pos = (ap * a_pos).unsqueeze(-1)
        g_neg = (ap * a_neg).unsqueeze(-1)

        up = u[:, :, :, k, :]
        u_pos = u_pad[:, 1 + dy:5 + dy, 1 + dx:5 + dx, k, :]
        u_neg = u_pad[:, 1 - dy:5 - dy, 1 - dx:5 - dx, k, :]
        sp_flux = g_pos * (u_pos - up) + g_neg * (u_neg - up)

        kp = (k + 1) & 3
        km = (k + 3) & 3
        ang_flux = c[:, :, :, k:k + 1] * (u[:, :, :, kp, :] - up) + c[:, :, :, km:km + 1] * (u[:, :, :, km, :] - up)
        outs.append(up + dts * sp_flux + dtc * ang_flux)
    return torch.stack(outs, dim=3)
