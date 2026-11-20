# haizklow/quadtree.py
# HaizkLow — Pointerless Morton Z-Order 4x4 Brick Quadtree Utilities
# Owns all Morton ordering, sparse brick gather/scatter, neighbor tables, and GT split masks.

from dataclasses import dataclass
from typing import List, Optional, Tuple
import torch
import torch.nn.functional as F

from .ops import build_bmap_and_neighbors_cuda, scatter_bricks_cuda


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
    """Precomputes Morton permutation tables for the dense L0 root grid."""
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
    f_raw: torch.Tensor                 # [N, C_pyr, T, T]
    f_emb: torch.Tensor                 # [N, C, T, T]
    state: torch.Tensor                 # [N, C, T, T]


@torch.no_grad()
def build_bmap_and_neighbors(
    b_idx: torch.Tensor, by_idx: torch.Tensor, bx_idx: torch.Tensor, B: int, G: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Builds [B, G, G] int32 brick_map and [8, N] int32 neighbor table (1-shot CUDA or CPU fallback)."""
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
    """Zero-sort Morton expansion: expanding sorted parents into q=0..3 preserves Z-order."""
    M = p_sel.numel()
    device = p_sel.device
    c_parent = p_sel.repeat_interleave(4)
    c_q = torch.arange(4, device=device, dtype=torch.int32).repeat(M)
    c_b = cur.b_idx[c_parent]
    c_bx = cur.bx_idx[c_parent] * 2 + (c_q & 1)
    c_by = cur.by_idx[c_parent] * 2 + ((c_q >> 1) & 1)
    c_morton = (cur.morton[c_parent] << 2) | c_q
    return c_b, c_by, c_bx, c_morton


def gather_bricks(
    dense: torch.Tensor, b_idx: torch.Tensor, by_idx: torch.Tensor, bx_idx: torch.Tensor, tile_size: int = 4
) -> torch.Tensor:
    B, C, H, W = dense.shape
    T = tile_size
    G = H // T
    return dense.view(B, C, G, T, G, T).permute(0, 2, 4, 1, 3, 5)[b_idx.long(), by_idx.long(), bx_idx.long()]


def scatter_bricks(
    bricks: torch.Tensor, b_idx: torch.Tensor, by_idx: torch.Tensor, bx_idx: torch.Tensor,
    B: int, G: int, tile_size: int = 4
) -> Tuple[torch.Tensor, torch.Tensor]:
    if bricks.is_cuda and tile_size == 4:
        return scatter_bricks_cuda(bricks, b_idx.to(torch.int32), by_idx.to(torch.int32), bx_idx.to(torch.int32), B, G)

    N, C, T, _ = bricks.shape
    device = bricks.device
    grid = torch.zeros(B, G, G, C, T, T, device=device, dtype=bricks.dtype)
    mask = torch.zeros(B, G, G, 1, T, T, device=device, dtype=bricks.dtype)
    ones = torch.ones(N, 1, T, T, device=device, dtype=bricks.dtype)

    idx = (b_idx.long(), by_idx.long(), bx_idx.long())
    grid = grid.index_put(idx, bricks)
    mask = mask.index_put(idx, ones)

    lvl_dense = grid.permute(0, 3, 1, 4, 2, 5).reshape(B, C, G * T, G * T)
    lvl_mask = mask.permute(0, 3, 1, 4, 2, 5).reshape(B, 1, G * T, G * T)
    return lvl_dense, lvl_mask


def gather_brick_halo_ref(x: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
    """Pure-PyTorch reference 1px halo gather (used on CPU and in CUDA gradcheck)."""
    N, C, T, _ = x.shape
    nb = nb_idx.long()
    x_z = torch.cat([x, torch.zeros(1, C, T, T, device=x.device, dtype=x.dtype)], dim=0)
    pad = torch.zeros(N, C, T + 2, T + 2, device=x.device, dtype=x.dtype)
    pad[:, :, 1:T + 1, 1:T + 1] = x
    pad[:, :, 0, 1:T + 1] = x_z[nb[0], :, T - 1, :]
    pad[:, :, T + 1, 1:T + 1] = x_z[nb[1], :, 0, :]
    pad[:, :, 1:T + 1, 0] = x_z[nb[2], :, :, T - 1]
    pad[:, :, 1:T + 1, T + 1] = x_z[nb[3], :, :, 0]
    pad[:, :, 0, 0] = x_z[nb[4], :, T - 1, T - 1]
    pad[:, :, 0, T + 1] = x_z[nb[5], :, T - 1, 0]
    pad[:, :, T + 1, 0] = x_z[nb[6], :, 0, T - 1]
    pad[:, :, T + 1, T + 1] = x_z[nb[7], :, 0, 0]
    return pad
