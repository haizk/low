# haizklow/se2_multigrid.py
# HaizkLow — Standalone Adaptive Morton-Quadtree + SE(2) Fiber Multigrid Engine (Phase 1 Fast)
#
# Zero U-Net. Zero Self-Attention. Zero Mamba. Zero permute() HBM copies.
# Conservative symmetric SE(2) conductance g_ij = p_i * p_j + Precomputed Morton neighbor table.

from dataclasses import dataclass
from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mipmap import MultiStatSE2Pyramid


def _part1by1(n: int) -> int:
    n &= 0x0000FFFF
    n = (n | (n << 8)) & 0x00FF00FF
    n = (n | (n << 4)) & 0x0F0F0F0F
    n = (n | (n << 2)) & 0x33333333
    n = (n | (n << 1)) & 0x55555555
    return n


def _morton2d(x: int, y: int) -> int:
    return _part1by1(x) | (_part1by1(y) << 1)


def batch_gt_split_masks(target: torch.Tensor, tile_size: int = 4, root_bricks: int = 8) -> List[torch.Tensor]:
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
    b_idx: torch.Tensor
    by_idx: torch.Tensor
    bx_idx: torch.Tensor
    morton: torch.Tensor
    brick_map: torch.Tensor
    nb_idx: torch.Tensor                # [8, N] precomputed neighbor brick indices (N = sentinel zero)
    parent_sel: Optional[torch.Tensor]
    forcing: torch.Tensor               # [N, 4, 12, 4, 4]
    state: torch.Tensor                 # [N, 4, D, 4, 4]


@torch.no_grad()
def precompute_neighbor_table(b_idx: torch.Tensor, by_idx: torch.Tensor,
                              bx_idx: torch.Tensor, bmap: torch.Tensor, G: int) -> torch.Tensor:
    """Computed ONCE per Quadtree level (eliminates ~2,000 redundant kernel launches!)."""
    N = b_idx.numel()
    offsets = ((-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1))
    nb_list = []
    neg1 = torch.full_like(b_idx, -1)
    sent = torch.full_like(b_idx, N)
    for dy, dx in offsets:
        ny = by_idx + dy
        nx = bx_idx + dx
        valid = (ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)
        idx = torch.where(valid, bmap[b_idx, ny.clamp(0, G - 1), nx.clamp(0, G - 1)], neg1)
        nb_list.append(torch.where(idx >= 0, idx, sent))
    return torch.stack(nb_list, dim=0)  # [8, N]


def gather_brick_halo_fast(x: torch.Tensor, nb_idx: torch.Tensor) -> torch.Tensor:
    """Fast 1px halo gather using precomputed nb_idx[8, N] -> [N, 4, C, T+2, T+2]."""
    N, K, C, T, _ = x.shape
    x_z = torch.cat([x, torch.zeros(1, K, C, T, T, device=x.device, dtype=x.dtype)], dim=0)
    pad = torch.zeros(N, K, C, T + 2, T + 2, device=x.device, dtype=x.dtype)
    pad[:, :, :, 1:T + 1, 1:T + 1] = x
    pad[:, :, :, 0, 1:T + 1] = x_z[nb_idx[0], :, :, T - 1, :]
    pad[:, :, :, T + 1, 1:T + 1] = x_z[nb_idx[1], :, :, 0, :]
    pad[:, :, :, 1:T + 1, 0] = x_z[nb_idx[2], :, :, :, T - 1]
    pad[:, :, :, 1:T + 1, T + 1] = x_z[nb_idx[3], :, :, :, 0]
    pad[:, :, :, 0, 0] = x_z[nb_idx[4], :, :, T - 1, T - 1]
    pad[:, :, :, 0, T + 1] = x_z[nb_idx[5], :, :, T - 1, 0]
    pad[:, :, :, T + 1, 0] = x_z[nb_idx[6], :, :, 0, T - 1]
    pad[:, :, :, T + 1, T + 1] = x_z[nb_idx[7], :, :, 0, 0]
    return pad


class SE2AnisotropicSmoother(nn.Module):
    """
    Zero-copy NCHW C4-shared SE(2) Conservative Anisotropic PDE Smoother.
    Computes permeability (p_i, a_i) ONCE per pixel -> g_ij = p_i * p_j (symmetric conservative flux).
    """
    def __init__(self, d_fiber: int = 16, f_ch: int = 12, hidden: int = 48):
        super().__init__()
        self.d_fiber = d_fiber

        # Single 1x1 conv computes BOTH spatial permeability p and angular permeability a
        self.gate_conv = nn.Conv2d(d_fiber + f_ch, 2 * d_fiber, kernel_size=1)

        # Zero-copy NCHW reaction MLP (GroupNorm(1, C) == LayerNorm across channels!)
        in_dim = 4 * d_fiber + f_ch + 2
        self.norm = nn.GroupNorm(1, in_dim)
        self.react = nn.Sequential(
            nn.Conv2d(in_dim, hidden, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(hidden, d_fiber, kernel_size=1),
        )
        nn.init.zeros_(self.react[-1].weight)
        nn.init.zeros_(self.react[-1].bias)
        self.dt_logit = nn.Parameter(torch.tensor(0.0))

    def _compute_gates(self, u: torch.Tensor, forcing: torch.Tensor):
        N, K, D, Hp, Wp = u.shape
        uf = torch.cat([u, forcing], dim=2).reshape(N * K, D + forcing.shape[2], Hp, Wp)
        gates = torch.sigmoid(self.gate_conv(uf)).view(N, K, 2 * D, Hp, Wp)
        p, a = gates.split(D, dim=2)
        return p, a

    def _step_from_padded(self, u: torch.Tensor, p: torch.Tensor, a: torch.Tensor,
                          up_pad: torch.Tensor, forcing: torch.Tensor,
                          rel_depth: float, is_finest: float) -> torch.Tensor:
        N, K, D, Hp, Wp = u.shape

        # 1. Extract strictly oriented +/- d_k neighbors for [u, p] together (1 halo gather!)
        up_pos = torch.stack([
            up_pad[:, 0, :, 1:-1, 2:  ],  # 0 deg   (East)
            up_pad[:, 1, :, :-2,  2:  ],  # 45 deg  (NE)
            up_pad[:, 2, :, 2:,   1:-1],  # 90 deg  (South)
            up_pad[:, 3, :, 2:,   2:  ],  # 135 deg (SE)
        ], dim=1)
        up_neg = torch.stack([
            up_pad[:, 0, :, 1:-1, :-2 ],  # 0 deg   (West)
            up_pad[:, 1, :, 2:,   :-2 ],  # 45 deg  (SW)
            up_pad[:, 2, :, :-2,  1:-1],  # 90 deg  (North)
            up_pad[:, 3, :, :-2,  :-2 ],  # 135 deg (NW)
        ], dim=1)

        u_pos, p_pos = up_pos.split(D, dim=2)
        u_neg, p_neg = up_neg.split(D, dim=2)

        # 2. Conservative symmetric anisotropic flux g_ij = p_i * p_j along theta_k
        fp = (p * p_pos) * (u_pos - u)
        fn = (p * p_neg) * (u_neg - u)
        diff_flux = fp + fn
        grad_flux = torch.abs(fp - fn)

        # 3. Conservative angular coupling ONLY with k-1 and k+1 (ZERO 0° <-> 90° bleed!)
        u_prev = torch.roll(u, shifts=1, dims=1)
        a_prev = torch.roll(a, shifts=1, dims=1)
        u_next = torch.roll(u, shifts=-1, dims=1)
        a_next = torch.roll(a, shifts=-1, dims=1)
        ang_flux = (a * a_prev) * (u_prev - u) + (a * a_next) * (u_next - u)

        # 4. Resolution-agnostic scale channels in NCHW
        scale_feat = torch.empty(N, K, 2, Hp, Wp, device=u.device, dtype=u.dtype)
        scale_feat[:, :, 0] = rel_depth
        scale_feat[:, :, 1] = is_finest

        # 5. Zero-copy NCHW reaction update
        x = torch.cat([u, diff_flux, grad_flux, ang_flux, forcing, scale_feat], dim=2)
        x = x.reshape(N * K, x.shape[2], Hp, Wp)
        du = self.react(self.norm(x)).view(N, K, D, Hp, Wp)
        dt = 0.5 * torch.sigmoid(self.dt_logit)
        return u + dt * du

    def smooth_bricks(self, u: torch.Tensor, lvl: QuadtreeLevel,
                      rel_depth: float, is_finest: float, steps: int = 1) -> torch.Tensor:
        for _ in range(steps):
            p, a = self._compute_gates(u, lvl.forcing)
            up_pad = gather_brick_halo_fast(torch.cat([u, p], dim=2), lvl.nb_idx)
            u = self._step_from_padded(u, p, a, up_pad, lvl.forcing, rel_depth, is_finest)
        return u

    def smooth_dense(self, u: torch.Tensor, forcing: torch.Tensor,
                     rel_depth: float, is_finest: float, steps: int = 1) -> torch.Tensor:
        for _ in range(steps):
            p, a = self._compute_gates(u, forcing)
            up_pad = F.pad(torch.cat([u, p], dim=2), (1, 1, 1, 1))
            u = self._step_from_padded(u, p, a, up_pad, forcing, rel_depth, is_finest)
        return u


class HaizkLow(nn.Module):
    def __init__(self, d_fiber: int = 16, hidden: int = 48, n_vcycles: int = 2,
                 smooth_steps: int = 1, tile_size: int = 4, root_bricks: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        self.d_fiber = d_fiber
        self.n_vcycles = n_vcycles
        self.smooth_steps = smooth_steps
        self.tile_size = tile_size
        self.root_bricks = root_bricks
        self.use_checkpoint = use_checkpoint

        self.pyramid = MultiStatSE2Pyramid(tile_size=tile_size, root_bricks=root_bricks)
        self.init_proj = nn.Conv2d(12, d_fiber, kernel_size=1)
        self.smoother = SE2AnisotropicSmoother(d_fiber=d_fiber, f_ch=12, hidden=hidden)

        self.restrict_conv = nn.Conv2d(d_fiber, d_fiber, kernel_size=2, stride=2)
        self.prolong_conv = nn.Conv2d(d_fiber, d_fiber, kernel_size=3, padding=1)
        self.rp_scale = nn.Parameter(torch.tensor(0.25))

        # Scale-aware brick split predictor (knows rel_depth so each level learns exact sparsity!)
        self.split_head = nn.Sequential(
            nn.Linear(2 * d_fiber + 12 + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.split_head[-1].bias, -1.0)

        self.out_head = nn.Conv2d(d_fiber, 2, kernel_size=1)
        self.lse_beta = 4.0

        G0 = root_bricks
        codes = [_morton2d(r % G0, r // G0) for r in range(G0 * G0)]
        morton_perm = torch.tensor(sorted(range(G0 * G0), key=lambda r: codes[r]), dtype=torch.long)
        inv_perm = torch.argsort(morton_perm)
        self.register_buffer("l0_morton_perm", morton_perm)
        self.register_buffer("l0_inv_perm", inv_perm)
        self.register_buffer("l0_by", morton_perm // G0)
        self.register_buffer("l0_bx", morton_perm % G0)
        self.register_buffer("l0_codes", torch.arange(G0 * G0, dtype=torch.long))

    def _init_state(self, f: torch.Tensor) -> torch.Tensor:
        N, K, C, T, _ = f.shape
        return self.init_proj(f.reshape(N * K, C, T, T)).view(N, K, self.d_fiber, T, T)

    def _gather_forcing_bricks(self, pyr_l: torch.Tensor, b_idx, by_idx, bx_idx) -> torch.Tensor:
        B, K, C, H, W = pyr_l.shape
        T = self.tile_size
        G = H // T
        return pyr_l.view(B, K, C, G, T, G, T).permute(0, 3, 5, 1, 2, 4, 6)[b_idx, by_idx, bx_idx]

    def _predict_split_logits(self, u: torch.Tensor, forcing: torch.Tensor, rel_depth: float) -> torch.Tensor:
        u_mean = u.mean(dim=(1, 3, 4))
        u_max = u.amax(dim=(1, 3, 4))
        f_max = forcing.amax(dim=(1, 3, 4))
        depth_col = torch.full((u.shape[0], 1), rel_depth, device=u.device, dtype=u.dtype)
        return self.split_head(torch.cat([u_mean, u_max, f_max, depth_col], dim=-1)).squeeze(-1)

    def _prolongate_to_children(self, parent_u: torch.Tensor, p_sel: torch.Tensor) -> torch.Tensor:
        M, K, D, T, _ = parent_u[p_sel].shape
        p = parent_u[p_sel].reshape(M * K, D, T, T)
        up = self.prolong_conv(F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False))
        return up.view(M, K, D, 2, T, 2, T).permute(0, 3, 5, 1, 2, 4, 6).reshape(M * 4, K, D, T, T)

    def _restrict_to_parent(self, child_u: torch.Tensor, parent_u: torch.Tensor,
                            p_sel: torch.Tensor) -> torch.Tensor:
        M = p_sel.numel()
        _, K, D, T, _ = child_u.shape
        x8 = child_u.view(M, 2, 2, K, D, T, T).permute(0, 3, 4, 1, 5, 2, 6).reshape(M * K, D, 2 * T, 2 * T)
        r = self.restrict_conv(x8).view(M, K, D, T, T)
        new_parent = parent_u.clone()
        new_parent[p_sel] = parent_u[p_sel] + (self.rp_scale * r).to(parent_u.dtype)
        return new_parent

    def _l0_bricks_to_dense(self, u0: torch.Tensor, B: int) -> torch.Tensor:
        G0, T, D = self.root_bricks, self.tile_size, self.d_fiber
        rm = u0.view(B, G0 * G0, 4, D, T, T)[:, self.l0_inv_perm]
        return rm.view(B, G0, G0, 4, D, T, T).permute(0, 3, 4, 1, 5, 2, 6).reshape(B, 4, D, G0 * T, G0 * T)

    def _dense_to_l0_bricks(self, dense: torch.Tensor) -> torch.Tensor:
        B, K, D, H, W = dense.shape
        G0, T = self.root_bricks, self.tile_size
        rm = dense.view(B, K, D, G0, T, G0, T).permute(0, 3, 5, 1, 2, 4, 6).reshape(B, G0 * G0, K, D, T, T)
        return rm[:, self.l0_morton_perm].reshape(B * G0 * G0, K, D, T, T)

    def _coarse_global_solve(self, u0_bricks: torch.Tensor, f32: torch.Tensor,
                             sub_coarse: List[torch.Tensor]) -> torch.Tensor:
        B, K, D = f32.shape[0], 4, self.d_fiber
        f16, f8 = sub_coarse

        u32 = self._l0_bricks_to_dense(u0_bricks, B)
        u32 = self.smoother.smooth_dense(u32, f32, rel_depth=0.0, is_finest=0.0, steps=self.smooth_steps)

        u16 = self.restrict_conv(u32.reshape(B * K, D, 32, 32)).view(B, K, D, 16, 16)
        u16 = self.smoother.smooth_dense(u16, f16, rel_depth=-0.25, is_finest=0.0, steps=1)

        u8 = self.restrict_conv(u16.reshape(B * K, D, 16, 16)).view(B, K, D, 8, 8)
        u8 = self.smoother.smooth_dense(u8, f8, rel_depth=-0.5, is_finest=0.0, steps=3)

        up16 = self.prolong_conv(F.interpolate(u8.reshape(B * K, D, 8, 8), scale_factor=2, mode="bilinear", align_corners=False))
        u16 = u16 + (self.rp_scale * up16.view(B, K, D, 16, 16)).to(u16.dtype)
        u16 = self.smoother.smooth_dense(u16, f16, rel_depth=-0.25, is_finest=0.0, steps=1)

        up32 = self.prolong_conv(F.interpolate(u16.reshape(B * K, D, 16, 16), scale_factor=2, mode="bilinear", align_corners=False))
        u32 = u32 + (self.rp_scale * up32.view(B, K, D, 32, 32)).to(u32.dtype)
        u32 = self.smoother.smooth_dense(u32, f32, rel_depth=0.0, is_finest=0.0, steps=self.smooth_steps)

        return self._dense_to_l0_bricks(u32)

    def _run_vcycle(self, states: List[torch.Tensor], levels: List[QuadtreeLevel],
                    pyr: List[torch.Tensor], sub_coarse: List[torch.Tensor]) -> List[torch.Tensor]:
        n_lvls = len(levels)
        states = list(states)

        for l in range(n_lvls - 1, 0, -1):
            rel_d = l / max(n_lvls - 1, 1)
            is_f = 1.0 if l == n_lvls - 1 else 0.0
            states[l] = self.smoother.smooth_bricks(states[l], levels[l], rel_d, is_f, steps=self.smooth_steps)
            states[l - 1] = self._restrict_to_parent(states[l], states[l - 1], levels[l].parent_sel)

        states[0] = self._coarse_global_solve(states[0], pyr[0], sub_coarse)

        for l in range(1, n_lvls):
            rel_d = l / max(n_lvls - 1, 1)
            is_f = 1.0 if l == n_lvls - 1 else 0.0
            delta = self._prolongate_to_children(states[l - 1], levels[l].parent_sel)
            states[l] = states[l] + (self.rp_scale * delta).to(states[l].dtype)
            states[l] = self.smoother.smooth_bricks(states[l], levels[l], rel_d, is_f, steps=self.smooth_steps)

        return states

    def _decode_quadtree_logits(self, states: List[torch.Tensor], levels: List[QuadtreeLevel],
                                B: int, H: int, W: int) -> torch.Tensor:
        device = states[0].device
        T = self.tile_size
        out = torch.zeros(B, 2, H, W, device=device, dtype=torch.float32)
        covered = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)

        for l in range(len(levels) - 1, -1, -1):
            lvl = levels[l]
            u = states[l]
            N, K, D, _, _ = u.shape
            f_logits = self.out_head(u.reshape(N * K, D, T, T)).view(N, K, 2, T, T).float()
            brick_logits = torch.logsumexp(self.lse_beta * f_logits, dim=1) / self.lse_beta  # [N, 2, T, T]

            G = lvl.grid_size
            grid = torch.zeros(B, G, G, 2, T, T, device=device, dtype=torch.float32)
            mask = torch.zeros(B, G, G, 1, T, T, device=device, dtype=torch.float32)
            ones = torch.ones(N, 1, T, T, device=device, dtype=torch.float32)

            grid = grid.index_put((lvl.b_idx, lvl.by_idx, lvl.bx_idx), brick_logits)
            mask = mask.index_put((lvl.b_idx, lvl.by_idx, lvl.bx_idx), ones)

            lvl_logits = grid.permute(0, 3, 1, 4, 2, 5).reshape(B, 2, G * T, G * T)
            lvl_mask = mask.permute(0, 3, 1, 4, 2, 5).reshape(B, 1, G * T, G * T)

            if G * T != H:
                lvl_logits = F.interpolate(lvl_logits, size=(H, W), mode="nearest")
                lvl_mask = F.interpolate(lvl_mask, size=(H, W), mode="nearest")

            take = lvl_mask * (1.0 - covered)
            out = out + lvl_logits * take
            covered = torch.clamp(covered + take, 0.0, 1.0)

        return out

    def forward(self, img: torch.Tensor, gt_split_masks: Optional[List[torch.Tensor]] = None,
                union_prob: float = 0.0, split_thresh: float = 0.55,
                adaptive_exit_eps: Optional[float] = None):
        B, _, H, W = img.shape
        device = img.device
        pyr, sub_coarse = self.pyramid(img)
        n_lvls = len(pyr)

        # ---- Stage 1: Coarse-to-Fine Adaptive Quadtree Growth (FMG) ----
        G0 = self.root_bricks
        b0 = torch.arange(B, device=device).repeat_interleave(G0 * G0)
        by0 = self.l0_by.repeat(B)
        bx0 = self.l0_bx.repeat(B)
        m0 = self.l0_codes.repeat(B)
        bmap0 = torch.full((B, G0, G0), -1, device=device, dtype=torch.long)
        bmap0[b0, by0, bx0] = torch.arange(B * G0 * G0, device=device)
        nb0 = precompute_neighbor_table(b0, by0, bx0, bmap0, G0)

        f0 = self._gather_forcing_bricks(pyr[0], b0, by0, bx0)
        u0 = self._init_state(f0)
        lvl0 = QuadtreeLevel(G0, b0, by0, bx0, m0, bmap0, nb0, None, f0, u0)
        lvl0.state = self.smoother.smooth_bricks(u0, lvl0, rel_depth=0.0, is_finest=0.0, steps=1)

        levels = [lvl0]
        split_logits_list = []
        split_targets_list = []

        if self.training and gt_split_masks is not None and union_prob > 0.0:
            use_union = (torch.rand(B, device=device) < union_prob)
        else:
            use_union = torch.zeros(B, device=device, dtype=torch.bool)

        for l in range(n_lvls - 1):
            cur = levels[l]
            rel_d_cur = l / max(n_lvls - 1, 1)
            s_logits = self._predict_split_logits(cur.state, cur.forcing, rel_d_cur)
            split_logits_list.append(s_logits)

            pred_mask = (torch.sigmoid(s_logits) > split_thresh)
            if gt_split_masks is not None:
                gt_s = gt_split_masks[l][cur.b_idx, cur.by_idx, cur.bx_idx]
                split_targets_list.append(gt_s)
                split_mask = torch.where(use_union[cur.b_idx], pred_mask | (gt_s > 0.5), pred_mask)
            else:
                split_mask = pred_mask

            # Safety guard: ensure at least 1 brick splits globally (avoids 24x per-batch GPU syncs!)
            if not split_mask.any():
                split_mask[s_logits.argmax()] = True

            # Zero-Sort Morton Expansion
            p_sel = torch.where(split_mask)[0]
            M = p_sel.numel()
            c_parent = p_sel.repeat_interleave(4)
            c_q = torch.arange(4, device=device).repeat(M)
            c_b = cur.b_idx[c_parent]
            c_bx = cur.bx_idx[c_parent] * 2 + (c_q & 1)
            c_by = cur.by_idx[c_parent] * 2 + ((c_q >> 1) & 1)
            c_morton = (cur.morton[c_parent] << 2) | c_q

            G_next = cur.grid_size * 2
            bmap_next = torch.full((B, G_next, G_next), -1, device=device, dtype=torch.long)
            bmap_next[c_b, c_by, c_bx] = torch.arange(4 * M, device=device)
            nb_next = precompute_neighbor_table(c_b, c_by, c_bx, bmap_next, G_next)

            f_next = self._gather_forcing_bricks(pyr[l + 1], c_b, c_by, c_bx)
            u_next = self._init_state(f_next) + (self.rp_scale * self._prolongate_to_children(cur.state, p_sel)).to(f_next.dtype)

            lvl_next = QuadtreeLevel(G_next, c_b, c_by, c_bx, c_morton, bmap_next, nb_next, p_sel, f_next, u_next)
            rel_d = (l + 1) / max(n_lvls - 1, 1)
            is_f = 1.0 if (l + 1) == n_lvls - 1 else 0.0
            lvl_next.state = self.smoother.smooth_bricks(u_next, lvl_next, rel_d, is_f, steps=1)
            levels.append(lvl_next)

        # ---- Stage 2: Weight-Tied Multigrid V-Cycles ----
        states = [lvl.state for lvl in levels]
        cycle_logits = []
        cycles_used = 0

        for k in range(self.n_vcycles):
            prev_finest = states[-1]
            if self.use_checkpoint and self.training:
                states = list(checkpoint(
                    lambda *s: tuple(self._run_vcycle(list(s), levels, pyr, sub_coarse)),
                    *states, use_reentrant=False
                ))
            else:
                states = self._run_vcycle(states, levels, pyr, sub_coarse)

            cycles_used = k + 1
            if self.training or k == self.n_vcycles - 1 or adaptive_exit_eps is not None:
                cycle_logits.append(self._decode_quadtree_logits(states, levels, B, H, W))

            if (not self.training) and adaptive_exit_eps is not None and k >= 1:
                if (states[-1] - prev_finest).abs().mean().item() < adaptive_exit_eps:
                    break

        finest = levels[-1]
        active_ratio = finest.b_idx.numel() / float(B * finest.grid_size * finest.grid_size)

        return {
            "logits": cycle_logits[-1],
            "cycle_logits": cycle_logits,
            "split_logits": split_logits_list,
            "split_targets": split_targets_list,
            "active_ratio": active_ratio,
            "cycles_used": cycles_used,
        }
