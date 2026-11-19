# haizklow/se2_multigrid.py
# HaizkLow — Standalone Adaptive Morton-Quadtree + SE(2) Fiber Multigrid Engine (Phase 1)
#
# Zero U-Net. Zero Self-Attention. Zero Mamba.
# Pure weight-tied SE(2) Anisotropic PDE Smoother + Morton 4x4 Brick Multigrid V-Cycles.

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
    """
    GPU-vectorized GT brick occupancy masks from [B, 2, H, W].
    Returns [L0 (B,8,8), L1 (B,16,16), ..., L_finest (B,H//4,W//4)] float {0,1}.
    """
    fg = (target.sum(dim=1, keepdim=True) > 0.5).float()  # [B, 1, H, W]
    cur = F.max_pool2d(fg, tile_size, tile_size)          # [B, 1, H//4, W//4]
    rev = [cur.squeeze(1)]
    while cur.shape[-1] > root_bricks:
        cur = F.max_pool2d(cur, 2, 2)
        rev.append(cur.squeeze(1))
    return list(reversed(rev))


@dataclass
class QuadtreeLevel:
    grid_size: int                  # G_l (number of 4x4 bricks per side)
    b_idx: torch.Tensor             # [N] int64 batch index
    by_idx: torch.Tensor            # [N] int64 brick y
    bx_idx: torch.Tensor            # [N] int64 brick x
    morton: torch.Tensor            # [N] int64 sorted Morton code per batch item
    brick_map: torch.Tensor         # [B, G_l, G_l] int64 -> compact idx 0..N-1 or -1
    parent_sel: Optional[torch.Tensor]  # [M] parent indices at l-1 that spawned this level (N = 4*M)
    forcing: torch.Tensor           # [N, 4, 12, 4, 4] pre-gathered SE(2) mipmap forcing
    state: torch.Tensor             # [N, 4, D, 4, 4] live SE(2) fiber state


def gather_brick_halo(u: torch.Tensor, lvl: QuadtreeLevel) -> torch.Tensor:
    """
    Exact 1-pixel sparse halo gather for active 4x4 bricks -> [N, 4, D, 6, 6].
    Matches dense zero-padding bit-for-bit (Phase 2 replaces this with fused CUDA shared-mem!).
    """
    N, K, D, T, _ = u.shape
    G = lvl.grid_size
    device = u.device

    # Sentinel zero brick at index N so inactive/OOB neighbors read 0 cleanly without branches
    u_z = torch.cat([u, torch.zeros(1, K, D, T, T, device=device, dtype=u.dtype)], dim=0)
    padded = torch.zeros(N, K, D, T + 2, T + 2, device=device, dtype=u.dtype)
    padded[:, :, :, 1:T + 1, 1:T + 1] = u

    b_idx, by_idx, bx_idx = lvl.b_idx, lvl.by_idx, lvl.bx_idx
    bmap = lvl.brick_map

    def nb(dy: int, dx: int) -> torch.Tensor:
        ny = by_idx + dy
        nx = bx_idx + dx
        valid = (ny >= 0) & (ny < G) & (nx >= 0) & (nx < G)
        idx = torch.where(
            valid,
            bmap[b_idx, ny.clamp(0, G - 1), nx.clamp(0, G - 1)],
            torch.full_like(b_idx, -1),
        )
        return torch.where(idx >= 0, idx, torch.full_like(idx, N))

    # 4 edges + 4 corners
    padded[:, :, :, 0, 1:T + 1] = u_z[nb(-1, 0), :, :, T - 1, :]
    padded[:, :, :, T + 1, 1:T + 1] = u_z[nb(1, 0), :, :, 0, :]
    padded[:, :, :, 1:T + 1, 0] = u_z[nb(0, -1), :, :, :, T - 1]
    padded[:, :, :, 1:T + 1, T + 1] = u_z[nb(0, 1), :, :, :, 0]
    padded[:, :, :, 0, 0] = u_z[nb(-1, -1), :, :, T - 1, T - 1]
    padded[:, :, :, 0, T + 1] = u_z[nb(-1, 1), :, :, T - 1, 0]
    padded[:, :, :, T + 1, 0] = u_z[nb(1, -1), :, :, 0, T - 1]
    padded[:, :, :, T + 1, T + 1] = u_z[nb(1, 1), :, :, 0, 0]
    return padded


class SE2AnisotropicSmoother(nn.Module):
    """
    C4 weight-shared SE(2) Orientation-Lifted PDE Smoother.
    Fibers k=0(0°), 1(45°), 2(90°), 3(135°):
      - Transport strictly along +/- d_k (ZERO direct 0° <-> 90° mixing!)
      - Curvature coupling only between adjacent angles k +/- 1 (+/- 45°)
      - Image forcing I_k pinned at every step so fine capillaries never blur out.
    """
    def __init__(self, d_fiber: int = 16, f_ch: int = 12, hidden: int = 48):
        super().__init__()
        self.d_fiber = d_fiber

        # Directional edge conductance gate g_theta([u, u_nb - u, I_k]) -> (0, 1)^D
        self.cond_gate = nn.Sequential(
            nn.Linear(2 * d_fiber + f_ch, d_fiber * 2),
            nn.GELU(),
            nn.Linear(d_fiber * 2, d_fiber),
        )
        # Angular turning gate g_ang([u, u_adj - u, I_k]) -> (0, 1)^D
        self.ang_gate = nn.Linear(2 * d_fiber + f_ch, d_fiber)

        # Nonlinear local reaction MLP Phi_theta (resolution-agnostic via [rel_depth, is_finest])
        in_dim = 4 * d_fiber + f_ch + 2
        self.norm = nn.LayerNorm(in_dim)
        self.react = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, d_fiber),
        )
        # Initialize reaction output small + damped step size dt in (0, 0.5)
        nn.init.zeros_(self.react[-1].weight)
        nn.init.zeros_(self.react[-1].bias)
        self.dt_logit = nn.Parameter(torch.tensor(0.0))  # dt = 0.5 * sigmoid(0) = 0.25

    def _step_from_padded(self, u: torch.Tensor, u_pad: torch.Tensor,
                          forcing: torch.Tensor, rel_depth: float, is_finest: float) -> torch.Tensor:
        """
        u:       [N, 4, D, Hp, Wp]
        u_pad:   [N, 4, D, Hp+2, Wp+2]
        forcing: [N, 4, 12, Hp, Wp]
        """
        # 1. Extract strictly oriented +/- d_k neighbors for each fiber
        u_pos = torch.stack([
            u_pad[:, 0, :, 1:-1, 2:  ],  # 0 deg   (East)
            u_pad[:, 1, :, :-2,  2:  ],  # 45 deg  (NE)
            u_pad[:, 2, :, 2:,   1:-1],  # 90 deg  (South)
            u_pad[:, 3, :, 2:,   2:  ],  # 135 deg (SE)
        ], dim=1)
        u_neg = torch.stack([
            u_pad[:, 0, :, 1:-1, :-2 ],  # 0 deg   (West)
            u_pad[:, 1, :, 2:,   :-2 ],  # 45 deg  (SW)
            u_pad[:, 2, :, :-2,  1:-1],  # 90 deg  (North)
            u_pad[:, 3, :, :-2,  :-2 ],  # 135 deg (NW)
        ], dim=1)

        # Move channel dim last for C4 weight-shared Linear ops: [N, 4, Hp, Wp, C]
        u_c = u.permute(0, 1, 3, 4, 2)
        f_c = forcing.permute(0, 1, 3, 4, 2)
        dp = u_pos.permute(0, 1, 3, 4, 2) - u_c
        dn = u_neg.permute(0, 1, 3, 4, 2) - u_c

        # 2. Gated anisotropic transport along theta_k (180°-symmetric sum + abs-diff)
        gp = torch.sigmoid(self.cond_gate(torch.cat([u_c, dp, f_c], dim=-1)))
        gn = torch.sigmoid(self.cond_gate(torch.cat([u_c, dn, f_c], dim=-1)))
        fp = gp * dp
        fn = gn * dn
        diff_flux = fp + fn
        grad_flux = torch.abs(fp - fn)

        # 3. Smooth angular coupling ONLY with k-1 and k+1 (NEVER orthogonal k+2!)
        u_prev = torch.roll(u_c, shifts=1, dims=1)
        u_next = torch.roll(u_c, shifts=-1, dims=1)
        d_ang = 0.5 * (u_prev + u_next) - u_c
        g_ang = torch.sigmoid(self.ang_gate(torch.cat([u_c, d_ang, f_c], dim=-1)))
        ang_flux = g_ang * d_ang

        # 4. Continuous resolution-agnostic scale conditioning
        scale_feat = torch.empty(*u_c.shape[:-1], 2, device=u.device, dtype=u.dtype)
        scale_feat[..., 0] = rel_depth
        scale_feat[..., 1] = is_finest

        # 5. Damped PDE update
        x = torch.cat([u_c, diff_flux, grad_flux, ang_flux, f_c, scale_feat], dim=-1)
        du = self.react(self.norm(x)).permute(0, 1, 4, 2, 3)  # [N, 4, D, Hp, Wp]
        dt = 0.5 * torch.sigmoid(self.dt_logit)
        return u + dt * du

    def smooth_bricks(self, u: torch.Tensor, lvl: QuadtreeLevel,
                      rel_depth: float, is_finest: float, steps: int = 2) -> torch.Tensor:
        for _ in range(steps):
            u_pad = gather_brick_halo(u, lvl)
            u = self._step_from_padded(u, u_pad, lvl.forcing, rel_depth, is_finest)
        return u

    def smooth_dense(self, u: torch.Tensor, forcing: torch.Tensor,
                     rel_depth: float, is_finest: float, steps: int = 2) -> torch.Tensor:
        for _ in range(steps):
            u_pad = F.pad(u, (1, 1, 1, 1))
            u = self._step_from_padded(u, u_pad, forcing, rel_depth, is_finest)
        return u


class HaizkLow(nn.Module):
    """
    Standalone Adaptive Morton-Quadtree + SE(2) Fiber Multigrid Architecture.
    """
    def __init__(self, d_fiber: int = 16, hidden: int = 48, n_vcycles: int = 3,
                 smooth_steps: int = 2, tile_size: int = 4, root_bricks: int = 8,
                 use_checkpoint: bool = False):
        super().__init__()
        self.d_fiber = d_fiber
        self.n_vcycles = n_vcycles
        self.smooth_steps = smooth_steps
        self.tile_size = tile_size
        self.root_bricks = root_bricks
        self.use_checkpoint = use_checkpoint

        self.pyramid = MultiStatSE2Pyramid(tile_size=tile_size, root_bricks=root_bricks)

        # Init state from raw forcing I_k
        self.init_proj = nn.Linear(12, d_fiber)

        # ONE shared SE(2) smoother across all levels and all V-cycles!
        self.smoother = SE2AnisotropicSmoother(d_fiber=d_fiber, f_ch=12, hidden=hidden)

        # Shared Morton [8x8 <-> 4x4] Restriction & Prolongation (applied per fiber)
        self.restrict_conv = nn.Conv2d(d_fiber, d_fiber, kernel_size=2, stride=2)
        self.prolong_conv = nn.Conv2d(d_fiber, d_fiber, kernel_size=3, padding=1)
        self.rp_scale = nn.Parameter(torch.tensor(0.25))

        # Brick Split Predictor (predicts whether a 4x4 brick needs subdivision)
        self.split_head = nn.Sequential(
            nn.Linear(2 * d_fiber + 12, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        # Negative bias keeps empty background OFF on Step 1 (Union Warmup keeps 100% of GT vessels ON!)
        nn.init.constant_(self.split_head[-1].bias, -1.0)

        # Fiber Output Head (each fiber independently predicts [Artery, Vein] logits)
        self.out_head = nn.Linear(d_fiber, 2)
        self.lse_beta = 4.0  # Smooth-max sharpness for crossing-preserving fiber collapse

        # Precompute L0 (8x8) Morton permutation tables
        G0 = root_bricks
        codes = [_morton2d(r % G0, r // G0) for r in range(G0 * G0)]
        morton_perm = torch.tensor(sorted(range(G0 * G0), key=lambda r: codes[r]), dtype=torch.long)
        inv_perm = torch.argsort(morton_perm)
        by0 = morton_perm // G0
        bx0 = morton_perm % G0
        self.register_buffer("l0_morton_perm", morton_perm)
        self.register_buffer("l0_inv_perm", inv_perm)
        self.register_buffer("l0_by", by0)
        self.register_buffer("l0_bx", bx0)
        self.register_buffer("l0_codes", torch.arange(G0 * G0, dtype=torch.long))

    def _gather_forcing_bricks(self, pyr_l: torch.Tensor, b_idx, by_idx, bx_idx) -> torch.Tensor:
        # pyr_l: [B, 4, 12, G*T, G*T] -> [N, 4, 12, T, T]
        B, K, C, H, W = pyr_l.shape
        T = self.tile_size
        G = H // T
        bricks = pyr_l.view(B, K, C, G, T, G, T).permute(0, 3, 5, 1, 2, 4, 6)
        return bricks[b_idx, by_idx, bx_idx]

    def _predict_split_logits(self, u: torch.Tensor, forcing: torch.Tensor) -> torch.Tensor:
        # u: [N, 4, D, T, T], forcing: [N, 4, 12, T, T] -> [N]
        u_mean = u.mean(dim=(1, 3, 4))                      # [N, D]
        u_max = u.amax(dim=(1, 3, 4))                       # [N, D]
        f_max = forcing.amax(dim=(1, 3, 4))                 # [N, 12]
        return self.split_head(torch.cat([u_mean, u_max, f_max], dim=-1)).squeeze(-1)

    def _prolongate_to_children(self, parent_u: torch.Tensor, p_sel: torch.Tensor) -> torch.Tensor:
        # parent_u[p_sel]: [M, 4, D, 4, 4] -> children delta [4*M, 4, D, 4, 4]
        M, K, D, T, _ = parent_u[p_sel].shape
        p = parent_u[p_sel].reshape(M * K, D, T, T)
        up = self.prolong_conv(F.interpolate(p, scale_factor=2, mode="bilinear", align_corners=False))
        return up.view(M, K, D, 2, T, 2, T).permute(0, 3, 5, 1, 2, 4, 6).reshape(M * 4, K, D, T, T)

    def _restrict_to_parent(self, child_u: torch.Tensor, parent_u: torch.Tensor,
                            p_sel: torch.Tensor) -> torch.Tensor:
        # child_u: [4*M, 4, D, 4, 4] -> add restricted residual to parent_u[p_sel]
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
        """
        Mini-V solve below L0: 32x32 -> 16x16 -> 8x8 (global!) -> 16x16 -> 32x32.
        Gives full-image global receptive field on V-Cycle #1 with zero Self-Attention!
        """
        B, K, D = f32.shape[0], 4, self.d_fiber
        f16, f8 = sub_coarse

        u32 = self._l0_bricks_to_dense(u0_bricks, B)
        u32 = self.smoother.smooth_dense(u32, f32, rel_depth=0.0, is_finest=0.0, steps=self.smooth_steps)

        # Restrict 32 -> 16 -> 8
        u16 = self.restrict_conv(u32.reshape(B * K, D, 32, 32)).view(B, K, D, 16, 16)
        u16 = self.smoother.smooth_dense(u16, f16, rel_depth=-0.25, is_finest=0.0, steps=2)

        u8 = self.restrict_conv(u16.reshape(B * K, D, 16, 16)).view(B, K, D, 8, 8)
        u8 = self.smoother.smooth_dense(u8, f8, rel_depth=-0.5, is_finest=0.0, steps=4)

        # Prolongate 8 -> 16 -> 32
        up16 = self.prolong_conv(F.interpolate(u8.reshape(B * K, D, 8, 8), scale_factor=2, mode="bilinear", align_corners=False))
        u16 = u16 + self.rp_scale * up16.view(B, K, D, 16, 16)
        u16 = self.smoother.smooth_dense(u16, f16, rel_depth=-0.25, is_finest=0.0, steps=2)

        up32 = self.prolong_conv(F.interpolate(u16.reshape(B * K, D, 16, 16), scale_factor=2, mode="bilinear", align_corners=False))
        u32 = u32 + self.rp_scale * up32.view(B, K, D, 32, 32)
        u32 = self.smoother.smooth_dense(u32, f32, rel_depth=0.0, is_finest=0.0, steps=self.smooth_steps)

        return self._dense_to_l0_bricks(u32)

    def _run_vcycle(self, states: List[torch.Tensor], levels: List[QuadtreeLevel],
                    pyr: List[torch.Tensor], sub_coarse: List[torch.Tensor]) -> List[torch.Tensor]:
        n_lvls = len(levels)
        states = list(states)

        # 1. Downward leg (Finest -> L1): Pre-smooth + Restrict residuals to parent
        for l in range(n_lvls - 1, 0, -1):
            rel_d = l / max(n_lvls - 1, 1)
            is_f = 1.0 if l == n_lvls - 1 else 0.0
            states[l] = self.smoother.smooth_bricks(states[l], levels[l], rel_d, is_f, steps=self.smooth_steps)
            states[l - 1] = self._restrict_to_parent(states[l], states[l - 1], levels[l].parent_sel)

        # 2. Bottom of V: Global Coarse Solve at L0 (32 -> 16 -> 8 -> 16 -> 32)
        states[0] = self._coarse_global_solve(states[0], pyr[0], sub_coarse)

        # 3. Upward leg (L1 -> Finest): Prolongate coarse correction + Post-smooth
        for l in range(1, n_lvls):
            rel_d = l / max(n_lvls - 1, 1)
            is_f = 1.0 if l == n_lvls - 1 else 0.0
            delta = self._prolongate_to_children(states[l - 1], levels[l].parent_sel)
            states[l] = states[l] + self.rp_scale * delta
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
            f_logits = self.out_head(u.permute(0, 1, 3, 4, 2)).float()     # [N, 4, T, T, 2]
            beta = self.lse_beta
            brick_logits = (torch.logsumexp(beta * f_logits, dim=1) / beta).permute(0, 3, 1, 2)

            G = lvl.grid_size
            grid = torch.zeros(B, G, G, 2, T, T, device=device, dtype=torch.float32)
            mask = torch.zeros(B, G, G, 1, T, T, device=device, dtype=torch.float32)
            ones = torch.ones(u.shape[0], 1, T, T, device=device, dtype=torch.float32)

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
                union_prob: float = 0.0, split_thresh: float = 0.45,
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

        f0 = self._gather_forcing_bricks(pyr[0], b0, by0, bx0)
        u0 = self.init_proj(f0.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)

        lvl0 = QuadtreeLevel(G0, b0, by0, bx0, m0, bmap0, None, f0, u0)
        u0 = self.smoother.smooth_bricks(u0, lvl0, rel_depth=0.0, is_finest=0.0, steps=1)
        lvl0.state = u0

        levels = [lvl0]
        split_logits_list = []
        split_targets_list = []

        # Sample which batch items use Union Warmup (GT | Pred) vs pure Pred
        if self.training and gt_split_masks is not None and union_prob > 0.0:
            use_union = (torch.rand(B, device=device) < union_prob)
        else:
            use_union = torch.zeros(B, device=device, dtype=torch.bool)

        for l in range(n_lvls - 1):
            cur = levels[l]
            s_logits = self._predict_split_logits(cur.state, cur.forcing)  # [N_l]
            split_logits_list.append(s_logits)

            pred_mask = (torch.sigmoid(s_logits) > split_thresh)
            if gt_split_masks is not None:
                gt_s = gt_split_masks[l][cur.b_idx, cur.by_idx, cur.bx_idx]  # [N_l]
                split_targets_list.append(gt_s)
                gt_bool = (gt_s > 0.5)
                split_mask = torch.where(use_union[cur.b_idx], pred_mask | gt_bool, pred_mask)
            else:
                split_mask = pred_mask

            # Hard VRAM safety cap during training: never split >50% of bricks at fine levels
            if self.training and l >= 1:
                max_m = max(B, int(0.50 * cur.b_idx.numel()))
                if split_mask.sum().item() > max_m:
                    topk_idx = torch.topk(s_logits, k=max_m).indices
                    cap_mask = torch.zeros_like(split_mask)
                    cap_mask[topk_idx] = True
                    split_mask = cap_mask

            # Safety guard: ensure >= 1 brick splits per batch item so M >= B always
            for b in range(B):
                b_sel = (cur.b_idx == b)
                if not (split_mask & b_sel).any():
                    b_indices = torch.where(b_sel)[0]
                    best_i = b_indices[s_logits[b_indices].argmax()]
                    split_mask[best_i] = True

            # Zero-Sort Morton Expansion: sorted parents -> 4 contiguous sorted children!
            p_sel = torch.where(split_mask)[0]                             # [M]
            M = p_sel.numel()
            c_parent = p_sel.repeat_interleave(4)                          # [4*M]
            c_q = torch.arange(4, device=device).repeat(M)                 # [4*M]
            c_b = cur.b_idx[c_parent]
            c_bx = cur.bx_idx[c_parent] * 2 + (c_q & 1)
            c_by = cur.by_idx[c_parent] * 2 + ((c_q >> 1) & 1)
            c_morton = (cur.morton[c_parent] << 2) | c_q

            G_next = cur.grid_size * 2
            bmap_next = torch.full((B, G_next, G_next), -1, device=device, dtype=torch.long)
            bmap_next[c_b, c_by, c_bx] = torch.arange(4 * M, device=device)

            f_next = self._gather_forcing_bricks(pyr[l + 1], c_b, c_by, c_bx)
            u_init = self.init_proj(f_next.permute(0, 1, 3, 4, 2)).permute(0, 1, 4, 2, 3)
            u_next = u_init + self.rp_scale * self._prolongate_to_children(cur.state, p_sel)

            lvl_next = QuadtreeLevel(G_next, c_b, c_by, c_bx, c_morton, bmap_next, p_sel, f_next, u_next)
            rel_d = (l + 1) / max(n_lvls - 1, 1)
            is_f = 1.0 if (l + 1) == n_lvls - 1 else 0.0
            u_next = self.smoother.smooth_bricks(u_next, lvl_next, rel_d, is_f, steps=1)
            lvl_next.state = u_next
            levels.append(lvl_next)

        # ---- Stage 2: Weight-Tied Multigrid V-Cycles (with Deep Supervision & Adaptive Exit) ----
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
                logits_k = self._decode_quadtree_logits(states, levels, B, H, W)
                cycle_logits.append(logits_k)

            # Adaptive Test-Time Early Exit when PDE residual converges!
            if (not self.training) and adaptive_exit_eps is not None and k >= 1:
                delta_norm = (states[-1] - prev_finest).abs().mean().item()
                if delta_norm < adaptive_exit_eps:
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
