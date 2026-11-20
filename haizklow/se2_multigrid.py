# haizklow/se2_multigrid.py
# HaizkLow v2.2 (Clean Baseline) — Imports all Quadtree/Morton ops cleanly from quadtree.py
# Cleanup fixes:
#   - Pixel-wise ChannelRMSNorm2d (kills catastrophic per-4x4-brick vs dense GroupNorm bug!)
#   - 1-shot CUDA int32 neighbor builder
#   - Cuts redundant coarse/intermediate smooth steps (83 -> 47 steps)
#   - Fixes adaptive_exit_eps (decodes dense logits ONCE after exit)

from typing import List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .mipmap import MultiStatSE2Pyramid
from .quadtree import (
    QuadtreeLevel,
    batch_gt_split_masks,
    build_bmap_and_neighbors,
    expand_morton_children,
    gather_brick_halo_ref,
    gather_bricks,
    make_l0_morton_tables,
    scatter_bricks,
)
from .ops import fused_halo_dw3x3


class ChannelRMSNorm2d(nn.Module):
    """Per-pixel channel-only RMSNorm: 100% invariant between (N,C,4,4) bricks & (B,C,H,W) dense grids."""
    def __init__(self, ch: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(1, ch, 1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.float().pow(2).mean(dim=1, keepdim=True) + self.eps).to(x.dtype)
        return x * rms * self.weight.to(x.dtype)


class ContractiveBrickSmoother(nn.Module):
    def __init__(self, ch: int = 32):
        super().__init__()
        self.ch = ch
        self.spatial_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=0, groups=ch)

        in_dim = 3 * ch + 2
        self.norm = ChannelRMSNorm2d(in_dim)
        self.pw = nn.Sequential(
            nn.Conv2d(in_dim, 48, kernel_size=1),
            nn.GELU(),
            nn.Conv2d(48, ch, kernel_size=1),
        )
        self.scale_film = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 2 * ch),
        )
        nn.init.zeros_(self.scale_film[-1].weight)
        nn.init.zeros_(self.scale_film[-1].bias)

        self.res_scale = nn.Parameter(torch.tensor(0.5))
        self.state_norm = ChannelRMSNorm2d(ch)

    def _step_from_spatial(self, u: torch.Tensor, spatial: torch.Tensor, f_emb: torch.Tensor,
                           rel_depth: float, is_finest: float) -> torch.Tensor:
        N, C, Hp, Wp = u.shape
        scale_feat = torch.empty(N, 2, Hp, Wp, device=u.device, dtype=u.dtype)
        scale_feat[:, 0] = rel_depth
        scale_feat[:, 1] = is_finest

        x = torch.cat([u, spatial, f_emb, scale_feat], dim=1)
        du = self.pw(self.norm(x)).to(u.dtype)

        s_vec = scale_feat[0, :, 0, 0]
        gamma, beta = self.scale_film(s_vec).to(u.dtype).chunk(2)
        du = du * (1.0 + gamma.view(1, C, 1, 1)) + beta.view(1, C, 1, 1)

        return self.state_norm(u + self.res_scale.to(u.dtype) * du).to(u.dtype)

    def smooth_bricks(self, u: torch.Tensor, lvl: QuadtreeLevel,
                      rel_depth: float, is_finest: float, steps: int = 1) -> torch.Tensor:
        for _ in range(steps):
            if u.is_cuda and u.shape[-1] == 4:
                spatial = fused_halo_dw3x3(u, self.spatial_conv.weight, self.spatial_conv.bias, lvl.nb_idx)
            else:
                spatial = self.spatial_conv(gather_brick_halo_ref(u, lvl.nb_idx))
            u = self._step_from_spatial(u, spatial, lvl.f_emb, rel_depth, is_finest)
        return u

    def smooth_dense(self, u: torch.Tensor, f_emb: torch.Tensor,
                     rel_depth: float, is_finest: float, steps: int = 1) -> torch.Tensor:
        for _ in range(steps):
            spatial = self.spatial_conv(F.pad(u, (1, 1, 1, 1)))
            u = self._step_from_spatial(u, spatial, f_emb, rel_depth, is_finest)
        return u


class HaizkLow(nn.Module):
    def __init__(self, ch: int = 32, out_ch: int = 2, n_vcycles: int = 3,
                 smooth_steps: int = 1, tile_size: int = 4, root_bricks: int = 8,
                 use_checkpoint: bool = False, **kwargs):
        super().__init__()
        self.ch = ch
        self.out_ch = out_ch
        self.n_vcycles = n_vcycles
        self.smooth_steps = smooth_steps
        self.tile_size = tile_size
        self.root_bricks = root_bricks
        self.use_checkpoint = use_checkpoint

        self.pyramid = MultiStatSE2Pyramid(tile_size=tile_size, root_bricks=root_bricks)
        self.init_proj = nn.Conv2d(20, ch, kernel_size=1)
        self.smoother = ContractiveBrickSmoother(ch=ch)

        self.restrict_conv = nn.Conv2d(ch, ch, kernel_size=2, stride=2, groups=ch)
        self.prolong_conv = nn.Conv2d(ch, ch, kernel_size=3, padding=1, groups=ch)
        nn.init.zeros_(self.restrict_conv.weight)
        nn.init.zeros_(self.restrict_conv.bias)
        nn.init.zeros_(self.prolong_conv.weight)
        nn.init.zeros_(self.prolong_conv.bias)
        self.restrict_blend = nn.Parameter(torch.tensor(0.0))

        self.split_head = nn.Sequential(
            nn.Linear(2 * ch + 20 + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.split_head[-1].bias, 0.0)
        self.out_head = nn.Conv2d(ch, out_ch, kernel_size=1)

        morton_perm, inv_perm, l0_by, l0_bx, sorted_codes = make_l0_morton_tables(root_bricks)
        self.register_buffer("l0_morton_perm", morton_perm)
        self.register_buffer("l0_inv_perm", inv_perm)
        self.register_buffer("l0_by", l0_by)
        self.register_buffer("l0_bx", l0_bx)
        self.register_buffer("l0_codes", sorted_codes)

    def _predict_split_logits(self, u: torch.Tensor, f_raw: torch.Tensor, rel_depth: float) -> torch.Tensor:
        u_mean = u.mean(dim=(2, 3))
        u_max = u.amax(dim=(2, 3))
        f_max = f_raw.amax(dim=(2, 3))
        depth_col = torch.full((u.shape[0], 1), rel_depth, device=u.device, dtype=u.dtype)
        return self.split_head(torch.cat([u_mean, u_max, f_max, depth_col], dim=-1)).squeeze(-1)

    def _restrict_op(self, x: torch.Tensor) -> torch.Tensor:
        return F.avg_pool2d(x, 2, 2) + self.restrict_conv(x)

    def _prolong_op(self, x: torch.Tensor) -> torch.Tensor:
        up = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        return up + self.prolong_conv(up)

    def _prolongate_to_children(self, parent_u: torch.Tensor, p_sel: torch.Tensor) -> torch.Tensor:
        p = parent_u[p_sel]
        M, C, T, _ = p.shape
        up = self._prolong_op(p)
        return up.view(M, C, 2, T, 2, T).permute(0, 2, 4, 1, 3, 5).reshape(M * 4, C, T, T)

    def _restrict_to_parent(self, child_u: torch.Tensor, parent_u: torch.Tensor,
                            p_sel: torch.Tensor) -> torch.Tensor:
        M = p_sel.numel()
        _, C, T, _ = child_u.shape
        x8 = child_u.view(M, 2, 2, C, T, T).permute(0, 3, 1, 4, 2, 5).reshape(M, C, 2 * T, 2 * T)
        r = self._restrict_op(x8).to(parent_u.dtype)
        gam = torch.sigmoid(self.restrict_blend).to(parent_u.dtype)
        new_parent = parent_u.clone()
        new_parent[p_sel] = (1.0 - gam) * parent_u[p_sel] + gam * r
        return new_parent

    def _l0_bricks_to_dense(self, u0: torch.Tensor, B: int) -> torch.Tensor:
        G0, T, C = self.root_bricks, self.tile_size, self.ch
        rm = u0.view(B, G0 * G0, C, T, T)[:, self.l0_inv_perm]
        return rm.view(B, G0, G0, C, T, T).permute(0, 3, 1, 4, 2, 5).reshape(B, C, G0 * T, G0 * T)

    def _dense_to_l0_bricks(self, dense: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dense.shape
        G0, T = self.root_bricks, self.tile_size
        rm = dense.view(B, C, G0, T, G0, T).permute(0, 2, 4, 1, 3, 5).reshape(B, G0 * G0, C, T, T)
        return rm[:, self.l0_morton_perm].reshape(B * G0 * G0, C, T, T)

    def _coarse_global_solve(self, u0_bricks: torch.Tensor, f32_emb: torch.Tensor,
                             sub_emb: List[torch.Tensor]) -> torch.Tensor:
        B = f32_emb.shape[0]
        f16, f8 = sub_emb

        u32 = self._l0_bricks_to_dense(u0_bricks, B)
        u32 = self.smoother.smooth_dense(u32, f32_emb, rel_depth=0.0, is_finest=0.0, steps=1)

        u16_0 = self._restrict_op(u32)
        u16 = self.smoother.smooth_dense(u16_0, f16, rel_depth=-0.25, is_finest=0.0, steps=1)

        u8_0 = self._restrict_op(u16)
        u8 = self.smoother.smooth_dense(u8_0, f8, rel_depth=-0.5, is_finest=0.0, steps=2)

        u16 = self.smoother.state_norm(u16 + self._prolong_op(u8 - u8_0).to(u16.dtype)).to(u16.dtype)
        u16 = self.smoother.smooth_dense(u16, f16, rel_depth=-0.25, is_finest=0.0, steps=1)

        u32 = self.smoother.state_norm(u32 + self._prolong_op(u16 - u16_0).to(u32.dtype)).to(u32.dtype)
        u32 = self.smoother.smooth_dense(u32, f32_emb, rel_depth=0.0, is_finest=0.0, steps=1)
        return self._dense_to_l0_bricks(u32)

    def _run_vcycle(self, states: List[torch.Tensor], levels: List[QuadtreeLevel],
                    f32_emb: torch.Tensor, sub_emb: List[torch.Tensor]) -> List[torch.Tensor]:
        n_lvls = len(levels)
        states = list(states)
        saved_parents: List[Optional[torch.Tensor]] = [None] * n_lvls

        for l in range(n_lvls - 1, 0, -1):
            rel_d = l / max(n_lvls - 1, 1)
            is_f = 1.0 if l == n_lvls - 1 else 0.0
            states[l] = self.smoother.smooth_bricks(states[l], levels[l], rel_d, is_f, steps=self.smooth_steps)
            states[l - 1] = self._restrict_to_parent(states[l], states[l - 1], levels[l].parent_sel)
            saved_parents[l - 1] = states[l - 1]

        states[0] = self._coarse_global_solve(states[0], f32_emb, sub_emb)

        for l in range(1, n_lvls):
            rel_d = l / max(n_lvls - 1, 1)
            is_f = 1.0 if l == n_lvls - 1 else 0.0
            coarse_delta = states[l - 1] - saved_parents[l - 1]
            delta_fine = self._prolongate_to_children(coarse_delta, levels[l].parent_sel)
            states[l] = self.smoother.state_norm(states[l] + delta_fine.to(states[l].dtype)).to(states[l].dtype)
            states[l] = self.smoother.smooth_bricks(states[l], levels[l], rel_d, is_f, steps=self.smooth_steps)

        return states

    def _decode_quadtree_logits(self, states: List[torch.Tensor], levels: List[QuadtreeLevel],
                                B: int, H: int, W: int) -> torch.Tensor:
        device = states[0].device
        T = self.tile_size
        out = torch.zeros(B, self.out_ch, H, W, device=device, dtype=torch.float32)
        covered = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)

        for l in range(len(levels) - 1, -1, -1):
            lvl = levels[l]
            brick_logits = self.out_head(states[l]).float()
            G = lvl.grid_size
            lvl_logits, lvl_mask = scatter_bricks(brick_logits, lvl.b_idx, lvl.by_idx, lvl.bx_idx, B, G, T)

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

        f32_emb = self.init_proj(pyr[0])
        sub_emb = [self.init_proj(s) for s in sub_coarse]

        # ---- Stage 1: Coarse-to-Fine Adaptive Quadtree Growth ----
        G0 = self.root_bricks
        b0 = torch.arange(B, device=device, dtype=torch.int32).repeat_interleave(G0 * G0)
        by0 = self.l0_by.repeat(B)
        bx0 = self.l0_bx.repeat(B)
        m0 = self.l0_codes.repeat(B)
        bmap0, nb0 = build_bmap_and_neighbors(b0, by0, bx0, B, G0)

        f0_raw = gather_bricks(pyr[0], b0, by0, bx0, self.tile_size)
        f0_emb = gather_bricks(f32_emb, b0, by0, bx0, self.tile_size)
        u0 = self.smoother.state_norm(f0_emb).to(f0_emb.dtype)
        lvl0 = QuadtreeLevel(G0, b0, by0, bx0, m0, bmap0, nb0, None, f0_raw, f0_emb, u0)
        lvl0.state = self.smoother.smooth_bricks(u0, lvl0, rel_depth=0.0, is_finest=0.0, steps=1)

        levels = [lvl0]
        split_logits_list, split_targets_list = [], []

        if self.training and gt_split_masks is not None and union_prob > 0.0:
            use_teacher = (torch.rand(B, device=device) < union_prob)
        else:
            use_teacher = torch.zeros(B, device=device, dtype=torch.bool)

        for l in range(n_lvls - 1):
            cur = levels[l]
            rel_d_cur = l / max(n_lvls - 1, 1)
            s_logits = self._predict_split_logits(cur.state, cur.f_raw, rel_d_cur)
            split_logits_list.append(s_logits)

            pred_mask = (torch.sigmoid(s_logits) > split_thresh)
            if gt_split_masks is not None:
                gt_s = gt_split_masks[l][cur.b_idx.long(), cur.by_idx.long(), cur.bx_idx.long()]
                split_targets_list.append(gt_s)
                split_mask = torch.where(use_teacher[cur.b_idx.long()], (gt_s > 0.5), pred_mask)
            else:
                split_mask = pred_mask

            if self.training and l >= 1:
                max_m = max(B, int(0.55 * cur.b_idx.numel()))
                if int(split_mask.sum().item()) > max_m:
                    topk_idx = torch.topk(s_logits, k=max_m).indices
                    cap = torch.zeros_like(split_mask)
                    cap[topk_idx] = True
                    split_mask = cap

            if not split_mask.any():
                split_mask[s_logits.argmax()] = True

            p_sel = torch.where(split_mask)[0]
            c_b, c_by, c_bx, c_morton = expand_morton_children(cur, p_sel)
            G_next = cur.grid_size * 2
            bmap_next, nb_next = build_bmap_and_neighbors(c_b, c_by, c_bx, B, G_next)

            f_raw_next = gather_bricks(pyr[l + 1], c_b, c_by, c_bx, self.tile_size)
            f_emb_next = self.init_proj(f_raw_next)
            u_next = self.smoother.state_norm(
                0.5 * f_emb_next + 0.5 * self._prolongate_to_children(cur.state, p_sel).to(f_emb_next.dtype)
            ).to(f_emb_next.dtype)

            lvl_next = QuadtreeLevel(G_next, c_b, c_by, c_bx, c_morton, bmap_next, nb_next, p_sel, f_raw_next, f_emb_next, u_next)
            rel_d = (l + 1) / max(n_lvls - 1, 1)
            is_f = 1.0 if (l + 1) == n_lvls - 1 else 0.0
            lvl_next.state = self.smoother.smooth_bricks(u_next, lvl_next, rel_d, is_f, steps=1)
            levels.append(lvl_next)

        # ---- Stage 2: Weight-Tied Multigrid V-Cycles ----
        states = [lvl.state for lvl in levels]
        cycle_logits = []
        cycles_used = 0
        n_run = int(torch.randint(2, self.n_vcycles + 2, (1,)).item()) if self.training else self.n_vcycles

        for k in range(n_run):
            prev_finest = states[-1]
            if self.use_checkpoint and self.training:
                states = list(checkpoint(
                    lambda *s: tuple(self._run_vcycle(list(s), levels, f32_emb, sub_emb)),
                    *states, use_reentrant=False
                ))
            else:
                states = self._run_vcycle(states, levels, f32_emb, sub_emb)

            cycles_used = k + 1
            if self.training:
                cycle_logits.append(self._decode_quadtree_logits(states, levels, B, H, W))
            elif adaptive_exit_eps is not None and k >= 1:
                if (states[-1] - prev_finest).abs().mean().item() < adaptive_exit_eps:
                    break

        if not self.training:
            cycle_logits.append(self._decode_quadtree_logits(states, levels, B, H, W))

        finest = levels[-1]
        active_ratio = finest.b_idx.numel() / float(B * finest.grid_size * finest.grid_size)

        return {
            "logits": cycle_logits[-1],
            "cycle_logits": cycle_logits,
            "split_logits": split_logits_list,
            "split_targets": split_targets_list,
            "finest_brick_map": finest.brick_map,
            "active_ratio": active_ratio,
            "cycles_used": cycles_used,
        }
