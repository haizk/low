# haizklow/se2_multigrid.py
# HaizkLow v3 — Hardware-Native NHWC D4-Equivariant SE(2) Fiber-Bundle Multigrid PDE Engine

from typing import List, Optional, Tuple
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
    fused_se2_laplacian_ref,
    gather_bricks_nhwc,
    make_l0_morton_tables,
    prolong_bricks,
    restrict_bricks,
    scatter_bricks_nhwc,
)
from .ops import fused_se2_laplacian


def _rms_norm_last(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if hasattr(F, "rms_norm"):
        return F.rms_norm(x, (x.shape[-1],), weight.to(x.dtype), eps)
    rms = torch.rsqrt(x.to(torch.float32).pow(2).mean(dim=-1, keepdim=True) + eps).to(x.dtype)
    return x * rms * weight.to(x.dtype)


def _noisy_or_logits(z_fibers: torch.Tensor) -> torch.Tensor:
    """Exact log-space Noisy-OR across K=4 orientation fibers: P(c) = 1 - prod_k (1 - sigmoid(z_k))."""
    S = F.softplus(z_fibers).sum(dim=-2)
    return torch.where(
        S > 1.0,
        S + torch.log1p(-torch.exp(-S.clamp_max(40.0))),
        torch.log(torch.expm1(S.clamp_min(1e-6))),
    )


class SE2FiberStem(nn.Module):
    """D4-Equivariant per-fiber image stem (computed ONCE per level, zero 4x isotropic copy)."""
    def __init__(self, fiber_dim: int = 16):
        super().__init__()
        h = max(32, 2 * fiber_dim)
        self.iso_proj = nn.Linear(12, h, bias=True)
        self.ori_proj = nn.Linear(6, h, bias=False)
        self.out_proj = nn.Linear(h, fiber_dim, bias=True)
        self.gate_bias_proj = nn.Linear(fiber_dim, 2, bias=True)
        nn.init.constant_(self.gate_bias_proj.bias[0], 0.0)
        nn.init.constant_(self.gate_bias_proj.bias[1], -1.0)

    def forward(self, f_raw: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f_iso = f_raw[..., :12]
        pos = f_raw[..., 12:16]
        neg = f_raw[..., 16:20]

        pos_adj = torch.roll(pos, 1, dims=-1) + torch.roll(pos, -1, dims=-1)
        neg_adj = torch.roll(neg, 1, dims=-1) + torch.roll(neg, -1, dims=-1)
        pos_orth = torch.roll(pos, 2, dims=-1)
        neg_orth = torch.roll(neg, 2, dims=-1)
        f_ori = torch.stack([pos, neg, pos_adj, neg_adj, pos_orth, neg_orth], dim=-1)

        h = F.gelu(self.iso_proj(f_iso).unsqueeze(-2) + self.ori_proj(f_ori))
        f_emb = self.out_proj(h)
        f_gate_bias = self.gate_bias_proj(f_emb)
        return f_emb, f_gate_bias


class SE2FiberSmoother(nn.Module):
    """D4-Equivariant SE(2) Gated Reaction-Diffusion Smoother (NHWC Tensor-Core + Fused SRAM)."""
    def __init__(self, fiber_dim: int = 16):
        super().__init__()
        D = fiber_dim
        H_mlp = max(32, 2 * D)

        self.u_to_gate = nn.Linear(D, 2, bias=False)
        self.pair_to_c = nn.Linear(D, 1, bias=False)

        self.dt_s_logit = nn.Parameter(torch.full((D,), 0.0))
        self.dt_c_logit = nn.Parameter(torch.full((D,), -0.8))

        self.norm_w = nn.Parameter(torch.ones(D))
        self.w1_u = nn.Linear(D, H_mlp, bias=True)
        self.w1_f = nn.Linear(D, H_mlp, bias=False)
        self.w2 = nn.Linear(H_mlp, D, bias=True)

        self.scale_film = nn.Sequential(
            nn.Linear(2, 16),
            nn.GELU(),
            nn.Linear(16, 2 * D),
        )
        nn.init.zeros_(self.scale_film[-1].weight)
        nn.init.zeros_(self.scale_film[-1].bias)

        self.alpha_logit = nn.Parameter(torch.full((D,), 0.0))

    def step_bricks(self, u: torch.Tensor, lvl: QuadtreeLevel, s_vec: torch.Tensor) -> torch.Tensor:
        # 1. Collinear continuation prob a [N,4,4,4] & +-45 deg turning prob c [N,4,4,4]
        g = self.u_to_gate(u) + lvl.f_gate_bias
        a = torch.sigmoid(g[..., 0])
        u_next_k = torch.roll(u, -1, dims=-2)
        c = torch.sigmoid(g[..., 1] + self.pair_to_c((u - u_next_k).abs()).squeeze(-1))

        # 2. 1-shot fused SRAM SE(2) Graph Laplacian (dt <= 0.25 -> strict L_inf contraction!)
        dt_s = 0.25 * torch.sigmoid(self.dt_s_logit)
        dt_c = 0.25 * torch.sigmoid(self.dt_c_logit)
        if u.is_cuda and u.shape[1] == 4:
            u_diff = fused_se2_laplacian(u, a, c, dt_s, dt_c, lvl.nb_idx)
        else:
            u_diff = fused_se2_laplacian_ref(u, a, c, dt_s, dt_c, lvl.nb_idx)

        # 3. Per-fiber Fisher-KPP Reaction MLP (cuBLAS HGEMM, zero torch.cat, zero H2D!)
        u_n = _rms_norm_last(u_diff, self.norm_w)
        h = F.gelu(self.w1_u(u_n) + self.w1_f(lvl.f_emb))
        du = self.w2(h)

        gamma, beta = self.scale_film(s_vec).chunk(2)
        du = du * (1.0 + gamma) + beta

        # 4. Bounded contractive convex update in [-1, 1]
        alpha = 0.5 * torch.sigmoid(self.alpha_logit).to(u.dtype)
        return (1.0 - alpha) * u_diff + alpha * torch.tanh(u_diff + du)


class CoarseGlobalSolver(nn.Module):
    """Fast Global Context Mixer on the tiny coarsest root grid (8x8 / 10x10 -> <0.15ms)."""
    def __init__(self, total_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(total_ch, total_ch, 3, padding=1, groups=total_ch),
            nn.Conv2d(total_ch, total_ch, 1),
            nn.GELU(),
            nn.Conv2d(total_ch, total_ch, 3, padding=1, groups=total_ch),
            nn.Conv2d(total_ch, total_ch, 1),
        )
        nn.init.zeros_(self.block[-1].weight)
        nn.init.zeros_(self.block[-1].bias)

    def forward(self, u_dense: torch.Tensor) -> torch.Tensor:
        u_tiny = F.avg_pool2d(u_dense, 4, 4)
        delta_tiny = self.block(u_tiny)
        delta = F.interpolate(delta_tiny, size=u_dense.shape[-2:], mode="bilinear", align_corners=False)
        return torch.tanh(u_dense + delta)


class HaizkLow(nn.Module):
    def __init__(self, ch: int = 64, out_ch: int = 2, n_vcycles: int = 2,
                 smooth_steps: int = 1, tile_size: int = 4, root_bricks: int = 8,
                 use_checkpoint: bool = False, split_thresh: float = 0.25, **kwargs):
        super().__init__()
        assert ch % 4 == 0, "ch must be divisible by K=4 SE(2) orientation fibers!"
        self.ch = ch
        self.K = 4
        self.D = ch // 4
        self.out_ch = out_ch
        self.n_vcycles = n_vcycles
        self.smooth_steps = smooth_steps
        self.tile_size = tile_size
        self.root_bricks = root_bricks
        self.use_checkpoint = use_checkpoint
        self.default_split_thresh = split_thresh

        self.pyramid = MultiStatSE2Pyramid(tile_size=tile_size, root_bricks=root_bricks)
        self.stem = SE2FiberStem(fiber_dim=self.D)
        self.smoother = SE2FiberSmoother(fiber_dim=self.D)
        self.coarse_solver = CoarseGlobalSolver(total_ch=ch)
        self.restrict_blend = nn.Parameter(torch.tensor(0.0))

        self.split_head = nn.Sequential(
            nn.Linear(2 * self.D + 20 + 1, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )
        nn.init.constant_(self.split_head[-1].bias, 0.0)

        h_head = max(32, 2 * self.D)
        self.head_u = nn.Linear(self.D, h_head, bias=True)
        self.head_f = nn.Linear(self.D, h_head, bias=False)
        self.head_out = nn.Linear(h_head, out_ch, bias=True)
        nn.init.constant_(self.head_out.bias, -2.5)

        morton_perm, inv_perm, l0_by, l0_bx, sorted_codes = make_l0_morton_tables(root_bricks)
        self.register_buffer("l0_morton_perm", morton_perm)
        self.register_buffer("l0_inv_perm", inv_perm)
        self.register_buffer("l0_by", l0_by)
        self.register_buffer("l0_bx", l0_bx)
        self.register_buffer("l0_codes", sorted_codes)

    def _predict_split_logits(self, u: torch.Tensor, f_raw: torch.Tensor, rel_depth: float) -> torch.Tensor:
        u_mean = u.mean(dim=(1, 2, 3))
        u_max = u.amax(dim=(1, 2, 3))
        f_max = f_raw.amax(dim=(1, 2))
        depth_col = torch.full((u.shape[0], 1), rel_depth, device=u.device, dtype=u.dtype)
        return self.split_head(torch.cat([u_mean, u_max, f_max, depth_col], dim=-1)).squeeze(-1)

    def _restrict_to_parent(self, child_u: torch.Tensor, parent_u: torch.Tensor,
                            p_sel: torch.Tensor) -> torch.Tensor:
        r = restrict_bricks(child_u).to(parent_u.dtype)
        gam = torch.sigmoid(self.restrict_blend).to(parent_u.dtype)
        new_parent = parent_u.clone()
        new_parent[p_sel] = (1.0 - gam) * parent_u[p_sel] + gam * r
        return new_parent

    def _l0_bricks_to_dense_nchw(self, u0: torch.Tensor, B: int) -> torch.Tensor:
        G0, T, C = self.root_bricks, self.tile_size, self.ch
        rm = u0.view(B, G0 * G0, T, T, C)[:, self.l0_inv_perm]
        return rm.view(B, G0, G0, T, T, C).permute(0, 5, 1, 3, 2, 4).reshape(B, C, G0 * T, G0 * T)

    def _dense_nchw_to_l0_bricks(self, dense: torch.Tensor) -> torch.Tensor:
        B, C, H, W = dense.shape
        G0, T = self.root_bricks, self.tile_size
        rm = dense.view(B, C, G0, T, G0, T).permute(0, 2, 4, 3, 5, 1).reshape(B, G0 * G0, T, T, C)
        return rm[:, self.l0_morton_perm].reshape(B * G0 * G0, T, T, self.K, self.D)

    def _run_vcycle(self, states: List[torch.Tensor], levels: List[QuadtreeLevel],
                    scale_vecs: torch.Tensor, B: int) -> List[torch.Tensor]:
        n_lvls = len(levels)
        states = list(states)
        saved_parents: List[Optional[torch.Tensor]] = [None] * n_lvls

        for l in range(n_lvls - 1, 0, -1):
            for _ in range(self.smooth_steps):
                states[l] = self.smoother.step_bricks(states[l], levels[l], scale_vecs[l])
            states[l - 1] = self._restrict_to_parent(states[l], states[l - 1], levels[l].parent_sel)
            saved_parents[l - 1] = states[l - 1]

        states[0] = self.smoother.step_bricks(states[0], levels[0], scale_vecs[0])
        u0_dense = self.coarse_solver(self._l0_bricks_to_dense_nchw(states[0], B))
        states[0] = self._dense_nchw_to_l0_bricks(u0_dense)
        states[0] = self.smoother.step_bricks(states[0], levels[0], scale_vecs[0])

        for l in range(1, n_lvls):
            coarse_delta = states[l - 1] - saved_parents[l - 1]
            delta_fine = prolong_bricks(coarse_delta[levels[l].parent_sel]).to(states[l].dtype)
            states[l] = torch.tanh(states[l] + delta_fine)
            for _ in range(self.smooth_steps):
                states[l] = self.smoother.step_bricks(states[l], levels[l], scale_vecs[l])

        return states

    def _decode_quadtree_logits(self, states: List[torch.Tensor], levels: List[QuadtreeLevel],
                                B: int, H: int, W: int) -> torch.Tensor:
        device = states[0].device
        T = self.tile_size
        out = torch.zeros(B, self.out_ch, H, W, device=device, dtype=torch.float32)
        covered = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)

        for l in range(len(levels) - 1, -1, -1):
            lvl = levels[l]
            h = F.gelu(self.head_u(states[l]) + self.head_f(lvl.f_emb))
            z_fibers = self.head_out(h).float()
            brick_logits = _noisy_or_logits(z_fibers)

            G = lvl.grid_size
            lvl_logits, lvl_mask = scatter_bricks_nhwc(brick_logits, lvl.b_idx, lvl.by_idx, lvl.bx_idx, B, G, T)

            if G * T != H:
                lvl_logits = F.interpolate(lvl_logits, size=(H, W), mode="nearest")
                lvl_mask = F.interpolate(lvl_mask, size=(H, W), mode="nearest")

            take = lvl_mask * (1.0 - covered)
            out = out + lvl_logits * take
            covered = torch.clamp(covered + take, 0.0, 1.0)
        return out

    def forward(self, img: torch.Tensor, gt_split_masks: Optional[List[torch.Tensor]] = None,
                union_prob: float = 0.0, split_thresh: Optional[float] = None,
                adaptive_exit_eps: Optional[float] = None,
                force_split_masks: Optional[List[torch.Tensor]] = None):
        B, _, H, W = img.shape
        device = img.device
        thresh = self.default_split_thresh if split_thresh is None else split_thresh

        pyr, _ = self.pyramid(img)
        n_lvls = len(pyr)

        # ---- Stage 1: Coarse-to-Fine Adaptive Quadtree Growth ----
        G0 = self.root_bricks
        b0 = torch.arange(B, device=device, dtype=torch.int32).repeat_interleave(G0 * G0)
        by0 = self.l0_by.repeat(B)
        bx0 = self.l0_bx.repeat(B)
        m0 = self.l0_codes.repeat(B)
        bmap0, nb0 = build_bmap_and_neighbors(b0, by0, bx0, B, G0)

        f0_raw = gather_bricks_nhwc(pyr[0], b0, by0, bx0, self.tile_size)
        f0_emb, f0_gb = self.stem(f0_raw)
        u0 = torch.tanh(f0_emb)

        # Precompute per-level (rel_depth, is_finest) on-GPU ONCE (zero H2D inside V-cycles!)
        rel_d = torch.linspace(0.0, 1.0, steps=max(n_lvls, 2), device=device, dtype=u0.dtype)[:n_lvls]
        is_f = torch.zeros(n_lvls, device=device, dtype=u0.dtype)
        is_f[-1] = 1.0
        scale_vecs = torch.stack([rel_d, is_f], dim=-1)

        lvl0 = QuadtreeLevel(G0, b0, by0, bx0, m0, bmap0, nb0, None, f0_raw, f0_emb, f0_gb, u0)
        lvl0.state = self.smoother.step_bricks(u0, lvl0, scale_vecs[0])

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

            if force_split_masks is not None:
                split_mask = (force_split_masks[l][cur.b_idx.long(), cur.by_idx.long(), cur.bx_idx.long()] > 0.5)
            else:
                pred_mask = (torch.sigmoid(s_logits) > thresh)
                if gt_split_masks is not None:
                    gt_s = gt_split_masks[l][cur.b_idx.long(), cur.by_idx.long(), cur.bx_idx.long()]
                    split_targets_list.append(gt_s)
                    split_mask = torch.where(use_teacher[cur.b_idx.long()], (gt_s > 0.5), pred_mask)
                else:
                    split_mask = pred_mask

            if self.training and l >= 1 and force_split_masks is None:
                max_m = max(B, int(0.55 * cur.b_idx.numel()))
                if int(split_mask.sum().item()) > max_m:
                    topk_idx = torch.topk(s_logits, k=max_m).indices
                    cap = torch.zeros_like(split_mask)
                    cap[topk_idx] = True
                    split_mask = cap

            p_sel = torch.where(split_mask)[0]
            if p_sel.numel() == 0:
                p_sel = s_logits.argmax().unsqueeze(0)

            c_b, c_by, c_bx, c_morton = expand_morton_children(cur, p_sel)
            G_next = cur.grid_size * 2
            bmap_next, nb_next = build_bmap_and_neighbors(c_b, c_by, c_bx, B, G_next)

            f_raw_next = gather_bricks_nhwc(pyr[l + 1], c_b, c_by, c_bx, self.tile_size)
            f_emb_next, f_gb_next = self.stem(f_raw_next)
            p_up = prolong_bricks(cur.state[p_sel]).to(f_emb_next.dtype)
            u_next = torch.tanh(0.5 * f_emb_next + 0.5 * p_up)

            lvl_next = QuadtreeLevel(
                G_next, c_b, c_by, c_bx, c_morton, bmap_next, nb_next, p_sel,
                f_raw_next, f_emb_next, f_gb_next, u_next
            )
            if l + 1 < n_lvls - 1:
                lvl_next.state = self.smoother.step_bricks(u_next, lvl_next, scale_vecs[l + 1])
            levels.append(lvl_next)

        # ---- Stage 2: Weight-Tied SE(2) Multigrid V-Cycles ----
        states = [lvl.state for lvl in levels]
        cycle_logits = []
        cycles_used = 0
        n_run = int(torch.randint(max(1, self.n_vcycles), self.n_vcycles + 2, (1,)).item()) if self.training else self.n_vcycles

        for k in range(n_run):
            prev_finest = states[-1]
            if self.use_checkpoint and self.training:
                states = list(checkpoint(
                    lambda *s: tuple(self._run_vcycle(list(s), levels, scale_vecs, B)),
                    *states, use_reentrant=False
                ))
            else:
                states = self._run_vcycle(states, levels, scale_vecs, B)

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
