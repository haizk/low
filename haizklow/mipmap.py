# haizklow/mipmap.py
# HaizkLow — O(1) Multi-Stat + SE(2) Oriented Ridge Image Pyramid
#
# Prevents coarse-level 1px capillary blindness & root-seed dilution by pooling:
#   mean_RGB (3), max_RGB (3), min_RGB (3), pos_ridge_k (1), neg_ridge_k (1), max_ridge_all (1)
# -> 12 forcing channels per orientation fiber k in {0:0deg, 1:45deg, 2:90deg, 3:135deg}.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStatSE2Pyramid(nn.Module):
    def __init__(self, tile_size=4, root_bricks=8):
        super().__init__()
        self.tile_size = tile_size
        self.root_bricks = root_bricks
        self.root_res = tile_size * root_bricks  # 32 px default

        # 4 fixed directional 2nd-derivative ridge stencils matching SE(2) fibers:
        # k=0: 0 deg (H), k=1: 45 deg (NE-SW), k=2: 90 deg (V), k=3: 135 deg (NW-SE)
        k0 = torch.tensor([
            [-0.5, -1.0, -0.5],
            [ 1.0,  2.0,  1.0],
            [-0.5, -1.0, -0.5],
        ], dtype=torch.float32) / 4.0
        k90 = k0.t().contiguous()
        k45 = torch.tensor([
            [-1.0, -0.5,  1.0],
            [-0.5,  2.0, -0.5],
            [ 1.0, -0.5, -1.0],
        ], dtype=torch.float32) / 4.0
        k135 = torch.flip(k45, dims=[1]).contiguous()

        kernels = torch.stack([k0, k45, k90, k135], dim=0).unsqueeze(1)  # [4, 1, 3, 3]
        self.register_buffer("ridge_kernels", kernels)
        self.register_buffer("luma_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _build_level_tensor(self, mean_rgb, max_rgb, min_rgb, pos_r, neg_r):
        # mean/max/min: [B, 3, H, W]; pos_r/neg_r: [B, 4, H, W]
        B, _, H, W = mean_rgb.shape
        shared = torch.cat([mean_rgb, max_rgb, min_rgb], dim=1)         # [B, 9, H, W]
        shared = shared.unsqueeze(1).expand(B, 4, 9, H, W)              # [B, 4, 9, H, W]
        max_all = torch.maximum(pos_r, neg_r).max(dim=1, keepdim=True).values  # [B, 1, H, W]
        max_all = max_all.unsqueeze(1).expand(B, 4, 1, H, W)            # [B, 4, 1, H, W]
        fiber_r = torch.stack([pos_r, neg_r], dim=2)                    # [B, 4, 2, H, W]
        return torch.cat([shared, fiber_r, max_all], dim=2)             # [B, 4, 12, H, W]

    @torch.no_grad()
    def _multiscale_ridge(self, lum):
        """Computes 3x3 oriented ridge at scales 1, 2, 4 so 1px, 3px, and 6px vessels ALL fire!"""
        H, W = lum.shape[-2:]
        pos_all = torch.zeros(lum.shape[0], 4, H, W, device=lum.device, dtype=lum.dtype)
        neg_all = torch.zeros_like(pos_all)
        cur = lum
        for s in (1, 2, 4):
            if s > 1:
                cur = F.avg_pool2d(cur, 2, 2)
            r = F.conv2d(cur, self.ridge_kernels, padding=1)
            rp, rn = F.relu(r), F.relu(-r)
            if s > 1:
                rp = F.interpolate(rp, size=(H, W), mode="bilinear", align_corners=False)
                rn = F.interpolate(rn, size=(H, W), mode="bilinear", align_corners=False)
            pos_all = torch.maximum(pos_all, rp)
            neg_all = torch.maximum(neg_all, rn)
        return pos_all, neg_all

    @torch.no_grad()
    def forward(self, img):
        B, C, H, W = img.shape
        assert H == W and H >= self.root_res and (H & (H - 1)) == 0, "H=W must be power of 2 >= 32"

        lum = (img * self.luma_w).sum(dim=1, keepdim=True)
        pos_r, neg_r = self._multiscale_ridge(lum)

        mean_rgb, max_rgb, min_rgb = img, img, img
        rev_levels = [self._build_level_tensor(mean_rgb, max_rgb, min_rgb, pos_r, neg_r)]

        cur_res = H
        while cur_res > self.root_res:
            mean_rgb = F.avg_pool2d(mean_rgb, 2, 2)
            max_rgb = F.max_pool2d(max_rgb, 2, 2)
            min_rgb = -F.max_pool2d(-min_rgb, 2, 2)
            pos_r = F.max_pool2d(pos_r, 2, 2)
            neg_r = F.max_pool2d(neg_r, 2, 2)
            rev_levels.append(self._build_level_tensor(mean_rgb, max_rgb, min_rgb, pos_r, neg_r))
            cur_res //= 2

        levels = list(reversed(rev_levels))
        sub_coarse = []
        for _ in range(2):
            mean_rgb = F.avg_pool2d(mean_rgb, 2, 2)
            max_rgb = F.max_pool2d(max_rgb, 2, 2)
            min_rgb = -F.max_pool2d(-min_rgb, 2, 2)
            pos_r = F.max_pool2d(pos_r, 2, 2)
            neg_r = F.max_pool2d(neg_r, 2, 2)
            sub_coarse.append(self._build_level_tensor(mean_rgb, max_rgb, min_rgb, pos_r, neg_r))

        return levels, sub_coarse
