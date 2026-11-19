# haizklow/mipmap.py
# HaizkLow v2 — O(1) Flat Multi-Stat + Multiscale Oriented Ridge Pyramid ([B, 17, H_l, W_l])
# Channels: mean_RGB (3) + max_RGB (3) + min_RGB (3) + pos_ridge_4dir (4) + neg_ridge_4dir (4) = 17.

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiStatSE2Pyramid(nn.Module):
    def __init__(self, tile_size=4, root_bricks=8):
        super().__init__()
        self.tile_size = tile_size
        self.root_bricks = root_bricks
        self.root_res = tile_size * root_bricks

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

        self.register_buffer("ridge_kernels", torch.stack([k0, k45, k90, k135], dim=0).unsqueeze(1))
        self.register_buffer("luma_w", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))

    @torch.no_grad()
    def _multiscale_ridge(self, lum):
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
        rev_levels = [torch.cat([mean_rgb, max_rgb, min_rgb, pos_r, neg_r], dim=1)]

        cur_res = H
        while cur_res > self.root_res:
            mean_rgb = F.avg_pool2d(mean_rgb, 2, 2)
            max_rgb = F.max_pool2d(max_rgb, 2, 2)
            min_rgb = -F.max_pool2d(-min_rgb, 2, 2)
            pos_r = F.max_pool2d(pos_r, 2, 2)
            neg_r = F.max_pool2d(neg_r, 2, 2)
            rev_levels.append(torch.cat([mean_rgb, max_rgb, min_rgb, pos_r, neg_r], dim=1))
            cur_res //= 2

        levels = list(reversed(rev_levels))
        sub_coarse = []
        for _ in range(2):
            mean_rgb = F.avg_pool2d(mean_rgb, 2, 2)
            max_rgb = F.max_pool2d(max_rgb, 2, 2)
            min_rgb = -F.max_pool2d(-min_rgb, 2, 2)
            pos_r = F.max_pool2d(pos_r, 2, 2)
            neg_r = F.max_pool2d(neg_r, 2, 2)
            sub_coarse.append(torch.cat([mean_rgb, max_rgb, min_rgb, pos_r, neg_r], dim=1))

        return levels, sub_coarse
