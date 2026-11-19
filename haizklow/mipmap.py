# haizklow/mipmap.py
# HaizkLow v2.1 — O(1) Multi-Stat + Local-Contrast + Green-Channel Multiscale Oriented Ridge Pyramid
# Channels (20): mean_RGB(3) + max_RGB(3) + min_RGB(3) + local_contrast_RGB(3) + pos_ridge_4dir(4) + neg_ridge_4dir(4)

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

    @torch.no_grad()
    def _multiscale_ridge(self, ch1):
        H, W = ch1.shape[-2:]
        pos_all = torch.zeros(ch1.shape[0], 4, H, W, device=ch1.device, dtype=ch1.dtype)
        neg_all = torch.zeros_like(pos_all)
        cur = ch1
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
        ratio = H // self.root_res
        assert H == W and H % self.root_res == 0 and ratio >= 1 and (ratio & (ratio - 1)) == 0, \
            f"H=W={H} must be root_res({self.root_res}) * 2^k"

        # Local background-subtracted RGB kills optic-disc glare & highlights central arterial reflex!
        local_bg = F.avg_pool2d(img, kernel_size=15, stride=1, padding=7)
        loc_c = img - local_bg

        # Green channel gives 3x stronger retinal vessel contrast (and equals gray on synthetic Crucible!)
        green = img[:, 1:2]
        pos_r, neg_r = self._multiscale_ridge(green)

        mean_rgb, max_rgb, min_rgb = img, img, img
        rev_levels = [torch.cat([mean_rgb, max_rgb, min_rgb, loc_c, pos_r, neg_r], dim=1)]

        cur_res = H
        while cur_res > self.root_res:
            mean_rgb = F.avg_pool2d(mean_rgb, 2, 2)
            max_rgb = F.max_pool2d(max_rgb, 2, 2)
            min_rgb = -F.max_pool2d(-min_rgb, 2, 2)
            loc_c = F.avg_pool2d(loc_c, 2, 2)
            pos_r = F.max_pool2d(pos_r, 2, 2)
            neg_r = F.max_pool2d(neg_r, 2, 2)
            rev_levels.append(torch.cat([mean_rgb, max_rgb, min_rgb, loc_c, pos_r, neg_r], dim=1))
            cur_res //= 2

        levels = list(reversed(rev_levels))
        sub_coarse = []
        for _ in range(2):
            mean_rgb = F.avg_pool2d(mean_rgb, 2, 2)
            max_rgb = F.max_pool2d(max_rgb, 2, 2)
            min_rgb = -F.max_pool2d(-min_rgb, 2, 2)
            loc_c = F.avg_pool2d(loc_c, 2, 2)
            pos_r = F.max_pool2d(pos_r, 2, 2)
            neg_r = F.max_pool2d(neg_r, 2, 2)
            sub_coarse.append(torch.cat([mean_rgb, max_rgb, min_rgb, loc_c, pos_r, neg_r], dim=1))

        return levels, sub_coarse
