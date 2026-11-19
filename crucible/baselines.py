# crucible/baselines.py
# Faithful Half-UNet baseline (Lu et al., 2022) for apples-to-apples Crucible comparison.

import torch
import torch.nn as nn
import torch.nn.functional as F


class GhostModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3):
        super().__init__()
        init_ch = out_ch // 2
        cheap_ch = out_ch - init_ch
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_ch, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm2d(init_ch),
            nn.ReLU(inplace=True),
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_ch, cheap_ch, 3, padding=1, groups=init_ch, bias=False),
            nn.BatchNorm2d(cheap_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        return torch.cat([x1, x2], dim=1)


class GhostBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            GhostModule(in_ch, out_ch),
            GhostModule(out_ch, out_ch),
        )

    def forward(self, x):
        return self.net(x)


class HalfUNet(nn.Module):
    def __init__(self, in_ch: int = 3, out_ch: int = 2, base_ch: int = 32):
        super().__init__()
        c = base_ch
        self.e1 = GhostBlock(in_ch, c)
        self.e2 = GhostBlock(c, c)
        self.e3 = GhostBlock(c, c)
        self.e4 = GhostBlock(c, c)
        self.e5 = GhostBlock(c, c)
        self.pool = nn.MaxPool2d(2, 2)
        self.dec = GhostBlock(c, c)
        self.head = nn.Conv2d(c, out_ch, 1)

    def forward(self, x):
        H, W = x.shape[-2:]
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        s4 = self.e4(self.pool(s3))
        s5 = self.e5(self.pool(s4))

        fused = (
            s1
            + F.interpolate(s2, size=(H, W), mode="bilinear", align_corners=False)
            + F.interpolate(s3, size=(H, W), mode="bilinear", align_corners=False)
            + F.interpolate(s4, size=(H, W), mode="bilinear", align_corners=False)
            + F.interpolate(s5, size=(H, W), mode="bilinear", align_corners=False)
        )
        return {"logits": self.head(self.dec(fused))}
