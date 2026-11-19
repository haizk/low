# crucible/crossing_maze.py
# HaizkLow — Synthetic Artery-Vein Crossing Maze Crucible
#
# Key traps built in to break standard local CNNs / Half-UNet:
#   1. Zero local A/V texture cue: arteries and veins have IDENTICAL intensity & noise
#      everywhere except a tiny colored root seed (Red=Artery, Blue=Vein).
#   2. 4-20 clean arteriovenous crossings per image (both channels active simultaneously).
#   3. Tapering down to 1-pixel capillaries (tests coarse-level anti-blindness).
#   4. Supports both Resolution Extrapolation (scale=H/256) and Path-Length Extrapolation (scale=1.0).

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


def draw_vascular_tree(H, W, start_xy, init_angles, rng, scale_factor=1.0,
                       max_depth=2, max_branches=10):
    mask = np.zeros((H, W), dtype=np.uint8)
    thin_mask = np.zeros((H, W), dtype=np.uint8)

    step_sz = 3.5 * scale_factor
    base_steps = int(min(H, W) * 1.05 / step_sz)
    base_width = max(1.5, 3.6 * scale_factor)

    queue = [
        (float(start_xy[0]), float(start_xy[1]), float(ang), base_width, 0, base_steps)
        for ang in init_angles
    ]
    branches_spawned = len(queue)

    while queue:
        x, y, ang, width, depth, steps = queue.pop(0)
        curv = rng.uniform(-0.025, 0.025)
        cooldown = 7
        pts = [(x, y)]
        widths = [width]

        for s in range(steps):
            curv = 0.90 * curv + rng.normal(0.0, 0.022)
            curv = float(np.clip(curv, -0.09, 0.09))
            ang += curv

            nx = x + np.cos(ang) * step_sz
            ny = y + np.sin(ang) * step_sz
            if nx < 4 or nx >= W - 4 or ny < 4 or ny >= H - 4:
                break

            taper = 1.0 - 0.72 * (s / max(steps, 1))
            cur_w = max(1.0, width * taper)
            pts.append((nx, ny))
            widths.append(cur_w)
            x, y = nx, ny

            cooldown -= 1
            if (depth < max_depth and branches_spawned < max_branches
                    and cooldown <= 0 and rng.random() < 0.12):
                cooldown = int(rng.integers(8, 14))
                sign = float(rng.choice([-1.0, 1.0]))
                b_ang = ang + sign * rng.uniform(0.35, 0.80)
                b_width = max(1.0, cur_w * rng.uniform(0.50, 0.75))
                b_steps = int(steps * rng.uniform(0.50, 0.75))
                queue.append((x, y, b_ang, b_width, depth + 1, b_steps))
                branches_spawned += 1
                ang -= sign * rng.uniform(0.06, 0.16)

        if len(pts) >= 2:
            pts_arr = np.round(np.array(pts)).astype(np.int32)
            w_int = np.maximum(1, np.round(np.array(widths)).astype(np.int32))
            start_i = 0
            while start_i < len(pts_arr) - 1:
                t_val = int(w_int[start_i])
                end_i = start_i + 1
                while end_i < len(pts_arr) - 1 and int(w_int[end_i]) == t_val:
                    end_i += 1
                seg = pts_arr[start_i:end_i + 1]
                cv2.polylines(mask, [seg], False, 1, thickness=t_val, lineType=cv2.LINE_AA)
                if t_val == 1:
                    cv2.polylines(thin_mask, [seg], False, 1, thickness=1, lineType=cv2.LINE_8)
                start_i = end_i

    return (mask > 0), (thin_mask > 0)


def generate_crossing_maze(H=256, W=256, seed=None, min_crossings=4, max_crossings=20,
                           physical_scale=None):
    """
    physical_scale=None -> scales thickness with resolution (Resolution Extrapolation)
    physical_scale=1.0  -> keeps pixel thickness fixed at 512x512 (2x Path-Length Extrapolation!)
    """
    rng = np.random.default_rng(seed)
    scale = (H / 256.0) if physical_scale is None else float(physical_scale)
    best = None

    for _ in range(12):
        mode = int(rng.integers(0, 2))
        if mode == 0:
            a_root = (int(12 * scale), int(rng.integers(int(0.25 * H), int(0.75 * H))))
            a_angs = [rng.uniform(-0.40, -0.08), rng.uniform(0.08, 0.40)]
            v_root = (int(rng.integers(int(0.25 * W), int(0.75 * W))), H - int(12 * scale))
            v_angs = [-np.pi / 2 + rng.uniform(-0.40, -0.08), -np.pi / 2 + rng.uniform(0.08, 0.40)]
        else:
            a_root = (int(12 * scale), int(rng.integers(int(0.18 * H), int(0.36 * H))))
            a_angs = [rng.uniform(0.08, 0.32), rng.uniform(0.36, 0.65)]
            v_root = (int(12 * scale), int(rng.integers(int(0.64 * H), int(0.82 * H))))
            v_angs = [rng.uniform(-0.32, -0.08), rng.uniform(-0.65, -0.36)]

        a_mask, a_thin = draw_vascular_tree(H, W, a_root, a_angs, rng, scale_factor=scale)
        v_mask, v_thin = draw_vascular_tree(H, W, v_root, v_angs, rng, scale_factor=scale)

        if np.hypot(a_root[0] - v_root[0], a_root[1] - v_root[1]) < 45 * scale:
            continue

        overlap = (a_mask & v_mask).astype(np.uint8)
        n_comp, _ = cv2.connectedComponents(overlap)
        n_crossings = n_comp - 1
        vessel_union = a_mask | v_mask
        fg_ratio = float(vessel_union.mean())

        cand = (a_mask, v_mask, a_thin, v_thin, a_root, v_root, overlap, n_crossings, fg_ratio)
        if best is None or abs(n_crossings - 9) < abs(best[7] - 9):
            best = cand
        if min_crossings <= n_crossings <= max_crossings and 0.05 <= fg_ratio <= 0.16:
            break

    a_mask, v_mask, a_thin, v_thin, a_root, v_root, overlap, n_crossings, _ = best
    vessel_union = a_mask | v_mask

    # Smooth uneven background illumination
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cx, cy = W * rng.uniform(0.3, 0.7), H * rng.uniform(0.3, 0.7)
    bg = 0.25 + 0.15 * np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * (0.45 * H) ** 2))

    # IDENTICAL vessel appearance for Artery and Vein
    vessel_soft = cv2.GaussianBlur(vessel_union.astype(np.float32), (3, 3), 0.6)
    base = bg + rng.uniform(0.28, 0.45) * vessel_soft
    img = np.stack([base, base, base], axis=0)

    # Tiny colored root cues ONLY (Red = Artery, Blue = Vein)
    r_cue = max(4, int(round(6 * scale)))
    a_cue = np.zeros((H, W), dtype=np.uint8)
    v_cue = np.zeros((H, W), dtype=np.uint8)
    cv2.circle(a_cue, a_root, r_cue, 1, -1)
    cv2.circle(v_cue, v_root, r_cue, 1, -1)

    img[0] = np.where(a_cue > 0, np.clip(img[0] + 0.55, 0, 1), img[0])
    img[1] = np.where(a_cue > 0, img[1] * 0.35, img[1])
    img[2] = np.where(a_cue > 0, img[2] * 0.35, img[2])

    img[2] = np.where(v_cue > 0, np.clip(img[2] + 0.55, 0, 1), img[2])
    img[0] = np.where(v_cue > 0, img[0] * 0.35, img[0])
    img[1] = np.where(v_cue > 0, img[1] * 0.35, img[1])

    img = np.clip(img + rng.normal(0.0, 0.04, size=img.shape).astype(np.float32), 0.0, 1.0)

    k = max(3, int(round(7 * scale)) | 1)
    crossing_zone = cv2.dilate(overlap, np.ones((k, k), np.uint8)) > 0
    thin_zone = (a_thin | v_thin) & (~crossing_zone)
    target = np.stack([a_mask.astype(np.float32), v_mask.astype(np.float32)], axis=0)

    return img, target, crossing_zone, thin_zone, n_crossings


def compute_gt_split_masks(target_2hw, tile_size=4, root_bricks=8):
    """Returns GT active brick masks [L0 (root), L1, ..., L_finest] for split BCE supervision."""
    H, W = target_2hw.shape[1], target_2hw.shape[2]
    fg = (target_2hw.sum(axis=0) > 0).astype(np.uint8)
    finest_g = H // tile_size
    assert H == W and (finest_g & (finest_g - 1)) == 0, "H=W must be power of 2"

    cur = fg.reshape(finest_g, tile_size, finest_g, tile_size).max(axis=(1, 3))
    levels = [cur]
    while cur.shape[0] > root_bricks:
        g = cur.shape[0] // 2
        cur = cur.reshape(g, 2, g, 2).max(axis=(1, 3))
        levels.append(cur)
    levels.reverse()
    return levels


class CrossingMazeDataset(Dataset):
    def __init__(self, n_samples=500, size=256, base_seed=0, physical_scale=None):
        self.n_samples = n_samples
        self.size = size
        self.base_seed = base_seed
        self.physical_scale = physical_scale

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        img, target, crossing_zone, thin_zone, _ = generate_crossing_maze(
            self.size, self.size, seed=self.base_seed + idx,
            physical_scale=self.physical_scale
        )
        return {
            "image": torch.from_numpy(img),                              # [3, H, W]
            "target": torch.from_numpy(target),                          # [2, H, W] (Artery, Vein)
            "crossing_zone": torch.from_numpy(crossing_zone),            # [H, W] bool
            "thin_zone": torch.from_numpy(thin_zone),                    # [H, W] bool
        }


@torch.no_grad()
def evaluate_crucible(logits, target, crossing_zone, thin_zone, thresh=0.5):
    """
    logits, target: [B, 2, H, W]
    crossing_zone, thin_zone: [B, H, W] bool
    """
    pred = (torch.sigmoid(logits) > thresh).float()
    eps = 1e-6

    # 1. Overall A/V mean Dice
    inter = (pred * target).sum(dim=(0, 2, 3))
    denom = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + eps
    av_dice = ((2.0 * inter + eps) / denom).mean().item()

    # 2. Crossing-Zone A/V F1 (the real Crossing Paradox killer test!)
    cz = crossing_zone.unsqueeze(1).expand_as(pred)
    cz_inter = (pred[cz] * target[cz]).sum()
    cz_denom = pred[cz].sum() + target[cz].sum() + eps
    crossing_f1 = ((2.0 * cz_inter + eps) / cz_denom).item()

    # 3. Thin 1px Capillary Recall (tests Anti-Blindness!)
    tz = thin_zone.unsqueeze(1).expand_as(pred)
    tz_tp = (pred[tz] * target[tz]).sum()
    tz_pos = target[tz].sum() + eps
    thin_recall = (tz_tp / tz_pos).item()

    return {
        "av_dice": av_dice,
        "crossing_f1": crossing_f1,
        "thin_recall": thin_recall,
    }


if __name__ == "__main__":
    for res in [256, 512]:
        img, tgt, cz, tz, nc = generate_crossing_maze(res, res, seed=7)
        lvls = compute_gt_split_masks(tgt, tile_size=4, root_bricks=8)
        fg_pct = (tgt.sum(0) > 0).mean() * 100.0
        print(f"\n[Crucible {res}x{res}] FG={fg_pct:.1f}% | Crossings={nc} | Thin1px={tz.mean()*100:.2f}%")
        for i, m in enumerate(lvls):
            g = m.shape[0]
            print(f"  L{i} ({g:3d}x{g:<3d} bricks = {g*4:3d}x{g*4:<3d}px): "
                  f"{m.sum():5d}/{g*g:5d} active ({m.mean()*100:5.1f}%)")