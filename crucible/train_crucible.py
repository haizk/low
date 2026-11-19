# crucible/train_crucible.py
# Head-to-head Crucible Showdown: Half-UNet vs. HaizkLow (Phase 1)
# Run on Kaggle T4: python crucible/train_crucible.py

import os
import sys
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crucible.crossing_maze import CrossingMazeDataset, evaluate_crucible
from crucible.baselines import HalfUNet
from haizklow.se2_multigrid import HaizkLow, batch_gt_split_masks


def av_seg_loss(logits, target):
    bce = F.binary_cross_entropy_with_logits(logits, target)
    prob = torch.sigmoid(logits)
    inter = (prob * target).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3)) + 1e-6
    dice = 1.0 - (2.0 * inter + 1e-6) / denom
    return bce + dice.mean()


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def benchmark_model(model, loader, device, is_haizklow=False, n_vcycles_eval=None):
    model.eval()
    orig_k = getattr(model, "n_vcycles", None)
    if is_haizklow and n_vcycles_eval is not None:
        model.n_vcycles = n_vcycles_eval

    totals = {
        "av_dice": 0.0,
        "av_sep": 0.0,
        "crossing_f1": 0.0,
        "thin_f1": 0.0,
        "vessel_dice": 0.0,
        "active_pct": 0.0,
    }
    n_batches = 0
    t_ms = 0.0

    for batch in loader:
        img = batch["image"].to(device)
        tgt = batch["target"].to(device)
        cz = batch["crossing_zone"].to(device)
        tz = batch["thin_zone"].to(device)

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ms += (time.perf_counter() - t0) * 1000.0 / img.shape[0]

        # Pure Blood Vessel Segmentation Dice (Union: Artery | Vein)
        prob = torch.sigmoid(out["logits"])
        pred_v = (prob.max(dim=1).values > 0.5).float()
        tgt_v = (tgt.max(dim=1).values > 0.5).float()
        v_inter = (pred_v * tgt_v).sum()
        totals["vessel_dice"] += ((2.0 * v_inter + 1e-6) / (pred_v.sum() + tgt_v.sum() + 1e-6)).item()

        # Honest Thin1px F1 (on vessels inside thin_zone, not inflated Recall!)
        tz2 = tz.unsqueeze(1).expand_as(tgt)
        pred_av = (prob > 0.5).float()
        t_inter = (pred_av[tz2] * tgt[tz2]).sum()
        totals["thin_f1"] += ((2.0 * t_inter + 1e-6) / (pred_av[tz2].sum() + tgt[tz2].sum() + 1e-6)).item()

        m = evaluate_crucible(out["logits"], tgt, cz, tz)
        for k in ("av_dice", "crossing_f1"):
            totals[k] += m[k]
        totals["active_pct"] += out.get("active_ratio", 1.0) * 100.0

        a_only = (tgt[:, 0] > 0.5) & (tgt[:, 1] < 0.5)
        v_only = (tgt[:, 1] > 0.5) & (tgt[:, 0] < 0.5)
        corr = (out["logits"][:, 0][a_only] > out["logits"][:, 1][a_only]).float().sum() + \
               (out["logits"][:, 1][v_only] > out["logits"][:, 0][v_only]).float().sum()
        totals["av_sep"] += (corr / (a_only.sum() + v_only.sum() + 1e-6)).item()

        n_batches += 1

    if is_haizklow and orig_k is not None:
        model.n_vcycles = orig_k

    return {k: v / n_batches for k, v in totals.items()} | {"ms_per_img": t_ms / n_batches}


def train_one(name, model, train_loader, val_loader, device, epochs=25, lr=2e-3, is_haizklow=False):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))
    print(f"\n================ Training {name} ({count_params(model):,} params) ================")

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        act_sum = 0.0
        # Union Warmup schedule: 1.0 -> 0.0 cleanly over first 35% of epochs
        union_prob = max(0.0, 1.0 - (ep - 1) / max(1, int(0.35 * epochs)))

        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = batch["target"].to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                if is_haizklow:
                    gt_splits = batch_gt_split_masks(tgt, model.tile_size, model.root_bricks)
                    out = model(img, gt_split_masks=gt_splits, union_prob=union_prob)

                    # Deep supervision across all V-cycles (later cycles weighted heavier)
                    n_c = len(out["cycle_logits"])
                    weights = [(i + 1) / n_c for i in range(n_c)]
                    w_sum = sum(weights)
                    seg_loss = sum(w * av_seg_loss(lg, tgt) for w, lg in zip(weights, out["cycle_logits"])) / w_sum

                    # Split BCE supervision across Quadtree levels
                    split_loss = sum(
                        F.binary_cross_entropy_with_logits(sl.float(), st.float())
                        for sl, st in zip(out["split_logits"], out["split_targets"])
                    ) / len(out["split_logits"])

                    loss = seg_loss + 1.0 * split_loss
                    act_sum += out["active_ratio"] * 100.0
                else:
                    out = model(img)
                    loss = av_seg_loss(out["logits"], tgt)
                    act_sum += 100.0

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()

        sched.step()
        if ep == 1 or ep % 5 == 0 or ep == epochs:
            val = benchmark_model(model, val_loader, device, is_haizklow=is_haizklow)
            print(
                f"Ep {ep:02d}/{epochs} | loss={loss_sum/len(train_loader):.4f} | "
                f"Vessel={val['vessel_dice']:.4f} | AV_Dice={val['av_dice']:.4f} | AV_Sep={val['av_sep']:.4f} | "
                f"Cross_F1={val['crossing_f1']:.4f} | Thin_F1={val['thin_f1']:.4f} | "
                f"Active={val['active_pct']:.1f}% | {val['ms_per_img']:.1f} ms/img"
            )
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HaizkLow Crucible] Running on {device}")

    train_ds = CrossingMazeDataset(n_samples=320, size=256, base_seed=0)
    val256_ds = CrossingMazeDataset(n_samples=48, size=256, base_seed=10_000)
    val512_res = CrossingMazeDataset(n_samples=24, size=512, base_seed=20_000, physical_scale=None)
    val512_len = CrossingMazeDataset(n_samples=24, size=512, base_seed=30_000, physical_scale=1.0)

    train_loader = DataLoader(train_ds, batch_size=8, shuffle=True, num_workers=2, pin_memory=True)
    val256_loader = DataLoader(val256_ds, batch_size=8, shuffle=False, num_workers=2)
    val512_res_loader = DataLoader(val512_res, batch_size=4, shuffle=False, num_workers=2)
    val512_len_loader = DataLoader(val512_len, batch_size=4, shuffle=False, num_workers=2)

    half_unet = train_one(
        "Half-UNet (C=32)", HalfUNet(base_ch=32),
        train_loader, val256_loader, device, epochs=25, lr=2e-3, is_haizklow=False
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    haizk = train_one(
        "HaizkLow v2 (C=32, K=3)",
        HaizkLow(ch=32, groups=4, n_vcycles=3, smooth_steps=1, use_checkpoint=False),
        train_loader, val256_loader, device, epochs=25, lr=3e-3, is_haizklow=True
    )

    print("\n================ ZERO-SHOT EXTRAPOLATION SHOWDOWN (Trained ONLY on 256x256!) ================")
    for tag, loader in [("512x512 Resolution Jump", val512_res_loader),
                        ("512x512 2x Path-Length Torture", val512_len_loader)]:
        r_hu = benchmark_model(half_unet, loader, device, is_haizklow=False)
        r_hl3 = benchmark_model(haizk, loader, device, is_haizklow=True, n_vcycles_eval=3)
        r_hl6 = benchmark_model(haizk, loader, device, is_haizklow=True, n_vcycles_eval=6)
        print(f"\n--- {tag} ---")
        print(f"  Half-UNet     : Vessel={r_hu['vessel_dice']:.4f} | AV_Dice={r_hu['av_dice']:.4f} | AV_Sep={r_hu['av_sep']:.4f} | Cross_F1={r_hu['crossing_f1']:.4f} | Thin_F1={r_hu['thin_f1']:.4f} | Active=100.0% | {r_hu['ms_per_img']:.1f} ms")
        print(f"  HaizkLow (K=3): Vessel={r_hl3['vessel_dice']:.4f} | AV_Dice={r_hl3['av_dice']:.4f} | AV_Sep={r_hl3['av_sep']:.4f} | Cross_F1={r_hl3['crossing_f1']:.4f} | Thin_F1={r_hl3['thin_f1']:.4f} | Active={r_hl3['active_pct']:.1f}% | {r_hl3['ms_per_img']:.1f} ms")
        print(f"  HaizkLow (K=6): Vessel={r_hl6['vessel_dice']:.4f} | AV_Dice={r_hl6['av_dice']:.4f} | AV_Sep={r_hl6['av_sep']:.4f} | Cross_F1={r_hl6['crossing_f1']:.4f} | Thin_F1={r_hl6['thin_f1']:.4f} | Active={r_hl6['active_pct']:.1f}% | {r_hl6['ms_per_img']:.1f} ms")

    # Quick C-style bottleneck profiler on one 512x512 batch
    sample_img = next(iter(val512_res_loader))["image"].to(device)
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
        haizk.eval()
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        pyr, sub = haizk.pyramid(sample_img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        _ = haizk(sample_img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        print(f"\n[Profile 512x512 B={sample_img.shape[0]}] Pyramid={(t1-t0)*1e3/sample_img.shape[0]:.1f} ms/img | Total={(t2-t1)*1e3/sample_img.shape[0]:.1f} ms/img")


if __name__ == "__main__":
    main()
