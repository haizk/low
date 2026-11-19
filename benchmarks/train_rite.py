# benchmarks/train_rite.py
# HaizkLow Track A — Real Clinical Retinal Artery-Vein Benchmark (RITE / DRIVE-AV, av3 format)
#
# Apples-to-apples showdown: Official Half-UNet (53.3K) vs. HaizkLow v2 (~29.8K)
# Native 584x565 resolution (zero-padded to 640x640, root_bricks=10 -> ZERO vessel downsampling blur!)
#
# Run on Kaggle T4:
#   python benchmarks/train_rite.py

import glob
import os
import sys
import time
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crucible.baselines import HalfUNet
from haizklow.se2_multigrid import HaizkLow, batch_gt_split_masks
from benchmarks.eval_topology import compute_cldice


DEFAULT_ROOT = "/kaggle/input/datasets/hezkielbramsetiawan/rite-dataset/RITE"
PAD_SIZE = 640  # Native 584x565 padded cleanly -> 40 -> 80 -> 160 -> 320 -> 640 (root_bricks=10)


def find_rite_root(root=DEFAULT_ROOT):
    if os.path.isdir(os.path.join(root, "train", "av3")):
        return root
    hits = glob.glob("/kaggle/input/**/av3/21.png", recursive=True)
    if hits:
        return os.path.dirname(os.path.dirname(os.path.dirname(hits[0])))
    raise FileNotFoundError(f"Could not locate RITE dataset at {root} or under /kaggle/input")


class RITEAV3Dataset(Dataset):
    """
    Loads RITE av3 format:
      enhanced/       -> enhanced RGB fundus image (584x565)
      enhanced_masks/ -> binary circular FOV mask (584x565)
      av3/            -> RGB [R=Artery, G=Vein, B=AllVessels] (584x565)
    """
    def __init__(self, root: str, split: str = "train", augment: bool = False, repeat: int = 1):
        self.augment = augment
        self.repeat = repeat
        img_paths = sorted(glob.glob(os.path.join(root, split, "enhanced", "*.png")))
        assert len(img_paths) > 0, f"No images found in {root}/{split}/enhanced"

        self.samples = []
        for ip in img_paths:
            name = os.path.basename(ip)
            mp = os.path.join(root, split, "enhanced_masks", name)
            ap = os.path.join(root, split, "av3", name)

            img_rgb = cv2.cvtColor(cv2.imread(ip, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
            fov = (cv2.imread(mp, cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
            av3_rgb = cv2.cvtColor(cv2.imread(ap, cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            self.samples.append((name, img_rgb, fov, av3_rgb))

    def __len__(self):
        return len(self.samples) * self.repeat

    def _pad_to_640(self, arr):
        H, W = arr.shape[:2]
        return cv2.copyMakeBorder(arr, 0, PAD_SIZE - H, 0, PAD_SIZE - W, cv2.BORDER_CONSTANT, value=0)

    def _augment(self, img_rgb, fov, av3_rgb):
        # 1. Random flips & 90-deg rotations (preserves exact pixel grid!)
        if np.random.rand() < 0.5:
            img_rgb = np.ascontiguousarray(img_rgb[:, ::-1])
            fov = np.ascontiguousarray(fov[:, ::-1])
            av3_rgb = np.ascontiguousarray(av3_rgb[:, ::-1])
        if np.random.rand() < 0.5:
            img_rgb = np.ascontiguousarray(img_rgb[::-1, :])
            fov = np.ascontiguousarray(fov[::-1, :])
            av3_rgb = np.ascontiguousarray(av3_rgb[::-1, :])
        k = np.random.randint(0, 4)
        if k > 0:
            img_rgb = np.ascontiguousarray(np.rot90(img_rgb, k))
            fov = np.ascontiguousarray(np.rot90(fov, k))
            av3_rgb = np.ascontiguousarray(np.rot90(av3_rgb, k))

        # 2. Continuous rotation (+/- 25 deg) + mild scale (0.92..1.08)
        H, W = img_rgb.shape[:2]
        ang = np.random.uniform(-25.0, 25.0)
        sc = np.random.uniform(0.92, 1.08)
        M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), ang, sc)
        img_rgb = cv2.warpAffine(img_rgb, M, (W, H), flags=cv2.INTER_LINEAR, borderValue=0)
        fov = cv2.warpAffine(fov, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)
        av3_rgb = cv2.warpAffine(av3_rgb, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=0)

        # 3. Mild photometric jitter (keeps subtle arterial orange vs venous purple intact!)
        img_f = img_rgb.astype(np.float32) / 255.0
        gain = np.random.uniform(0.90, 1.10, size=(1, 1, 3)).astype(np.float32)
        bias = np.random.uniform(-0.04, 0.04, size=(1, 1, 3)).astype(np.float32)
        gamma = np.random.uniform(0.90, 1.12)
        img_f = np.clip(img_f, 0.0, 1.0) ** gamma
        img_f = np.clip(img_f * gain + bias, 0.0, 1.0)
        img_f = img_f * fov[..., None]
        return img_f, fov, av3_rgb

    def __getitem__(self, idx):
        name, img_rgb, fov, av3_rgb = self.samples[idx % len(self.samples)]
        orig_h, orig_w = img_rgb.shape[:2]

        img_rgb = self._pad_to_640(img_rgb)
        fov = self._pad_to_640(fov)
        av3_rgb = self._pad_to_640(av3_rgb)

        if self.augment:
            img_f, fov, av3_rgb = self._augment(img_rgb, fov, av3_rgb)
        else:
            img_f = (img_rgb.astype(np.float32) / 255.0) * fov[..., None]

        # Exact verified av3 parsing: R=Artery, G=Vein, B=AllVessels
        R = (av3_rgb[..., 0] > 127)
        G = (av3_rgb[..., 1] > 127)
        B = (av3_rgb[..., 2] > 127)

        fov_b = (fov > 0)
        uncertain = B & (~R) & (~G)
        valid_av = fov_b & (~uncertain)
        crossing = (R & G).astype(np.uint8)
        cross_zone = (cv2.dilate(crossing, np.ones((7, 7), np.uint8)) > 0) & fov_b

        target = np.stack([R, G, B], axis=0).astype(np.float32)  # [3, 640, 640]
        return {
            "image": torch.from_numpy(img_f.transpose(2, 0, 1).copy()),  # [3, 640, 640]
            "target": torch.from_numpy(target),                          # [3, 640, 640]
            "fov": torch.from_numpy(fov_b),                              # [640, 640] bool
            "valid_av": torch.from_numpy(valid_av),                      # [640, 640] bool
            "cross_zone": torch.from_numpy(cross_zone),                  # [640, 640] bool
            "orig_h": orig_h,
            "orig_w": orig_w,
        }


def masked_bce_dice(logits, target, mask):
    """Exact FOV / valid-mask BCE + SoftDice per channel."""
    m = mask.float()
    bce_map = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    bce = (bce_map * m).sum() / (m.sum() + 1e-6)

    prob = torch.sigmoid(logits) * m
    tgt = target * m
    inter = (prob * tgt).sum(dim=(0, 2, 3))
    denom = prob.sum(dim=(0, 2, 3)) + tgt.sum(dim=(0, 2, 3)) + 1e-6
    dice = 1.0 - (2.0 * inter + 1e-6) / denom
    return bce + dice.mean()


def rite_av3_loss(logits, target, fov, valid_av):
    # Channels 0,1 (Artery, Vein): supervised inside valid_av (FOV minus uncertain vessels)
    loss_av = masked_bce_dice(logits[:, :2], target[:, :2], valid_av.unsqueeze(1))
    # Channel 2 (AllVessels): supervised inside full FOV (includes uncertain capillaries!)
    loss_v = masked_bce_dice(logits[:, 2:3], target[:, 2:3], fov.unsqueeze(1))
    return loss_av + 0.5 * loss_v


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate_rite(model, loader, device, is_haizklow=False, n_vcycles_eval=None, compute_topo=False):
    model.eval()
    orig_k = getattr(model, "n_vcycles", None)
    if is_haizklow and n_vcycles_eval is not None:
        model.n_vcycles = n_vcycles_eval

    # Accumulate pixel-level TP/FP/FN cleanly across the exact 20 untouched 584x565 test eyes
    inter_av = torch.zeros(2, device=device)
    denom_av = torch.zeros(2, device=device)
    inter_v, denom_v = 0.0, 0.0
    sep_corr, sep_tot = 0.0, 0.0
    cz_inter, cz_denom = 0.0, 0.0
    cl_scores = []
    act_sum, n_batches, t_ms = 0.0, 0, 0.0

    for batch in loader:
        img = batch["image"].to(device)
        tgt = batch["target"].to(device)
        fov = batch["fov"].to(device)
        vav = batch["valid_av"].to(device)
        cz = batch["cross_zone"].to(device)
        H0, W0 = int(batch["orig_h"][0]), int(batch["orig_w"][0])

        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
            out = model(img)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t_ms += (time.perf_counter() - t0) * 1000.0 / img.shape[0]

        # Crop strictly back to exact untouched native 584x565 test pixels!
        lg = out["logits"][:, :, :H0, :W0].float()
        tg = tgt[:, :, :H0, :W0]
        fv = fov[:, :H0, :W0]
        va = vav[:, :H0, :W0]
        cz_c = cz[:, :H0, :W0]

        pred = (torch.sigmoid(lg) > 0.5).float()

        # 1. Artery & Vein Dice (inside valid_av)
        vm = va.unsqueeze(1)
        p_av, t_av = pred[:, :2] * vm, tg[:, :2] * vm
        inter_av += (p_av * t_av).sum(dim=(0, 2, 3))
        denom_av += p_av.sum(dim=(0, 2, 3)) + t_av.sum(dim=(0, 2, 3))

        # 2. All-Vessel Dice (inside full FOV)
        fm = fv.unsqueeze(1)
        p_v, t_v = pred[:, 2:3] * fm, tg[:, 2:3] * fm
        inter_v += (p_v * t_v).sum().item()
        denom_v += (p_v.sum() + t_v.sum()).item()

        # 3. Pure A/V Separation Accuracy on known non-crossing vessel pixels
        a_only = (tg[:, 0] > 0.5) & (tg[:, 1] < 0.5) & va
        v_only = (tg[:, 1] > 0.5) & (tg[:, 0] < 0.5) & va
        sep_corr += (lg[:, 0][a_only] > lg[:, 1][a_only]).float().sum().item()
        sep_corr += (lg[:, 1][v_only] > lg[:, 0][v_only]).float().sum().item()
        sep_tot += (a_only.sum() + v_only.sum()).item()

        # 4. Crossing-Zone F1
        cz2 = cz_c.unsqueeze(1).expand_as(p_av)
        cz_inter += (pred[:, :2][cz2] * tg[:, :2][cz2]).sum().item()
        cz_denom += (pred[:, :2][cz2].sum() + tg[:, :2][cz2].sum()).item()

        # 5. Topological clDice (on final evals so epochs stay fast!)
        if compute_topo:
            p_np = p_av.cpu().numpy()
            t_np = t_av.cpu().numpy()
            for b in range(p_np.shape[0]):
                for c in range(2):
                    cl_scores.append(compute_cldice(p_np[b, c], t_np[b, c]))

        act_sum += out.get("active_ratio", 1.0) * 100.0
        n_batches += 1

    if is_haizklow and orig_k is not None:
        model.n_vcycles = orig_k

    eps = 1e-6
    av_dices = (2.0 * inter_av + eps) / (denom_av + eps)
    return {
        "a_dice": av_dices[0].item(),
        "v_dice": av_dices[1].item(),
        "av_dice": av_dices.mean().item(),
        "vessel_dice": (2.0 * inter_v + eps) / (denom_v + eps),
        "av_sep": sep_corr / (sep_tot + eps),
        "cross_f1": (2.0 * cz_inter + eps) / (cz_denom + eps),
        "cl_dice": float(np.mean(cl_scores)) if cl_scores else 0.0,
        "active_pct": act_sum / n_batches,
        "ms_per_img": t_ms / n_batches,
    }


def train_rite_model(name, model, train_loader, test_loader, device,
                     epochs=120, lr=2e-3, is_haizklow=False):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda"))

    print(f"\n================ RITE Training: {name} ({count_params(model):,} params) ================")
    best_av = 0.0

    for ep in range(1, epochs + 1):
        model.train()
        loss_sum = 0.0
        # Scheduled Teacher-to-Pred anneal over first 30% of epochs
        teacher_prob = max(0.0, 1.0 - (ep - 1) / max(1, int(0.30 * epochs)))

        for batch in train_loader:
            img = batch["image"].to(device)
            tgt = batch["target"].to(device)
            fov = batch["fov"].to(device)
            vav = batch["valid_av"].to(device)
            opt.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", dtype=torch.float16, enabled=(device.type == "cuda")):
                if is_haizklow:
                    # Supervise Quadtree splitter on Channel 2 (AllVessels, including uncertain!)
                    gt_splits = batch_gt_split_masks(tgt[:, 2:3], model.tile_size, model.root_bricks)
                    out = model(img, gt_split_masks=gt_splits, union_prob=teacher_prob)

                    n_c = len(out["cycle_logits"])
                    weights = [(i + 1) / n_c for i in range(n_c)]
                    w_sum = sum(weights)
                    seg_loss = sum(
                        w * rite_av3_loss(lg, tgt, fov, vav)
                        for w, lg in zip(weights, out["cycle_logits"])
                    ) / w_sum

                    split_loss = sum(
                        F.binary_cross_entropy_with_logits(sl.float(), st.float())
                        for sl, st in zip(out["split_logits"], out["split_targets"])
                    ) / len(out["split_logits"])

                    loss = seg_loss + 1.0 * split_loss
                else:
                    out = model(img)
                    loss = rite_av3_loss(out["logits"], tgt, fov, vav)

            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            loss_sum += loss.item()

        sched.step()
        if ep == 1 or ep % 10 == 0 or ep == epochs:
            res = evaluate_rite(model, test_loader, device, is_haizklow=is_haizklow, compute_topo=(ep == epochs))
            best_av = max(best_av, res["av_dice"])
            topo_str = f" | clDice={res['cl_dice']:.4f}" if ep == epochs else ""
            print(
                f"Ep {ep:03d}/{epochs} | loss={loss_sum/len(train_loader):.4f} | "
                f"Vessel={res['vessel_dice']:.4f} | AV_Dice={res['av_dice']:.4f} "
                f"(A={res['a_dice']:.3f}/V={res['v_dice']:.3f}) | AV_Sep={res['av_sep']:.4f} | "
                f"Cross_F1={res['cross_f1']:.4f}{topo_str} | Active={res['active_pct']:.1f}% | "
                f"{res['ms_per_img']:.1f} ms"
            )
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = find_rite_root()
    print(f"[HaizkLow Track A — RITE Clinical Eyes] Device={device} | Root={root}")

    # repeat=1 -> 20 augmented views per epoch (5 batches of 4), 80 epochs = 400 steps!
    train_ds = RITEAV3Dataset(root, split="train", augment=True, repeat=1)
    test_ds = RITEAV3Dataset(root, split="test", augment=False, repeat=1)

    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    half_unet = train_rite_model(
        "Half-UNet Official (C=32)",
        HalfUNet(in_ch=3, out_ch=3, base_ch=32),
        train_loader, test_loader, device, epochs=80, lr=2e-3, is_haizklow=False
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()

    haizk = train_rite_model(
        "HaizkLow v2 (C=32, K=3)",
        HaizkLow(ch=32, groups=4, out_ch=3, n_vcycles=3, smooth_steps=1,
                 tile_size=4, root_bricks=10, use_checkpoint=False),
        train_loader, test_loader, device, epochs=80, lr=2.5e-3, is_haizklow=True
    )

    # Test-Time Compute Scaling Check on Real Unseen Human Eyes (K=3 vs K=6)!
    print("\n================ FINAL RITE CLINICAL TEST SUMMARY (20 Unseen Eyes, Native 584x565) ================")
    for tag, m, is_hl, k_ev in [
        ("Half-UNet (53.3K)     ", half_unet, False, None),
        ("HaizkLow v2 (K=3, 30K)", haizk,     True,  3),
        ("HaizkLow v2 (K=6, 30K)", haizk,     True,  6),
    ]:
        r = evaluate_rite(m, test_loader, device, is_haizklow=is_hl, n_vcycles_eval=k_ev, compute_topo=True)
        print(
            f"  {tag} | Vessel={r['vessel_dice']:.4f} | AV_Dice={r['av_dice']:.4f} "
            f"(A={r['a_dice']:.4f}/V={r['v_dice']:.4f}) | AV_Sep={r['av_sep']:.4f} | "
            f"Cross_F1={r['cross_f1']:.4f} | clDice={r['cl_dice']:.4f} | "
            f"Active={r['active_pct']:.1f}% | {r['ms_per_img']:.1f} ms"
        )


if __name__ == "__main__":
    main()
