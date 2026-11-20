# benchmarks/eval_topology.py
# HaizkLow — Q1 Clinical Retinal A/V Topology Metrics (clDice, Betti-0/1, Crossing F1, AV_Sep)

import cv2
import numpy as np
import torch


def fast_skeletonize(binary_mask: np.ndarray) -> np.ndarray:
    """Morphological skeletonization (uses skimage if installed, pure OpenCV fallback)."""
    try:
        from skimage.morphology import skeletonize
        return skeletonize(binary_mask > 0).astype(np.uint8)
    except ImportError:
        img = (binary_mask > 0).astype(np.uint8) * 255
        skel = np.zeros_like(img)
        elem = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
        while True:
            eroded = cv2.erode(img, elem)
            opened = cv2.dilate(eroded, elem)
            skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
            img = eroded
            if cv2.countNonZero(img) == 0:
                break
        return (skel > 0).astype(np.uint8)


def compute_cldice(pred_bin: np.ndarray, gt_bin: np.ndarray) -> float:
    """Exact topological Centerline Dice (Shit et al., CVPR 2021)."""
    pred_b = (pred_bin > 0).astype(np.uint8)
    gt_b = (gt_bin > 0).astype(np.uint8)
    if pred_b.sum() == 0 or gt_b.sum() == 0:
        return 0.0
    s_pred = fast_skeletonize(pred_b)
    s_gt = fast_skeletonize(gt_b)
    tprec = (s_pred & gt_b).sum() / (s_pred.sum() + 1e-6)
    tsens = (s_gt & pred_b).sum() / (s_gt.sum() + 1e-6)
    return float(2.0 * tprec * tsens / (tprec + tsens + 1e-6))


def compute_betti_numbers(binary_mask: np.ndarray) -> tuple[int, int]:
    """2D digital topology Betti numbers (8-connected FG, 4-connected BG): (beta_0, beta_1)."""
    fg = (binary_mask > 0).astype(np.uint8)
    if fg.sum() == 0:
        return 0, 0
    n_fg, _ = cv2.connectedComponents(fg, connectivity=8)
    beta_0 = int(n_fg - 1)
    padded_bg = np.pad(1 - fg, pad_width=1, mode="constant", constant_values=1)
    n_bg, _ = cv2.connectedComponents(padded_bg, connectivity=4)
    beta_1 = int(max(0, n_bg - 2))
    return beta_0, beta_1


def compute_betti_errors(pred_bin: np.ndarray, gt_bin: np.ndarray) -> tuple[int, int]:
    pb0, pb1 = compute_betti_numbers(pred_bin)
    gb0, gb1 = compute_betti_numbers(gt_bin)
    return abs(pb0 - gb0), abs(pb1 - gb1)


def parse_rite_rgb_mask(mask_rgb: np.ndarray):
    R = mask_rgb[..., 0] > 127
    G = mask_rgb[..., 1] > 127
    B = mask_rgb[..., 2] > 127

    white = R & G & B
    cross = G & (~R) & (~B)
    artery = (R & (~B)) | cross
    vein = (B & (~R)) | cross
    valid_av = ~white
    split_fg = (artery | vein | white).astype(np.float32)

    cross_zone = cv2.dilate(cross.astype(np.uint8), np.ones((7, 7), np.uint8)) > 0
    target_2hw = np.stack([artery.astype(np.float32), vein.astype(np.float32)], axis=0)
    return target_2hw, valid_av, split_fg, cross_zone


@torch.no_grad()
def evaluate_clinical_av(logits: torch.Tensor, target: torch.Tensor,
                         valid_av: torch.Tensor, cross_zone: torch.Tensor, thresh: float = 0.5):
    prob = torch.sigmoid(logits)
    pred = (prob > thresh).float()
    vm = valid_av.unsqueeze(1).expand_as(pred)
    eps = 1e-6

    p_v, t_v = pred * vm, target * vm
    inter = (p_v * t_v).sum(dim=(0, 2, 3))
    denom = p_v.sum(dim=(0, 2, 3)) + t_v.sum(dim=(0, 2, 3)) + eps
    av_dice = ((2.0 * inter + eps) / denom).mean().item()

    a_only = (target[:, 0] > 0.5) & (target[:, 1] < 0.5) & valid_av
    v_only = (target[:, 1] > 0.5) & (target[:, 0] < 0.5) & valid_av
    corr = (logits[:, 0][a_only] > logits[:, 1][a_only]).float().sum() + \
           (logits[:, 1][v_only] > logits[:, 0][v_only]).float().sum()
    av_sep = (corr / (a_only.sum() + v_only.sum() + eps)).item()

    cz = cross_zone.unsqueeze(1).expand_as(pred)
    if cz.any():
        cz_inter = (pred[cz] * target[cz]).sum()
        cz_denom = pred[cz].sum() + target[cz].sum() + eps
        cross_f1 = ((2.0 * cz_inter + eps) / cz_denom).item()
    else:
        cross_f1 = 0.0

    pred_np = pred.cpu().numpy()
    tgt_np = target.cpu().numpy()
    vm_np = valid_av.cpu().numpy()
    cl_scores, b0_errs, b1_errs = [], [], []
    for b in range(pred_np.shape[0]):
        for c in range(2):
            p_m = pred_np[b, c] * vm_np[b]
            t_m = tgt_np[b, c] * vm_np[b]
            cl_scores.append(compute_cldice(p_m, t_m))
            eb0, eb1 = compute_betti_errors(p_m, t_m)
            b0_errs.append(eb0)
            b1_errs.append(eb1)

    return {
        "av_dice": av_dice,
        "av_sep": av_sep,
        "cross_f1": cross_f1,
        "cl_dice": float(np.mean(cl_scores)),
        "betti0_err": float(np.mean(b0_errs)),
        "betti1_err": float(np.mean(b1_errs)),
    }
