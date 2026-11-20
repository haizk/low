# benchmarks/bench_latency.py
# HaizkLow v3 — CUDA Gradcheck (All 3 Kernels) + Honest Sparse & Dense Scaling (256^2 -> 2048^2)
#
# Run on Kaggle T4 (or CPU):
#   python benchmarks/bench_latency.py

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crucible.baselines import HalfUNet
from crucible.crossing_maze import generate_crossing_maze
from haizklow.se2_multigrid import HaizkLow
from haizklow.quadtree import (
    batch_gt_split_masks,
    build_bmap_and_neighbors,
    fused_se2_laplacian_ref,
    prolong_bricks_ref,
    restrict_bricks_ref,
)
from haizklow.ops import fused_se2_laplacian, prolong_bricks_nhwc, restrict_bricks_nhwc


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_cuda_verification(device: torch.device):
    if device.type != "cuda":
        return
    print("\n=== 1. HaizkLow v3 CUDA Kernel Verification (fp64 gradcheck + exactness) ===")
    B, G, D = 2, 4, 4
    mask = torch.rand(B, G, G, device=device) < 0.55
    b_idx, by_idx, bx_idx = torch.where(mask)
    N = b_idx.numel()
    _, nb_idx = build_bmap_and_neighbors(b_idx, by_idx, bx_idx, B, G)

    u64 = torch.randn(N, 4, 4, 4, D, device=device, dtype=torch.float64, requires_grad=True)
    a64 = torch.sigmoid(torch.randn(N, 4, 4, 4, device=device, dtype=torch.float64)).requires_grad_(True)
    c64 = torch.sigmoid(torch.randn(N, 4, 4, 4, device=device, dtype=torch.float64)).requires_grad_(True)
    dts64 = torch.full((D,), 0.15, device=device, dtype=torch.float64, requires_grad=True)
    dtc64 = torch.full((D,), 0.10, device=device, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(fused_se2_laplacian, (u64, a64, c64, dts64, dtc64, nb_idx), eps=1e-6, atol=1e-4)
    with torch.no_grad():
        diff = (fused_se2_laplacian(u64, a64, c64, dts64, dtc64, nb_idx)
                - fused_se2_laplacian_ref(u64, a64, c64, dts64, dtc64, nb_idx)).abs().max().item()
    print(f"  [PASS] fused_se2_laplacian  fp64 gradcheck | max_diff={diff:.2e}")

    p64 = torch.randn(5, 4, 4, 8, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(prolong_bricks_nhwc, (p64,), eps=1e-6, atol=1e-4)
    with torch.no_grad():
        diff_p = (prolong_bricks_nhwc(p64) - prolong_bricks_ref(p64)).abs().max().item()
    print(f"  [PASS] prolong_bricks_nhwc  fp64 gradcheck | max_diff={diff_p:.2e}")

    ch64 = torch.randn(20, 4, 4, 8, device=device, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(restrict_bricks_nhwc, (ch64,), eps=1e-6, atol=1e-4)
    with torch.no_grad():
        diff_r = (restrict_bricks_nhwc(ch64) - restrict_bricks_ref(ch64)).abs().max().item()
    print(f"  [PASS] restrict_bricks_nhwc fp64 gradcheck | max_diff={diff_r:.2e}")


@torch.no_grad()
def bench_call(fn, device: torch.device, warmup: int = 5, iters: int = 20):
    use_amp = (device.type == "cuda")
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = fn()
    if use_amp:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = fn()
    if use_amp:
        torch.cuda.synchronize()

    ms = (time.perf_counter() - t0) * 1000.0 / iters
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if use_amp else 0.0
    act_pct = out.get("active_ratio", 1.0) * 100.0 if isinstance(out, dict) else 100.0
    return ms, peak_mb, act_pct


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HaizkLow v3 Latency & Memory Benchmark] Device={device}")
    run_cuda_verification(device)

    hu = HalfUNet(in_ch=3, out_ch=3, base_ch=32).to(device).eval()
    hl_demo = HaizkLow(ch=64, out_ch=3, n_vcycles=2, tile_size=4, root_bricks=8).to(device).eval()
    print(f"\n=== 2. End-to-End Inference Scaling (fp16={device.type=='cuda'}, B=1, K=2 V-cycles) ===")
    print(f"Half-UNet (C=32)        : {count_params(hu):,} params")
    print(f"HaizkLow v3 (K=4, D=16) : {count_params(hl_demo):,} params\n")

    print(f"{'Res':>9s} | {'Half-UNet':>17s} | {'v3 Sparse (GT tree)':>28s} | {'v3 Dense (100%)':>18s} | {'Pyr':>6s}")
    print("-" * 90)

    for res, rb in [(256, 8), (512, 8), (640, 10), (1024, 8), (2048, 8)]:
        hl = HaizkLow(ch=64, out_ch=3, n_vcycles=2, tile_size=4, root_bricks=rb).to(device).eval()

        img_np, tgt_np, _, _, _ = generate_crossing_maze(res, res, seed=7)
        x = torch.from_numpy(img_np).unsqueeze(0).to(device)
        tgt = torch.from_numpy(tgt_np).unsqueeze(0).to(device)
        gt_masks = batch_gt_split_masks(tgt, tile_size=4, root_bricks=rb)

        hu_ms, hu_mb, _ = bench_call(lambda: hu(x), device)
        sp_ms, sp_mb, sp_act = bench_call(lambda: hl(x, force_split_masks=gt_masks), device)
        pyr_ms, _, _ = bench_call(lambda: hl.pyramid(x), device, warmup=3, iters=10)

        if res <= 1024:
            dn_ms, dn_mb, _ = bench_call(lambda: hl(x, split_thresh=-10.0), device, warmup=3, iters=10)
            dn_str = f"{dn_ms:6.2f}ms {dn_mb:6.1f}MB"
        else:
            dn_str = "   (skipped @2K) "

        speedup = hu_ms / max(sp_ms, 1e-6)
        print(
            f"{res:4d}x{res:<4d} | {hu_ms:6.2f}ms {hu_mb:6.1f}MB | "
            f"{sp_ms:6.2f}ms {sp_mb:5.1f}MB ({sp_act:4.1f}%) {speedup:5.2f}x | "
            f"{dn_str:>18s} | {pyr_ms:5.2f}ms"
        )


if __name__ == "__main__":
    main()
