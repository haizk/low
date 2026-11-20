# benchmarks/bench_latency.py
# HaizkLow — Honest Wall-Clock Latency, Peak VRAM, CUDA Gradcheck & Stage Profiler
#
# Run on Kaggle T4 (or CPU):
#   python benchmarks/bench_latency.py

import os
import sys
import time
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from crucible.baselines import HalfUNet
from haizklow.se2_multigrid import HaizkLow
from haizklow.quadtree import build_bmap_and_neighbors, gather_brick_halo_ref
from haizklow.ops import fused_halo_dw3x3


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def run_cuda_verification(device: torch.device):
    if device.type != "cuda":
        return
    print("\n=== 1. CUDA Kernel Verification (fp64 gradcheck + exactness) ===")
    B, G, C = 2, 5, 6
    mask = torch.rand(B, G, G, device=device) < 0.45
    b_idx, by_idx, bx_idx = torch.where(mask)
    N = b_idx.numel()
    _, nb_idx = build_bmap_and_neighbors(b_idx, by_idx, bx_idx, B, G)

    u64 = torch.randn(N, C, 4, 4, device=device, dtype=torch.float64, requires_grad=True)
    w64 = torch.randn(C, 1, 3, 3, device=device, dtype=torch.float64, requires_grad=True)
    b64 = torch.randn(C, device=device, dtype=torch.float64, requires_grad=True)

    assert torch.autograd.gradcheck(fused_halo_dw3x3, (u64, w64, b64, nb_idx), eps=1e-6, atol=1e-4)
    ref = torch.nn.functional.conv2d(gather_brick_halo_ref(u64, nb_idx), w64, b64, groups=C)
    out = fused_halo_dw3x3(u64, w64, b64, nb_idx)
    print(f"  [PASS] fused_halo_dw3x3 fp64 gradcheck | max_diff={float((out - ref).abs().max()):.2e}")


@torch.no_grad()
def bench_model(model: torch.nn.Module, x: torch.Tensor, device: torch.device, warmup: int = 5, iters: int = 20):
    model.eval()
    use_amp = (device.type == "cuda")
    if use_amp:
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.empty_cache()

    for _ in range(warmup):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
    if use_amp:
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            out = model(x)
    if use_amp:
        torch.cuda.synchronize()

    ms_per_img = (time.perf_counter() - t0) * 1000.0 / (iters * x.shape[0])
    peak_mb = torch.cuda.max_memory_allocated(device) / (1024 ** 2) if use_amp else 0.0
    act_pct = out.get("active_ratio", 1.0) * 100.0
    return ms_per_img, peak_mb, act_pct


@torch.no_grad()
def profile_pyramid(model: HaizkLow, x: torch.Tensor, device: torch.device, iters: int = 15) -> float:
    model.eval()
    use_amp = (device.type == "cuda")
    with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
        _ = model.pyramid(x)
    if use_amp:
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        with torch.amp.autocast("cuda", dtype=torch.float16, enabled=use_amp):
            _ = model.pyramid(x)
    if use_amp:
        torch.cuda.synchronize()
    return (time.perf_counter() - t0) * 1000.0 / (iters * x.shape[0])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[HaizkLow Latency & Memory Benchmark] Device={device}")
    run_cuda_verification(device)

    hu = HalfUNet(in_ch=3, out_ch=3, base_ch=32).to(device)
    print(f"\n=== 2. End-to-End Inference Scaling (fp16={device.type=='cuda'}, B=1) ===")
    print(f"Half-UNet (C=32) params: {count_params(hu):,}")

    configs = [(256, 8), (512, 8), (640, 10), (1024, 8)]
    for res, rb in configs:
        hl = HaizkLow(ch=32, out_ch=3, n_vcycles=3, smooth_steps=1, tile_size=4, root_bricks=rb).to(device)
        if res == 256:
            print(f"HaizkLow params        : {count_params(hl):,}\n")
            print(f"{'Res':>9s} | {'Half-UNet ms':>12s} {'VRAM':>8s} | {'HaizkLow ms':>11s} {'VRAM':>8s} {'Act%':>6s} | {'Pyr ms':>7s} {'Speedup':>7s}")
            print("-" * 82)

        x = torch.rand(1, 3, res, res, device=device)
        hu_ms, hu_mb, _ = bench_model(hu, x, device)
        hl_ms, hl_mb, act = bench_model(hl, x, device)
        pyr_ms = profile_pyramid(hl, x, device)
        print(
            f"{res:4d}x{res:<4d} | {hu_ms:10.2f}ms {hu_mb:6.1f}MB | "
            f"{hl_ms:9.2f}ms {hl_mb:6.1f}MB {act:5.1f}% | {pyr_ms:5.2f}ms {hu_ms/max(hl_ms,1e-6):6.2f}x"
        )


if __name__ == "__main__":
    main()
