# HaizkLow

> **A standalone, hardware-native Adaptive Quadtree Multigrid + $SE(2)$ Fiber PDE engine for sparse spatial perception.**  
> *No U-Net. No dense pixel-grid waste. No fake-FLOPs benchmarks.*

---

## 1. Thesis: Why Current Vision Architectures Are Broken

Modern segmentation models (U-Net, Half-UNet, Swin, Vision Mamba) are trapped by three broken assumptions:

1. **The 90% Waste Problem:** In curvilinear and boundary-heavy domains (artery–vein vasculature, neurons, satellite roads, micro-cracks, object contours), 85–95% of the image is empty background. Dense grids burn memory bandwidth and compute equally everywhere.
2. **The FLOPs Lie:** "Lightweight" CNNs rely on depthwise-separable convolutions, small channel-attention ops, and full-scale feature additions that look tiny in parameter/FLOP tables while stalling on HBM/RAM memory bandwidth in real wall-clock execution.
3. **The Crossing Paradox:** Locally, thin arteries and veins are visually indistinguishable. Standard $3 \times 3$ convolutions mix perpendicular vessels at arteriovenous (A/V) crossings—causing topological breaks and impossible A/V identity switches.

**HaizkLow** deletes the deep feedforward encoder–decoder stack and replaces it with **one hardware-native computational primitive**: a weight-tied **Anisotropic Multigrid V-Cycle** operating over a **Pointerless Morton Linear Quadtree of `4×4` micro-bricks** with **$SE(2)$ orientation-lifted state fibers**.

---

## 2. Architecture Overview

```text
Raw Image (H x W)
   │
   ▼
[O(1) Multi-Stat Image Pyramid]  ──► mean, min, max, std, max-ridge energy (prevents 1px blindness)
   │
   ▼
[Coarse-to-Fine Adaptive Quadtree Growth (FMG)]
   Level 0 (8x8 bricks = 32x32 px)     ──► 100% dense root (64 bricks)
   Level 1 (16x16 max bricks)          ──► split active bricks via CSR prefix-sum
   Level 2 (32x32 max bricks)          ──► split active bricks
   Level 3 (64x64 max bricks)          ──► split active bricks
   Level 4 (128x128 max bricks = 512²) ──► ~10–15% active (~2,450 bricks ≈ 2.4 MB in L2 cache!)
   │
   ▼
[Weight-Tied Multigrid V-Cycle x K]  (Shared ~50K–150K params total)
   Fine (L4) ──► SE(2) Smoother ──► Restrict (code >> 2) ──► ... ──► Coarse Solve (L0)
                                                                           │
   Fine (L4) ◄── SE(2) Post-Smooth ◄── Prolongate ((code << 2)|q) ◄────────┘
   │
   ▼
[Adaptive Test-Time Exit]  ──► stop when ||u^(k+1) - u^(k)|| < eps
   │
   ▼
[Fiber Collapse Head] ──► Artery / Vein / Crossing / Background
```

---

## 3. Core Systems & Mathematical Specification

### 3.1 Pointerless Linear Quadtree with `4×4` Micro-Bricks (VDB-Style)
Pointer-based trees (`Node *nw, *ne, *sw, *se`) cause warp divergence and cache misses. Instead, every active node at level $\ell$ is a dense **`4×4` pixel micro-brick** (`TILE = 4`, compile-time configurable to `8`) indexed by a **Morton Z-Order key**:

```c
static inline uint32_t part1by1(uint32_t n) {
    n &= 0x0000ffffu;
    n = (n | (n << 8)) & 0x00ff00ffu;
    n = (n | (n << 4)) & 0x0f0f0f0fu;
    n = (n | (n << 2)) & 0x33333333u;
    n = (n | (n << 1)) & 0x55555555u;
    return n;
}

static inline uint32_t morton2d_encode(uint16_t bx, uint16_t by) {
    return part1by1(bx) | (part1by1(by) << 1);
}

// O(1) hierarchy navigation (1 clock cycle):
// parent_code = code >> 2;
// child_code  = (code << 2) | (dy << 1) | dx;
```

**Why `4×4` micro-bricks beat `1×1` sparse pixels on real silicon:**
* **Interior pixels:** Zero hash/grid lookups—100% coalesced shared-memory / L1 cache access.
* **Halo borders only:** Morton neighbor lookups happen only on brick boundaries (cutting lookup overhead by 4×–8×).
* **Uniform hierarchy:** Every level $\ell \in \{0 \dots L\}$ uses the exact same brick tensor layout and CUDA/C kernel.
* **Memory footprint (`TILE=4`, `K=4` fibers, `D=8` channels, `fp16`):**  
  $4 \times 4 \times 4 \times 8 \times 2\text{ B} = \mathbf{1\text{ KB/brick}}$. At 15% sparsity on $512^2$ (~2,450 bricks), the full finest-level state is **~2.4 MB**, fitting entirely inside a Tesla T4's 4 MB L2 cache.

| Level | Max Brick Grid | Effective Resolution | Typical Active Sparsity |
| :--- | :---: | :---: | :---: |
| L0 (Root) | $8 \times 8$ | $32 \times 32$ | 100% (64 bricks) |
| L1 | $16 \times 16$ | $64 \times 64$ | ~60% |
| L2 | $32 \times 32$ | $128 \times 128$ | ~40% |
| L3 | $64 \times 64$ | $256 \times 256$ | ~25% |
| L4 | $128 \times 128$ | $512 \times 512$ | ~10–15% |
| L5 *(zero-shot)* | $256 \times 256$ | $1024 \times 1024$ | ~8–10% |

---

### 3.2 Dynamic CSR Packing & Zero-Malloc GPU Arena
To support true variable per-image sparsity across a batch without `cudaMalloc` stalls:
* Pre-allocate a single reusable **GPU/CPU Workspace Arena** sized to `max_bricks`.
* Pack ragged active brick arrays across batch items `b = 0..B-1` using **CSR prefix-sum offsets** (`batch_offsets[B+1]`, computed via CUB/Thrust exclusive scan on GPU or linear scan on CPU).
* Maintain a compact flat `brick_map[B][grid_H * grid_W]` (`int32_t`, returning compact index `0..N-1` or `-1` if unrefined, triggering O(1) parent fallback on 2:1 balanced boundaries).

---

### 3.3 Coarse-to-Fine Tree Growth & Anti-Blindness Protection
To prevent thin 1-pixel capillaries from disappearing inside coarse $16 \times 16$ cells:
* **Phase 1 — Multi-Stat Mipmap Pyramid:** Precompute $O(1)$ per-level image summaries containing `[mean_RGB, min_intensity, max_intensity, std, max_ridge_energy]` (Laplacian / Hessian vesselness max-pool) so even a single faint capillary spikes coarse split energy.
* **Phase 2 (v2) — Lateral Wake-Up Refinement:** If an active fine vessel reaches the border of an unsplit neighbor brick during V-cycles, it dynamically wakes up and subdivides that neighbor.
* **Split Supervision:** Ground-truth split target $s_b^* = 1$ if brick $b$ contains any foreground/boundary variance, else $0$ (clean BCE supervision—no unstable RL/Gumbel estimators).

---

### 3.4 $SE(2)$ Orientation-Lifted Anisotropic PDE Smoother
Inspired by **V1 orientation columns in visual cortex** and PDEs on $SE(2) = \mathbb{R}^2 \times S^1$, each pixel holds $K = 4$ discrete orientation fibers ($\theta_k \in \{0^\circ, 45^\circ, 90^\circ, 135^\circ\}$, naturally packing into a `float4` SIMD register) with $D$ latent channels per fiber:

$$u_i = \{u_{i,k} \in \mathbb{R}^D\}_{k=0}^{3}$$

One relaxation step updates each orientation fiber $k$ via **anisotropic intra-fiber transport** (strong along $\theta_k$), **smooth angular turning** ($k \pm 1$, i.e., $\pm 45^\circ$ for curves & bifurcations), and **image forcing** $I_i^{(\ell)}$:

$$u_{i,k}^{(t+1)} = u_{i,k}^{(t)} + \Delta t \cdot \Phi_\theta\!\Big(u_{i,k}^{(t)},\; \underbrace{\sum_{j \in \mathcal{N}_k(i)} g_\theta(u_{i,k}^{(t)}, u_{j,k}^{(t)}, \vec{d}_{ij})\,(u_{j,k}^{(t)} - u_{i,k}^{(t)})}_{\text{Directional transport along } \theta_k},\; \underbrace{\Psi_\theta(u_{i,k-1}^{(t)}, u_{i,k+1}^{(t)})}_{\text{Curvature / bifurcation coupling}},\; I_i^{(\ell)}\Big)$$

**Why this solves A/V crossings natively:**
* Orthogonal fibers ($\theta_k$ vs. $\theta_{k+2}$, i.e., $0^\circ$ vs. $90^\circ$) have **no direct mixing path** inside the transport operator.
* At a crossing pixel, a North–South artery flows on the $90^\circ$ fiber while an East–West vein flows on the $0^\circ$ fiber **at the exact same $(x,y)$ coordinate with zero identity collision**.
* Collapsing across fibers at the output head natively detects **Artery**, **Vein**, and **Crossing** pixels simultaneously.

---

### 3.5 Multigrid V-Cycle & Training Stability Protocol
The entire network shares **one set of weights** across V-cycles:
1. **Pre-Smooth** ($\mathcal{S}_\theta$) on active bricks at level $\ell$.
2. **Restrict** ($\mathcal{R}_\theta$): Pool residuals from active children `(code << 2) | {0..3}` into parent `code >> 2`.
3. **Coarse Solve:** Run 4–8 cheap smoothing iterations on the tiny $8 \times 8$ root brick grid (global image context in microseconds).
4. **Prolongate** ($\mathcal{P}_\theta$): Broadcast coarse corrections back down to active children + **Post-Smooth** ($\mathcal{S}_\theta$).

**Stability rules (non-negotiable):**
* **Union Warmup (Splitter):** Early epochs build trees using `GT_active ∪ Pred_active`, annealing smoothly to pure `Pred_active` to eliminate train/inference distribution shift.
* **Deep Supervision:** Apply segmentation + topology loss after **every** V-cycle ($k = 1 \dots K$).
* **Damped Step Size:** Constrain $\Delta t \in (0, 0.5)$ via sigmoid + gradient checkpointing per V-cycle to keep VRAM low on single-GPU setups (e.g., Kaggle T4 15 GB).

---

## 4. Generality Beyond Retinal Vessels

Mathematically, **HaizkLow** targets **sparse 1D curvilinear manifolds and high-curvature boundaries embedded in 2D/3D grids**:
* **Medical:** Retinal A/V segmentation, coronary angiography, 3D brain MRA/CTA vessels, neuron/axon connectomics.
* **Geospatial:** Satellite road networks (highway overpasses = A/V crossings), rivers, utility lines.
* **Industrial:** Surface micro-cracks, PCB trace inspection, semiconductor defect detection.

---

## 5. Falsifiable Kill-Criteria (Benchmark Gates)

We do not claim a breakthrough without clearing **all four gates**:

1. **Pure-Core-First Gate:** Build and test pure standalone `HaizkLow` first (**0 U-Net, 0 Self-Attention, 0 Mamba**, target **<150K params**). External helpers (e.g., coarse-root global mixing) are only permitted if strict ablations prove pure multigrid coarse solving cannot converge without them—never as Lego clutter.
2. **Crucible Gate (Synthetic Crossing Maze):** Must decisively beat parameter-matched Half-UNet / U-Net baselines on synthetic crossing vascular trees (identical non-root intensity, 4–6 crossings) **AND** demonstrate zero-shot $256 \to 512$ resolution extrapolation before moving to real medical datasets.
3. **Honest Wall-Clock Latency Gate:** Custom CUDA C kernel (Kaggle T4) **and** portable pure C99 + SIMD CPU backend must beat dense Half-UNet in **real measured ms/image** at $512^2$ and $1024^2$—not just theoretical FLOPs.
4. **Medical Topology Gate:** Competitive or superior on **RITE (DRIVE-AV)**, **HRF-AV**, and **LES-AV** specifically on **Crossing-Region F1**, **clDice**, and **Betti-0 / Betti-1 connectivity errors**.

---

## 6. Execution Roadmap

- [ ] **Phase 1 — Crucible & Math Proof (PyTorch Vectorized on Kaggle T4)**
  - Build synthetic `Crossing Maze` vascular tree generator (identical branches, root-only A/V cues, 4–6 crossings).
  - Implement vectorized `4×4` Morton Brick Quadtree + Multi-Stat Mipmaps + $SE(2)$ Multigrid V-Cycle.
  - Validate Union Warmup, Deep Supervision, crossing accuracy, zero-shot $256 \to 512$ jump, and Half-UNet baseline comparison.
- [ ] **Phase 2 — Native C/CUDA Engine (`load_inline` + Pure C99/SIMD)**
  - Write fused CUDA C kernels (`forward` + analytical `backward`) for `4×4` brick $SE(2)$ smoothing, Morton halo gather, Restriction/Prolongation, and CUB CSR compaction.
  - Pass `torch.autograd.gradcheck` and profile real ms/image, VRAM, and L2 cache hit rate vs. cuDNN dense convs.
  - Implement portable C99 + AVX2/NEON inference engine ("any hardware" target).
- [ ] **Phase 3 — Real Artery–Vein Benchmarks & Paper**
  - Train & evaluate on RITE (DRIVE-AV), HRF-AV, and LES-AV.
  - Run full ablations: `1×1` vs `4×4` bricks, `K=1` vs `K=4` $SE(2)$ fibers, V-cycle depth $K$, adaptive early-exit, and real hardware latency.

---

## 7. Planned Repository Structure

```text
haizk/low/
├── README.md                 # Architecture spec & kill-criteria (this file)
├── crucible/
│   ├── crossing_maze.py      # Synthetic A/V crossing tree generator & evaluator
│   └── baselines.py          # Parameter-matched Half-UNet / U-Net baselines
├── haizklow/
│   ├── mipmap.py             # O(1) Multi-Stat Image Pyramid (mean/min/max/std/ridge)
│   ├── quadtree.py           # Morton linear quadtree, CSR arena, Union Warmup splitter
│   ├── se2_multigrid.py      # SE(2) orientation-lifted V-Cycle engine
│   └── ops.py                # PyTorch autograd wrapper for custom C/CUDA kernels
├── csrc/
│   ├── morton.h              # Header-only Morton bit-twiddling & 2:1 balance utils
│   ├── haizklow_cuda.cu      # Fused CUDA C 4x4 brick SE(2) smoother + R/P kernels
│   └── haizklow_cpu.c        # Portable pure C99 + AVX2/NEON inference engine
└── benchmarks/
    ├── bench_latency.py      # Real wall-clock ms, VRAM, L2 efficiency benchmarks
    └── eval_topology.py      # Crossing F1, clDice, Betti-0/1 connectivity metrics
```