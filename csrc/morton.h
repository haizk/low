// csrc/morton.h
// HaizkLow — Header-only Pointerless 2D Morton Z-Order Quadtree Primitives
// Compiles cleanly under C99/C11 (gcc/clang) and CUDA C/C++ (nvcc).
//
// Smoke test:
//   gcc -O3 -DMORTON_SELFTEST -x c csrc/morton.h -o /tmp/test_morton && /tmp/test_morton

#ifndef HAIZKLOW_MORTON_H
#define HAIZKLOW_MORTON_H

#include <stdint.h>

#ifdef __CUDACC__
  #define HL_HD __host__ __device__ __forceinline__
#else
  #define HL_HD static inline
#endif

#define HL_MORTON_X_MASK 0x55555555u
#define HL_MORTON_Y_MASK 0xaaaaaaaau

// Spread lower 16 bits of n into even bit positions (0, 2, 4, ..., 30)
HL_HD uint32_t hl_part1by1(uint32_t n) {
    n &= 0x0000ffffu;
    n = (n | (n << 8)) & 0x00ff00ffu;
    n = (n | (n << 4)) & 0x0f0f0f0fu;
    n = (n | (n << 2)) & 0x33333333u;
    n = (n | (n << 1)) & 0x55555555u;
    return n;
}

// Inverse of hl_part1by1: compact even bit positions back into lower 16 bits
HL_HD uint32_t hl_compact1by1(uint32_t n) {
    n &= 0x55555555u;
    n = (n | (n >> 1)) & 0x33333333u;
    n = (n | (n >> 2)) & 0x0f0f0f0fu;
    n = (n | (n >> 4)) & 0x00ff00ffu;
    n = (n | (n >> 8)) & 0x0000ffffu;
    return n;
}

// Encode brick coordinates (bx, by) -> 32-bit Morton Z-order key
HL_HD uint32_t hl_morton2d_encode(uint16_t bx, uint16_t by) {
    return hl_part1by1((uint32_t)bx) | (hl_part1by1((uint32_t)by) << 1);
}

// Decode 32-bit Morton key -> brick coordinates (bx, by)
HL_HD void hl_morton2d_decode(uint32_t code, uint16_t *bx, uint16_t *by) {
    *bx = (uint16_t)hl_compact1by1(code);
    *by = (uint16_t)hl_compact1by1(code >> 1);
}

// O(1) Quadtree hierarchy navigation (1 clock cycle)
// Note: expanding sorted parents into q=0,1,2,3 preserves Morton order with ZERO sorting!
HL_HD uint32_t hl_morton2d_parent(uint32_t code) {
    return code >> 2;
}

// quadrant q in {0:(0,0), 1:(1,0), 2:(0,1), 3:(1,1)}
HL_HD uint32_t hl_morton2d_child(uint32_t parent_code, uint32_t q) {
    return (parent_code << 2) | (q & 3u);
}

// Branchless bitwise cardinal neighbor navigation (no decode/re-encode!)
// Preconditions: caller guards bounds (bx + 1 < grid_w, bx > 0, by + 1 < grid_h, by > 0)
HL_HD uint32_t hl_morton2d_inc_x(uint32_t code) {
    return (((code | HL_MORTON_Y_MASK) + 1u) & HL_MORTON_X_MASK) | (code & HL_MORTON_Y_MASK);
}

HL_HD uint32_t hl_morton2d_dec_x(uint32_t code) {
    return (((code & HL_MORTON_X_MASK) - 1u) & HL_MORTON_X_MASK) | (code & HL_MORTON_Y_MASK);
}

HL_HD uint32_t hl_morton2d_inc_y(uint32_t code) {
    return (((code | HL_MORTON_X_MASK) + 1u) & HL_MORTON_Y_MASK) | (code & HL_MORTON_X_MASK);
}

HL_HD uint32_t hl_morton2d_dec_y(uint32_t code) {
    return (((code & HL_MORTON_Y_MASK) - 1u) & HL_MORTON_Y_MASK) | (code & HL_MORTON_X_MASK);
}

#ifdef MORTON_SELFTEST
#include <assert.h>
#include <stdio.h>

int main(void) {
    for (uint16_t y = 1; y < 127; ++y) {
        for (uint16_t x = 1; x < 127; ++x) {
            uint32_t c = hl_morton2d_encode(x, y);
            uint16_t dx, dy;
            hl_morton2d_decode(c, &dx, &dy);
            assert(dx == x && dy == y);

            hl_morton2d_decode(hl_morton2d_inc_x(c), &dx, &dy);
            assert(dx == x + 1 && dy == y);
            hl_morton2d_decode(hl_morton2d_dec_x(c), &dx, &dy);
            assert(dx == x - 1 && dy == y);
            hl_morton2d_decode(hl_morton2d_inc_y(c), &dx, &dy);
            assert(dx == x && dy == y + 1);
            hl_morton2d_decode(hl_morton2d_dec_y(c), &dx, &dy);
            assert(dx == x && dy == y - 1);

            uint32_t p = hl_morton2d_parent(c);
            uint32_t q = ((uint32_t)(y & 1) << 1) | (uint32_t)(x & 1);
            assert(p == hl_morton2d_encode(x >> 1, y >> 1));
            assert(hl_morton2d_child(p, q) == c);
        }
    }
    printf("[HaizkLow] morton.h selftest PASSED (126x126 exhaustive).\n");
    return 0;
}
#endif

#endif // HAIZKLOW_MORTON_H