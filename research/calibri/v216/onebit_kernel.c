/* CALIBRI v216 one physical packed extra bit per 8 source weights.
 * 16 adjacent vocabulary rows share ONE 8-component residual direction.
 * Source-only FP16 direction + packed 1-bit signs, no full dense matrix.
 * Add into existing true native C CB59 LM-head GEMV outputs.
 * Portable native CPU, no global allocation; no test or internet access.
 */
#include <stdint.h>
#include <stddef.h>
#include <math.h>

int calibri_onebit_head_add(const float *x, float *out,
                           const uint8_t *packed, const float *direction,
                           int rows, int cols, int block_rows) {
    if (!x || !out || !packed || !direction ||
        rows <= 0 || cols <= 0 || block_rows != 16 ||
        rows % block_rows != 0 || cols % 8 != 0)
        return -1;
    const int groups = cols / 8;
    /* Require 16 rows / group; bit array packs row-major. */
    if ((rows * groups) % 8 != 0) return -2;
    for (int block = 0; block < rows / 16; ++block) {
        const float *d = direction + (size_t)block * 8;
        /* Compute each 8D dot once for 16 neighboring rows. */
        for (int row_in_block = 0; row_in_block < 16; ++row_in_block) {
            const int row = block * 16 + row_in_block;
            const size_t bit_begin = (size_t)row * (size_t)groups;
            float total = 0.0f;
            for (int group = 0; group < groups; ++group) {
                const float *v = x + (size_t)group * 8;
                float value = 0.0f;
                for (int k = 0; k < 8; ++k)
                    value += d[k] * v[k];
                const size_t bit_index = bit_begin + (size_t)group;
                const uint8_t bit = (packed[bit_index >> 3] >>
                                    (bit_index & 7)) & 1u;
                total += bit ? value : -value;
            }
            out[row] += total;
        }
    }
    return 0;
}
