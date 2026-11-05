/* CALIBRI v177 mixed 4/5/6-bit direct-resident lookup-GEMV.
 * Only compressed codes, per-matrix shared codebook and row scales are used.
 * No dense matrix allocation; embedding reconstructs requested rows only.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

static inline uint32_t lookup(const uint8_t *packed, size_t size,
                              uint64_t index, unsigned bits) {
    uint64_t pos = index * (uint64_t)bits;
    size_t offset = (size_t)(pos >> 3);
    unsigned shift = (unsigned)(pos & 7u);
    uint32_t data = (uint32_t)packed[offset];
    if (shift + bits > 8 && offset + 1 < size)
        data |= ((uint32_t)packed[offset+1]) << 8;
    return (data >> shift) & ((1u << bits) - 1u);
}

int cb177_linear(const uint8_t *packed, size_t size,
                 const float *scale, const float *centroid,
                 const float *input, int64_t batch, int64_t rows,
                 int64_t columns, int64_t bits, float *output) {
    if (!packed || !scale || !centroid || !input || !output ||
        (bits != 4 && bits != 5 && bits != 6) || batch <= 0 ||
        rows <= 0 || columns <= 0 || columns % 8)
        return -1;
    int64_t groups = columns / 8;
    if (groups > INT64_MAX / rows ||
        (uint64_t)groups * (uint64_t)rows > SIZE_MAX / (uint64_t)bits)
        return -2;
    uint64_t count = (uint64_t)groups * (uint64_t)rows;
    if ((uint64_t)size * 8ull < count * (uint64_t)bits) return -3;
    size_t k = (size_t)1 << bits;
    if ((size_t)groups > SIZE_MAX / (k*sizeof(float))) return -4;
    float *lut = (float*)malloc((size_t)groups*k*sizeof(float));
    if (!lut) return -5;
    for (int64_t batch_ix=0; batch_ix<batch; batch_ix++) {
        const float *x = input + batch_ix*columns;
        for (int64_t group=0;group<groups;group++) {
            float *row_lut = lut + (size_t)group*k;
            const float *block = x + 8*group;
            for (size_t code=0;code<k;code++) {
                const float *c = centroid+code*8;
                float accum=0.0f;
                for (int j=0;j<8;j++) accum += block[j]*c[j];
                row_lut[code]=accum;
            }
        }
        for (int64_t row=0;row<rows;row++) {
            float accum=0.0f;
            for (int64_t group=0;group<groups;group++) {
                uint64_t code_index=(uint64_t)row*(uint64_t)groups+(uint64_t)group;
                unsigned code=lookup(packed,size,code_index,(unsigned)bits);
                accum += lut[(size_t)group*k+(size_t)code];
            }
            output[batch_ix*rows+row] = accum*scale[row];
        }
    }
    free(lut);
    return 0;
}

int cb177_embedding(const uint8_t *packed, size_t size,
                    const float *scale, const float *centroid,
                    const int64_t *ids, int64_t count, int64_t rows,
                    int64_t columns, int64_t bits, float *output) {
    if (!packed || !scale || !centroid || !ids || !output ||
        (bits != 4 && bits != 5 && bits != 6) || count <= 0 ||
        rows <= 0 || columns <= 0 || columns % 8)
        return -1;
    int64_t groups=columns/8;
    if (groups > INT64_MAX/rows) return -2;
    if ((uint64_t)size*8ull < (uint64_t)rows*(uint64_t)groups*(uint64_t)bits)
        return -3;
    for (int64_t t=0;t<count;t++) {
        int64_t row=ids[t];
        if (row<0 || row>=rows) return -4;
        for (int64_t g=0;g<groups;g++) {
            unsigned code=lookup(packed,size,(uint64_t)row*groups+g,(unsigned)bits);
            const float *c=centroid+8*code;
            float *out=output+t*columns+g*8;
            for (int j=0;j<8;j++) out[j]=c[j]*scale[row];
        }
    }
    return 0;
}
