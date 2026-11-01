/*
 * CALIBRI v20: per-row packed INT5 reconstruction with bounded scratch.
 * Bytes decoded ONCE per row per batch, never entire dense matrix.
 * O(cols) temporary RAM independent of number of output rows.
 * CPU reference only. No runtime model, GPU, SIMD/ARM claims.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API int calibri_packed5_matmul_rowcache(
    const uint8_t *payload, size_t payload_bytes,
    const float *scales, const float *inputs, float *out,
    int rows, int cols, int batches)
{
    if (!payload || !scales || !inputs || !out ||
        rows<=0 || cols<=0 || batches<=0)
        return -1;
    const size_t total=(size_t)rows*(size_t)cols;
    if (rows>65535 || cols>65535 || total>(SIZE_MAX-7)/5)
        return -2;
    const size_t expected=(total*5+7)/8;
    if (payload_bytes!=expected) return -3;
    float *one_row=(float*)malloc((size_t)cols*sizeof(float));
    if (!one_row) return -4;
    for (int r=0;r<rows;++r) {
        if ((cols&7)==0) {
            const size_t groups=(size_t)cols/8;
            const uint8_t *base=payload+(size_t)r*groups*5;
            for (size_t g=0;g<groups;++g) {
                const uint8_t *p=base+5*g;
                const uint64_t packed=((uint64_t)p[0]<<32)|
                    ((uint64_t)p[1]<<24)|((uint64_t)p[2]<<16)|
                    ((uint64_t)p[3]<<8)|((uint64_t)p[4]);
                for (int j=0;j<8;++j) {
                    const int sym=(int)((packed>>(35-5*j))&31u);
                    const int val=sym>=16?sym-32:sym;
                    one_row[g*8+j]=(float)val*scales[r];
                }
            }
        } else {
            for (int c=0;c<cols;++c) {
                const size_t bit=((size_t)r*cols+(size_t)c)*5;
                const size_t idx=bit>>3;
                const int shift=(int)(bit&7);
                uint32_t duo=((uint32_t)payload[idx])<<8;
                if (idx+1<expected) duo|=(uint32_t)payload[idx+1];
                const int sym=(int)((duo>>(11-shift))&31u);
                one_row[c]=(float)(sym>=16?sym-32:sym)*scales[r];
            }
        }
        for (int b=0;b<batches;++b) {
            const float *x=inputs+(size_t)b*cols;
            float acc=0.0f;
            for (int c=0;c<cols;++c) {
                acc+=one_row[c]*x[c];
            }
            out[(size_t)b*rows+r]=acc;
        }
    }
    free(one_row);
    return 0;
}
