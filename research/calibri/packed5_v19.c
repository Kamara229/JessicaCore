/*
 * CALIBRI v19 CPU research-only packed INT5 GEMM.
 * Decode each 5-bit coefficient inline during dot products, never allocate
 * or materialize a dense (rows x cols) weight matrix.
 * This is NOT a production kernel, SIMD implementation, GPU or Android port.
 */
#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

API int calibri_packed5_matmul(
    const uint8_t *payload, size_t payload_bytes,
    const float *scales, const float *inputs, float *out,
    int rows, int cols, int batches)
{
    if (!payload || !scales || !inputs || !out ||
        rows <= 0 || cols <= 0 || batches <= 0)
        return -1;
    const size_t total = (size_t)rows*(size_t)cols;
    if (rows > 65535 || cols > 65535 || total > (SIZE_MAX-7)/5)
        return -2;
    const size_t expected = (total*5 + 7)/8;
    if (payload_bytes != expected) return -3;
    for (int b=0;b<batches;++b) {
        const float *x = inputs + (size_t)b*cols;
        for (int r=0;r<rows;++r) {
            float acc=0.0f;
            if ((cols & 7)==0) {
                const size_t groups=(size_t)cols/8;
                const uint8_t *bytes=payload + (size_t)r*groups*5;
                for (size_t g=0;g<groups;++g) {
                    const uint8_t *p=bytes+g*5;
                    const uint64_t packed=((uint64_t)p[0]<<32)|
                       ((uint64_t)p[1]<<24)|((uint64_t)p[2]<<16)|
                       ((uint64_t)p[3]<<8)|((uint64_t)p[4]);
                    for (int j=0;j<8;++j) {
                        const int raw=(int)((packed>>(35-5*j))&31u);
                        const int signed_val=raw>=16?raw-32:raw;
                        acc += (float)signed_val*x[g*8+j];
                    }
                }
            } else {
                for (int c=0;c<cols;++c) {
                    const size_t bit=((size_t)r*(size_t)cols+(size_t)c)*5;
                    const size_t idx=bit>>3;
                    const int shift=(int)(bit&7);
                    uint32_t duo=((uint32_t)payload[idx])<<8;
                    if (idx+1<expected) duo|=(uint32_t)payload[idx+1];
                    int val=(int)((duo>>(11-shift))&31u);
                    if (val>=16)val-=32;
                    acc+=(float)val*x[c];
                }
            }
            out[(size_t)b*rows+r]=acc*scales[r];
        }
    }
    return 0;
}
