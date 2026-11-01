/*
 * CALIBRI v21: x86 AVX2+FMA row-buffer INT5 reference.
 * Native execution from packed 5-bit stream, only one float row scratch.
 * Fallback function returns -5 for non-AVX2/FMA processors.
 * DO NOT ship as a platform-neutral Android implementation.
 */
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#if defined(__x86_64__) || defined(__i386__)
#include <immintrin.h>
#endif

#if defined(_WIN32)
#define API __declspec(dllexport)
#else
#define API __attribute__((visibility("default")))
#endif

#if defined(__x86_64__) || defined(__i386__)
__attribute__((target("avx2,fma")))
static float dot_f32_avx2(const float *a,const float *b,int n)
{
    __m256 acc0=_mm256_setzero_ps(),acc1=_mm256_setzero_ps();
    int c=0;
    for (;c+16<=n;c+=16) {
        const __m256 x0=_mm256_loadu_ps(a+c);
        const __m256 y0=_mm256_loadu_ps(b+c);
        const __m256 x1=_mm256_loadu_ps(a+c+8);
        const __m256 y1=_mm256_loadu_ps(b+c+8);
        acc0=_mm256_fmadd_ps(x0,y0,acc0);
        acc1=_mm256_fmadd_ps(x1,y1,acc1);
    }
    acc0=_mm256_add_ps(acc0,acc1);
    for (;c+8<=n;c+=8) {
        acc0=_mm256_fmadd_ps(_mm256_loadu_ps(a+c),
                             _mm256_loadu_ps(b+c),acc0);
    }
    float parts[8];
    _mm256_storeu_ps(parts,acc0);
    float result=0.0f;
    for (int i=0;i<8;++i)result+=parts[i];
    for (;c<n;++c)result+=a[c]*b[c];
    return result;
}
#endif

API int calibri_packed5_matmul_rowcache_simd(
    const uint8_t *payload,size_t payload_bytes,
    const float *scales,const float *inputs,float *out,
    int rows,int cols,int batches)
{
#if !defined(__x86_64__) && !defined(__i386__)
    (void)payload;(void)payload_bytes;(void)scales;(void)inputs;
    (void)out;(void)rows;(void)cols;(void)batches;
    return -5;
#else
    __builtin_cpu_init();
    if (!__builtin_cpu_supports("avx2") || !__builtin_cpu_supports("fma"))
        return -5;
    if (!payload || !scales || !inputs || !out ||
        rows<=0 || cols<=0 || batches<=0) return -1;
    const size_t total=(size_t)rows*(size_t)cols;
    if (rows>65535 || cols>65535 || total>(SIZE_MAX-7)/5) return -2;
    const size_t needed=(total*5+7)/8;
    if (needed!=payload_bytes) return -3;
    float *one_row=(float*)malloc((size_t)cols*sizeof(float));
    if (!one_row)return -4;
    for (int r=0;r<rows;++r) {
        if ((cols&7)==0) {
            const size_t groups=(size_t)cols/8;
            const uint8_t *p=payload+(size_t)r*groups*5;
            for (size_t g=0;g<groups;++g) {
                const uint8_t *q=p+5*g;
                const uint64_t word=((uint64_t)q[0]<<32)|
                  ((uint64_t)q[1]<<24)|((uint64_t)q[2]<<16)|
                  ((uint64_t)q[3]<<8)|((uint64_t)q[4]);
                for (int j=0;j<8;++j){
                    const int v=(int)((word>>(35-5*j))&31u);
                    one_row[g*8+j]=(float)(v>=16?v-32:v)*scales[r];
                }
            }
        } else {
            for (int c=0;c<cols;++c) {
                const size_t at=((size_t)r*cols+(size_t)c)*5;
                const size_t bi=at>>3;
                const int shift=(int)(at&7);
                uint32_t bits=(uint32_t)payload[bi]<<8;
                if (bi+1<needed)bits|=payload[bi+1];
                const int v=(int)((bits>>(11-shift))&31u);
                one_row[c]=(float)(v>=16?v-32:v)*scales[r];
            }
        }
        for (int b=0;b<batches;++b) {
            const float *x=inputs+(size_t)b*cols;
            out[(size_t)b*rows+r]=dot_f32_avx2(one_row,x,cols);
        }
    }
    free(one_row);
    return 0;
#endif
}
