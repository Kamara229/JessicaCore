/* Research packed K8/K16 LUT GEMV. No dense weight reconstruction.
 * Input packing is little-endian FG03, 3 or 4 bits per group of 8.
 * CPU accumulation may differ from dense GEMV's FP32 reduction order.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>

int fg17_gemv(const uint8_t *packed, size_t bytes, const float *centers,
             const float *scales, const float *x, float *y,
             size_t rows, size_t cols, int bits) {
    if (!packed || !centers || !scales || !x || !y ||
        (bits != 3 && bits != 4) || !rows || !cols || cols % 8) return 1;
    const size_t groups = cols / 8, k = (size_t)1 << bits;
    if (rows > SIZE_MAX/groups || rows*groups > (SIZE_MAX-7)/(size_t)bits ||
        bytes != (rows*groups*(size_t)bits+7)/8 ||
        groups > SIZE_MAX/k/sizeof(float)) return 2;
    float *lut = malloc(groups*k*sizeof(float));
    if (!lut) return 3;
    for (size_t g=0;g<groups;g++) {
        for(size_t j=0;j<k;j++) {
            float d=0;
            for(size_t i=0;i<8;i++) d += x[8*g+i]*centers[8*j+i];
            lut[g*k+j]=d;
        }
    }
    #pragma omp parallel for schedule(static) num_threads(2)
    for(size_t r=0;r<rows;r++) {
        float sum=0;
        size_t pos=r*groups*(size_t)bits;
        for(size_t g=0;g<groups;g++,pos+=(size_t)bits) {
            size_t b=pos>>3, shift=pos&7;
            unsigned v=packed[b];
            if(shift+(size_t)bits>8 && b+1<bytes) v |= (unsigned)packed[b+1]<<8;
            size_t code=(v>>shift)&(k-1);
            sum+=lut[g*k+code];
        }
        y[r]=sum*scales[r];
    }
    free(lut);
    return 0;
}
