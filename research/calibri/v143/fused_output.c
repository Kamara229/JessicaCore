/*
 * CALIBRI v143 fusion: SAME physical CB140 source as v140/v141, no training.
 * Original 5-bit base codewords and paid 1-bit residual share ONE pass over
 * output rows, replacing CB59-GEMV plus residual-GEMV with a 64-entry LUT.
 * Encoded base/index streams are NEVER converted into dense source weights.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

static inline uint8_t read5(const uint8_t *a, size_t i) {
    size_t start=i*5u, at=start>>3;
    unsigned int shift=(unsigned int)(start&7u);
    unsigned int val=(unsigned int)a[at];
    if(shift>3u)val|=(unsigned int)a[at+1]<<8;
    return (uint8_t)((val>>shift)&31u);
}
static inline uint8_t read1(const uint8_t *a, size_t i) {
    return (uint8_t)((a[i>>3]>>(i&7u))&1u);
}

/* out: [batch, vocab]; matrices come from two paid packed streams:
 * 5-bit base vectors of shape [32,8] and 1-bit residual vectors [32,2,8].
 * The same row scales are shared. This is the exact combined weight:
 * W[r,g,j]=scale[r]*(base[c(r,g),j]+delta[c(r,g),bit(r,g),j]).
 */
int calibri143_fused_output(
    const uint8_t *five,size_t five_len,
    const uint8_t *one,size_t one_len,
    const float *base_centers,const float *delta_centers,
    const float *scales,const float *input,
    int64_t batch,int64_t rows,int64_t width,float *output) {
    if(!five||!one||!base_centers||!delta_centers||!scales||
       !input||!output||batch<1||rows<1||width<8||width%8||
       rows>INT32_MAX/8||width>INT32_MAX/64)return 1;
    size_t groups=(size_t)width/8u;
    size_t code_count=(size_t)rows*groups;
    if(five_len!=(code_count*5u+7u)/8u || one_len!=(code_count+7u)/8u)
        return 2;
    /* Only 72 * 64 * 4 = 18KiB for 135M: shared dynamic LUT. */
    float *lut=(float*)malloc(groups*64u*sizeof(float));
    if(!lut)return 4;
    for(int64_t n=0;n<batch;n++) {
        const float *x=input+(size_t)n*(size_t)width;
        float *out=output+(size_t)n*(size_t)rows;
        for(size_t g=0;g<groups;g++) {
            const float *v=x+8u*g;
            float *dest=lut+64u*g;
            for(unsigned int c=0;c<32u;c++) {
                const float *b=base_centers+8u*c;
                for(unsigned int bit=0;bit<2u;bit++) {
                    const float *r=delta_centers+8u*(c*2u+bit);
                    float sum=0.0f;
                    for(unsigned int j=0;j<8u;j++)
                        sum+=v[j]*(b[j]+r[j]);
                    dest[c*2u+bit]=sum;
                }
            }
        }
        for(int64_t row=0;row<rows;row++) {
            size_t offset=(size_t)row*groups;
            float sum=0.0f;
            for(size_t g=0;g<groups;g++) {
                size_t pos=offset+g;
                unsigned int symbol=read5(five,pos)*2u+read1(one,pos);
                sum+=lut[g*64u+symbol];
            }
            out[row]=sum*scales[row];
        }
    }
    free(lut);
    return 0;
}
