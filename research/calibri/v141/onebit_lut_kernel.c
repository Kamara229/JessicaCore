/*
 * CPU direct packed one-bit residual, per-input-group LUT.
 * Same C ABI and byte-identical physical CB140 as v140.
 * Avoids 8 multiplies per (vocab row × group) by precomputing 32x2
 * vector products for each input group. No dense trained matrix materialized.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>

static inline unsigned int read5(const uint8_t *data, size_t pos) {
    size_t start=pos*5u, byte=start>>3;
    unsigned int shift=(unsigned int)(start&7u);
    unsigned int b=(unsigned int)data[byte];
    if(shift>3u) b |= (unsigned int)data[byte+1]<<8;
    return (b>>shift)&31u;
}

static inline unsigned int read1(const uint8_t *data, size_t pos) {
    return (data[pos>>3]>>((unsigned int)pos&7u))&1u;
}

static int checked_geometry(size_t base_len, size_t plane_len,
                            int64_t rows, int64_t dims) {
    if(rows<=0 || dims<=0 || dims%8 || rows>INT32_MAX/8 ||
       dims>INT32_MAX/64) return 1;
    size_t groups=(size_t)rows*((size_t)dims/8u);
    if(base_len!=(groups*5u+7u)/8u || plane_len!=(groups+7u)/8u)
        return 2;
    return 0;
}

/* Embedding row decode: small already-optimal on-demand operation. */
int calibri141_embedding_delta(const uint8_t *base,size_t base_len,
        const uint8_t *plane,size_t plane_len,const float *centers,
        const float *scales,const int64_t *indices,int64_t count,
        int64_t rows,int64_t dims,float *out) {
    if(!base||!plane||!centers||!scales||!indices||!out||count<=0)
        return 1;
    int rc=checked_geometry(base_len,plane_len,rows,dims);
    if(rc) return rc;
    size_t ngroups=(size_t)dims/8;
    for(int64_t n=0;n<count;n++) {
        int64_t row=indices[n];
        if(row<0||row>=rows)return 3;
        float scale=scales[row];
        for(size_t g=0;g<ngroups;g++) {
            size_t pos=(size_t)row*ngroups+g;
            const float *c=centers+((read5(base,pos)*2u+read1(plane,pos))*8u);
            float *target=out+(size_t)n*(size_t)dims+g*8u;
            for(unsigned int j=0;j<8u;j++)target[j]=scale*c[j];
        }
    }
    return 0;
}

/* Exact same residual GEMV as v140 except grouped product caching.
 * L[ngroup][32][2] = input group dot paid residual centroids.
 * Extra allocated only 72×64 float values = 18KB for SmolLM2-135M.
 */
int calibri141_output_delta(const uint8_t *base,size_t base_len,
        const uint8_t *plane,size_t plane_len,const float *centers,
        const float *scales,const float *inputs,int64_t ninputs,
        int64_t rows,int64_t dims,float *out) {
    if(!base||!plane||!centers||!scales||!inputs||!out||ninputs<=0)
        return 1;
    int rc=checked_geometry(base_len,plane_len,rows,dims);
    if(rc)return rc;
    size_t groups=(size_t)dims/8;
    float *lut=(float*)malloc(groups*64u*sizeof(float));
    if(!lut)return 4;
    for(int64_t n=0;n<ninputs;n++) {
        const float *x=inputs+(size_t)n*(size_t)dims;
        for(size_t g=0;g<groups;g++) {
            const float *v=x+g*8u;
            float *line=lut+g*64u;
            for(unsigned int code=0;code<32u;code++) {
                for(unsigned int bit=0;bit<2u;bit++) {
                    const float *r=centers+(code*2u+bit)*8u;
                    float value=0.0f;
                    for(unsigned int j=0;j<8u;j++) value+=v[j]*r[j];
                    line[code*2u+bit]=value;
                }
            }
        }
        float *target=out+(size_t)n*(size_t)rows;
        for(int64_t row=0;row<rows;row++) {
            size_t start=(size_t)row*groups;
            float sum=0.0f;
            for(size_t g=0;g<groups;g++) {
                size_t pos=start+g;
                unsigned int symbol=read5(base,pos)*2u+read1(plane,pos);
                sum+=lut[g*64u+symbol];
            }
            target[row]=sum*scales[row];
        }
    }
    free(lut);
    return 0;
}
