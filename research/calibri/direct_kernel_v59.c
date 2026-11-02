/* CALIBRI v59: direct compressed lookup-GEMV and demand embedding.
 *
 * No full dense W reconstruction: 5-bit code per 8D trained source
 * vector remains resident, C reconstructs only the 8-value LUT
 * contributions or requested embedding rows. This is a CPU reference
 * kernel to prove working compact execution, NOT an optimized BLAS/GPU.
 *
 * Packed codes: logical index (row*(ncols/8)+group)*5, LSB-first.
 * Row scales/codebooks are originally FP16 on disk; Python expands
 * only those tiny metadata arrays into FP32 for this C kernel.
 * No pointer to original teacher weights exists in either kernel.
 */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include <string.h>

#define CALIBRI_CODES 32
#define CALIBRI_VECTOR 8

static inline unsigned int read5(const uint8_t *data,size_t nbytes,uint64_t index){
    uint64_t start=index*5ULL;
    size_t by=(size_t)(start/8ULL);
    unsigned int shift=(unsigned int)(start%8ULL);
    if(by>=nbytes) return 0;
    unsigned int value=(unsigned int)data[by];
    if(shift>3 && by+1<nbytes)
        value|=((unsigned int)data[by+1])<<8;
    return (value>>shift)&31U;
}

int cb59_linear(const uint8_t *packed,size_t packed_len,
                const float *scales,const float *codebook,
                const float *input,int64_t batch,
                int64_t outrows,int64_t ncols,float *output){
    if(!packed || !scales || !codebook || !input || !output ||
        batch<=0 || outrows<=0 || ncols<=0 || ncols%8!=0)
        return -1;
    const int64_t ngroup=ncols/8;
    if((uint64_t)packed_len*8 < (uint64_t)(outrows*ngroup)*5ULL)
        return -2;
    if(ngroup>INT32_MAX/32) return -3;
    float *lut=(float*)malloc((size_t)ngroup*CALIBRI_CODES*sizeof(float));
    if(!lut) return -4;
    for(int64_t b=0;b<batch;b++){
        const float *x=input+b*ncols;
        for(int64_t g=0;g<ngroup;g++){
            const float *v=x+g*8;
            float *line=lut+g*32;
            for(int c=0;c<32;c++){
                const float *centroid=codebook+c*8;
                float sum=0.0f;
                for(int k=0;k<8;k++) sum+=v[k]*centroid[k];
                line[c]=sum;
            }
        }
        for(int64_t row=0;row<outrows;row++){
            float acc=0.0f;
            for(int64_t group=0;group<ngroup;group++){
                uint64_t ix=(uint64_t)row*ngroup+(uint64_t)group;
                unsigned int c=read5(packed,packed_len,ix);
                acc+=lut[group*32+c];
            }
            output[b*outrows+row]=acc*scales[row];
        }
    }
    free(lut);
    return 0;
}

int cb59_embedding(const uint8_t *packed,size_t packed_len,
                   const float *scales,const float *codebook,
                   const int64_t *ids,int64_t count,
                   int64_t total_rows,int64_t ncols,float *output){
    if(!packed || !scales || !codebook || !ids || !output ||
       count<=0 || total_rows<=0 || ncols<=0 || ncols%8!=0)
        return -1;
    int64_t ngroup=ncols/8;
    if((uint64_t)packed_len*8 < (uint64_t)(total_rows*ngroup)*5ULL)
        return -2;
    for(int64_t t=0;t<count;t++){
        int64_t row=ids[t];
        if(row<0 || row>=total_rows)return -3;
        float factor=scales[row];
        for(int64_t g=0;g<ngroup;g++){
            uint64_t index=(uint64_t)row*ngroup+(uint64_t)g;
            const float *c=codebook+8*read5(packed,packed_len,index);
            float *out=output+t*ncols+g*8;
            for(int k=0;k<8;k++)out[k]=c[k]*factor;
        }
    }
    return 0;
}
