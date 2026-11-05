/* CALIBRI v187: 4D (output row pair, input col group, 2, 4)
 * compressed source. Each 2x4 tile stores ONE 5-bit code, same
 * 0.625 paid bpp as original 1x8 CB59, plus original FP16 row scales
 * and same 32x8 FP16 per-matrix dictionary. No dense model weights.
 *
 * This C pilot changes operator geometry and its compute pattern.
 * It has not yet passed complete-model answer-quality gates.
 */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <math.h>
#include "../direct_kernel_v59.c"

int cb187_linear(const uint8_t* packed,size_t packed_len,
    const float* scales,const float* codebook,
    const float* input,int64_t batch,int64_t outrows,
    int64_t ncols,float* output) {
    if(!packed||!scales||!codebook||!input||!output||
        batch<=0||outrows<=0||outrows%2||ncols<=0||ncols%4)
        return -1;
    const int64_t ngroup=ncols/4;
    if((uint64_t)packed_len*8ULL <
       (uint64_t)(outrows/2)*ngroup*5ULL)
        return -2;
    if(ngroup>INT32_MAX/64)return -3;
    // The source covers 2 output rows simultaneously, so the LUT
    // holds one dot product for each 4-value half of the 8D centroid.
    float* lut=(float*)malloc((size_t)ngroup*32*2*sizeof(float));
    if(!lut)return -4;
    for(int64_t b=0;b<batch;b++){
        const float* x=input+b*ncols;
        for(int64_t g=0;g<ngroup;g++){
            const float* x4=x+4*g;
            float* l=lut+g*64;
            for(int64_t c=0;c<32;c++){
                const float* v=codebook+c*8;
                float a=0.f,z=0.f;
                for(int t=0;t<4;t++){
                    a+=v[t]*x4[t];
                    z+=v[4+t]*x4[t];
                }
                l[2*c]=a;
                l[2*c+1]=z;
            }
        }
        for(int64_t pair=0;pair<outrows/2;pair++){
            float result0=0.f,result1=0.f;
            for(int64_t g=0;g<ngroup;g++){
                uint64_t index=(uint64_t)pair*ngroup+g;
                unsigned code=read5(packed,packed_len,index);
                const float* contribution=lut+g*64+code*2;
                result0+=contribution[0];
                result1+=contribution[1];
            }
            const int64_t r=pair*2;
            output[b*outrows+r]=result0*scales[r];
            output[b*outrows+r+1]=result1*scales[r+1];
        }
    }
    free(lut);
    return 0;
}

int cb187_embedding(const uint8_t* packed,size_t packed_len,
    const float* scales,const float* codebook,
    const int64_t* ids,int64_t count,
    int64_t outrows,int64_t ncols,float* output) {
    if(!packed||!scales||!codebook||!ids||!output||
       count<=0||outrows<=0||outrows%2||ncols<=0||ncols%4)
        return -1;
    const int64_t ngroup=ncols/4;
    if((uint64_t)packed_len*8ULL <
       (uint64_t)(outrows/2)*ngroup*5ULL)
        return -2;
    for(int64_t i=0;i<count;i++){
        const int64_t row=ids[i];
        if(row<0||row>=outrows)return -3;
        const int64_t pair=row/2;
        const int64_t half=row%2;
        const float scale=scales[row];
        for(int64_t g=0;g<ngroup;g++){
            const uint64_t index=(uint64_t)pair*ngroup+g;
            unsigned code=read5(packed,packed_len,index);
            const float* values=codebook+8*code+4*half;
            float* result=output+i*ncols+4*g;
            for(int t=0;t<4;t++)
                result[t]=values[t]*scale;
        }
    }
    return 0;
}
