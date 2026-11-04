/* Backward dL/dx for FROZEN packed CB59, never materializes dense W.
 * Row-major W: each 5bit code represents 8 coefficients and row scale.
 * This is NOT a weight gradient; trainable external rank1/2 factors
 * receive gradients from ordinary PyTorch autograd.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
static inline uint32_t ix5(const uint8_t *data,size_t bytes,uint64_t i){
    uint64_t bit=i*5ULL;
    size_t by=(size_t)(bit>>3);
    unsigned sh=(unsigned)(bit&7);
    if(by>=bytes)return 0;
    uint32_t w=data[by];
    if(sh>3 && by+1<bytes)w|=((uint32_t)data[by+1])<<8;
    return (w>>sh)&31U;
}
int cb59_grad_input(const uint8_t *packed,size_t packed_len,
                    const float *scales,const float *codebook,
                    const float *grad_out,int64_t batch,
                    int64_t rows,int64_t cols,float *grad_x){
    if(!packed||!scales||!codebook||!grad_out||!grad_x||
       batch<1||rows<1||cols<8||cols%8)return -1;
    uint64_t groups=(uint64_t)cols/8ULL;
    if((uint64_t)packed_len*8 < (uint64_t)rows*groups*5ULL)return -2;
    for(int64_t b=0;b<batch;b++){
        float *dx=grad_x+b*cols;
        const float *gy=grad_out+b*rows;
        memset(dx,0,(size_t)cols*sizeof(float));
        for(int64_t r=0;r<rows;r++){
            float value=gy[r]*scales[r];
            if(value==0.0f)continue;
            for(uint64_t g=0;g<groups;g++){
                uint32_t c=ix5(packed,packed_len,(uint64_t)r*groups+g);
                const float *w=codebook+(size_t)c*8;
                float *dest=dx+g*8;
                for(unsigned k=0;k<8;k++)dest[k]+=value*w[k];
            }
        }
    }
    return 0;
}
