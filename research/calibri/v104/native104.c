/* CALIBRI v104: C packed-resident K32V8 native GEMV with 8 lexical banks.
 *
 * The original trained head AND embedding consume 5 bits per 8 weight
 * groups, with one bank for each contiguous eighth of the full 49,152
 * vocab. 8 additional physically paid FP16 codebook banks in CB104.
 * No full source dense head, vectors, or teacher weights materialized.
 * Strict bounds + original 5-bit little-endian ABI.
 */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
#include <string.h>
#include <math.h>
#define CB104_BANKS 8
#define CB104_CODES 32

static inline uint32_t read5(const uint8_t *raw,size_t bytes,uint64_t ix){
    uint64_t bits=ix*5ULL;
    size_t i=(size_t)(bits>>3);
    unsigned int shift=(unsigned int)(bits&7U);
    uint32_t x=raw[i];
    if(shift>3 && i+1<bytes)x|=((uint32_t)raw[i+1])<<8;
    return (x>>shift)&31U;
}
static inline int valid(const uint8_t *packed,size_t size,const float *scales,
             const float *banks,int64_t rows,int64_t cols){
    if(!packed||!scales||!banks||rows<=0||cols<=0||cols%8)return 0;
    if(rows!=49152||cols!=576||rows%CB104_BANKS)return 0;
    if((uint64_t)size*8 < (uint64_t)(rows*(cols/8))*5ULL)return 0;
    return 1;
}
int clb104_head(const uint8_t *packed,size_t size,const float *scales,
        const float *banks,const float *input,int64_t batch,
        int64_t rows,int64_t cols,float *output){
    if(!input||!output||batch<=0||!valid(packed,size,scales,banks,rows,cols))
        return -1;
    const int64_t groups=cols/8;
    const int64_t rows_per_bank=rows/CB104_BANKS;
    const size_t items=(size_t)CB104_BANKS*(size_t)groups*CB104_CODES;
    float *lut=(float*)malloc(items*sizeof(float));
    if(!lut)return -2;
    for(int64_t t=0;t<batch;t++){
        const float *x=input+t*cols;
        for(int bank=0;bank<CB104_BANKS;bank++){
            const float *dict=banks+(size_t)bank*CB104_CODES*8;
            for(int64_t g=0;g<groups;g++){
                const float *v=x+g*8;
                float *dst=lut+((size_t)bank*groups+g)*CB104_CODES;
                for(int c=0;c<CB104_CODES;c++){
                    const float *w=dict+c*8;
                    float acc=0.0f;
                    for(int k=0;k<8;k++)acc+=v[k]*w[k];
                    dst[c]=acc;
                }
            }
        }
        for(int64_t row=0;row<rows;row++){
            int bank=(int)(row/rows_per_bank);
            float val=0.0f;
            for(int64_t group=0;group<groups;group++){
                const uint64_t pos=(uint64_t)row*(uint64_t)groups+(uint64_t)group;
                uint32_t code=read5(packed,size,pos);
                val+=lut[((size_t)bank*groups+group)*CB104_CODES+code];
            }
            output[t*rows+row]=val*scales[row];
        }
    }
    free(lut);
    return 0;
}
int clb104_embedding(const uint8_t *packed,size_t size,const float *scales,
        const float *banks,const int64_t *tokens,int64_t count,
        int64_t rows,int64_t cols,float *output){
    if(!tokens||!output||count<=0||
       !valid(packed,size,scales,banks,rows,cols))return -1;
    int64_t groups=cols/8;
    int64_t rows_per_bank=rows/CB104_BANKS;
    for(int64_t t=0;t<count;t++){
        int64_t row=tokens[t];
        if(row<0||row>=rows)return -2;
        const float *dict=banks+(row/rows_per_bank)*CB104_CODES*8;
        const float factor=scales[row];
        for(int64_t g=0;g<groups;g++){
            uint32_t code=read5(packed,size,(uint64_t)row*groups+g);
            const float *centroid=dict+code*8;
            float *dst=output+t*cols+g*8;
            for(int k=0;k<8;k++)dst[k]=centroid[k]*factor;
        }
    }
    return 0;
}
