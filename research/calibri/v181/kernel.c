/* v181 diagnostic-only high-rate native 5-bit source; V=1 or V=2.
 * No pretrained dense weight matrix is materialized; only a 32xV LUT
 * and 1/2-wide source code index stream are used.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>
#include <math.h>

static inline unsigned read5_181(const uint8_t *source,size_t size,uint64_t index){
    uint64_t pos=index*5ull;
    size_t by=(size_t)(pos>>3);
    unsigned shift=(unsigned)(pos&7u);
    unsigned value=source[by];
    if(shift>3&&by+1<size)value|=((unsigned)source[by+1])<<8;
    return (value>>shift)&31u;
}

int cb181_linear(const uint8_t *packed,size_t size,const float *scales,
       const float *centers,const float *x,int64_t batch,int64_t rows,
       int64_t cols,int64_t group,float *out){
    if(!packed||!scales||!centers||!x||!out||(group!=1&&group!=2)||
       batch<=0||rows<=0||cols<=0||cols%group)return -1;
    int64_t groups=cols/group;
    if(groups>INT64_MAX/rows|| (uint64_t)rows*(uint64_t)groups>SIZE_MAX/5)return -2;
    if((uint64_t)size*8u<(uint64_t)rows*(uint64_t)groups*5u)return -3;
    if((uint64_t)groups>SIZE_MAX/(32*sizeof(float)))return -4;
    float *lut=malloc(groups*32*sizeof(float));
    if(!lut)return -5;
    for(int64_t b=0;b<batch;b++){
        const float *vec=x+b*cols;
        for(int64_t g=0;g<groups;g++){
            const float *v=vec+g*group;
            float *l=lut+g*32;
            for(int k=0;k<32;k++){
                const float *c=centers+k*group;
                float t=0.0f;
                for(int j=0;j<group;j++)t+=v[j]*c[j];
                l[k]=t;
            }
        }
        for(int64_t row=0;row<rows;row++){
            float sum=0.0f;
            for(int64_t g=0;g<groups;g++){
                uint64_t code=(uint64_t)row*groups+(uint64_t)g;
                sum+=lut[g*32+read5_181(packed,size,code)];
            }
            out[b*rows+row]=sum*scales[row];
        }
    }
    free(lut);
    return 0;
}
int cb181_embedding(const uint8_t *packed,size_t size,const float *scales,
        const float *centers,const int64_t *ids,int64_t count,int64_t rows,
        int64_t cols,int64_t group,float *out){
    if(!packed||!scales||!centers||!ids||!out||(group!=1&&group!=2)||
       count<=0||rows<=0||cols<=0||cols%group)return -1;
    int64_t groups=cols/group;
    if(groups>INT64_MAX/rows)return -2;
    if((uint64_t)size*8u<(uint64_t)rows*(uint64_t)groups*5u)return -3;
    for(int64_t b=0;b<count;b++){
        int64_t row=ids[b];
        if(row<0||row>=rows)return -4;
        for(int64_t g=0;g<groups;g++){
            unsigned code=read5_181(packed,size,(uint64_t)row*groups+(uint64_t)g);
            const float *c=centers+code*group;
            for(int j=0;j<group;j++)out[b*cols+g*group+j]=c[j]*scales[row];
        }
    }
    return 0;
}
