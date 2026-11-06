/* CALIBRI v218: partitioned K32V8 codebooks, resident packed native C.
 *
 * No dense 2D matrix is materialized in the inference runtime.
 * v206's original 5-bit per 8D vector, row scale, 1D weights and ABI are
 * retained. Tied embedding/head source uses a local 32x8 FP16 dictionary
 * per fixed block of 128 or 256 vocabulary rows. All extra dictionaries
 * are physically charged in the signed sidecar and must fit <=0.700 bpp.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

static inline unsigned int read_code5(const uint8_t* bytes,
                                      size_t nbytes, uint64_t i) {
    const uint64_t bits=i*5ull;
    const size_t by=(size_t)(bits>>3);
    const unsigned int shift=(unsigned int)(bits&7);
    if(by>=nbytes) return 0;
    unsigned int w=(unsigned int)bytes[by];
    if(shift>3 && by+1<nbytes)w|=(unsigned int)bytes[by+1]<<8;
    return (w>>shift)&31U;
}

int cb218_part_linear(const uint8_t* packed,size_t nbytes,
                       const float* scales,const float* dictionary,
                       const float* activations,int64_t batch,
                       int64_t nrows,int64_t ncols,int64_t rows_per_block,
                       float* output) {
    if(!packed||!scales||!dictionary||!activations||!output
        ||batch<=0||nrows<=0||ncols<=0||rows_per_block<=0
        ||nrows%rows_per_block||ncols%8) return -1;
    const int64_t groups=ncols/8;
    if((uint64_t)nbytes*8ull<(uint64_t)nrows*(uint64_t)groups*5ull)return -2;
    if(groups>INT32_MAX/32)return -3;
    float* lut=(float*)malloc((size_t)groups*32*sizeof(float));
    if(!lut)return -4;
    for(int64_t t=0;t<batch;t++){
        const float* input=activations+(size_t)t*(size_t)ncols;
        float* out=output+(size_t)t*(size_t)nrows;
        for(int64_t block=0;block<nrows/rows_per_block;block++){
            const float* centers=dictionary+(size_t)block*32u*8u;
            for(int64_t group=0;group<groups;group++){
                const float* value=input+group*8;
                float* line=lut+group*32;
                for(int i=0;i<32;i++){
                    const float* c=centers+i*8;
                    float z=0.0f;
                    for(int k=0;k<8;k++)z+=value[k]*c[k];
                    line[i]=z;
                }
            }
            const int64_t start=block*rows_per_block;
            const int64_t stop=start+rows_per_block;
            for(int64_t row=start;row<stop;row++){
                const uint64_t base=(uint64_t)row*(uint64_t)groups;
                float sum=0.0f;
                for(int64_t g=0;g<groups;g++){
                    const unsigned int idx=read_code5(packed,nbytes,base+(uint64_t)g);
                    sum+=lut[g*32+(int64_t)idx];
                }
                out[row]=sum*scales[row];
            }
        }
    }
    free(lut);
    return 0;
}

int cb218_part_embedding(const uint8_t* packed,size_t nbytes,
                          const float* scales,const float* dictionary,
                          const int64_t* ids,int64_t count,
                          int64_t nrows,int64_t ncols,int64_t rows_per_block,
                          float* output) {
    if(!packed||!scales||!dictionary||!ids||!output
       ||count<=0||nrows<=0||ncols<=0||rows_per_block<=0
       ||nrows%rows_per_block||ncols%8)return -1;
    const int64_t groups=ncols/8;
    if((uint64_t)nbytes*8ull<(uint64_t)nrows*(uint64_t)groups*5ull)return -2;
    for(int64_t i=0;i<count;i++){
        const int64_t row=ids[i];
        if(row<0||row>=nrows)return -3;
        const int64_t block=row/rows_per_block;
        const float* centers=dictionary+(size_t)block*32u*8u;
        const float scale=scales[row];
        for(int64_t g=0;g<groups;g++){
            const uint64_t index=(uint64_t)row*(uint64_t)groups+(uint64_t)g;
            const unsigned int code=read_code5(packed,nbytes,index);
            const float* c=centers+code*8u;
            float* result=output+(size_t)i*(size_t)ncols+(size_t)g*8u;
            for(int k=0;k<8;k++)result[k]=c[k]*scale;
        }
    }
    return 0;
}
