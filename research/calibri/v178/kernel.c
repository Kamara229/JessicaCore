/* CALIBRI v178: native five-bit K32/V8 direct matvec and embedding,
 * with physically-paid per-column scale factors; no dense trained source.
 *
 * Approx W[row,col] = row_scale[row] * centroid[code, col%8] * col_scale[col].
 * Linear does scaled inputs; embedding scales decoded coordinates.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <limits.h>

static inline unsigned get5(const uint8_t *data,size_t size,uint64_t ix){
    uint64_t offset=ix*5u;
    size_t by=(size_t)(offset>>3);
    unsigned shift=(unsigned)(offset&7u);
    unsigned word=data[by];
    if(shift>3 && by+1<size)word|=((unsigned)data[by+1])<<8;
    return (word>>shift)&31u;
}

int cb178_linear(const uint8_t *packed,size_t size,
                 const float *row_scales,const float *codebook,
                 const float *column_scales,const float *input,
                 int64_t batch,int64_t rows,int64_t columns,float *output){
    if(!packed || !row_scales || !codebook || !column_scales || !input || !output ||
       batch<=0 || rows<=0 || columns<=0 || columns%8)return -1;
    int64_t groups=columns/8;
    if(groups>INT64_MAX/rows || (uint64_t)groups*(uint64_t)rows > SIZE_MAX/5u)
        return -2;
    if((uint64_t)size*8u < (uint64_t)rows*(uint64_t)groups*5u) return -3;
    if((size_t)groups>SIZE_MAX/(32*sizeof(float)))return -4;
    float *lut=(float*)malloc((size_t)groups*32*sizeof(float));
    if(!lut)return -5;
    for(int64_t b=0;b<batch;b++){
        const float *x=input+b*columns;
        for(int64_t g=0;g<groups;g++){
            float *out_lut=lut+g*32;
            float scaled_input[8];
            for(int j=0;j<8;j++){
                int64_t col=g*8+j;
                scaled_input[j]=x[col]*column_scales[col];
            }
            for(unsigned k=0;k<32;k++){
                const float *cent=codebook+k*8;
                float sum=0.0f;
                for(int j=0;j<8;j++) sum+=scaled_input[j]*cent[j];
                out_lut[k]=sum;
            }
        }
        for(int64_t row=0;row<rows;row++){
            float accumulator=0.0f;
            for(int64_t g=0;g<groups;g++){
                uint64_t index=(uint64_t)row*groups+(uint64_t)g;
                accumulator+=lut[g*32+get5(packed,size,index)];
            }
            output[b*rows+row]=accumulator*row_scales[row];
        }
    }
    free(lut);
    return 0;
}

int cb178_embedding(const uint8_t *packed,size_t size,
                    const float *row_scales,const float *codebook,
                    const float *column_scales,const int64_t *ids,
                    int64_t count,int64_t rows,int64_t columns,float *output){
    if(!packed || !row_scales || !codebook || !column_scales || !ids || !output ||
       count<=0 || rows<=0 || columns<=0 || columns%8)return -1;
    int64_t groups=columns/8;
    if(groups>INT64_MAX/rows)return -2;
    if((uint64_t)size*8u < (uint64_t)rows*(uint64_t)groups*5u)return -3;
    for(int64_t t=0;t<count;t++){
        int64_t row=ids[t];
        if(row<0 || row>=rows)return -4;
        for(int64_t g=0;g<groups;g++){
            uint64_t ix=(uint64_t)row*groups+(uint64_t)g;
            unsigned code=get5(packed,size,ix);
            const float *cent=codebook+code*8;
            float *target=output+t*columns+g*8;
            for(int j=0;j<8;j++){
                int64_t col=g*8+j;
                target[j]=cent[j]*row_scales[row]*column_scales[col];
            }
        }
    }
    return 0;
}
