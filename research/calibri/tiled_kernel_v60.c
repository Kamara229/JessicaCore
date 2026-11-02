/* CALIBRI CB60 direct execution: K32V8 compressed source, 256-row tiled
 * FP16 centroid books. All actual model weights remain 5bit code IDs.
 *
 * For EACH input and 256-row tile, precompute 32-centroid x ncols/8
 * dot-product LUT, then stream 5bit indices directly from resident source.
 * Never construct full trained dense matrix. Tied token embedding decodes
 * ONLY requested rows, using the tile's own correct source codebook.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>

static inline unsigned int source5(const uint8_t *data,size_t len,uint64_t ix){
    uint64_t bit=ix*5ULL;
    size_t offset=(size_t)(bit>>3);
    unsigned int sh=(unsigned int)(bit&7ULL);
    if(offset>=len)return 0;
    unsigned int value=(unsigned int)data[offset];
    if(sh>3&&offset+1<len)value|=(unsigned int)data[offset+1]<<8;
    return (value>>sh)&31U;
}

int cb60_linear(const uint8_t *packed,size_t packed_len,
                const float *scales,const float *books,const float *input,
                int64_t batch,int64_t outrows,int64_t ncols,
                int64_t tile_rows,float *output){
    if(!packed||!scales||!books||!input||!output||
       batch<=0||outrows<=0||ncols<=0||ncols%8||tile_rows<=0)
       return -1;
    int64_t group=ncols/8;
    if((uint64_t)packed_len*8ULL < (uint64_t)(outrows*group)*5ULL)
        return -2;
    if(group>INT32_MAX/32)return -3;
    float *lut=(float*)malloc((size_t)group*32*sizeof(float));
    if(!lut)return -4;
    for(int64_t b=0;b<batch;b++){
        const float *x=input+b*ncols;
        for(int64_t first=0;first<outrows;first+=tile_rows){
            int64_t last=first+tile_rows;
            if(last>outrows)last=outrows;
            const float *book=books+(first/tile_rows)*256;
            for(int64_t g=0;g<group;g++){
                const float *v=x+g*8;
                float *line=lut+g*32;
                for(int k=0;k<32;k++){
                    float acc=0.0f;
                    const float *center=book+k*8;
                    for(int j=0;j<8;j++)acc+=v[j]*center[j];
                    line[k]=acc;
                }
            }
            for(int64_t r=first;r<last;r++){
                float sum=0.0f;
                for(int64_t g=0;g<group;g++){
                    unsigned int code=source5(packed,packed_len,
                                              (uint64_t)r*group+g);
                    sum+=lut[g*32+code];
                }
                output[b*outrows+r]=sum*scales[r];
            }
        }
    }
    free(lut);
    return 0;
}

int cb60_embedding(const uint8_t *packed,size_t packed_len,
                   const float *scales,const float *books,
                   const int64_t *ids,int64_t count,int64_t rows,
                   int64_t ncols,int64_t tile_rows,float *output){
    if(!packed||!scales||!books||!ids||!output||
       count<=0||rows<=0||ncols<=0||ncols%8||tile_rows<=0)return -1;
    int64_t group=ncols/8;
    if((uint64_t)packed_len*8ULL < (uint64_t)(rows*group)*5ULL)return -2;
    for(int64_t p=0;p<count;p++){
        int64_t r=ids[p];
        if(r<0||r>=rows)return -3;
        const float *book=books+(r/tile_rows)*256;
        for(int64_t g=0;g<group;g++){
            const float *center=book+8*source5(packed,packed_len,
                                      (uint64_t)r*group+g);
            float *dst=output+p*ncols+g*8;
            for(int j=0;j<8;j++)dst[j]=center[j]*scales[r];
        }
    }
    return 0;
}
