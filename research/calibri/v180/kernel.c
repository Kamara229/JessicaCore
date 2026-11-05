/* CALIBRI v180: direct codebook-native 5-bit GEMV with inverse rotation.
 * Uses the EXACT same five-bit CB59 C codec, never reconstructs complete
 * trained dense matrices. FWHT is applied only to live FP32 activations
 * for linear, and requested decoded embedding rows for embedding.
 */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "../direct_kernel_v59.c"

static int block_legal(int64_t n, int64_t block){
    return (block==32 || block==64) && n>0 && n%block==0;
}
static float sign_for(int64_t i){
    uint32_t x=(uint32_t)((uint64_t)(i+1)*2654435761ull);
    x^=x>>16;
    return (x&1u) ? -1.f : 1.f;
}
static void fwht(float *v,int64_t block){
    for(int64_t k=1;k<block;k*=2)
      for(int64_t ix=0;ix<block;ix+=2*k)
        for(int64_t j=0;j<k;j++){
          float a=v[ix+j],b=v[ix+k+j];
          v[ix+j]=a+b;
          v[ix+k+j]=a-b;
        }
    const float norm=(block==32)?0.1767766952966369f:0.125f;
    for(int64_t j=0;j<block;j++)v[j]*=norm;
}

int cb180_linear(const uint8_t *packed,size_t packed_len,
                 const float *row_scales,const float *centroids,
                 const float *input,int64_t batch,int64_t outrows,
                 int64_t ncols,int64_t block,float *output){
    if(!packed || !row_scales || !centroids || !input || !output ||
       !block_legal(ncols,block) || batch<=0 || outrows<=0 ||
       (uint64_t)ncols*(uint64_t)batch > SIZE_MAX/sizeof(float))
        return -1;
    size_t count=(size_t)batch*(size_t)ncols;
    float *rotated=(float*)malloc(count*sizeof(float));
    if(!rotated)return -2;
    for(int64_t b=0;b<batch;b++){
      float *x=rotated+b*ncols;
      const float *src=input+b*ncols;
      for(int64_t col=0;col<ncols;col++)x[col]=src[col]*sign_for(col);
      for(int64_t col=0;col<ncols;col+=block)fwht(x+col,block);
    }
    int rc=cb59_linear(packed,packed_len,row_scales,centroids,rotated,
                        batch,outrows,ncols,output);
    free(rotated);
    return rc;
}

int cb180_embedding(const uint8_t *packed,size_t packed_len,
                    const float *row_scales,const float *centroids,
                    const int64_t *ids,int64_t count,
                    int64_t total_rows,int64_t ncols,
                    int64_t block,float *output){
    if(!block_legal(ncols,block) || !output || count<=0)
        return -1;
    int rc=cb59_embedding(packed,packed_len,row_scales,centroids,ids,
                          count,total_rows,ncols,output);
    if(rc)return rc;
    for(int64_t b=0;b<count;b++){
       float *out=output+b*ncols;
       for(int64_t col=0;col<ncols;col+=block)fwht(out+col,block);
       for(int64_t col=0;col<ncols;col++)out[col]*=sign_for(col);
    }
    return 0;
}
