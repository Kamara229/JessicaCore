/* CALIBRI v184: fully paid joint rank-two correction, direct packed C.
 * All indices/codebooks are decoded by the established original native
 * CB59 kernel; these functions ADD shared rank-2 source correction.
 * NO original dense trained model matrices are loaded at inference.
 */
#include <stdint.h>
#include <stddef.h>
#include "../direct_kernel_v59.c"

int cb184_linear(const uint8_t *packed,size_t packed_len,
                 const float *row_scales,const float *centroids,
                 const float *input,int64_t batch,int64_t outrows,
                 int64_t ncols,float *output,
                 const int8_t *a,float a_scale,const float *shared_b,
                 int64_t rank){
    if(!a||!shared_b||rank!=2||a_scale<=0.0f)return -18;
    int rc=cb59_linear(packed,packed_len,row_scales,centroids,input,
                       batch,outrows,ncols,output);
    if(rc)return rc;
    for(int64_t b=0;b<batch;b++){
        const float *x=input+b*ncols;
        float latent[2]={0.f,0.f};
        for(int64_t k=0;k<rank;k++){
            const float *vec=shared_b+k*ncols;
            float dot=0.f;
            for(int64_t c=0;c<ncols;c++)dot+=vec[c]*x[c];
            latent[k]=dot*a_scale;
        }
        float *y=output+b*outrows;
        for(int64_t row=0;row<outrows;row++){
            const int8_t *v=a+row*rank;
            y[row]+=((float)v[0])*latent[0]+((float)v[1])*latent[1];
        }
    }
    return 0;
}

int cb184_embedding(const uint8_t *packed,size_t packed_len,
                    const float *row_scales,const float *centroids,
                    const int64_t *ids,int64_t count,int64_t total_rows,
                    int64_t ncols,float *output,
                    const int8_t *a,float a_scale,
                    const float *shared_b,int64_t rank){
    if(!a||!shared_b||rank!=2||a_scale<=0.0f)return -18;
    int rc=cb59_embedding(packed,packed_len,row_scales,centroids,ids,
                          count,total_rows,ncols,output);
    if(rc)return rc;
    for(int64_t i=0;i<count;i++){
        int64_t row=ids[i];
        if(row<0||row>=total_rows)return -19;
        const int8_t *v=a+row*rank;
        float *y=output+i*ncols;
        for(int64_t col=0;col<ncols;col++)
            y[col]+=a_scale*(((float)v[0])*shared_b[col]+
                             ((float)v[1])*shared_b[ncols+col]);
    }
    return 0;
}
