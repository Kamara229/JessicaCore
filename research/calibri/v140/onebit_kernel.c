/*
 * CALIBRI v140 CPU pilot: one extra residual bit per K32/V8 group.
 * The original CB59 source is still served by its own direct LUT-GEMV kernel.
 * This unit adds only a paid low-bit *residual* without dense trained source
 * matrices. Both embedding decode and tied lm_head GEMV share the SAME bitplane.
 */
#include <stdint.h>
#include <stddef.h>

static inline uint8_t five_bit(const uint8_t *a,size_t position) {
    size_t bit=position*5, byte=bit>>3;
    unsigned int val=(unsigned int)a[byte]>>((unsigned int)bit&7);
    // Code buffer always stores whole groups in the real CB59 source.
    if((bit&7)>3) val|=(unsigned int)a[byte+1]<<(8-((unsigned int)bit&7));
    return (uint8_t)(val&31u);
}

static inline uint8_t one_bit(const uint8_t *a,size_t position) {
    return (a[position>>3]>>(position&7))&1u;
}

/* out: [num_rows, hidden]. The supplied output array is pre-zeroed. */
int calibri140_embedding_delta(const uint8_t *base,size_t base_len,
    const uint8_t *plane,size_t plane_len,const float *centers,
    const float *row_scales,const int64_t *indices,int64_t num_indices,
    int64_t rows,int64_t dim,float *out) {
    if(!base||!plane||!centers||!row_scales||!indices||!out||
       num_indices<1||rows<1||dim<=0||dim%8) return 1;
    const size_t groups_per_row=(size_t)dim/8;
    const size_t groups=(size_t)rows*groups_per_row;
    if(base_len!=(groups*5+7)/8 || plane_len!=(groups+7)/8) return 2;
    for(int64_t n=0;n<num_indices;n++) {
        int64_t row=indices[n];
        if(row<0||row>=rows) return 3;
        const float scale=row_scales[row];
        for(size_t g=0;g<groups_per_row;g++) {
            size_t position=(size_t)row*groups_per_row+g;
            uint8_t code=five_bit(base,position), bit=one_bit(plane,position);
            const float *r=centers+((size_t)(code*2+bit)*8);
            float *target=out+(size_t)n*(size_t)dim+g*8;
            for(size_t j=0;j<8;j++) target[j]=scale*r[j];
        }
    }
    return 0;
}

/* out: [num_vectors, rows]; the original CB59 projection is added in Python. */
int calibri140_output_delta(const uint8_t *base,size_t base_len,
    const uint8_t *plane,size_t plane_len,const float *centers,
    const float *row_scales,const float *inputs,int64_t num_vectors,
    int64_t rows,int64_t dim,float *out) {
    if(!base||!plane||!centers||!row_scales||!inputs||!out||
       num_vectors<1||rows<1||dim<=0||dim%8) return 1;
    const size_t groups_per_row=(size_t)dim/8;
    const size_t groups=(size_t)rows*groups_per_row;
    if(base_len!=(groups*5+7)/8 || plane_len!=(groups+7)/8) return 2;
    for(int64_t v=0;v<num_vectors;v++) {
        const float *x=inputs+(size_t)v*(size_t)dim;
        float *y=out+(size_t)v*(size_t)rows;
        for(int64_t row=0;row<rows;row++) {
            float result=0.0f;
            size_t offset=(size_t)row*groups_per_row;
            for(size_t g=0;g<groups_per_row;g++) {
                size_t pos=offset+g;
                uint8_t code=five_bit(base,pos),bit=one_bit(plane,pos);
                const float *r=centers+((size_t)(code*2+bit)*8);
                const float *z=x+g*8;
                float dot=0.0f;
                for(size_t j=0;j<8;j++) dot+=z[j]*r[j];
                result+=dot;
            }
            y[row]=result*row_scales[row];
        }
    }
    return 0;
}
