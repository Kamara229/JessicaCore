/* Direct packed codebook linear, inference-only reference CPU kernel. */
#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>
int cb_linear(const float *x,const float *c,const float *s,const uint8_t *p,
              float *y,int n,int o,int g,int k,int bits) {
    if(n<1 || o<1 || g<1 || k!=(1<<bits) || bits<3 || bits>5) return -1;
    float *table=malloc((size_t)g*k*sizeof(float));
    if(!table)return -2;
    unsigned mask=(1u<<bits)-1u;
    for(int t=0;t<n;t++){
        for(int j=0;j<g;j++){
            const float *v=x+((size_t)t*g+j)*8;
            for(int z=0;z<k;z++){
                float a=0;
                for(int d=0;d<8;d++)a+=v[d]*c[z*8+d];
                table[j*k+z]=a;
            }
        }
        for(int r=0;r<o;r++){
            float a=0;
            for(int j=0;j<g;j++){
                size_t b=((size_t)r*g+j)*bits;
                size_t q=b>>3;unsigned shift=b&7;
                unsigned code=((unsigned)p[q]|((unsigned)p[q+1]<<8))>>shift;
                a+=table[j*k+(code&mask)];
            }
            y[(size_t)t*o+r]=a*s[r];
        }
    }
    free(table);return 0;
}
