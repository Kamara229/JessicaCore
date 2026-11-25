/* FG19: block-aligned 3/4-bit lookup GEMV, research only.
   All CALIBRI body and vocabulary shapes have groups divisible by 8.
   Generic remainder is rejected, not silently dropped. */
#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>

int fg19_gemv(const uint8_t *p, size_t bytes, const float *c,
             const float *s, const float *x, float *y,
             size_t rows, size_t cols, int bits) {
    if (!p || !c || !s || !x || !y || !rows || !cols ||
        cols%64 || (bits!=3 && bits!=4)) return 1;
    size_t g=cols/8, k=(size_t)1<<bits;
    size_t stride=g*(size_t)bits/8;
    if (rows>SIZE_MAX/stride || bytes!=rows*stride ||
        g>SIZE_MAX/k/sizeof(float)) return 2;
    float *lut=malloc(g*k*sizeof(float)); if(!lut)return 3;
    for(size_t i=0;i<g;i++)
        for(size_t j=0;j<k;j++){
            float v=0;
            for(size_t t=0;t<8;t++)v+=x[8*i+t]*c[8*j+t];
            lut[i*k+j]=v;
        }
    /* Avoid OpenMP startup overhead on tiny attention matrices. */
    #pragma omp parallel for schedule(static) num_threads(2) if(rows*g>=100000)
    for(size_t r=0;r<rows;r++){
        const uint8_t *q=p+r*stride;
        float a=0,b=0,d=0,e=0;
        if(bits==3){
            for(size_t i=0;i<g;i+=8,q+=3){
                unsigned v=(unsigned)q[0]|((unsigned)q[1]<<8)|((unsigned)q[2]<<16);
                const float *l=lut+8*i;
                a+=l[v&7]+l[32+((v>>12)&7)];
                b+=l[8+((v>>3)&7)]+l[40+((v>>15)&7)];
                d+=l[16+((v>>6)&7)]+l[48+((v>>18)&7)];
                e+=l[24+((v>>9)&7)]+l[56+((v>>21)&7)];
            }
        }else{
            for(size_t i=0;i<g;i+=8,q+=4){
                const float *l=lut+16*i;
                a+=l[q[0]&15]+l[64+(q[2]&15)];
                b+=l[16+(q[0]>>4)]+l[80+(q[2]>>4)];
                d+=l[32+(q[1]&15)]+l[96+(q[3]&15)];
                e+=l[48+(q[1]>>4)]+l[112+(q[3]>>4)];
            }
        }
        y[r]=((a+b)+(d+e))*s[r];
    }
    free(lut);return 0;
}
