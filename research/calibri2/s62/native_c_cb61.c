/* CALIBRI S62 independent native C decoder for S61 CB61ZL01.
 *
 * Exactly reverses the physically paid zlib CB59 envelope on CPU.
 * No model-weight changes, no QA/test prompts, no GPU.
 * Does NOT implement a C neural inference engine; throughput of native-C
 * generation remains unmeasured and must not be claimed.
 *
 * Build: cc -O2 -std=c11 -Wall -Wextra -Werror native_c_cb61.c -o cb61check -lz -lcrypto
 * Usage: cb61check S45.cb59 S61.s61 S56.safetensors S61_manifest.json report.json
 */
#define _POSIX_C_SOURCE 200809L
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <time.h>
#include <zlib.h>
#include <openssl/sha.h>

static const char S45_SHA[]="12a725b07d37a0dae870d7b47666488265a1fa71a58bebf10da1fc7f25a7c398";
static const char S56_SHA[]="503926839336ed7161786fa6d8ccb2a4ae6fee876cb566d0e52db80ff128f6d4";
static const size_t ORIGINAL_SIZE=3141513;
static const size_t S56_SIZE=39040;
static const double PARAMS=36380480.0;

static void die(const char *msg) {fprintf(stderr,"S62_CHECK_FAILED: %s\n",msg);exit(2);}
static uint32_t le32(const unsigned char *p) {
    return (uint32_t)p[0]|((uint32_t)p[1]<<8)|((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
}
static uint64_t le64(const unsigned char *p) {
    return (uint64_t)le32(p)|((uint64_t)le32(p+4)<<32);
}
static double seconds(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC,&t)!=0)die("monotonic clock unavailable");
    return (double)t.tv_sec+(double)t.tv_nsec/1e9;
}
static int doubles_cmp(const void *a,const void *b){
    const double x=*(const double*)a,y=*(const double*)b;
    return (x>y)-(x<y);
}
static unsigned char *read_file(const char *path,size_t *n) {
    FILE *f=fopen(path,"rb");
    if(!f)die("cannot open input file");
    if(fseek(f,0,SEEK_END)!=0)die("cannot seek input file");
    long size=ftell(f);
    if(size<0||size>12000000)die("input file too large or invalid");
    if(fseek(f,0,SEEK_SET)!=0)die("cannot reset input file");
    *n=(size_t)size;
    unsigned char *mem=malloc(*n?*n:1);
    if(!mem)die("malloc input failed");
    if(fread(mem,1,*n,f)!=*n)die("short file read");
    if(fclose(f)!=0)die("file close failed");
    return mem;
}
static void hex_sha(const unsigned char *data,size_t n,char out[65]){
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if(!SHA256(data,n,digest))die("SHA256 failed");
    static const char hex[]="0123456789abcdef";
    for(size_t i=0;i<SHA256_DIGEST_LENGTH;i++){
       out[i*2]=hex[digest[i]>>4];
       out[i*2+1]=hex[digest[i]&15];
    }
    out[64]=0;
}
static void verify_legacy_cb59(const unsigned char *p,size_t n){
    if(n!=ORIGINAL_SIZE || n<13+36)die("invalid legacy CB59 original byte length");
    if(memcmp(p,"CB59",4)!=0 || p[4]!=1)die("invalid CB59 magic/version");
    if((uint64_t)13+le32(p+5)+le32(p+9)+36!=n)die("CB59 metadata/body/CRC length disagreement");
    if(crc32(0,p,n-36)!=le32(p+n-36))die("CB59 footer CRC mismatch");
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if(!SHA256(p,n-32,digest) || memcmp(digest,p+n-32,32)!=0)die("CB59 footer SHA mismatch");
    char h[65];hex_sha(p,n,h);
    if(strcmp(h,S45_SHA)!=0)die("CB59 original S45 immutable SHA changed");
}
static void decode(const unsigned char *zip,size_t n,unsigned char *out,size_t capacity){
    if(n<52+1||memcmp(zip,"CB61ZL01",8)!=0)die("CB61 header/magic invalid");
    if(le64(zip+8)!=ORIGINAL_SIZE || capacity!=ORIGINAL_SIZE)die("CB61 original length invalid");
    if(n-52>UINT32_MAX)die("CB61 compressed stream too long");
    uLongf output_len=(uLongf)capacity;
    int code=uncompress(out,&output_len,zip+52,(uLong)(n-52));
    if(code!=Z_OK || output_len!=capacity)die("CB61 native C zlib inflate failed");
    if(crc32(0,out,capacity)!=le32(zip+16))die("CB61 zlib header CRC mismatch");
    unsigned char digest[SHA256_DIGEST_LENGTH];
    if(!SHA256(out,capacity,digest)||memcmp(digest,zip+20,32)!=0)
        die("CB61 zlib header SHA mismatch");
    verify_legacy_cb59(out,capacity);
}
int main(int argc,char **argv){
    if(argc!=6)die("expected original, envelope, tunnel, manifest, output-report arguments");
    size_t raw_n,zip_n,side_n,manifest_n;
    unsigned char *raw=read_file(argv[1],&raw_n);
    unsigned char *zip=read_file(argv[2],&zip_n);
    unsigned char *side=read_file(argv[3],&side_n);
    unsigned char *manifest=read_file(argv[4],&manifest_n);
    verify_legacy_cb59(raw,raw_n);
    char tunnel_sha[65];hex_sha(side,side_n,tunnel_sha);
    if(side_n!=S56_SIZE || strcmp(tunnel_sha,S56_SHA))die("S56 paid Green Tunnel changed");
    unsigned char *restored=malloc(raw_n);
    if(!restored)die("malloc restore failed");
    decode(zip,zip_n,restored,raw_n);
    if(memcmp(restored,raw,raw_n)!=0)die("S61 C bytes differ from S45 original");
    /* Real CPU clock measurements, warm in-process, no neural inference. */
    double times[51];
    for(int k=0;k<51;k++){
        double t0=seconds();
        decode(zip,zip_n,restored,raw_n);
        times[k]=(seconds()-t0)*1000.0;
    }
    qsort(times,51,sizeof(double),doubles_cmp);
    size_t paid=zip_n+side_n+manifest_n;
    if((double)paid*8.0/PARAMS>0.7)die("physical all-inclusive paid bpp >0.7");
    FILE *f=fopen(argv[5],"wb");
    if(!f)die("cannot create report");
    fprintf(f,"{\n");
    fprintf(f,"  \"stage\": \"CALIBRI_S62_NATIVE_C_S61_LOSSLESS_STARTUP\",\n");
    fprintf(f,"  \"actual_C_compiled_decompress_and_SHA_verified\": true,\n");
    fprintf(f,"  \"C_decoder_bytewise_identical_to_original_CB59\": true,\n");
    fprintf(f,"  \"CB59_original_immutable_sha256\": \"%s\",\n",S45_SHA);
    fprintf(f,"  \"S56_unchanged_controller_sha256\": \"%s\",\n",S56_SHA);
    fprintf(f,"  \"S61_physical_stored_source_bytes\": %zu,\n",zip_n);
    fprintf(f,"  \"S56_paid_tunnel_bytes\": %zu,\n",side_n);
    fprintf(f,"  \"S61_physical_manifest_bytes\": %zu,\n",manifest_n);
    fprintf(f,"  \"S61_all_inclusive_bytes\": %zu,\n",paid);
    fprintf(f,"  \"S61_all_inclusive_bpp\": %.12f,\n",paid*8.0/PARAMS);
    fprintf(f,"  \"native_C_startup_decode_median_ms\": %.6f,\n",times[25]);
    fprintf(f,"  \"native_C_startup_decode_p95_ms\": %.6f,\n",times[48]);
    fprintf(f,"  \"native_C_steady_state_extra_inference_decode_per_token\": 0,\n");
    fprintf(f,"  \"native_C_neural_generation_or_FP32_quality_measured\": false,\n");
    fprintf(f,"  \"GPU_used\": false,\n");
    fprintf(f,"  \"Jessica_chat_test_run\": false,\n");
    fprintf(f,"  \"main_Render_Supabase_APK_modified\": false\n}\n");
    if(fclose(f)!=0)die("report write failed");
    printf("S62_NATIVE_C_SUCCESS actual_restored_S45_SHA=%s bpp=%.12f median_startup_ms=%.4f p95_ms=%.4f\\n",
           S45_SHA,paid*8.0/PARAMS,times[25],times[48]);
    free(raw);free(zip);free(side);free(manifest);free(restored);
    return 0;
}
