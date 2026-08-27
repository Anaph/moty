/* hw_sve2.h — ARM SVE2 backend.
 * Uses svdot_s32 (scalable vector dot product) for int8×int8 and int4×int8.
 */
#ifndef HW_SVE2_H
#define HW_SVE2_H
/* <arm_sve.h> is already included by hw.h before this backend */

int32_t moty_hw_dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    svint32_t a0=svdup_n_s32(0), a1=svdup_n_s32(0);
    int B=(int)svcntb(), i=0;
    for(; i+2*B<=n; i+=2*B){
        svbool_t pg=svptrue_b8();
        a0=svdot_s32(a0, svld1_s8(pg,w+i),   svld1_s8(pg,x+i));
        a1=svdot_s32(a1, svld1_s8(pg,w+i+B), svld1_s8(pg,x+i+B));
    }
    if(i+B<=n){ svbool_t pg=svptrue_b8();
        a0=svdot_s32(a0, svld1_s8(pg,w+i), svld1_s8(pg,x+i)); i+=B; }
    int32_t sum=svaddv_s32(svptrue_b32(), svadd_s32_x(svptrue_b32(), a0, a1));
    for(; i<n; i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}

int32_t moty_hw_dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    svint32_t acc=svdup_n_s32(0);
    int B=(int)svcntb(), i=0;
    for(; i+2*B<=I; i+=2*B){
        svbool_t pg=svptrue_b8();
        svuint8_t pk=svld1_u8(pg, w4+(i>>1));
        svuint8_t lo=svand_u8_x(pg, pk, svdup_n_u8(0x0F));
        svuint8_t hi=svand_u8_x(pg, svlsr_n_u8_x(pg, pk, 4), svdup_n_u8(0x0F));
        svint8_t w0=svsub_n_s8_x(pg, svreinterpret_s8_u8(svzip1_u8(lo,hi)), 8);
        svint8_t w1=svsub_n_s8_x(pg, svreinterpret_s8_u8(svzip2_u8(lo,hi)), 8);
        acc=svdot_s32(acc, w0, svld1_s8(pg, x+i));
        acc=svdot_s32(acc, w1, svld1_s8(pg, x+i+B));
    }
    int32_t sum=svaddv_s32(svptrue_b32(), acc);
    for(; i+1<I; i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}

float moty_hw_dot_f32(const float *a, const float *b, int n) {
    svbool_t pg4=svptrue_b32();
    svfloat32_t acc=svdup_n_f32(0);
    int i=0, W=(int)svcntw();
    for(; i+4*W<=n; i+=4*W){
        acc=svmla_f32_x(pg4, acc, svld1_f32(pg4,a+i),     svld1_f32(pg4,b+i));
        acc=svmla_f32_x(pg4, acc, svld1_f32(pg4,a+i+W),   svld1_f32(pg4,b+i+W));
        acc=svmla_f32_x(pg4, acc, svld1_f32(pg4,a+i+2*W), svld1_f32(pg4,b+i+2*W));
        acc=svmla_f32_x(pg4, acc, svld1_f32(pg4,a+i+3*W), svld1_f32(pg4,b+i+3*W));
    }
    for(; i+W<=n; i+=W) acc=svmla_f32_x(pg4, acc, svld1_f32(pg4,a+i), svld1_f32(pg4,b+i));
    float s=svaddv_f32(pg4, acc);
    for(; i<n; i++) s+=a[i]*b[i];
    return s;
}

float moty_hw_dot_f32i8(const float *x, const int8_t *w, int n) {
    float s=0;
    for(int i=0; i<n; i++) s+=x[i]*(float)w[i];
    return s;
}

#endif /* HW_SVE2_H */
