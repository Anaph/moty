/* hw_neon.h — ARM NEON backend (with optional DOTPROD).
 * Implements dot_i8i8 via vdotq_s32 (DOTPROD) or vmull+vmlal (baseline NEON).
 * dot_i4i8 via nibble unpack + vdot/vmull.
 */
#ifndef HW_NEON_H
#define HW_NEON_H

static inline float neon_hsum_f32(float32x4_t v) {
#ifdef __aarch64__
    return vaddvq_f32(v);
#else
    float32x2_t r = vadd_f32(vget_low_f32(v), vget_high_f32(v));
    return vget_lane_f32(vpadd_f32(r, r), 0);
#endif
}
static inline int32_t neon_hsum_s32(int32x4_t v) {
#ifdef __aarch64__
    return vaddvq_s32(v);
#else
    int32x2_t r = vpadd_s32(vget_low_s32(v), vget_high_s32(v));
    return vget_lane_s32(vpadd_s32(r, r), 0);
#endif
}

#ifdef __ARM_FEATURE_DOTPROD
/* ---- DOTPROD path ---- */
int32_t moty_hw_dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    int32_t sum=0; int i=0;
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(; i+64<=n; i+=64){
        a0=vdotq_s32(a0,vld1q_s8(w+i),    vld1q_s8(x+i));
        a1=vdotq_s32(a1,vld1q_s8(w+i+16), vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vld1q_s8(w+i+32), vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vld1q_s8(w+i+48), vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(; i+16<=n; i+=16) acc=vdotq_s32(acc,vld1q_s8(w+i),vld1q_s8(x+i));
    sum=neon_hsum_s32(acc);
    for(; i<n; i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}

int32_t moty_hw_dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum=0; int i=0;
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0),a2=vdupq_n_s32(0),a3=vdupq_n_s32(0);
    for(; i+64<=I; i+=64){
        uint8x16_t byA=vld1q_u8(w4+(i>>1)), byB=vld1q_u8(w4+(i>>1)+16);
        uint8x16x2_t zA=vzipq_u8(vandq_u8(byA,m4q), vshrq_n_u8(byA,4));
        uint8x16x2_t zB=vzipq_u8(vandq_u8(byB,m4q), vshrq_n_u8(byB,4));
        a0=vdotq_s32(a0,vsubq_s8(vreinterpretq_s8_u8(zA.val[0]),b8q),vld1q_s8(x+i));
        a1=vdotq_s32(a1,vsubq_s8(vreinterpretq_s8_u8(zA.val[1]),b8q),vld1q_s8(x+i+16));
        a2=vdotq_s32(a2,vsubq_s8(vreinterpretq_s8_u8(zB.val[0]),b8q),vld1q_s8(x+i+32));
        a3=vdotq_s32(a3,vsubq_s8(vreinterpretq_s8_u8(zB.val[1]),b8q),vld1q_s8(x+i+48));
    }
    int32x4_t acc=vaddq_s32(vaddq_s32(a0,a1),vaddq_s32(a2,a3));
    for(; i+32<=I; i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4));
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q),vld1q_s8(x+i));
        acc=vdotq_s32(acc,vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q),vld1q_s8(x+i+16));
    }
    sum=neon_hsum_s32(acc);
    for(; i+1<I; i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}

#else
/* ---- baseline NEON (no DOTPROD) ---- */
int32_t moty_hw_dot_i8i8(const int8_t *w, const int8_t *x, int n) {
    int32_t sum=0; int i=0;
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0);
    for(; i+64<=n; i+=64){
        int8x16_t wA0=vld1q_s8(w+i),wA1=vld1q_s8(w+i+16),wB0=vld1q_s8(w+i+32),wB1=vld1q_s8(w+i+48);
        int8x16_t xA0=vld1q_s8(x+i),xA1=vld1q_s8(x+i+16),xB0=vld1q_s8(x+i+32),xB1=vld1q_s8(x+i+48);
        int16x8_t p;
        p=vmull_s8(vget_low_s8(wA0),vget_low_s8(xA0)); p=vmlal_s8(p,vget_high_s8(wA0),vget_high_s8(xA0));
        a0=vpadalq_s16(a0,p);
        p=vmull_s8(vget_low_s8(wA1),vget_low_s8(xA1)); p=vmlal_s8(p,vget_high_s8(wA1),vget_high_s8(xA1));
        a0=vpadalq_s16(a0,p);
        p=vmull_s8(vget_low_s8(wB0),vget_low_s8(xB0)); p=vmlal_s8(p,vget_high_s8(wB0),vget_high_s8(xB0));
        a1=vpadalq_s16(a1,p);
        p=vmull_s8(vget_low_s8(wB1),vget_low_s8(xB1)); p=vmlal_s8(p,vget_high_s8(wB1),vget_high_s8(xB1));
        a1=vpadalq_s16(a1,p);
    }
    int32x4_t acc=vaddq_s32(a0,a1);
    for(; i+32<=n; i+=32){
        int8x16_t wA=vld1q_s8(w+i),wB=vld1q_s8(w+i+16);
        int8x16_t xA=vld1q_s8(x+i),xB=vld1q_s8(x+i+16);
        int16x8_t p2;
        p2=vmull_s8(vget_low_s8(wA),vget_low_s8(xA)); p2=vmlal_s8(p2,vget_high_s8(wA),vget_high_s8(xA));
        acc=vpadalq_s16(acc,p2);
        p2=vmull_s8(vget_low_s8(wB),vget_low_s8(xB)); p2=vmlal_s8(p2,vget_high_s8(wB),vget_high_s8(xB));
        acc=vpadalq_s16(acc,p2);
    }
    sum=neon_hsum_s32(acc);
    for(; i<n; i++) sum+=(int32_t)w[i]*x[i];
    return sum;
}

int32_t moty_hw_dot_i4i8(const uint8_t *w4, const int8_t *x, int I) {
    int32_t sum=0; int i=0;
    const uint8x16_t m4q=vdupq_n_u8(0x0F); const int8x16_t b8q=vdupq_n_s8(8);
    int32x4_t a0=vdupq_n_s32(0),a1=vdupq_n_s32(0);
    for(; i+64<=I; i+=64){
        uint8x16_t byA=vld1q_u8(w4+(i>>1)), byB=vld1q_u8(w4+(i>>1)+16);
        uint8x16x2_t zA=vzipq_u8(vandq_u8(byA,m4q), vshrq_n_u8(byA,4));
        uint8x16x2_t zB=vzipq_u8(vandq_u8(byB,m4q), vshrq_n_u8(byB,4));
        int8x16_t wA0=vsubq_s8(vreinterpretq_s8_u8(zA.val[0]),b8q);
        int8x16_t wA1=vsubq_s8(vreinterpretq_s8_u8(zA.val[1]),b8q);
        int8x16_t wB0=vsubq_s8(vreinterpretq_s8_u8(zB.val[0]),b8q);
        int8x16_t wB1=vsubq_s8(vreinterpretq_s8_u8(zB.val[1]),b8q);
        int8x16_t xA0=vld1q_s8(x+i),xA1=vld1q_s8(x+i+16);
        int8x16_t xB0=vld1q_s8(x+i+32),xB1=vld1q_s8(x+i+48);
        int16x8_t p;
        p=vmull_s8(vget_low_s8(wA0),vget_low_s8(xA0)); p=vmlal_s8(p,vget_high_s8(wA0),vget_high_s8(xA0));
        a0=vpadalq_s16(a0,p);
        p=vmull_s8(vget_low_s8(wA1),vget_low_s8(xA1)); p=vmlal_s8(p,vget_high_s8(wA1),vget_high_s8(xA1));
        a0=vpadalq_s16(a0,p);
        p=vmull_s8(vget_low_s8(wB0),vget_low_s8(xB0)); p=vmlal_s8(p,vget_high_s8(wB0),vget_high_s8(xB0));
        a1=vpadalq_s16(a1,p);
        p=vmull_s8(vget_low_s8(wB1),vget_low_s8(xB1)); p=vmlal_s8(p,vget_high_s8(wB1),vget_high_s8(xB1));
        a1=vpadalq_s16(a1,p);
    }
    int32x4_t acc=vaddq_s32(a0,a1);
    for(; i+32<=I; i+=32){
        uint8x16_t by=vld1q_u8(w4+(i>>1));
        uint8x16x2_t z=vzipq_u8(vandq_u8(by,m4q), vshrq_n_u8(by,4));
        int8x16_t w0=vsubq_s8(vreinterpretq_s8_u8(z.val[0]),b8q);
        int8x16_t w1=vsubq_s8(vreinterpretq_s8_u8(z.val[1]),b8q);
        int8x16_t x0=vld1q_s8(x+i), x1=vld1q_s8(x+i+16);
        int16x8_t p2;
        p2=vmull_s8(vget_low_s8(w0),vget_low_s8(x0)); p2=vmlal_s8(p2,vget_high_s8(w0),vget_high_s8(x0));
        acc=vpadalq_s16(acc,p2);
        p2=vmull_s8(vget_low_s8(w1),vget_low_s8(x1)); p2=vmlal_s8(p2,vget_high_s8(w1),vget_high_s8(x1));
        acc=vpadalq_s16(acc,p2);
    }
    sum=neon_hsum_s32(acc);
    for(; i+1<I; i+=2){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]+((int)(b>>4)-8)*x[i+1]; }
    if(i<I){ uint8_t b=w4[i>>1]; sum+=((int)(b&0xF)-8)*x[i]; }
    return sum;
}
#endif /* DOTPROD */

float moty_hw_dot_f32(const float *a, const float *b, int n) {
    float32x4_t acc=vdupq_n_f32(0);
    int i=0;
    for(; i+16<=n; i+=16){
        float32x4_t a0=vld1q_f32(a+i), b0=vld1q_f32(b+i);
        acc=vfmaq_f32(acc,a0,b0);
        float32x4_t a1=vld1q_f32(a+i+4), b1=vld1q_f32(b+i+4);
        acc=vfmaq_f32(acc,a1,b1);
        float32x4_t a2=vld1q_f32(a+i+8), b2=vld1q_f32(b+i+8);
        acc=vfmaq_f32(acc,a2,b2);
        float32x4_t a3=vld1q_f32(a+i+12), b3=vld1q_f32(b+i+12);
        acc=vfmaq_f32(acc,a3,b3);
    }
    for(; i+4<=n; i+=4) acc=vfmaq_f32(acc,vld1q_f32(a+i),vld1q_f32(b+i));
    float s=neon_hsum_f32(acc);
    for(; i<n; i++) s+=a[i]*b[i];
    return s;
}

float moty_hw_dot_f32i8(const float *x, const int8_t *w, int n) {
    float s=0;
    for(int i=0; i<n; i++) s+=x[i]*(float)w[i];
    return s;
}

#endif /* HW_NEON_H */
