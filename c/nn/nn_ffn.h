#ifndef NN_FFN_H
#define NN_FFN_H
static void dense_ffn(Model *m, Layer *l, float *x, int S, float *out) {
    int I = m->c.inter;
    scr_reset(&m->base.scr);
    scr_reserve(&m->base.scr, 2*scr_al((int64_t)S*I*4));
    float *gb = scr_take(&m->base.scr, (int64_t)S*I*4), *ub = scr_take(&m->base.scr, (int64_t)S*I*4);
    mat_apply(gb, x, &l->gate, S); mat_apply(ub, x, &l->up, S);
    for (int64_t i = 0; i < (int64_t)S*I; i++) { float v=gb[i]; gb[i]=(v/(1.f+expf(-v)))*ub[i]; }
    mat_apply(out, gb, &l->down, S);
}
#endif
