/* Motore di inferenza OLMoE in C puro, con EXPERT-STREAMING dal disco.
 * Nato come porting del motore Python (engine.py) per validare il core MoE
 * (REF: stessi token id del riferimento) prima di scalare a GLM-5.2; ora e' il
 * RIFERIMENTO MoE pulito: cache+topk riusabili via moe.h, interfaccia ad env
 * (SNAP/PROMPT/REF/...), chat interattiva, sampling. glm.c resta il motore MoE
 * di produzione (cache ricca io_uring/pin/CUDA); questo e' lo starting point
 * per un nuovo engine MoE.
 *
 * Densa (embed, attn, router, norme, lm_head) residente in RAM (float32).
 * Expert letti dal disco on-demand via pread+fadvise(DONTNEED), cache LRU
 * per-layer (moe.h). Matmul multi-thread OpenMP (niente BLAS).
 *
 * Uso (variabili d'ambiente, come qwen/gemma):
 *   SNAP=<dir snapshot HF>            config.json + tokenizer.json + *.safetensors
 *   EXPERT_CACHE=16                   expert residenti per layer (LRU)
 *   EBITS=8                           bit di quantizzazione expert (2..8; storage int8)
 *   REF=ref.json                      validazione greedy vs prompt_ids/full_ids
 *   PROMPT="..."                      one-shot; senza PROMPT ne' REF -> chat su stdin
 *   NGEN=256 CTX=4096 TEMP=0.7 NUCLEUS=0.95 SEED=n TOKENS=1
 *   CHAT_TEMPLATE=0                   0=prompt grezzo (OLMoE non ha template standard)
 */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include "st.h"
#include "nn.h"
#include "moe.h"
#include "tok.h"

#define ENGINE_TAG "olmoe"

static int g_tokens_dump = 0;   /* TOKENS=1 (runtime.h non e' incluso qui) */

/* ---------- config ---------- */
typedef struct {
    int hidden, n_layers, n_heads, n_kv_heads, head_dim;
    int n_experts, topk, inter, vocab;
    float theta, eps; int norm_topk;
} Cfg;

/* ---------- pesi densi per-layer ---------- */
typedef struct {
    float *in_ln, *post_ln, *q, *k, *v, *o, *qn, *kn, *gate;
} Layer;

typedef struct {
    Cfg c;
    shards S;
    int quant_bits;        /* bit di quantizzazione degli expert (2..8); storage int8, niente f32 (#134) */
    float *embed, *lm_head, *final_norm;
    Layer *L;
    ExpertCache *cache;    /* [n_layers], LRU per-layer (moe.h) */
    uint64_t clock; long long hits, miss;   /* aggregati per reporting (somma sui layer) */
    /* kv-cache per-layer: K,V come [H * maxT * head_dim] */
    float **K, **V; int kv_len, max_t;
    Scratch scr;             /* P5: arena per-Model (sampler) */
    double dense_load_s;
} Model;

/* ---------- caricamento ---------- */
static void load_cfg(Cfg *c, const char *snap) {
    char path[2048]; snprintf(path, sizeof(path), "%s/config.json", snap);
    FILE *f = fopen(path, "rb"); if(!f){perror(path);exit(1);}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf = malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    jval *r = json_parse(buf);
    c->hidden    = (int)json_get(r,"hidden_size")->num;
    c->n_layers  = (int)json_get(r,"num_hidden_layers")->num;
    c->n_heads   = (int)json_get(r,"num_attention_heads")->num;
    c->n_kv_heads= (int)json_get(r,"num_key_value_heads")->num;
    c->n_experts = (int)json_get(r,"num_experts")->num;
    c->topk      = (int)json_get(r,"num_experts_per_tok")->num;
    c->inter     = (int)json_get(r,"intermediate_size")->num;
    c->vocab     = (int)json_get(r,"vocab_size")->num;
    c->head_dim  = c->hidden / c->n_heads;
    jval *th = json_get(r,"rope_theta");  c->theta = th ? (float)th->num : 10000.f;
    jval *ep = json_get(r,"rms_norm_eps"); c->eps   = ep ? (float)ep->num : 1e-5f;
    jval *nt = json_get(r,"norm_topk_prob"); c->norm_topk = (nt && nt->t==J_BOOL) ? nt->boolean : 0;
    if (c->topk > 64) { fprintf(stderr,"[olmoe] topk=%d > 64 (hard cap del buffer di moe_topk)\n", c->topk); exit(1); }
    json_free(r); free(buf);
}

static float *load_t(Model *m, const char *name) {
    int64_t n = st_numel(&m->S, name);
    if (n < 0) { fprintf(stderr, "missing %s\n", name); exit(1); }
    float *p = falloc(n);
    st_read_f32(&m->S, name, p, 0);   /* densa: niente DONTNEED, resta residente */
    return p;
}

static void model_init(Model *m, const char *snap, int cap, int bits) {
    memset(m, 0, sizeof(*m));
    m->quant_bits = bits;
    load_cfg(&m->c, snap);
    st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    m->embed      = load_t(m, "model.embed_tokens.weight");
    m->lm_head    = load_t(m, "lm_head.weight");
    m->final_norm = load_t(m, "model.norm.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    char nm[256];
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        #define LD(field, suffix) snprintf(nm,sizeof(nm),"model.layers.%d." suffix,i); l->field = load_t(m,nm)
        LD(in_ln,  "input_layernorm.weight");
        LD(post_ln,"post_attention_layernorm.weight");
        LD(q, "self_attn.q_proj.weight"); LD(k, "self_attn.k_proj.weight");
        LD(v, "self_attn.v_proj.weight"); LD(o, "self_attn.o_proj.weight");
        LD(qn,"self_attn.q_norm.weight"); LD(kn,"self_attn.k_norm.weight");
        LD(gate, "mlp.gate.weight");
        #undef LD
    }
    m->cache = calloc(c->n_layers, sizeof(ExpertCache));
    for (int i = 0; i < c->n_layers; i++) expert_cache_init(&m->cache[i], cap, c->n_experts);
    m->dense_load_s = now_s() - t0;
}

/* legge un weight expert dal disco e lo quantizza in q[O,I]+scale[O].
 * Container pre-quantizzato (convert_olmoe.py: int8 + scale f32 in "name.qs"):
 * lettura raw diretta — meta' I/O e zero quantize_rows a runtime. */
static void load_expert_w(Model *m, const char *name, int8_t *q, float *scale, int O, int I, float *tmp) {
    st_tensor *t = st_find(&m->S, name);
    if (t && t->dtype == 3) {                    /* I8/U8: container moty */
        char qs[300]; snprintf(qs, sizeof(qs), "%s.qs", name);
        st_read_raw(&m->S, name, q, 1);
        st_read_f32(&m->S, qs, scale, 1);
        return;
    }
    st_read_f32(&m->S, name, tmp, 1);            /* pread + fadvise DONTNEED */
    quantize_rows(tmp, q, scale, O, I, m->quant_bits);
}

/* hook di moe.h: carica g/u/d di UN expert (selezionato da expert_get) dal disco. */
static void olmoe_load_expert(void *ctx, int layer, int eid, ExpertSlot *s, int inter, int hidden) {
    Model *m = (Model *)ctx;
    int64_t ng = (int64_t)inter*hidden, nd = (int64_t)hidden*inter;
    float *tmp = falloc(ng > nd ? ng : nd);
    char nm[256];
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.gate_proj.weight",layer,eid); load_expert_w(m,nm,s->g,s->gs,inter,hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.up_proj.weight",  layer,eid); load_expert_w(m,nm,s->u,s->us,inter,hidden,tmp);
    snprintf(nm,sizeof(nm),"model.layers.%d.mlp.experts.%d.down_proj.weight",layer,eid); load_expert_w(m,nm,s->d,s->ds,hidden,inter,tmp);
    free(tmp);
}

/* ---------- RoPE su un vettore di una testa (head_dim) a posizione assoluta pos ---------- */
static void rope_head(float *x, int pos, const Cfg *c) {
    int h = c->head_dim / 2;
    for (int j = 0; j < h; j++) {
        float inv = powf(c->theta, -2.0f * j / c->head_dim);
        float ang = pos * inv, cs = cosf(ang), sn = sinf(ang);
        float a = x[j], b = x[j+h];
        x[j]   = a*cs - b*sn;
        x[j+h] = b*cs + a*sn;
    }
}

/* attenzione sui token nuovi x[S,hidden]; pos_base = posizione assoluta del primo token nuovo */
static void attention(Model *m, Layer *l, int layer, float *x, int S, int pos_base, float *out) {
    Cfg *c = &m->c; int H = c->n_heads, hd = c->head_dim, D = c->hidden;
    float *q = falloc((int64_t)S*D), *k = falloc((int64_t)S*D), *vv = falloc((int64_t)S*D);
    matmul(q, x, l->q, S, D, D);
    matmul(k, x, l->k, S, D, D);
    matmul(vv, x, l->v, S, D, D);
    /* qk-norm sull'intero vettore hidden, poi RoPE per testa */
    for (int s = 0; s < S; s++) {
        rmsnorm_row(q + (int64_t)s*D, q + (int64_t)s*D, l->qn, D, c->eps);
        rmsnorm_row(k + (int64_t)s*D, k + (int64_t)s*D, l->kn, D, c->eps);
        int pos = pos_base + s;
        for (int hh = 0; hh < H; hh++) { rope_head(q + (int64_t)s*D + hh*hd, pos, c); rope_head(k + (int64_t)s*D + hh*hd, pos, c); }
    }
    /* scrive k,v nella kv-cache alle posizioni pos_base..pos_base+S-1 */
    for (int s = 0; s < S; s++) for (int hh = 0; hh < H; hh++) {
        int t = pos_base + s;
        memcpy(m->K[layer] + ((int64_t)hh*m->max_t + t)*hd, k + (int64_t)s*D + hh*hd, hd*sizeof(float));
        memcpy(m->V[layer] + ((int64_t)hh*m->max_t + t)*hd, vv + (int64_t)s*D + hh*hd, hd*sizeof(float));
    }
    int Tk = pos_base + S;             /* numero di key totali disponibili */
    float scale = 1.f / sqrtf((float)hd);
    float *ctx = falloc((int64_t)S*D);
    #pragma omp parallel
    {
        float *sc = falloc(Tk);        /* punteggi per thread, dimensionati sulle key reali */
        #pragma omp for collapse(2) schedule(static)
        for (int hh = 0; hh < H; hh++) {
            for (int s = 0; s < S; s++) {
                int qpos = pos_base + s;
                const float *qv = q + (int64_t)s*D + hh*hd;
                for (int t = 0; t <= qpos; t++) {          /* causale: t <= qpos */
                    const float *kv = m->K[layer] + ((int64_t)hh*m->max_t + t)*hd;
                    float acc = 0; for (int dd = 0; dd < hd; dd++) acc += qv[dd]*kv[dd];
                    sc[t] = acc * scale;
                }
                softmax_row(sc, qpos+1);
                float *cx = ctx + (int64_t)s*D + hh*hd;
                for (int dd = 0; dd < hd; dd++) cx[dd] = 0;
                for (int t = 0; t <= qpos; t++) {
                    const float *vrow = m->V[layer] + ((int64_t)hh*m->max_t + t)*hd;
                    float a = sc[t];
                    for (int dd = 0; dd < hd; dd++) cx[dd] += a * vrow[dd];
                }
            }
        }
        free(sc);
    }
    matmul(out, ctx, l->o, S, D, D);
    free(q); free(k); free(vv); free(ctx);
}

/* MoE sui token x[S,hidden] -> out[S,hidden]. Router softmax + top-K greedy
 * (moe.h, bit-identico al naive) + expert_get (cache LRU) + matmul_q per expert. */
static void moe(Model *m, Layer *l, int layer, float *x, int S, float *out) {
    Cfg *c = &m->c; int D = c->hidden, E = c->n_experts, K = c->topk, I = c->inter;
    float *logits = falloc((int64_t)S*E);
    matmul(logits, x, l->gate, S, D, E);
    memset(out, 0, (int64_t)S*D*sizeof(float));
    float *g = falloc(I), *u = falloc(I), *hh = falloc(D);
    int idx[64]; float w[64];
    for (int s = 0; s < S; s++) {
        float *pr = logits + (int64_t)s*E;
        softmax_row(pr, E);
        moe_topk(pr, E, K, idx, w, c->norm_topk);
        const float *xs = x + (int64_t)s*D;
        for (int kk = 0; kk < K; kk++) {
            ExpertSlot *e = expert_get(&m->cache[layer], m, layer, idx[kk],
                                       olmoe_load_expert, I, D);
            matmul_q(g, xs, e->g, e->gs, D, I);     /* gate_proj [I,D] */
            matmul_q(u, xs, e->u, e->us, D, I);     /* up_proj   [I,D] */
            for (int i = 0; i < I; i++) { float gv = g[i]; g[i] = (gv / (1.f + expf(-gv))) * u[i]; }
            matmul_q(hh, g, e->d, e->ds, I, D);      /* down_proj [D,I] */
            float weight = w[kk];
            float *os = out + (int64_t)s*D;
            for (int d = 0; d < D; d++) os[d] += weight * hh[d];
        }
    }
    free(logits); free(g); free(u); free(hh);
}

/* un passo: token nuovi ids[S] a posizione pos_base. Ritorna logits dell'ultimo token (malloc'd). */
static float *step(Model *m, const int *ids, int S, int pos_base) {
    Cfg *c = &m->c; int D = c->hidden;
    float *x = falloc((int64_t)S*D);
    for (int s = 0; s < S; s++) memcpy(x + (int64_t)s*D, m->embed + (int64_t)ids[s]*D, D*sizeof(float));
    float *nrm = falloc((int64_t)S*D), *tmp = falloc((int64_t)S*D);
    for (int i = 0; i < c->n_layers; i++) {
        Layer *l = &m->L[i];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->in_ln, D, c->eps);
        attention(m, l, i, nrm, S, pos_base, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
        for (int s = 0; s < S; s++) rmsnorm_row(nrm + (int64_t)s*D, x + (int64_t)s*D, l->post_ln, D, c->eps);
        moe(m, l, i, nrm, S, tmp);
        for (int64_t j = 0; j < (int64_t)S*D; j++) x[j] += tmp[j];
    }
    m->kv_len = pos_base + S;
    /* solo l'ultimo token -> logits */
    float *last = falloc(D);
    rmsnorm_row(last, x + (int64_t)(S-1)*D, m->final_norm, D, c->eps);
    float *logit = falloc(c->vocab);
    matmul(logit, last, m->lm_head, 1, D, c->vocab);
    free(x); free(nrm); free(tmp); free(last);
    return logit;
}

static void kv_alloc(Model *m, int max_t) {
    Cfg *c = &m->c;
    m->max_t = max_t; m->kv_len = 0;
    m->K = calloc(c->n_layers, sizeof(float*)); m->V = calloc(c->n_layers, sizeof(float*));
    for (int i = 0; i < c->n_layers; i++) {
        m->K[i] = falloc((int64_t)c->n_heads * max_t * c->head_dim);
        m->V[i] = falloc((int64_t)c->n_heads * max_t * c->head_dim);
    }
}

/* genera n_new token dopo il prompt (greedy se g_temp<=0, sampling altrimenti).
 * pick_tok di nn.h con g_temp=0 e' argmax puro -> bit-identico al REF originale.
 * echo!=0 + T: decodifica e stampa i token generati (chat/one-shot).
 * Ritorna il numero di token generati (ng); *dt_out raccoglie il tempo (NULL ok). */
static int generate(Model *m, Tok *T, const int *prompt, int np, int n_new,
                    int *out, int echo, double *dt_out) {
    Cfg *c = &m->c;
    for (int i = 0; i < np; i++) out[i] = prompt[i];
    double t0 = now_s();
    float *logit = step(m, prompt, np, 0);          /* PREFILL */
    int len = np, ng = 0;
    for (int s = 0; s < n_new; s++) {
        int t = pick_tok(&m->scr, logit, c->vocab); free(logit);
        out[len++] = t; ng++;
        if (g_tokens_dump) fprintf(stderr, "%d ", t);
        if (echo && T) { char buf[64]; int bn = tok_decode(T, &t, 1, buf, 63); fwrite(buf,1,bn,stdout); fflush(stdout); }
        if (is_stop(t)) break;
        if (s == n_new - 1) break;
        logit = step(m, &out[len-1], 1, len-1);      /* DECODE */
    }
    if (g_tokens_dump) fprintf(stderr, "\n");
    if (dt_out) *dt_out = now_s() - t0;
    return ng;
}

/* ---------- ref.json (validazione) ---------- */
static int *read_int_array(jval *o, const char *key, int *n_out) {
    jval *a = json_get(o, key);
    int *r = malloc(a->len * sizeof(int));
    for (int i = 0; i < a->len; i++) r[i] = (int)a->kids[i]->num;
    *n_out = a->len; return r;
}

static int run_ref(Model *m, const char *refpath) {
    FILE *f = fopen(refpath, "rb"); if(!f){perror(refpath);return 1;}
    fseek(f,0,SEEK_END); long n=ftell(f); fseek(f,0,SEEK_SET);
    char *buf=malloc(n+1); if(fread(buf,1,n,f)!=(size_t)n){} buf[n]=0; fclose(f);
    jval *ref = json_parse(buf);
    int np, nfull; int *prompt = read_int_array(ref,"prompt_ids",&np); int *full = read_int_array(ref,"full_ids",&nfull);
    int n_new = nfull - np;
    if (n_new <= 0) { fprintf(stderr,"ref.json: full_ids non estende prompt_ids\n"); return 1; }
    kv_alloc(m, nfull + 1);
    int *out = malloc(nfull * sizeof(int));
    g_temp = 0;                              /* REF e' greedy: pick_tok -> argmax (bit-identico) */
    double dt; generate(m, NULL, prompt, np, n_new, out, 0, &dt);
    int match = 0;
    printf("Reference: ");  for (int i=np;i<nfull;i++) printf("%d ", full[i]);
    printf("\nC engine : ");  for (int i=np;i<nfull;i++) { printf("%d ", out[i]); if (out[i]==full[i]) match++; }
    printf("\nMatching tokens: %d/%d\n", match, n_new);
    /* hit/miss ora per-layer (moe.h): aggrega per il reporting */
    long long hits = 0, miss = 0;
    for (int i = 0; i < m->c.n_layers; i++) { hits += m->cache[i].hits; miss += m->cache[i].misses; }
    double tot = (double)hits + miss;
    printf("Speed: %.2f tok/s | PEAK RSS %.2f GB\n", n_new/dt, rss_gb());
    printf("Expert cache hit rate: %.1f%%  (hit=%lld miss=%lld)\n", tot?100.0*hits/tot:0.0, hits, miss);
    json_free(ref); free(buf); free(prompt); free(full); free(out);
    return match == n_new ? 0 : 2;
}

/* ---------- main (interfaccia ad env, come qwen/gemma) ---------- */
int main(int argc, char **argv) { (void)argc; (void)argv;
    const char *snap = getenv("SNAP");
    if (!snap) { fprintf(stderr, "set SNAP=<snapshot directory>\n"); return 1; }
    int cap  = getenv("EXPERT_CACHE") ? atoi(getenv("EXPERT_CACHE")) : 16;
    int bits = getenv("EBITS")        ? atoi(getenv("EBITS"))        : 8;
    if (bits < 2 || bits > 8) {   /* expert storage is int8_t: bits>8 truncates in quantize_rows (#134). f32 mode is not implemented here — int8 is already token-exact vs the oracle. */
        fprintf(stderr, "EBITS deve essere 2..8 (got %d): storage expert int8, niente modo f32\n", bits);
        return 1;
    }
    int ngen  = getenv("NGEN")  ? atoi(getenv("NGEN"))  : 256;
    int maxctx= getenv("CTX")   ? atoi(getenv("CTX"))   : 4096;
    if (getenv("TEMP"))    g_temp = (float)atof(getenv("TEMP"));
    if (getenv("NUCLEUS")) g_nuc  = (float)atof(getenv("NUCLEUS"));
    if (getenv("SEED"))    g_rng  = (uint64_t)strtoull(getenv("SEED"),NULL,10) | 1u;
    g_tokens_dump = getenv("TOKENS") && atoi(getenv("TOKENS"));

    Model m; model_init(&m, snap, cap, bits);
    fprintf(stderr, "[" ENGINE_TAG "] %d layer, hidden %d, %d teste (hd %d), %d expert top-%d, inter %d, vocab %d, cache %d expert/layer @%dbit | load %.1fs | RSS %.2f GB\n",
            m.c.n_layers, m.c.hidden, m.c.n_heads, m.c.head_dim, m.c.n_experts, m.c.topk,
            m.c.inter, m.c.vocab, cap, bits, m.dense_load_s, rss_gb());

    const char *refpath = getenv("REF");
    if (refpath) return run_ref(&m, refpath);

    /* generazione: serve il tokenizer */
    Tok T; char tokpath[2048]; snprintf(tokpath, sizeof(tokpath), "%s/tokenizer.json", snap);
    tok_load(&T, tokpath);
    /* ricarica config.json per eos_token_id (load_cfg non lo tiene) -> stop set */
    char cpath[2048]; snprintf(cpath, sizeof(cpath), "%s/config.json", snap);
    FILE *cf = fopen(cpath, "rb");
    if (cf) {
        fseek(cf,0,SEEK_END); long cn=ftell(cf); fseek(cf,0,SEEK_SET);
        char *cb=malloc(cn+1); if(fread(cb,1,cn,cf)!=(size_t)cn){} cb[cn]=0; fclose(cf);
        jval *cfg = json_parse(cb);
        jval *e = json_get(cfg,"eos_token_id");
        if (e) stop_add(e->t==J_ARR ? (int)e->kids[0]->num : (int)e->num);
        json_free(cfg); free(cb);
    }

    const char *prompt = getenv("PROMPT");
    int *hist = malloc(maxctx * sizeof(int));
    if (prompt) {                                   /* one-shot */
        int k = tok_encode(&T, prompt, (int)strlen(prompt), hist, maxctx - 2);
        int cur = ngen; if (k + cur + 1 > maxctx) cur = maxctx - k - 1;
        kv_alloc(&m, maxctx);
        generate(&m, &T, hist, k, cur, hist, 1, NULL);
        printf("\n");
        return 0;
    }

    /* chat interattiva: KV persistente, storia append-only */
    fprintf(stderr, "[" ENGINE_TAG "] chat interattiva: scrivi e premi invio (Ctrl-D per uscire)\n");
    kv_alloc(&m, maxctx);
    int len = 0;
    char *line = NULL; size_t lcap = 0;
    for (;;) {
        fprintf(stderr, "\n> "); fflush(stderr);
        ssize_t nr = getline(&line, &lcap, stdin);
        if (nr < 0) break;
        while (nr > 0 && (line[nr-1]=='\n' || line[nr-1]=='\r')) line[--nr]=0;
        if (!nr) continue;
        int k = tok_encode(&T, line, (int)nr, hist + len, maxctx - len - 2);
        if (len + k + 8 >= maxctx) {
            fprintf(stderr, "[" ENGINE_TAG "] contesto pieno, reset della conversazione\n");
            len = 0; m.kv_len = 0;
            k = tok_encode(&T, line, (int)nr, hist, maxctx - 2);
        }
        int cur = ngen; if (len + k + cur + 1 > maxctx) cur = maxctx - len - k - 1;
        int ng = generate(&m, &T, hist + len, k, cur, hist + len, 1, NULL);
        len += k + ng;
    }
    return 0;
}
