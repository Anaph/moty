/* Caricamento pesi: load_t/load_mat_bits, budget streaming, micro-RSS.
 * Estratto da runtime.h (P2). Include DOPO Cfg/Layer/Model
 * e le dichiarazioni degli hook. Un'istanza per TU (tutto static). */
#ifndef RT_MODEL_LOAD_H
#define RT_MODEL_LOAD_H

/* open phases (MotyCommon.load_ph): where the seconds of a model open go */
enum { LP_INDEX, LP_SMALL, LP_EMBED, LP_HEAD, LP_LAYERS, LP_FUSE, LP_TOK, LP_KV, LP_N };
static const char *const lp_name[LP_N] = { "index", "small", "embed", "head", "layers", "fuse", "tokenizer", "kv" };
#define LP_MARK(m, ph, t) do { double _t = now_s(); (m)->base.load_ph[ph] += _t - (t); (t) = _t; } while (0)

/* ---------- caricamento pesi ---------- */
static float *load_t(Model *m, const char *name, int64_t expect) {
    st_tensor *t = st_expect(&m->S, name, expect);
    float *p = falloc(t->numel);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

/* gruppo delle scale int4 (QGROUP): 32 = blocco Q4_0 di GGUF; 0 = per riga */

/* carica [O,I] e quantizza secondo bits: 0=f32, 8=int8+scala per riga,
 * 4=int4 impacchettato con scale per gruppo (g_qgroup; 0 -> per riga) */
/* GGUF Q4_0 + QBITS=4 con gruppo 32: repack LOSSLESS (pura permutazione di
 * nibble, gguf.h) invece di dequant+requant — girano gli stessi bit del file */
static void load_mat_q4_0(Model *m, Mat *w, const char *name, int O, int I) {
    st_tensor *t = st_expect(&m->S, name, (int64_t)O*I);
    void *raw = balloc(t->nbytes, name);
    st_read_raw(&m->S, name, raw, 0);
    w->q4 = balloc((int64_t)O*(I/2), name); w->qs = falloc((int64_t)O*(I/32));
    gguf_repack_q4_0(raw, w->q4, w->qs, O, I);
    free(raw);
    w->gs = 32; w->fmt = WF_I4G;
}

/* Q8_TENSORS=<glob>[,<glob>...] ('*' matches any run of characters):
 * tensors (snapshot names; the tied head is "lm_head.weight") that take
 * Q8R4 instead of Q4R4 under QBITS=4 Q4FMT=r4 — mixed precision for the
 * tensors int4 damages most (docs/performance.md). */
static int r4_glob(const char *p, const char *s) {
    for (; *p && *p != ','; p++, s++) {
        if (*p == '*') {
            for (const char *t = s; ; t++) { if (r4_glob(p + 1, t)) return 1; if (!*t) return 0; }
        }
        if (*s != *p) return 0;
    }
    return *s == 0;
}
static int r4_wants_q8(const char *name) {
    const char *p = g_q8_tensors;
    for (; p && *p; p = strchr(p, ',') ? strchr(p, ',') + 1 : NULL)
        if (r4_glob(p, name)) return 1;
    return 0;
}

/* QBITS=4 + Q4FMT=r4: rows read from disk in ~4 MB chunks (never the whole
 * f32 matrix — the lm_head of a 130k-vocab model would be 800 MB) and
 * packed into 4-row blocks in parallel (nn/quant.c, hw/hw_q4r4.h): Q4R4,
 * or Q8R4 when `pname` matches Q8_TENSORS. A pre-packed container decides
 * by itself: "<name>" + F16 "<name>.s16", 64 bytes per block-group = Q4R4,
 * 128 = Q8R4. */
/* int4 lm_head rows [v0, v1) re-packed from the int8 embedding table */
typedef struct { Mat *h; const int8_t *eq; const float *eqs; int D; int64_t rb; } HeadI4Job;
static void head_i4_part(void *c_, int64_t v0, int64_t v1, int tid) {
    const HeadI4Job *c = c_; int D = c->D;
    for (int64_t v = v0; v < v1; v++) {
        float row[2048];
        const int8_t *er = c->eq + v*D;
        for (int i = 0; i < D; i++) row[i] = er[i] * c->eqs[v];
        pack_int4(row, c->h->q4 + v*c->rb, c->h->qs + v, 1, D);
    }
}

/* R4 packing of the 4-row blocks of one chunk read from disk */
typedef struct { Mat *w; const float *chunk; int rr, o0, I, nb, gb; } R4PackJob;
static void r4_pack_part(void *c_, int64_t b0, int64_t b1, int tid) {
    const R4PackJob *c = c_; Mat *w = c->w; int I = c->I, nb = c->nb, rr = c->rr;
    for (int b = (int)b0; b < (int)b1; b++) {
        int nr = rr - b*4 < 4 ? rr - b*4 : 4, ob = c->o0/4 + b;
        if (c->gb == 128) moty_pack_q8r4_block(c->chunk + (int64_t)b*4*I, nr, I, (int8_t *)w->q4 + (int64_t)ob*nb*128, w->s16 + (int64_t)ob*nb*4);
        else moty_pack_q4r4_block(c->chunk + (int64_t)b*4*I, nr, I, w->q4 + (int64_t)ob*nb*64, w->s16 + (int64_t)ob*nb*4);
    }
}

static void load_mat_r4(Model *m, Mat *w, const char *name, const char *pname, int O, int I) {
    int nb = I / 32, O4 = (O + 3) / 4;
    char sn[256]; snprintf(sn, sizeof sn, "%s.s16", name);
    int packed = st_has(&m->S, sn), gb;
    if (packed) {
        int64_t nq = st_nbytes(&m->S, name);
        gb = nq == (int64_t)O4*nb*64 ? 64 : nq == (int64_t)O4*nb*128 ? 128 : 0;
        if (!gb || st_nbytes(&m->S, sn) != (int64_t)O4*nb*4*2) {
            moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] %s: packed R4 size mismatch (O=%d I=%d)\n", name, O, I);
        }
    } else gb = r4_wants_q8(pname) ? 128 : 64;
    w->O = O; w->I = I; w->fmt = gb == 128 ? WF_Q8R4 : WF_Q4R4;
    if (packed) {                                   /* pre-packed container: used in place (mmap) */
        const void *q = st_map(&m->S, name, 1), *d = st_map(&m->S, sn, 2);
        if (q && d) { w->q4 = (uint8_t *)q; w->s16 = (uint16_t *)d; w->borrowed = 2; return; }
    }
    w->q4 = balloc((int64_t)O4*nb*gb, name);
    w->s16 = balloc((int64_t)O4*nb*4*sizeof(uint16_t), name);
    if (packed) {                                   /* no mapping: raw read */
        st_read_raw(&m->S, name, w->q4, 0);
        st_read_raw(&m->S, sn, w->s16, 0);
        return;
    }
    st_expect(&m->S, name, (int64_t)O*I);
    int rows = (int)(((4 << 20) / ((int64_t)I*4)) & ~3); if (rows < 4) rows = 4;
    float *chunk = falloc((int64_t)rows*I);
    for (int o0 = 0; o0 < O; o0 += rows) {
        int rr = O - o0 < rows ? O - o0 : rows;
        st_read_slice_f32(&m->S, name, (int64_t)o0*I, (int64_t)rr*I, chunk, 0);
        R4PackJob pj = { w, chunk, rr, o0, I, nb, gb };
        moty_par_for((rr + 3) / 4, 0, r4_pack_part, &pj);
    }
    free(chunk);
}

static void quantize_from_disk(Model *m, const char *name, int8_t *q, float *qs, int64_t N, int I, int rows);

static void load_mat_bits(Model *m, Mat *w, const char *name, int O, int I, int bits) {
    mat_reset_storage(w);            /* fmt = WF_F32 di default */
    mat_reset_storage(w);
    w->O = O; w->I = I;
    if ((bits == 4 || bits == 8) && I % 32 == 0) {
        /* a pre-packed container (Q4R4/Q8R4 by stored size) is read as it
         * was packed, whatever QBITS / Q4FMT say: the file defines the layout */
        char sn[256]; snprintf(sn, sizeof sn, "%s.s16", name);
        if (st_has(&m->S, sn) || (bits == 4 && g_q4fmt)) { load_mat_r4(m, w, name, name, O, I); return; }
    }
    if (bits == 8) {             /* row chunks: bit-identical to quantize_rows on the whole matrix */
        st_expect(&m->S, name, (int64_t)O*I);
        w->q = balloc((int64_t)O*I, name); w->qs = falloc(O);
        quantize_from_disk(m, name, w->q, w->qs, O, I, 0);
        w->fmt = WF_I8;
        return;
    }
    if (bits == 4 && g_qgroup == 32 && I % 32 == 0 && st_dtype(&m->S, name) == ST_Q4_0) {
        load_mat_q4_0(m, w, name, O, I);
        return;
    }
    w->f = load_t(m, name, (int64_t)O*I);
    if (bits == 8) {
        w->q = balloc((int64_t)O*I, name); w->qs = falloc(O);
        quantize_rows(w->f, w->q, w->qs, O, I, 8);
        free(w->f); w->f = NULL;
        w->fmt = WF_I8;
    } else if (bits == 4) {
        int gs = g_qgroup;
        int64_t rb = ((int64_t)I+1)/2, ng = gs > 0 ? ((int64_t)I+gs-1)/gs : 1;
        w->q4 = balloc((int64_t)O*rb, name); w->qs = falloc((int64_t)O*ng);
        if (gs > 0) pack_int4_grouped(w->f, w->q4, w->qs, O, I, gs);
        else pack_int4(w->f, w->q4, w->qs, O, I);
        w->gs = gs;
        free(w->f); w->f = NULL;
        w->fmt = gs > 0 ? WF_I4G : WF_I4;
    } else if (bits == 2) {
        int64_t rb = ((int64_t)I+3)/4;
        w->q4 = balloc((int64_t)O*rb, name); w->qs = falloc(O);
        pack_int2(w->f, w->q4, w->qs, O, I, 2);
        free(w->f); w->f = NULL;
        w->fmt = WF_I2;
    } else if (bits == -1) { /* nativo Q4_K/Q6_K: raw, nessuna re-quant */
        st_tensor *t = st_expect(&m->S, name, (int64_t)O*I);
        if (t->dtype == ST_Q4_K || t->dtype == ST_Q6_K) {
            void *raw = balloc(t->nbytes, name);
            st_read_raw(&m->S, name, raw, 0);
            w->q4 = (uint8_t*)raw;
            w->fmt = (t->dtype == ST_Q4_K) ? WF_Q4K : WF_Q6K;
        } else {
            w->f = load_t(m, name, (int64_t)O*I);
            w->fmt = WF_F32;
        }
    }
    /* bits==0: fmt resta WF_F32 (default), w->f residente */
}

static void load_mat(Model *m, Mat *w, const char *name, int O, int I) {
    load_mat_bits(m, w, name, O, I, m->base.qbits);
}

/* legge un tensore [N,I] dal disco a blocchi di righe e lo quantizza int8
 * per riga in (q, qs): il transiente f32 e' un blocco da ~4 MB, mai l'intera
 * matrice. quantize_rows lavora per riga, quindi il risultato e' bit-identico
 * alla quantizzazione one-shot comunque si spezzi. rows<1 -> blocco dal
 * budget; i test passano un rows minuscolo per esercitare il loop. */
static void quantize_from_disk(Model *m, const char *name, int8_t *q, float *qs,
                               int64_t N, int I, int rows) {
    static float *chunk = NULL; static int64_t ccap = 0;
    if (rows < 1) rows = (int)((4 << 20) / ((int64_t)I * 4));
    if (rows < 1) rows = 1;
    grow((void **)&chunk, &ccap, (int64_t)rows*I, sizeof(float), "chunk quantizzazione");
    for (int64_t o = 0; o < N; o += rows) {
        int64_t rr = N - o < rows ? N - o : rows;
        st_read_slice_f32(&m->S, name, o*(int64_t)I, rr*I, chunk, 0);
        quantize_rows(chunk, q + o*I, qs + o, (int)rr, I, 8);
    }
}

/* a container (SAVE_PACKED, format moty-r4 v2) stores the table as moty
 * keeps it: I8 "model.embed_tokens.weight" [V*D] + F32 ".qs" [V] */
static int embed_is_q8(Model *m) {
    return st_has(&m->S, "model.embed_tokens.weight.qs") && st_dtype(&m->S, "model.embed_tokens.weight") == 3;
}

/* embed int8 per riga (QBITS=8): la tabella non passa MAI intera per la RAM */
static int g_embed_chunk_rows = 0;      /* 0 = auto; i test lo stringono */
static void load_embed_q8(Model *m) {
    Cfg *c = &m->c; int D = c->hidden; int64_t V = c->vocab;
    if (embed_is_q8(m)) {                         /* pre-quantized: in place (mmap) or one raw read */
        const char *en = "model.embed_tokens.weight", *sn = "model.embed_tokens.weight.qs";
        if (st_nbytes(&m->S, en) != V*D || st_nbytes(&m->S, sn) != V*4)
            moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] %s: int8 table size mismatch (vocab %lld, hidden %d)\n", en, (long long)V, D);
        const void *q = st_map(&m->S, en, 1), *qs = st_map(&m->S, sn, 4);
        if (q && qs) { m->base.embed_q = (int8_t *)q; m->base.embed_qs = (float *)qs; return; }
        m->base.embed_q = balloc(V*D, "embed int8"); m->base.embed_qs = falloc(V);
        st_read_raw(&m->S, en, m->base.embed_q, 0); st_read_raw(&m->S, sn, m->base.embed_qs, 0);
        return;
    }
    st_expect(&m->S, "model.embed_tokens.weight", V*D);
    m->base.embed_q  = balloc(V*D, "embed int8");
    m->base.embed_qs = falloc(V);
    quantize_from_disk(m, "model.embed_tokens.weight", m->base.embed_q, m->base.embed_qs,
                       V, D, g_embed_chunk_rows);
}

static int64_t layer_f32_bytes(Model *m, int li) {
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    int64_t b = 0;
    for (int j = 0; j < n; j++) b += (int64_t)r[j].O*r[j].I*4;
    return b;
}

/* rilettura di un layer streamato. QBITS=0/4: f32 in stream_buf come sempre
 * (per int4 l'impacchettamento a OGNI step costerebbe piu' del risparmio).
 * QBITS=8: lettura a blocchi di righe + quantize_rows nello scratch int8 —
 * il transiente f32 e' un blocco, lo scratch e' 4x piu' piccolo e le matrici
 * streamate girano sullo stesso kernel int8 di quelle residenti (la
 * quantizzazione per riga rende il risultato bit-identico al load residente). */
static void layer_stream_in(Model *m, int li) {
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    if (m->base.qbits == 8) {
        int64_t qoff = 0, soff = 0;
        for (int j = 0; j < n; j++) {
            int O = r[j].O, I = r[j].I;
            quantize_from_disk(m, r[j].name, m->base.stream_q + qoff, m->base.stream_qs + soff, O, I, 0);
            mat_reset_storage(r[j].mat);
            r[j].mat->q = m->base.stream_q + qoff; r[j].mat->qs = m->base.stream_qs + soff;
            r[j].mat->O = O; r[j].mat->I = I; r[j].mat->fmt = WF_I8;
            qoff += (int64_t)O*I; soff += O;
        }
        return;
    }
    int64_t off = 0;
    for (int j = 0; j < n; j++) {
        st_read_f32(&m->S, r[j].name, m->base.stream_buf + off, 0);  /* drop=0: la page cache aiuta */
        mat_reset_storage(r[j].mat);
        r[j].mat->f = m->base.stream_buf + off;
        r[j].mat->O = r[j].O; r[j].mat->I = r[j].I;
        off += (int64_t)r[j].O*r[j].I;
    }
}

static void layer_prefetch(Model *m, int li) {
#ifndef _WIN32                       /* su Windows WILLNEED e' sincrono: niente overlap */
    MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, li, r);
    for (int j = 0; j < n; j++) st_prefetch(&m->S, r[j].name);
#else
    (void)m; (void)li;
#endif
}

/* ---------- micro-RSS (MICRO=1): consumo di RAM minimo assoluto ----------
 * NESSUN peso resta residente: l'embedding si legge per riga (gather nello
 * step del motore), ogni matmul rilegge la propria matrice dal disco a blocchi
 * di g_micro_chunk byte in uno scratch costante. Con g_micro_drop=1 (default)
 * ogni blocco viene anche scartato dalla page cache subito dopo l'uso: il
 * footprint e' davvero solo attivazioni + KV + tokenizer, pensato per limiti
 * HARD (cgroup/embedded). Prezzo: l'intero modello transita dal disco a OGNI
 * token — la velocita' e' bandwidth-del-disco, non della RAM. */
static int64_t g_micro_chunk = 4 << 20; /* byte f32 dello scratch di streaming */

/* y[S,O] = x[S,I] @ W^T leggendo W dal disco a blocchi di righe; installata in
 * g_mat_stream_fn cosi' mat_apply (nn.h) la usa per le Mat con sh!=NULL.
 * Scratch statico che cresce e basta: contratto di chiamata SERIALE, come
 * matmul_q_s. Bit-identica al percorso f32 residente (stesse righe, stesso
 * dot_f32). */
typedef struct { float *y; const float *x, *buf; int S, I, O, o0; } StreamJob;
static void stream_rows(void *c_, int64_t r0, int64_t r1, int tid) {
    const StreamJob *c = c_; int S = c->S, I = c->I, O = c->O;
    for (int64_t o = r0; o < r1; o++)
        for (int s = 0; s < S; s++)
            c->y[(int64_t)s*O + c->o0 + o] = dot_f32(c->x + (int64_t)s*I, c->buf + o*I, I);
}

static void mat_stream(float *y, const float *x, const Mat *w, int S) {
    shards *Sh = (shards *)w->sh;
    int I = w->I, O = w->O;
    int rows = (int)(g_micro_chunk / ((int64_t)I * 4));
    if (rows < 1) rows = 1;
    if (rows > O) rows = O;
    static float *buf = NULL; static int64_t cap = 0;
    grow((void **)&buf, &cap, (int64_t)rows * I, sizeof(float), "scratch micro");
    for (int o0 = 0; o0 < O; o0 += rows) {
        int r = O - o0 < rows ? O - o0 : rows;
        st_read_slice_f32(Sh, w->sname, (int64_t)o0 * I, (int64_t)r * I, buf, g_micro_drop);
        StreamJob sj = { y, x, buf, S, I, O, o0 };
        moty_par_for(r, 0, stream_rows, &sj);
    }
}

/* prepara una Mat streamata: dims validate contro il file, nessun dato letto */
static void mat_stream_init(Model *m, Mat *w, const char *name, int O, int I) {
    st_expect(&m->S, name, (int64_t)O*I);
    mat_reset_storage(w);
    w->O = O; w->I = I;
    w->sh = &m->S; w->sname = strdup(name);
}

/* micro-RSS: ogni Mat diventa un descrittore streamato, zero pesi residenti */
static void model_init_micro(Model *m) {
#if ENGINE_MICRO
    Cfg *c = &m->c;
    moty_nn_set_stream_fn(mat_stream);   /* M3: setter invece del simbolo */
    mat_stream_init(m, &m->base.lm_head,
                    m->base.lm_tied ? "model.embed_tokens.weight" : "lm_head.weight",
                    c->vocab, c->hidden);
    for (int i = 0; i < c->n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++) mat_stream_init(m, r[j].mat, r[j].name, r[j].O, r[j].I);
    }
    m->base.n_resident = 0;                    /* verita': zero layer residenti */
    fprintf(stderr, "[" ENGINE_TAG "] micro-RSS: 0 pesi residenti, matmul streamato a blocchi da %lld MB, page cache %s\n",
            (long long)(g_micro_chunk >> 20), g_micro_drop ? "scartata (MICRO_DROP=0 per tenerla)" : "attiva");
#else
    (void)m;
    moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] MICRO=1 non supportato da questo motore\n");
#endif
}

/* quanti layer di matrici stanno nel budget, con la stima onesta della parte
 * fissa (embed/testa/norme/KV) e dello scratch di streaming */
static int budget_resident(Model *m, int64_t budget_bytes, int ctx_hint) {
    Cfg *c = &m->c; int D = c->hidden;
    int64_t max_lb = 0;
    for (int i = 0; i < c->n_layers; i++) { int64_t b = layer_f32_bytes(m, i); if (b > max_lb) max_lb = b; }
    /* embed (e l'eventuale lm_head separato): f32 oppure int8+scala
     * (con QBITS=4 embed e testa restano comunque int8) */
    int64_t vd = m->base.qbits ? (int64_t)c->vocab*D + (int64_t)c->vocab*4
                          : (int64_t)c->vocab*D*4;
    int64_t fixed = vd + (int64_t)D*4;                          /* + final_norm */
    if (!m->base.lm_tied) fixed += vd;
    fixed += (int64_t)c->n_layers * 8 * D * 4;                  /* norme/vettori: stima larga */
    fixed += fixed_bytes(m, ctx_hint > 0 ? ctx_hint : 4096);    /* hook: KV, PLE... */
    /* scratch di streaming: int8 con QBITS=8 (layer_stream_in quantizza), f32 altrimenti */
    int64_t scratch = (m->base.qbits == 8) ? max_lb/4 + max_lb/64 : max_lb;
    int64_t used = fixed + scratch;
    int R = 0;
    for (; R < c->n_layers; R++) {
        int64_t lb = layer_f32_bytes(m, R);
        if (m->base.qbits == 8) lb = lb/4 + lb/64;                   /* int8 + scale */
        else if (m->base.qbits == 4) lb = lb/8 + lb/32;              /* int4 + scale di gruppo (gs=32) */
        if (used + lb > budget_bytes) break;
        used += lb;
    }
    fprintf(stderr, "[" ENGINE_TAG "] budget %.2f GB -> %d/%d layer residenti (fisso %.2f GB, scratch %.2f GB)\n",
            budget_bytes/1073741824.0, R, c->n_layers, fixed/1073741824.0, scratch/1073741824.0);
#if ENGINE_MICRO
    /* il classico non scende sotto embed + scratch: budget irrealizzabile */
    if (budget_bytes < fixed + scratch)
        fprintf(stderr, "[" ENGINE_TAG "] budget sotto il pavimento residente (%.2f GB): per la RSS minima usa MICRO=1\n",
                (fixed + scratch)/1073741824.0);
#endif
    return R;
}

/* scratch di streaming, dimensionato sul massimo dei layer EFFETTIVAMENTE
 * streamati (i >= n_resident), non sul massimo globale */
static void stream_scratch_alloc(Model *m) {
    Cfg *c = &m->c;
    int64_t smax = 0, rmax = 0;
    for (int i = m->base.n_resident; i < c->n_layers; i++) {
        int64_t b = layer_f32_bytes(m, i); if (b > smax) smax = b;
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        int64_t rows = 0; for (int j = 0; j < n; j++) rows += r[j].O;
        if (rows > rmax) rmax = rows;
    }
    if (m->base.qbits == 8) {
        m->base.stream_q = balloc(smax/4, "scratch stream int8");  /* 1 byte per elemento f32 */
        m->base.stream_qs = falloc(rmax);
    } else {
        m->base.stream_buf = falloc(smax/4);
    }
}

/* budget_bytes==0 -> tutto residente (comportamento classico) */
static void model_init_ex(Model *m, const char *snap, int qbits, int64_t budget_bytes, int ctx_hint) {
    memset(m, 0, sizeof(*m));
    m->base.qbits = qbits;
    double tp = now_s();
    /* GGUF: l'indice va costruito PRIMA del config (i metadati SONO il config) */
    if (g_gguf) gguf_index(&m->S, &g_gguf_meta, g_gguf);
    load_cfg(&m->c, snap);
    if (!g_gguf) st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    int D = c->hidden;
    /* final norm: model.norm.weight (standard); se assente (LFM2: token_embd_norm) lo fornira" load_small */
    if (st_has(&m->S, "model.norm.weight"))
        m->base.final_norm = load_t(m, "model.norm.weight", D);
    m->base.lm_tied = c->tie_emb || !st_has(&m->S, "lm_head.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    /* 1) parte piccola SEMPRE residente (hook: norme, vettori, stati, PLE...) */
    LP_MARK(m, LP_INDEX, tp);
    load_small(m);
    LP_MARK(m, LP_SMALL, tp);
    /* micro-RSS: nessun peso residente, embed compreso (gather per riga nello
     * step del motore); ogni Mat diventa un descrittore streamato. */
    if (g_micro && embed_is_q8(m))
        moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] MICRO=1 needs an f32/bf16 embedding table (this container stores it int8)\n");
    if (g_micro) { model_init_micro(m); m->base.load_s = now_s() - t0; return; }
    /* QBITS!=0 copre anche l'embed, ma SEMPRE a int8 (anche con QBITS=4):
     * l'lm_head e' il GEMV piu' sensibile alla quantizzazione e l'int4 li'
     * risparmierebbe poco rispetto alle matrici dei layer */
    /* EMBED=disk: no resident table, embed_row gathers the row from the file
     * (the micro-RSS branch). Only valid when nothing else reads the table:
     * a tied lm_head must then be packed separately (QBITS=4 Q4R4). */
    int head_q4r4 = m->base.lm_tied && D % 32 == 0          /* a container's packed head, or QBITS=4 r4 */
                    && (st_has(&m->S, "lm_head.weight.s16") || (m->base.qbits == 4 && g_q4fmt));
    if (g_embed_disk && m->base.lm_tied && !head_q4r4) {
        moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] EMBED=disk con lm_head legato richiede QBITS=4 Q4FMT=r4\n");
    }
    if (g_embed_disk && embed_is_q8(m))
        moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] EMBED=disk: this container's table is int8 in place (mmap) already\n");
    if (g_embed_disk) st_expect(&m->S, "model.embed_tokens.weight", (int64_t)c->vocab*D);
    else if (m->base.qbits > 0 || moty_getenv("EMBED_Q8") || embed_is_q8(m)) load_embed_q8(m);  /* qbits=-1 (native): f32 embed */
    else m->base.embed = load_t(m, "model.embed_tokens.weight", (int64_t)c->vocab*D);
    LP_MARK(m, LP_EMBED, tp);
    if (m->base.lm_tied) {
        mat_reset_storage(&m->base.lm_head);
        m->base.lm_head.O = c->vocab; m->base.lm_head.I = D;
        if (head_q4r4) {
            /* tied head in Q4R4 packed from the original rows (not from the
             * int8 table: no double quantization); a container stores it
             * pre-packed as lm_head.weight */
            load_mat_r4(m, &m->base.lm_head, st_has(&m->S, "lm_head.weight.s16") ? "lm_head.weight"
                                                   : "model.embed_tokens.weight", "lm_head.weight", c->vocab, D);
        } else if (m->base.qbits == 4 && m->base.embed_q && D <= 2048) {
            /* lm_head separato in INT4: il GEMV del logit e' ~43% del traffico
             * per-token in decode (262MB→131). Lookup embed resta int8. */
            int64_t V = c->vocab, rb = (D+1)/2;
            m->base.lm_head.q4 = balloc(V*rb, "lm_head i4");
            m->base.lm_head.qs = falloc(V);
            m->base.lm_head.fmt = WF_I4;
            HeadI4Job hj = { &m->base.lm_head, m->base.embed_q, m->base.embed_qs, D, rb };
            moty_par_for(V, 0, head_i4_part, &hj);
        } else {
            m->base.lm_head.f = m->base.embed; m->base.lm_head.q = m->base.embed_q; m->base.lm_head.qs = m->base.embed_qs;
            m->base.lm_head.fmt = m->base.embed_q ? WF_I8 : WF_F32;
        }
    } else {
        load_mat_bits(m, &m->base.lm_head, "lm_head.weight", c->vocab, D, m->base.qbits);
    }
    LP_MARK(m, LP_HEAD, tp);
    /* 2) budget -> quanti layer di matrici stanno residenti */
    m->base.n_resident = budget_bytes > 0 ? budget_resident(m, budget_bytes, ctx_hint)
                                     : c->n_layers;
    /* 3) matrici: residenti (QBITS onorato) o streamate (dims impostate, f=NULL) */
    for (int i = 0; i < c->n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++) {
            if (i < m->base.n_resident) load_mat(m, r[j].mat, r[j].name, r[j].O, r[j].I);
            else {
                st_expect(&m->S, r[j].name, (int64_t)r[j].O*r[j].I);
                mat_reset_storage(r[j].mat);
                r[j].mat->O = r[j].O; r[j].mat->I = r[j].I;
            }
        }
    }
    if (m->base.n_resident < c->n_layers) stream_scratch_alloc(m);
    LP_MARK(m, LP_LAYERS, tp);
    m->base.load_s = now_s() - t0;
}

/* ---------- SAVE_PACKED=<dir>: write a pre-packed container (moty-q4r4 v2) ----------
 * A safetensors file (moty container convention, cf. olmoe/glm "name.qs"):
 * every resident WF_Q4R4 (WF_Q8R4) matrix as U8 (I8) "<name>" (its 4-row
 * blocks) + F16 "<name>.s16"; the lm_head as "lm_head.weight" (+.s16), also
 * when tied. v2 (made to be used in place through mmap, docs/api.md):
 *   - the token embedding as moty keeps it: I8 "model.embed_tokens.weight"
 *     + F32 "model.embed_tokens.weight.qs" (per-row scale), not the bf16/f32
 *     table that every open re-quantized;
 *   - the codes of the matrices the engine fuses (q/k/v, gate/up: one
 *     buffer after moty_mat_fuse_rows) written back to back, their scales
 *     likewise, so the fused matrix is a view of the mapping (no copy);
 *   - only the tensors the loader read (a VL snapshot's vision tower is
 *     dropped); other tensors copied byte for byte with their dtype.
 * Data 64-byte aligned. config/tokenizer files are copied. A v1 container
 * (and pack_r4.py output) still loads; SAVE_PACKED from it upgrades it. */
#ifdef _WIN32
#include <direct.h>
#define rt_mkdir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define rt_mkdir(p) mkdir(p, 0755)
#endif
static int pk_name_is(Model *m, const char *sname, Mat **out) {
    char cb[512]; const char *name = st_canon(sname, cb, sizeof cb);
    if (!strcmp(name, "lm_head.weight") && (m->base.lm_head.fmt == WF_Q4R4 || m->base.lm_head.fmt == WF_Q8R4)) { *out = &m->base.lm_head; return 1; }
    for (int i = 0; i < m->c.n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++)
            if (!strcmp(r[j].name, name) && (r[j].mat->fmt == WF_Q4R4 || r[j].mat->fmt == WF_Q8R4)) { *out = r[j].mat; return 1; }
    }
    return 0;
}
static int64_t pk_qbytes(const Mat *w) { return (int64_t)(w->O + 3) / 4 * (w->I / 32) * (w->fmt == WF_Q8R4 ? 128 : 64); }
static int64_t pk_sbytes(const Mat *w) { return (int64_t)(w->O + 3) / 4 * (w->I / 32) * 8; }
static void model_save_packed(Model *m, const char *snap, const char *dir) {
    if (g_gguf || m->base.qbits != 4 || !g_q4fmt || m->base.n_resident < m->c.n_layers) {
        moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] SAVE_PACKED: needs SNAP=, QBITS=4 Q4FMT=r4, all layers resident\n");
    }
    rt_mkdir(dir);
    typedef struct { char name[256]; const char *dt; int64_t n, bytes; const void *mem; int src, glue; } PkT;   /* glue: no padding before it */
    int cap = 2 * m->S.n + 8, nt = 0;
    PkT *t = calloc(cap, sizeof(PkT));
    typedef struct { char name[256]; Mat *w; int done; } PkM;
    PkM *pm = calloc(m->S.n + 1, sizeof(PkM)); int np = 0;
    static const char *dtn[4] = { "BF16", "F16", "F32", "U8" };
    int head_done = 0;
    /* 1) plain tensors (what the loader read), in file order */
    for (int i = 0; i < m->S.n; i++) {
        st_tensor *st = &m->S.t[i]; Mat *w;
        if (pk_name_is(m, st->name, &w)) {
            snprintf(pm[np].name, 256, "%s", st->name); pm[np++].w = w;
            if (!strcmp(st->name, "lm_head.weight")) head_done = 1;
            continue;
        }
        if (!st->used) continue;
        { size_t L = strlen(st->name); if (L > 4 && !strcmp(st->name + L - 4, ".s16")) continue; }   /* a packed matrix's scales */
        char cb[512]; const char *cn = st_canon(st->name, cb, sizeof cb);                      /* VL: model.language_model.X */
        if (!strcmp(cn, "model.embed_tokens.weight.qs")) continue;                                  /* written with the table */
        if (!strcmp(cn, "model.embed_tokens.weight") && m->base.embed_q && !m->base.embed) {
            int64_t V = m->c.vocab, D = m->c.hidden;
            snprintf(t[nt].name, 256, "%s", st->name); t[nt].dt = "I8"; t[nt].n = t[nt].bytes = V*D; t[nt].mem = m->base.embed_q; t[nt].src = -1; nt++;
            snprintf(t[nt].name, 256, "%s.qs", st->name); t[nt].dt = "F32"; t[nt].n = V; t[nt].bytes = V*4; t[nt].mem = m->base.embed_qs; t[nt].src = -1; nt++;
            continue;
        }
        if (st->dtype > 3) { moty_fail_code(MOTY_FAIL_FORMAT, "SAVE_PACKED: %s: block dtype unsupported\n", st->name); }
        snprintf(t[nt].name, 256, "%s", st->name); t[nt].dt = dtn[st->dtype]; t[nt].n = st->numel;
        t[nt].bytes = st->nbytes; t[nt].mem = NULL; t[nt].src = i; nt++;
    }
    if (!head_done && (m->base.lm_head.fmt == WF_Q4R4 || m->base.lm_head.fmt == WF_Q8R4)) {  /* tied head */
        snprintf(pm[np].name, 256, "lm_head.weight"); pm[np++].w = &m->base.lm_head;
    }
    /* 2) packed matrices: groups whose codes are back to back in memory (a
     * fused buffer) stay back to back; codes of every group first, then the
     * scales in the same order */
    int *ord = calloc(np + 1, sizeof(int)), *glue = calloc(np + 1, sizeof(int)); int no = 0;
    for (int i = 0; i < np; i++) {
        if (pm[i].done) continue;
        int grp[64], g = 1; grp[0] = i; pm[i].done = 1;
        for (int grew = 1; grew && g < 64; ) {
            grew = 0;
            for (int j = 0; j < np && g < 64; j++) {
                if (pm[j].done) continue;
                const Mat *f = pm[grp[0]].w, *l = pm[grp[g-1]].w, *w = pm[j].w;
                if (w->q4 == l->q4 + pk_qbytes(l)) { grp[g++] = j; pm[j].done = 1; grew = 1; }
                else if (w->q4 + pk_qbytes(w) == f->q4) { memmove(grp + 1, grp, g * sizeof(int)); grp[0] = j; g++; pm[j].done = 1; grew = 1; }
            }
        }
        for (int k = 0; k < g; k++) { glue[no] = k > 0; ord[no++] = grp[k]; }
    }
    for (int k = 0; k < no; k++) {
        Mat *w = pm[ord[k]].w;
        snprintf(t[nt].name, 256, "%s", pm[ord[k]].name); t[nt].dt = w->fmt == WF_Q8R4 ? "I8" : "U8";
        t[nt].n = t[nt].bytes = pk_qbytes(w); t[nt].mem = w->q4; t[nt].src = -1; t[nt].glue = glue[k]; nt++;
    }
    for (int k = 0; k < no; k++) {
        Mat *w = pm[ord[k]].w;
        snprintf(t[nt].name, 256, "%s.s16", pm[ord[k]].name); t[nt].dt = "F16";
        t[nt].n = pk_sbytes(w) / 2; t[nt].bytes = pk_sbytes(w); t[nt].mem = w->s16; t[nt].src = -1; t[nt].glue = glue[k]; nt++;
    }
    free(ord); free(glue); free(pm);
    /* header; each tensor starts on a 64-byte boundary of the file, except the
     * members of a fused group after the first (back to back: one view) */
    int64_t hcap = 256 + (int64_t)nt * 400; char *hdr = malloc(hcap); int64_t hl = 0, off = 0;
    int64_t *at = calloc(nt + 1, sizeof(int64_t));
    hl += snprintf(hdr + hl, hcap - hl, "{\"__metadata__\":{\"format\":\"moty-q4r4\",\"version\":\"2\"}");
    for (int i = 0; i < nt; i++) {
        if (!t[i].glue) off = (off + 63) & ~(int64_t)63;
        at[i] = off;
        hl += snprintf(hdr + hl, hcap - hl, ",\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
                       t[i].name, t[i].dt, (long long)t[i].n, (long long)off, (long long)(off + t[i].bytes));
        off += t[i].bytes;
    }
    hl += snprintf(hdr + hl, hcap - hl, "}");
    while ((8 + hl) % 64) hdr[hl++] = ' ';                          /* data section 64-byte aligned */
    char path[2048]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb"); if (!f) { moty_fail_code(MOTY_FAIL_IO, "%s: %s", path, strerror(errno)); }
    uint64_t h64 = (uint64_t)hl;
    fwrite(&h64, 8, 1, f); fwrite(hdr, 1, hl, f);
    void *buf = NULL; int64_t bcap = 0, pos = 0; static const char zero[64];
    for (int i = 0; i < nt; i++) {
        if (at[i] > pos) { fwrite(zero, 1, (size_t)(at[i] - pos), f); pos = at[i]; }
        const void *src = t[i].mem;
        if (!src) {
            if (t[i].bytes > bcap) {               /* local buffer: not grow(), which registers statics */
                void *nb = realloc(buf, (size_t)t[i].bytes);
                if (!nb) moty_fail_code(MOTY_FAIL_OOM, "OOM save copy (%lld byte)", (long long)t[i].bytes);
                buf = nb; bcap = t[i].bytes;
            }
            st_read_raw(&m->S, m->S.t[t[i].src].name, buf, 1);
            src = buf;
        }
        if (fwrite(src, 1, t[i].bytes, f) != (size_t)t[i].bytes) { moty_fail_code(MOTY_FAIL_IO, "%s: %s", "write", strerror(errno)); }
        pos += t[i].bytes;
    }
    fclose(f); free(buf); free(hdr); free(at);
    static const char *aux[] = { "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja" };
    for (size_t k = 0; k < sizeof aux / sizeof aux[0]; k++) {
        char src[2048], dst[2048]; snprintf(src, sizeof src, "%s/%s", snap, aux[k]); snprintf(dst, sizeof dst, "%s/%s", dir, aux[k]);
        long n; char *d = NULL; FILE *fs = fopen(src, "rb"); if (!fs) continue; fclose(fs);
        d = slurp_file(src, &n);
        FILE *fd = fopen(dst, "wb"); if (!fd) { moty_fail_code(MOTY_FAIL_IO, "%s: %s", dst, strerror(errno)); } fwrite(d, 1, n, fd); fclose(fd); free(d);
    }
    fprintf(stderr, "[" ENGINE_TAG "] SAVE_PACKED: %d tensors, %.1f MB -> %s (moty-q4r4 v2)\n", nt, off / 1048576.0, path);
}

static void model_init(Model *m, const char *snap, int qbits) {
    model_init_ex(m, snap, qbits, 0, 0);
}

/* KV_BITS=8: KV-cache int8 con scala per (testa_kv, posizione). Default 0
 * (f32): la numerica di REF non cambia mai in silenzio. */

/* riga id dell'embedding -> dst[D] moltiplicata per scale (gemma passa
 * sqrt(D), qwen 1): f32 residente, int8 dequant, oppure micro-RSS (lettura
 * della sola riga dal disco; drop=0, le righe calde sono minuscole). Era
 * open-coded in tre punti fra i due motori. */
static void embed_row(Model *m, int id, float scale, float *dst) {
    int D = m->c.hidden;
    if (m->base.inj && id == m->base.inj_tok) {     /* EMBEDS: the next external row, as is */
        if (m->base.inj_used >= m->base.inj_n) {
            moty_fail_code(MOTY_FAIL_FORMAT, "[" ENGINE_TAG "] EMBEDS: more placeholder tokens (%d) than rows (%d)\n",
                    m->base.inj_used + 1, m->base.inj_n);
        }
        memcpy(dst, m->base.inj + (int64_t)m->base.inj_used++ * D, D*sizeof(float));
        return;
    }
    if (m->base.embed) {
        const float *er = m->base.embed + (int64_t)id*D;
        if (scale == 1.f) memcpy(dst, er, D*sizeof(float));
        else for (int i = 0; i < D; i++) dst[i] = er[i] * scale;
    } else if (m->base.embed_q) {
        const int8_t *er = m->base.embed_q + (int64_t)id*D;
        float es = m->base.embed_qs[id] * scale;
        for (int i = 0; i < D; i++) dst[i] = er[i] * es;
    } else {
        st_read_slice_f32(&m->S, "model.embed_tokens.weight", (int64_t)id*D, D, dst, 0);
        if (scale != 1.f) for (int i = 0; i < D; i++) dst[i] *= scale;
    }
}

#endif /* RT_MODEL_LOAD_H */
