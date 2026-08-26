/* Caricamento pesi: load_t/load_mat_bits, budget streaming, micro-RSS.
 * Estratto da runtime.h (P2). Include DOPO Cfg/Layer/Model
 * e le dichiarazioni degli hook. Un'istanza per TU (tutto static). */
#ifndef RT_MODEL_LOAD_H
#define RT_MODEL_LOAD_H

/* ---------- caricamento pesi ---------- */
static float *load_t(Model *m, const char *name, int64_t expect) {
    st_tensor *t = st_expect(&m->S, name, expect);
    float *p = falloc(t->numel);
    st_read_f32(&m->S, name, p, 0);
    return p;
}

/* gruppo delle scale int4 (QGROUP): 32 = blocco Q4_0 di GGUF; 0 = per riga */
static int g_qgroup = 32;

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

static void load_mat_bits(Model *m, Mat *w, const char *name, int O, int I, int bits) {
    mat_reset_storage(w);            /* fmt = WF_F32 di default */
    mat_reset_storage(w);
    w->O = O; w->I = I;
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
    load_mat_bits(m, w, name, O, I, m->qbits);
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

/* embed int8 per riga (QBITS=8): la tabella non passa MAI intera per la RAM */
static int g_embed_chunk_rows = 0;      /* 0 = auto; i test lo stringono */
static void load_embed_q8(Model *m) {
    Cfg *c = &m->c; int D = c->hidden; int64_t V = c->vocab;
    st_expect(&m->S, "model.embed_tokens.weight", V*D);
    m->embed_q  = balloc(V*D, "embed int8");
    m->embed_qs = falloc(V);
    quantize_from_disk(m, "model.embed_tokens.weight", m->embed_q, m->embed_qs,
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
    if (m->qbits == 8) {
        int64_t qoff = 0, soff = 0;
        for (int j = 0; j < n; j++) {
            int O = r[j].O, I = r[j].I;
            quantize_from_disk(m, r[j].name, m->stream_q + qoff, m->stream_qs + soff, O, I, 0);
            mat_reset_storage(r[j].mat);
            r[j].mat->q = m->stream_q + qoff; r[j].mat->qs = m->stream_qs + soff;
            r[j].mat->O = O; r[j].mat->I = I; r[j].mat->fmt = WF_I8;
            qoff += (int64_t)O*I; soff += O;
        }
        return;
    }
    int64_t off = 0;
    for (int j = 0; j < n; j++) {
        st_read_f32(&m->S, r[j].name, m->stream_buf + off, 0);  /* drop=0: la page cache aiuta */
        mat_reset_storage(r[j].mat);
        r[j].mat->f = m->stream_buf + off;
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
static int     g_micro = 0;             /* attivato da MICRO=1 (engine_main) o dai test */
static int     g_micro_drop = 1;        /* MICRO_DROP=0 -> lascia vivere la page cache */
static int64_t g_micro_chunk = 4 << 20; /* byte f32 dello scratch di streaming */

/* y[S,O] = x[S,I] @ W^T leggendo W dal disco a blocchi di righe; installata in
 * g_mat_stream_fn cosi' mat_apply (nn.h) la usa per le Mat con sh!=NULL.
 * Scratch statico che cresce e basta: contratto di chiamata SERIALE, come
 * matmul_q_s. Bit-identica al percorso f32 residente (stesse righe, stesso
 * dot_f32). */
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
        #pragma omp parallel for schedule(static)
        for (int o = 0; o < r; o++)
            for (int s = 0; s < S; s++)
                y[(int64_t)s*O + o0 + o] = dot_f32(x + (int64_t)s*I, buf + (int64_t)o*I, I);
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
    g_mat_stream_fn = mat_stream;
    mat_stream_init(m, &m->lm_head,
                    m->lm_tied ? "model.embed_tokens.weight" : "lm_head.weight",
                    c->vocab, c->hidden);
    for (int i = 0; i < c->n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++) mat_stream_init(m, r[j].mat, r[j].name, r[j].O, r[j].I);
    }
    m->n_resident = 0;                    /* verita': zero layer residenti */
    fprintf(stderr, "[" ENGINE_TAG "] micro-RSS: 0 pesi residenti, matmul streamato a blocchi da %lld MB, page cache %s\n",
            (long long)(g_micro_chunk >> 20), g_micro_drop ? "scartata (MICRO_DROP=0 per tenerla)" : "attiva");
#else
    (void)m;
    fprintf(stderr, "[" ENGINE_TAG "] MICRO=1 non supportato da questo motore\n");
    exit(1);
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
    int64_t vd = m->qbits ? (int64_t)c->vocab*D + (int64_t)c->vocab*4
                          : (int64_t)c->vocab*D*4;
    int64_t fixed = vd + (int64_t)D*4;                          /* + final_norm */
    if (!m->lm_tied) fixed += vd;
    fixed += (int64_t)c->n_layers * 8 * D * 4;                  /* norme/vettori: stima larga */
    fixed += fixed_bytes(m, ctx_hint > 0 ? ctx_hint : 4096);    /* hook: KV, PLE... */
    /* scratch di streaming: int8 con QBITS=8 (layer_stream_in quantizza), f32 altrimenti */
    int64_t scratch = (m->qbits == 8) ? max_lb/4 + max_lb/64 : max_lb;
    int64_t used = fixed + scratch;
    int R = 0;
    for (; R < c->n_layers; R++) {
        int64_t lb = layer_f32_bytes(m, R);
        if (m->qbits == 8) lb = lb/4 + lb/64;                   /* int8 + scale */
        else if (m->qbits == 4) lb = lb/8 + lb/32;              /* int4 + scale di gruppo (gs=32) */
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
    for (int i = m->n_resident; i < c->n_layers; i++) {
        int64_t b = layer_f32_bytes(m, i); if (b > smax) smax = b;
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        int64_t rows = 0; for (int j = 0; j < n; j++) rows += r[j].O;
        if (rows > rmax) rmax = rows;
    }
    if (m->qbits == 8) {
        m->stream_q = balloc(smax/4, "scratch stream int8");  /* 1 byte per elemento f32 */
        m->stream_qs = falloc(rmax);
    } else {
        m->stream_buf = falloc(smax/4);
    }
}

/* budget_bytes==0 -> tutto residente (comportamento classico) */
static void model_init_ex(Model *m, const char *snap, int qbits, int64_t budget_bytes, int ctx_hint) {
    memset(m, 0, sizeof(*m));
    m->qbits = qbits;
    /* GGUF: l'indice va costruito PRIMA del config (i metadati SONO il config) */
    if (g_gguf) gguf_index(&m->S, &g_gguf_meta, g_gguf);
    load_cfg(&m->c, snap);
    if (!g_gguf) st_init(&m->S, snap);
    Cfg *c = &m->c;
    double t0 = now_s();
    int D = c->hidden;
    /* final norm: model.norm.weight (standard); se assente (LFM2: token_embd_norm) lo fornira" load_small */
    if (st_has(&m->S, "model.norm.weight"))
        m->final_norm = load_t(m, "model.norm.weight", D);
    m->lm_tied = c->tie_emb || !st_has(&m->S, "lm_head.weight");
    m->L = calloc(c->n_layers, sizeof(Layer));
    /* 1) parte piccola SEMPRE residente (hook: norme, vettori, stati, PLE...) */
    load_small(m);
    /* micro-RSS: nessun peso residente, embed compreso (gather per riga nello
     * step del motore); ogni Mat diventa un descrittore streamato. */
    if (g_micro) { model_init_micro(m); m->load_s = now_s() - t0; return; }
    /* QBITS!=0 copre anche l'embed, ma SEMPRE a int8 (anche con QBITS=4):
     * l'lm_head e' il GEMV piu' sensibile alla quantizzazione e l'int4 li'
     * risparmierebbe poco rispetto alle matrici dei layer */
    if (m->qbits > 0 || getenv("EMBED_Q8")) load_embed_q8(m);  /* qbits=-1 (native): f32 embed */
    else m->embed = load_t(m, "model.embed_tokens.weight", (int64_t)c->vocab*D);
    if (m->lm_tied) {
        mat_reset_storage(&m->lm_head);
        m->lm_head.O = c->vocab; m->lm_head.I = D;
        if (m->qbits == 4 && m->embed_q && D <= 2048) {
            /* lm_head separato in INT4: il GEMV del logit e' ~43% del traffico
             * per-token in decode (262MB→131). Lookup embed resta int8. */
            int64_t V = c->vocab, rb = (D+1)/2;
            m->lm_head.q4 = balloc(V*rb, "lm_head i4");
            m->lm_head.qs = falloc(V);
            m->lm_head.fmt = WF_I4;
            #pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < V; v++) {
                float row[2048];
                const int8_t *er = m->embed_q + v*D;
                float es = m->embed_qs[v];
                for (int i = 0; i < D; i++) row[i] = er[i] * es;
                pack_int4(row, m->lm_head.q4 + v*rb, m->lm_head.qs + v, 1, D);
            }
        } else {
            m->lm_head.f = m->embed; m->lm_head.q = m->embed_q; m->lm_head.qs = m->embed_qs;
            m->lm_head.fmt = m->embed_q ? WF_I8 : WF_F32;
        }
    } else {
        load_mat_bits(m, &m->lm_head, "lm_head.weight", c->vocab, D, m->qbits);
    }
    /* 2) budget -> quanti layer di matrici stanno residenti */
    m->n_resident = budget_bytes > 0 ? budget_resident(m, budget_bytes, ctx_hint)
                                     : c->n_layers;
    /* 3) matrici: residenti (QBITS onorato) o streamate (dims impostate, f=NULL) */
    for (int i = 0; i < c->n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++) {
            if (i < m->n_resident) load_mat(m, r[j].mat, r[j].name, r[j].O, r[j].I);
            else {
                st_expect(&m->S, r[j].name, (int64_t)r[j].O*r[j].I);
                mat_reset_storage(r[j].mat);
                r[j].mat->O = r[j].O; r[j].mat->I = r[j].I;
            }
        }
    }
    if (m->n_resident < c->n_layers) stream_scratch_alloc(m);
    m->load_s = now_s() - t0;
}

static void model_init(Model *m, const char *snap, int qbits) {
    model_init_ex(m, snap, qbits, 0, 0);
}

/* KV_BITS=8: KV-cache int8 con scala per (testa_kv, posizione). Default 0
 * (f32): la numerica di REF non cambia mai in silenzio. */
static int g_kv_bits = 0;

/* riga id dell'embedding -> dst[D] moltiplicata per scale (gemma passa
 * sqrt(D), qwen 1): f32 residente, int8 dequant, oppure micro-RSS (lettura
 * della sola riga dal disco; drop=0, le righe calde sono minuscole). Era
 * open-coded in tre punti fra i due motori. */
static void embed_row(Model *m, int id, float scale, float *dst) {
    int D = m->c.hidden;
    if (m->embed) {
        const float *er = m->embed + (int64_t)id*D;
        if (scale == 1.f) memcpy(dst, er, D*sizeof(float));
        else for (int i = 0; i < D; i++) dst[i] = er[i] * scale;
    } else if (m->embed_q) {
        const int8_t *er = m->embed_q + (int64_t)id*D;
        float es = m->embed_qs[id] * scale;
        for (int i = 0; i < D; i++) dst[i] = er[i] * es;
    } else {
        st_read_slice_f32(&m->S, "model.embed_tokens.weight", (int64_t)id*D, D, dst, 0);
        if (scale != 1.f) for (int i = 0; i < D; i++) dst[i] *= scale;
    }
}

#endif /* RT_MODEL_LOAD_H */
