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

/* QBITS=4 + Q4FMT=r4: rows read from disk in ~4 MB chunks (never the whole
 * f32 matrix — the lm_head of a 130k-vocab model would be 800 MB) and
 * packed into Q4R4 4-row blocks in parallel (nn/quant.c, hw/hw_q4r4.h). */
static void load_mat_q4r4(Model *m, Mat *w, const char *name, int O, int I) {
    int nb = I / 32, O4 = (O + 3) / 4;
    w->q4 = balloc((int64_t)O4*nb*64, name);
    w->s16 = balloc((int64_t)O4*nb*4*sizeof(uint16_t), name);
    /* pre-packed container (SAVE_PACKED): U8 blocks + F16 "<name>.s16" -> raw read */
    char sn[256]; snprintf(sn, sizeof sn, "%s.s16", name);
    if (st_has(&m->S, sn)) {
        if (st_nbytes(&m->S, name) != (int64_t)O4*nb*64 || st_nbytes(&m->S, sn) != (int64_t)O4*nb*4*2) {
            fprintf(stderr, "[" ENGINE_TAG "] %s: packed Q4R4 size mismatch (O=%d I=%d)\n", name, O, I); exit(1);
        }
        st_read_raw(&m->S, name, w->q4, 0);
        st_read_raw(&m->S, sn, w->s16, 0);
        w->O = O; w->I = I; w->fmt = WF_Q4R4;
        return;
    }
    st_expect(&m->S, name, (int64_t)O*I);
    int rows = (int)(((4 << 20) / ((int64_t)I*4)) & ~3); if (rows < 4) rows = 4;
    float *chunk = falloc((int64_t)rows*I);
    for (int o0 = 0; o0 < O; o0 += rows) {
        int rr = O - o0 < rows ? O - o0 : rows;
        st_read_slice_f32(&m->S, name, (int64_t)o0*I, (int64_t)rr*I, chunk, 0);
        #pragma omp parallel for schedule(static)
        for (int b = 0; b < (rr + 3) / 4; b++) {
            int nr = rr - b*4 < 4 ? rr - b*4 : 4, ob = o0/4 + b;
            moty_pack_q4r4_block(chunk + (int64_t)b*4*I, nr, I, w->q4 + (int64_t)ob*nb*64, w->s16 + (int64_t)ob*nb*4);
        }
    }
    free(chunk);
    w->O = O; w->I = I; w->fmt = WF_Q4R4;
}

static void quantize_from_disk(Model *m, const char *name, int8_t *q, float *qs, int64_t N, int I, int rows);

static void load_mat_bits(Model *m, Mat *w, const char *name, int O, int I, int bits) {
    mat_reset_storage(w);            /* fmt = WF_F32 di default */
    mat_reset_storage(w);
    w->O = O; w->I = I;
    if (bits == 4 && g_q4fmt && I % 32 == 0) { load_mat_q4r4(m, w, name, O, I); return; }
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

/* embed int8 per riga (QBITS=8): la tabella non passa MAI intera per la RAM */
static int g_embed_chunk_rows = 0;      /* 0 = auto; i test lo stringono */
static void load_embed_q8(Model *m) {
    Cfg *c = &m->c; int D = c->hidden; int64_t V = c->vocab;
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
    load_small(m);
    /* micro-RSS: nessun peso residente, embed compreso (gather per riga nello
     * step del motore); ogni Mat diventa un descrittore streamato. */
    if (g_micro) { model_init_micro(m); m->base.load_s = now_s() - t0; return; }
    /* QBITS!=0 copre anche l'embed, ma SEMPRE a int8 (anche con QBITS=4):
     * l'lm_head e' il GEMV piu' sensibile alla quantizzazione e l'int4 li'
     * risparmierebbe poco rispetto alle matrici dei layer */
    /* EMBED=disk: no resident table, embed_row gathers the row from the file
     * (the micro-RSS branch). Only valid when nothing else reads the table:
     * a tied lm_head must then be packed separately (QBITS=4 Q4R4). */
    int head_q4r4 = m->base.lm_tied && m->base.qbits == 4 && g_q4fmt && D % 32 == 0;
    if (g_embed_disk && m->base.lm_tied && !head_q4r4) {
        fprintf(stderr, "[" ENGINE_TAG "] EMBED=disk con lm_head legato richiede QBITS=4 Q4FMT=r4\n"); exit(1);
    }
    if (g_embed_disk) st_expect(&m->S, "model.embed_tokens.weight", (int64_t)c->vocab*D);
    else if (m->base.qbits > 0 || getenv("EMBED_Q8")) load_embed_q8(m);  /* qbits=-1 (native): f32 embed */
    else m->base.embed = load_t(m, "model.embed_tokens.weight", (int64_t)c->vocab*D);
    if (m->base.lm_tied) {
        mat_reset_storage(&m->base.lm_head);
        m->base.lm_head.O = c->vocab; m->base.lm_head.I = D;
        if (head_q4r4) {
            /* tied head in Q4R4 packed from the original rows (not from the
             * int8 table: no double quantization); a container stores it
             * pre-packed as lm_head.weight */
            load_mat_q4r4(m, &m->base.lm_head, st_has(&m->S, "lm_head.weight.s16") ? "lm_head.weight"
                                                   : "model.embed_tokens.weight", c->vocab, D);
        } else if (m->base.qbits == 4 && m->base.embed_q && D <= 2048) {
            /* lm_head separato in INT4: il GEMV del logit e' ~43% del traffico
             * per-token in decode (262MB→131). Lookup embed resta int8. */
            int64_t V = c->vocab, rb = (D+1)/2;
            m->base.lm_head.q4 = balloc(V*rb, "lm_head i4");
            m->base.lm_head.qs = falloc(V);
            m->base.lm_head.fmt = WF_I4;
            #pragma omp parallel for schedule(static)
            for (int64_t v = 0; v < V; v++) {
                float row[2048];
                const int8_t *er = m->base.embed_q + v*D;
                float es = m->base.embed_qs[v];
                for (int i = 0; i < D; i++) row[i] = er[i] * es;
                pack_int4(row, m->base.lm_head.q4 + v*rb, m->base.lm_head.qs + v, 1, D);
            }
        } else {
            m->base.lm_head.f = m->base.embed; m->base.lm_head.q = m->base.embed_q; m->base.lm_head.qs = m->base.embed_qs;
            m->base.lm_head.fmt = m->base.embed_q ? WF_I8 : WF_F32;
        }
    } else {
        load_mat_bits(m, &m->base.lm_head, "lm_head.weight", c->vocab, D, m->base.qbits);
    }
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
    m->base.load_s = now_s() - t0;
}

/* ---------- SAVE_PACKED=<dir>: write a pre-packed Q4R4 container ----------
 * A safetensors snapshot (moty container convention, cf. olmoe/glm "name.qs"):
 * every resident WF_Q4R4 matrix as U8 "<name>" (its 4-row blocks) + F16
 * "<name>.s16"; the lm_head as "lm_head.weight" (+.s16), also when tied; all
 * other tensors copied byte for byte with their dtype (flattened shape).
 * config/tokenizer files are copied. Loading it with QBITS=4 skips the
 * bf16 read + quantization (load_mat_q4r4 raw path). */
#ifdef _WIN32
#include <direct.h>
#define rt_mkdir(p) _mkdir(p)
#else
#include <sys/stat.h>
#define rt_mkdir(p) mkdir(p, 0755)
#endif
static int pk_name_is(Model *m, const char *name, Mat **out) {
    if (!strcmp(name, "lm_head.weight") && m->base.lm_head.fmt == WF_Q4R4) { *out = &m->base.lm_head; return 1; }
    for (int i = 0; i < m->c.n_layers; i++) {
        MatRef r[MAX_LAYER_MATS]; int n = layer_matrefs(m, i, r);
        for (int j = 0; j < n; j++) if (!strcmp(r[j].name, name) && r[j].mat->fmt == WF_Q4R4) { *out = r[j].mat; return 1; }
    }
    return 0;
}
static void model_save_packed(Model *m, const char *snap, const char *dir) {
    if (g_gguf || m->base.qbits != 4 || !g_q4fmt || m->base.n_resident < m->c.n_layers) {
        fprintf(stderr, "[" ENGINE_TAG "] SAVE_PACKED: needs SNAP=, QBITS=4 Q4FMT=r4, all layers resident\n"); exit(1);
    }
    rt_mkdir(dir);
    typedef struct { char name[256]; const char *dt; int64_t n, bytes; const void *mem; int src; } PkT;
    int cap = 2 * m->S.n + 4, nt = 0;               /* a packed matrix -> 2 entries */
    PkT *t = calloc(cap, sizeof(PkT));
    static const char *dtn[4] = { "BF16", "F16", "F32", "U8" };
    int head_done = 0;
    for (int i = 0; i < m->S.n; i++) {
        st_tensor *st = &m->S.t[i]; Mat *w;
        if (pk_name_is(m, st->name, &w)) {
            int64_t nb = w->I / 32, O4 = (w->O + 3) / 4;
            snprintf(t[nt].name, 256, "%s", st->name); t[nt].dt = "U8"; t[nt].n = t[nt].bytes = O4*nb*64; t[nt].mem = w->q4; t[nt].src = -1; nt++;
            snprintf(t[nt].name, 256, "%s.s16", st->name); t[nt].dt = "F16"; t[nt].n = O4*nb*4; t[nt].bytes = O4*nb*8; t[nt].mem = w->s16; t[nt].src = -1; nt++;
            if (!strcmp(st->name, "lm_head.weight")) head_done = 1;
        } else {
            if (st->dtype > 3) { fprintf(stderr, "SAVE_PACKED: %s: block dtype unsupported\n", st->name); exit(1); }
            snprintf(t[nt].name, 256, "%s", st->name); t[nt].dt = dtn[st->dtype]; t[nt].n = st->numel;
            t[nt].bytes = st->nbytes; t[nt].mem = NULL; t[nt].src = i; nt++;
        }
    }
    if (!head_done && m->base.lm_head.fmt == WF_Q4R4) {           /* tied head */
        Mat *w = &m->base.lm_head; int64_t nb = w->I / 32, O4 = (w->O + 3) / 4;
        snprintf(t[nt].name, 256, "lm_head.weight"); t[nt].dt = "U8"; t[nt].n = t[nt].bytes = O4*nb*64; t[nt].mem = w->q4; t[nt].src = -1; nt++;
        snprintf(t[nt].name, 256, "lm_head.weight.s16"); t[nt].dt = "F16"; t[nt].n = O4*nb*4; t[nt].bytes = O4*nb*8; t[nt].mem = w->s16; t[nt].src = -1; nt++;
    }
    int64_t hcap = 256 + (int64_t)nt * 400; char *hdr = malloc(hcap); int64_t hl = 0, off = 0;
    hl += snprintf(hdr + hl, hcap - hl, "{\"__metadata__\":{\"format\":\"moty-q4r4\"}");
    for (int i = 0; i < nt; i++) {
        hl += snprintf(hdr + hl, hcap - hl, ",\"%s\":{\"dtype\":\"%s\",\"shape\":[%lld],\"data_offsets\":[%lld,%lld]}",
                       t[i].name, t[i].dt, (long long)t[i].n, (long long)off, (long long)(off + t[i].bytes));
        off += t[i].bytes;
    }
    hl += snprintf(hdr + hl, hcap - hl, "}");
    while (hl % 8) hdr[hl++] = ' ';                                /* data 8-byte aligned */
    char path[2048]; snprintf(path, sizeof path, "%s/model.safetensors", dir);
    FILE *f = fopen(path, "wb"); if (!f) { perror(path); exit(1); }
    uint64_t h64 = (uint64_t)hl;
    fwrite(&h64, 8, 1, f); fwrite(hdr, 1, hl, f);
    void *buf = NULL; int64_t bcap = 0;
    for (int i = 0; i < nt; i++) {
        if (t[i].mem) { if (fwrite(t[i].mem, 1, t[i].bytes, f) != (size_t)t[i].bytes) { perror("write"); exit(1); } continue; }
        grow(&buf, &bcap, t[i].bytes, 1, "save copy");
        st_read_raw(&m->S, m->S.t[t[i].src].name, buf, 1);
        if (fwrite(buf, 1, t[i].bytes, f) != (size_t)t[i].bytes) { perror("write"); exit(1); }
    }
    fclose(f); free(buf); free(hdr);
    static const char *aux[] = { "config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json" };
    for (size_t k = 0; k < sizeof aux / sizeof aux[0]; k++) {
        char src[2048], dst[2048]; snprintf(src, sizeof src, "%s/%s", snap, aux[k]); snprintf(dst, sizeof dst, "%s/%s", dir, aux[k]);
        long n; char *d = NULL; FILE *fs = fopen(src, "rb"); if (!fs) continue; fclose(fs);
        d = slurp_file(src, &n);
        FILE *fd = fopen(dst, "wb"); if (!fd) { perror(dst); exit(1); } fwrite(d, 1, n, fd); fclose(fd); free(d);
    }
    fprintf(stderr, "[" ENGINE_TAG "] SAVE_PACKED: %d tensors, %.1f MB -> %s\n", nt, off / 1048576.0, path);
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
