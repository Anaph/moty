#ifndef MOTY_QWEN_Q41_STREAM_H
#define MOTY_QWEN_Q41_STREAM_H

/* Private to the Qwen engine. Included after model loading helpers so
 * the streaming path can share layer_matrefs(), load_mat() and fixed_bytes().
 * A queue slot owns its weights until the inference thread submits the next
 * layer for that slot. The dispatch groups publish completed reads. */

#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <pthread.h>
#include <sys/mman.h>
#endif

#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
static void qwen_q41_panel_close(QwenQ41Input *in) {
    if (!in->panel_bytes) return;
    close(in->panel_fd);
    in->panel_bytes = 0;
}

static void qwen_q41_panels_open(Model *m) {
    const char *packed = moty_getenv("MOTY_Q41_PACKED");
    if (!m->q41_window || !packed || !moty_q41_uses_fp16() ||
        (strcmp(packed,"stream") && strcmp(packed,"stream-all"))) return;
    int panels = 0;
    for (int i = m->base.n_resident; i < m->c.n_layers; i++) {
        QwenQ41Input *in = m->q41_layers[i].ffn;
        for (int j = 0; j < 3; j++) {
            if (in[j].resident) continue;
            Q41CacheHeader disk;
            int fd = st_q41_cache_open(&m->S,in[j].tensor,in[j].mat->I,in[j].mat->O,&disk);
            if (fd < 0) continue;
            in[j].panel_fd = fd;
            in[j].panel_bytes = disk.bytes;
            in[j].panel_hash = disk.checksum;
            in[j].panel_version = disk.version;
        }
        /* A gate/up pair uses one layout so it can keep the fused kernel. */
        if (!!in[0].panel_bytes != !!in[1].panel_bytes) {
            qwen_q41_panel_close(&in[0]);
            qwen_q41_panel_close(&in[1]);
        }
        for (int j = 0; j < 3; j++) panels += in[j].panel_bytes != 0;
    }
    if (panels)
        fprintf(stderr,"[qwen] native Q4_1 budget: %d panel cache entries for decode\n",panels);
}
#endif

/* Pocket keeps attention and the tied head resident. Only the cold FFN
 * projections use the raw GGUF windows. */
static void qwen_q41_stream_open(Model *m, int64_t budget, int ctx) {
    Cfg *c = &m->c;
    if (c->hybrid || m->base.lm_head.fmt != WF_Q4_1 || !m->base.lm_tied)
        moty_fail_code(MOTY_FAIL_FORMAT, "Native Pocket streaming requires dense layers and a tied Q4_1 head");
    if (ctx <= 0) ctx = 4096;
    if (c->max_pos > 0 && ctx > c->max_pos) ctx = c->max_pos;
    Mat *head = &m->base.lm_head;
    int64_t fixed = fixed_bytes(m, ctx) + (int64_t)c->n_layers * 8 * c->hidden * sizeof(float);
    if (head->gs == Q41_INDEXED_GS) fixed += q41_indexed_bytes(head->I,head->O);
    else if (head->gs == Q41_PACKED_GS) fixed += (int64_t)(head->O/4)*(head->I/64)*Q41_PANEL_BYTES;
    else fixed += st_nbytes(&m->S,"model.embed_tokens.weight");
    int check_group64 = q41_group64_enabled();
    m->q41_layers = bzalloc((int64_t)c->n_layers*sizeof(*m->q41_layers), "Pocket FFN windows");
    int64_t total_ffn = 0, window = 0;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    const char *packed = moty_getenv("MOTY_Q41_PACKED");
    const char *prefill_attention = moty_getenv("MOTY_Q41_PREFILL_ATTN");
    int64_t attention_window = 0;
    if (packed && !strcmp(packed,"stream-all") && g_st_map && moty_q41_uses_fp16() &&
        (!prefill_attention || strcmp(prefill_attention,"panel"))) {
        m->q41_attention = bzalloc((int64_t)c->n_layers*sizeof(*m->q41_attention),
                                    "Pocket attention tensor references");
        m->q41_attention_group64 = check_group64 ? 64 : 0;
    }
#endif
    for (int i = 0; i < c->n_layers; i++) {
        MatRef refs[MAX_LAYER_MATS];
        int n = layer_matrefs(m,i,refs);
        QwenQ41Layer *layer = &m->q41_layers[i];
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
        int64_t attention_bytes = 0;
#endif
        /* layer_matrefs lists the three FFN projections after attention. */
        for (int j = 0; j < n; j++) {
            MatRef *r = &refs[j];
            const st_tensor *t = st_expect(&m->S,r->name,(int64_t)r->O*r->I);
            if (j < n-3) {
                load_mat(m,r->mat,r->name,r->O,r->I);
                if (r->mat->fmt == WF_F32) fixed += t->numel*(int64_t)sizeof(float);
                else if (r->mat->gs == Q41_PACKED_GS) {
                    fixed += (int64_t)(r->O/4)*(r->I/64)*Q41_PANEL_BYTES;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
                    if (m->q41_attention) {
                        m->q41_attention[i][j] = t;
                        attention_bytes += scr_al(t->nbytes);
                    }
#endif
                } else fixed += t->nbytes;
                continue;
            }
            if (t->dtype != ST_Q4_1 || r->I%Q4_1_ELEMS ||
                t->nbytes != (int64_t)r->O*(r->I/Q4_1_ELEMS)*Q4_1_BYTES)
                moty_fail_code(MOTY_FAIL_FORMAT,"Native Pocket streaming requires original Q4_1 blocks in %s",r->name);
            mat_reset_storage(r->mat);
            r->mat->fmt = WF_Q4_1; r->mat->O = r->O; r->mat->I = r->I;
            layer->ffn[j-(n-3)] = (QwenQ41Input){.mat=r->mat,.tensor=t,.check_group64=check_group64};
            layer->bytes += scr_al(t->nbytes);
        }
        total_ffn += layer->bytes;
        if (layer->bytes > window) window = layer->bytes;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
        if (attention_bytes > attention_window) attention_window = attention_bytes;
#endif
    }
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    if (attention_window) {
        m->q41_attention_raw = balloc(attention_window,"Pocket prefill attention window");
        fixed += attention_window;
    }
#endif
    int64_t available = budget - fixed;
    int resident = 0;
    int prefix = c->n_layers;
    const char *placement = moty_getenv("MOTY_Q41_RESIDENT_PREFIX");
    if (placement && *placement) {
        char *end;
        long count = strtol(placement,&end,10);
        if (!*end && count >= 0 && count < prefix) prefix = (int)count;
    }
    if (total_ffn <= available) resident = c->n_layers;
    else {
        m->q41_window_count = 1;
#ifdef __APPLE__
        const char *windows = moty_getenv("MOTY_Q41_WINDOWS");
        int limit = windows && !strcmp(windows,"4") ? Q41_MAX_WINDOWS : Q41_WINDOWS;
        while (m->q41_window_count < limit && available >= (m->q41_window_count+1)*window)
            m->q41_window_count++;
#endif
        m->q41_window_bytes = window;
        available -= m->q41_window_count*window;
        while (resident < prefix && m->q41_layers[resident].bytes <= available)
            available -= m->q41_layers[resident++].bytes;
        m->q41_window = balloc(m->q41_window_count*window,"Pocket raw Q4_1 FFN windows");
    }
    m->base.n_resident = resident;
    for (int i = 0; i < resident; i++) {
        MatRef refs[MAX_LAYER_MATS];
        int n = layer_matrefs(m,i,refs);
        for (int j = n-3; j < n; j++)
            load_mat(m,refs[j].mat,refs[j].name,refs[j].O,refs[j].I);
    }
    int resident_down = 0;
    if (prefix < c->n_layers) {
        const char *order = moty_getenv("MOTY_Q41_RESIDENT_DOWN");
        int stride = order && !strcmp(order,"slot") && m->q41_window_count > 1
                   ? m->q41_window_count : 1;
        /* Slot order can leave a complete window's down pages untouched. */
        for (int slot = 0; slot < stride; slot++) {
            for (int i = resident+slot; i < c->n_layers; i += stride) {
                QwenQ41Input *down = &m->q41_layers[i].ffn[2];
                int64_t bytes = scr_al(down->tensor->nbytes);
                if (bytes > available) break;
                Mat *w = down->mat;
                load_mat(m,w,down->tensor->name,w->O,w->I);
                available -= bytes;
                down->resident = 1;
                resident_down++;
            }
        }
    }
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    qwen_q41_panels_open(m);
    const char *prefill_panels = moty_getenv("MOTY_Q41_STREAM_PREFILL");
    if (prefill_panels && *prefill_panels && m->q41_window) {
        char *end;
        long count = strtol(prefill_panels,&end,10);
        if (!*end && count > 1 && count <= INT_MAX) m->q41_prefill_panels = (int)count;
    }
#endif
    fprintf(stderr,"[qwen] native Q4_1 budget: %d/%d FFN layers resident, %d raw windows (%.2f MiB total)\n",
            resident,c->n_layers,m->q41_window_count,(double)(m->q41_window_count*window)/1048576.);
    if (resident_down)
        fprintf(stderr,"[qwen] native Q4_1 budget: %d additional resident down projections\n",resident_down);
}

static int qwen_q41_read_bytes(int fd, uint8_t *dst, int64_t offset, int64_t bytes) {
    int64_t done = 0;
    while (done < bytes) {
        ssize_t n = pread(fd,dst+done,(size_t)(bytes-done),offset+done);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) return n == 0 ? EIO : errno;
        done += n;
    }
    return 0;
}

#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
static void qwen_q41_file_window_close(QwenQ41FileWindow *window) {
    if (window->address && munmap(window->address,window->capacity))
        moty_fail_code(MOTY_FAIL_IO,"Cannot release Q4_1 file window");
    window->address = NULL;
    window->capacity = 0;
    window->writable = 0;
}

/* Each slot owns its address range. Replace its backing only after the
 * caller has finished the previous projection and submitted the next read. */
static uint8_t *qwen_q41_file_window_map(QwenQ41FileWindow *window, size_t bytes,
                                         int fd, size_t file_bytes) {
    size_t page = window->page_size;
    if (file_bytes > bytes) bytes = file_bytes;
    if (bytes > SIZE_MAX-(page-1)) return NULL;
    size_t capacity = (bytes+page-1)/page*page;
    int anonymous = fd < 0;
    if (capacity > window->capacity) {
        qwen_q41_file_window_close(window);
        int protection = anonymous ? PROT_READ|PROT_WRITE : PROT_NONE;
        void *base = mmap(NULL,capacity,protection,MAP_PRIVATE|MAP_ANON,-1,0);
        if (base == MAP_FAILED) return NULL;
        window->address = base;
        window->capacity = capacity;
        window->writable = anonymous;
    }
    if (anonymous && window->writable) return window->address;
    void *view = mmap(window->address,anonymous ? window->capacity : file_bytes,
                      PROT_READ|(anonymous ? PROT_WRITE : 0),
                      MAP_FIXED|MAP_PRIVATE|(anonymous ? MAP_ANON : 0),fd,0);
    if (view == MAP_FAILED) {
        qwen_q41_file_window_close(window);
        return NULL;
    }
    window->writable = anonymous;
    return view;
}

static void qwen_q41_prepare_copy(QwenQ41Read *job) {
    if (!job->file_window) return;
    uint8_t *scratch = qwen_q41_file_window_map(job->file_window,job->tensor->nbytes,-1,0);
    job->dst = scratch ? scratch : job->copy_dst;
}
#endif

static void qwen_q41_read_raw(QwenQ41Read *job) {
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    qwen_q41_prepare_copy(job);
#endif
    const st_tensor *t = job->tensor;
    int fail = qwen_q41_read_bytes(t->fd,job->dst,t->off,t->nbytes);
    if (fail) job->error = fail;
}

static void qwen_q41_read(void *arg) {
    QwenQ41Read *job = arg;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    /* Check the first payload already in the decode window. A failed
     * sidecar read falls back within the same raw-sized tensor slot. */
    const QwenQ41Input *panel = job->panel;
    if (job->map_panel && panel->panel_verified && panel->panel_fd >= 0) {
        size_t bytes = job->tensor->nbytes;
        size_t length = sizeof(Q41CacheHeader)+panel->panel_bytes;
        struct stat info;
        uint8_t *view = NULL;
        if (!fstat(panel->panel_fd,&info) && info.st_size >= 0 && (uint64_t)info.st_size == length)
            view = qwen_q41_file_window_map(job->file_window,bytes,panel->panel_fd,length);
        if (view) {
            const volatile uint8_t *pages = view;
            for (size_t i = 0; i < length; i += job->file_window->page_size) (void)pages[i];
            job->dst = view+sizeof(Q41CacheHeader);
            job->packed = 1;
            job->file_window->mapped_panel = 1;
            return;
        }
    }
    if (panel) {
        qwen_q41_prepare_copy(job);
        if (!qwen_q41_read_bytes(panel->panel_fd,job->dst,sizeof(Q41CacheHeader),panel->panel_bytes) &&
            (panel->panel_verified ||
             q41_cache_hash_payload(panel->panel_version,job->dst,panel->panel_bytes) == panel->panel_hash)) {
            job->packed = 1;
            return;
        }
        job->invalid_panel = 1;
    }
#endif
    qwen_q41_read_raw(job);
}

static int qwen_q41_use_panels(const Model *m, int S) {
    return S == 1 || S <= m->q41_prefill_panels;
}

static void qwen_q41_read_jobs(QwenQ41Read jobs[3], const QwenQ41Layer *layer,
                               uint8_t *dst, int use_panels) {
    for (int j = 0; j < 3; j++) {
        const QwenQ41Input *in = &layer->ffn[j];
        if (in->resident) {
            jobs[j] = (QwenQ41Read){0};
            continue;
        }
        const st_tensor *t = in->tensor;
        QwenQ41Read job = {.tensor=t,.dst=dst};
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
        if (use_panels && in->panel_bytes) job.panel = in;
#else
        (void)use_panels;
#endif
        jobs[j] = job;
        dst += scr_al(t->nbytes);
    }
}

static void qwen_q41_stream_bind(Model *m, int li, QwenQ41Read jobs[3],
                                 int first, int end) {
    QwenQ41Layer *layer = &m->q41_layers[li];
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
    if (first == 0 && jobs[0].packed != jobs[1].packed) {
        int packed_projection = jobs[0].packed ? 0 : 1;
        QwenQ41Read *job = &jobs[packed_projection];
        job->packed = 0;
        qwen_q41_read_raw(job);
    }
#endif
    for (int j = first; j < end; j++) {
        if (!jobs[j].tensor) continue;
        QwenQ41Input *input = &layer->ffn[j];
        if (jobs[j].error)
            moty_fail_code(MOTY_FAIL_IO,"Cannot read native Q4_1 tensor %s: %s",
                           input->tensor->name,strerror(jobs[j].error));
        Mat *w = input->mat;
        w->q4 = jobs[j].dst;
        w->borrowed = 1;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
        if (jobs[j].packed) {
            if (!input->panel_verified) input->panel_verified = 1;
            w->gs = Q41_PACKED_GS;
            continue;
        }
        if (jobs[j].panel) {
            qwen_q41_panel_close(input);
            if (jobs[j].invalid_panel) {
                /* This step already uses the raw block. Repair the sidecar
                 * for later opens without changing the current window. */
                Q41CacheHeader disk;
                int fd = st_q41_cache_open_format(&m->S,input->tensor,w->I,w->O,&disk,0,1);
                if (fd >= 0) close(fd);
            }
        }
        if (w->gs == Q41_PACKED_GS) w->gs = input->raw_gs;
#endif
        if (input->check_group64) {
            if (q41_same_params_per_pair(w->q4,w->I,input->tensor->nbytes)) w->gs = 64;
#if defined(__APPLE__) && defined(__aarch64__) && defined(__ARM_NEON)
            input->raw_gs = w->gs;
#endif
            input->check_group64 = 0;
        }
    }
}

static void qwen_q41_stream_load(Model *m, int li, int S) {
    QwenQ41Read jobs[3];
    qwen_q41_read_jobs(jobs,&m->q41_layers[li],m->q41_window,qwen_q41_use_panels(m,S));
    for (int j = 0; j < 3; j++)
        if (jobs[j].tensor) qwen_q41_read(&jobs[j]);
    qwen_q41_stream_bind(m,li,jobs,0,3);
}

#ifdef __APPLE__
typedef struct {
    dispatch_group_t ready, down_ready;
    QwenQ41Read jobs[Q41_ATTN_MATRICES];
#if defined(__aarch64__) && defined(__ARM_NEON)
    QwenQ41FileWindow file_windows[Q41_FFN_MATRICES];
#endif
} QwenQ41Batch;

typedef struct {
    QwenQ41Read *job;
    dispatch_group_t ready;
} QwenQ41Task;

enum { Q41_QUEUE_SIZE = 16 };
_Static_assert(Q41_QUEUE_SIZE >= Q41_MAX_WINDOWS*Q41_FFN_MATRICES+Q41_ATTN_MATRICES,
               "Q4_1 reader queue too small");

struct QwenQ41Queue {
    pthread_mutex_t mutex;
    pthread_cond_t work;
    pthread_t readers[Q41_READERS];
    QwenQ41Batch batch[Q41_MAX_WINDOWS];
#if defined(__aarch64__) && defined(__ARM_NEON)
    QwenQ41Batch attention;
#endif
    QwenQ41Task tasks[Q41_QUEUE_SIZE];
    unsigned take, put;
    int n_readers, stop, read_ahead, split_down;
    unsigned staged;
};

static void *qwen_q41_reader(void *arg) {
    QwenQ41Queue *q = arg;
    pthread_mutex_lock(&q->mutex);
    for (;;) {
        while (q->take == q->put && !q->stop)
            pthread_cond_wait(&q->work,&q->mutex);
        if (q->take == q->put) break;
        QwenQ41Task task = q->tasks[q->take++ % Q41_QUEUE_SIZE];
        pthread_mutex_unlock(&q->mutex);
        qwen_q41_read(task.job);
        dispatch_group_leave(task.ready);
        pthread_mutex_lock(&q->mutex);
    }
    pthread_mutex_unlock(&q->mutex);
    return NULL;
}

static void qwen_q41_queue_close(Model *m) {
    QwenQ41Queue *q = m->q41_queue;
    if (!q) return;
    pthread_mutex_lock(&q->mutex);
    q->stop = 1;
    pthread_cond_broadcast(&q->work);
    pthread_mutex_unlock(&q->mutex);
    for (int j = 0; j < q->n_readers; j++) pthread_join(q->readers[j],NULL);
#if defined(__aarch64__) && defined(__ARM_NEON)
    int mapped_gate = 0, mapped_down = 0;
#endif
    for (int j = 0; j < m->q41_window_count; j++) {
#if defined(__aarch64__) && defined(__ARM_NEON)
        mapped_gate += q->batch[j].file_windows[0].mapped_panel;
        mapped_down += q->batch[j].file_windows[2].mapped_panel;
        for (int p = 0; p < Q41_FFN_MATRICES; p++)
            qwen_q41_file_window_close(&q->batch[j].file_windows[p]);
#endif
        if (q->batch[j].ready) dispatch_release(q->batch[j].ready);
        if (q->batch[j].down_ready) dispatch_release(q->batch[j].down_ready);
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (q->attention.ready) dispatch_release(q->attention.ready);
    if (q->batch[0].file_windows[0].page_size)
        fprintf(stderr,"[qwen] Q4_1 file windows used: gate=%d down=%d\n",mapped_gate,mapped_down);
#endif
    pthread_cond_destroy(&q->work);
    pthread_mutex_destroy(&q->mutex);
    free(q);
    m->q41_queue = NULL;
}

static void qwen_q41_stream_close(Model *m) {
    qwen_q41_queue_close(m);
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (m->q41_window)
        for (int i = m->base.n_resident; i < m->c.n_layers; i++)
            for (int j = 0; j < 3; j++) qwen_q41_panel_close(&m->q41_layers[i].ffn[j]);
#endif
}

static QwenQ41Queue *qwen_q41_queue_open(Model *m) {
    if (m->q41_queue || m->q41_window_count < 2) return m->q41_queue;
    QwenQ41Queue *q = calloc(1,sizeof(*q));
    if (!q) return NULL;
    if (pthread_mutex_init(&q->mutex,NULL)) {
        free(q);
        return NULL;
    }
    if (pthread_cond_init(&q->work,NULL)) {
        pthread_mutex_destroy(&q->mutex);
        free(q);
        return NULL;
    }
    m->q41_queue = q;
    const char *split = moty_getenv("MOTY_Q41_SPLIT_DOWN");
    q->split_down = split && !strcmp(split,"1");
    const char *ahead = moty_getenv("MOTY_Q41_READ_AHEAD");
    if (ahead && *ahead) {
        char *end;
        long count = strtol(ahead,&end,10);
        if (!*end && count > 0 && count <= Q41_MAX_WINDOWS)
            q->read_ahead = count < m->q41_window_count ? (int)count : m->q41_window_count;
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    const char *file_windows = moty_getenv("MOTY_Q41_FILE_WINDOWS");
    long page_size = file_windows && !strcmp(file_windows,"1") ? sysconf(_SC_PAGESIZE) : 0;
#endif
    for (int j = 0; j < m->q41_window_count; j++) {
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (page_size > 0) {
            q->batch[j].file_windows[0].page_size = (size_t)page_size;
            q->batch[j].file_windows[2].page_size = (size_t)page_size;
        }
#endif
        q->batch[j].ready = dispatch_group_create();
        if (q->split_down) q->batch[j].down_ready = dispatch_group_create();
        if (!q->batch[j].ready || (q->split_down && !q->batch[j].down_ready)) {
            qwen_q41_queue_close(m);
            return NULL;
        }
    }
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (m->q41_attention_raw) {
        q->attention.ready = dispatch_group_create();
        if (!q->attention.ready) {
            qwen_q41_queue_close(m);
            return NULL;
        }
    }
#endif
    for (int j = 0; j < Q41_READERS; j++) {
        if (pthread_create(&q->readers[j],NULL,qwen_q41_reader,q)) break;
        q->n_readers++;
    }
    if (!q->n_readers) qwen_q41_queue_close(m);
    return m->q41_queue;
}

static QwenQ41Batch *qwen_q41_batch_for_layer(Model *m, int layer) {
    return &m->q41_queue->batch[(layer-m->base.n_resident)%m->q41_window_count];
}

static void qwen_q41_enqueue(QwenQ41Queue *q, QwenQ41Batch *batch, int count,
                              int split_down) {
    pthread_mutex_lock(&q->mutex);
    for (int j = 0; j < count; j++) {
        if (!batch->jobs[j].tensor) continue;
        dispatch_group_t ready = split_down && j == 2 ? batch->down_ready : batch->ready;
        dispatch_group_enter(ready);
        q->tasks[q->put++ % Q41_QUEUE_SIZE] = (QwenQ41Task){&batch->jobs[j],ready};
    }
    pthread_cond_broadcast(&q->work);
    pthread_mutex_unlock(&q->mutex);
}

static void qwen_q41_stream_prefetch(Model *m, int layer, int S) {
    QwenQ41Queue *q = m->q41_queue;
    QwenQ41Batch *batch = qwen_q41_batch_for_layer(m,layer);
    uint8_t *dst = m->q41_window + (batch-q->batch)*m->q41_window_bytes;
    qwen_q41_read_jobs(batch->jobs,&m->q41_layers[layer],dst,qwen_q41_use_panels(m,S));
#if defined(__aarch64__) && defined(__ARM_NEON)
    if (batch->file_windows[0].page_size) {
        for (int p = 0; p < Q41_FFN_MATRICES; p++) {
            QwenQ41Read *job = &batch->jobs[p];
            QwenQ41FileWindow *window = &batch->file_windows[p];
            if (job->tensor && window->page_size && m->q41_layers[layer].ffn[p].panel_bytes) {
                job->file_window = window;
                job->copy_dst = job->dst;
                job->map_panel = S == 1 && job->panel;
            }
        }
    }
#endif
    /* At most one batch per window is in flight. */
    qwen_q41_enqueue(q,batch,Q41_FFN_MATRICES,q->split_down && S == 1);
}

static void qwen_q41_stage_decode(Model *m, int slot) {
    qwen_q41_stream_prefetch(m,m->base.n_resident+slot,1);
    m->q41_queue->staged |= 1u << slot;
}

#if defined(__aarch64__) && defined(__ARM_NEON)
static void qwen_q41_attention_prefetch(Model *m, int layer) {
    QwenQ41Batch *batch = &m->q41_queue->attention;
    uint8_t *dst = m->q41_attention_raw;
    int count = 0;
    for (int j = 0; j < Q41_ATTN_MATRICES; j++) {
        const st_tensor *t = m->q41_attention[layer][j];
        if (!t) continue;
        batch->jobs[count++] = (QwenQ41Read){.tensor=t,.dst=dst};
        dst += scr_al(t->nbytes);
    }
    if (count) qwen_q41_enqueue(m->q41_queue,batch,count,0);
}
#endif
#endif

#ifdef __APPLE__
static inline QwenQ41Queue *qwen_q41_begin_batch(Model *m, int S) {
    int layers = m->c.n_layers;
    QwenQ41Queue *queue = qwen_q41_queue_open(m);
    if (queue) {
#if defined(__aarch64__) && defined(__ARM_NEON)
        if (S > 1 && m->q41_attention_raw) qwen_q41_attention_prefetch(m,0);
#endif
        int first = m->base.n_resident;
        int end = first+m->q41_window_count;
        for (int i = first; i < end && i < layers; i++) {
            if (queue->staged & (1u << (i-first))) {
                QwenQ41Batch *batch = qwen_q41_batch_for_layer(m,i);
                /* Staged scalar jobs may finish on separate groups. A
                 * multi-token caller binds all projections together. */
                if (S > 1 && batch->down_ready)
                    dispatch_group_wait(batch->down_ready,DISPATCH_TIME_FOREVER);
                if (qwen_q41_use_panels(m,S)) continue;
                /* A multi-token step needs the raw layout in this window. */
                dispatch_group_wait(batch->ready,DISPATCH_TIME_FOREVER);
            }
            qwen_q41_stream_prefetch(m,i,S);
        }
        queue->staged = 0;
    } else if (m->q41_window) {
        m->q41_window_count = 1;
    }
    return queue;
}
#endif

/* Return the layer whose down projection still needs a separate wait. */
static inline int qwen_q41_load_ffn(Model *m, int layer, int S) {
    int pending_down = -1;
    OP_T(t_read);
#ifdef __APPLE__
    QwenQ41Queue *q = m->q41_queue;
    if (q) {
        QwenQ41Batch *batch = qwen_q41_batch_for_layer(m,layer);
        dispatch_group_wait(batch->ready,DISPATCH_TIME_FOREVER);
        if (S == 1 && q->split_down && batch->jobs[2].tensor) pending_down = layer;
        qwen_q41_stream_bind(m,layer,batch->jobs,0,pending_down >= 0 ? 2 : 3);
    } else
#endif
    qwen_q41_stream_load(m,layer,S);
    OP_ACC(OP_FFN_READ_READY, t_read);
    return pending_down;
}

static inline void qwen_q41_finish_down(Model *m, int layer) {
#ifdef __APPLE__
    if (layer < 0) return;
    QwenQ41Batch *batch = qwen_q41_batch_for_layer(m,layer);
    OP_T(t_read);
    dispatch_group_wait(batch->down_ready,DISPATCH_TIME_FOREVER);
    qwen_q41_stream_bind(m,layer,batch->jobs,2,3);
    OP_ACC(OP_FFN_READ_DOWN, t_read);
#endif
}

#endif
