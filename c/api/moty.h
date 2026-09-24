/* moty.h — the moty library: load a model, run it on token ids (optionally
 * with injected embedding rows), stream the answer through a callback.
 * C99, no moty internals. docs/api.md has the full contract; in short:
 *   - no exit()/abort(): every failure is a moty_status + message;
 *   - a handle is used by one thread at a time; calls on different handles
 *     take turns (one process-wide lock: the kernels share one thread pool
 *     and scratch buffers); moty_abort() is safe from any thread;
 *   - the library never writes to stdout. */
#ifndef MOTY_H
#define MOTY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define MOTY_API_VERSION 1

typedef struct moty_model moty_model;

typedef enum {
    MOTY_OK = 0,
    MOTY_ERR_ARG = -1,          /* bad argument (NULL, size mismatch, out of range) */
    MOTY_ERR_IO = -2,           /* a file could not be opened or read */
    MOTY_ERR_FORMAT = -3,       /* config/weights/tokenizer not understood */
    MOTY_ERR_OOM = -4,          /* allocation failed (the handle stays usable / nothing leaked) */
    MOTY_ERR_CONTEXT = -5,      /* prompt + answer would exceed the context */
    MOTY_ERR_ABORTED = -6,      /* moty_abort(): the conversation has been reset */
    MOTY_ERR_UNSUPPORTED = -7   /* architecture / feature not served by the library */
} moty_status;

typedef struct {
    int size;             /* = sizeof(moty_options); set by moty_options_init */
    int threads;          /* 0 = every CPU of the process's affinity mask */
    int threads_decode;   /* 0 = threads; e.g. 3 on a 4-core board shared with another workload */
    int ctx;              /* max tokens in the conversation, 0 = 4096 (capped by the model) */
    int qbits;            /* 4 (default) or 8; a pre-packed container keeps its own layout */
    int kv_bits;          /* 0 = f32 KV cache (default), 8 = int8 */
    int embed_disk;       /* 1 = read token-embedding rows from the file (saves vocab x hidden bytes) */
    int head_topk;        /* 0 = off; >0: two-stage lm_head shortlist (Q4R4 heads) */
    int prefill_chunk;    /* tokens per prefill step: abort latency / activation memory
                           * (default 64; 0 = the whole prompt in one step) */
    int log_level;        /* 0 silent, 1 errors (default), 2 info (moty_set_log or stderr) */
    int pool_spin_us;     /* pthread pool: a worker polls this long after a parallel region before
                           * it sleeps (default 1000; 0 = sleep at once: ~0 CPU when idle, slower decode) */
    int pin_threads;      /* 1 (default): workers pinned one per CPU, the calling thread pinned for
                           * the duration of a call and its affinity restored; 0: no pinning */
    /* --- v1.1 (additive: a caller built against v1 passes the smaller size) --- */
    int mmap_weights;     /* pre-packed containers: 1 (default) use the weights in place through a
                           * read-only mapping (no copy; clean file pages the kernel can reclaim),
                           * 2 = also read the whole file during open (MAP_POPULATE), 0 = copy */
} moty_options;
#define MOTY_OPTIONS_V1_SIZE ((int)offsetof(moty_options, mmap_weights))
void moty_options_init(moty_options *o);

typedef struct {
    int      size;              /* = sizeof(moty_sampling); set by moty_sampling_init */
    float    temperature;       /* 0 = greedy (default) */
    float    top_p;             /* nucleus mass when temperature > 0 (default 0.95) */
    uint64_t seed;              /* 0 = keep the sampler's current state */
    int      max_new_tokens;    /* default 64 */
    int      ignore_eos;        /* 1: always generate max_new_tokens */
} moty_sampling;
void moty_sampling_init(moty_sampling *s);

typedef struct {
    int    prompt_tokens, new_tokens;
    double prefill_s, decode_s;
    int    stop;                /* MOTY_STOP_* */
} moty_stats;
enum { MOTY_STOP_LENGTH = 0, MOTY_STOP_EOS = 1, MOTY_STOP_CALLBACK = 2, MOTY_STOP_ABORT = 3 };

/* one generated token; piece = its text (UTF-8 bytes, may be a partial
 * character), not NUL-terminated. Return non-zero to stop generating. */
typedef int (*moty_token_cb)(void *user, int32_t token, const char *piece, int piece_len);

/* open a model directory (moty container or HF snapshot: config.json +
 * safetensors [+ tokenizer.json]). err (optional) receives the message. */
moty_status moty_model_open(const char *path, const moty_options *opt, moty_model **out,
                            char *err, size_t err_len);
/* releases everything the handle owns (weights, KV, tokenizer, files) */
void        moty_model_close(moty_model *m);

int moty_model_vocab(const moty_model *m);
int moty_model_hidden(const moty_model *m);      /* width of injected embedding rows */
int moty_model_ctx(const moty_model *m);         /* effective context */
int moty_model_image_token(const moty_model *m); /* config image_token_id, -1 if none */
int moty_model_n_past(const moty_model *m);      /* tokens in the conversation so far */
double moty_model_load_s(const moty_model *m);   /* seconds moty_model_open took */

/* text -> ids; chat_template = 1 wraps text as one user turn of the model's
 * template. Returns the count (<= cap) or a negative moty_status
 * (MOTY_ERR_UNSUPPORTED: the directory has no tokenizer.json). */
int moty_tokenize(moty_model *m, const char *text, int add_bos, int chat_template, int32_t *ids, int cap);
/* id -> text bytes written to buf (not NUL-terminated), 0 for unknown ids */
int moty_token_piece(moty_model *m, int32_t id, char *buf, int cap);

/* Appends ids[0..n) to the conversation and generates. Injection: the k-th
 * position of ids equal to embed_token takes row k of
 * embeds[n_rows][moty_model_hidden()] (float32, contiguous); the number of
 * embed_token positions must equal n_rows (else MOTY_ERR_ARG); embeds may
 * be NULL (no injection). The rows are only read during this call. The
 * previous call's last token (sampled, not yet in the KV cache) is fed
 * first, so consecutive calls continue one conversation. st may be NULL. */
moty_status moty_generate(moty_model *m, const int32_t *ids, int n,
                          const float *embeds, int n_rows, int32_t embed_token,
                          const moty_sampling *s, moty_token_cb cb, void *user, moty_stats *st);

/* Sticky abort, safe from any thread until moty_model_close: the running
 * moty_generate stops within one decode step or prefill chunk, and every
 * moty_generate refuses to start (MOTY_ERR_ABORTED) until moty_abort_clear
 * or moty_reset. An aborted call leaves an empty conversation. */
void        moty_abort(moty_model *m);
void        moty_abort_clear(moty_model *m);
moty_status moty_reset(moty_model *m);   /* forget the conversation (KV, recurrent state); clears the abort */
const char *moty_last_error(const moty_model *m);

/* free the kernels' process-wide scratch buffers and join the pool's worker
 * threads (they restart on the next call): between requests an idle
 * process then holds no moty memory beyond open handles and no moty threads */
void        moty_release_scratch(void);

/* log sink for the library's messages (level 1 error, 2 info), process-wide;
 * NULL = stderr. The message has no trailing newline. */
typedef void (*moty_log_cb)(void *user, int level, const char *msg);
void        moty_set_log(moty_log_cb cb, void *user);
const char *moty_version(void);          /* "moty <version> (<thread backend>)" */

#ifdef __cplusplus
}
#endif
#endif
