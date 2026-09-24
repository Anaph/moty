/* engine_api.h — what an engine translation unit exports to the library
 * (api/api.c). runtime.h instantiates one MotyEngineOps per engine that
 * defines ENGINE_API_ID before including it; everything behind the ops is
 * the engine's own static code (hooks, loader, tokenizer).
 *
 * Calling contract: api.c calls open/prefill/step/... with a MotyTrap armed
 * (a failure longjmps back to it) and, during open, with the open tracker
 * set (every allocation the instance keeps is owned by the handle). Calls
 * are serialized by api.c. */
#ifndef MOTY_RUNTIME_ENGINE_API_H
#define MOTY_RUNTIME_ENGINE_API_H
#include <stdint.h>
#include <stdatomic.h>

typedef struct {
    int qbits, ctx, kv_bits, embed_disk, head_topk, q4fmt;   /* q4fmt: 1 Q4R4, 0 legacy grouped */
    int log_level;                                           /* 2: print the load banner */
    const char *q8_tensors;
    void **inst_out;          /* set to the instance as soon as it exists: a failed open closes it */
} MotyEngineOpen;

typedef struct MotyEngineOps {
    const char *id;                                   /* "lfm2", "qwen" */
    int   (*accepts)(const char *model_type);         /* config.json model_type */
    void *(*open)(const char *dir, const MotyEngineOpen *o);
    void  (*close)(void *inst);                       /* non-heap resources; the heap is the handle's */
    int   (*vocab)(void *inst);
    int   (*hidden)(void *inst);
    int   (*ctx)(void *inst);
    int   (*image_token)(void *inst);                 /* config image_token_id or -1 */
    int   (*has_tokenizer)(void *inst);
    int   (*bos)(void *inst);                         /* -1: none */
    int   (*is_stop)(void *inst, int tok);
    /* ids[0..n) at positions pos.. ; rows: injected at inj_tok positions.
     * Returns malloc'd logits of the last position, or NULL if *abort was
     * raised between prefill chunks (chunk: tokens per step, 0 = all). */
    float *(*prefill)(void *inst, const int *ids, int n, int pos, const float *rows, int n_rows, int inj_tok,
                      int chunk, const atomic_int *abort);
    float *(*step)(void *inst, int tok, int pos);     /* one decode step: malloc'd logits */
    void  *(*scratch)(void *inst);                    /* the instance's Scratch for sampling */
    void  (*reset)(void *inst);
    int   (*encode)(void *inst, const char *text, int add_bos, int chat_template, int *ids, int cap);
    int   (*piece)(void *inst, int tok, char *buf, int cap);
} MotyEngineOps;

extern const MotyEngineOps moty_engine_lfm2;
extern const MotyEngineOps moty_engine_qwen;
#endif
