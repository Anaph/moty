/* fail.h — fatal errors and allocation ownership for the embeddable library.
 *
 * moty_fail(): the one way the load/run paths report an unrecoverable error.
 * Inside a library call (a MotyTrap is armed on this thread) it records the
 * message and longjmps back to the call's entry, which returns an error
 * code; otherwise (the command-line engines) it prints the message and
 * exits, as the code always did.
 *
 * Open tracker: while a model is being opened, every allocation made by
 * the runtime (moty_balloc & co., and the engine translation units' libc
 * calls through runtime/track.h) is recorded in the handle's MotyTrack;
 * frees remove entries. What is left is exactly what the model owns, and
 * moty_track_free_all() releases it on close or after a failed open. The
 * tracker is thread-local and off outside moty_model_open, so the per-token
 * paths pay one TLS load per allocation. */
#ifndef NN_FAIL_H
#define NN_FAIL_H
#include <setjmp.h>
#include <errno.h>
#include <string.h>
#include <stddef.h>
#include <stdint.h>

enum { MOTY_FAIL_GENERIC = 1, MOTY_FAIL_IO, MOTY_FAIL_FORMAT, MOTY_FAIL_OOM };

typedef struct MotyTrap { jmp_buf jb; int code; char msg[512]; } MotyTrap;

/* arm/disarm on the calling thread (nesting: the previous trap is returned
 * by arm and restored by disarm) */
MotyTrap *moty_trap_arm(MotyTrap *t);
void      moty_trap_disarm(MotyTrap *prev);
#define MOTY_TRAP_TRY(t) (setjmp((t)->jb) == 0)

#if defined(__GNUC__)
__attribute__((noreturn, format(printf, 2, 3)))
#endif
void moty_fail_code(int code, const char *fmt, ...);
#define moty_fail(...) moty_fail_code(MOTY_FAIL_GENERIC, __VA_ARGS__)

/* allocation ownership during open */
typedef struct MotyTrack { void **p; int64_t n, cap; int64_t live; int big; } MotyTrack;
/* big = 1: blocks >= 256 KiB allocated under this tracker come from mmap
 * (a model's persistent memory: returned to the OS on close) */
MotyTrack *moty_track_set(MotyTrack *t);     /* returns the previous tracker */
void       moty_track_free_all(MotyTrack *t);
void       moty_track_merge(MotyTrack *dst, MotyTrack *src);
void      *moty_trk_malloc(size_t n);
void      *moty_trk_calloc(size_t n, size_t sz);
void      *moty_trk_realloc(void *p, size_t n);
char      *moty_trk_strdup(const char *s);
void       moty_trk_free(void *p);

/* getenv for the library's code paths: NULL once moty_env_disable() ran
 * (moty_model_open: no configuration from the host's shared environment) */
const char *moty_getenv(const char *name);
void        moty_env_disable(void);

/* the kernels' grow() buffers (process-wide, reused across calls) */
void       moty_release_scratch_all(void);
#endif
