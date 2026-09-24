/* track.h — the libc heap calls of every moty translation unit (through
 * nn/nn_alloc.h; engines include it first, for the static code they
 * instantiate from io/, tok/, util/, runtime/) go through nn/alloc.c's
 * tracker wrappers: a library call can then own exactly what it allocated
 * and release it (nn/fail.h). With no tracker set they are plain libc
 * calls. The system headers come first: their declarations stay untouched.
 * nn/alloc.c, which implements the wrappers, defines MOTY_NO_TRACK_MACROS. */
#ifndef MOTY_NN_TRACK_H
#define MOTY_NN_TRACK_H
#define _GNU_SOURCE
#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include "nn/fail.h"
#ifndef MOTY_NO_TRACK_MACROS
#undef malloc
#undef calloc
#undef realloc
#undef strdup
#undef free
#define malloc(n)      moty_trk_malloc(n)
#define calloc(n, s)   moty_trk_calloc(n, s)
#define realloc(p, n)  moty_trk_realloc(p, n)
#define strdup(s)      moty_trk_strdup(s)
#define free(p)        moty_trk_free(p)
#endif
#endif
