/* api_internal.h — between the public API (api.c) and what links it: the
 * engine registry (api/registry.c, the library) and the command-line
 * engines, which open through their own ops. */
#ifndef MOTY_API_INTERNAL_H
#define MOTY_API_INTERNAL_H
#include "api/moty.h"
#include "runtime/engine_api.h"
/* ops: one engine (the command-line engines), or NULL: the first of the
 * NULL-terminated engines[] that accepts the config's model_type */
moty_status moty_model_open_with(const MotyEngineOps *ops, const MotyEngineOps *const *engines, const char *path,
                                 const moty_options *opt, moty_model **out, char *err, size_t err_len);
#endif
