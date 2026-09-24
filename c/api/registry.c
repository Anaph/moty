/* registry.c — the engines libmoty serves (runtime/engine_api.h) and the
 * public open that picks one by config.json model_type. Only libmoty links
 * it: a command-line engine opens through its own ops. */
#include "api/api_internal.h"
#include "nn/fail.h"

static const MotyEngineOps *const engines[] = { &moty_engine_lfm2, &moty_engine_qwen, NULL };

moty_status moty_model_open(const char *path, const moty_options *opt, moty_model **out, char *err, size_t err_len) {
    moty_env_disable();            /* the library takes no configuration from the environment */
    return moty_model_open_with(NULL, engines, path, opt, out, err, err_len);
}
