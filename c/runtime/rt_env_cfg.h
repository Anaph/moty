/* Env: wrapper su libmoty-runtime (M4). RunEnv resta il nome storico. */
#ifndef RT_ENV_CFG_H
#define RT_ENV_CFG_H
#include "runtime/config.h"

typedef MotyRunConfig RunEnv;

#ifndef MOTY_CORE_NO_LEGACY
#define budget_from_env moty_rt_budget_from_env
#define omp_hot_tune    moty_rt_omp_hot_tune
#define parse_env       moty_rt_parse_env
#endif

#endif /* RT_ENV_CFG_H */
