/* api_run.c — the library API tests (api_tests.c) without gtest: for a
 * target where only a C cross compiler is at hand, e.g. the sanitizer runs
 * on the board (docs/api.md, "Tests"):
 *
 *   SRC="nn/*.c hw/hw.c runtime/env.c runtime/kvcache.c runtime/moecache.c io/tier.c api/api.c \
 *        api/registry.c engines/lfm2.c engines/qwen.c tests/api_tests.c tests/api_run.c"
 *   aarch64-linux-gnu-gcc -O1 -g -fsanitize=thread -I. -Itests -DMOTY_THREADPOOL -DMOTY_NO_MAIN \
 *       -pthread $SRC -lm -o api_run
 *   ./api_run [TestName]          # prints each test and "ALL OK" / "FAILED" */
#include <stdio.h>
#include <string.h>
int ap_lifecycle(void); int ap_continuation(void); int ap_abort_callback(void); int ap_abort_thread(void);
int ap_inject(void); int ap_errors(void); int ap_cycles(void);
int main(int argc, char **argv) {
    struct { const char *n; int (*f)(void); } t[] = { {"Lifecycle", ap_lifecycle}, {"Continuation", ap_continuation},
        {"AbortCallback", ap_abort_callback}, {"AbortThread", ap_abort_thread}, {"Inject", ap_inject},
        {"Errors", ap_errors}, {"Cycles", ap_cycles} };
    int fail = 0;
    for (unsigned i = 0; i < sizeof t / sizeof *t; i++) {
        if (argc > 1 && strcmp(argv[1], t[i].n)) continue;
        int r = t[i].f(); printf("%-14s %s\n", t[i].n, r ? "FAIL" : "ok"); fflush(stdout); fail += r != 0;
    }
    printf("%s\n", fail ? "FAILED" : "ALL OK");
    return fail;
}
