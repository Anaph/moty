/* rssrun CMD ARGS... : run CMD, then print peak RSS (ru_maxrss) and wall time
 * to stderr. The child gets oom_score_adj=1000 so under memory pressure the
 * kernel kills the benchmark first, not the device's other processes. */
#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <time.h>
#include <sys/wait.h>
#include <sys/resource.h>
int main(int argc, char **argv) {
    if (argc < 2) { fprintf(stderr, "usage: rssrun cmd args...\n"); return 2; }
    struct timespec t0, t1; clock_gettime(CLOCK_MONOTONIC, &t0);
    pid_t p = fork();
    if (p == 0) {
        FILE *f = fopen("/proc/self/oom_score_adj", "w"); if (f) { fputs("1000", f); fclose(f); }
        execvp(argv[1], argv + 1); perror("execvp"); _exit(127);
    }
    int st; struct rusage ru; wait4(p, &st, 0, &ru);
    clock_gettime(CLOCK_MONOTONIC, &t1);
    fprintf(stderr, "[rssrun] peak_rss_mb=%.1f wall_s=%.2f status=%d\n", ru.ru_maxrss / 1024.0,
            (t1.tv_sec - t0.tv_sec) + (t1.tv_nsec - t0.tv_nsec) * 1e-9, WIFEXITED(st) ? WEXITSTATUS(st) : 128 + WTERMSIG(st));
    return WIFEXITED(st) ? WEXITSTATUS(st) : 1;
}
