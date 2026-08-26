/* Stand-alone runner for c/tests/simd_tests.c (the SIMD parity suite): calls
 * every st_* function and exits non-zero on any failure. Used by the ARM CI
 * job (cross-compile + qemu-user) because the gtest harness is x86-oriented.
 * Convention of simd_tests.c: 0 = pass, 1 = fail, 2 = skip. */
#include <stdio.h>

int st_dot_i8i8(void), st_dot_i4i8(void), st_dot_f32i8(void), st_dot_f32(void);
int st_qrow(void), st_dn_rows(void), st_report(void);

int main(void) {
    int (*fns[])(void) = { st_dot_i8i8, st_dot_i4i8, st_dot_f32i8,
                           st_dot_f32, st_qrow, st_dn_rows, st_report };
    const char *names[] = { "dot_i8i8","dot_i4i8","dot_f32i8","dot_f32","qrow","dn_rows","report" };
    int fails = 0;
    for (unsigned k = 0; k < sizeof(fns)/sizeof(fns[0]); k++) {
        int r = fns[k]();
        fprintf(stderr, "[%s] %s\n", r==0?"PASS":(r==2?"SKIP":"FAIL"), names[k]);
        if (r == 1) fails++;
    }
    fprintf(stderr, fails ? "%d FAILS\n" : "ALL PASS\n", fails);
    return fails ? 1 : 0;
}
