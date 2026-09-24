/* rknn_encode — run a vision encoder (.rknn, Rockchip NPU) on one image and
 * write its output rows as raw little-endian f32: the EMBEDS= file of a
 * moty text model (docs/performance.md §5.9, tools/rknn/README.md).
 *
 *   rknn_encode <model.rknn> <image.rgb|image.ppm> <out.f32> [reps]
 *
 * image: uint8 RGB at the model's input size, raw (H*W*3 bytes) or binary
 * PPM (P6). The model's own normalisation is used (input type UINT8, NHWC).
 * librknnrt is loaded at run time (dlopen): RKNN_LIB=<path>, default
 * "librknnrt.so". Build (not part of moty's build; needs rknn_api.h from
 * Rockchip's rknn-toolkit2, runtime version of the board):
 *
 *   aarch64-linux-gnu-gcc -O2 -I$RKNN_INCLUDE rknn_encode.c -ldl -o rknn_encode
 */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "rknn_api.h"

static double now_ms(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t); return t.tv_sec * 1e3 + t.tv_nsec * 1e-6; }
static long mem_avail_kb(void) {
    FILE *f = fopen("/proc/meminfo", "r"); char l[256]; long v = -1;
    while (f && fgets(l, sizeof l, f)) if (sscanf(l, "MemAvailable: %ld", &v) == 1) break;
    if (f) fclose(f);
    return v;
}
static void *slurp(const char *p, long *n) {
    FILE *f = fopen(p, "rb"); if (!f) { perror(p); exit(1); }
    fseek(f, 0, SEEK_END); *n = ftell(f); fseek(f, 0, SEEK_SET);
    void *b = malloc(*n); if (!b || fread(b, 1, *n, f) != (size_t)*n) { perror(p); exit(1); }
    fclose(f); return b;
}

#define API(name) static __typeof__(name) *p_##name
API(rknn_init); API(rknn_query); API(rknn_inputs_set); API(rknn_run);
API(rknn_outputs_get); API(rknn_outputs_release); API(rknn_destroy);

int main(int argc, char **argv) {
    if (argc < 4) { fprintf(stderr, "usage: %s <model.rknn> <image.rgb|.ppm> <out.f32> [reps]\n", argv[0]); return 1; }
    int reps = argc > 4 ? atoi(argv[4]) : 1; if (reps < 1) reps = 1;
    const char *lib = getenv("RKNN_LIB") ? getenv("RKNN_LIB") : "librknnrt.so";
    void *h = dlopen(lib, RTLD_NOW);
    if (!h) { fprintf(stderr, "dlopen %s: %s\n", lib, dlerror()); return 1; }
#define LOAD(name) if (!(p_##name = (__typeof__(name) *)dlsym(h, #name))) { fprintf(stderr, "dlsym %s\n", #name); return 1; }
    LOAD(rknn_init); LOAD(rknn_query); LOAD(rknn_inputs_set); LOAD(rknn_run);
    LOAD(rknn_outputs_get); LOAD(rknn_outputs_release); LOAD(rknn_destroy);

    long mem0 = mem_avail_kb(), msz;
    double t0 = now_ms();
    void *model = slurp(argv[1], &msz);
    rknn_context ctx;
    int ret = p_rknn_init(&ctx, model, (uint32_t)msz, 0, NULL);
    if (ret != RKNN_SUCC) { fprintf(stderr, "rknn_init: %d\n", ret); return 1; }
    double t_init = now_ms() - t0;
    long mem1 = mem_avail_kb();
    rknn_input_output_num io; p_rknn_query(ctx, RKNN_QUERY_IN_OUT_NUM, &io, sizeof io);
    rknn_tensor_attr ia; memset(&ia, 0, sizeof ia); ia.index = 0;
    p_rknn_query(ctx, RKNN_QUERY_INPUT_ATTR, &ia, sizeof ia);
    rknn_tensor_attr oa; memset(&oa, 0, sizeof oa); oa.index = 0;
    p_rknn_query(ctx, RKNN_QUERY_OUTPUT_ATTR, &oa, sizeof oa);
    /* NHWC input: dims [1, H, W, C]; NCHW: [1, C, H, W] */
    int H = ia.fmt == RKNN_TENSOR_NCHW ? ia.dims[2] : ia.dims[1];
    int W = ia.fmt == RKNN_TENSOR_NCHW ? ia.dims[3] : ia.dims[2];
    fprintf(stderr, "model: %u input(s) %u output(s); input %ux%ux%ux%u %s %s; output n_elems %u\n",
            io.n_input, io.n_output, ia.dims[0], ia.dims[1], ia.dims[2], ia.dims[3],
            get_type_string(ia.type), get_format_string(ia.fmt), oa.n_elems);

    long isz; unsigned char *img = slurp(argv[2], &isz), *px = img;
    if (isz > 2 && img[0] == 'P' && img[1] == '6') {      /* binary PPM: P6 W H 255 <pixels> */
        int pw, ph, pm, off = 0;
        if (sscanf((char *)img, "P6 %d %d %d%n", &pw, &ph, &pm, &off) != 3 || pm != 255) { fprintf(stderr, "bad PPM\n"); return 1; }
        px = img + off + 1; isz -= off + 1;
        if (pw != W || ph != H) { fprintf(stderr, "image %dx%d, model wants %dx%d\n", pw, ph, W, H); return 1; }
    }
    if (isz != (long)W * H * 3) { fprintf(stderr, "image: %ld bytes, model wants %d (%dx%d RGB)\n", isz, W * H * 3, W, H); return 1; }

    rknn_input in; memset(&in, 0, sizeof in);
    in.index = 0; in.buf = px; in.size = (uint32_t)(W * H * 3); in.type = RKNN_TENSOR_UINT8; in.fmt = RKNN_TENSOR_NHWC;
    rknn_output out; double best = 1e30, sum = 0;
    for (int r = 0; r < reps; r++) {
        double a = now_ms();
        if ((ret = p_rknn_inputs_set(ctx, 1, &in)) != RKNN_SUCC) { fprintf(stderr, "inputs_set: %d\n", ret); return 1; }
        if ((ret = p_rknn_run(ctx, NULL)) != RKNN_SUCC) { fprintf(stderr, "run: %d\n", ret); return 1; }
        memset(&out, 0, sizeof out); out.index = 0; out.want_float = 1;
        if ((ret = p_rknn_outputs_get(ctx, 1, &out, NULL)) != RKNN_SUCC) { fprintf(stderr, "outputs_get: %d\n", ret); return 1; }
        double dt = now_ms() - a; sum += dt; if (dt < best) best = dt;
        if (r == reps - 1) {
            FILE *f = fopen(argv[3], "wb");
            if (!f || fwrite(out.buf, 1, out.size, f) != out.size) { perror(argv[3]); return 1; }
            fclose(f);
        }
        p_rknn_outputs_release(ctx, 1, &out);
    }
    long mem2 = mem_avail_kb();
    p_rknn_destroy(ctx);
    long mem3 = mem_avail_kb();
    fprintf(stderr, "rknn_encode: init %.0f ms | encode best %.1f ms mean %.1f ms (%d runs) | %u floats -> %s | "
            "MemAvailable kB: before %ld, loaded %ld (%+ld), after run %ld, after destroy %ld\n",
            t_init, best, sum / reps, reps, oa.n_elems, argv[3], mem0, mem1, mem1 - mem0, mem2, mem3);
    free(model); free(img); dlclose(h);
    return 0;
}
