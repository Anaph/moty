/* moty.c — unified multi-model inference engine.
 *
 * Auto-detects the model architecture from the GGUF metadata and dispatches
 * to the matching model implementation. Each model_X.c is compiled to a .o
 * with its main() renamed to model_X_main via -Dmain=model_X_main. This file
 * provides the real main() that reads the GGUF arch and selects the model.
 *
 * To add a new model:
 *   1. Write model_X.c (any name, any architecture) following the runtime.h
 *      hook pattern. It MUST have int main(int,char**).
 *   2. Add a Makefile rule: compile with -Dmain=model_X_main → model_X.o
 *   3. Add an entry to the models[] table below.
 *   4. Add model_X.o to the moty link line.
 *
 * That's it. No changes to engine code, no shared types to modify. Each model
 * is fully independent (own Cfg/Layer/Model structs, own hooks) but shares
 * the compute kernels (nn_attn.h, nn_conv.h, nn_ffn.h, nn_moe_sigmoid.h,
 * simd.h, etc.) via #include.
 *
 * Build: make moty
 * Run:   GGUF=model.gguf PROMPT="..." NGEN=128 ./moty
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

/* Forward declarations: each model_X.o exports model_X_main (renamed from main
 * via -Dmain=model_X_main at compile time). */
int lfm2_main(int argc, char **argv);
int qwenmoe_main(int argc, char **argv);
int gemma_main(int argc, char **argv);
int olmoe_main(int argc, char **argv);
int qwen_main(int argc, char **argv);
int glm_main(int argc, char **argv);

/* Registry: arch string → entry point. Extend this table to add models. */
typedef struct { const char *arch; int (*fn)(int, char**); const char *desc; } ModelEntry;
static const ModelEntry models[] = {
    {"lfm2moe",   lfm2_main,   "Liquid Foundation Model 2 (LFM2.5-8B-A1B)"},
    {"qwen35moe", qwenmoe_main, "Qwen3.5-MoE (Qwen3.6-35B-A3B)"},
    {"qwen3moe",  qwenmoe_main, "Qwen3-MoE"},
    {"gemma2",    gemma_main,  "Gemma 2"},
    {"gemma3",    gemma_main,  "Gemma 3"},
    {"gemma3n",   gemma_main,  "Gemma 3N"},
    {"olmoe",     olmoe_main,  "OLMoE (dense MoE)"},
    {"qwen3",     qwen_main,   "Qwen3 (dense)"},
    {"glm_moe_dsa", glm_main, "GLM-5.2 MoE (MLA + DSA indexer)"},
    {"qwen2",     qwen_main,   "Qwen2 (dense)"},
    {NULL, NULL, NULL},
};

#ifdef MOTY_PLUGINS
/* ---- M5: motori out-of-tree come .so ----
 * Un plugin e' una shared library che esporta:
 *   const char *moty_plugin_arch(void);   stringa arch GGUF servita
 *   int          moty_plugin_main(int, char**);
 * Si caricano con MOTY_PLUGIN=percorso/plugin.so (lista separata da ':'), anche
 * piu' di uno; l'ultimo che dichiara un arch gia' presente VINCE (override).
 * Build di riferimento: make plugins/sample.so (oppure -fPIC -shared a mano). */
#include <dlfcn.h>
#define MOTY_MAX_PLUGINS 16
static ModelEntry plugin_entries[MOTY_MAX_PLUGINS];
static void *plugin_handles[MOTY_MAX_PLUGINS];
static int n_plugins = 0;

static void plugins_load(void) {
    const char *list = getenv("MOTY_PLUGIN");
    if (!list) return;
    char pathbuf[2048];
    while (*list && n_plugins < MOTY_MAX_PLUGINS) {
        const char *sep = strchr(list, ':');
        size_t len = sep ? (size_t)(sep - list) : strlen(list);
        if (len == 0 || len >= sizeof pathbuf) { if (sep) list = sep + 1; else break; continue; }
        memcpy(pathbuf, list, len); pathbuf[len] = 0;
        void *h = dlopen(pathbuf, RTLD_NOW | RTLD_LOCAL);
        if (!h) { fprintf(stderr, "[plugin] %s: %s\n", pathbuf, dlerror()); if (sep) list = sep + 1; else break; continue; }
        const char *(*arch_fn)(void) = (const char *(*)(void))dlsym(h, "moty_plugin_arch");
        int (*main_fn)(int, char**) = (int (*)(int, char**))dlsym(h, "moty_plugin_main");
        if (!arch_fn || !main_fn) {
            fprintf(stderr, "[plugin] %s: servono moty_plugin_arch + moty_plugin_main\n", pathbuf);
            dlclose(h);
        } else {
            plugin_handles[n_plugins] = h;
            plugin_entries[n_plugins] = (ModelEntry){ arch_fn(), main_fn, "plugin" };
            n_plugins++;
            fprintf(stderr, "[plugin] %s: arch '%s'\n", pathbuf, arch_fn());
        }
        if (sep) list = sep + 1; else break;
    }
}
static const ModelEntry *plugins_find(const char *arch) {
    for (int i = 0; i < n_plugins; i++)
        if (plugin_entries[i].arch && !strcmp(plugin_entries[i].arch, arch))
            return &plugin_entries[i];
    return NULL;
}
#endif /* MOTY_PLUGINS */

/* Minimal GGUF header reader to extract the architecture string without
 * pulling in the full gguf.h (which would conflict with each model's TU). */
static const char *detect_arch(const char *path, char *buf, int bufsz) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    uint32_t magic, version;
    if (fread(&magic, 4, 1, f) != 1 || fread(&version, 4, 1, f) != 1) { fclose(f); return NULL; }
    /* skip n_tensors and n_kv (uint32 for v1/v2, uint64 for v3) */
    if (version >= 3) fseek(f, 16, SEEK_CUR);  /* 2 × uint64 */
    else              fseek(f, 8,  SEEK_CUR);  /* 2 × uint32 */
    /* scan KV pairs for general.architecture */
    for (int i = 0; i < 256; i++) {
        uint64_t klen;
        if (fread(&klen, 8, 1, f) != 1) break;  /* GGUF strings: uint64 length */
        if (klen == 0 || klen > 1024) break;
        char key[1024];
        if (fread(key, 1, klen, f) != klen) break;
        key[klen] = 0;
        uint32_t vt;
        if (fread(&vt, 4, 1, f) != 1) break;
        if (strcmp(key, "general.architecture") == 0 && vt == 8) {
            uint64_t slen;
            if (fread(&slen, 8, 1, f) != 1) break;
            if (slen >= (uint64_t)bufsz) slen = bufsz - 1;
            if (fread(buf, 1, slen, f) != slen) break;
            buf[slen] = 0;
            fclose(f);
            return buf;
        }
        /* skip value of any other type */
        if (vt == 0 || vt == 7) { uint64_t sl; if(fread(&sl,8,1,f)!=1) break; fseek(f, (long)sl, SEEK_CUR); }
        else if (vt == 4 || vt == 5 || vt == 6) fseek(f, 4, SEEK_CUR);
        else if (vt == 8) fseek(f, 1, SEEK_CUR);
        else if (vt == 10) fseek(f, 8, SEEK_CUR);
        else if (vt == 2) {
            uint32_t at, n; if (fread(&at,4,1,f)!=1 || fread(&n,4,1,f)!=1) break;
            int esz = (at==4||at==5||at==6)?4 : (at==8)?1 : (at==10)?8 : 0;
            if (at==7||at==0) { for(uint32_t j=0;j<n&&j<8192;j++){uint64_t sl;if(fread(&sl,8,1,f)!=1)break;fseek(f,(long)sl,SEEK_CUR);} }
            else if (esz) fseek(f, (long long)n * esz, SEEK_CUR);
            else break;
        }
        else break;
    }
    fclose(f);
    return NULL;
}

int main(int argc, char **argv) {
    const char *gguf = getenv("GGUF");
    if (!gguf || !*gguf) {
        fprintf(stderr, "moty: set GGUF=<file.gguf>\n");
        fprintf(stderr, "Supported architectures:\n");
        for (const ModelEntry *m = models; m->arch; m++)
            fprintf(stderr, "  %s — %s\n", m->arch, m->desc);
        return 1;
    }

    /* detect architecture */
    char arch_buf[128];
    const char *arch = detect_arch(gguf, arch_buf, sizeof(arch_buf));
    if (!arch) {
        fprintf(stderr, "moty: cannot read architecture from %s\n", gguf);
        return 1;
    }
    fprintf(stderr, "[moty] %s: arch=%s\n", gguf, arch);

#ifdef MOTY_PLUGINS
    plugins_load();
    const ModelEntry *plug = plugins_find(arch);
    if (plug) return plug->fn(argc, argv);   /* override del registry statico */
#endif

    /* dispatch */
    for (const ModelEntry *m = models; m->arch; m++) {
        if (strcmp(m->arch, arch) == 0) {
            return m->fn(argc, argv);
        }
    }

    fprintf(stderr, "moty: unsupported architecture '%s'\n", arch);
    fprintf(stderr, "Supported:\n");
    for (const ModelEntry *m = models; m->arch; m++)
        fprintf(stderr, "  %s — %s\n", m->arch, m->desc);
    return 1;
}
