/* plugins/sample.c — esempio MINIMO di motore out-of-tree (M5).
 * Build: make plugins/sample.so   (o: gcc -I.. -I. -fPIC -shared)
 * Run:   MOTY_PLUGIN=plugins/sample.so moty GGUF=<file con arch "sample">
 * Un motore vero implementa il pattern runtime.h (docs/adding-a-model.md);
 * qui bastano i due simboli del contratto plugin. */
#include <stdio.h>
const char *moty_plugin_arch(void) { return "sample"; }
int moty_plugin_main(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("[sample plugin] ciao dal motore out-of-tree\n");
    return 0;
}
