# libmoty — the embeddable library

`libmoty` runs a moty model inside another process: open a model directory,
feed token ids or text (optionally with injected embedding rows, e.g. an NPU
image encoder's output), stream the answer through a callback, abort from
another thread, reset between requests, close. The public header is
[`c/api/moty.h`](../c/api/moty.h); this page is its contract.

The command-line engines use the same code: `lfm2` / `qwen` with `PROMPT=` or
`PROMPT_IDS=` (and optionally `EMBEDS=`) run their one-shot generation through
this API (`runtime/runtime.h: cli_api_oneshot`), so their speed is the API's
speed. Validation (`REF=`), perplexity (`PPL=`), packing (`SAVE_PACKED=`),
streaming (`MEM_GB=`), GGUF files, `MICRO=` and the interactive chat keep the
engines' direct path.

## Build

```
make clean && make THREADPOOL=1 libmoty.a        # c/libmoty.a
cmake -B build -S c -DMOTY_LIBRARY_ONLY=ON        # pthread pool by default
cmake --build build --target moty_lib             # build/libmoty.a; install: include/moty.h, lib/libmoty.a
```

Static, position-independent (CMake target), C99 header, links `-lm
-lpthread`. The library build uses moty's own pthread pool (no OpenMP
runtime needed); an OpenMP build works too (`THREADPOOL=0`). Global symbols:
only `moty_*` (no `main`: the engines are compiled with `MOTY_NO_MAIN`). No
constructors, destructors, `atexit`, signal handlers; nothing is written to
stdout.

Engines in the library: `lfm2` (config `model_type` `lfm2`, `lfm2_vl` — dense
LFM2 / LFM2.5 and the LFM2.5-VL text backbone) and `qwen` (`qwen3`, `qwen2`,
`llama` — Qwen3 dense and Llama-family dense, e.g. MiniCPM5-1B). Other
engines (MoE, Gemma, GLM, Qwen3.5 hybrid) are command-line only.

## Model directory

`moty_model_open(dir, ...)` reads exactly:

| file | |
|---|---|
| `config.json` | architecture; `model_type` picks the engine; `image_token_id` (root) for `moty_model_image_token` |
| every `*.safetensors` file in `dir` | the weights, as one shard set (sorted by name) |
| `tokenizer.json` | optional: without it only token ids work (`moty_tokenize` returns `MOTY_ERR_UNSUPPORTED`) |

Everything else in the directory is ignored (signature manifests, a
`chat_template.jinja`: the chat template of each engine is built in). A moty
container (`tools/ref/pack_r4.py`, `SAVE_PACKED=`) keeps its own int4/int8
layout; an HF snapshot is quantized at open (`qbits` 4 or 8). `qbits` is not
used for a pre-packed container.

### Container v2 (fast open)
`SAVE_PACKED=` writes format `moty-q4r4` version 2, made to be used in place
through a read-only `mmap` (`mmap_weights`, below):
- the token embedding as moty keeps it (I8 table + F32 per-row scales
  `<name>.qs`), instead of the bf16/f32 table every open re-quantized;
- the codes of the matrices an engine fuses (q/k/v, gate/up) back to back,
  their scales likewise: the fused matrix is a view of the mapping;
- only the tensors the engine reads (a VL snapshot's vision tower is
  dropped); 64-byte aligned.

Upgrade any container (v1 from `pack_r4.py` or an older `SAVE_PACKED`):

```
SNAP=<v1 dir> QBITS=4 Q4FMT=r4 SAVE_PACKED=<v2 dir> ./qwen     # or ./lfm2
```

Generation is bit-identical (checked: 64/64 ids, v1/v2 x copy/map, VisionPsy
and LFM2.5-VL). v1 containers keep loading (mapped too; the tensors that
are fused get copied, their mapped pages are then released).

## Calls

| call | |
|---|---|
| `moty_options_init`, `moty_sampling_init` | fill defaults (and the `size` ABI guard: `moty_options` accepts its v1 size, `MOTY_OPTIONS_V1_SIZE`; fields added since keep their defaults) |
| `moty_model_open` / `moty_model_close` | a handle owns weights, KV cache, tokenizer, open files |
| `moty_model_vocab / hidden / ctx / image_token / n_past / load_s` | model facts; `hidden` = width of injected rows |
| `moty_tokenize(m, text, add_bos, chat_template, ids, cap)` | text → ids; special tokens written in the text are recognized as HF does (the LFM2.5-VL prompt `<\|image_start\|>` + 256 × `<image>` + `<\|image_end\|>Describe the image.` with the chat template gives HF's 272 ids exactly) |
| `moty_token_piece` | id → text bytes |
| `moty_generate(m, ids, n, embeds, n_rows, embed_token, sampling, cb, user, stats)` | append to the conversation and generate |
| `moty_abort` / `moty_abort_clear` | sticky stop, any thread |
| `moty_reset` | empty conversation (also clears an abort) |
| `moty_last_error` | message of the handle's last failure |
| `moty_release_scratch` | free the kernels' scratch, join the pool's threads |
| `moty_set_log` | log sink (level 1 error, 2 info), process-wide; default stderr |
| `moty_version` | `"moty <version> (<thread backend>)"` |

### Generation
- `ids[0..n)` are appended; the previous call's last sampled token (not yet in
  the KV cache) is fed first, so consecutive calls continue one conversation.
  `moty_reset` between unrelated requests.
- Injection: the k-th position of `ids` equal to `embed_token` takes row k of
  `embeds` (float32, contiguous `[n_rows][moty_model_hidden()]`). The count of
  `embed_token` positions must equal `n_rows` (else `MOTY_ERR_ARG`); the rows
  are only read during the call.
- The callback gets each token and its text; a non-zero return stops.
  `stats` gives prompt/new token counts, prefill/decode seconds and why it
  stopped (`MOTY_STOP_LENGTH / EOS / CALLBACK / ABORT`).
- Greedy by default (`temperature` 0); nucleus sampling with `temperature` > 0.

### Abort
`moty_abort(m)` sets a flag on the handle: a running `moty_generate` returns
`MOTY_ERR_ABORTED` within one decode step or one prefill chunk
(`prefill_chunk`, default 64 tokens: 0.86 s measured on a Cortex-A53), and every later
`moty_generate` refuses to start until `moty_abort_clear` or `moty_reset`
(a stop that lands between "decide to generate" and the call is not lost).
Safe from any thread until `moty_model_close`; not concurrently with close.
An aborted or failed call leaves an empty conversation.

### Errors
No `exit()` / `abort()` in the library: a failure deep in the loader or a
kernel (bad file, missing tensor, allocation failure) unwinds to the API call
(`setjmp`/`longjmp`, `nn/fail.h`), which frees what the call allocated and
returns a `moty_status` with a message (`err` buffer for open,
`moty_last_error` afterwards). A failed open leaves nothing allocated and no
file open.

## Threads
- Calls on one handle must not overlap (except `moty_abort`). Calls on
  different handles are serialized by one process-wide lock: the compute
  kernels share one thread pool and scratch buffers, and on a 4-core board
  two concurrent inferences would only contend. Several handles may be open.
- The pool (pthread build) is shared by all handles and created on first
  use: `threads` workers (default: the CPUs of the process mask), each call
  running with its handle's `threads` / `threads_decode`. With `pin_threads`
  (default 1) workers are pinned one per CPU and the calling thread is pinned
  to the first CPU for the duration of a call, its own affinity restored
  after. After a parallel region a worker polls for `pool_spin_us` (default
  1000 µs) and then sleeps on a condition variable: an idle pool uses no CPU.
  `pool_spin_us = 0` sleeps at once. `moty_release_scratch()` joins the
  workers; the next call starts them again.

## Memory
- `mmap_weights` (default 1, v1.1): the packed weights of a container are
  used in place in a read-only private mapping of `model.safetensors`, not
  copied. They are clean file pages: RSS counts them, but under memory
  pressure the kernel can drop them (and re-read them from flash if touched)
  instead of the process holding the same bytes as anonymous memory, and
  open no longer holds two copies (page cache + heap) at once. `2`
  populates the mapping during open (the whole file read up front, no page
  faults in the first prefill; measured slower and larger, docs/performance.md
  5.15); `0` copies as before. Board B, VisionPsy int8: open 0.2 s instead of
  6.1–6.4 s, RSS during a generate 20 MB anonymous + 375 MB file instead of
  449 MB anonymous.
- A handle's persistent blocks of 256 KiB or more (KV cache, scratch
  arenas, copied weights) are `mmap`ed and `munmap`ed at close, so the memory goes back to the
  OS instead of glibc's heap; the weight mapping is unmapped at close. Everything the handle allocated during open is
  recorded (`nn/fail.h` open tracker) and freed at close.
- The kernels' scratch buffers are process-wide (reused by every call, sized
  by the largest prompt so far); `moty_release_scratch()` frees them.
- `tests/api_tests.c` (Cycles) opens, generates, closes and releases four
  times with RSS flat within 2 MB and the thread count back to the baseline;
  `api/examples/moty_cycle.c` does the same on a real model and prints RSS,
  threads and `MemAvailable` per cycle.

## Environment and process state
The library reads no environment variables: `moty_model_open` switches the
process's moty code to `moty_getenv() == NULL` (the command-line engines keep
reading theirs). No re-exec.

Process-wide state that remains, all serialized by the API lock:
- the thread pool (above), the kernels' scratch buffers (`moty_grow`
  statics), the RoPE cos/sin table (per rope base and width, grows with
  position), the sampler's temperature/nucleus/RNG state (set from each
  call's `moty_sampling`);
- the runtime's layout switches (`g_q4fmt`, `g_kv_bits`, ...), set from the
  handle's options at open;
- the big-block registry (pointer → size of the `mmap`ed blocks), the log
  callback.

## Tests
`c/tests/api_tests.c` (gtest `Api.*`): lifecycle (open/generate/close twice,
same tokens), continuation (4 + 8 tokens = 12 at once), abort from the
callback and from another thread, sticky abort and clear, injection (rows,
other rows, count mismatch), error paths (missing directory, unknown
architecture, missing tensor mid-load, bad options, context overflow, bad
ids, no tokenizer), cycles (RSS, threads).

Sanitizers: on the Cortex-A53 board (GCC 12 runtimes, kernel with
`mmap_rnd_bits` 18) the seven tests pass 10 of 10 runs under
ThreadSanitizer and 10 of 10 under AddressSanitizer / LeakSanitizer with
zero reports (built with `tests/api_run.c`, the gtest-free driver; build
line in its header). ThreadSanitizer found one race there — the pool's
spin/pin settings written by each call while idle workers read them — now
atomics. On an x86 host whose kernel randomizes mmap with 32 bits, GCC
12's sanitizer runtimes break before `main` (an empty
`int main(void){return 0;}` built with `-fsanitize=address` crashes 6 of
40 runs, with `-fsanitize=thread` 20 of 20: "unexpected memory mapping");
use a board, a kernel with `vm.mmap_rnd_bits=28`, or newer runtimes.

## Measured on the board

RV1126B, 4× Cortex-A53, its own vision workload running, LFM2.5-350M
text prompt (66 tokens) and LFM2.5-VL-450M image prompt (272 tokens), 64
new tokens, 4 threads / decode on 3, medians of 5 interleaved runs:

| | prefill tok/s | decode tok/s |
|---|---|---|
| text, command line before the API (direct loop) | 41.0 | 15.19 |
| text, through the API | 45.7 | 15.92 |
| text, through the API, `prefill_chunk` 64 | 46.1 | 15.53 |
| image prompt, direct loop | 42.3 | 11.83 |
| image prompt, through the API | 41.8 | 11.90 |
| image prompt, through the API, `prefill_chunk` 64 | 43.7 | 12.45 |

`api/examples/moty_probe.c` (image prompt, 4 threads):

| | |
|---|---|
| abort during the prefill, `prefill_chunk` 64 / 0 | returns after 0.86 s / 4.9 s |
| abort during the decode | returns after 2–38 ms |
| process CPU over 3 s idle with a model open, `pool_spin_us` 1000 / 0 | 0.008 s / 0.000 s |

`api/examples/moty_cycle.c`, 4 cycles open → 64-token image answer → close
→ `moty_release_scratch()`: RSS 407 MB open, 13.7–15.1 MB after close,
9.6–10.8 MB after release, threads back to 1, MemAvailable within 4 MB of
its level. Tracing every allocation through 4 warm cycles leaves 0 bytes
allocated by moty alive; over 30–40 cycles on x86 the RSS after release
stays within 12–14 MB (glibc keeps some freed heap in its arenas; its
in-use count grows ~0.4 KB per cycle, also with one thread and no pool).

## Example

```c
moty_options o; moty_options_init(&o); o.threads = 4; o.threads_decode = 3;
moty_model *m; char err[256];
if (moty_model_open("/models/lfm2.5-vl-lm", &o, &m, err, sizeof err)) { puts(err); return; }
int32_t ids[512];
int n = moty_tokenize(m, prompt_with_image_tokens, 1, 1, ids, 512);
moty_sampling s; moty_sampling_init(&s); s.max_new_tokens = 64;
moty_stats st;
moty_generate(m, ids, n, npu_rows, 256, moty_model_image_token(m), &s, on_token, user, &st);
moty_reset(m);                 /* next frame */
moty_model_close(m);
moty_release_scratch();        /* between requests: no moty memory or threads left */
```
