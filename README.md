# Moty

**Tiny engine, immense model.** Pure-C LLM inference with zero runtime
dependencies: no BLAS, no Python at runtime, no GPU required. Weights load
straight from HuggingFace safetensors snapshots.

Standalone engines, one per architecture:

| Engine | Model family | Notes |
|---|---|---|
| `olmoe` | OLMoE | Reference GQA-MoE engine |
| `qwen` | Qwen3 dense / Qwen3.5 hybrid / Llama-family dense (e.g. MiniCPM5-1B) | GQA + optional QK-norm; Gated DeltaNet + Gated Attention |
| `lfm2` | LFM2 / LFM2.5 dense (e.g. LFM2.5-350M) and LFM2-MoE | short-conv + GQA hybrid; HF snapshot or GGUF |
| `gemma` | Gemma 4 (e.g. 12B-it, text-only) | Sliding/global hybrid, p-RoPE, GeGLU, SP tokenizer |

## Build

```
make            # builds all engines (from repo root or c/)
make qwen       # just one engine
make portable   # portable CPU baseline (x86-64-v3 / armv8-a / power8)
make test       # test suite (GoogleTest via a separate CMake build path)

# CMake build of record (same sources, same libraries):
cmake -B c/build -S c && cmake --build c/build -j       # portable tier
cmake -B c/build -S c -DMOTY_ARCH=native && cmake --build c/build -j
ctest --test-dir c/build                                # the full suite
```

Both builds produce the same layered libraries
(`libmoty-hw` → `libmoty-nn` → `libmoty-runtime`) and the engine
binaries; the Makefile additionally covers the exotic cross targets
(ARMv7, PPC64, portable-v4/sve2) and docker, CMake covers the desktop
paths and is what CI-grade tooling should consume.

Requirements: a C compiler (gcc/clang) and GNU make. Linux, macOS, Windows
(MinGW/MSYS2), *BSD and PowerPC are supported. OpenMP is used when available.

The engines build with the C compiler alone. `make test` additionally needs
cmake ≥ 3.24 and a C++ compiler for the GoogleTest harness (test logic itself
is plain C; the C++ is confined to thin gtest glue). The first configure
downloads a pinned gtest via FetchContent unless a system GTest is installed.

## Docker

No toolchain on the host needed — the multi-stage `Dockerfile` builds the
portable binaries and ships a slim runtime image (debian-slim + libgomp,
all four engines in `/usr/local/bin`):

```bash
make docker                 # docker build -t moty .
make docker-test            # runs the full gtest suite INSIDE the build:
                            # the image fails to build if a test fails

# models live on the host, mounted read-only at /models
SNAP=/models/Qwen3-0.6B make docker-run                  # interactive chat
docker run --rm -it -v ~/models:/models:ro \
  -e GGUF=/models/Qwen3-0.6B-Q4_K_M.gguf -e PROMPT="hi" moty
```

Or with compose (services `qwen`, `gemma`, `olmoe`; the whole env-knob table
passes through from the host):

```bash
docker compose build
SNAP=/models/Qwen3-0.6B QBITS=8 docker compose run --rm qwen
MODELS_DIR=/data/llm GGUF=/models/model.gguf docker compose run --rm qwen
```

The image builds `make portable` (x86-64-v3 on amd64, armv8-a on arm64), so
it runs on any modern host of the same architecture; rebuild with
`--build-arg ARCH=native` for the fastest binaries on one specific machine,
and `--build-arg BASE=<mirror>/debian:bookworm-slim` behind a registry
mirror. A hard RAM cap composes naturally with the engine's own knobs:
`docker run -m 256m -e MICRO=1 ...`.

## SIMD

The shared kernels (contract in `c/hw/hw.h`) are selected at compile time by `-march`
(one `#ifdef` ladder, no runtime CPUID dispatch):

| Build | f32 kernel | int8 kernel |
| --- | --- | --- |
| `make portable` (x86-64-v3) | AVX2+FMA | AVX2 (maddubs) |
| `make portable-v4` (x86-64-v4) | AVX512F | AVX2 (v4 has no VNNI) |
| `make` / `ARCH=native` (x86) | AVX512F where present | +AVX512-VNNI on supporting CPUs |
| ARM (NEON baseline) | NEON | NEON; +dotprod (SDOT) with `ARCH=native` on supporting cores |

Quantized matmuls (`QBITS=8`) run the per-row Q8_0 scheme: activations are
quantized to int8 per row and the dot is pure integer (`dot_i8i8`). `IDOT=0`
selects the exact f32×int8 path instead — use it for byte-exact `REF`
comparisons (olmoe now shares the same int8 scheme, so its numerics also
shift slightly unless `IDOT=0`). The engines print the compiled tiers in
their startup banner (`idot ... | f32 ...`).

`make test-native` builds and runs the test suite with `-march=native`, so
the reference-based tests exercise the SIMD kernels your CPU actually has.

**ARMv8.0 without dotprod (Cortex-A53 class).** On aarch64 `QBITS=4` uses
the **Q4R4** layout (`Q4FMT=r4`, the default there): 4-row blocks,
group-32 f16 scales, int8 group-32 activations, NEON GEMV for decode and a
4×4 register-tiled GEMM for prefill built from `SMULL`/`SMLAL` only (no
`SDOT`, no fp16 arithmetic). The non-matmul row ops (RMSNorm, SiLU·up,
softmax, short conv, residual adds) have NEON versions with scalar
references (`hw/hw_ops.h`). On other targets `QBITS=4` keeps the grouped
int4 (`Q4FMT=g`, VNNI fused paths on AVX512); `Q4FMT=r4` runs the portable
scalar Q4R4 kernels, which is how x86 tests them. Details:
[docs/architecture.md](docs/architecture.md#int4-q4r4-and-the-armv80-neon-kernels),
numbers: [docs/performance.md](docs/performance.md#5-small-arm-boards-int4-on-cortex-a53).

## Run

Each engine is driven by environment variables and reads a HuggingFace
snapshot directory (config.json + tokenizer.json + *.safetensors):

```
# one-shot prompt
SNAP=/path/to/Qwen3-4B PROMPT="Hello!" ./c/qwen

# interactive chat (persistent KV cache)
SNAP=/path/to/Qwen3-4B ./c/qwen
```

Common environment variables (qwen engine):

| Var | Default | Meaning |
|---|---|---|
| `SNAP` | — | model snapshot directory (required) |
| `PROMPT` | — | one-shot prompt; if unset, interactive chat on stdin |
| `NGEN` | 256 | max new tokens |
| `CTX` | 4096 | context length |
| `TEMP` / `NUCLEUS` / `SEED` | 0.7 / 0.95 | sampling (TEMP=0 → greedy) |
| `CHAT_TEMPLATE` | 1 | wrap prompt in the model's chat format |
| `THINK` | 0 | Qwen3 thinking mode (0 pre-closes the think block) |
| `QBITS` | 0 | 8 → int8-quantize weights **and embeddings** at load (~4× less RAM); 4 → int4 layer weights with group-wise scales (~8× on layers; embeddings stay int8; the lm_head stays int8 except with Q4R4, where a tied head is packed to Q4R4 from the original rows) |
| `QGROUP` | 32 | int4 scale group size (multiple of 16; 0 → one scale per row). 32 matches the GGUF Q4_0 block. Q4R4 always uses 32 |
| `Q4FMT` | `r4` on aarch64, else `g` | `QBITS=4` layout: `r4` = Q4R4 (ARMv8.0 NEON kernels, 4.5 bits/weight), `g` = legacy grouped int4 |
| `EMBED` | `ram` | `disk` → no resident embedding table: the row of each input token is read from the snapshot (bf16/f32, no int8 rounding). With a tied lm_head it requires `QBITS=4 Q4FMT=r4` (the head is packed separately). Saves vocab×hidden bytes — 200 MB on MiniCPM5-1B |
| `SAVE_PACKED` | — | `<dir>`: load with `QBITS=4 Q4FMT=r4`, write a pre-packed Q4R4 container (`<dir>/model.safetensors` + config/tokenizer files) and exit. Loading `SNAP=<dir>` with `QBITS=4` then skips bf16 reading and packing |
| `THREADS` | — | cap the OpenMP team; overrides `OMP_NUM_THREADS`; applied before load |
| `MEM_GB` | — | RAM budget in GiB: layers beyond the budget stream from disk each step |
| `MEM_FRAC` | — | same budget as a fraction (0..1) of total physical RAM; `MEM_GB` wins |
| `MICRO` | 0 | 1 → micro-RSS mode (qwen only): **no** weights resident, minimum possible RAM; see below |
| `MICRO_DROP` | 1 | 0 → let streamed weight pages live in the OS page cache (faster, more memory charged to the process's cgroup) |
| `KV_BITS` | 0 | 8 → int8 KV cache with one scale per (kv-head, position): 4× less KV RAM/traffic (1.2 GB → 300 MB at 4B/4k ctx). Queries stay f32, so the quantization error is one-sided |
| `PREFILL_CHUNK` | 0 | feed the prompt to the model in blocks of ≤N tokens: caps prefill activation peaks (≈1.6 GB at S=4096 on 4B → tens of MB at N=256), **bit-identical** output. Off by default: with `MEM_GB` each block re-reads the streamed layers from disk |
| `GGUF` | — | single-file GGUF model instead of `SNAP` (qwen): weights, config and tokenizer all come from the file; see below |
| `REF` | — | ref.json with prompt_ids/full_ids for greedy validation |
| `REF_LOGITS` | — | with `REF`: write the raw f32 logits of the last prompt position to this file (compare with `tools/ref/cmp_logits.py`) |
| `PPL` | — | `{"ids":[...]}` JSON: teacher-forced perplexity over the ids (decode path, one token per step) instead of generating |
| `PPL_OUT` / `PPL_N` | — | with `PPL`: write the per-position argmax ids (for top-1 agreement vs a reference, `tools/ref/cmp_ppl.py`); cap the token count |
| `IGNORE_EOS` | 0 | 1 → keep generating past end-of-turn tokens (fixed-length benchmark runs) |
| `THREADS_DECODE` | = `THREADS` | team size for the single-token decode steps only (prefill keeps `THREADS`). On a board whose cores also run another process one fewer thread than cores decodes faster: every decode matmul ends in a barrier that waits for a preempted thread |
| `HEAD_TOPK` | 0 | K > 0 → two-stage lm_head for Q4R4 heads: a 1-bit copy of the head picks K candidate rows, exact int4 logits only for them, the rest -1e30 (top-K-truncated distribution). ~2× cheaper head; the argmax equals the full int4 head's at 99.76 % of positions for K=512 on LFM2.5-350M. Ignored by `REF`/`PPL` |
| `TOKENS` | 0 | 1 → dump generated token ids to stderr |
| `TTA` | off | **experimental** test-time adaptation: `cache` (neural cache), `bias` (online logit bias) or `lora` (online low-rank lm_head adapter); see [docs/online-learning.md](docs/online-learning.md) |
| `TTA_N` / `TTA_LAMBDA` / `TTA_THETA` / `TTA_LR` | 2048 / 0.1 / 1.0 / 0.1 | cache size, mix weight (capped at 0.5), similarity temperature, bias/lora learning rate (lora defaults to 1e-3) |
| `TTA_RANK` | 4 | rank of the `TTA=lora` online adapter (capped at 64) |
| `LORA` | — | safetensors file (or dir) with LoRA adapters to load at startup (qwen only) |
| `TRAIN` | — | corpus.txt → run the LoRA fine-tuner instead of generating (qwen only, see below) |
| `TRAIN_CTX` / `TRAIN_STRIDE` / `TRAIN_EPOCHS` | 512 / CTX / 1 | training window, window stride, epochs |
| `TRAIN_LR` / `TRAIN_WD` / `TRAIN_CE_CHUNK` | 1e-4 / 0 / 32 | AdamW learning rate, weight decay, CE chunk size |
| `LORA_RANK` / `LORA_ALPHA` / `LORA_LAYERS` / `LORA_HEAD` | 8 / 2·rank / 4 / 0 | adapter rank, scale, how many top layers to adapt, train an lm_head adapter too |
| `LORA_OUT` | lora.safetensors | where the trainer saves the adapters |

Validation and measurement tooling lives in `tools/ref/` (HF transformers
reference ids, logits, perplexity and fake-quant scripts) and `tools/a53/`
(bandwidth/kernel benchmark and run-matrix helpers for small ARM boards).
Building the libraries and an engine with `-DMOTY_PROF` adds a
per-operation table (`[prof-op]`, ms and ms/token for every phase: norms,
projections, RoPE, attention, conv, FFN, lm_head, ...) after each prefill
and each decode turn. The layer timers live in `libmoty-nn`, so rebuild
everything: `make clean && make lfm2 CFLAGS="-I. -O2 -march=native -fopenmp -pthread -DMOTY_PROF"`.

`TTA` (qwen only, default off — zero cost when unset) adapts predictions to
the text being generated: the neural cache mixes in a distribution over
recently seen continuations, the bias variant runs closed-form SGD on a
persistent logit bias. Adaptation state is cleared on every context reset
and REF validation mode structurally bypasses it.

On startup the engines seed hot-thread OpenMP defaults (`OMP_WAIT_POLICY=active`
etc.) and re-exec themselves once so libgomp picks them up; any `OMP_`/`GOMP_`
variable you set yourself wins, and `MOTY_NO_OMP_TUNE=1` disables the whole
mechanism.

`MEM_GB`/`MEM_FRAC` (qwen and gemma) trade speed for memory: the engine keeps
as many layers resident as fit the budget (embeddings, norms and recurrent
state always stay resident) and re-reads the remaining layers from the
safetensors on every step, prefetching the next layer while the current one
computes. With `QBITS=8` the streamed layers are quantized on read and run
the same int8 kernel as resident ones (4× smaller scratch and disk traffic,
token output identical to fully-resident int8); at `QBITS=0/4` they run f32.
Token output is identical at any budget. Unset → everything resident. The
classic path never goes below embeddings + one layer of scratch; if your
budget is under that floor the engine tells you to use `MICRO=1`.

## GGUF

The qwen engine runs GGUF files directly — one file, nothing else needed:

```bash
GGUF=Qwen3-0.6B-Q4_K_M.gguf PROMPT="hi" ./qwen
```

Weights are indexed into the same loader the safetensors path uses (names
translated from the llama.cpp scheme), the config is synthesized from the
`<arch>.*` metadata, and the tokenizer comes from `tokenizer.ggml.*`
(byte-level BPE only — the Qwen family). Supported tensor types: F32, F16,
BF16, Q8_0, Q4_0 and the K-quants Q4_K/Q5_K/Q6_K (dequantized on load, then
requantized per `QBITS`). Special case: Q4_0 with `QBITS=4` is repacked
**losslessly** into the engine's group-wise int4 (same nibble encoding, same
32-element scale blocks) — you run exactly the bits that are in the file.
`REF=`, `MEM_GB`, `MICRO=1`, `KV_BITS` and LoRA adapters all work as with a
snapshot directory.

`MICRO=1` (qwen only) is the mode for **hard** memory limits (cgroup,
embedded): nothing of the model stays resident. Embedding rows are gathered
from disk per token, every matmul re-reads its matrix in constant-size 4 MB
chunks, and the lm_head streams the same way; with the default `MICRO_DROP=1`
each chunk is evicted from the page cache right after use, so the footprint
is only activations + KV cache + tokenizer. The context default drops to 256
(`CTX` still wins) because the KV cache is the last big allocation. Token
output is bit-identical to the resident f32 path. The price is honest: the
whole model transits from disk on *every* token, so decode speed is disk
bandwidth divided by model size (~1–3 s/token for a 4B model on NVMe).
Ballpark RSS: Qwen3-0.6B ≈ 100–150 MB, 4B ≈ 150–250 MB, dominated by KV and
tokenizer, not weights. Incompatible with `TRAIN` (which needs resident f32
weights); `LORA` adapters work (they are small and stay resident).

## Fine-tuning (LoRA)

The qwen engine ships a dependency-free LoRA fine-tuner (hand-written
backward pass + AdamW, `c/engines/qwen_train.h`). Set `TRAIN=` to a plain-text
corpus and the binary trains rank-r adapters on the top `LORA_LAYERS`
layers (q/k/v/o + gate/up/down, optionally the lm_head) instead of
generating, then saves them as a single safetensors file:

```bash
TRAIN=corpus.txt SNAP=$SNAP/Qwen3-0.6B/snapshots/<hash> \
  LORA_LAYERS=4 LORA_RANK=8 TRAIN_EPOCHS=2 ./c/qwen
LORA=lora.safetensors SNAP=$SNAP/Qwen3-0.6B/snapshots/<hash> ./c/qwen
```

Training v1 requires a dense Qwen3 checkpoint, f32 weights (`QBITS=0`) and
everything resident (no `MEM_GB`). Setting `LORA=` during training warm
starts from previously saved adapters; at inference `LORA=` works with any
mode (chat, `PROMPT`, `REF`) and quantization, and adapters with all-zero
B matrices are exact no-ops. Every adapter tensor in the file must match a
known name and shape or the engine refuses to start. The backward pass is
verified by a finite-difference gradient check over every adapter
parameter (`qt_grad_fd`).

## Qwen engine notes

`qwen` runs two architecture families from the same binary, selected by the
model's config.json:

- **Qwen3 dense** (0.6B–32B): GQA attention with per-head QK-RMSNorm, RoPE
  (theta from config), SwiGLU MLP, tied embeddings where the checkpoint uses
  them.
- **Qwen3.5 hybrid** (Qwen3-Next lineage, e.g. Qwen3.5-4B): `layer_types`
  mixes **Gated DeltaNet** linear-attention layers (recurrent state instead of
  a KV cache — memory does not grow with context) with **Gated Attention**
  full-attention layers (output gate, partial RoPE).

Memory: a 4B model needs ~16 GB RAM at f32; `QBITS=8` quantizes everything
including the embedding table (~4 GB, loaded chunk-wise so the peak never
spikes above the final footprint).
In chat mode the recurrent DeltaNet state is append-only: editing history
requires a full conversation reset (the engine does this automatically when
the context fills up).

### Validating against a reference (REF mode)

`REF=<file> SNAP=<snapshot> ./c/qwen` greedy-decodes and compares token ids
against a reference file, printing the match count (exit 0 on full match,
2 otherwise). The file format is plain JSON:

```json
{"prompt_ids": [151644, 872, ...], "full_ids": [151644, 872, ..., 785, 6722]}
```

`full_ids` must extend `prompt_ids`; the engine generates
`len(full_ids) - len(prompt_ids)` tokens greedily from `prompt_ids` and
requires an exact id-by-id match (f32 build). Produce the reference with any
tool that runs the original model — e.g. with `transformers`:

```python
tok = AutoTokenizer.from_pretrained(m); model = AutoModelForCausalLM.from_pretrained(m, torch_dtype=torch.float32)
ids = tok(prompt, return_tensors="pt").input_ids
out = model.generate(ids, max_new_tokens=24, do_sample=False, num_beams=1)
json.dump({"prompt_ids": ids[0].tolist(), "full_ids": out[0].tolist()}, open("ref.json","w"))
```

### Small ARM boards (int4 on Cortex-A53)

The end-to-end recipe used for LFM2.5-350M and MiniCPM5-1B on a
4× Cortex-A53 board (details and numbers in
[docs/performance.md](docs/performance.md#5-small-arm-boards-int4-on-cortex-a53)):

```bash
# host: cross-build static binaries for ARMv8.0
make -C c lfm2 qwen CC=aarch64-linux-gnu-gcc AR=aarch64-linux-gnu-ar ARCH=armv8-a \
     LDFLAGS="-lm -fopenmp -pthread -static"
# host (any arch): pack once into a Q4R4 container (loads ~5x faster on the board)
SNAP=/models/LFM2.5-350M QBITS=4 Q4FMT=r4 SAVE_PACKED=/models/LFM2.5-350M-q4r4 ./c/lfm2
# board
SNAP=LFM2.5-350M-q4r4 QBITS=4 THREADS=4 PROMPT="..." ./lfm2
# fastest decode on a board shared with another workload (see docs/performance.md §5.7)
SNAP=LFM2.5-350M-q4r4 QBITS=4 THREADS=4 THREADS_DECODE=3 HEAD_TOPK=512 PROMPT="..." ./lfm2
SNAP=MiniCPM5-1B-q4r4 QBITS=4 EMBED=disk THREADS=4 PROMPT="..." ./qwen
```

`EMBED=disk` keeps the 1B model's embedding table out of RAM (it is the
largest single tensor); peak RSS is then ~560 MB for MiniCPM5-1B and
~300 MB for LFM2.5-350M.

### Tokenizer parity (tok_oracle)

The test build also produces `c/tests/build/tok_oracle` — a corpus-scale
parity harness: `./tok_oracle <tokenizer.json> < cases.tsv` where each line
is `TEXT\tID,ID,...` (escapes: `\n \t \r \\`). Run it against ids produced
by the reference tokenizer before debugging model-level mismatches.

## Layout

```
c/engines/   one .c per model family + moty.c (multi-model dispatcher)
c/nn/        shared compute kernels (attention, conv, MoE, FFN, sampler...)
c/hw/        hardware abstraction: AVX512/AVX2/NEON/SVE2/VSX/scalar backends
c/runtime/   engine scaffolding (load/KV/generation hooks) + MoE expert cache
c/io/        safetensors, GGUF, io_uring, tiering
c/tok/       tokenizers (BPE byte-level, SentencePiece)
c/util/      json, compat shims, grammar, profiling
c/tests/     test suite: C logic + GoogleTest glue (make test)
Dockerfile   multi-stage image: build -> (test) -> slim runtime
docker-compose.yml  services qwen/gemma/olmoe with /models mounted
docs/        architecture + how-to guides
```

Documentation:

- [docs/architecture.md](docs/architecture.md) — module map, layering
  rules, kernel contracts, scratch arena, token flow
- [docs/adding-a-model.md](docs/adding-a-model.md) — support a new model
  architecture end-to-end (hooks, registration, validation, tests)
- [docs/supporting-layers.md](docs/supporting-layers.md) — extract or add
  a shared layer in `nn/` (design rules, template, mandatory tests)
- [docs/modularization-plan.md](docs/modularization-plan.md) — phased plan:
  paste-in headers → layered libraries (hw/nn/runtime), ops-table engines,
  explicit layer variants, unified build
- [docs/performance.md](docs/performance.md) — memory-bandwidth model,
  measured optimizations, int4 on Cortex-A53 (speed, quality, llama.cpp
  comparison)
- [tools/ref/README.md](tools/ref/README.md) — HF transformers reference
  scripts (greedy ids, logits, perplexity, int4 fake-quant)
- [tools/a53/README.md](tools/a53/README.md) — measurement helpers for
  small ARM boards
- [docs/gemma-plan.md](docs/gemma-plan.md) — engine review + the gemma
  improvement plan (bring-up on a real model, shared-attention migration,
  arena, VNNI int4, sliding-window KV)

### Gemma engine notes

`gemma` runs the Gemma 4 text stack: sliding-window attention interleaved
with global layers (p-RoPE, optionally larger global head_dim), per-head
q/k/v RMSNorm, sandwich norms, GeGLU, optional KV-sharing / K=V / per-layer
embeddings — all config-driven. A few checkpoint conventions could not be
verified offline and sit behind loud probes (see VERIFY comments in
`c/engines/gemma.c`); on first run against a real snapshot, resolve any reported
tensor-name/shape mismatch, validate the tokenizer with `tok_oracle`, then
gate with REF mode. `GEMMA_NORM_PLAIN=1` switches the RMSNorm convention
from `(1+w)` to `w` if REF parity points at the norm.

## License

See [LICENSE](LICENSE).
