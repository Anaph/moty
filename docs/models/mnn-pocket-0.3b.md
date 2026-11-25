# MNN-Pocket-0.3B on CPU

Use the official QAT Q4_1 GGUF. The upstream MNN export uses
W4A16 with 64 weights per group. Its GGUF stores the original
UINT4 codes in 32-weight blocks with FP16 scale and minimum.
Moty consumes those blocks directly and shares the packed table
between input embeddings and the tied output projection.
[GGUF model card](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF#qat-q4_1-gguf) · [MNN model card](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN#model-details)

## Run the model

From the repository root:

```sh
make -C c -j4 moty
model="$PWD/MNN-Pocket-0.3B-Instruct-Q4_1.gguf"
curl -fL -o "$model" \
  https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF/resolve/791b7e92fa790a5ca5bbd7604feb1ce1d2429d14/MNN-Pocket-0.3B-Instruct-Q4_1.gguf
GGUF="$model" QBITS=-1 THREADS=4 CTX=512 TEMP=0 NGEN=64 \
  PROMPT='Explain how a rainbow forms, step by step.' ./c/moty
```

- `QBITS=-1` selects native GGUF weights in Qwen. Qwen uses this
  mode automatically for compatible GGUFs when `QBITS` is unset. `QBITS=0` selects float32
  weights and `QBITS=8` selects the existing int8 conversion.
- `Q41_ACT=f16` uses NEON FMLAL and FP32 accumulation when the build
  supports it. ARMv8.0 uses FP32 activations and NEON FP32 accumulation.
  `Q41_ACT=f32` selects that precision explicitly. Inputs that overflow or
  become nonfinite when packed as FP16 use the FP32 matmul.
- `MMAP=0` loads a copy of the packed GGUF. `MEM_GB` and `MEM_FRAC` with
  explicit `QBITS=-1` enable FFN streaming for this model. The budget
  guides placement. The process RSS also includes scratch, tokenizer
  data, file-backed pages and runtime overhead.
- The original Hugging Face snapshot runs via `SNAP=/path/to/snapshot`.
  For the recommended native QAT path use the Q4_1 GGUF above.

## What runs on NEON

Q/K/V and fused `gate/up` share activation preparation. The Pocket
FFN writes SwiGLU output directly into the input buffer for `down`.
Dense prefill can reuse a shared workspace across layers and chunks.
Long FP32 GQA decode uses 128-position blocks for Pocket's 64-wide
heads, starting at 2048 visible positions. Each query head retains
its own full-context softmax. The int8 KV cache and other head shapes
continue to use their existing attention paths.

On Apple FMLAL the opt-in lossless cache packs Group64 panels from the
original GGUF blocks. A verified sidecar may supply a read-only file view
for streamed `gate` and `down` at a reusable address. Workers keep each
slot until its consumers have finished. Missing or invalid panels fall
back to the original tensor. The tied indexed head stores repeated
scale/min pairs once and still computes all logits.
Stale sidecar headers and lengths trigger an atomic replacement. A resident
checksum failure triggers one rebuild attempt. A checksum failure found by a
streaming reader uses raw Q4_1 for the current run and attempts to repair the
sidecar for the next open.

| Option | Purpose |
| --- | --- |
| `MOTY_Q41_PACKED=all` | Create or use local panels for eligible projections with a resident model. |
| `MOTY_Q41_PACKED=stream` / `stream-all` | Read streamed FFN panels. `stream-all` also packs eligible attention projections. |
| `MOTY_Q41_INDEXED_HEAD=1` | Use the lossless indexed layout for a compatible tied head. |
| `MOTY_Q41_CACHE_HASH=parallel` | Use version 2 panel checksums. |
| `MOTY_Q41_FILE_WINDOWS=1` | Reuse verified `gate` and `down` file views during scalar decode. |
| `MOTY_Q41_WINDOWS=4`, `MOTY_Q41_READ_AHEAD=3`, `MOTY_Q41_SPLIT_DOWN=1` | Configure the Apple FFN reader queue and overlap reads with projection work. |
| `MOTY_Q41_RESIDENT_PREFIX`, `MOTY_Q41_RESIDENT_DOWN=slot` | Select the resident FFN prefix and placement of extra resident `down` matrices. |
| `MOTY_Q41_PREFILL_ATTN=panel`, `MOTY_Q41_STREAM_PREFILL=N` | Keep resident attention panels during prefill and read FFN panels for batches up to `N`. |
| `MOTY_Q41_PANEL_PREFILL=1`, `MOTY_Q41_PREFILL_PAR=1` | Reuse expanded panels within each worker tile and distribute eligible prefill normalization and FP32 KV writes. |
| `MOTY_Q41_SHORT_PREFILL=1` | On FMLAL, use an eight-token panel tile with one output row at a time. Small activation buffers are packed by the calling thread. |
| `MOTY_Q41_AUTO_CHUNK=512` | Allow compatible long Pocket prefill to use bounded chunks. Explicit `PREFILL_CHUNK` takes precedence. |

The optional history lookahead uses existing greedy verification. Use
`MOTY_Q41_LOOKAHEAD=2`, `4` or `auto` to evaluate it separately. Its hit rate
depends on the request, so the RSS results below use ordinary decode.

## Follow-up on prefill at half RSS

Apple M5 Max, pthread build, official GGUF SHA-256
`a281c127e388b16537fd3f5d31ab32d3f87739e6dc514ac1093c80d9eac8d097`.
Each row uses seven pairs after warm-up, with alternating order and
warm version 2 sidecars. The fixed cases generate 128 tokens each.
The eight ordinary prompts generated 201 tokens in total with EOS enabled.
Peak RSS includes file-backed pages and is the maximum from
`/usr/bin/time -l` in each series. Rate and latency changes are paired
medians. TTFT and request time include process startup.

| Workload | Budget (GiB) | Original → candidate RSS (MiB) | Prefill | Decode | TTFT | Request time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eight ordinary prompts, 12 workers | 0.1065 | 205.55 → 100.38 | +3.65% | +6.64% | −15.63% | −10.41% |
| 256 IDs, 12 workers | 0.108 | 224.27 → 108.08 | +2.11% | +5.52% | −5.23% | −5.45% |
| 1536 IDs, 4 workers | 0.145 | 299.14 → 147.34 | +6.72% | +3.45% | −6.46% | −5.98% |
| 3584 IDs, 12 workers | 0.202 | 434.08 → 206.30 | +1.52% | +2.67% | −1.63% | −1.48% |
| 3584 IDs, 12 workers | 0.207 | 434.02 → 215.98 | +2.13% | +4.25% | −2.32% | −2.34% |

The table records the clean pthread build after the Qwen refactor
(SHA-256 `b7a90d1bbfa47fd1a9601111737e4ba79eaea6992465cb525d35cb45f0403997`).
All generated token IDs matched the saved native Q4_1 baseline (SHA-256
`3e70bcb62710c0bad7cb56a0937fd2653371c3a13d0619bd60c15983a0c09178`).
For a fresh checkout, the benchmark defaults to comparing streaming
and resident modes in the same binary. Pass `--baseline /path/to/binary`
to use a saved native baseline. The benchmark records both binary hashes
and the model hash.

## Reproduce on Apple Silicon

Build in a fresh checkout with the pthread backend. The runner uses
Python 3.11 or newer. Prepare both cache
layouts next to the writable copy of the GGUF before timing. The resident
run builds the ordinary Group64 head first. The second run builds the
indexed head while reusing the other sidecars.

```sh
make -C c -j4 THREADPOOL=1 ARCH=apple-m1 moty
for indexed in 0 1; do
  GGUF="$model" QBITS=-1 Q41_ACT=f16 MOTY_Q41_PACKED=all \
    MOTY_Q41_CACHE_HASH=parallel MOTY_Q41_INDEXED_HEAD="$indexed" \
    CTX=512 TEMP=0 NGEN=1 PROMPT='Hello' ./c/moty >/dev/null
done
python3 tools/ref/pocket_bench.py --model "$model" --candidate ./c/moty > pocket-results.json
```

The [benchmark runner](../../tools/ref/pocket_bench.py) checks generated
IDs in each pair and confirms that the candidate mapped both gate and down
panels, then writes prefill and decode rates, latency and RSS
to JSON. Use `--workload text`, `256`, `1536`, `3584` or
`3584-speed` to run a single case. The same
[prompts and token IDs](../../tools/ref/pocket_bench_inputs.json) can
also be used with `PROMPT_IDS` or `PPL`.

## Quality and compatibility

The post-refactor benchmark matched all 585 generated IDs across the
four distinct workloads. The clean build also matched all 2310 top-1 IDs
between resident and streaming execution at perplexity 119.9085.
CTest passed 176 cases with four platform skips. The standalone
Q4_1 and GQA tests passed on Cortex-A53 QEMU.

The optimization series used this quality suite.

| Build | Generated IDs matched | Teacher-forced top-1 IDs matched | Perplexity |
| --- | ---: | ---: | ---: |
| Apple M5 Max, FP16 activations | 457 | 2310 | 119.9085 |
| Cortex-A53 QEMU, FP32 activations | 329 | 2310 | 119.8882 |

The M5 Max logits matched bit for bit at the first generation position
on the fixed inputs and on short prompts covering 15 lengths from 2 to
64. Tests cover panel tails, causal bounds and the 2048-position GQA
transition, chunk overrides, and fallbacks for FP32 activations,
int8 KV and buffered loading. The
[CI workflow](../../.github/workflows/ci.yml) runs the Q4_1 and GQA
kernel tests with `qemu-aarch64 -cpu cortex-a53` and exercises FMLAL,
the pthread backend and the FFN file windows on macOS arm64. Cortex-A53 speed
requires physical hardware.

To repeat the teacher-forced comparison, write the included token IDs
to a PPL input file and compare `PPL_OUT` from the resident and streaming
runs. For example:

```sh
python3 -c 'import json; x=json.load(open("tools/ref/pocket_bench_inputs.json")); print(json.dumps({"ids":x["ppl"]}))' > pocket-ppl.json
GGUF="$model" QBITS=-1 Q41_ACT=f16 THREADS=12 \
  PPL=pocket-ppl.json PPL_OUT=resident.top1 ./c/moty
GGUF="$model" QBITS=-1 Q41_ACT=f16 THREADS=12 CTX=4096 MEM_GB=0.202 \
  MOTY_Q41_PACKED=stream-all MOTY_Q41_CACHE_HASH=parallel \
  MOTY_Q41_FILE_WINDOWS=1 PPL=pocket-ppl.json PPL_OUT=stream.top1 ./c/moty
cmp resident.top1 stream.top1
```

See the [experiment log](mnn-pocket-0.3b-experiments.md) for individual
kernel profiles, comparisons with float32 and int8, and the variants
evaluated along the way.
