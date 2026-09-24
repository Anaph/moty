# CPU inference performance: caches, memory, and cores

*Research note for Moty's dense engines (qwen, gemma). Companion to the
optimizations landed on this branch; numbers from `c/tests/build/bench` unless
stated otherwise.*

## 1. The memory-bandwidth wall

Single-token decode of a dense transformer reads **every weight byte exactly
once per token**: each matrix element participates in one multiply of a GEMV
and is never reused. Arithmetic intensity is ~0.5 FLOP/byte at f32 (2 FLOP per
4-byte weight) — orders of magnitude to the left of any CPU roofline knee. The
consequence is a hard ceiling:

&nbsp;&nbsp;&nbsp;&nbsp;**tok/s ≤ RAM bandwidth ÷ weight bytes per token**

At Qwen3-4B scale (~4B params):

| storage | bytes/token | 25 GB/s (2-ch DDR4) | 60 GB/s (2-ch DDR5) | 250 GB/s (Apple M-max) |
|---|---|---|---|---|
| f32 | ~16 GB | 1.5 tok/s | 3.7 | 15 |
| int8 (`QBITS=8`) | ~4 GB | 6 | 15 | 60 |
| int4 (`QBITS=4`) | ~2 GB | 12 | 30 | 120 |

Everything else is noise at decode: KV-cache reads are ~288 KB/token at 4k
context (0.01% of weight traffic), activations and the DeltaNet S-state are
kilobytes. Cores beyond the point of bandwidth saturation add nothing —
`THREADS` sweeps plateau exactly where aggregate read bandwidth peaks.

**Prefill is the exception.** With S tokens in flight, each weight row can be
applied to all S activations per read — arithmetic intensity scales with S and
the workload becomes compute-bound. This asymmetry drives the batching work
below.

Cache-level view of the hot loops (4B dims): a weight row is 2.5–38 KB — it
transits L1/L2 once and is gone; the only *resident* data are the activation
vector (10–38 KB, stays hot in L1/L2 across all O rows) and, at prefill, the
S×I activation block. The K/V cache layout `[kv_head][t][hd]` streams
t-contiguously per head (full cache-line utilization). There is no blocking
scheme that makes decode weight traffic cacheable — the working set is the
model.

## 2. What this branch implements (measured)

Container: 4 shared cores, AVX512-VNNI, portable-build kernels for the table
(scalar tier — relative effects transfer; native-tier absolute numbers in
`bench` output).

1. **OpenMP hot-thread tuning + re-exec** (`runtime.h: omp_hot_tune`, ported
   measured **66.9 s → 20.9 s** on a 32-core Zen5 matmul workload
   run). Dense decode enters ~250–290 tiny parallel regions per token; with
   the default passive wait policy the team goes to sleep between them.
   `OMP_WAIT_POLICY=active` + `GOMP_SPINCOUNT` + `OMP_PROC_BIND=close` +
   `OMP_DYNAMIC=FALSE` are seeded (never overriding user-set values) and the
   process re-execs once so libgomp picks them up. Kill-switch:
   `MOTY_NO_OMP_TUNE=1`. On the 4-core noisy container the effect is neutral
   (spin competes with oversubscription); the win grows with core count.
2. **`THREADS=N`** — first thread-count knob in the engines; applied before
   load so weight quantization, scratch sizing and all regions obey it.
   `OMP_NUM_THREADS` still works.
3. **Per-model attention scratch** (`Model.att_sc`): the attention region
   used to `malloc`/`free` a per-thread score buffer on **every layer, every
   token** (2×36 allocations/token at 4B); now one buffer sized
   `threads × max_t` lives on the Model, indexed by `omp_get_thread_num()`
   (the big-MoE engine's pattern).
4. **Batched activation quantization** (`matmul_q_s`): the int8 GEMV now
   quantizes all S activation rows once and reads each weight row **once for
   all S tokens** inside a single parallel region. Measured on the container:
   34.5 → 50.9 → **58.9 GFLOP/s** at S = 1 → 8 → 64 with weight traffic
   constant — the prefill weight-reuse win, bit-identical numerics. The
   batched `mlp()` removes S× region re-forks per layer on top.
5. **DeltaNet single region**: conv + recurrence share one parallel region
   per token (was two). Minor; taken because it is free.
6. **int8 embeddings under `QBITS=8`** — the embedding table used to stay
   f32 resident, so with a tied lm_head the largest single GEMV of decode
   (V×D: 622 MB at 0.6B, 1.5 GB at 4B) still streamed f32 bytes after every
   other matrix was quantized. Now the table is quantized per-row at load
   (chunk-wise, so the load transient is one chunk, not the table), the
   input gather dequantizes one row, and the tied head runs the int8 kernel:
   decode bytes/token at `QBITS=8` are now genuinely ~4× below f32.
7. **Micro-RSS mode** (`MICRO=1`, qwen) — the opposite trade: minimum
   resident memory instead of maximum speed. No weight is resident (embedding
   rows gathered per token, every GEMV re-reads its matrix in constant 4 MB
   chunks, page cache dropped after use); output stays bit-identical to the
   resident f32 path. Decode cost becomes *disk* bandwidth ÷ model bytes —
   the same wall as §1 with the disk in place of RAM. For hard cgroup /
   embedded limits where tok/s is secondary.
8. **int4 weights (`QBITS=4`)** — the "single largest available win" from
   the deferred list, now landed: bit-exact-validated int4 kernels
   (packing, exact f32×int4 matmul, group-wise scales `QGROUP`, default 32)
   lifted into the shared core. Layer weights halve again vs int8 (~2 GB at
   4B → ~2× decode ceiling per §1); embeddings and the lm_head deliberately
   stay int8 (the head is the most quantization-sensitive GEMV). The int4
   IDOT kernel (`dot_i4i8`) is ported and tested but not yet wired — the
   exact kernel already wins 4× on weight traffic.
9. **int8 KV cache (`KV_BITS=8`)** — promoted from the deferred list since
   at 4k context the f32 KV (1.2 GB at 4B) rivals quantized weights.
   Per-(head, position) scales, quantize-on-write, one-sided error (queries
   stay f32 via `dot_f32i8`). Works with gemma's sliding windows, shared-KV
   aliasing and k_eq_v.
10. **Chunked prefill (`PREFILL_CHUNK`)** — bounds the S-proportional
   activation peak (mlp scratch alone is 2·S·inter floats) and the permanent
   `matmul_q_s` scratch growth to a constant, bit-identically; opt-in
   because with `MEM_GB` every chunk re-reads the streamed layers.
11. **GGUF reading (`GGUF=`)** — single-file models; Q4_0 repacks
   losslessly onto the int4 kernels, K-quants dequantize on load. Not a
   speed feature per se, but it removes the f32 load transient and lets the
   engine start from pre-quantized files.

## 3. Deferred optimizations, cost/benefit at 4B

Ordered by expected value (items 1 and 6 of the original list have since
landed as §2.8/§2.9 above):

1. **Speculative decoding** — the structural escape from the wall: draft
   cheaply, verify K tokens in one batched forward (weight bytes amortize
   over K like prefill). Qwen3.5's cheap linear layers or an n-gram draft
   both fit; a working MTP/n-gram speculation loop exists in the tree to model on.
2. **Weight interleave for VNNI** — reorder int8 rows so the dot kernel loads
   are perfectly sequential across the unrolled accumulators (llama.cpp /
   [Neural Speed](https://arxiv.org/abs/2411.19542)-style fused layouts
   reach >90% of bandwidth on INT4 GEMV). Moderate win over the current
   row-major int8 (already sequential per row); real gain appears with int4.
3. **NUMA placement** — first-touch or interleaved weight allocation +
   binding the team per socket; only matters on multi-socket / chiplet-split
   machines ([ArcLight](https://arxiv.org/abs/2603.07770) reports the
   cross-NUMA bottleneck dominating many-core CPU inference).
4. **Hugepages** — `MADV_HUGEPAGE` on the big weight buffers cuts TLB misses
   during streaming; single-digit % on Linux, ~10 lines.
5. **KV-cache paging** — int8 KV landed (§2.9); page-level eviction only
   matters at 100k+ context, and the hybrid architectures (DeltaNet, sliding
   windows) already bound this structurally.
6. **Async prefetch of streamed layers** — the MEM_GB path already issues
   `WILLNEED` for layer i+1; a dedicated I/O thread
   ([async KV prefetching](https://arxiv.org/abs/2504.06319) analog) could
   overlap more aggressively.

## 4. LoRA training cost model

Training (TRAIN mode, f32 base required) is bandwidth-heavy in one place: the
chunked cross-entropy head. Per window of S tokens with chunk Sc, the lm_head
matrix (V×D ≈ 1.5 GB f32 at 4B) streams once forward and once backward per
chunk: ≈ 2·(S/Sc)·1.5 GB ≈ 50 GB per 512-token window at Sc=32 — a few
seconds at desktop bandwidth, dominating the window unless the trained-layer
count is large. The trained-layer backward is ~2× the forward FLOPs of those
layers; activation stash ≈ 124 MB per trained layer at S=512 (dominated by
the H×S×S attention probabilities). Adapter/optimizer state is megabytes.
Practical guidance: keep `TRAIN_CTX` moderate (256–512), raise
`TRAIN_CE_CHUNK` if RAM allows, and prefer few high-layers over many.

## 5. Small ARM boards: int4 on Cortex-A53

Target: **Rockchip RV1126B, 4× Cortex-A53 @ 1.6 GHz (ARMv8.0: NEON, no
dotprod, no fp16 arithmetic), ~1 GB RAM**, static aarch64 binaries
(`ARCH=armv8-a`). The boards ran their own vision workload during every
measurement (≈20–25 % CPU plus ISP memory traffic), so absolute numbers
carry that load; all comparisons below are interleaved runs on the same
board, and two boards with different background load are reported
separately (A: heavier, B: lighter). Prompt 66 tokens, 64 generated
tokens with `IGNORE_EOS=1`, median of 3 interleaved repetitions, peak RSS
from `ru_maxrss` (`tools/a53/rssrun.c`), load time from the engine banner.

### 5.1 Machine limits (`c/tests/bench_a53.c`)

| | board A | board B |
|---|---|---|
| read bandwidth, 4 threads, `PRFM` 1–2 KB ahead (median) | 3.1–3.3 GB/s | 4.3–4.8 GB/s |
| int8 `SMLAL` peak, 4 threads | 36.7 GMAC/s (7.2 MAC/cycle/core) | — |

Decode is bandwidth-bound (§1): the ceiling is bandwidth ÷ weight bytes
per token. Prefill is bound by the kernel: per group the GEMM spends ~16
cycles in `SMLAL` and ~18 in the horizontal reduction and scaling
(`ADDP`/`SADDLP`/`SCVTF`/`FMLA`); an inline-asm body without register
spills was bit-identical but slower, so the intrinsics stay.

### 5.2 Throughput

**LFM2.5-350M** (board A, int4 loaded from the bf16 snapshot):

| engine | format | threads | prefill tok/s | decode tok/s | peak RSS | load |
|---|---|---|---|---|---|---|
| moty | int8 (`QBITS=8`) | 1 / 2 / 4 | 8.0 / 15.1 / 21.0 | 3.24 / 5.34 / 5.58 | 385 MB | 10–16 s |
| moty | **int4 Q4R4** (`QBITS=4`) | 1 / 2 / 4 | **11.8 / 23.6 / 34.1** | **6.05 / 10.56 / 10.21** | 300 MB | 13–23 s |
| llama.cpp | Q4_0 | 1 / 2 / 4 | 5.35 / 10.57 / 16.58 | 3.45 / 6.22 / 6.88 | 242 MB | — |

On board B the same int4 binary from a `SAVE_PACKED` container: 40.6
prefill / 14.65 decode tok/s at 4 threads, load 3.1–3.7 s.

**MiniCPM5-1B** (board B, int4 from a `SAVE_PACKED` container with
`EMBED=disk`):

| engine | format | threads | prefill tok/s | decode tok/s | peak RSS | load |
|---|---|---|---|---|---|---|
| moty | **int4 Q4R4** | 1 / 2 / 4 | **4.9 / 10.0 / 16.8** | **2.78 / 4.76 / 6.20** | 556 MB | 3.9–4.6 s |
| moty | int4 Q4R4, packed at load from bf16 | 4 | 18.0 | 6.72 | 555 MB | 23.9 s |
| llama.cpp | Q4_0 | 1 / 2 / 4 | 2.14 / 4.21 / 7.29 | 1.45 / 2.74 / 4.05 | 686 MB | — |

The int8 path does not fit: its resident peak is 930 MB (measured on
x86 with `EMBED=disk`) on a ~1 GB board. Run through the `MEM_GB`
streamer instead it re-reads and re-quantizes bf16 layers from eMMC every
token (0.05–0.08 tok/s) — int4 is what makes a 1B model usable here.
With the int8 embedding table resident (`EMBED=ram`) the int4 run peaks
at 751 MB, too close to the limit on a board that also runs its own
workload; `EMBED=disk` saves the 200 MB table.

### 5.3 Against the ceilings

| board | model | format | bytes/token | decode ceiling | measured (best) | share |
|---|---|---|---|---|---|---|
| A | LFM2.5-350M | moty int4 | 199 MB | 15.6–16.6 | 10.56 | 64–68 % |
| A | LFM2.5-350M | llama.cpp Q4_0 | 217 MB | 14.3–15.2 | 6.88 | 45–48 % |
| A | LFM2.5-350M | moty int8 | 354 MB | 8.7–9.3 | 5.58 | 60–64 % |
| B | LFM2.5-350M | moty int4 | 199 MB | 21.6–24.1 | 14.65 | 61–68 % |
| B | MiniCPM5-1B | moty int4 | 495 MB | 8.7–9.7 | 6.20 | 64–71 % |
| B | MiniCPM5-1B | llama.cpp Q4_0 | 660 MB | 6.5–7.3 | 4.05 | 56–62 % |

Prefill reaches 27–32 % of the `SMLAL` peak (LFM2.5 287 M MAC/token ×
34.1–40.6 tok/s; MiniCPM5 680 M × 16.8). Against llama.cpp at 4 threads:
prefill ×2.1 (LFM2.5) / ×2.3 (MiniCPM5), decode ×1.5 on both.

### 5.4 What moved the numbers (interleaved A/B on the board)

- **Kernel choice** (`bench_a53 gemv`, 45 MB of int4 weights, cold): the
  4-row Q4R4 block with 1 KB `PRFM PLDL1KEEP` streamed 2.28 GB/s of
  weights at 4 threads vs 1.65 for the existing row-major int4 kernel
  (hot cache: 11.6 vs 7.4 G weights/s).
- **Row ops in NEON** (`hw/hw_ops.h`) + RoPE table, LFM2.5 prefill
  ms/token: SiLU·up 7.1 → 2.0 (scalar `expf` was the third-largest item),
  conv depthwise 1.5 → 0.3, QK-norm + RoPE 1.3 → 0.1, attention core
  1.3 → 0.5, RMSNorm 1.1 → 0.3.
- **Fused q/k/v and gate/up** (one GEMV and one parallel region instead of
  three / two): prefill 31.2 → 33.5, decode 9.3 → 9.6 tok/s.
- **Prefetching the f16 scale stream** too: decode 9.0 → 10.3 tok/s.
- **Pre-packed container** (`SAVE_PACKED`): load 23.9 → 4.6 s for
  MiniCPM5-1B on the board (x86: 13.1 → 1.9 s; LFM2.5 3.5 → 0.5 s),
  bit-identical output.
- Tried and rejected: group size 64 (fewer non-MAC instructions per
  weight, but LFM2.5 KL 0.75 → 1.00), an inline-asm GEMM body (slower),
  a single interleaved weight+scale stream (no faster than two prefetched
  streams, needs a format change). OpenMP hot-thread tuning was neutral
  on this board.

### 5.5 Quality

Teacher-forced over 2047 tokens of mixed text (prose, Python, Russian,
Markdown; `tools/ref/ppl_text_long.txt`, `PPL=` / `hf_ppl.py`), on x86. Top-1 is agreement of the
argmax with HF transformers f32 at every position.

| model | path | PPL | top-1 vs f32 |
|---|---|---|---|
| LFM2.5-350M | HF f32 | 245.4 | — |
| LFM2.5-350M | moty int8 (`QBITS=8`) | 230.3 | 83.9 % |
| LFM2.5-350M | **moty int4 Q4R4** | **343.2** | **61.5 %** |
| MiniCPM5-1B | HF f32 | 24.45 | — |
| MiniCPM5-1B | moty int8 | 24.90 | 90.3 % |
| MiniCPM5-1B | **moty int4 Q4R4** | **30.10** | **71.6 %** |

At f32 both engines match transformers: 32/32 greedy tokens on four
prompts (English, code, Russian, arithmetic) and max |Δlogit| ≤ 5e-5. The
board (NEON) and x86 (scalar reference) int4 paths agree: LFM2.5 PPL on
a 290-token text 175.3 vs 176.6, argmax equal at 281 of 290 positions.

**Be aware of the int4 loss on LFM2.5-350M.** A 350M model loses a lot at
4 bits: perplexity +40 % and the int4 argmax differs from f32 at almost 4
positions in 10. This is the weight quantization itself, not the kernels
or the int8 activations: HF fake-quant of the weights alone with the
same rule gives PPL 319 / 62.4 % top-1 (`tools/ref/hf_qsim.py`), and the
no-clip scale rule is already the best of the group-32 rules tried
(the Q4_0 rule 60.4 %, symmetric amax/7 56.9 %, same fake-quant). The
llama.cpp Q4_0 file uses the Q4_0 rule, so it should land in the same
place; its perplexity was not measured here. For LFM2.5-350M prefer int8 where memory allows (it is ~2×
slower on decode); MiniCPM5-1B tolerates int4 much better. Better int4
quality would need calibration-based methods (e.g. AWQ/GPTQ-style scale
search on activations), which this loader does not do.

### 5.6 llama.cpp reference build

llama.cpp at commit 42916d8, models converted with its
`convert_hf_to_gguf.py` and `llama-quantize Q4_0` (LFM2.5 207 MiB,
MiniCPM5 629 MiB, 4.9 bits/weight). Built statically for aarch64 with
`-mcpu=cortex-a53` (no OpenMP, its own thread pool); the binary contains
no `sdot`/`udot`/`smmla` and no fp16 vector arithmetic, i.e. the same
instruction class as moty's kernels. Numbers from `llama-bench -p 66 -n 64`
with `-t 1/2/4`, interleaved with the moty runs.

### 5.7 Second speed pass (LFM2.5-350M, board B)

Same method (board B, its vision workload running, container load,
prompt 66 / 64 generated tokens, `IGNORE_EOS=1`, medians of 3 interleaved
repetitions; peak RSS 300 MB in every row, 309 MB with `HEAD_TOPK`):

| threads | build | prefill tok/s | decode tok/s |
|---|---|---|---|
| 1 | before | 12.7 | 7.28 |
| 1 | after | **14.1** | **8.36** |
| 2 | before | 24.9 | 12.55 |
| 2 | after | **27.9** | **14.12** |
| 4 | before | 43.1 | 14.65 |
| 4 | after | **48.0** | 14.92 |
| 4 | after, `THREADS_DECODE=3` | 47.7 | 15.35 |
| 4 | after, `THREADS_DECODE=3 HEAD_TOPK=512` | 44.6 | **17.24** |

(Prefill in the last row: one run hit a background burst, 46.0/44.6/35.6;
the knobs do not touch prefill.)

Per-op decode profile, ms/token (`-DMOTY_PROF` builds, median of 4; the
"before" runs caught more background load, 67–81 ms in other rounds):

| op | before | after (`THREADS_DECODE=3 HEAD_TOPK=512`) |
|---|---|---|
| ffn gate+up / down | 27.7 / 18.0 | 24.5 / 12.8 |
| conv in / out | 8.5 / 3.7 | 5.6 / 2.1 |
| qkv / o | 3.9 / 1.7 | 2.3 / 1.2 |
| lm_head | 11.1 | 6.1 |
| attention core | 2.1 | 1.2 |
| silu·up | 2.1 | 1.2 |
| total | 80.8 | 58.2 |

Prefill profile: 23.1 → 21.2 ms/token (gate+up 10.6 → 9.8, down 6.1 →
5.7, attention core 0.33 → 0.17, silu·up 0.64 → 0.47).

What paid off, each measured on the board:

- **Decode GEMV, two groups per iteration.** With a hot cache the
  one-group kernel spent 68 cycles per 4×32 group: the in-order core does
  not overlap one group's reduction chain with the next group's
  multiplies. Two groups per loop body: 56 cycles; one core 1.51 → 1.77
  GB/s of weights, two cores 2.67 → 3.04.
- **Prefill GEMM with vector scales** (`gemm4t`): 156 → ~124 instructions
  per group, 4.30 → 4.90 GMAC/s per core (I = 1024).
- **Four exp chains per iteration** in silu·up and softmax (80 → 35 µs
  for 4608 elements) and **register-resident attention rows** (one head at
  130 positions 32.7 → 13.4 µs).
- **`THREADS_DECODE=3`.** An empty OpenMP region costs 8 µs median at any
  team size here, but with 4 threads the mean is 36–51 µs (p99 0.3–1 ms):
  a thread that shares its core with the vision workload gets preempted
  and every decode matmul ends in a barrier that waits for it (one core
  was 38 % busy with that workload, the others 11–15 %). 3 threads: mean
  10–12 µs. Decode is bandwidth-bound, so the 4th thread adds little
  bandwidth but all of the stall risk; prefill (compute-bound) keeps 4.
- **`HEAD_TOPK=512`**: the head (37.7 of the 199 MB per token) 10.3 →
  5.5 ms. Opt-in because it is not exact: over 2047 teacher-forced
  positions the argmax equals the full int4 head's at 99.76 % (K=512),
  99.41 % (K=256). Candidate generators rejected on HF hidden states: an
  SVD of the head (rank 256: 94 % top-1 recall in the top 512).

Tried, no gain (measured): `-mtune=cortex-a53` (same cycles), reciprocal
estimate + Newton instead of the SiLU division (38 vs 35 µs), a parallel
silu·up in decode (slower than serial on the shared cores), far L2
prefetch and 768/1536/2048-byte prefetch distances in the GEMV, two
4-row blocks per GEMV iteration (faster hot, 1.86 → 1.43 GB/s cold: two
interleaved weight streams), blocking the prefill GEMM along I (4.31 →
4.0–4.2 GMAC/s), a compiler barrier against accumulator spills in the
GEMM (no change: spills are not the bottleneck), pinning the decode
threads to the three least busy cores (within noise).

Where the remaining gap goes:

- **Decode** reaches 17.2 tok/s; the matmuls stream ~3.1–3.3 GB/s against
  ~4.4 GB/s of read bandwidth with 3 threads. The in-order GEMV itself
  caps one core at ~1.8 GB/s with a cold cache (56 cycles per 72-byte
  group), the vision workload takes ~20 % of the CPU time, and barriers
  still wait for preempted threads — together that is the ~25 % gap. The
  rest of a decode step (norms, conv, attention, silu, sampling) is 3.4
  ms of 58.
- **Prefill** reaches 48 tok/s = 13.8 GMAC/s, 38 % of the 36.7 GMAC/s
  `SMLAL` peak. Per 4×4×32 tile the GEMM executes 64 multiply
  instructions and ~60 others: the unpack of the nibbles, the pairwise
  reduction of the int16 lanes to one int32 per row (3 `ADDP` + `SADDLP`
  per token — already minimal for 4 rows), the int→float conversion and
  the two scale FMAs. That is the cost of group-32 scales on both
  weights and activations; at ~3 MAC/cycle per core and ~80 % of the CPU
  available the matmuls run at their ceiling.

### 5.8 Quantization study: which int4 for LFM2.5-350M

Evaluation: teacher-forced over 2047 tokens of `tools/ref/ppl_text_long.txt`;
PPL, top-1 agreement with HF transformers f32 (PPL 245.4) and, in the HF
fake-quant simulation, mean KL(f32 ‖ quantized). Calibration (imatrix,
AWQ, GPTQ): the first 2048 tokens of `tools/ref/calib_text.txt`, disjoint
from the evaluation text. The simulation reproduces moty: Q4R4 simulated
PPL 339 / top-1 61.6 % vs moty 343.2 / 61.5 %.

**Sensitivity** (`hf_sens.py`: one group int4, the rest f32, KL):
layer 0 0.36 and layer 1 0.35 dominate, then layer 2 0.11 and 0.03–0.08
for the rest; by type w2 0.23, w3 0.20, v 0.12, w1 0.12, conv out 0.11,
attention out 0.07, conv in 0.06, q/k 0.03, lm_head 0.013 (the head is
not the problem), int8 embedding 0.001. Some input channels carry 100–800×
the median mean-square activation.

**Weight schemes** (`hf_qstudy.py`, every Linear + head, int8 embedding):

| scheme | MB/token | PPL | KL | top-1 |
|---|---|---|---|---|
| Q4R4 (sym, g32, f16 scale, no-clip rule) | 199.4 | 339 | 0.773 | 61.6 % |
| sym g16 / g64 / g128 | 221.5 / 188.3 / 182.7 | 151 / 320 / 1478 | 0.898 / 1.030 / 1.975 | 59.7 / 54.9 / 43.8 % |
| g32, int8 scales × f32 per row | 188.3 | 311 | 0.785 | 61.7 % |
| asymmetric g32 (Q4_1: scale + min) | 221.5 | 1863 | 1.807 | 47.1 % |
| Q5 g32 | 243.7 | 210 | 0.195 | 78.4 % |
| Q8 g32 / int8 per row | 376.6 / 354.4 | 268 / 234 | 0.007 / 0.017 | 95.6 / 92.6 % |

Finer groups and Q4_1 lower the reconstruction error of every matrix (and
help alone: layer 0 at g16 or Q4_1 costs KL 0.11 instead of 0.36), yet
lose on the whole model; the damage is concentrated in FFN w3 (KL 0.20 at
g32 → 0.30 g16 → 0.68 Q4_1). Q5 is the best 5-bit option in quality but
not on the A53: a Q5 GEMV (nibbles + a high-bit plane) takes 113 cycles
per group instead of 56 and decodes a 9216×1024 matrix 1.6× slower at 4
threads for 22 % more bytes.

**Calibration** (on Q4R4, same format): imatrix-weighted scale search KL
1.058 and AWQ 0.846 are *worse* than round-to-nearest (0.773) — with
input-channel outliers this large the scale search trades exact extremes
for weighted MSE; **GPTQ** 0.606, top-1 66.1 % (moty: PPL 285.1, top-1
64.6 %), no format change. imatrix helps only the finer formats (g16
0.592, Q4_1 0.576).

**Pareto on board B** (moty; RV1126B, 4 threads, decode on 3,
container load; speed medians of 6 interleaved runs; KL from the
simulation with moty's int8 activations):

| layout (`pack_r4.py`) | MB/token | prefill tok/s | decode tok/s | RSS | PPL | top-1 | KL |
|---|---|---|---|---|---|---|---|
| Q4R4, round-to-nearest (before) | 199.4 | 45.7 | 15.89 | 300 MB | 343.2 | 61.5 % | 0.789 |
| Q4R4, GPTQ | 199.4 | 41.2* | 16.09 | 300 MB | 285.1 | 64.6 % | 0.601 |
| **GPTQ + Q8R4 layers 0–1** | 217.8 | **45.2** | **15.51** | 318 MB | **219.9** | **70.5 %** | 0.454 |
| GPTQ + Q8R4 layers 0–1 + all w2 | 250.7 | 38.3 | 13.91 | 350 MB | 284.8 | 72.3 % | 0.332 |
| Q8R4 everywhere | 376.6 | 37.4 | 9.39 | 470 MB | 255.1 | 94.7 % | 0.013 |
| existing int8 (`QBITS=8`, per-row activations) | 354.4 | — | — | — | 230.3 | 83.9 % | — |

\* The two Q4R4 rows are the same format and kernels; their prefill
difference is background load (runs 34–48 tok/s for both).

**Recommendation for LFM2.5-350M on the A53: GPTQ-calibrated Q4R4 with
Q8R4 for layers 0 and 1** (`pack_r4.py --method gptq --q8
"model.layers.0.*,model.layers.1.*"`, or `Q8_TENSORS=` the same globs for
a round-to-nearest pack inside moty). +9.2 % bytes, decode −2–3 %, prefill
within noise, and top-1 agreement with f32 61.5 → 70.5 %, PPL 343 → 220.
The two sensitive layers carry most of the int4 damage; GPTQ removes a
further share at no cost. If quality matters more than speed, Q8R4
everywhere reaches 94.7 % top-1 at 60 % of the int4 decode speed — and
beats the existing `QBITS=8` path (83.9 %) because its activations are
quantized per group of 32 instead of per row.

### 5.9 LFM2.5-VL-450M: the language model on the board, images from an NPU encoder

The vision tower and projector run elsewhere (here: converted to RKNN
for the RV1126B NPU by a separate effort); moty runs the LFM2 language
model of `LiquidAI/LFM2.5-VL-450M` from its own checkpoint and takes the
256 projected image rows through `EMBEDS=` at the image-token positions
of the processor's prompt (`PROMPT_IDS=`, 272 tokens for one 512×512
tile and "Describe the image.").

Exactness (x86, f32): with the HF image features moty matches HF's first
logits to 3 decimals (top-5 identical) and its 64-token greedy answer
word for word; COCO val2017 #39769 through `tools/ref/hf_vl_ref.py`: 15/15
tokens, max |Δlogit| 1.3e-5. With the NPU encoder's features (cosine to
HF: mean 0.990, min 0.61; relative L2 0.21) the f32 answer is still
identical to HF's.

On board B (4 threads, decode on 3, container load; medians of 2):

| LM format | image rows | load | prefill 272 tok (= time to first token) | decode tok/s | peak RSS | answer tokens = HF (teacher-forced, of 64) |
|---|---|---|---|---|---|---|
| Q4R4 (RTN) | HF | 3.0 s | 43.4 tok/s (6.3 s) | 15.29 | 309 MB | 60 |
| Q4R4 (RTN) | NPU | 2.1 s | 47.5 tok/s (5.7 s) | 14.85 | 309 MB | 60 |
| GPTQ + Q8R4 layers 0–1 | HF | 3.3 s | 45.2 tok/s (6.0 s) | 14.50 | 327 MB | 56 |
| GPTQ + Q8R4 layers 0–1 | NPU | 2.7 s | 42.8 tok/s (6.4 s) | 14.28 | 327 MB | 55 |

64 answer tokens are a small sample; on the 2047-token evaluation text
(x86, top-1 against moty f32, PPL 34.0) the VL language model gives
Q4R4 PPL 39.7 / 81.3 % and GPTQ + Q8R4 layers 0–1 PPL 34.8 / 86.2 % — the
same recommendation as for LFM2.5-350M holds, with a smaller gap (this
model loses far less to int4: 81 % vs 61.5 % top-1). On image inputs it
does not — §5.14 replaces this recommendation for the VL case.

### 5.10 LFM2.5-VL-450M end to end on one board: NPU encoder + moty

Board B (RV1126B, its vision workload running), one 512×512 tile, prompt
"Describe the image." (272 tokens, 256 image rows):

1. `tools/rknn/rknn_encode` (librknnrt 2.3.2 via `dlopen`, not part of
   moty's build) runs the vision tower + projector compiled for the NPU
   (fp16; rknn-toolkit2 2.3.2 from an ONNX export of the HF modules, whose
   f32 output matches HF's `get_image_features` to cosine 0.99999999; the
   toolkit simulator gives cosine ≥ 0.993 vs HF). On the board its output
   is bit-identical to the other team's NPU run of the same model; vs HF
   f32: cosine mean 0.990, min 0.61 (fp16 on the NPU; see §5.9 for the
   effect on answers).
2. moty `lfm2` with `EMBEDS=` (the recommended GPTQ + Q8R4 container).

`tools/rknn/vl_pipeline.sh`, two cold runs (each step a fresh process):

| step | time |
|---|---|
| NPU encoder: init / encode | 1.8 s / 1.52 s (3.6 s wall for the step) |
| LM: load / prefill 272 tokens | 3.2–3.9 s / 6.4–6.7 s |
| time to first token, cold (encoder + LM load + prefill) | ≈ 13.5 s |
| time to first token, LM already loaded | ≈ 8 s (encode + prefill) |
| decode | 13.9–14.4 tok/s |
| peak RSS of the LM | 327 MB |
| MemAvailable while the encoder is loaded | −425 MB (DMA buffers; fully returned at process exit, not at `rknn_destroy`) |

Run the encoder as its own process before the LM: together they would
need ~750 MB on a ~1 GB board that also runs its own workload.

The answer ("The image shows a cluttered desk with various electronic
devices and cables. In the top left corner, there's a box labeled
"GOTSCHLICH" …") describes the same scene as HF's ("… a cluttered
workspace … a white cord with a black plug …"); teacher-forced on HF's
64-token answer the board agrees on 55/64 tokens with the NPU rows (60/64
with Q4R4, §5.9).

**NPU sharing.** The board's own detector used the NPU at 0 % load during
these runs (`/proc/rknpu/load`), so the encoder ran at full speed. To see
what a busy detector would suffer, its model (960×544 input, 67–73
ms/frame alone) was run at the same time as the VL encoder: detector mean
73 → 125.5 ms per frame (+72 %, frames queue behind the encoder; best
frame unchanged), VL encode 1522 → 1555 ms (+2 %). A deployment that needs
steady detector latency should schedule the VL encoder between detector
frames or accept ~1.5 s NPU pauses.

### 5.11 MiniCPM5-1B: GPTQ and Q8R4 mixes

Same method as §5.8: teacher-forced over 2047 tokens of
`ppl_text_long.txt` against HF transformers f32 (PPL 24.45), GPTQ
calibrated on the first 2048 tokens of `calib_text.txt`, KL from the
`hf_qstudy.py` simulation with moty's int8 activations (it reproduces
moty: round-to-nearest Q4R4 simulated PPL 30.11 / top-1 71.9 % vs moty
30.10 / 71.6 %).

**Sensitivity** (`hf_sens.py`, KL with one group int4): unlike LFM2.5,
no layer stands out — layers 4, 0, 1, 23 lead with 0.024–0.029, the rest
are close behind. By type: FFN down 0.073, attention out 0.054, v 0.050,
FFN up 0.049, gate 0.036, lm_head 0.028, q 0.027, k 0.012; an int8 head
or embedding costs 0.0004 / 0.0001. The v and k projections are small
(2 KV heads × 128 = 256 rows each), so v is the cheapest int8 per unit of
KL; k rides along for 0.6 % more bytes.

**Pareto on board B** (RV1126B, its vision workload running; prompt 66
tokens, 64 decode tokens; 4 threads, decode on 3, `EMBED=disk`; speed
medians of 3 interleaved runs; PPL / top-1 measured with moty on x86 from
the same containers, KL simulated):

| layout (`pack_r4.py`) | MB/token | prefill tok/s | decode tok/s | peak RSS | PPL | top-1 | KL |
|---|---|---|---|---|---|---|---|
| Q4R4, round-to-nearest (before) | 495.0 | 19.2 | 6.23 | 556 MB | 30.10 | 71.6 % | 0.302 |
| Q4R4, GPTQ | 495.0 | 19.2 | 6.23 | 556 MB | 27.26 | 79.0 % | 0.178 |
| **GPTQ + Q8R4 v, k** | 504.4 | **19.1** | **6.31** | 565 MB | 27.66 | **81.5 %** | 0.159 |
| GPTQ + Q8R4 v, k, attention out | 542.2 | 16.8 | 5.93 | 602 MB | 27.60 | 82.6 % | 0.135 |
| GPTQ + Q8R4 FFN down | 579.9 | 16.2 | 5.50 | 637 MB* | 26.18† | 82.2 %† | 0.131 |
| GPTQ + Q8R4 layers 0, 1, 4, 23 | 551.6 | — | — | — | 27.06† | 81.2 %† | 0.155 |
| Q8R4 everywhere | 935.0 | — | — | — | 24.42† | 96.3 %† | 0.003 |

Speed and RSS were measured with round-to-nearest containers of each
layout; GPTQ changes code values only (same bytes, same kernels), so the
GPTQ rows take the speed of their layout. The round-to-nearest v, k
container gives PPL 29.40 / top-1 73.6 %: GPTQ is the larger step.
\* MemAvailable fell to 135 MB during each of the three runs — below the
150 MB margin kept for the board's own workload, so this layout does not
fit next to it. † simulation only. Q8R4 everywhere (≈ 1 GB of weights)
does not fit on the board. Decode on all 4 threads instead of 3: 5.44
tok/s (−13 %), as for LFM2.5.

**Recommendation for MiniCPM5-1B on the A53: GPTQ-calibrated Q4R4 with
Q8R4 for the v and k projections** (`pack_r4.py --method gptq --q8
"*.self_attn.v_proj.weight,*.self_attn.k_proj.weight"`): +1.9 % bytes,
prefill and decode within noise of plain int4, top-1 agreement with f32
71.6 → 81.5 %. The moty PPL of this row is 0.4 above plain GPTQ while
KL, simulated PPL (26.99 vs 27.21) and top-1 all favour it — a small
difference in the other direction on one 2047-token text. Adding the
attention output buys another 1.1 points of top-1 for −6 % decode and
−12 % prefill; FFN down is the most sensitive type but also the largest
and does not fit next to the board's workload. Layer-wise mixes, which
worked for LFM2.5, do not pay here: layers 0, 1, 4, 23 in int8 cost 6×
the bytes of v, k for similar quality.

### 5.12 moty vs llama.cpp on the board, LFM2.5-350M and MiniCPM5-1B

Board B (RV1126B, its vision workload running), llama.cpp 42916d8 built
as in §5.6 plus `Q4_K_M` from the same `llama-quantize`. One prompt of 66
tokens, 64 decode tokens; moty on 4 threads with decode on 3
(`THREADS_DECODE=3`, its best setting), llama.cpp at `-t 3` and `-t 4`
(`llama-bench -p 66 -n 64`); medians of 3 interleaved runs, 24 runs in
all, none near the memory margin. moty loads its `SAVE_PACKED` container
(`EMBED=disk` for MiniCPM5); llama.cpp mmaps the GGUF.

Quality in one protocol, llama-perplexity's: `ppl_text_long.txt` in
1024-token chunks with BOS at each chunk start, the second half of each
chunk scored (1022 positions). moty: top-1 agreement with HF
transformers f32 on exactly those positions; llama.cpp: `Same top p`
against its own f16 GGUF (`--kl-divergence`), which is an easier target
than HF f32 — the comparison favours llama.cpp slightly.

| model | engine | format | weights on disk | prefill tok/s | decode tok/s | peak RSS | top-1 |
|---|---|---|---|---|---|---|---|
| LFM2.5-350M | moty | Q4R4 | 334 MB | **46.6** | **15.80** | 300 MB | 56.9 % |
| LFM2.5-350M | moty | **GPTQ + Q8R4 layers 0–1** (§5.8) | 352 MB | **45.1** | **15.32** | 318 MB | **70.4 %** |
| LFM2.5-350M | llama.cpp | Q4_0, t3 / t4 | 219 MB | 15.4 / 18.6 | 9.39 / 10.17 | 242 MB | 58.4 % |
| LFM2.5-350M | llama.cpp | Q4_K_M, t3 / t4 | 229 MB | 15.3 / 19.1 | 8.99 / 9.69 | 252 MB | 64.2 % |
| MiniCPM5-1B | moty | Q4R4 | 896 MB | **19.0** | **5.95** | 556 MB | 69.1 % |
| MiniCPM5-1B | moty | **GPTQ + Q8R4 v, k** (§5.11) | 906 MB | **18.1** | **6.32** | 565 MB | **82.5 %** |
| MiniCPM5-1B | llama.cpp | Q4_0, t3 / t4 | 665 MB | 6.1 / 7.5 | 3.76 / 3.87 | 686 MB | 72.9 % |
| MiniCPM5-1B | llama.cpp | Q4_K_M, t3 / t4 | 688 MB | 6.1 / 6.8 | 3.68 / 3.95 | 707 MB | 81.8 % |

At equal quality the comparison is between moty's recommended layouts
and llama.cpp's Q4_K_M: prefill 2.4× (LFM2.5) / 2.7× (MiniCPM5), decode
1.6× / 1.6× faster, top-1 +6 / +0.7 points. The llama.cpp build uses
the same instruction class (§5.6); the per-op profiles of §5.4 and §5.7
show where moty's time goes, llama.cpp's were not profiled here. moty's
disk size is larger because the container keeps the embedding table in
the snapshot's bf16 (LFM2.5 134 MB, MiniCPM5 401 MB), of which a decode
step reads one row (`EMBED=disk` keeps it out of RAM for MiniCPM5). llama.cpp's peak
RSS is lower for LFM2.5 (242 vs 300 MB) and higher for MiniCPM5 (686–707
vs 556–565 MB). Loading: moty 3.3–4.2 s from the container;
`llama-bench` does not report a load time (§5.2).

### 5.13 Without OpenMP: the `THREADPOOL` build

A toolchain without an OpenMP runtime (the board vendor's clang ships no
target libomp: `-fopenmp` fails to link with `undefined symbol:
__kmpc_fork_call`, reproduced with clang 18.1.6 via `zig cc`) used to
build moty single-threaded. `make THREADPOOL=1` runs the same parallel
loops on moty's own pthread pool (docs/architecture.md, "Parallel loops").

Board B, LFM2.5-350M (GPTQ + Q8R4 layers 0–1 container), prompt 66 /
decode 64 tokens, interleaved runs; "main" is the OpenMP binary of the
tree before the change, "OpenMP" / "pool" the same tree built with gcc
12.2 both ways (static), "clang pool" `zig cc` 0.13 (clang 18.1.6,
`-mcpu=cortex-a53`, dynamic glibc, no libomp), prefill / decode tok/s:

| threads | main | OpenMP | pool | clang pool | runs each |
|---|---|---|---|---|---|
| 1 | 13.6 / 7.59 | 13.7 / 7.66 | 14.2 / 8.56 | 15.0 / 8.60 | 3 |
| 2 | 26.8 / 12.71 | 26.8 / 13.45 | 26.8 / 13.60 | 28.3 / 13.54 | 3 |
| 3 | 39.1 / 15.02 | 38.9 / 15.58 | 38.8 / 15.68 | 41.0 / 15.88 | 3 |
| 4 | 46.0 / 14.23 | 45.3 / 15.13 | 45.5 / 15.57 | 48.1 / 15.36 | 9 / 27 / 27 / 9 |
| 4, `THREADS_DECODE=3` | 42.9 / 14.68 | 45.7 / 15.38 | 42.4 / 14.28 | 48.1 / 15.19 | 6 |

Medians. On this board single runs at 4 threads scatter by ±15 % (the
board's own workload shares the cores): over 27 runs each the pool and
OpenMP are equal (prefill 45.5 vs 45.3, best 46.9 vs 46.6; decode 15.57
vs 15.13); the 6-run `THREADS_DECODE=3` row is within that scatter (pool
decode runs alternate 13.2–13.3 / 15.2–15.4 like OpenMP's 13.3–13.5 /
15.3–15.8). The clang build is 4–6 % faster in prefill from code
generation alone. Unpinned workers (`MOTY_POOL_PIN=0`) decode slower
(14.04, 8 runs). The pool build does not re-exec itself (no OpenMP
environment to seed).

### 5.14 LFM2.5-VL-450M: which int4 on image inputs

(The whole recipe, scripts and results in one place:
[docs/models/lfm2.5-vl-450m.md](models/lfm2.5-vl-450m.md).)

On live tiles from the other board (NPU encoder rows, prompt "Describe the
image.", greedy 64 tokens) the GPTQ + Q8R4-layers-0–1 container recommended
for text (§5.8, §5.10) diverged from the f32 language model after ~20
tokens and invented objects (a computer monitor, keyboard and mouse), while
the f32 LM on the same rows kept its answer. Reference below: the f32 LM on
the same NPU rows (the NPU encoder alone already changes answers vs HF's own
encoder, which no LM quantization can repair). 33 live tiles (one night-IR
scene over an hour, `xchg`), HF fake-quant with moty's int8 activations
(`tools/ref/hf_vlq.py`) and, for the finalists, moty itself on x86
(greedy leading tokens equal to the reference, teacher-forced top-1 over the
64 answer positions). Decode on board B: 272-token image prompt, 64 tokens,
4 threads / decode on 3, medians of 3.

| layout (`pack_r4.py`) | MB/token | decode tok/s | KL | leading tokens = f32 (of 64) | moty: leading / teacher-forced | tiles inventing a computer set |
|---|---|---|---|---|---|---|
| f32 weights, moty's int8 activations (ceiling) | — | — | 0.0003 | 59.2 (29 of 33 all 64) | — | 0 |
| Q4R4, round-to-nearest | 199.4 | 15.35 | 0.052 | 20.4 | 21.5 / 93.3 % | 12 |
| GPTQ (text) + Q8R4 layers 0–1 (the text recommendation) | 217.7 | 14.06 | 0.095 | 15.3 | 11.2 / 84.2 % | 24 |
| GPTQ (text + tiles) | 199.4 | 15.35* | 0.044 | 28.6 | 24.0 / 92.9 % | 12 |
| RTN + Q8R4 head | 232.9 | 13.93 | 0.041 | 25.4 | — | 7 |
| RTN + Q8R4 layers 8–15 | 270.7 | 11.96 | 0.021 | 39.0 | — | 3 |
| **RTN + Q8R4 head + layers 10–15** | 286.4 | **11.49** | 0.015 | **42.5** (2 all 64) | **42.0 / 96.4 %** (3 all 64) | **0** |
| GPTQ (text + tiles) + Q8R4 head + layers 10–15 | 286.4 | 11.49* | 0.014 | 29.8 | 35.6 / 95.4 % | 21 |
| Q8R4 everywhere | 376.6 | 10.00 | 0.0005 | 55.0 (20 all 64) | — | 3 |

\* same format and bytes as the row above it (speed by construction).
"Computer set": monitor, keyboard, mouse, laptop or screen in the answer but
not in the reference answer (the scene has an on-screen clock, so a few
"screen" mentions may be legitimate — the f32 ceiling has none).

What the image inputs change:
- **Sensitivity moves to the end of the network.** One tensor group in
  int4, the rest f32 (KL, the first 4 tiles): lm_head 0.011, FFN w2 0.011, w3 / v / w1
  0.006, attention out 0.004, conv out / in 0.003, q / k 0.002; by layer
  14 0.006, 10 0.005, 15 0.005 … layer 0 0.0016, layer 1 0.0003 — on text
  layers 0 and 1 cost 0.36 / 0.35 (§5.8). The text recipe protects the
  wrong layers.
- **GPTQ calibrated on text hurts images**: its Hessians come from text
  activations; on image-conditioned inputs it doubles the error of plain
  rounding (KL 0.095 vs 0.052). Calibrating on the answer positions of
  image sequences (16 tiles of brownai test images, colour and grey, not
  the evaluation tiles; the image-row positions left out of the Hessians —
  on the first 4 tiles, with them KL 0.056, without 0.046) brings it below
  rounding in KL (0.044 on the 33 tiles), but every
  GPTQ variant still invents the computer set on more tiles than its
  round-to-nearest twin: a lower average error, a worse answer.
- moty's int8 activations cost almost nothing (29 of 33 tiles identical
  over 64 tokens with f32 weights); the damage is in the int4 weights.

**Recommendation for LFM2.5-VL-450M on the A53: round-to-nearest Q4R4 with
Q8R4 for the lm_head and layers 10–15**
(`pack_r4.py --method rtn --q8 "lm_head.weight,model.layers.1?.*"` on the
VL snapshot): no invented objects on the 33 tiles and about three times
the matching prefix of the text recipe (42.0 vs 11.2 tokens in moty), at
11.5 tok/s decode instead of 14.1. For
text-only use of LFM2.5-350M §5.8 stands. `pack_r4.py --calib-tiles` (VL
calibration: text positions of image sequences) is kept for further
studies; `tools/ref/hf_vl_rows.py` makes calibration tiles from images.
