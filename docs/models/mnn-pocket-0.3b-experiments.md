# MNN-Pocket-0.3B optimization experiments

[Current setup, checked results and reproducible benchmark](mnn-pocket-0.3b.md).

This log records prototypes, intermediate results and rejected variants. Commands
under `.model-cache/` describe the original local experiments. The current setup
page uses tooling and fixtures included in the repository.

The authors recommend the QAT Q4_1 GGUF for on-device inference. Their MNN export uses W4A16 with groups of 64 weights. The official GGUF describes its conversion in `pocket.qat_conversion`: it keeps the UINT4 codes, splits each group of 64 into two 32-weight blocks, and stores an FP16 scale and minimum in each block. [GGUF model card](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF#qat-q4_1-gguf) · [MNN model card](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-MNN#model-details)

Moty reads the Q4_1 blocks directly from the GGUF. The input embeddings and tied output head share one packed table. The CPU kernel applies each block's scale and minimum with float32 accumulation. AArch64 builds with NEON FMLAL use FP16 matmul activations by default. Token lookup decodes only the requested embedding row into the activation buffer. With `MMAP=0`, the loader reads the same packed bytes into RAM.

## CPU benchmark

Apple M5 Max, 48 GiB, internal SSD on APFS, macOS 26.7.1, Apple Clang 21.0.0 and OpenMP. Both binaries used the default `-O2` build and base commit `d5c0260`. The baseline binary included the earlier Q4_1 loader that expanded weights to float32 or requantized them to int8. All modes used [the same official GGUF revision](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B-GGUF/tree/791b7e92fa790a5ca5bbd7604feb1ce1d2429d14), with SHA-256 `a281c127e388b16537fd3f5d31ab32d3f87739e6dc514ac1093c80d9eac8d097`.

Four threads, context 512, greedy sampling, 128 generated tokens with the EOS limit disabled. Each configuration had one warm-up process per prompt length followed by five measured processes in rotating order with a warm file cache. The table shows medians. Peak RSS from macOS `/usr/bin/time -l` includes resident file-backed pages. Wall time includes startup.

| Input | Weight mode | Peak RSS (MiB) | Prefill (tok/s) | Decode (tok/s) | Wall time (s) |
|---|---|---:|---:|---:|---:|
| 19-token chat | Previous float32 (`QBITS=0`) | 1217.55 | 127.9 | 105.55 | 1.46 |
| 19-token chat | Native Q4_1 (`QBITS=-1`) | 205.84 | 400.6 | 159.76 | 0.85 |
| 19-token chat | Previous int8 (`QBITS=8`) | 321.14 | 1097.0 | 503.93 | 0.37 |
| 256 token IDs | Previous float32 (`QBITS=0`) | 1231.91 | 133.8 | 104.18 | 3.24 |
| 256 token IDs | Native Q4_1 (`QBITS=-1`) | 220.45 | 356.5 | 150.52 | 1.57 |
| 256 token IDs | Previous int8 (`QBITS=8`) | 336.61 | 998.9 | 472.30 | 0.63 |

Run the short-input comparison from the repository root with `pocket_gguf` set to the official Q4_1 file. Explicit `QBITS=0` and `QBITS=8` select the previous execution paths in the current binary.

```sh
make -C c -j4 qwen moty
for bits in 0 -1 8; do
  /usr/bin/time -l env GGUF="$pocket_gguf" QBITS="$bits" THREADS=4 CTX=512 \
    TEMP=0 NGEN=128 IGNORE_EOS=1 \
    PROMPT="Explain how a rainbow forms, step by step." ./c/moty > /dev/null
done
```

For the 256-token run, set `PROMPT_IDS` to a JSON file with `{"ids":[...]}` containing the first 256 token IDs of [the reference passage](../../tools/ref/ppl_text_long.txt), tokenized with the official snapshot. Unset `PROMPT` for that run.

## Optimized native kernel

Profiling the original native kernel on the 256-token input put the feed-forward projections at 76% of prefill time and 60% of decode time. The tied output head accounted for another 18% of decode time. These projections all use the same Q4_1 matmul.

The AArch64 kernel now centers the nibble codes and applies the scale and minimum once per block. It computes the sum of the input activations once per block and reuses that sum across output rows. During prefill, each worker reuses a small activation tile across its output rows, improving cache locality. Both paths use the original packed file and float32 activations and accumulation.

Both binaries were built from the same checkout before and after the kernel change. The table uses the same model, prompt lengths, build flags, thread count and timing commands as above, forcing `QBITS=-1` on both binaries. Each row is the median of seven interleaved runs with one warm-up process per binary and prompt length.

| Input | Native kernel | Peak RSS (MiB) | Prefill (tok/s) | Decode (tok/s) | Wall time (s) |
|---|---|---:|---:|---:|---:|
| 19-token chat | Before | 205.83 | 407.3 | 166.79 | 0.82 |
| 19-token chat | Optimized | 205.94 | 419.7 | 189.98 | 0.73 |
| 256 token IDs | Before | 220.45 | 366.8 | 155.34 | 1.53 |
| 256 token IDs | Optimized | 220.64 | 460.7 | 177.71 | 1.28 |

The optimized kernel matched the previous native kernel on all 201 generated tokens across the eight chat prompts below and on all 128 tokens from the 256-token input with `PREFILL_CHUNK=0`, `16`, and `1`. At the last prompt position, the maximum absolute logit difference was `1.45e-5`. The teacher-forced run matched all 2,310 top-1 IDs and measured perplexity `119.8882` (previous native kernel `119.8881`).

## W4A16 with NEON

The Q4_1 matmul converts each input activation to FP16 once and reuses the packed activation buffer across weight rows. The auxiliary block sums use those same rounded activations. NEON `FMLAL` and `FMLAL2` multiply the FP16 activations by the exact centered nibble values and accumulate in FP32. The stored QAT weights stay in their original Q4_1 blocks. Builds without FMLAL use FP32 activations. A matmul also uses the FP32 path if converting its inputs produces an infinity or NaN.

Use `Q41_ACT=f16` or `Q41_ACT=f32` with the benchmark command above and set `QBITS=-1` for both runs. A build targeting ARMv8.0 uses the FP32 path. For ARM targets with hardware FMLAL, select an architecture with `+fp16fml` or use `ARCH=native` on the target. The startup log reports the selected activation precision.

Apple M5 Max, four threads, context 512, 128 generated tokens and seven interleaved repetitions per input. Both modes used the same binary and the warm-up and timing procedure above.

| Input | Activations | Peak RSS (MiB) | Prefill (tok/s) | Decode (tok/s) | Wall time (s) |
|---|---|---:|---:|---:|---:|
| 19-token chat | FP32 | 205.95 | 407.0 | 187.83 | 0.73 |
| 19-token chat | FP16 with NEON | 206.09 | 502.4 | 253.38 | 0.55 |
| 256 token IDs | FP32 | 220.66 | 457.0 | 175.04 | 1.30 |
| 256 token IDs | FP16 with NEON | 221.91 | 537.6 | 233.39 | 1.03 |

All 329 generated tokens matched the FP32 activation path across the eight chat prompts and the 256-token input. FP16 generated the same 128 tokens with `PREFILL_CHUNK=0`, `16`, and `1`. All 2,310 top-1 IDs matched in teacher-forced evaluation. Perplexity was `119.9075` with FP16 activations and `119.8882` with FP32 activations.

## Multi-row NEON, joint FFN and long-context GQA

The FP16 path now reuses loaded activations across four rows and processes `gate` and `up` in one parallel region. Workers write activated tiles directly to the `down` input. The Q4_1 file remains mapped in its original layout. Where Pocket's adjacent blocks have verified identical scale and offset, decode shares those parameters while applying each block's FP32 update in the original order. For single-token attention at 2,048 or more visible positions, pairs of query heads share K/V loads over 128-position blocks. Each head computes its own FP32 softmax over the full visible range, and the value partials are summed in block order.

The table compares the previous FMLAL binary and the optimized binary on M5 Max. Both ran the same GGUF with a 128-token generation limit, `TEMP=0` and `IGNORE_EOS=1`. Each cell is a median of seven interleaved runs following one warm-up per binary. Time to first token includes process startup and is measured at the first stdout byte. RSS includes resident file-backed pages.

| Input / threads | Prefill before → after (tok/s) | Decode before → after (tok/s) | First token before → after (ms) | Peak RSS before → after (MiB) |
|---|---:|---:|---:|---:|
| 256 / 4 | 531.7 → 599.2 | 234.65 → 260.63 | 491.9 → 448.4 | 221.92 → 219.42 |
| 256 / 12 | 1172.8 → 1309.4 | 439.95 → 440.15 | 229.5 → 217.5 | 224.67 → 222.14 |
| 1536 / 4 | 434.1 → 473.4 | 185.54 → 201.31 | 3552.0 → 3266.2 | 313.75 → 298.78 |
| 3584 / 4 | 322.9 → 348.3 | 140.60 → 157.23 | 11111.4 → 10315.0 | 462.42 → 427.77 |

Isolated four-thread measurements on the model's actual matrices showed `gate/up + SiLU` at **0.0797 → 0.0650 → 0.0625 ms** for one token, with the joint tiled kernel and then with parameter sharing. The same three stages took **0.0435 → 0.0377 → 0.0372 ms** for `down` and **0.7390 → 0.6874 → 0.6612 ms** for the tied output head. With 256 inputs, the joint FFN's gate/up stage took **8.84 → 7.92 ms**, and `down` took **4.31 → 3.89 ms**. Eight-token tiles performed better than four in this comparison. The paired attention candidate alone moved four-thread decode from **142.4 to 146.0 tok/s** in a two-run screening comparison at 3584 inputs. Dynamic scheduling for Q4_1 row jobs was slower at 12 threads, so the static schedule remains in use.

For a short-context comparison, set `pocket_before` to the saved previous binary and `pocket_gguf` to the official GGUF. Set `pocket_ids` to a JSON file with the first 256 reference token IDs. For the longer cases, use 1536 tokens with `CTX=2048` and 3584 tokens with `CTX=4096`. The 3584-token input repeats the reference passage to reach that length.

```sh
for executable in "$pocket_before" ./c/moty; do
  /usr/bin/time -l env GGUF="$pocket_gguf" QBITS=-1 Q41_ACT=f16 \
    THREADS=4 CTX=512 TEMP=0 NGEN=128 IGNORE_EOS=1 \
    PROMPT_IDS="$pocket_ids" "$executable"
done
```

The full path matched **457 generated tokens** over ten inputs and **2,310 teacher-forced top-1 IDs**. Perplexity was **119.9085**, compared with **119.9075** before. The 256-token generation also matched with `PREFILL_CHUNK=1` and `16`. The measured gains are below the planned latency targets, and the 12-thread decode result is effectively flat.

On ARMv8.0, the two-row FP32 NEON kernel, joint FFN, and GQA passed the standalone tests with `qemu-aarch64 -cpu cortex-a53`. Their Cortex-A53 performance still needs a board measurement. For board benchmarking, use `MOTY_Q41_ROWS=1 MOTY_Q41_FFN=1 MOTY_Q41_GQA=1` to select the candidates. FMLAL builds select these paths automatically with FP16 activations. Setting any of those variables to `0` selects the previous implementation for that operation. `MOTY_Q41_GROUP64=0` disables the parameter-sharing fast path independently.

## Shared Q/K/V preparation and fewer attention barriers

For resident Q4_1 query, key and value projections, Qwen prepares the shared input once and runs the three matrices in one parallel region. Workers write to the separate output buffers. This avoids packing the same activations three times and leaves the mapped weights in their original layout. FMLAL builds enable the path with FP16 activations. Use `MOTY_Q41_QKV=0` to select separate projections, or `MOTY_Q41_QKV=1` to test the shared path with ARMv8.0 NEON.

Long decode can assign the score, softmax, value and merge phases of a query pair to the same worker. This reduces the paired GQA kernel from four parallel regions to one. The 128-position blocks, per-head FP32 softmax and ordered value reduction stay the same. FMLAL builds using FP16 activations select the new schedule when each worker can receive at least one pair task. Pocket has eight tasks, including two single-head tails. Use `MOTY_Q41_GQA_PAIR=0` to select the four-region schedule and `MOTY_Q41_GQA_PAIR=1` to test the pair schedule.

These M5 Max medians compare the preceding and current binaries with their default schedules and FP16 activations. Each configuration ran seven times after warm-up, alternating which binary ran first.

| Input / threads | Prefill before → after (tok/s) | Decode before → after (tok/s) | First token before → after (ms) | Peak RSS before → after (MiB) |
|---|---:|---:|---:|---:|
| 256 / 4 | 603.2 → 601.8 | 259.40 → 261.52 | 445.7 → 446.8 | 219.41 → 219.66 |
| 256 / 12 | 1316.0 → 1315.5 | 433.64 → 455.77 | 216.5 → 215.9 | 222.98 → 223.05 |
| 1536 / 4 | 475.3 → 477.1 | 201.11 → 202.57 | 3252.5 → 3241.0 | 298.75 → 298.75 |
| 3584 / 4 | 353.1 → 352.5 | 161.18 → 165.28 | 10170.9 → 10188.1 | 427.77 → 427.77 |
| 3584 / 12 | 890.4 → 893.6 | 276.15 → 280.09 | 4044.2 → 4024.5 | 433.09 → 429.88 |

At 3,584 inputs with shared Q/K/V enabled, the one-region schedule raised four-thread decode from 158.92 to 160.25 tok/s in a separate experiment. With twelve threads it reduced throughput from 276.83 to 271.27 tok/s, so those runs use the four-region schedule by default.

On M5 Max, the optimized binary matched all 457 generated tokens and 2,310 teacher-forced top-1 IDs from the preceding configuration. Both binaries measured perplexity 119.9085. The 256-token generation also matched with prefill chunks of 1 and 16 tokens. At 3,584 inputs with twelve threads, all 128 generated IDs matched with each of `KV_BITS=0` and `KV_BITS=8`.

The ARMv8.0 build passed the Q4_1 and GQA kernel tests under QEMU 10.0 with `-cpu cortex-a53`. With FP32 activations, toggling the two new paths preserved all 329 generated tokens across the eight chats and the 256-token input, and all 2,310 teacher-forced top-1 IDs. Both configurations measured perplexity 119.8882. On ARMv8.0 the two paths remain opt-in pending performance measurements on hardware.

To isolate these changes, use the timing command from the preceding section with the same GGUF and prompt IDs. For the Q/K/V comparison, set `MOTY_Q41_GQA_PAIR=0` for both runs and alternate `MOTY_Q41_QKV` between `0` and `1`. For a long attention comparison, set `MOTY_Q41_QKV=1` and alternate `MOTY_Q41_GQA_PAIR` between `0` and `1`. Run one warm-up per configuration, then take the median of seven interleaved runs. The Cortex-A53 functional test is the [ARMv8.0 CI step](../../.github/workflows/ci.yml).

With `pocket-ref.json` from the reference commands below, compare both paths on one binary as follows. Use `Q41_ACT=f32` when running the ARMv8.0 build under `qemu-aarch64 -cpu cortex-a53`.

```sh
for mode in 0 1; do
  GGUF="$pocket_gguf" QBITS=-1 Q41_ACT=f16 THREADS=4 PPL=pocket-ref.json \
    PPL_OUT="pocket-schedule-$mode.top1" MOTY_Q41_QKV="$mode" \
    MOTY_Q41_GQA=1 MOTY_Q41_GQA_PAIR="$mode" ./c/moty
done
cmp pocket-schedule-0.top1 pocket-schedule-1.top1
```

## Register tiling for FFN and four-head GQA

The Group64 FMLAL `gate/up` decode kernel consumes each row's nibble codes as they are unpacked. Both projections share the packed FP16 input and a parallel region. The kernel preserves the two FP32 updates per Group64 block. Use `MOTY_Q41_FFN_DECODE_ROWS=4` for the previous 4+4 tile, `2` for the 2+2 tile and `1` to reproduce the 1+1 comparison.

On the M5 Max build, sampling and disassembly of the two hot loops showed vector spills and reloads through the stack with 4+4. The sampled 2+2 loop contains zero vector loads or stores referencing the stack pointer.

On Pocket's layer-0 matrices at four threads, the complete `gate/up` stage including SiLU and the output write took 0.0632, 0.0594 and 0.0615 ms with 4+4, 2+2 and 1+1 rows respectively. At 256 inputs the existing 4×8 tile took 7.79 ms for `gate/up` and 3.89 ms for `down`. The initial 2×8 trial took 8.08 and 3.96 ms, and 4×4 took 8.70 and 4.37 ms.

The experimental 4+2+1 GQA layout shares K/V loads across four query heads in its largest group. The full-sequence FP32 softmax, 128-position blocks, and block-order value reduction are common to both layouts. Use `MOTY_Q41_GQA_FOUR=1` for 4+2+1 or `0` for 2+2+2+1, with `MOTY_Q41_GQA_PAIR=0` to schedule blocks and `1` to schedule whole groups.

In seven paired full-model runs within one binary with 256 inputs, 2+2 increased decode throughput from 255.14 to 265.75 tok/s at four threads and from 458.64 to 477.15 tok/s at twelve. The separate comparison against the saved binary measured a 2.1% latency reduction at four threads, below the 3% rollout threshold. The FMLAL path defaults to 2+2 with twelve workers. With 3584 inputs, 4+2+1 measured 162.38 tok/s with block scheduling and 161.64 tok/s with group scheduling, compared with 164.69 tok/s in the four-thread baseline. At twelve threads the block schedule measured 281.59 versus 277.44 tok/s. The default long-context layout remains 2+2+2+1.

The final prefill comparison at four threads measured 556.5 tok/s with the reordered 2×8 trial and 599.3 tok/s with the existing 4×8 tile. Both FP16 trials remain excluded from default selection. The archived `prefill` experiment binary accepts `MOTY_Q41_PREFILL_ROWS=2 MOTY_Q41_TILE=8` to reproduce the reordered trial.

The table compares the saved baseline with the selected default schedules on M5 Max. Each case uses the same GGUF, 128 generated tokens and seven paired repetitions after warm-up. The order alternates between pairs.

| Input / threads | Prefill before → after (tok/s) | Decode before → after (tok/s) | First token before → after (ms) | Peak RSS before → after (MiB) |
|---|---:|---:|---:|---:|
| 256 / 4 | 587.7 → 582.1 | 258.60 → 258.90 | 456.2 → 460.2 | 219.67 → 219.66 |
| 256 / 12 | 1330.2 → 1313.3 | 455.92 → 480.91 | 214.0 → 216.8 | 223.02 → 223.23 |
| 1536 / 4 | 483.3 → 480.8 | 205.89 → 206.75 | 3198.7 → 3215.4 | 298.78 → 298.77 |
| 1536 / 12 | 1108.6 → 1110.4 | 354.75 → 366.70 | 1406.9 → 1405.7 | 301.05 → 302.25 |
| 3584 / 4 | 370.5 → 369.2 | 169.88 → 167.47 | 9694.7 → 9727.0 | 427.78 → 427.80 |
| 3584 / 12 | 900.0 → 897.6 | 281.65 → 289.97 | 4003.3 → 4013.4 | 432.12 → 432.09 |

Decode latency fell by 5.2% at 256 inputs and by 2.9% at 3584 inputs with twelve threads. Peak RSS rose by at most 1.20 MiB. The planned end-to-end latency reductions remain unmet.

The M5 Max comparison matched all 457 generated tokens and all 2310 teacher-forced top-1 IDs. Perplexity was 119.9085 on both binaries. The 256-token input also matched with `PREFILL_CHUNK=0`, `1`, and `16`. At 3584 inputs and twelve threads, both `KV_BITS=0` and `KV_BITS=8` matched all 128 generated IDs.

The ARMv8.0 build passed the kernel checks under `qemu-aarch64 -cpu cortex-a53`. With the manual 2×2 prefill and 4+2+1 GQA settings, it matched all 329 generated tokens and all 2310 top-1 IDs against 2×8 prefill and 2+2+2+1 GQA in the same build. Both configurations measured perplexity 119.8882. On the emulator, the two prefill tiles measured 1307 versus 2192 ms for the complete layer-0 `gate/up` stage and 671 versus 1127 ms for `down` at 256 inputs. These are emulator wall times. Cortex-A53 hardware performance remains unverified.

To reproduce the full-model measurements, save the prior binary as `pocket_before` and use the timing command from the previous section. For single-binary comparisons prepend the experimental settings to the same command. For FP16 prefill 4×4 use `MOTY_Q41_TILE=4`. On the ARMv8.0 build, `Q41_ACT=f32 MOTY_Q41_ROWS=1 MOTY_Q41_FFN=1 MOTY_Q41_TILE=2` selects the 2×2 prefill trial. The [Cortex-A53 CI step](../../.github/workflows/ci.yml) runs the kernel checks under QEMU.

## Lossless panels, SIMD across KV positions and FFN scheduling

- `MOTY_Q41_PACKED=1` uses a local mmap cache for the FFN projections and output head. Each Group64 panel holds the original UINT4 codes and one copy of the original FP16 scale/min per group. The loader checks the GGUF file identity, tensor shape, on-disk size and payload checksum. It falls back to GGUF for each unavailable or invalid panel file. The tied embedding reads its rows from the same packed panels as the output head. Pocket creates 85 files of the form `<gguf>.moty-q41-<offset>` alongside the GGUF, totaling 134.87 MiB. Cache entries are published by atomic rename.
- `MOTY_Q41_FILE_WINDOWS=1` enables [reusable read-only windows](#reusable-file-windows-for-gate-and-down) for `gate` and `down` during packed streaming decode on Apple Silicon.
- `MOTY_Q41_KV4=1` stores the FP32 keys in four-position tiles. Each NEON lane computes one score using the existing FP32 reduction tree. This keeps one K representation throughout the session. `MOTY_Q41_KV4_PREFILL_PAIR=1` enables K reuse across two neighboring prefill queries with their own causal bounds. The int8 KV path stays available through `KV_BITS=8`.
- `MOTY_Q41_FFN_TEAM=1` evaluates resident Q4_1 `gate/up` and `down` in one OpenMP region for single-token decode without LoRA. The master prepares the `down` input between two dependency barriers. The ordinary kernel path handles other configurations.

At 256 input tokens on M5 Max, the panel cache raised four-thread decode from 261.09 to 276.99 tok/s and reduced peak RSS from 219.73 to 205.30 MiB. Prefill fell from 599.4 to 562.3 tok/s and TTFT rose from 448.4 to 491.9 ms. On twelve threads the panel cache used 14.50 MiB less peak RSS and ran decode at 450.67 versus 458.64 tok/s. The memory gain is available in the manual mode.

The OpenMP probe measured dispatch and synchronization as each region's elapsed time minus its longest worker span. Estimates were 2.1% of short decode at four threads and 10.5% at twelve, including approximately 0.9% and 4.2% inside FFN respectively. The single FFN region raised decode throughput from 257.64 to 260.43 tok/s at four threads and from 479.52 to 482.55 at twelve. The separate pthread build reached 479.24 tok/s versus 458.64 for OpenMP at twelve threads in the first series.

The combined configuration measured as follows against the saved binary. Each comparison is the median of seven repetitions after warm-up, with alternating order, the same GGUF and 128 generated tokens.

| Input / threads | Prefill before → after (tok/s) | Decode before → after (tok/s) | First token before → after (ms) | Peak RSS before → after (MiB) |
|---|---:|---:|---:|---:|
| 256 / 4 | 611.2 → 575.2 | 269.42 → 283.93 | 439.4 → 479.8 | 219.67 → 205.31 |
| 256 / 12 | 1349.9 → 1302.6 | 485.47 → 488.15 | 211.3 → 232.8 | 224.14 → 209.80 |
| 1536 / 4 | 500.0 → 472.1 | 209.74 → 214.64 | 3094.8 → 3289.0 | 298.77 → 284.45 |
| 1536 / 12 | 1158.8 → 1102.7 | 362.34 → 356.67 | 1347.8 → 1431.0 | 302.28 → 288.20 |
| 3584 / 4 | 367.8 → 372.2 | 164.36 → 174.95 | 9766.9 → 9666.5 | 427.80 → 413.45 |
| 3584 / 12 | 847.8 → 828.0 | 274.15 → 256.99 | 4249.2 → 4370.6 | 433.61 → 421.12 |

With only the tiled K and paired prefill enabled, 3584-input prefill reached 395.7 tok/s at four threads and 890.1 tok/s at twelve. Decode reached 163.61 and 271.73 tok/s respectively. Building the 85 panel files took 286 ms in a separate loader probe. A repeat mmap and checksum pass took a median of 22.8 ms. The first-run and repeat TTFT measurements for the panel-only process were 647.8 and 486.9 ms on the fresh cache path.

All three changes remain available by explicit opt-in. The combined mode exceeded the TTFT regression limit on the short input and the decode regression limit on the long twelve-thread input. The end-to-end targets in the plan remain unmet.

With all experimental flags enabled, the quality run matched all 457 generated IDs on M5 Max and all 329 in Cortex-A53 QEMU. Each architecture matched all 2310 teacher-forced top-1 IDs. Perplexity stayed at 119.9085 and 119.8882 respectively. The Apple run covered `PREFILL_CHUNK=0/1/16`. With `KV_BITS=8` and `MOTY_Q41_KV4=1`, the long-context fallback matched all 128 generated IDs. The kernel suite also checks byte-exact panel reversal, rejection of corrupted cache payloads, and odd and even causal score tails. Physical Cortex-A53 speed remains unmeasured.

To reproduce, use the benchmark command above with a fixed GGUF and token IDs, changing one setting at a time. For the combined configuration set all four switches to `1`:

```sh
env GGUF="$pocket_gguf" QBITS=-1 Q41_ACT=f16 THREADS=4 CTX=4096 \
  TEMP=0 NGEN=128 IGNORE_EOS=1 PROMPT_IDS="$pocket_ids" \
  MOTY_Q41_PACKED=1 MOTY_Q41_KV4=1 MOTY_Q41_KV4_PREFILL_PAIR=1 \
  MOTY_Q41_FFN_TEAM=1 ./c/moty
```

Use `CTX=512` with 256 input IDs and `CTX=2048` with 1536 IDs. Compare `THREADPOOL=1` using a separate clean build with the same compiler flags otherwise. For panel cache creation time, run from a fresh GGUF path, then repeat the same command against the generated cache. The ARMv8.0 functional test uses `qemu-aarch64 -cpu cortex-a53`.

## Three-head value accumulation for long decode

For single-token FP32 KV attention, the block schedule can compute the value stage in groups of 3+3+1 query heads. Each NEON load of a 16-channel strip is reused by three heads. The score and softmax stages keep their existing schedules. Value accumulation preserves the FP32 FMA order inside each 128-position block, and the final reduction preserves the order of the blocks.

On M5 Max with 3,584 input IDs, 128 generated tokens and 12 threads, a profiled run reduced the median value stage from 0.706 to 0.473 ms per generated token. The full GQA stage went from 1.590 to 1.340 ms/token. In seven paired runs of the same release binary, turning on the new value kernel reduced decode latency by a median of 6.98%. Median throughput rose from 277.45 to 300.79 tok/s. In seven additional paired runs of the final build, the default was 9.22% faster in decode latency than the explicit fallback, with median throughput of 306.65 versus 277.72 tok/s. Control runs at 256 and 1536 inputs with twelve threads and 3584 inputs with four threads stayed within 1.4% in paired median decode latency against the prior binary.

The standard FP16/FMLAL block schedule selects this value kernel by default with twelve workers. Set `MOTY_Q41_GQA_V3=0` to select the previous value kernel. To test it manually when the pair schedule is normally selected, set `MOTY_Q41_GQA_PAIR=0 MOTY_Q41_GQA_V3=1`. On the ARMv8.0 build add `MOTY_Q41_GQA=1` to enable the existing block attention path with FP32 activations. The visible-context threshold is 2048. Prefill and `KV_BITS=8` use their existing paths.

The final binary matched all 457 generated tokens on M5 Max and 329 in Cortex-A53 QEMU. All 2310 teacher-forced top-1 IDs matched on both architectures. Perplexity was unchanged at 119.9085 and 119.8882 respectively. The kernel tests check byte-exact outputs across both value layouts at 2048, 2049, 2177 and 2179 visible positions, including the tiled K path. Physical Cortex-A53 performance remains unmeasured.

## Shared prefill attention and exact history lookahead

The twelve-thread FMLAL path shares the row-major FP32 KV cache across four consecutive prefill queries for Pocket at `CTX >= 2048`. Each query retains its own causal extent, the original score reduction and softmax, and the original position order for accumulating V. `MOTY_Q41_PP_ATTN=0` selects the previous path. Set it to `2` or `4` to test either tile on NEON. The int8 KV cache and the tiled K experiment keep their existing paths.

### Projection and decode experiments

The FMLAL prefill prototypes use a fixed output row and sixteen or thirty-two tokens to reuse each unpacked Q4_1 block. The lossless panel path and the optional block-major activation layout preserve FP16 rounding and the order of FP32 updates. These are the median timings for layer 0 on M5 Max, with hidden width 896 and FFN width 2560.

| 256-token projection | Existing mmap | 16-token tile | 32-token tile | 16-token tile with block-major activations | Existing lossless panel |
|---|---:|---:|---:|---:|---:|
| `gate/up` with SwiGLU | 3.258 ms | 3.378 ms | 3.416 ms | 3.548 ms | 3.476 ms |
| `down` | 1.754 ms | 1.958 ms | 2.145 ms | 1.999 ms | 2.052 ms |

The disassembly showed FMLAL/FMLAL2 and vector stack traffic in the tiled prefill kernel. On the full 256-token input, PP was 1335 tok/s with the existing kernel, 1282 and 1100 with the 16- and 32-token tiles, and 1258 with packed 16-token tiles. Keeping the existing kernel also preserves its scheduling for Q/K/V and O. The decode microbench measured 0.0556 ms for the mmap `gate/up` kernel and 0.0591 ms for the 2+2 packed prototype. The packed `down` measurement was 0.0424 ms versus 0.0428 ms from mmap. Packing only the tied head still computed the entire vocabulary and failed to clear the full-model improvement threshold. Head-only cache creation produced one 31.5 MiB file. Median TTFT was 407 ms on first use at three fresh GGUF paths and 216 ms on repeat opens. These runs measured panel creation over the existing filesystem cache.

With identical decode kernels on 8, 10 and 12 workers, the short decode yielded 373, 435 and 485 tok/s. The long decode yielded 243, 268 and 298 tok/s. The measured trend favored retaining twelve workers. The earlier OpenMP probe attributed about 4.2% of short decode to dispatch and synchronization within FFN at twelve workers. Combining the FFN into one region recovered about 0.6% on the model, so the default keeps the current regions. On 3584 inputs with four-query attention, `PREFILL_CHUNK=256` yielded 1053 tok/s versus 1057 with the unchunked run and cut peak RSS from 434 to 325 MiB. In the separate row-major sweep, `PREFILL_CHUNK=64/128/256` yielded 917/924/950 tok/s at 309/313/325 MiB, versus 933 tok/s at 433 MiB without chunking.

The following is a separate three-run instrumented comparison at 3584 input IDs, 128 output IDs, and twelve workers. Times are cumulative across layers. The score, softmax, V and merge columns cover decode only. All three layouts generated the same IDs.

| Layout | Prefill attention (ms) | Prefill KV store (ms) | Decode KV store (ms) | Score (ms) | Softmax (ms) | V (ms) | Merge (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Row-major, individual prefill queries | 1439.23 | 10.35 | 1.30 | 83.16 | 26.62 | 59.76 | 8.50 |
| Row-major, four prefill queries | 1027.54 | 10.48 | 1.32 | 82.57 | 26.38 | 59.53 | 8.84 |
| Tiled K, two prefill queries, two decode heads × four positions | 1261.53 | 12.86 | 2.13 | 83.48 | 27.30 | 59.78 | 8.56 |

At the new default, long-context GQA still took approximately 1.40 ms per generated token and accounted for about 42% of decode time. Scoring was its largest stage at 0.65 ms/token, followed by V at 0.47 ms/token. The combined FFN projections took about 1.1 ms/token. The tiled K layout did not lower decode scoring time and increased write cost.

### Exact lookahead

`MOTY_Q41_LOOKAHEAD=2`, `4`, `auto` or `backoff` enables an experimental mode for one-shot, greedy Pocket generation. It finds suffixes in committed history and verifies proposals with the full output head. `auto` requires two prior continuations to agree on at least two tokens. `backoff` searches up to 8192 committed IDs and votes on continuations with up to eight tokens of suffix context. It requires at least 80% agreement, and requires more observations for shorter suffixes. Verification shares projection work and schedules long GQA blocks across the verified queries. Each query retains single-token arithmetic on either side of the 2048-position attention threshold. After a mismatch, the logical KV length rolls back to the accepted prefix and the verified model's actual token becomes the pending input. The ordinary path handles interactive sessions, non-greedy sampling, LoRA, tiled or int8 KV, and models with recurrent state.

In the 256-token benchmark, output contains 14 distinct IDs and 16 distinct 4-grams out of 125. The median TPS was 486 with the ordinary path, 664 with two drafts, 744 with four, and 719 in `auto`. Verified proposals were 73/76, 88/91 and 81/82 respectively. The 3584-token benchmark contains 66 distinct output IDs and 117 distinct 4-grams. Its medians were 306, 301, 263 and 321 TPS with the same settings, with 34/80, 39/134 and 17/23 proposals accepted respectively.

On the eight prompts in the quality suite, all four modes generated the same 201 IDs with EOS enabled. Median paired decode throughput changed by −3.30% for two drafts, −6.34% for four, and −0.57% for `auto`. Those modes accepted 6/23, 7/39 and 0/0 proposals respectively. The observed throughput gain on the repetitive fixture did not generalize to these prompts, so lookahead stays opt-in.

### Full-model acceptance

Before edits, seven runs gave 1355.6 PP and 467.03 TPS on 256 inputs and 901.3 PP and 292.59 TPS on 3584. These fix the four target rates at 2711.2 PP, 934.06 TPS, 1802.6 PP and 585.18 TPS. The saved binary was also rerun in every acceptance pair to account for runtime variation. The table lists separate medians after one warm-up per configuration and seven repetitions alternating order. TTFT is elapsed process time to the first output byte. Wall time includes process startup and shutdown. Both decode series generated 128 IDs with EOS ignored.

| Input / workers | Mode | PP (tok/s) | TPS (tok/s) | TTFT (ms) | Wall (ms) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|
| 256 / 12 | Saved | 1355.1 | 492.49 | 210.05 | 472.37 | 224.22 |
| 256 / 12 | Default | 1357.4 | 492.10 | 209.91 | 471.99 | 224.22 |
| 256 / 12 | Default + `auto` | 1349.7 | 727.17 | 210.63 | 388.77 | 224.48 |
| 1536 / 12 | Saved | 1193.2 | 373.54 | 1308.88 | 1654.80 | 302.33 |
| 1536 / 12 | Default | 1246.4 | 371.42 | 1254.57 | 1603.91 | 302.92 |
| 3584 / 12 | Saved | 954.6 | 307.07 | 3775.70 | 4197.78 | 433.19 |
| 3584 / 12 | Default | 1065.0 | 303.85 | 3385.88 | 3833.40 | 433.80 |
| 3584 / 12 | Default + `auto` | 1063.6 | 322.30 | 3390.49 | 3792.75 | 435.80 |
| 3584 / 4 | Saved | 368.9 | 174.91 | 9735.25 | 10472.06 | 427.84 |
| 3584 / 4 | Default | 370.8 | 176.52 | 9686.91 | 10416.16 | 427.88 |

Median paired changes at twelve threads were +11.13% PP, −0.30% TPS, −9.94% TTFT and +0.14% RSS on the 3584-token run with default settings. On the 1536-token control, PP gained 4.93% and TPS changed by −0.83%. The four default rates still need approximately 100%, 90%, 69% and 93% further growth respectively to reach the fixed targets. The ordinary decode also falls short of the intermediate ×1.5 target.

The SHA‑256 values identify the measured binaries and fixtures.

| Artifact | SHA‑256 |
|---|---|
| Saved executable | `539a9f719facd453e8dc90e572a402f35375760337c7d3f356957f4bcc864889` |
| Measured executable | `d2149a002184db3aac4089569c7a49ddf93afd232825427dd9850854a3bc4dc0` |
| Quality-tested release with the interactive fallback | `3f52bff384d6a36e1ad1803ecf8c0421dafdeed3d10e2dacfb52900e90cd642b` |
| Q4_1 GGUF | `a281c127e388b16537fd3f5d31ab32d3f87739e6dc514ac1093c80d9eac8d097` |
| 256 token IDs | `bf0ab9bcc8cda21d888b64a753d17e24fc71d2a3799aa4c07911030057c8c370` |
| 1536 token IDs | `d88ae9b8e3c16a3234a7ac70e6f7c5d7a73ee8c0168291d5bb4c6a978495174a` |
| 3584 token IDs | `d8af03bb78dc45e14ee7fd06d38cc886073248e14b6b0825d798c0a0381e9ad3` |

The correctness run matched all 457 generated IDs on M5 Max and all 329 on ARMv8.0 in QEMU Cortex-A53. Each architecture matched all 2310 teacher-forced top-1 IDs. Perplexity stayed at 119.9085 on Apple and 119.8882 in QEMU. The Apple comparison exercised `PREFILL_CHUNK=0/1/16`. The kernel tests covered prefill tile tails and causal bounds, as well as the single-token fallback at 2047 and exact batch attention from 2048 through 2052 positions. The replay build compared full logits bit for bit. Its long four-draft run reached a first rejection at each draft position, with counts of 16, 9, 4 and 2. Further comparisons matched 64 tokens generated across the long-attention threshold, the 17- and 63-token limits, and all 128 tokens with `KV_BITS=8`. The interactive fallback matched both turns of the persistent-KV test. Native tests reported 174 passes and four platform skips. Physical Cortex-A53 speed remains unmeasured.

From the repository root, retain the existing executable before building with `make -C c -j qwen moty`. Use identical files for both commands and alternate the order after a warm-up. Use `CTX=512/2048/4096` for the 256/1536/3584 input fixtures respectively. The following command uses the long fixture.

```sh
env GGUF="$pocket_gguf" QBITS=-1 Q41_ACT=f16 THREADS=12 CTX=4096 \
  TEMP=0 TOKENS=1 NGEN=128 IGNORE_EOS=1 PROMPT_IDS="$pocket_ids" \
  /usr/bin/time -l ./c/moty
```

To isolate the default attention change, repeat with `MOTY_Q41_PP_ATTN=0`. To compare the tiled score kernel, use `MOTY_Q41_KV4=1 MOTY_Q41_KV4_PREFILL_PAIR=1`. To test the projections, use `MOTY_Q41_PP_TILE=16/32` one value at a time, optionally with `MOTY_Q41_PP_XPACK=1`. `MOTY_Q41_PACKED=gate/down/ffn/head` selects which tensor families use panels, while `MOTY_Q41_PACKED=1` selects all existing panel families. Add `MOTY_Q41_PANEL_DUAL=1` for the packed 2+2 `gate/up` kernel. For the exact lookahead check, rebuild the Qwen engine with `-DMOTY_Q41_VERIFY_CHECK` and run the same fixtures with `MOTY_Q41_LOOKAHEAD=2`, `4` and `auto` individually. The test build compares every verified logit row with its sequential counterpart. The ARMv8.0 CI job runs the kernel tests with `qemu-aarch64 -cpu cortex-a53`.

Local run artifacts are under `.model-cache/neon-x2/`. `bench.py` records every pair in JSON, `profile/run.sh` runs the instrumented attention breakdown, and `verify-release.py` and `verify-a53.py` compare generated IDs and teacher-forced predictions. The numbered prompt fixtures are `pocket-prefill-256.json` and `q41-bottleneck/prompt{1536,3584}.json` under `.model-cache/`.

## Row-major GQA decode and compact FFN trials

The long-decode path on Apple with twelve OpenMP workers and FMLAL computes row-major FP32 scores in query-head groups of 3+3+1. It schedules scores by 128-position block, keeps the existing value grouping, and runs the four dependent phases in one OpenMP region. The merge holds four adjacent vectors per head and still sums blocks in position order. The two FP32 chains and score reduction are unchanged. Single-token and batched verification paths share work sizing and dispatch for the four phases. The four-thread head-owner schedule and ARMv8.0 defaults stay on their existing paths.

### FFN experiments

The FMLAL variants use 2×4 or 2×6 tiles on the original Q4_1 rows for `gate/up` and `down`. Each pair of rows shares FP16 activation loads, applies the original per-block updates, and accumulates in FP32. The separate decode experiments prefetch upcoming `gate/up` codes and write SwiGLU directly to the intermediate, or partition `down` work at output cache-line boundaries. The refactor shares the same centered-code FMLAL dot product between the mmap and panel decode kernels. It also keeps the specialized PP code out of the generic row loops.

On a 256-ID, 128-output, twelve-worker run, the saved binary and the new defaults measured 1334.3 and 1335.6 tok/s in prefill. The 2×4 and 2×6 `gate/up` candidates reached 1283.0 and 1221.6 tok/s. The best compact `down` candidate reached 1315.7 tok/s. For decode, the new default, prefetch with direct write, and cache-line scheduling reached 478.08, 471.82 and 456.85 tok/s respectively. The compact kernels emitted FMLAL and FMLAL2, with vector stack traffic still present in the larger tile. These candidates remain opt-in.

Isolated layer-0 medians use the same Pocket weights. For 256 tokens, the mmap, 2×4, 2×6 and panel kernels took 3.225, 3.454, 3.846 and 3.458 ms for `gate/up` including SwiGLU. The corresponding `down` times were 1.720, 1.874, 1.959 and 1.831 ms. For one token, the prefetch and direct write trial took 0.0588 ms versus 0.0599 ms, and the cache-line `down` trial took 0.0417 ms versus 0.0451 ms. These isolated gains reversed in the full-model measurements.

An instrumented short decode recorded 3556 `gate/up` calls across all layers. Cumulative medians were 1.62 ms for activation preparation, 78.87 ms for projections on the slowest worker, 1.01 ms for SwiGLU, 2.18 ms for output writes and 5.13 ms for dispatch and waiting. Prefetch with direct write reduced the write estimate to 1.35 ms while the projection estimate reached 79.21 ms.

To reproduce the FFN trials, set `MOTY_Q41_COMPACT_PP=4` or `6` for `gate/up`, or `MOTY_Q41_COMPACT_DOWN=4` or `6` for `down`. `MOTY_Q41_GATE_DECODE=1` enables the prefetch and direct write trial. `MOTY_Q41_DOWN_ALIGN=1` enables the output partition trial. Compare mmap with `MOTY_Q41_PACKED=gate` and `MOTY_Q41_PACKED=down` in separate runs.

### GQA experiments and acceptance

The first screen compared 2 query heads × 4 positions and 3 × 2 against the previous 2 × 2 score kernel on the same KV cache. The next screen compared block-first scoring, the 16-channel merge and the single OpenMP region. The combination uses no additional KV representation or score buffer. Individual phases and the combination are independently selectable for profiling.

On 3584 input IDs at twelve workers, seven alternating repetitions gave a median paired TPS gain of 0.78% for the 3×2 score tile alone, 5.30% for block-first scoring alone, and 8.44% for the complete schedule. The full schedule is the new default. The table lists separate medians for each binary. TTFT is elapsed process time to the first output byte and wall time includes startup and shutdown.

| Input / workers | Binary | PP (tok/s) | TPS (tok/s) | TTFT (ms) | Wall (ms) | Peak RSS (MiB) |
|---|---|---:|---:|---:|---:|---:|
| 256 / 12 | Saved | 1334.3 | 481.05 | 213.04 | 481.34 | 223.09 |
| 256 / 12 | New | 1335.6 | 478.08 | 213.14 | 483.84 | 223.36 |
| 3584 / 12 | Saved | 1049.5 | 307.12 | 3436.13 | 3858.09 | 434.00 |
| 3584 / 12 | New | 1044.3 | 332.16 | 3453.77 | 3844.17 | 434.00 |
| 1536 / 12 | Saved | 1230.7 | 349.45 | 1269.89 | 1640.75 | 302.89 |
| 1536 / 12 | New | 1235.2 | 370.89 | 1266.08 | 1619.90 | 302.89 |
| 1536 / 4 | Saved | 506.2 | 212.78 | 3055.84 | 3665.70 | 298.84 |
| 1536 / 4 | New | 503.2 | 212.73 | 3072.77 | 3678.87 | 298.86 |

The paired long-run change in peak RSS was +0.005%. The short paired changes were −0.17% PP and −0.82% TPS. At 1536 IDs, the paired changes were −0.08% PP and +1.67% TPS at twelve workers, and +0.04% PP and +0.72% TPS at four. The long-run speed target for this iteration cleared +5% TPS. Short prefill and decode stayed close to the saved binary, so the short-context targets and the overall ×2 target remain open.

Instrumented three-run medians below are milliseconds per generated token. The gap between score wall time and the longest worker span estimates dispatch and synchronization. In the one-region build, the merge column includes the cost of joining the team.

| Score kernel and schedule | Score | Longest score worker | Softmax | V | Merge | Total GQA |
|---|---:|---:|---:|---:|---:|---:|
| Previous 2×2, head-group order | 0.569 | 0.531 | 0.180 | 0.465 | 0.064 | 1.308 |
| 2×4, head-group order | 0.570 | 0.531 | 0.183 | 0.473 | 0.066 | 1.324 |
| 3×2, head-group order | 0.493 | 0.451 | 0.181 | 0.468 | 0.066 | 1.237 |
| 3×2, block-first | 0.412 | 0.370 | 0.182 | 0.477 | 0.066 | 1.168 |
| 3×2, block-first, one team and wide merge | 0.399 | 0.372 | 0.171 | 0.458 | 0.066 | 1.120 |

The score stage now costs less than V. The FFN projections and GQA each take about 1.1 ms per output token in the instrumented long decode. On the short input, FFN remains the largest optimization target.

On M5 Max, use the earlier timing command with `CTX=4096`, `THREADS=12`, 3584 input IDs and 128 output IDs. Set `MOTY_Q41_GQA_SCORE=24` or `32` to select a score tile. The command below disables the four new GQA options for an in-binary comparison with the fallback.

```sh
env GGUF="$pocket_gguf" QBITS=-1 Q41_ACT=f16 THREADS=12 CTX=4096 \
  TEMP=0 TOKENS=1 NGEN=128 IGNORE_EOS=1 PROMPT_IDS="$pocket_ids" \
  MOTY_Q41_GQA_SCORE=0 MOTY_Q41_GQA_BLOCK_FIRST=0 \
  MOTY_Q41_GQA_MERGE=0 MOTY_Q41_GQA_TEAM=0 /usr/bin/time -l ./c/moty
```

The warm-up and seven alternating comparisons are recorded under `.model-cache/neon-exact`. The same directory contains the saved executable, SHA‑256 values, instrumentation and the separate projection benchmark. The local suite passed 174 tests with four platform skips. The QEMU Cortex-A53 kernel suite passed. The full-model runs matched 457 generated IDs on Apple, 329 in Cortex-A53 QEMU, and 2310 top-1 IDs on each. Perplexity stayed at 119.9085 on Apple and 119.8882 in QEMU. The kernel tests compare outputs bit for bit across the new score tiles, odd causal tails, the 2048-position boundary and the batched layouts. Model runs also matched with `PREFILL_CHUNK=0/1/16`, across the long-attention threshold and on the `KV_BITS=8` fallback. Physical Cortex-A53 speed remains unmeasured.

## Reducing peak RSS with the original Q4_1 weights and FP32 KV

The target is half of the saved executable's peak RSS on both primary M5 Max workloads. The initial seven runs gave 222.50 MiB at 256 input IDs and 433.78 MiB at 3584, with 128 output tokens and twelve workers. The corresponding targets are 111.25 MiB and 216.89 MiB. `vmmap` recorded 180.7 MiB of resident clean GGUF pages after prefill on both inputs. At 3584 inputs it also showed 96.2 MiB retained by the Darwin allocator in empty large blocks. The peak RSS figures use `/usr/bin/time -l`.

- The GGUF index borrows the tensor mapping for metadata. A `MMAP=0` reader retains the bounded copy path. The tokenizer uses smaller hash entries for text input and skips the encoding hash tables for explicit `PROMPT_IDS` while preserving detokenization and stop IDs.
- The dense Pocket path reuses one arena for residuals, attention and FFN intermediates. Attention and FFN share the same storage after each phase finishes. On Apple NEON, `MOTY_Q41_WORKSPACE=0` restores per-operation allocations.
- The Apple FMLAL twelve-worker path processes prefill batches of at least 1024 inputs in blocks of 256 with one arena for the request. Decode keeps its original arithmetic and the KV cache stays FP32. If any chunk rejects FP16 activation packing, it discards the provisional K/V suffix and replays the original full batch. The chunked path preserves absolute RoPE positions and keeps any preceding prefix. Set `MOTY_Q41_AUTO_CHUNK=0` to disable it, or set a block size explicitly. A positive `PREFILL_CHUNK` still selects the explicitly configured path. The arena and automatic chunking can be enabled manually on Cortex-A53.
- `MOTY_Q41_PACKED=all` tests lossless packing for all eligible Q4_1 matrices, including Q/K/V and the output projection. The joint Q/K/V path still shares activation preparation across raw and packed matrices. The previous packed values retain their scope. The lower RSS comes with slower prefill, so this mode stays opt-in.

Per-layer striping with full precision replay also generated the same IDs. It saved roughly 114 MiB at 3584 inputs, while processing whole-model blocks saved roughly another 24 MiB. The per-layer experiment used two full residual buffers. The shipped approach replays from the input ID sequence and has one bounded residual buffer.

The packed trial reached 184.62 and 278.02 MiB on the 256- and 3584-input workloads, with paired PP changes of −7.29% and −9.18%. The final unpacked path uses one arena for the complete request. The table lists separate medians from seven paired runs after warm-up, with alternating order and 128 generated IDs per run.

| Input / workers | Peak RSS, saved → new (MiB) | PP, saved → new (tok/s) | TPS, saved → new (tok/s) | TTFT, saved → new (ms) |
|---|---:|---:|---:|---:|
| 256 / 12 | 224.25 → 202.27 | 1345.9 → 1345.0 | 474.99 → 479.24 | 211.29 → 208.20 |
| 1536 / 12 | 302.92 → 238.00 | 1220.6 → 1211.6 | 372.20 → 372.28 | 1279.91 → 1285.98 |
| 1536 / 4 | 298.88 → 270.67 | 502.4 → 504.1 | 211.75 → 211.11 | 3078.08 → 3063.96 |
| 3584 / 12 | 433.78 → 295.86 | 1010.3 → 998.1 | 331.68 → 329.15 | 3568.65 → 3608.90 |

The median paired changes at 256 inputs were −9.76% peak RSS, −0.42% PP and −0.09% TPS. At 3584 inputs they were −31.82% peak RSS, −0.41% PP and −0.29% TPS. The remaining gaps to the fixed ×2 RSS targets are approximately 91 MiB and 79 MiB respectively.

At the first output byte, `vmmap` measured a long-run peak physical footprint of 111.8 MiB versus 248.7 MiB on the saved binary. That Darwin metric excludes the clean file pages counted in RSS. The original weights occupy about 181 MiB of RSS and 3584 rows of full FP32 K/V account for another 98 MiB across 28 layers. Those components alone exceed the long RSS target, and the resident GGUF pages alone exceed the short target. Sampling 22 MiB of the model payload gave a lossless ratio of 100% with LZ4 and 88% with LZFSE. A warm `pread` probe reached 28, 52 and 72 GB/s on one, two and four workers respectively. It measured file transfer in isolation from inference and establishes the bandwidth costs that a weight-streaming implementation would need to overlap.

The 3584-token prefill comparison matched the final logits and all visible FP32 K/V bytes bit for bit. It also matched after forcing a full-batch replay from the second chunk. The model matched 457 generated IDs on Apple, 329 in Cortex-A53 QEMU and all 2310 teacher-forced top-1 IDs on each. Perplexity stayed at 119.9085 and 119.8882 respectively. The short input matched all 128 output IDs under `MMAP=0`. The local suite passed 175 tests with four platform skips, and the Cortex-A53 QEMU kernel suite passed.

Run the earlier benchmark command with the same GGUF and 128 generated tokens. Use `CTX=512/2048/4096` for the 256/1536/3584 inputs. For an in-binary comparison, set `MOTY_Q41_WORKSPACE=0 MOTY_Q41_AUTO_CHUNK=0` on the fallback run. Local logs, paired JSON, saved binary checksums and the memory diagnostics are in `.model-cache/rss-x2`.

### Experimental streaming of native Q4_1 layers

The CLI prototype keeps the tied head and a configurable prefix of layers resident. For the remaining layers it defers access to the GGUF weight pages, validates the Q4_1 group parameters once, and fills a bounded queue with `pread`. A layer runs directly from the original Q4_1 bytes in its queue slot, using the same NEON kernels and FP32 KV cache. The first prototype used two slots with the readers assigned to one layer at a time. The second assigns readers to individual slots and can load several future layers concurrently. Each slot is 5.20 MiB on Pocket.

The two-slot trial with eight resident layers and four readers reached 108.28 MiB and 299.89 tok/s on the 256-token input in the first screen. Its resident comparison reached 476.05 tok/s. A four-slot trial with six resident layers and two readers per slot reached a median of 108.53 MiB and 343.91 tok/s in a three-run screen, with 477.57 tok/s in the resident comparison. Longer paired runs below use this same four-slot layout on both primary workloads. The PP and decode comparisons preserve all 128 generated IDs on each workload.

| Input / workers | Resident → streaming peak RSS (MiB) | Resident → streaming PP (tok/s) | Resident → streaming TPS (tok/s) |
|---|---:|---:|---:|
| 256 / 12 | 202.25 → 108.48 | 1354.5 → 1344.9 | 489.56 → 283.53 |
| 3584 / 12 | 295.91 → 202.06 | 1048.1 → 1041.8 | 338.33 → 265.99 |

These are separate medians from seven alternating pairs after warm-up. The median paired PP changes were −0.74% and −0.86%. The median paired TPS changes were −42.08% and −21.70%. Both peaks clear the fixed 111.25 and 216.89 MiB targets. The decode regressions keep the combined goal open. On the short workload, the foreground spent a median of 0.148 s waiting for the reader queue across the request. On the long workload the wait was 0.063 s. The four-slot mode reads roughly 114 MiB per generated step from the warm file cache, and the reads share resources with inference even when they overlap.

The prototype matched all 457 generated IDs and 2310 teacher-forced top-1 IDs on M5 Max. Perplexity stayed at 119.9085. The short fixture also matched with `PREFILL_CHUNK=0/1/16`.

The Darwin mapping probe left RSS at 184.39 MiB after both `MADV_DONTNEED` and `MS_INVALIDATE` on touched read-only GGUF pages. Explicit map, fault and unmap of warm 5 MiB ranges reached about 18 GB/s and about 27 GB/s with `MADV_WILLNEED`, measured by touching one byte per page. The warmed four-worker `pread` probe reached 72 GB/s in the previous section. This rules out the advisory calls as RSS controls on the measured host and shows the cost of explicit remapping before inference work.

The prototype source, the loader overlay and the raw logs are local artifacts under `.model-cache/rss-stream`. `build.py` generates the CLI translation unit, `io.h` contains the queue, and `bench.py` runs the fixed token-ID workloads. The existing runtime defaults continue to use the resident Q4_1 path.

```sh
python3 .model-cache/rss-stream/bench.py 256 12 7 base h6d4r2
python3 .model-cache/rss-stream/bench.py 3584 12 7 base h6d4r2
python3 .model-cache/rss-stream/verify-ring.py
```

### Compact decode transport and verified batching

The follow-up builds a contiguous sidecar of the 196 Q4_1 layer matrices. It checks the shared parameters in every pair of blocks and stores one copy of the parameters in the existing lossless Group64 panel layout. The sidecar preserves every weight code. The hybrid prototype keeps the same resident prefix and reads raw Q4_1 during prefill to preserve throughput. During single-token decode and short verification batches it reads lossless panels for the streamed layers. For the six-resident-layer configuration, cold reads shrink from 114.38 to 102.94 MiB per model pass. Both layouts reuse the same queue slots. The sampled marginal entropy of the four-bit codes was 3.77 bits per code.

On the 256-input repeated-output benchmark, the hybrid prototype with the actual history-based `auto` draft mode measured 109.80 MiB and 586.37 tok/s versus 202.27 MiB and 494.65 tok/s for the resident control. The median paired TPS change was +18.54% and PP changed by −0.45% over seven alternating runs. All tokens matched. This fixture has 14 distinct output IDs across 128 tokens. The speed result depends on accepting enough proposals to amortize weight reads.

The same hybrid mode produced the same IDs on eight ordinary text prompts with EOS enabled and three 256-ID slices of the teacher-forced passage. Across three paired runs, the ordinary prompts gave no proposals and achieved 58.26% of resident decode throughput. The 256-ID slices used a fixed 128-token output with EOS ignored. They accepted 203/208 proposals and delivered 104.54% of resident decode throughput. The ordinary text result keeps automatic streaming from meeting the general speed requirement.

On 3584 inputs, nine resident layers and the hybrid transport without proposals gave 291.03 tok/s at 212.78 MiB in the three-run screen, versus 339.04 tok/s at 295.83 MiB for the resident control. Ten resident layers with history proposals gave 305.80 tok/s at 220.31 MiB in seven pairs, versus 340.65 tok/s for the resident control. That configuration exceeds the 216.89 MiB limit.

An isolated verification-capacity test fed reference continuations to the existing exact verifier. With two or four reference proposals per call it measured 465.66 and 541.53 tok/s on the long fixture at 203.80 and 205.91 MiB. The verifier scored every proposal and reproduced the reference output. These measurements exclude draft generation cost and cannot qualify as end-to-end generation results. At the measured batch cost, the four-proposal verifier would need about 2.15 accepted draft tokens per call on average to match the resident throughput, before counting any cost of producing drafts.

The tied output projection on intermediate hidden states matched the final top-1 ID on 2.1% of 193 ordinary-prompt steps after six resident layers, and 0.8% of 127 long-fixture steps. After 24 layers the rates were 66.8% and 75.6%. On 511 teacher-forced positions they were 0.8% and 48.5% respectively. The shallow projection has low agreement across the measured inputs. The late projection would already need most of the streamed layer weights to produce a draft.

Keeping all attention projections resident and staging only the FFN lets the current layer compute attention before it waits for its weights. With five resident FFNs and three staging slots this raised the long fixture to 329.34 tok/s at 217.36 MiB in a two-run screen. Removing one resident FFN, or reducing the queue to two slots, passed the RSS limit and yielded 312.07 or 308.53 tok/s respectively in the following three-run comparisons. This remains a separate prototype.

The `backoff` history search increased accepted proposals on the long fixed fixture from 17/23 in nine batches to 93/104 in 26 batches. Using the checked-in heuristic with the hybrid streaming prototype, separate medians from seven alternating pairs were:

| Input / workers | Peak RSS ceiling (MiB) | Resident → streaming peak RSS (MiB) | Resident → streaming PP (tok/s) | Resident → streaming TPS (tok/s) |
|---|---:|---:|---:|---:|
| 256 / 12 | 111.25 | 202.23 → 110.81 | 1370.8 → 1363.7 | 496.91 → 614.06 |
| 3584 / 12 | 216.89 | 295.91 → 216.23 | 1034.5 → 1036.1 | 338.72 → 473.79 |

Median paired TPS improved by 24.18% and 39.72%, while PP changed by −0.48% and −0.15%. The generated sequences matched the resident runs exactly. One long run reached 217.36 MiB. Two recombinations of the long context also matched and reached 150.98% of resident decode throughput across three paired passes, with an observed maximum of 218.52 MiB. The eight ordinary text prompts returned the same sequences, rejected all four proposals in each pass, and achieved a median of 59.27% of resident decode throughput. The `backoff` predictor is available by explicit opt-in. The streaming path stays in the local test harness while it has a decode penalty on ordinary prompts.

The final hybrid build with `backoff` matched the 457 generated IDs and 2310 teacher-forced top-1 IDs on M5 Max. Perplexity remained at 119.9085. The local suite passed 175 tests with four platform skips, and the Cortex-A53 QEMU kernel suite passed. Coverage includes conflicting suffix votes, a source occurrence beyond the original 1024-token search window, and generation with `PREFILL_CHUNK=0/1/16`.

The follow-up files under `.model-cache/rss-stream` include `pack_layers.py` for the sidecar, `build-panel.py` and `build-hybrid.py` for the hybrid prototype, `build-selective.py` for FFN-only streaming, and `bench-follow.py` with the saved run configurations. `run-robust.py` compares exact IDs and timing on additional prompt sets. The `ideal2` and `ideal4` configurations use reference proposals for measurement only.

### Tied head screening for exact greedy decoding

The tied output projection occupies 35.0 MiB in the original GGUF. Sharing the parameters of consecutive blocks and keeping the upper three bits of each code reduces the resident screening table to 24.5 MiB. The original rows remain in the GGUF. For each row, the three-bit midpoint gives a provisional logit. The omitted bit changes the real-valued score by at most half the scale times the sum of absolute activations in each group. One prototype also adds the error from its int8 activation approximation. Both prototypes add a margin for FP32 arithmetic. A threshold from exact logits of seed rows determines which original rows to read and evaluate with the existing Q4_1 kernel.

Across 457 recorded head inputs, offline screening with FP16 activations left a median of 9, 9 and 2 original rows on ordinary text, 256-token and 3584-token inputs, respectively. Dropping two bits raised the same medians to 58,530, 62,159 and 42,796 out of 65,536. Tracking the population of the omitted bit planes to tighten the bound still left a median of 52,481 rows on ordinary text in the two-bit trial.

The local NEON benchmark uses twelve workers and a warm GGUF. It includes approximate scoring, fetching up to sixteen seeds, searching and reading the candidates, and exact rescoring. All 457 top-1 IDs matched in each version and the recorded logits stayed within the upper bounds. The measurements below are medians in microseconds per output projection from each version's pass over the captured activations.

| Head inputs | Resident, SDOT run | 3-bit SDOT + exact | Resident, FMLAL run | 3-bit FMLAL + exact |
|---|---:|---:|---:|---:|
| Eight ordinary text prompts, 201 steps | 333 | 479 | 340 | 549 |
| Fixed 256-token input, 128 steps | 295 | 1415 | 337 | 528 |
| Fixed 3584-token input, 128 steps | 294 | 430 | 337 | 522 |

The FMLAL scan eliminates the int8 activation error and cuts the median number of candidate rows on the short fixed input from 1053 to 10. The scan alone still takes 401 µs on that input. The SDOT scan is faster and its wider bound leads to more row reads. The candidate and baseline shared a process in the microbenchmark, so the 10.5 MiB difference is a table size comparison. These figures do not establish end-to-end RSS or full-logit equivalence. The screening prototypes remain local because all measured workloads took longer in the head.

The capture, packing, bound calculation and NEON sources are under `.model-cache/rss-head`. The result logs are `bounds.log`, `bench-seeds16.log` and `bench-fmlal.log`.

```sh
p=.model-cache/rss-head
python3 "$p/capture.py"
python3 "$p/bounds.py"
python3 "$p/pack.py"
for source in bench-dot bench; do
  clang -O3 -mcpu=apple-m1 -Ic "$p/$source.c" c/libmoty-nn.a c/libmoty-hw.a \
    -L/opt/homebrew/opt/libomp/lib -lomp -lm -o "$p/$source"
  Q41_ACT=f16 "$p/$source" .model-cache/MNN-Pocket-0.3B-Instruct-Q4_1.gguf \
    "$(cat "$p/offset")" "$p/head3.bin" 12 "$p"/natural*.bin "$p"/fixed*.bin
done
```

### Reading cold FFN panels on the compute workers

This CLI prototype keeps attention and the tied head resident. Prefill uses the existing FFN queue and original Q4_1 layout. During single-token decode, each FFN worker reads its own contiguous rows from the lossless Group64 sidecar and runs the same NEON kernel using that slice. The workers reuse partitions of the prefill queue buffer after the queue drains. An optional single OpenMP region covers both gate/up and down, with a barrier to finish SwiGLU and prepare the down input. Verification batches keep the queue path.

The twelve-worker screen used two resident FFNs at 256 input tokens and five at 3584. Each input produced the same 128 IDs as the resident control. These medians are from three passes after warm-up. Decode uses greedy generation without draft proposals.

| Input / workers | Implementation | PP (tok/s) | TPS (tok/s) | Peak RSS (MiB) |
|---|---|---:|---:|---:|
| 256 / 12 | Resident control | 1357.0 | 491.93 | 202.34 |
| 256 / 12 | FFN queue | 1360.3 | 262.56 | 103.81 |
| 256 / 12 | Compute-worker reads, 16 panels per stripe | 1335.7 | 187.92 | 104.09 |
| 3584 / 12 | Resident control | 1045.3 | 340.77 | 295.86 |
| 3584 / 12 | FFN queue | 1038.1 | 293.87 | 211.27 |
| 3584 / 12 | Compute-worker reads, 16 panels per stripe | 1036.8 | 176.87 | 210.34 |

The short follow-up used 1, 2, 3, 4, 6 or 8 workers for cold FFN alone, while other operators retained twelve. Six workers with the 16-panel stripe yielded 188.85 tok/s, compared with 491.42 for the resident control and 286.71 for the FFN queue in that pass. A large slice per worker reached 160.16 tok/s with four workers and 131.89 tok/s with six. The whole-model slowdown persists across worker counts.

In a separate warmed read-only trial, `pread` on one shared descriptor reached 87.3 GB/s with six workers and 95.1 GB/s with twelve, using 128 KiB requests into reused worker buffers. With 32 KiB requests the rates were 60.6 and 32.2 GB/s. Separate descriptors reached 90.4 and 91.8 GB/s at 128 KiB. Disabling read-ahead on those descriptors gave 93.1 and 95.4 GB/s. The isolated read rate depends on operation size and concurrency. The end-to-end direct path also puts reads on the FFN critical path and loses overlap with the preceding operators.

The short diagnostic timed the slowest worker in each of 6,604 FFN stages. With twelve workers and 16-panel stripes, those workers spent a cumulative 364 ms reading panels and 114 ms computing. The stage totals include per-worker scheduling delays while executing those calls. Using passive OpenMP waits lowered the measured read time to 213 ms, while the full decode time rose from 0.66 s to 1.18 s. Scheduling and synchronization outside the measured calls also affect the outcome.

The prototype remains local. Its runs met the RSS ceilings on the two fixed workloads, while the read queue preserved more decode throughput.

### Shared background readers for the cold FFN queue

A shared FIFO assigns the earliest unclaimed stripe to the next reader. It decouples the number of reader threads from the number of queue slots. In the best first screen, four readers serviced two slots with 1 MiB stripes. The selector keeps three FFNs resident at 256 input tokens and six at 3584. The GGUF and original FP32 K/V buffers are unchanged. Compared with the resident control, all 128 generated IDs matched on each workload.

| Input / workers | Resident → shared pool PP (tok/s) | Resident → shared pool TPS (tok/s) | Resident → shared pool peak RSS (MiB) |
|---|---:|---:|---:|
| 256 / 12 | 1362.1 → 1344.4 | 494.41 → 411.56 | 202.25 → 108.11 |
| 3584 / 12 | 1036.9 → 1036.1 | 335.41 → 316.10 | 295.95 → 214.11 |

Medians are from three alternating passes. Median paired TPS changes were −16.88% and −5.76%. Foreground wait for the queue across the complete request was 35 ms and 3 ms respectively. In the same passes, the older two-slot queue reached 288.45 and 292.42 tok/s with two and five resident FFNs. A one-slot pool can spend the freed space on one more resident FFN. In the short screening pass it reached up to 240.59 tok/s at 107.84 MiB, since it can only request the next cold layer after the current FFN releases the slot. Fusing the two projection stages into one OpenMP region gave no consistent gain.

The next overlay schedules the `gate/up` stripes first and lets the FFN begin when those stripes finish. The `down` stage waits for its own input weights. The sidecar's byte layout stays the same. A three-pass screen with four readers measured 405.78 tok/s at 108.14 MiB with 768 KiB stripes on the short input, compared with 388.80 tok/s when the same binary waited for all three matrices and read 1 MiB stripes. With the 1 MiB stripe size fixed, starting early delivered 394.40 tok/s. On the long input the early-start 1 MiB variant measured 310.57 tok/s at 214.16 MiB versus 319.50 tok/s for the resident control. Prefill uses the original full readiness check.

On eight ordinary text requests, the shared pool and early-start variants generated the same 201 tokens as the resident path, with EOS enabled. Across three passes the median aggregate decode rates were 515.12 tok/s for resident, 425.39 tok/s for the pool and 429.00 tok/s for early start. The median of each pass's highest process peak RSS was 197.81 MiB for resident and 103.23 MiB for both streaming variants. In a separate 256-token test, giving the four background readers user-initiated QoS reached 406.81 tok/s versus 399.88 at default QoS. Utility QoS fell to 329.14 tok/s. The resident control reached 469.42 tok/s in those passes.

With the short RSS budget held at about 108 MiB, replacing one resident FFN with a third queue slot raised median decode throughput from 405.07 to 412.78 tok/s. A fourth slot and one resident FFN gave 396.89 tok/s. The resident control reached 470.81 tok/s. The early-start variant with three slots reached 411.39 tok/s in that comparison.

Both the six-worker direct-read variant with 16-panel stripes and the early-start pool with 768 KiB stripes matched all 457 generated IDs and all 2310 teacher-forced top-1 IDs. Perplexity remained 119.9085. The short input matched with `PREFILL_CHUNK=0/1/16`. These overlays remain experimental because decode on ordinary text still trails the resident path.

The CLI overlays, binaries and raw logs are local under `.model-cache/rss-direct`. The commands below compare the initial pool with the resident binary and the older pair of per-slot reader teams.

```sh
p=.model-cache/rss-direct
bash "$p/build.sh"
bash "$p/build-pool.sh"
python3 "$p/bench.py" 256 12 3 base direct2-0-0 pool3-4-1024-2
python3 "$p/bench.py" 3584 12 3 base direct5-0-0 pool6-4-1024-2
python3 "$p/build-staged.py"
bash "$p/build-staged.sh"
python3 "$p/bench.py" 256 12 3 base staged3-4-1024-2 staged3-4-768-2s
python3 "$p/bench-natural.py" 3 base pool3-4-1024-2 staged3-4-1024-2s
python3 "$p/verify.py" staged3-4-768-2s
```

### Independent gate/up and down queues

The split overlay gives gate/up one more queue slot than down. The extra slot holds 2.46 MiB of lossless panels. The other slots reuse the prefill buffers. A gate/up slot can be recycled once SwiGLU has been written. A down slot is recycled after its projection completes. Prefill and batched verification wait on complete layers. An optional priority rule orders the requests by layer, with gate/up ahead of down for the same layer.

| Input / workers | Path | PP (tok/s) | TPS (tok/s) | Peak RSS (MiB) |
|---|---|---:|---:|---:|
| 256 / 12 | Resident | 1349.7 | 469.37 | 201.97 |
| 256 / 12 | Full-layer pool | 1348.5 | 396.23 | 108.11 |
| 256 / 12 | Split queues, 768 KiB stripes | 1351.0 | 414.61 | 110.66 |
| 3584 / 12 | Resident | 985.9 | 324.04 | 295.88 |
| 3584 / 12 | Full-layer pool | 986.4 | 312.61 | 213.97 |
| 3584 / 12 | Split queues, 1024 KiB stripes | 983.1 | 304.01 | 216.58 |

These are medians of three alternating passes. The short split cut foreground queue wait from 35 ms to 10 ms across the request. The long split was slower than the full-layer pool. The best short packed-prefix variant in a subsequent screen kept four FFNs resident and measured 428.31 tok/s at 109.70 MiB, against 461.26 tok/s for the resident control. Its PP was 1316.2 against 1332.1 tok/s. The three-resident-FFN variant with a packed tied head and five readers reached 423.11 tok/s at 106.98 MiB in the first screen of that layout.

On eight ordinary text prompts, the packed-head, five-reader variant matched all 201 tokens and reached an aggregate median of 449.12 tok/s at 102.25 MiB. In the same comparison the full-layer pool reached 415.90 tok/s at 103.20 MiB and the resident control reached 505.31 tok/s at 197.81 MiB. The packed-head variant also matched all 457 generated IDs and all 2310 teacher-forced top-1 IDs. Perplexity remained 119.9085. The short fixture matched with `PREFILL_CHUNK=0/1/16`.

In a separate operator profile of 128 short-context decode steps, the resident control took 282.4 ms, the full-layer pool took 323.4 ms and the packed-head split took 303.4 ms. The split reduced total foreground queue wait from 32 ms to 8 ms. The same split spent 65.25 ms in down, which includes the down readiness check, versus 55.10 ms in the resident control. Its attention core took 31.60 ms versus 25.76 ms for resident, even though it uses the same attention path. This comparison shows that shrinking the queue wait alone leaves operator time to recover. A bounded 25 µs spin on the foreground wait raised the full-layer pool from 406.10 to 412.05 tok/s in its own short screen. A 60 µs spin yielded 396.07 tok/s.

With moty's pthread compute backend and a 100 µs pool spin, the split queue reached 448.14 tok/s at 105.69 MiB on the short fixture. The corresponding pthread resident executable reached 511.77 tok/s. The OpenMP split in the same screen reached 443.20 tok/s versus 477.06 for its resident control. The backend switch therefore did not close the gap.

The split overlays remain local. The full-layer pool is still the better trade-off on the long workload. Ordinary prompt decode with split queues still trails the resident control by 11%.

```sh
p=.model-cache/rss-direct
python3 "$p/build-split.py"
bash "$p/build-split.sh"
python3 "$p/bench.py" 256 12 3 base pool3-4-1024-2 split3-4-768-2
python3 "$p/bench.py" 3584 12 3 base pool6-4-1024-2 split6-4-1024-2
python3 "$p/bench-natural.py" 3 base pool3-4-1024-2 split3-5-768-2ho
python3 "$p/verify.py" split3-5-768-2ho
```

### Reader limits during decode operators

The local dispatch experiment caps the number of new reader tasks during selected operators. It lets already-issued reads finish and restores the full pool before any foreground weight wait. Prefill and verification batches run with the full pool. On the 256-token input, the shared control with independent queues reached 436.19 tok/s at 106.92 MiB. Its resident control reached 470.61 tok/s. Stopping new reads during down reached 417.24 tok/s. A two-reader cap reached 412.38 during down, 416.36 during the attention core and 390.64 tok/s during both complete attention and down. The gate/up cap of one reader reached 404.42 tok/s. These are medians from three alternating runs with identical greedy IDs. The caps traded overlap for lower reader concurrency and lowered whole-request decode throughput.

On the 3584-token input, the full-layer pool without throttling reached 303.31 tok/s. The two-reader attention-core cap reached 305.17, and the complete-attention cap reached 299.56. The resident control reached 329.82 tok/s. These comparisons left the speed gap open.

### Bounded mmap windows and operator timing

A read-only mapping trial used the same lossless sidecar and replaced each anonymous slot with a file-backed view at decode. Lazy faults with `MADV_WILLNEED` reached 212.91 tok/s at 104.38 MiB on the 256-token fixture. Prefaulting with four readers without advice reached 300.62 tok/s and had a median observed peak of 419.16 MiB. The same binary using buffered `pread` reached 416.16 tok/s at 104.64 MiB. Mapping with both advice and prefaulting reached 208.11 tok/s at 106.28 MiB. All generated IDs matched in the three-run screen.

The parallel FFN packing trial wrote the same FP16 activations and FP32 sums in disjoint groups of 32 immediately after SwiGLU. The best three-resident-layer packing trial reached 433.76 tok/s versus 432.19 with the reference packing schedule and 479.11 for resident. Reusing an arena for single-token intermediate rows reached 428.32 tok/s with streaming, versus 442.74 on the same streaming configuration with the arena disabled. The resident-path reuse here applies only to the existing normalization buffer after the last layer, which removes one 3584-byte activation allocation per Pocket forward.

For visible lengths up to 512, the paired GQA trial reused K and V across adjacent heads while preserving each head's FP32 reduction order and final causal position. The model run reached 442.17 to 442.18 tok/s for one or two pairs per KV head, versus 443.03 with the existing schedule. A standalone check matched softmax scores and output values bit for bit at 4096 tested input lengths, including every length from 1 to 512 for eight input sets. The model runs also matched their 128 generated IDs. The slower transport and operator variants remain local experiments.

A separate layout screen held the same number of FFNs resident and changed their layer positions. Keeping the first three FFNs resident gave 444.98 tok/s at 107.02 MiB on the short fixture. Evenly spaced, first-and-last, and middle-cluster layouts gave 421.31 to 429.61 tok/s. On the long fixture the six-layer prefix reached 317.72 tok/s at 214.05 MiB. Spreading the six resident layers gave 314.94 to 317.59 tok/s. The screen used three alternating runs and matched all generated IDs.

The local harness, overlays and logs are under `.model-cache/rss-direct`.

```sh
p=.model-cache/rss-direct
python3 "$p/build-reader-budget.py"
bash "$p/build-budget.sh"
python3 "$p/bench.py" 256 12 3 base budget3-5-768-2ho budget3-5-768-2hom
python3 "$p/build-mapped-queue.py"
bash "$p/build-map-queue.sh"
python3 "$p/bench.py" 256 12 3 base mapped3-4-1024-2h mapped3-4-1024-2hv
bash "$p/build-fused-pack.sh"
python3 "$p/bench.py" 256 12 3 base pack3-5-768-2ho pack3-5-768-2hov
bash "$p/build-short-gqa.sh"
python3 "$p/bench.py" 256 12 3 base short3-5-768-2ho short3-5-768-2hoi
```

### Reusing the final normalization buffer

The Qwen forward reuses its existing normalization buffer for the final hidden vector. It removes one 3584-byte allocation on Pocket. Over fourteen alternating pairs of otherwise identical builds at 256 input tokens, the median paired change was +0.06% TPS, −0.10% PP and −0.02% peak RSS. This change removes the allocation without a measurable reduction in process peak RSS. At 3584 inputs, the seven comparisons against the saved resident binary gave −0.30% TPS and −0.13% PP.

The updated binary matched all 457 generated IDs and 2310 teacher-forced top-1 IDs on Apple. Perplexity stayed at 119.9085. The native suite passed 175 tests with four platform skips. The ARMv8.0 build passed the Cortex-A53 QEMU kernel suite and matched 32 generated IDs in a short-context smoke comparison.

### Shared gate/up panels and exact KV storage

The local sidecar interleaves pairs of gate and up rows inside each Group64 panel. Reversing the transform reproduced every parameter and weight byte across the 28 layers. The FMLAL kernels retain the original per-row FP32 updates. Converting four scale/minimum pairs in parallel avoids the individual conversions of the first prototype. The cold joint panel is used only during single-token forward. Prefill and verification batches load the original layout.

Both the 2+2 and 4+4 kernels matched the separate panel path bit for bit across all 168 layer, precision and tile checks, including FP16 overflow fallback and explicit FP32. With twelve workers and the last Pocket layer, the existing single-projection passes measured 59.13 µs for the complete gate/up projection, including SwiGLU. The separate 2+2 kernel measured 59.49 µs. The interleaved 2+2 and 4+4 kernels measured 58.22 and 59.01 µs in the same sweep.

On the 256-token fixture, the separate streaming control reached 381.85 tok/s, the interleaved 2+2 variant reached 394.53 tok/s and the interleaved 4+4 variant reached 397.32 tok/s. The latter's median paired improvement against separate panels was 3.34%. Its median paired gap to the resident executable was 9.32%. The two interleaved versions used 107.17 MiB peak RSS. These results are from seven alternating passes. On the eight ordinary text prompts, interleaved 2+2 reached 398.26 tok/s versus 401.50 for separate panels and 497.07 for the resident control. Every generated ID matched. The interleaved 2+2 build also matched all 457 generated IDs, all 2310 teacher-forced top-1 IDs and perplexity 119.9085. It matched the short fixture with `PREFILL_CHUNK=0/1/16`. The interleaved layout remains a local experiment.

The same quality corpus rejected the existing `KV_BITS=8` mode as a replacement for FP32. It changed two of eight ordinary response sequences and 41 of 2310 top-1 IDs. Perplexity rose from 119.9085 to 120.0959, a 0.156% increase. On the fixed 3584-token input the 128 generated IDs matched. In that single comparison, RSS decreased from 295.83 to 223.12 MiB and decode fell from 318.51 to 150.24 tok/s.

A reversible FP32 codec keeps each sign and mantissa and substitutes a four-bit index for common exponent values. An out-of-dictionary exponent selects the original FP32 representation for the sixteen-value group. A NEON round trip used tables calibrated on a fixed 3584-token capture, then reconstructed a distinct 3584-token text cache bit for bit. Encoded storage, including tags and tables, was 85.98 MiB versus 98.00 MiB of raw K and V. One worker reconstructed the whole cache in 1.924 ms, versus 1.454 ms for a raw memory copy on the same host. These are standalone storage and decode results. The following section measures the same approach on the full model.

```sh
p=.model-cache/rss-joint
python3 "$p/pack.py"
bash "$p/build.sh"
bash "$p/acceptance.sh"
python3 "$p/check-kv.py"
bash "$p/dump-kv.sh"
clang -O3 -mcpu=apple-m1 "$p/kv-codec.c" -o "$p/kv-codec"
"$p/kv-codec" "$p/kv-long.bin" "$p/kv-text.bin"
```

### Exact KV storage inside GQA

The local full-model overlay uses 224 bytes for a normal 64-value FP32 KV row. It stores sign and mantissa intact and replaces the exponent with a four-bit code. For a row with an uncommon exponent, an index refers to a reusable page of original exponent bytes. Decoding reconstructs the original 32-bit values. The tests also exercised exception slots across writes to the same positions and slices starting or ending on either side of a 128-row boundary.

The first full-model variant reconstructed one layer's visible FP32 cache before running the original attention. On the 3584-token workload, storing both K and V in this format lowered resident peak RSS from 295.76 to 288.68 MiB. Decode moved from 311.47 to 278.24 tok/s in a two-pass comparison using the same executable. The process peak includes the shared reconstruction buffer, exception pages and live allocations in the rest of the engine.

For single-token decode at the existing GQA threshold, each worker now reconstructs one 128-position block and immediately consumes it for the 3+3+1 query groups. The score, softmax, value accumulation and merge arithmetic stay in FP32 with the original ordering. Prefill, small contexts and verification batches use the layer staging path. The following are medians over three alternating runs with twelve workers and 128 matching generated IDs per run.

| Long-context transport | PP (tok/s) | TPS (tok/s) | Peak RSS (MiB) |
|---|---:|---:|---:|
| Resident control | 1031.9 | 332.45 | 295.92 |
| Six resident FFNs, raw FP32 KV | 1033.5 | 310.31 | 213.89 |
| Eight resident FFNs, exact KV, layer staging | 1028.1 | 278.15 | 214.91 |
| Eight resident FFNs, exact KV, block reconstruction | 1034.1 | 304.74 | 215.72 |
| Nine resident FFNs, packed tied head, exact KV, block reconstruction | 1031.5 | 308.91 | 215.94 |

The block scheme recovered 26.59 tok/s over layer staging at the same resident FFN count. The other layout trial used planar mantissa bytes and vectorized exponent encoding. Its long run with eight resident FFNs measured 306.95 tok/s, compared with 307.29 for the original encoded layout and 318.57 for the raw-KV streaming control in the same set. The codec saved space for more resident FFNs, while their throughput remained below the raw-KV configuration.

Releasing the shared reconstruction area on entry to single-token GQA lowered the median peak for the eight-layer planar run from 215.80 to 215.50 MiB. Its median decode was 300.17 tok/s versus 304.51 with the area retained and 316.41 with the raw-KV streaming control. Nine resident FFNs with release reached a median 216.61 MiB, with one run at 217.03 MiB. Exact history verification with this release path preserved all generated IDs on the long fixture. Its two runs peaked at 219.78 and 222.80 MiB because batched verification uses the layer staging area.

With the codec disabled and both K and V read directly from FP32 memory, the block schedule itself reached 308.11 tok/s versus 311.96 for the regular six-layer streaming control. This isolates the change in scheduling from the cost of reconstructing values.

The standalone GQA check matched the original output bit for bit at 2048, 2049, 2177, 3584, 3711 and 3712 positions for every K/V compression mask. It also confirmed fallback at 2047. With K and V captured from the first Pocket layer, the 3584-position microbenchmark took 155.60 µs with the original kernel, 154.39 µs with the new scheduling and raw data, and 163.67 µs with both cache matrices compressed. An independent synthetic check covered out-of-dictionary exponent values and returned the same bits.

Keeping raw FP32 K and V and packing the tied head plus seven resident FFNs reached 317.59 tok/s at 211.62 MiB. The six-resident-FFN raw KV control reached 316.82 tok/s at 214.09 MiB in that three-run screen. The resident executable reached 331.83 tok/s. PP was 1023.8 tok/s with seven packed FFNs, versus 1039.5 for six with the original FFN layout. This combination also stayed below the RSS ceiling and kept all 128 IDs.

The planar build matched all 457 generated IDs and 2310 teacher-forced top-1 IDs. Perplexity stayed at 119.9085. It also matched the 256-token input with `PREFILL_CHUNK=0/1/16`. The codec code is an executable experiment under `.model-cache/rss-kv`, and the shipped runtime still uses the existing FP32 cache.

```sh
p=.model-cache/rss-kv
bash "$p/build.sh"
python3 .model-cache/rss-direct/bench.py 3584 12 3 base codec6 codec8b codec8bf codec9bfh
python3 .model-cache/rss-direct/verify.py codec8bfp
```

### Independent residency for FFN projections

The local overlay keeps gate/up together and selects which down projections stay resident independently. Cold weights still use bounded buffers and preserve their GGUF values. A layer can read gate/up, down, or the entire FFN depending on what is already resident. For the short fixture, with twelve workers and identical generated IDs, retaining three complete FFNs gave 410.85 tok/s at 103.20 MiB. Keeping eleven down projections gave 392.80 tok/s at 105.95 MiB. Keeping five gate/up pairs gave 396.98 tok/s at 104.45 MiB. Distributing either family across the layer list raised these last rates to 403.09 and 400.60 respectively. The resident control reached 485.45 tok/s. Each number is the median of three alternating runs.

In a follow-up with five complete FFNs resident, packing only Q/K/V reduced peak RSS from 110.58 to 109.06 MiB. Packing only O reached 110.16 MiB, and packing Q/K/V plus O reached 107.77 MiB. PP changed from 1264.9 tok/s without packed attention to 1243.8 with packed Q/K/V and 1252.4 with all packed attention. The six-resident-FFN variant with all attention packed reached 111.47 MiB against the 111.25 MiB ceiling. Its PP was 1193.7 tok/s against 1292.3 for the resident control in that comparison.

Balancing short ranges over all reader threads did not reverse the result. In a separate paired screen the complete-FFN control reached 375.75 tok/s and the balanced variant reached 379.05. Retaining eleven down projections with balanced reads gave 378.53 tok/s, using 105.67 MiB versus 103.47 MiB for the balanced complete-FFN configuration. The five gate/up pairs with balanced reads reached 354.70 tok/s. On 3584 inputs, seven complete FFNs gave 295.89 tok/s at 211.72 MiB. Keeping 21 down projections at 211.75 MiB gave 285.38 tok/s, and ten gate/up pairs at 210.67 MiB gave 287.04 tok/s.

Rebuilding the same source and runtime libraries with Apple Clang's `-O3 -mcpu=apple-m5` did not close the remaining gap. With 3584 input tokens, the new build reached 307.97 tok/s for seven packed FFNs, compared with 307.17 using the original streaming build and 318.99 using the resident build with the new compiler options. Both streaming builds used about 212 MiB. On the short workload the native build reached 397.17 tok/s with five packed FFNs at 110.67 MiB, versus 465.58 tok/s for its resident counterpart. Generated IDs were identical across each three-run comparison.

On eight ordinary text prompts the whole-FFN and down-only policies both matched all 201 generated IDs. Their aggregate median rates were 397.90 and 392.69 tok/s, compared with 506.93 for the resident control. The balanced down-resident build also matched all 457 generated IDs and 2310 teacher-forced top-1 IDs, with perplexity 119.9085.

Allowing some attention projections to stream made room for more FFNs under the same approximate RSS budget. On the short fixture, retaining five complete FFNs and all attention gave 385.18 tok/s at 107.80 MiB. Keeping seven FFNs and attention for twenty layers gave 353.15 tok/s at 109.12 MiB. With nine FFNs and attention for twelve layers, this fell to 327.06 tok/s at 108.80 MiB. On the long fixture, the all-attention control reached 295.31 tok/s at 211.62 MiB. The variant with ten FFNs and fifteen resident attention layers reached 266.74 tok/s at 209.02 MiB. Adopting an attention-cold slot before attention shortens the overlap available to hide the reads. All of these comparisons preserved the generated IDs.

The local sources and logs are under `.model-cache/rss-partial`.

```sh
p=.model-cache/rss-partial
python3 "$p/build.py"
python3 "$p/build-attention.py"
python3 "$p/build-native.py"
python3 .model-cache/rss-direct/bench.py 256 12 3 base partial3-3-4-1024w partial0-11-4-1024wa partial5-0-4-1024wa
python3 .model-cache/rss-direct/bench.py 256 12 3 base partial5-5-4-1024at28wko partial7-7-4-1024at20wko
python3 .model-cache/rss-direct/verify.py partial0-11-4-1024wa
```

### Bounded reads for the tied output head

The isolated head transport test streams the complete lossless Group64 panel file through two buffers. It measures the full projection after the input vector is ready. Across 128 captured Pocket inputs, all 65,536 logits matched the mapped projection bit for bit on every input and every tested buffer size. The fastest measured streamed result was 819.9 µs versus 349.6 µs mapped, with 15.75 MiB of buffers and four readers. At 3.94 MiB of buffers the best measured time was 1172.5 µs versus 339.3 µs mapped. Retaining one OpenMP team for the whole projection produced 887.0 µs with four readers at 15.75 MiB. The standalone figures measure projection latency and queue buffers.

```sh
p=.model-cache/rss-partial
clang -O2 -mcpu=apple-m5 -Ic "$p/head-transport.c" c/libmoty-nn.a c/libmoty-hw.a \
  -lm -L/opt/homebrew/opt/libomp/lib -lomp -o "$p/head-transport"
Q41_ACT=f16 "$p/head-transport" \
  .model-cache/MNN-Pocket-0.3B-Instruct-Q4_1.gguf.moty-q41-2442848 \
  .model-cache/rss-head/fixed256.bin 4096 4 12
```

### Vectorized scale and minimum for packed GEMV

The 4-row FMLAL kernel can load four FP16 scale/minimum pairs with one NEON load, convert in two vectors and select the appropriate parameter lane in each FP32 update. The update order and block reductions match the original packed kernel.

All 28 packed down and gate/up layer checks and the full tied head matched bit for bit. In the long sweeps on the last layer, down went from 48.06 to 47.41 µs, gate/up from 57.27 to 56.19 µs, and the head from 370.91 to 355.81 µs. A separate test found less than 1 µs difference between a 128-byte-aligned head payload and the same bytes with an eight-byte offset.

With split queues and four resident FFNs, seven paired 256-input runs measured 427.21 tok/s using the old kernel and 443.87 using vector parameters at 109.45 MiB. The median paired TPS gain was 3.30% over that streaming control. The median paired gap to the resident executable was 9.47%. The same build matched all 457 generated IDs, 2310 teacher-forced top-1 IDs and perplexity 119.9085. On eight ordinary prompts, the vector-parameter path gave 451.53 tok/s versus 438.23 for separate scalar parameter loads in a three-pass comparison, with all 201 IDs matching.

One OpenMP region can span gate/up, the foreground wait for down, and the down projection. On the seven-run short fixture, combining the vector path with the shared region reached 431.34 tok/s at 109.70 MiB. On 3584 inputs the six-resident-FFN combination reached 317.04 tok/s at 210.84 MiB versus 336.41 in the resident control. Both timing sets preserved all generated IDs.

Packing the completed SwiGLU slices inside the shared team reached 436.52 tok/s at 109.62 MiB. The same trial reached 443.49 without parallel input packing in that team and 444.35 with separate projection calls. Parallel packing stays in the local experiment.

For the existing resident packed path, directly replacing the four-row GEMV kernel moved decode from 501.32 to 510.09 tok/s on 256 inputs. The seven-run paired median was +1.60% TPS, −0.26% PP and effectively flat RSS at 187.62 MiB. On 3584 inputs, the three-run paired median was +2.54% TPS and −0.88% PP. The vector parameter load is now the implementation of the existing packed FMLAL GEMV. The format and raw Q4_1 kernels are shared with the earlier tests.

The rebuilt `moty` matched 457 generated IDs and all 2310 teacher-forced top-1 IDs with `MOTY_Q41_PACKED=1`. Perplexity stayed at 119.9085, including the same short fixture outputs with `PREFILL_CHUNK=0/1/16`. The four-worker packed control at 1536 input tokens reached 224.48 tok/s against 214.93 in the earlier packed executable. All three paired runs generated identical tokens. The native suite passed 175 tests with four skips, and the ARMv8.0 kernel tests passed under `qemu-aarch64 -cpu cortex-a53`.

```sh
p=.model-cache/rss-params
python3 "$p/build.py"
python3 "$p/build-once.py"
bash "$p/build-once.sh"
python3 "$p/build-fixed.py"
python3 .model-cache/rss-direct/bench.py 256 12 7 base params4-bso params4-vso
python3 .model-cache/rss-direct/bench.py 256 12 7 fixedold fixedvec
python3 .model-cache/rss-direct/verify.py prodpacked
```

### One worker team per decode token

The prototype holds one OpenMP team over a single-token Qwen forward. It first used one barrier to publish each operator and a second barrier to join its workers. With four resident packed FFNs and split queues, the two-barrier variant reached 436.04 tok/s versus 452.80 with separate regions on the 256-input fixture. At 3584 inputs, the seven-resident-FFN variant reached 310.17 versus 324.68 tok/s. Each comparison used three alternating runs.

A second scheduler publishes each job through an atomic generation counter. Every worker acknowledges completion on its own cache line. With split queues and four resident packed FFNs, the short comparison used seven alternating runs with one second between processes. PP, TPS and RSS are separate medians.

| 256 inputs / 12 workers | PP (tok/s) | TPS (tok/s) | Peak RSS (MiB) |
|---|---:|---:|---:|
| Resident control | 1346.8 | 490.78 | 202.25 |
| Separate regions, split queues | 1339.9 | 448.09 | 109.66 |
| Shared team, per-worker acknowledgement | 1341.1 | 450.96 | 109.70 |

The median paired decode gain over separate regions was 1.89%. Against the resident control, the median paired changes were −8.11% TPS, −1.31% PP and +3.82% TTFT. The highest measured RSS for the per-worker variant was 110.92 MiB against the fixed ceiling of 111.25 MiB. All three executables returned identical IDs for the 128 generated tokens.

The long-context prototype can also execute the existing score, softmax, V and merge phases inside the decode team. During consecutive long runs the resident control fell below 150 tok/s. A standalone rerun recovered to 340.03 tok/s. With a 12-second pause between processes, two alternating runs gave the following preliminary numbers.

| 3584 inputs / 12 workers | PP (tok/s) | TPS (tok/s) | Peak RSS (MiB) |
|---|---:|---:|---:|
| Resident control | 1046.1 | 333.10 | 296.50 |
| Separate regions, full-layer queues | 1027.3 | 319.47 | 212.45 |
| Shared team, GQA using OpenMP phase barriers | 1024.3 | 333.39 | 211.94 |

The shared team returned to resident decode speed in this short screen. PP was 2.08% lower, and its highest RSS was 211.95 MiB against 216.89 MiB. The earlier timing series with the slow resident controls is excluded from these results.

The standalone GQA comparison exercised 2047, 2048, 2049, 2177, 3584, 3711 and 3712 visible positions. The 2047 case selected fallback and outputs from 2048 onward matched bit for bit at four and twelve workers. A twelve-worker synthetic GQA comparison at 3584 positions used 56 calls per pass and seven alternating passes. The separate region, team with OpenMP phase barriers and team with per-worker acknowledgements for each phase measured 150.13, 324.18 and 31.89 µs per call, respectively. The synthetic test uses one set of K/V buffers and excludes FFN reads, so the phase dispatch results require a full-model comparison.

On the full model, dispatching each GQA phase separately through the same team's acknowledgements reached 328.78 tok/s at 211.97 MiB versus 338.79 tok/s for resident. The three alternating pairs gave −3.78% TPS, −1.38% PP and a maximum RSS of 212.14 MiB. The synthetic dispatch benefit did not carry over to the model.

Increasing the automatic prefill chunk from 256 to 320 or 384 preserved all 128 generated IDs in the warm-ups. Peak RSS was 211.95, 213.58 and 215.48 MiB respectively. The resident control fell to 121.20 tok/s in the timed series, so that series was stopped and no prefill speed conclusion was drawn.

The per-worker variants with and without OpenMP GQA phase barriers matched all 457 generated IDs and all 2310 teacher-forced top-1 IDs, with perplexity 119.9085. The 256-input fixture also matched for `PREFILL_CHUNK=0/1/16`. With `KV_BITS=8`, the long-context prototypes and the short-context idle variant matched all 128 generated IDs of the resident fallback on the 256-input fixture.

A two-run scheduling screen compared smaller row ranges and shorter spin windows. Work-stealing thresholds of 8, 16 and 64 jobs reached 448.00, 438.70 and 420.04 tok/s. Allowing workers to sleep after 10, 30 and 100 µs gave 145.81, 426.59 and 455.46 tok/s at the same RSS budget. The per-worker control reached 446.59 and the resident control 484.13 tok/s. In seven further alternating runs, the 100 µs window reached 454.32 tok/s at 109.50 MiB. The highest RSS was 111.12 MiB. Median paired TPS increased by 1.69% over continuous polling and remained 7.04% below the resident control. Relative to resident, PP changed by −0.90% and TTFT by +2.37%. All 128 generated IDs matched on every run.

The 100 µs variant also matched 457 generated IDs, 2310 top-1 IDs and perplexity 119.9085. Reducing the gate/up queue from three slots to two made room for a fifth resident FFN. This kept all 128 IDs and reached 110.92 MiB, with a maximum of 110.95 MiB in the three-run test. The resident control slowed to roughly 312 tok/s in that series. The smaller queue has a validated memory result and needs a stable speed comparison.

With one second between processes, the eight ordinary requests reached 543.74 tok/s for resident, 467.14 tok/s at 104.45 MiB for four resident FFNs, and 474.25 tok/s at 108.45 MiB for five resident FFNs. Both streaming runs used the 100 µs spin window and matched all 201 generated IDs in each of two passes.

For the 2+2 packed gate/up kernel, loading the four scale/minimum pairs from two panels as vectors kept SwiGLU outputs bit-identical across all 28 Pocket layers. On the last layer the isolated latency changed from 56.34 to 55.59 µs over 512 inputs. On the full 256-input fixture, three alternating runs gave 453.70 tok/s for separate panel walks, 450.10 for the existing joint kernel and 424.45 for the joint kernel with vector parameter loads. The three paths matched every generated ID.

An operator profile after a 45-second pause matched the 128 generated IDs on all three alternating runs. With four resident FFNs, decode took 281.0 ms compared with 263.1 ms for the resident control. The down projection and its readiness check took 65.15 versus 50.99 ms. Q/K normalization and RoPE took 11.68 versus 3.39 ms. The OpenMP team reported its coordinator as parallel even while it was running serial model code, so RoPE bypassed its position table and recomputed the trigonometric values for each head. The prototype now allows the coordinator to use the original table between submitted jobs. Worker callbacks still use the parallel fallback.

The restored cache raised the four-resident-FFN variant to 461.52 tok/s at 109.44 MiB. The seven-run paired gain was 1.52% over the same worker schedule with uncached RoPE. The median paired change from the resident control was −6.75% TPS and −1.00% PP. The highest observed RSS was 109.69 MiB. With five resident FFNs and the smaller queue, decode reached 455.72 tok/s and one run exceeded the 111.25 MiB ceiling at 112.06 MiB. Both variants matched all 457 generated IDs and all 2310 top-1 IDs with perplexity 119.9085.

In the repeat profile, Q/K normalization and RoPE took 3.66 ms per 128 decoded tokens in the four-resident-FFN variant and 3.35 ms for resident. Down including its readiness check took 69.39 versus 50.89 ms. This was the largest remaining measured operator difference. The five-resident-FFN variant moved down to 52.40 ms and increased unattributed time to 26.33 ms, versus 9.63 ms with four. These are medians of three alternating runs. On the eight ordinary requests with two seconds between processes, the four-FFN variant reached 477.00 tok/s at 104.56 MiB, versus 518.46 at 197.82 MiB for resident. The five-FFN variant reached 449.97 tok/s at 105.92 MiB. All 201 output IDs matched in both passes.

The streaming and shared-team variants remain local experiments. The four-FFN variant met the short RSS ceiling and retained a decode penalty on both the fixed fixture and the ordinary requests.

```sh
p=.model-cache/rss-team
python3 "$p/build.py"
python3 "$p/build-mailbox.py"
python3 "$p/build-gqa.py"
python3 "$p/build-balanced.py"
python3 "$p/build-idle.py"
python3 "$p/build-dual.py"
python3 "$p/build-compact-queue.py"
python3 "$p/build-serial-rope.py"
bash "$p/build-profile.sh"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench.py 256 12 7 base forward4-bs forward4-ms
MOTY_BENCH_COOLDOWN=12 python3 .model-cache/rss-direct/bench.py 3584 12 2 base forward7-b forward7-mg
MOTY_BENCH_COOLDOWN=12 python3 .model-cache/rss-direct/bench.py 3584 12 3 base forward7-m
MOTY_BENCH_COOLDOWN=12 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  base forward7-mg forward7-mga forward7-mgc
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench.py 256 12 2 \
  base forward4-ms forward4-msd forward4-mse forward4-msf forward4-msi forward4-msj forward4-msk
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench.py 256 12 7 base forward4-ms forward4-msk
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench.py 256 12 3 base forward4-ms forward4-msu forward4-msv
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench.py 256 12 3 base forward4-msk forward5-mskz
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 base forward4-msk forward5-msk
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 7 \
  base forward4-msk forward4-mskr forward5-mskzr
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench-natural.py 2 base forward4-mskr forward5-mskzr
python3 .model-cache/rss-direct/verify.py forward4-ms
python3 .model-cache/rss-direct/verify.py forward4-msk
python3 .model-cache/rss-direct/verify.py forward7-mg
python3 .model-cache/rss-direct/verify.py forward7-m
python3 .model-cache/rss-direct/verify.py forward4-mskr
python3 .model-cache/rss-direct/verify.py forward5-mskzr
python3 "$p/check-kv8.py"
python3 "$p/profile.py" base forward4-mskr forward5-mskzr
```

### Split queue and down readiness

The split-queue profile now measures the foreground waits for gate/up and down independently. With four resident FFNs, three gate/up slots and two down slots, the 256-input fixture spent 16.96 ms waiting for down across 128 decoded tokens. It spent 1.45 ms waiting for gate/up and 4.04 ms publishing future gate/up reads. Reassigning one gate/up slot to the down queue with a layer-based priority reduced down waits to 1.18 ms while gate/up waits grew to 12.55 ms and publication reached 7.18 ms. Whole-model decode took 291.0 and 304.4 ms, respectively. Figures are medians of three alternating instrumented runs.

In the corresponding three-run throughput screen, the resident control reached 492.51 tok/s. The split FIFO reached 464.11 tok/s at 109.44 MiB. Ordering reads by layer reached 461.31 tok/s at 109.45 MiB, and advancing down reads further reached 442.88 tok/s. With two gate/up slots and three or four down slots, results were 436.61 and 438.24 tok/s. Skipping the wait mutex when its readiness counter showed a complete matrix reached 461.92 tok/s. All paths generated identical IDs.

The progressive variant lets each compute worker begin when its output-row range is ready, sharing the existing FP16 activation preparation. With 768 KiB stripes, it used that path on about 1,734 down calls in the instrumented fixture. Time attributed to down, including waits in the compute workers, rose from 69.32 to 102.00 ms. The separate throughput screen saw the resident control fall to 362.14 tok/s, so it provides token and RSS checks only. The progressive path matched 457 generated IDs, 2310 teacher-forced top-1 IDs and perplexity 119.9085.

A second reader prototype claims stripes from an atomic FIFO cursor and parks idle readers on the existing condition variable. Under stable controls, the locked FIFO, atomic cursor and atomic cursor with the ready-counter fast path delivered 464.44, 455.35 and 463.49 tok/s respectively. The resident control reached 493.22 tok/s. All three paired comparisons passed the short RSS ceiling and returned identical IDs. The atomic cursor also matched all 457 generated IDs and 2310 teacher-forced top-1 IDs, with perplexity 119.9085.

Publishing reads and signaling completion only when a thread is waiting reached 459.60 tok/s at 108.62 MiB versus 457.63 tok/s for the locked FIFO and 489.33 tok/s for resident. It matched the same 457 generated IDs and 2310 top-1 IDs with perplexity 119.9085.

With four resident FFNs and the conditional-wake scheduler, retaining three slots for each family reached 472.11 tok/s at 110.69 MiB. Peak RSS stayed at or below 111.00 MiB in three alternating pairs. The resident control reached 486.54 tok/s and the original three-gate/two-down queue reached 463.72 tok/s. An instrumented repeat measured 0.97 ms of foreground down waits and 12.75 ms of gate/up waits with three slots for each family. It preserved all generated IDs.

Using the same weight budget for separate resident prefixes, two gate/up pairs plus eight down matrices reached 472.29 tok/s at 109.45 MiB. The same binary holding four resident matrices in each family reached 466.32 tok/s at 109.72 MiB. The resident control reached 489.27 tok/s. All results are three-run medians and match the 128 decode IDs.

Sampling the atomic claim cursor at the foreground wait boundary found unclaimed stripes on four of 3,136 down waits with two down slots. In the other 3,132 waits, readers had already claimed every stripe and at least one read was still in flight. With a third down slot, down waits fell to 91 across three requests. The remaining stalls shifted to gate/up, where readers had already claimed every stripe in 1,924 of 1,925 waits. Moving the next gate/up publication behind current down readiness reached 463.72 tok/s with two down slots and 473.72 tok/s with three. The resident control reached 492.05 tok/s in that three-run screen.

Keeping gate/up stripes at 768 KiB and changing only down to 256, 512 or 1280 KiB gave 397.04, 434.09 and 432.90 tok/s. The 768 KiB down control reached 463.73 tok/s. The median resident control was 496.17 tok/s. These three alternating passes preserved every generated ID.

Seven alternating pairs with the resident control narrowed the short-context speed gap. Separate resident prefixes with three slots for each family met the 111.25 MiB ceiling on every run.

| 256 inputs / 12 workers | PP (tok/s) | TPS (tok/s) | Paired TPS change | Median peak RSS (MiB) | Highest RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Resident control | 1354.5 | 494.68 | | 202.22 | |
| Four resident FFNs, three slots per family | 1336.9 | 476.11 | −3.95% | 110.97 | 111.67 |
| Two resident gate/up pairs, eight down matrices, two down slots | 1344.8 | 467.20 | −5.92% | 109.30 | 109.73 |
| Two resident gate/up pairs, eight down matrices, three down slots | 1342.7 | 471.79 | −4.55% | 110.03 | 110.69 |

The last row had paired changes of −0.87% PP and +2.66% TTFT. All 128 generated IDs matched across all runs. Across two alternating passes of the eight ordinary requests, the same configuration reached 487.61 tok/s at 105.75 MiB compared with 530.38 tok/s at 197.58 MiB for resident. It matched all 201 output IDs in each pass. The full suite matched 457 generated IDs, 2310 teacher-forced top-1 IDs and perplexity 119.9085, including the short fixture with `PREFILL_CHUNK=0/1/16`. The short-context throughput requirement keeps the queue variants experimental. Logs and binary digests are in `.model-cache/rss-down`.

```sh
p=.model-cache/rss-down
python3 "$p/build.py"
python3 "$p/build-progressive.py"
python3 "$p/build-dispatch.py"
python3 "$p/build-wake.py"
python3 "$p/build-family.py"
python3 "$p/build-demand-profile.py"
python3 "$p/build-defer.py"
python3 "$p/build-stripes.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base down4-10-0 down4-10-1 down4-10-3 down4-01-1 down4-02-1 down4-10-0f
python3 "$p/profile.py" base down4-10-0 down4-01-1 down4-10-0p down4-10-0l
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base down4-10-0 down4-10-0l down4-10-0fl
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base down4-10-0 down4-11-0 down4-11-0w
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base downfam4-4-10 downfam3-6-10 downfam2-8-10 downfam5-2-10
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base down4-10-0w down4-10-0y down4-11-0y
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base down4-10-0w downstripe10-256 downstripe10-512 downstripe10-1280
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 7 \
  base down4-11-0w downfam2-8-10 downfam2-8-11
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench-natural.py 2 base downfam2-8-11
python3 "$p/demand.py"
python3 .model-cache/rss-direct/verify.py down4-10-0l
python3 .model-cache/rss-direct/verify.py down4-10-0p
python3 .model-cache/rss-direct/verify.py down4-10-0w
python3 .model-cache/rss-direct/verify.py downfam2-8-11
```

### Reassigning memory after prefill

A short reader sweep kept the three-slot queues and the two-gate/eight-down resident allocation. With three, four, five and six readers, decode reached 417.99, 455.11, 466.67 and 457.72 tok/s. Five readers at user-initiated and utility QoS reached 460.86 and 435.42 tok/s. The resident control reached 481.12 tok/s. These are three-run medians with identical generated IDs. The default-priority five-reader configuration reached an individual 111.36 MiB peak, which exceeds the short ceiling.

The phase prototype releases the kernel scratch from the most recent multi-token pass just before single-token decode. It can then load additional packed FFN panels into the resident set. On return to a multi-token pass, it releases those temporary panels and restores the original queue. The phase probe measured 1.56 MiB reclaimed from batch scratch at the 256-token transition. Loading one more gate/up pair and one down matrix reached 491.09 tok/s at 111.91 MiB, versus 491.26 tok/s for resident in the three-run screen.

Separate anonymous mappings for raw prefill and packed decode buffers reached 493.51 tok/s at a median of 111.05 MiB with the same resident panels. One timed run reached 113.12 MiB. Reusing the prefix of each raw slot and unmapping its page-aligned tail reclaimed another 0.81 MiB. In the instrumented transition, the prefill high-water mark was 107.17 MiB, and the final peak was 111.09 MiB. Moving one or two more initial resident down matrices into the decode-only set reduced the prefill high-water mark to 105.94 or 104.72 MiB respectively, with final observed peaks of 110.80 and 111.14 MiB. The probes matched all 128 fixture IDs and all 32 IDs across two interactive turns in the same process.

In the three-run prefix-reuse screen, the resident control fell from 492.49 to 421.37 tok/s between consecutive pairs. The timing series is excluded. The three versions with the same final allocation of three gate/up pairs and nine down matrices peaked at 111.14, 111.14 and 111.11 MiB. All generated IDs matched.

In the next three alternating runs, the prefix-reuse configuration reached 498.88 tok/s versus 492.01 for resident and recorded one 112.12 MiB peak. Reducing the gate/up queue to two slots made room for more resident gate/up panels. The configuration that fit every screen peaked at 109.91 MiB in seven follow-up runs, with 449.68 tok/s versus 496.75 for resident. It returned the same 128 IDs and was dropped from speed acceptance.

Packing the attention output projection losslessly on the last 20 or all 28 layers made room while retaining three queue slots per family. The 20-layer configuration reached 494.83 tok/s at 110.62 MiB. The 28-layer configuration reached 501.39 tok/s at 110.38 MiB. Its three timed peaks were 110.36, 110.38 and 110.39 MiB. The resident control reached 497.75 tok/s. The 28-layer configuration's PP was 1329.5 versus 1351.8 tok/s. All 128 generated IDs matched in each run.

In seven alternating pairs, the 28-layer configuration had median paired changes of −0.07% TPS and −1.88% PP. One run reached 111.70 MiB. Prefill tiles of 16 and 32 gave −8.99% and −18.71% paired PP, respectively. Packing Q on additional layers provided more room. It also exposed an imbalance in mixed packed/raw QKV scheduling, where one packed job computes four rows and one raw job computes one. Dispatching both layouts in groups of four rows gave 1316.5 PP and 500.20 TPS at 109.22 MiB with Q packed on all 28 layers. The corresponding reference reached 1347.0 PP and 498.15 TPS. This was a three-run screen with identical generated IDs.

A further prototype uses one temporary buffer to read the original Q4_1 query and output projections during multi-token prefill. Decode uses the resident packed panels. Packing both Q and O on all 28 layers with this transient raw path reached 1330.7 PP and 490.40 TPS at 109.50 MiB. The resident control reached 1372.9 PP and 499.43 TPS. The packed-Q16 balanced path in the same three-run screen reached 1329.7 PP and 500.18 TPS at 109.64 MiB. The comparison uses identical generated IDs.

Seven follow-up pairs with the raw prefill path reached 1338.1 PP and 494.08 TPS at 109.23 MiB. The resident control reached 1370.0 PP and 498.64 TPS. Median paired changes were −2.12% PP, −0.94% TPS, +6.42% TTFT and +3.64% request time. The highest observed RSS was 109.52 MiB. Every run returned the same 128 IDs.

On eight ordinary requests, the same configuration reached 510.60 tok/s at 107.17 MiB compared with 545.47 tok/s at 197.82 MiB for the resident control. Both passes matched all 201 generated IDs. The full quality suite matched 457 generated IDs and 2310 teacher-forced top-1 IDs. Perplexity was 119.9085 on both paths. The short fixture also matched with `PREFILL_CHUNK=0/1/16`.

With four gate/up pairs and twelve down matrices resident at prefill, the same phase method ran the 3584-input fixture at 1029.5 PP and 339.89 TPS. In two alternating pairs the resident control reached 1057.4 PP and 337.24 TPS. Median paired changes were −2.64% PP, +0.79% TPS, +3.00% TTFT and +2.62% request time. Peak RSS was 212.83 MiB in the median and 212.95 MiB at most. All 128 generated IDs matched.

At a fresh GGUF path, the short prototype built 69 resident panel cache entries on first open. TTFT was 335.06 ms on first open and 239.35 ms on reopening, compared with 205.08 ms for resident at that path. The corresponding peaks were 109.27, 109.22 and 204.22 MiB, and all generated IDs matched. This test used the existing cold-FFN transport file. The phase variant remains experimental while prefill, ordinary decode and first-open latency trail the resident control.

```sh
p=.model-cache/rss-phase
python3 "$p/build.py"
python3 "$p/build-banks.py"
python3 "$p/build-reuse.py"
python3 "$p/build-peak.py"
python3 "$p/build-attn.py"
python3 "$p/build-weighted.py"
python3 "$p/build-raw-prefill.py"
MOTY_PHASE_PROFILE_BINARY=peak-profile python3 "$p/probe.py" \
  downfam2-8-11-x11tb downfam2-8-11-x11tr downfam2-7-11-x12tr downfam2-6-11-x13tr
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base downfam2-8-11-x10tr downfam2-8-11-x11tr-o20 downfam2-8-11-x11tr-o28
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base downfam2-8-11-x11tr-o28-q16 downfam2-8-11-x11tr-o28-q16w downfam2-8-11-x11tr-o28-q28w
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base downfam2-8-11-x11tr-o28-q16w downfam2-8-11-x11tr-o28-q16-raw downfam2-8-11-x11tr-o28-q28-raw
MOTY_BENCH_COOLDOWN=4 python3 .model-cache/rss-direct/bench.py 256 12 7 \
  base downfam2-8-11-x11tr-o28-q28-raw
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base downfam2-8-11-x11tr-o28-q28-raw
python3 .model-cache/rss-direct/verify.py downfam2-8-11-x11tr-o28-q28-raw
MOTY_BENCH_COOLDOWN=13 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  base downfam4-12-11-x11tr-o28-q28-raw
python3 "$p/cold.py"
```

### Overlapping attention I/O and building panel caches in batches

The prefill prototype reads the contiguous raw O and Q slice for a layer through one or two extra slots in the existing reader pool. The slots are unmapped on entry to decode. On the 256-token screen, synchronous reads, one slot and two slots reached 1340.6, 1346.9 and 1342.0 PP respectively. All 128 generated IDs matched. The screen did not establish a prefill gain from the second slot.

The readers can validate Group64 pairs in the first prefetched raw FFN buffer and publish readiness once the validation is complete. This removes a separate read of every cold FFN at model startup. With two attention slots, median TTFT changed from 241.50 to 230.67 ms in three alternating pairs. The resident control reached 226.43 ms. PP was 1340.8 versus 1341.0 tok/s for the two streaming variants, and TPS was 492.37 versus 495.18. The deferred variant's highest RSS in this screen was 110.97 MiB. The runs preserved every generated ID.

In seven pairs, the deferred two-slot variant recorded median paired changes of +1.24% TTFT, +1.13% request time, −1.42% PP and −1.44% TPS against resident. One run reached 112.14 MiB and exceeded the 111.25 MiB ceiling. All 457 generated IDs and 2310 teacher-forced top-1 IDs matched in the full quality suite. Perplexity was 119.9085, with the short fixture also matching under `PREFILL_CHUNK=0/1/16`.

On 3584 inputs, the same method with the larger resident prefix reached 1035.2 PP and 334.56 TPS. The resident control reached 1044.55 PP and 335.69 TPS. The two paired changes were −0.89% PP, −0.33% TPS, +1.04% TTFT and +0.97% request time. Highest RSS was 214.47 MiB, and all 128 generated IDs matched in both pairs.

On eight ordinary requests, adding one resident gate/up pair at the prefill-to-decode transition reached 515.21 tok/s at 109.52 MiB, compared with 503.90 tok/s at 107.16 MiB for the existing allocation and 550.81 tok/s for resident. Another gate/up pair raised peak RSS above the 111.25 MiB ceiling in both passes. Each pass matched all 201 output IDs.

With deferred validation and two slots, the same additional gate/up pair reached 506.80 tok/s at 109.59 MiB on the ordinary requests. The resident and existing streaming allocations reached 535.01 and 502.60 tok/s respectively in that two-pass test. All output IDs matched.

The production Q4_1 cache builder now reads and writes batches of panels in one bounded buffer. The cache header, checksum and payload layout are the same. In five alternating pairs with both executables warmed and fresh GGUF aliases, median first-open TTFT changed from 335.90 to 272.31 ms. The median paired reduction was 18.68%. Reopening changed from 243.33 to 241.90 ms. The highest first-open RSS was 110.19 MiB for the batched builder. All 69 cache files matched the old builder byte for byte in every pair, and all runs matched the 128 generated IDs. The test used the same pre-existing cold FFN transport file on both paths.

Keeping synchronous attention reads, deferred validation and balanced QKV decode met the short RSS ceiling in all seven pairs. The highest observed RSS was 109.48 MiB. Median paired changes against resident were −2.32% PP, −1.01% TPS, +1.65% TTFT and +0.28% request time. Without QKV balancing, PP and TPS changed by −3.13% and −3.15%, with a maximum of 109.84 MiB. Every run in both variants matched all 128 generated IDs. The balanced variant also matched the full suite of 457 generated IDs and 2310 top-1 IDs, with perplexity 119.9085.

The native suite completed with 175 passes and four platform skips. The current ARMv8.0 build passed the Q4_1 kernel suite under `qemu-aarch64 -cpu cortex-a53`. Logs and the provisional loader overlays are in `.model-cache/rss-overlap`.

```sh
p=.model-cache/rss-overlap
python3 "$p/build.py"
python3 "$p/build-defer.py"
bash "$p/build-batched.sh"
python3 "$p/cold.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base downfam2-8-11-x11tr-o28-q28-raw-a1 downfam2-8-11-x11tr-o28-q28-raw-a2 \
  downfam2-8-11-x11tr-o28-q28-raw-a1-verifyio downfam2-8-11-x11tr-o28-q28-raw-a2-verifyio
python3 .model-cache/rss-direct/verify.py downfam2-8-11-x11tr-o28-q28-raw-a2-verifyio
MOTY_BENCH_COOLDOWN=13 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  base downfam4-12-11-x11tr-o28-q28-raw-a2-verifyio
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 7 \
  base downfam2-8-11-x11tr-o28-q28-raw-a0-verifyio downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio
python3 .model-cache/rss-direct/verify.py downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio
```

### Using the remaining decode memory

The phase prototype promotes selected FFN panels from the stream to resident storage after prefill. With twelve threads and the same eight ordinary requests, allocating more of the remaining budget to `down` kept peak RSS below 111.25 MiB. The fastest ordinary-request allocation in this two-pass screen added one gate/up pair.

| Panels promoted after prefill | Aggregate decode tok/s | Median of the per-pass maximum RSS (MiB) |
|---|---:|---:|
| Resident control | 542.36 | 197.82 |
| One gate/up pair, one down | 494.78 | 107.18 |
| Two gate/up pairs, one down | 504.20 | 109.50 |
| Two gate/up pairs, two down | 496.38 | 110.78 |
| One gate/up pair, three down | 503.78 | 109.66 |

Each pass matched all 201 generated IDs. Prefetching panels for the following token into released buffer slots during `lm_head` or earlier in the layer stack also preserved output IDs. In three alternating short-fixture passes the same allocation with prefetch disabled reached 493.52 TPS. Prefetching two gate/up pairs during `lm_head` reached 486.29 TPS, starting these reads earlier reached 487.71 TPS, and starting three reads for each family earlier reached 483.86 TPS. All four paths stayed below 110 MiB in this screen. Using NEON widening shifts to unpack Q4_1 nibbles changed short-fixture decode from 493.13 to 491.53 TPS in a separate three-pass screen. These variants remain in the local overlays.

An exact NEON greedy reduction changed the profiled sampling cost from 8.6 to 5.1 µs/token. On the eight ordinary requests, the existing picker, parallel NEON picker and serial NEON picker reached 495.81, 494.82 and 500.32 TPS respectively. The resident control reached 543.09 TPS. RSS stayed below 108 MiB for the three streaming runs in each pass and all 201 IDs matched. The sampling stage accounts for less than 0.5% of decode time, so it cannot close the throughput gap.

Queue profiling over two passes of the ordinary requests measured 1982.1 µs/token for the one-gate/one-down promotion policy and 1836.6 µs/token for resident. Of the streaming time, 81.5 µs/token went to waiting for gate/up panels and 3.9 µs/token to waiting for down panels. Switching the prefill storage and promoting panels cost another 37.1 µs/token averaged over the 402 generated tokens. Publishing read requests cost 28.8 µs/token, including 18.9 µs/token for the rolling gate/up queue. These costs identify the gate/up producer and the phase transition as the priorities for ordinary requests.

```sh
p=.model-cache/rss-ordinary
b=downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b" downfam2-8-11-x21tr-o28-q28-raw-a0w-verifyio \
  downfam2-8-11-x22tr-o28-q28-raw-a0w-verifyio \
  downfam2-8-11-x13tr-o28-q28-raw-a0w-verifyio
python3 "$p/build.py"
python3 "$p/build-ring.py"
python3 "$p/build-shll.py"
python3 "$p/build-sampler.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-h0b" "$b-h2g" "$b-h2gr" "$b-h3br"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-h0b" "$b-h0bs"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-h0b" "$b-h0b-vmax" "$b-h0b-smax"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-h0b" "$b-h0b-vmax" "$b-h0b-smax"
python3 "$p/build-wait-profile.py"
MOTY_QUEUE_PROFILE_BINARY="$p/wait-profile" python3 "$p/profile-natural.py" base "$b-h0b"
```

### Balancing gate/up stripes and loading promoted panels in parallel

This prototype keeps the lossless panel bytes and decode memory layout. Only the gate/up reader splits its requests into nearly equal, 16 KiB-aligned stripes. Raw prefill and down continue to use the original stripe size. Mode `n0` retains the original gate/up reads.

| Gate/up read policy | 256-input decode tok/s | Highest RSS in three runs (MiB) |
|---|---:|---:|
| Resident control | 495.43 | 202.84 |
| Original 768 KiB stripes | 493.12 | 109.45 |
| Three balanced stripes | 500.45 | 109.47 |
| Four balanced stripes | 447.82 | 109.52 |
| Five balanced stripes | 390.85 | 109.47 |
| Six balanced stripes | 377.14 | 110.95 |

Three stripes reached 499.02 TPS in the repeat comparison, where the original method reached 499.91 TPS and resident reached 497.88 TPS. Two stripes reached 467.07 TPS in that comparison. Ordinary-request profiling measured 70.4 µs/token waiting for gate/up with the original split and 84.0 µs/token with three stripes, so the short-fixture gain did not generalize.

The second overlay allocates the same buffers for panels promoted after prefill, then uses the existing readers to fill them. Mode `ap1` waits for all promotions before entering the forward pass. Mode `ap2` waits for each promotion at first use. It drains outstanding reads before releasing promoted storage. The initial ordinary-request screen reached 496.93 TPS for the original split, 468.43 TPS for three stripes alone, 506.80 TPS for parallel promotion with a barrier and 507.16 TPS for promotion with first-use waits. The resident control reached 550.10 TPS. Each pass produced all 201 reference IDs. One run of three stripes alone had 4,890 involuntary context switches on a request where the control had 1,575.

The follow-up used the same executable for all four streaming modes, including `ap0` with parallel promotion disabled. With the original stripes, first-use waits changed aggregate decode from 478.55 to 494.97 TPS. With three stripes, they changed it from 496.52 to 498.95 TPS. The resident control reached 529.03 TPS. The highest streaming RSS was 107.34 MiB across both passes, and all 201 IDs matched each time.

With first-use waits, the 256-input repeat reached 498.10 TPS at a maximum of 111.86 MiB. That exceeded the 111.25 MiB ceiling in two of three runs even though the panel allocation count was unchanged. All warm-ups and measured runs matched the 128 reference IDs.

On the 3584-input fixture, the resident control reached 1028.45 PP and 319.17 TPS, while the first-use variant reached 1013.45 PP and 313.38 TPS. Its maximum of 217.75 MiB exceeded the 216.89 MiB ceiling in both measured passes. Every run matched all 128 output IDs.

Removing the promoted down matrix on the short fixture gave 472.96 TPS and a peak of 111.48 MiB in three first-use runs. The same allocation with synchronous promotion reached 471.77 TPS at a maximum of 109.50 MiB. All 128 IDs matched on every run. The first-use variant still fails the short memory gate, and the ordinary-request throughput gap to resident keeps both overlays experimental.

The first-use variant matched all 457 generated IDs and all 2310 teacher-forced top-1 IDs. Both paths measured perplexity 119.9085. It matched the short fixture with `PREFILL_CHUNK=0/1/16` and a separate two-turn test matched 32 IDs after switching back to prefill.

```sh
p=.model-cache/rss-gate
b=downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio-h0b
l=downfam4-12-11-x11tr-o28-q28-raw-a2-verifyio-h0b
lean=downfam2-8-11-x10tr-o28-q28-raw-a0w-verifyio-h0b
python3 "$p/build.py"
python3 "$p/build-promotion.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-n0" "$b-n3" "$b-n4" "$b-n5" "$b-n6"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-n0" "$b-n2" "$b-n3" "$b-n3-ap2"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-n0" "$b-n3" "$b-n3-ap1" "$b-n3-ap2"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-n0-ap0" "$b-n0-ap2" "$b-n3-ap0" "$b-n3-ap2"
MOTY_BENCH_COOLDOWN=10 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  base "$l-n0" "$l-n3-ap2"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-n3-ap0" "$lean-n3-ap0" "$lean-n3-ap2"
python3 .model-cache/rss-direct/verify.py "$b-n3-ap2"
MOTY_PHASE_PROFILE_BINARY="$PWD/$p/promotion-profile" python3 .model-cache/rss-phase/probe.py "$b-n3-ap2"
```

### Starting gate/up before the complete read arrives

A reader now publishes one completion bit per stripe in this overlay. A compute worker can start a row range when both its gate and up panels are ready. The worker waits for every stripe that overlaps the range, including panels that cross a stripe boundary. The path prepares FP16 activations once and calls the same packed projection and SwiGLU code for each range. Slot reuse still waits for completion of the whole read. The byte-range check covered stripe sizes from 64 to 4096 KiB, unequal worker ranges and crossing panels.

In a three-pass screen on the 256-token fixture, the full-read, partial-read spin and partial-read yield modes reached 470.19, 475.71 and 490.76 TPS. Resident reached 485.19 TPS. The yield mode peaked at 109.53 MiB, and all modes matched the 128 output IDs on every pass. The spin mode also had a 408.88 TPS pass.

The second overlay adds an alternative that permits a worker to claim any ready tile of up to 16 four-row panels. A tile is available once every overlapping stripe has published completion. An atomic claim gives one worker ownership of its outputs. The row arithmetic and prepared activation buffer are shared with the full-read path. Three alternating short-fixture runs measured 488.33 TPS for a full read, 465.27 TPS for fixed-range spin and 461.65 TPS for ready-tile claims. Combining ready-tile claims with three balanced stripes gave 434.52 TPS. Resident reached 483.62 TPS. All paths matched every output ID.

On eight ordinary requests, the full-read, fixed-range yield and ready-tile variants reached 497.97, 490.57 and 477.95 TPS respectively across two passes. Resident reached 536.18 TPS. Every pass matched all 201 output IDs, and the streaming variants peaked at or below 107.32 MiB. Neither partial-read strategy improved throughput on ordinary requests.

The operator profile shows where the wait went. The full-read control spent 82.9 µs/token in foreground gate/up readiness and 657.8 µs/token in the gate/up kernel. Fixed-range yield reduced the foreground timer to 0.6 µs/token, with 728.4 µs/token inside gate/up. Ready-tile claims raised the kernel timer to 868.0 µs/token. These modes dispatched with a partial read 1550 and 1950 times over 402 generated tokens. Including worker waits in the projection timer is essential when comparing the scheduling policies.

Ready-tile claims matched all 457 generated IDs and all 2310 teacher-forced top-1 IDs. Perplexity was 119.9085 for both paths, and the short fixture matched with `PREFILL_CHUNK=0/1/16`.

A separate overlay polls for the next request briefly after a decode read before parking. It leaves the prefill reader behavior unchanged. On 256 inputs, immediate parking reached 477.89 TPS. Polling for 4 µs with one reader, 12 µs with one reader, and 12 µs with two readers reached 470.89, 482.51 and 479.92 TPS. The peak for one reader at 12 µs was 109.47 MiB. All variants matched every generated ID in the three-pass comparison.

On the ordinary requests, polling with one reader at 12 µs reached 495.36 TPS, compared with 500.44 TPS with immediate parking and 538.64 TPS for resident. All 201 IDs matched in each pass. Polling and partial gate/up readiness remain local experiments.

```sh
p=.model-cache/rss-progress
b=downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio-h0b
python3 "$p/build.py"
python3 "$p/build-steal.py"
python3 "$p/build-idle.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-n0-pg0" "$b-n0-pg1" "$b-n0-pg2"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-n0-sg0" "$b-n0-sg1" "$b-n0-sg3" "$b-n3-sg3"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-n0-sg0" "$b-n0-pg2" "$b-n0-sg3"
python3 .model-cache/rss-ordinary/profile-natural.py base "$b-n0-sg0" "$b-n0-pg2" "$b-n0-sg3"
python3 .model-cache/rss-direct/verify.py "$b-n0-sg3"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-ri0x1" "$b-ri4x1" "$b-ri12x1" "$b-ri12x2"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-ri0x1" "$b-ri12x1"
```

### Dictionary transport for FFN scale and minimum

There are 34,547 distinct FP16 scale/minimum pairs across the 84 FFN matrices. About one third of all groups have a minimum equal to exactly −8 times the scale. The local transport packs each original pair as one 16-bit index into a 138,188-byte table. The Q4_1 codes stay byte-identical, and a round-trip check compared every parameter and code in all 84 matrices. Packed FFN payload decreased from 108,380,160 to 102,359,040 bytes, a reduction of 5.56% before the dictionary and index.

The single-token FMLAL kernel loads the original pair from the table and applies the same FP32 updates in the same order. The prefill path still uses the previous transport. A second layout moves the indices to the end of each four-row panel, keeping all Q4 code vectors aligned within the panel. Its round-trip check also covers all 84 matrices. The original GGUF is shared by all runs.

On the first three alternating 256-input passes, the same streaming executable reached 470.10 TPS using the original panels, 487.55 using indexed gate/up, 489.82 using indexed down and 495.17 using both. Resident reached 489.71 TPS. With both families indexed the maximum RSS was 109.31 MiB, and all 128 IDs matched in every comparison. The measured total read volume was 10,282.00 MiB for all indexed FFNs versus 10,881.04 MiB for the original panels.

On eight ordinary requests, indexed interleaved FFN panels reached 514.89 TPS compared with 497.46 TPS using the original panels in the same executable and 535.62 TPS for resident. The indexed passes matched all 201 generated IDs and peaked at 107.00 MiB. The aligned-code layout reached 491.74 TPS on the short fixture and 486.64 TPS on the ordinary requests. Its matched control reached 492.32 and 489.12 TPS respectively. All generated IDs matched across both layouts.

On 3584 inputs, two alternating passes gave 331.28 TPS for resident, 335.15 TPS for the original streaming panels, 328.70 TPS for interleaved indices and 326.72 TPS for aligned codes. The indexed variants peaked at 214.52 and 214.16 MiB, respectively. Interleaved indices reached 1028.9 PP, versus 1044.5 PP for resident. Every run returned the same 128 IDs.

Disassembly of the first indexed kernel showed a 6 KiB stack frame with a Darwin stack probe on every call because the FP32 fallback kept a staging panel on its stack. Outlining the FP32 fallback and specializing the indexed FP16 loop removed the stack probe from the FP16 path and reduced its frame to 48 bytes. The short fixture reached 498.02 TPS with the specialization, 479.95 TPS with the first indexed kernel and 474.69 TPS with the original panels in the specialized executable. Resident reached 483.99 TPS in the same three-pass screen. The highest RSS of the specialized indexed variant was 109.42 MiB.

On the ordinary requests, the specialization reached 508.07 TPS versus 486.05 TPS for the original panels in the same executable and 534.41 TPS for resident. The specialized indexed path peaked at 107.05 MiB and matched all 201 IDs in both passes. The interleaved, aligned-code and specialized paths each matched all 457 generated IDs, all 2310 teacher-forced top-1 IDs and perplexity 119.9085. Their short fixtures also matched with `PREFILL_CHUNK=0/1/16`.

The specialized path reached 1002.15 PP and 317.63 TPS on 3584 inputs in two alternating pairs. Resident reached 1022.45 PP and 328.30 TPS. The highest peak was 214.48 MiB, and all 128 output IDs matched. A separate `Q41_ACT=f32` comparison matched all 128 short-fixture IDs for each indexed variant against the original panels in the same executable. The dictionary path stays experimental while prefill, long decode and ordinary decode trail resident. The performance comparisons used prebuilt sidecars.

```sh
p=.model-cache/rss-scale
b=downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio-h0b
l=downfam4-12-11-x11tr-o28-q28-raw-a2-verifyio-h0b
python3 "$p/probe.py"
python3 "$p/pack.py"
python3 "$p/build.py"
python3 "$p/build-planar.py"
python3 "$p/build-fast.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-dict0" "$b-dict1" "$b-dict2" "$b-dict3"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-plane0" "$b-dict3" "$b-plane3"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-dict0" "$b-dict3" "$b-plane0" "$b-plane3"
MOTY_BENCH_COOLDOWN=8 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  base "$l-dict0" "$l-dict3" "$l-plane3"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-fast0" "$b-dict3" "$b-fast3"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-fast0" "$b-fast3"
MOTY_BENCH_COOLDOWN=7 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  base "$l-fast3"
for mode in dict3 plane3 fast3; do
  python3 .model-cache/rss-direct/verify.py "$b-$mode"
done
python3 "$p/verify-f32.py"
```

### Placing indexed panels within the decode budget

On the 256-input fixture, the indexed transport with one gate/up pair and one down matrix promoted after prefill reached 474.95 TPS. Two gate/up pairs and one down reached 481.33 TPS and stayed at or below 111.23 MiB across three measured runs. On the eight ordinary requests, that larger allocation reached 480.26 TPS compared with 498.87 TPS for the one-pair allocation. Two pairs without the promoted down matrix reached 500.97 TPS at a maximum of 108.19 MiB. The resident control varied from 485.47 to 527.21 TPS across the three ordinary-request passes. All output IDs matched.

The compact-bank prototype sizes the reusable prefill prefix and any additional decode slots from the actual indexed range size for each family. On the short fixture with two promoted gate/up pairs and one down, the observed maximum declined from 111.27 to 110.88 MiB. Decode changed from 483.60 to 478.18 TPS. Promoting three gate/up pairs and no down still reached 111.97 MiB. All fixed runs kept the same 128 IDs. On ordinary requests with the two-pair allocation, the highest RSS with original and compact banks was 109.34 and 108.97 MiB respectively.

Releasing the one-shot tokenizer hash tables through the general allocator after encoding produced a peak of 108.73 MiB in the compact-bank two-pair ordinary screen, versus 108.97 MiB with the tables retained. The overlay keeps the vocabulary strings, added tokens and decoding tables available throughout generation. The mapped variant gives the two encoding hash tables private mappings so they can be explicitly unmapped after encoding.

With the same two-pair placement, the mapped trial decreased maximum RSS from 108.97 to 104.94 MiB and reached 501.10 TPS versus 493.78 TPS for the retained-table comparison. With a third promoted gate/up pair and one down matrix, it reached 505.11 TPS at a maximum of 107.22 MiB. Resident reached 529.94 TPS in those two ordinary-request passes. Both mapped variants matched all 201 output IDs. A separate check matched the complete UTF-8 output on all eight prompts and 32 IDs across two interactive turns.

The native GGUF one-shot `PROMPT` path now releases its encoding hash tables before generation. On Darwin, large one-shot tables use private mappings so the release removes their resident pages. Across five alternating ordinary-request passes with the resident native binary, the median of the per-pass peak RSS dropped from 197.48 to 193.66 MiB. The median paired changes were +0.09% decode throughput, −0.81% prefill throughput and +1.02% time to first token. All 201 IDs and output bytes matched on every pass. The interactive path retains its encoding tables, and token-ID input continues to skip their construction. The full Apple run matched 457 generated IDs, 2310 top-1 IDs and perplexity 119.9085. The native suite passed 175 tests with four skips. The Q4_1 suite and an eight-output one-shot comparison ran under Cortex-A53 QEMU.

On 3584 inputs, compact banks with one or two promoted gate/up pairs reached 328.10 and 323.58 TPS, with peaks of 214.47 and 214.27 MiB respectively. The resident control reached 334.37 TPS and 1038.85 PP. The compact variants reached 1016.7 and 1014.55 PP, with all 128 output IDs matching in both passes.

Another overlay precomputes the dictionary's FP32 scale and centered bias using the same fused multiply-add as the kernel. The FP16 path can read both coefficients directly, at the cost of a larger dictionary. In the identical one-pair placement, short decode moved from 487.21 to 481.23 TPS and ordinary decode from 499.43 to 488.22 TPS. The expanded table therefore remains disabled. The variant with two promoted gate/up pairs matched all 457 generated IDs, 2310 teacher-forced top-1 IDs and perplexity 119.9085, including the short runs with `PREFILL_CHUNK=0/1/16`.

```sh
p=.model-cache/rss-scale
s=tr-o28-q28-raw-a0w-verifyio-h0b
a=downfam2-8-11-x11$s
b=downfam2-8-11-x20$s
c=downfam2-8-11-x21$s
d=downfam2-8-11-x12$s
e=downfam2-8-11-x30$s
f=downfam2-8-11-x31$s
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 3 \
  base "$a-dict3" "$b-dict3" "$c-dict3" "$d-dict3"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$a-dict3" "$b-dict3" "$c-dict3" "$d-dict3"
python3 "$p/build-banks.py"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$c-bank0" "$c-bank1" "$c-bank2" "$e-bank2" "$f-bank2"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$c-bank0" "$c-bank1" "$e-bank1"
python3 "$p/build-expand.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$a-bank1-cp0" "$a-bank1-cp1" "$b-bank1-cp1"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$a-bank2-cp0" "$a-bank2-cp1" "$b-bank2-cp1"
python3 .model-cache/rss-direct/verify.py "$b-bank2-cp1"
python3 "$p/build-token-map.py"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$c-tokmap1" "$c-tokmap2" "$f-tokmap2"
python3 .model-cache/rss-direct/verify.py "$f-tokmap2"
python3 "$p/verify-mapped.py"
l=tr-o28-q28-raw-a2-verifyio-h0b-bank1
MOTY_BENCH_COOLDOWN=7 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  base "downfam4-12-11-x11$l" "downfam4-12-11-x21$l"
bash "$p/build-native-encoder.sh"
python3 "$p/bench-encoder.py"
python3 .model-cache/rss-direct/verify.py encoder-native
```

### Borrowing tokenizer strings from GGUF metadata

For one-shot text, GGUF metadata remains available until the encoder is released. The native path now indexes the original merge strings using their on-disk separator. Byte-level BPE maps input spaces to a Unicode symbol, so the ASCII space in a merge key is reserved for the separator. The change avoids copying merge keys into the tokenizer-owned string pool. The reusable tokenizer continues to own its keys.

For Pocket, the copied pool falls from 1,177,708 to 589,374 bytes with borrowed merges. The combined prototype keeps 291 bytes of copied control strings and a separate array of token lengths. A direct tokenizer comparison matched 2333 encoded IDs across 129 text inputs and the decoded bytes of every one of the 65,536 vocabulary entries after releasing the encoder. Both memory-mapped and copied GGUF headers passed the comparison.

A separate prototype also borrows non-control token strings, storing their explicit decode lengths. This layout was tested separately from borrowed merge keys. On eight ordinary requests over three alternating passes, the control, merge-only and combined modes reached 545.83, 546.26 and 538.03 TPS, at median per-pass peaks of 195.22, 193.16 and 192.61 MiB. All 201 generated IDs matched. The follow-up includes measurements of time to first token and byte-for-byte output.

With three promoted gate/up pairs and one down matrix, the compact streaming control reached 520.73 TPS and the prototype borrowing both sets of strings reached 518.71 TPS. Its highest peak dropped from 107.30 to 106.67 MiB. Using the space for another down matrix reached 515.61 TPS at 108.17 MiB, and promoting four gate/up pairs without a down matrix reached 513.32 TPS at 107.27 MiB. The resident control reached 532.01 TPS. Each pass matched all 201 IDs. These placements have no measured ordinary-request throughput advantage over the compact streaming control.

The combined prototype matched all 457 generated IDs and 2310 teacher-forced top-1 IDs at perplexity 119.9085. The fixed-ID timing screen included a control run below half of its usual decode throughput. It establishes ID equivalence for the fixture, and the production comparison uses the ordinary text workload in alternating pairs.

The merge-only production change used five alternating ordinary-request pairs. Median per-pass peaks were 194.45 and 194.17 MiB before and after the change. Paired medians changed by −0.35% for decode throughput, −3.20% for prefill throughput and −0.35% for time to first token. All 201 generated IDs, the input token counts and the full output bytes matched on each pass. The native suite passed 175 tests with four skips. ARMv8.0 passed the Q4_1 kernel checks and matched the earlier one-shot output and eight IDs under Cortex-A53 QEMU.

With explicit IDs on the 256-token fixture, three alternating native pairs measured 1371.0 versus 1350.6 PP and 497.03 versus 494.72 TPS before and after the change. The 3584-token comparison preserved all 128 IDs, but both binaries ran far below their earlier sustained throughput. The 3584-token series provides a quality check and is excluded from speed comparisons. The full native quality run matched all 457 generated IDs and all 2310 top-1 IDs at perplexity 119.9085.

```sh
p=.model-cache/rss-tok
python3 "$p/build.py"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 3 toknew0 toknew1 toknew2
python3 "$p/bench-native.py"
python3 .model-cache/rss-direct/verify.py toknew2
clang -O2 -I"$p/include" -Ic "$p/check-tok.c" c/libmoty-runtime.a c/libmoty-nn.a c/libmoty-hw.a \
  -lm -L/opt/homebrew/opt/libomp/lib -lomp -o "$p/check-tok"
for mapped in 1 0; do
  "$p/check-tok" .model-cache/MNN-Pocket-0.3B-Instruct-Q4_1.gguf tools/ref/ppl_text_long.txt "$mapped"
done
b=downfam2-8-11-x31tr-o28-q28-raw-a0w-verifyio-h0b-bank1
c=downfam2-8-11-x32tr-o28-q28-raw-a0w-verifyio-h0b-bank1
d=downfam2-8-11-x40tr-o28-q28-raw-a0w-verifyio-h0b-bank1
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-borrow0" "$b-borrow2" "$c-borrow2" "$d-borrow2"
bash "$p/build-production.sh"
python3 "$p/bench-production.py"
MOTY_BENCH_COOLDOWN=3 python3 .model-cache/rss-direct/bench.py 256 12 3 tokprod-before tokprod-after
MOTY_BENCH_COOLDOWN=6 python3 .model-cache/rss-direct/bench.py 3584 12 3 tokprod-before tokprod-after
python3 .model-cache/rss-direct/verify.py tokprod-after
```

### Mapping residency and GGUF header release

At the first generated byte on the 256-input fixture, `vmmap -resident` reported the following file-backed resident bytes. The per-region snapshots separate clean file mappings from the process's peak RSS. Each snapshot pauses generation and is used for memory attribution.

| Mode | Original GGUF (MiB) | Packed sidecars (MiB) |
|---|---:|---:|
| Native raw | 183.00 | 0 |
| Native packed FFN and tied head | 33.40 | 135.19 |
| Compact streaming | 7.03 | 70.81 |
| Compact streaming with `MADV_RANDOM` | 7.03 | 70.81 |

SHA-256 comparison found no duplicate or zero-filled blocks among 11,563 full 16 KiB blocks starting at the original tensor payload and 8,379 full blocks starting at the packed cache payload. Independent Deflate compression retained 89.46% and 95.67% of the respective input sizes. This experiment estimates the potential of payload block sharing and local compression.

The advisory comparison reached 478.77 TPS with random access advice and 480.49 TPS with the original advice on the 256-input fixture. On eight ordinary requests, the modes reached 516.56 and 523.18 TPS. Their median per-pass peaks were both 102.02 MiB. All 128 fixed IDs and 201 ordinary IDs matched.

The GGUF header occupies the first 2,441,216 complete bytes of the mapping on this host, before the page containing the earliest tensor. One-shot generation finishes reading the header after tokenization and cache setup. A prototype released that prefix and cleared the metadata index before entering generation. A second prototype replaces the prefix with an inaccessible anonymous reservation so the complete address range stays owned until the shard mapping is closed.

On the short fixture, the reservation trial changed median native RSS from 202.05 to 199.77 MiB. With indexed streaming it changed median RSS from 109.20 to 106.81 MiB, and decode from 463.61 to 460.04 TPS. A larger allocation with three gate/up pairs and one down stayed within the short RSS ceiling at a maximum of 110.88 MiB. On eight ordinary requests, the smaller and larger streaming allocations with the reservation reached 516.86 and 522.33 TPS, while the resident control reached 543.81 TPS. Both allocations matched all output IDs. The streaming decode speed gap remains.

The native Darwin CLI now releases this mapped prefix for Pocket QAT Q4_1 on explicit-ID inputs. The loader leaves the page containing the first tensor available and preserves the full address reservation for shard close. The initial short fixture measured 202.81 → 199.75 MiB, 1350.5 → 1364.6 PP and 492.24 → 490.48 TPS across three alternating pairs. On 3584 explicit IDs, two alternating pairs measured 295.65 → 294.15 MiB, 1046.25 → 1047.0 PP and 336.56 → 340.56 TPS. All 128 generated IDs matched on both fixtures.

The text-input trial increased median native peak RSS from 193.19 to 194.48 MiB in two passes. Production enablement is scoped to explicit IDs. In a follow-up after scoping, three alternating short-fixture pairs measured 202.36 → 199.77 MiB, 1353.8 → 1351.6 PP and 488.39 → 495.33 TPS. The final text-input comparison gave 193.50 → 194.45 MiB and 550.96 → 543.84 TPS in two passes. The header-release branch is inactive in those text-input runs. Interactive and copied-file validation checks cover the other input paths.

The final native code matched all 457 output IDs, all 2310 top-1 IDs and perplexity 119.9085. The Darwin fixture exercises releasing a full metadata page, reading the first tensor through the same pointer and closing the reserved range. The native suite passed 175 tests with four skips. Cortex-A53 QEMU passed the Q4_1 kernel suite and matched an eight-token native text request.

```sh
p=.model-cache/rss-residency
python3 "$p/build.py"
python3 "$p/snapshot.py"
python3 "$p/pages.py"
b=downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio-h0b-bank1
c=downfam2-8-11-x31tr-o28-q28-raw-a0w-verifyio-h0b-bank1
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  base "$b-advice0" "$b-advice1"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-advice0" "$b-advice1"
python3 "$p/build-trim.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  vtrim0 vtrim1 "$b-trim0" "$b-trim1"
python3 "$p/build-reserve.py"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 \
  vguard0 vguard1 "$b-guard0" "$b-guard1" "$c-guard1"
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 \
  base "$b-guard1" "$c-guard1"
python3 .model-cache/rss-direct/verify.py "$c-guard1"
bash "$p/build-native.sh"
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 3 meta-before meta-after
MOTY_BENCH_COOLDOWN=6 python3 .model-cache/rss-direct/bench.py 3584 12 2 meta-before meta-after
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 meta-before meta-text-trim
MOTY_BENCH_COOLDOWN=1 python3 .model-cache/rss-direct/bench-natural.py 2 meta-before meta-after
python3 "$p/verify-copy.py"
python3 .model-cache/rss-direct/verify.py meta-after
```

### Independent reader limits for prefill and decode

The phase-specific queue trial keeps five readers for prefill and sets the active limit to three, four or five when switching to the compact decode bank. The reader count can return to five when an interactive request enters prefill again. A separate prefill stripe setting lets raw and indexed reads use different sizes. Each stripe descriptor carries its chosen size, and the worker reads it after acquiring the published request.

In the first 3584-input screen the resident control ranged from 152.70 to 306.16 TPS. Prefill also varied substantially among the cases with identical prefill settings. The streaming cases peaked between 209.53 and 212.84 MiB, and every run matched all 128 output IDs. This series does not isolate an effect on throughput.

A second overlay parks phase-disabled readers on their own condition variable. Enqueuing ordinary work wakes the active reader pool, and changing the phase wakes the parked readers. Shutdown wakes both sets. The paired timing overlay changes the active limit every eight generated steps within a single request. It alternates the start order between runs and excludes the first two steps of each group from the rate calculation.

In a separate profile on 3584 inputs, foreground gate/up waits across 128 generated tokens were 232.89 ms for five active readers, 111.50 ms for four and 92.10 ms for four with separate parking. The corresponding down waits were 0.33, 9.10 and 10.76 ms. Prefill wall time ranged from 4.10 to 8.63 seconds despite the same prefill settings. The profiles localize the stalls and do not establish a steady-state speedup.

The four-reader variant with separate parking matched 457 generated IDs and 2310 teacher-forced top-1 IDs at perplexity 119.9085. Its short fixture matched with `PREFILL_CHUNK=0/1/16`, and two interactive turns matched all 32 IDs.

Alternating limits within each request gave median changes of +0.32% for four readers and +0.33% for three readers on the short fixture, relative to five. On the long fixture, the results for four readers ranged from −13.55% to +217.39%, and those for three readers ranged from −4.63% to +19.34%. The five-reader bands themselves ranged from 8.49 to 63.92 TPS on long inputs. All paired requests matched the 128 reference IDs. Even the within-request measurements lacked stable long-context throughput, so the scheduler remains experimental.

```sh
p=.model-cache/rss-stage
l=downfam4-12-11-x11tr-o28-q28-raw-a2-verifyio-h0b-bank1
b=downfam2-8-11-x11tr-o28-q28-raw-a0w-verifyio-h0b-bank1
python3 "$p/build.py"
MOTY_BENCH_COOLDOWN=5 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  meta-before "$l-rd5t0" "$l-rd4t0" "$l-rd3t0" "$l-rd4t1024"
python3 "$p/build-park.py"
MOTY_BENCH_COOLDOWN=8 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  meta-after "$l-rd5t0-pk0" "$l-rd4t0-pk0" "$l-rd4t0-pk1"
python3 "$p/profile.py" 3584 "$l-rd5t0-pk0" "$l-rd4t0-pk0" "$l-rd4t0-pk1"
python3 .model-cache/rss-direct/verify.py "$l-rd4t0-pk1"
MOTY_PHASE_PROFILE_BINARY="$PWD/$p/park-profile" \
  python3 .model-cache/rss-phase/probe.py "$b-rd4t0-pk1"
python3 "$p/build-paired.py"
python3 "$p/paired.py" 3584 "$l-rd5t0-pk1"
python3 "$p/paired.py" 256 "$b-rd5t0-pk1"
```

### FP32 KV residency and one-shot token-ID input

The KV allocation experiment compares the original context capacity, delayed allocation at the original capacity, and delayed allocation at the prompt length plus the generation limit. The bounded case was tested with separate per-layer buffers and a shared FP32 arena. The instrumented executable counts unique Darwin pages intersecting the K and V arrays with `mincore` before prefill and at the first single-token decode step.

| Input tokens | Original / bounded capacity | Original / shared-arena KV before prefill (MiB) | Original / shared-arena KV after prefill (MiB) |
|---:|---:|---:|---:|
| 15 | 512 / 31 | 0.00 / 0.00 | 1.75 / 0.84 |
| 256 | 512 / 384 | 0.00 / 0.00 | 7.00 / 7.00 |
| 3584 | 4096 / 3712 | 0.00 / 0.00 | 98.00 / 98.00 |

The unused capacity is predominantly virtual. The resident KV grew by the same amount on both fixed fixtures, and the instrumented runs matched all 128 generated IDs for every mode at each length. On eight ordinary prompts, all 201 IDs matched in each of two passes. The shared arena changed the native per-pass peak from 193.08 to 192.95 MiB and aggregate decode from 533.97 to 532.20 tok/s.

In two alternating short-fixture passes, the shared arena changed median RSS from 199.56 to 199.69 MiB with mapped weights and from 110.79 to 110.76 MiB with streamed weights. The long-fixture changes were 292.87 to 292.10 MiB and 209.30 to 209.90 MiB. Long decode measurements ran at 60 to 129 tok/s across all variants, well below earlier measurements on the same host. This series is used for RSS and token comparisons. The streaming prototype matched 457 generated IDs and 2310 top-1 IDs, with perplexity 119.9085 and the same short output for `PREFILL_CHUNK=0/1/16`. The existing allocator remains the default.

The direct `PROMPT_IDS` path now copies parsed values into the existing history buffer, frees the JSON arena and source text before prefill, and allocates the text input buffer only for text or interactive input.

Seven alternating native pairs gave median peak RSS of 199.72 → 199.70 MiB on the 256-input fixture and 293.34 → 292.73 MiB on the 3584-input fixture. The 128 generated IDs matched in every pair. The short fixture also matched with `KV_BITS=8`, `PREFILL_CHUNK=1/16` and exact lookahead. Lookahead accepted 85 of 86 proposals in both binaries. Invalid, empty or oversized ID arrays were rejected before prefill. The resident control varied from 168 to 337 TPS on the short fixture and from 124 to 159 TPS on the long fixture. The short-fixture paired decode change was −14.01%, and the long-fixture change was +1.89%. These runs do not establish speed parity.

A follow-up used the same executable and alternated whether it released the JSON and source or retained them with a second token buffer. In nine pairs on the 256-input fixture, the paired PP and TPS changes were +3.54% and +0.96% with four threads and −4.86% and +4.16% with twelve. The twelve-thread medians were 159.40 and 157.19 TPS. The same-binary comparison did not repeat the previous paired decode drop. The control remained below its earlier throughput.

The final native CLI matched all 457 generated IDs and all 2310 teacher-forced top-1 IDs at perplexity 119.9085. The local suite passed 175 tests with four platform skips. Under `qemu-aarch64 -cpu cortex-a53`, the ARMv8.0 Q4_1 kernel suite passed and a 256-input run matched all 32 generated IDs against the binary built with the original prompt parser.

```sh
p=.model-cache/rss-kvsize
python3 "$p/build.py"
python3 "$p/probe.py"
python3 .model-cache/rss-direct/bench-natural.py 2 kvsize0 kvsize1 kvsize2 kvsize3
b=downfam2-8-11-x31tr-o28-q28-raw-a0w-verifyio-h0b-bank1-rd5t0-pk0
l=downfam4-12-11-x11tr-o28-q28-raw-a2-verifyio-h0b-bank1-rd5t0-pk0
MOTY_BENCH_COOLDOWN=2 python3 .model-cache/rss-direct/bench.py 256 12 2 \
  kvsize0 kvsize3 "$b-sz0" "$b-sz3"
MOTY_BENCH_COOLDOWN=5 python3 .model-cache/rss-direct/bench.py 3584 12 2 \
  kvsize0 kvsize3 "$l-sz0" "$l-sz3"
python3 .model-cache/rss-direct/verify.py "$b-sz3"
bash "$p/build-ids.sh"
python3 "$p/ids-check.py"
python3 .model-cache/rss-direct/verify.py ids-after
bash "$p/check-a53.sh"
python3 "$p/parser-ab.py"
```

### FP16 KV storage with FP32 attention

The local storage experiment switches K, V or both to FP16 for Pocket. Prefill and short decode expand the visible rows into a shared FP32 layer buffer. Long single-token decode instead expands one block per worker and keeps the original FP32 reduction order. The whole-model quality comparison uses the same binary, GGUF and twelve-worker setting for every case. The control perplexity is 119.9085.

| FP16 storage | Matching generated IDs | Matching teacher-forced top-1 IDs | Perplexity |
|---|---:|---:|---:|
| K | 405 / 457 | 2307 / 2310 | 119.8901 |
| V | 457 / 457 | 2308 / 2310 | 119.8954 |
| K and V | 405 / 457 | 2309 / 2310 | 119.8776 |

K rounding changed one of the eight ordinary responses. All three variants matched the 128 generated IDs on the two fixed inputs. The V-only experiment preserved the complete generation corpus and failed the teacher-forced equality check. No FP16 cache mode passed the quality gate, so no end-to-end performance claim or production change follows from this trial.

A separate NEON prototype widens V directly from the FP16 cache in the value loop and updates FP32 accumulators in position order. Its 896 context values matched bit for bit against the original GQA reading the same FP16-rounded values already expanded to FP32, at all six visible lengths from 2048 to 3712. On a captured first-layer cache at 3584 positions, the single timing series measured 179.06 µs with FP32 loads and 180.00 µs with direct FP16 loads. This isolates the widening cost for the tested kernel and uses inputs already rounded in both cases.

```sh
p=.model-cache/rss-kvhalf
python3 "$p/build-storage.py"
OMP_WAIT_POLICY=PASSIVE MOTY_KV_HALF_BINARY=store python3 "$p/quality.py"
python3 "$p/build-direct.py"
clang -O2 -mcpu=apple-m1 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
  -flto=auto -Ic "$p/check-direct.c" "$p/gqa-direct.c" \
  c/libmoty-nn.a c/libmoty-hw.a -lm -L/opt/homebrew/opt/libomp/lib -lomp -o "$p/check-direct"
Q41_ACT=f16 "$p/check-direct" .model-cache/rss-joint/kv-long.bin
```

### 24-bit KV storage and seven-head GQA

The 24-bit cache experiment uses the FP32 sign and exponent field widths and rounds the significand to 16 bits. A 64-value row occupies 192 bytes in four groups. NEON stores the high halfwords and middle bytes separately and reconstructs FP32 words before attention. The local converter matched a scalar round-to-nearest-even implementation on 1,024,000 inputs, including ties, signed zero, subnormals, infinities and NaNs.

| 24-bit storage | Matching generated IDs | Matching teacher-forced top-1 IDs | Perplexity |
|---|---:|---:|---:|
| K | 457 / 457 | 2308 / 2310 | 119.8994 |
| V | 457 / 457 | 2308 / 2310 | 119.8942 |
| K and V | 457 / 457 | 2308 / 2310 | 119.9029 |

At teacher-forced position 431 the FP32 top two logits differed by about 0.00020, and each 24-bit variant changed the selected ID. A debug control copied unmodified FP32 cache values through the same staging and long-decode schedule. Its top-1 file matched the original byte for byte and both runs measured perplexity 119.9085. The 24-bit variants remain local experiments.

In two alternating passes at 3584 input tokens, native peak RSS changed from 292.71 to 272.10 MiB. With the same streamed-weight allocation, peak RSS changed from 209.12 to 189.23 MiB. Keeping nine gate/up pairs and seventeen down matrices resident raised the 24-bit result to 207.30 MiB. All fixed runs generated the same 128 IDs. The native control ranged from 121.67 to 332.08 TPS and the streaming control from 22.51 to 38.24 TPS, so the series establishes RSS results for this fixture and gives insufficient evidence for a throughput comparison.

In two 3584-token FP32 captures, fewer than 1.2% of K rows and 0.35% of V rows exceeded a 16-value exponent span. A reversible per-row exponent layout with a one-byte origin would occupy about 86.23 MiB of payload versus 98 MiB, before allocator and exception index overhead. These figures estimate storage from captured values.

With FP32 KV, a separate kernel shares each V load across all seven query heads in strips of four or eight channels. Both variants produced bit-identical attention output at visible lengths 2048, 2049, 2177, 3584, 3711 and 3712. Visibility 2047 selected the existing fallback. On a captured first-layer cache at 3584 positions, the median paired changes in full GQA time were +2.24% and +0.06% across seven series of 300 iterations per mode. This screen did not justify changing the value kernel.

```sh
p=.model-cache/rss-kv24
python3 "$p/build.py"
python3 "$p/build.py" stream
clang -O2 -mcpu=apple-m1 "$p/check-codec.c" -o "$p/check-codec"
"$p/check-codec"
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
python3 "$p/build-debug.py"
bash "$p/debug.sh"
.model-cache/validation/bin/python "$p/kv-range.py"
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  kv24-native-fp32 kv24-native-kv kv24-stream-4-12-fp32 kv24-stream-4-12-kv \
  kv24-stream-9-17-kv
p=.model-cache/rss-gqa7
python3 "$p/build.py"
clang -O2 -mcpu=apple-m1 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
  -flto=auto -Ic "$p/check.c" "$p/gqa-native.c" c/libmoty-nn.a c/libmoty-hw.a \
  -lm -L/opt/homebrew/opt/libomp/lib -lomp -o "$p/check"
Q41_ACT=f16 "$p/check" .model-cache/rss-joint/kv-long.bin
```

### Reversible per-row FP32 KV storage

The experimental codec encodes FP32 exponents relative to the largest exponent in each 64-value row. Rows whose exponent range fits in sixteen codes have a 224-byte payload and a one-byte origin. An exception row keeps its original exponent bytes. Signs and all mantissa bits are preserved. The row layout uses no model-specific calibration table. During long decode, each worker expands at most 128 rows at a time into its own FP32 buffer before running the existing score or value reductions.

The codec reconstructed all 401,408 rows in each of two independent 3584-token captures byte for byte. The captures contained 3022 and 3079 exponent exceptions. The same check rewrote an already allocated exception row and verified the replacement. The model matched all 457 generated IDs and 2310 teacher-forced top-1 IDs, with perplexity 119.9085. The block-check output matched bit for bit for all six visible lengths and all four compression masks. For a first-layer capture at 3584 positions, one timing series measured 271.99 µs with the original GQA and 282.51 µs with both cache matrices encoded.

Two alternating pairs on the 3584-input fixture used twelve workers and matched all 128 output IDs on every run. The RSS target is 216.89 MiB.

| Weights / KV | PP (tok/s) | TPS (tok/s) | Median peak RSS (MiB) | Highest peak RSS (MiB) |
|---|---:|---:|---:|---:|
| Native mapped / raw | 991.10 | 332.71 | 292.78 | 292.89 |
| Native mapped / reversible | 992.50 | 314.60 | 287.02 | 287.23 |
| Streaming 4 gate/up and 12 down / raw | 983.65 | 315.92 | 211.52 | 211.66 |
| Streaming 4 gate/up and 12 down / reversible | 978.05 | 312.83 | 204.65 | 206.50 |
| Streaming 7 gate/up and 15 down / reversible | 974.10 | 307.20 | 216.80 | 217.41 |

The smaller streaming configuration freed 6.87 MiB with reversible KV and lost 0.98% in median decode throughput against the same placement with raw KV. It was 5.98% slower than the native mapped control in this screen. Increasing both resident matrix counts did not recover the throughput and exceeded the peak RSS ceiling in one run.

A follow-up writes the unmodified new FP32 rows into the staging buffer and unpacks only the existing prefix. The NEON unpacker also widens the shared sign and mantissa byte with one signed operation. This cut median paired GQA time by 1.59% relative to the first decoder across seven short timing series. The prefill change preserved the full native quality corpus, the 256-input cases with `PREFILL_CHUNK=0/1/16`, `KV_BITS=8` and exact lookahead, and output IDs on text-derived fixtures of lengths 2047, 2048, 2049, 2177 and 3584.

With seven resident gate/up pairs and twelve down matrices, the faster variant reached 998.10 PP and 321.66 TPS at a maximum of 212.62 MiB in the two-pair follow-up. The mapped control reached 1021.25 PP and 333.92 TPS. The paired short-fixture screen favored raw FP32 KV. The raw streaming configuration reached 491.00 TPS at a maximum of 110.94 MiB, versus 490.44 TPS in the mapped control. The reversible short variant reached 471.74 TPS and exceeded the short memory ceiling at 111.56 MiB.

In a second long-context comparison, the selected 7/12 placement reached 321.11 TPS and 213.33 MiB at its peak, against 334.54 TPS mapped. The median paired decode changes in the two screens were −3.67% and −4.01%. Retaining two additional gate/up pairs exceeded the memory ceiling in one run. Encoding V alone with the 7/12 placement stayed below the ceiling at 216.56 MiB but reached 310.19 TPS in the two-pair screen.

The complete accelerated 7/12 streaming variant matched the mapped FP32 control on all 457 generated IDs and 2310 teacher-forced top-1 IDs. Both measured perplexity 119.9085.

Seven short-fixture pairs with raw streaming KV reached a maximum of 110.92 MiB and matched all generated IDs. Paired changes against mapped weights were +1.06% TPS, −2.38% PP and +2.18% TTFT. On eight ordinary text requests, two passes with matching IDs measured 514.12 aggregate TPS and a 106.73 MiB maximum for raw streaming, compared with 526.22 TPS mapped. Both fixtures meet their RSS ceilings in the selected configurations. The long fixture and ordinary requests still show a decode speed gap, so the configurations remain experimental.

```sh
p=.model-cache/rss-kvrow
python3 "$p/build.py"
clang -O2 -mcpu=apple-m1 "$p/check.c" -o "$p/check"
for fixture in long text; do
  "$p/check" ".model-cache/rss-joint/kv-$fixture.bin"
done
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
clang -O2 -mcpu=apple-m1 -Xclang -fopenmp -I/opt/homebrew/opt/libomp/include \
  -flto=auto -Ic "$p/check-gqa.c" .model-cache/rss-kvhalf/gqa.c \
  c/libmoty-nn.a c/libmoty-hw.a -lm -L/opt/homebrew/opt/libomp/lib -lomp -o "$p/check-gqa"
Q41_ACT=f16 "$p/check-gqa" .model-cache/rss-joint/kv-long.bin
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  kvrow-native-fp32 kvrow-native-kv kvrow-stream-4-12-fp32 kvrow-stream-4-12-kv \
  kvrow-stream-7-15-kv
python3 "$p/build-direct.py"
OMP_WAIT_POLICY=PASSIVE bash "$p/verify-direct.sh"
python3 "$p/check-unpack.py"
MOTY_TEST_KVROW_BUILD_FAST=1 python3 "$p/build-direct.py"
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  kvrow-native-fp32 kvrow-stream-4-12-kvf kvrow-stream-7-12-kvf kvrow-stream-4-18-kvf
MOTY_KVROW_SHORT=1 MOTY_BENCH_COOLDOWN=1 python3 "$p/bench.py" 256 12 2 \
  kvrow-native-fp32 kvrow-stream-2-8-fp32 kvrow-stream-2-8-kvf
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  kvrow-native-fp32 kvrow-stream-7-12-kvf kvrow-stream-9-12-kvf kvrow-stream-7-12-vf
MOTY_KVROW_SHORT=1 MOTY_BENCH_COOLDOWN=1 python3 "$p/bench.py" 256 12 7 \
  kvrow-native-fp32 kvrow-stream-2-8-fp32
MOTY_KVROW_SHORT=1 python3 "$p/bench-natural.py" 2 \
  kvrow-native-fp32 kvrow-stream-2-8-fp32
OMP_WAIT_POLICY=PASSIVE python3 "$p/combined-quality.py"
```

### Streaming with raw FP32 KV and GQA softmax

The 3584-input decode profile compared mapped weights with raw FP32 KV, streamed weights with raw FP32 KV, and streamed weights with reversible KV. In the two raw streaming runs, gate/up and down waited 1.28 and 1.58 ms combined across 128 generated tokens. Publishing the following gate/up windows took 10.90 and 11.12 ms. GQA measured 144.07–144.29 ms with raw streaming KV, 139.87–143.27 ms mapped, and 154.81–160.45 ms with reversible streaming KV.

The targeted-wakeup trial uses one semaphore per reader and an atomic parked-reader mask. With the same seven-gate/twelve-down allocation and raw KV, the existing condition-variable path measured 306.25 TPS. Waking one reader or waking up to the number of stripes measured 286.37 and 310.99 TPS respectively in two alternating passes. All generated IDs matched. The second mode peaked at 219.53 MiB, and native control ranged from 254.96 to 322.59 TPS in this series. The targeted wakeups remain experimental.

Changing the raw streaming allocation to seven gate/up pairs and ten down matrices reached 322.07 TPS versus 325.53 TPS mapped, at a maximum of 216.48 MiB in two alternating pairs. The partial reversible layouts retained the larger weight allocation and encoded V on twelve or sixteen layers, or encoded both matrices on ten layers. All four tested partial layouts exceeded 216.89 MiB in at least one run.

The experimental NEON softmax unrolls the maximum and normalization loops. It uses the original exponential approximation and four FP32 sum chains. For finite inputs in the isolated check, every output matched bit for bit at all lengths from 1 through 4224. The 3584-to-3711-element timings across seven alternating series had medians of 1.963 µs with the previous loop and 1.535 µs with the wider loop. In the first model screen, using the wider loop for GQA and issuing the first two weight windows before the preceding output projection reached 329.37 TPS at a maximum of 216.56 MiB. Mapped control reached 332.97 TPS.

Applying the wider softmax to prefill and decode, allowing 1536 KiB prefill read stripes, and issuing two windows of each kind before the head produced the following seven-pair results. All 128 generated IDs matched in each pair.

| 3584-input configuration | PP (tok/s) | TPS (tok/s) | Paired PP / TPS change | Highest peak RSS (MiB) |
|---|---:|---:|---:|---:|
| Mapped weights | 1035.80 | 337.88 | Reference | 294.53 |
| Streamed, seven gate/up and ten down | 1019.90 | 332.98 | −1.76% / −1.60% | 216.66 |
| Streamed, seven gate/up and nine down | 1019.40 | 334.52 | −1.32% / −1.27% | 217.64 |

On a text-derived 3584-input fixture, the corresponding two-pair speed changes were −0.72% PP and −1.48% TPS for 7/10, and −0.93% PP and +1.39% TPS for 7/9. The highest peaks were 216.36 and 215.16 MiB respectively. On the ordinary short requests, seven passes gave 521.80 aggregate TPS mapped, 502.96 streamed with the original softmax and 495.01 streamed using the wider loop throughout. Every pass matched all 201 output IDs. The wider loop therefore remains a long-context experiment in these configurations.

In the initial prefill-chunk sweep, 384 tokens per chunk reached 1027.10 PP and 334.73 TPS, versus 1023.65 PP and 335.41 TPS mapped. Paired changes were +0.34% PP and −0.21% TPS. Its maximum RSS was 219.77 MiB. Both 320 and 448 tokens per chunk also exceeded the memory ceiling with the same resident allocation.

Keeping fewer down matrices resident before prefill and promoting them afterward reached 1017.45 PP with seven gate/up pairs and eight down matrices during prefill and three additional down matrices in decode. The 384-token configuration reached 319.81 TPS and a maximum of 217.70 MiB. Another 384-token placement with seven gate/up and seven down matrices stayed below the ceiling at 216.52 MiB but measured −3.85% paired TPS.

A three-query-head, three-position score tile preserved all context floats in the isolated GQA check at the six long visible lengths. It did not improve the seven-series GQA comparison. The selected 7/10 raw-KV configuration and the short 2/8 configuration built with the wider softmax passed all 457 generated IDs and 2310 teacher-forced top-1 IDs at perplexity 119.9085. The short fixture also matched with `PREFILL_CHUNK=0/1/16`.

The short fixed fixture reached a maximum of 110.88 MiB in four pairs with the wide-softmax 2/8 configuration. Its paired changes were +1.94% TPS and −1.52% PP. On eight ordinary text requests the smaller live KV made room for one more resident gate/up pair and one more down matrix. In seven alternating passes with that 3/9 placement, all 201 output IDs matched. Peak RSS stayed at or below 110.36 MiB and the paired decode change was −0.89% (522.07 aggregate TPS versus 526.96 mapped). The tested placements remain separate because their RSS margins depend on request length.

For the dense Pocket prefill workspace, an experimental allocation reuses the normalized input buffer as the output of attention and FFN. Each of those operations finishes reading the normalized rows before its final projection writes the output. This saves one hidden-width buffer for the chunk, 1.31 MiB at 384 tokens. Across two pairs and the same 7/8 weight placement, the peak maximum changed from 217.22 to 215.91 MiB. The variant that promoted two down matrices after prefill peaked at 213.06 MiB, with 1026.10 PP and 327.14 TPS in the two-pair screen.

Seven follow-up pairs with that variant reached 1030.30 PP and 330.84 TPS versus 1032.50 PP and 332.97 TPS mapped. The median paired changes were −0.26% PP, −0.77% TPS, +0.33% TTFT and +0.15% total request time. The highest observed RSS was 216.45 MiB, below the 216.89 MiB ceiling. The fixed fixture preserved all 128 generated IDs in every run.

On the text-derived 3584-input fixture the same variant matched all 128 IDs, but its highest observed peak was 217.25 MiB. Its two-pair changes were −0.30% PP and −1.74% TPS. The full quality check of this long variant and the short 3/9 placement matched all 457 generated IDs, all 2310 teacher-forced top-1 IDs and perplexity 119.9085. The implementation remains experimental because the text-derived input exceeded the RSS ceiling and measured a throughput gap. Executable, GGUF and fixture hashes are recorded in `.model-cache/rss-close-gap/sha256.txt`.

```sh
p=.model-cache/rss-close-gap
bash "$p/profile.sh"
python3 "$p/build.py"
python3 "$p/build-softmax.py"
python3 "$p/build-wide-all.py"
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 7 \
  native low-raw-7-10-w0a2t1536s2 low-raw-7-9-w0a2t1536s2
MOTY_GAP_TEXT=1 MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  native low-raw-7-10-w0a2t1536s2 low-raw-7-9-w0a2t1536s2
MOTY_KVROW_SHORT=1 python3 "$p/bench-natural.py" 7 native low-raw-2-8-w0 low-raw-2-8-w0s2
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  native low-raw-7-10-w0a2s1 low-raw-7-10-w0a2c320s2 \
  low-raw-7-10-w0a2c384s2 low-raw-7-10-w0a2c448s2
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  native low-raw-7-8-w0a2c384s2p13 low-raw-7-7-w0a2c384s2p14 \
  low-raw-7-8-w0a2c384s2p12 low-raw-7-7-w0a2c384s2
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py" low-raw-7-10-w0a2t1536s2 low-raw-2-8-w0s2
python3 "$p/build-prefill-reuse.py"
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  native low-raw-7-8-w0a2c384s5 low-raw-7-8-w0a2c384s4 \
  low-raw-7-8-w0a2c384s4p12 low-raw-7-9-w0a2c384s4
MOTY_KVROW_SHORT=1 python3 "$p/bench-natural.py" 7 native low-raw-3-9-w0 low-raw-2-8-w0p41
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 7 native low-raw-7-8-w0a2c384s4p12
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py" low-raw-7-8-w0a2c384s4p12 low-raw-3-9-w0
MOTY_GAP_TEXT=1 MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 2 \
  native low-raw-7-8-w0a2c384s4p12
```

### Reusing the native prefill normalization buffer

The native Pocket workspace now uses two hidden-width buffers. Attention finishes reading the normalized rows in Q/K/V projection, and FFN finishes reading them in gate/up. Their output projections can write back into the same normalization buffer before the residual addition. The change applies where the existing dense Pocket prefill workspace is active, including automatic chunks. The allocated workspace saves one hidden-width buffer per chunk, or 0.875 MiB at 256 tokens and 1.31 MiB at 384 tokens.

| Input tokens / 12 workers | Median peak RSS before → after (MiB) | Paired PP change | Paired TPS change |
|---:|---:|---:|---:|
| 256 | 199.95 → 199.17 | +0.20% | +0.19% |
| 3584 | 292.92 → 292.06 | −0.32% | +0.97% |

Both comparisons used seven alternating pairs. Every pair produced the same 128 generated IDs. The maximum of the seven peaks changed from 201.75 to 200.61 MiB on the short input and from 294.58 to 292.20 MiB on the long input.

With 384-token automatic chunks, every byte of the final prefill logits and visible FP32 K/V matched the prior workspace. The combined snapshot was 103,022,592 bytes. Full validation matched all 457 generated IDs, 2310 top-1 IDs and perplexity 119.9085. The short fixture also matched with `PREFILL_CHUNK=0/1/16/64/128/384`, `KV_BITS=8`, `MMAP=0`, FP32 Q4_1 activations and separate Q/K/V and FFN projections. Local CTest passed 175 tests with four platform skips. The ARMv8.0 Q4_1 kernel suite passed under Cortex-A53 QEMU, and all 32 generated IDs matched with the workspace explicitly enabled.

The earlier raw-KV streaming prototype was instrumented at prefill chunk boundaries, the phase transition and 16-token decode intervals. In three fixed-input and three text-derived runs, its first recorded high-water mark occurred in the decode interval ending at 3664 visible tokens. Peaks in that series were 212.94–213.22 MiB. The earlier 217.25 MiB outlier on text-derived input did not recur under instrumentation.

The extra grouped-prefill score rows reserve 0.5625 MiB at `CTX=4096`. An overlay maps the score scratch only for prefill and unmaps it before decode. Streaming with that allocation matched all 128 IDs on the fixed and text-derived fixtures, and all 32 IDs and output bytes across two interactive turns. The highest text-fixture peak was 216.28 MiB with scoped scores and 215.95 MiB with the persistent allocation in the two-pair screen. The fixed-fixture maxima were 215.41 and 215.58 MiB. These timings ran below the earlier host control throughput. The RSS measurements did not show a repeatable whole-process reduction from this lifetime change alone.

Sharing one RoPE table-row lookup across Q and K heads was bit-identical in 16,384 standalone cases covering 4096 positions and four normalization masks. It did not improve the isolated latency. A separate mapping experiment releases the consumed prefix of the precomputed RoPE table at the long prefill-to-decode transition. At position 3584 that releases 917,504 bytes. A standalone check reproduced every retained and rewound row bit for bit, then grew the table and rechecked the retained suffix. When combined with scoped prefill scores, three alternating pairs on the 3584-token text fixture matched all 128 IDs, with a highest observed peak of 216.28 MiB. The paired changes were +1.17% PP and +0.59% TPS in that screen.

For the fixed 3584-input fixture, the same combination reached a maximum of 216.03 MiB in two pairs, with −2.12% paired PP and −5.82% paired TPS. It passed the full 457 generated IDs, 2310 top-1 IDs and perplexity 119.9085. Two interactive turns, the first with 2627 input tokens and the second with 15, preserved all 32 IDs and the output bytes. These results leave streaming experimental because the repeated fixed-input comparisons have not established speed parity.

A fused residual addition and RMSNorm produced bit-identical residual and normalized rows in 1024 standalone cases, including an aliased output. It cut median paired time by 30.28% for the isolated 896-wide row. Full-model comparisons remained below mapped decode throughput. The combined streaming variant with this kernel also reached 217.39 MiB on the text-derived input, so the kernel stays experimental.

```sh
p=.model-cache/rss-workspace
bash "$p/build-native.sh"
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 256 12 7 before after
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 7 before after
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench.py" 3584 12 3 before-384 after-384
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/proof.py"
bash "$p/check-a53.sh"
python3 "$p/locate.py"
python3 "$p/build-scores.py"
MOTY_GAP_TEXT=1 MOTY_BENCH_COOLDOWN=2 python3 "$p/bench-scores.py" 3584 12 2 \
  before 2-full 2-scoped 1-scoped
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench-scores.py" 3584 12 2 before 2-full 2-scoped 1-scoped
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/build-rope-trim.py"
MOTY_GAP_TEXT=1 MOTY_BENCH_COOLDOWN=2 python3 "$p/bench-trim.py" 3584 12 3 \
  before 2-full 2-trim 2-both
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench-trim.py" 3584 12 2 before 2-full 2-trim 2-both
OMP_WAIT_POLICY=PASSIVE python3 "$p/trim-quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive-long.py"
python3 "$p/build-residnorm.py"
MOTY_BENCH_COOLDOWN=2 python3 "$p/bench-residnorm.py" 3584 12 2 before 2-full 2-norm 2-combo
MOTY_GAP_TEXT=1 MOTY_BENCH_COOLDOWN=2 python3 "$p/bench-residnorm.py" 3584 12 2 \
  before 2-full 2-norm 2-combo
```

### Native FFN streaming with a memory budget

The Pocket CLI accepts `QBITS=-1` together with `MEM_GB` or `MEM_FRAC`.
Attention matrices, norms and the tied head remain resident. The budget
selects how many FFN layers remain resident after accounting for the KV
capacity and weight windows. The remaining gate, up and down projections
are read from GGUF into those windows and use the existing Q4_1 kernels
with the original codes, scales and minima. On Apple, when two or three
windows fit, four persistent readers fill the windows while the compute
workers run. The engine waits after the attention stage and reuses a
window after its FFN projections complete. A tighter budget and other
platforms use one synchronous window.

`MEM_GB` is a placement budget in GiB. The process also needs memory for
tokenization and inference scratch, and the fixed tensors and a single
window set a floor on placement. Use the model's usual context limit and
measure peak RSS for the request. The resident path is the default when the
budget is absent.

The first native streaming implementation used two windows and three
new dispatch jobs per layer. Seven alternating pairs on M5 Max used
12 workers, FP16 activations, a warm file cache and 128 generated tokens
per request. The comparison used the same GGUF with the previous mapped
executable. PP and TPS are run medians.

| Input tokens | `MEM_GB` | Median peak RSS, mapped → streamed (MiB) | Highest streamed peak (MiB) | PP, mapped → streamed | TPS, mapped → streamed |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.102 | 199.05 → 105.72 | 107.19 | 1337.8 → 1326.8 | 473.72 → 279.09 |
| 3584 | 0.203 | 292.14 → 206.97 | 207.06 | 1017.6 → 1017.3 | 332.04 → 268.10 |

The highest streamed peaks meet the original 111.25 and 216.89 MiB
ceilings. Median paired PP changes were −0.67% and +0.39%, and TPS changes
were −41.56% and −18.88%. TTFT changed by −2.31% and −0.56%, while total
request time changed by +38.49% and +1.67%. On eight ordinary text prompts,
three passes matched all 201 generated IDs per pass. Their median
aggregate decode rates were 522.25 and 303.59 TPS, with a median of 99.94
MiB for the highest streamed peak in each pass. The decode cost keeps this
placement opt-in.

The operation profile isolates the decode cost. Waiting for FFN reads
took 173.17 ms of the 455.5 ms short decode and 83.28 ms of the 484.0 ms
long decode. Enqueuing those reads took another 6.95 and 6.94 ms. The
combined time in the gate/up and down kernels was 138.76 and 138.79 ms,
versus 142.36 and 139.97 ms mapped. For 256 input tokens, the engine
streams about 102.5 MiB of raw FFN data per generated token.
Splitting each matrix into two or three reads, waiting only after the
current attention stage, and retaining another FFN layer with a higher
budget did not close the throughput gap in the short screens.

On M5 Max the streamed path matched all 457 generated IDs and all 2310
teacher-forced top-1 IDs. Perplexity was 119.9085 in both modes. The tests
also covered explicit prefill chunks, FP32 activations, `KV_BITS=8`,
`MMAP=0` and separate Q/K/V and FFN projections. The small GGUF regression
checks bitwise prefill and decode logits with a single window, Apple
double buffering and enough budget for all layers. The local suite passed
176 tests with four platform skips. The ARMv8.0 suite passed under
`qemu-aarch64 -cpu cortex-a53`. The model matched 329 generated IDs and
2310 top-1 IDs in that configuration, with perplexity unchanged at
119.8882. The QEMU runs establish compatibility. Executable and input
hashes and paired timing logs are in `.model-cache/rss-native-stream`.

```sh
make -C c ARCH=apple-m1 moty
p=.model-cache
GGUF="$p/MNN-Pocket-0.3B-Instruct-Q4_1.gguf"
for n in 256 3584; do
  case "$n" in
    256) ids="$p/pocket-prefill-256.json"; ctx=512; budget=0.102 ;;
    3584) ids="$p/q41-bottleneck/prompt3584.json"; ctx=4096; budget=0.203 ;;
  esac
  /usr/bin/time -l env GGUF="$GGUF" QBITS=-1 Q41_ACT=f16 THREADS=12 \
    CTX="$ctx" PROMPT_IDS="$ids" TEMP=0 TOKENS=1 NGEN=128 IGNORE_EOS=1 \
    MEM_GB="$budget" ./c/moty
done
```

### Persistent readers for native FFN streaming

Four persistent readers replace the per-layer dispatch jobs. Each read
still transfers one original GGUF matrix. The queue can hold three FFN
layers when the budget allows it. Window reuse is synchronized with the
end of the FFN projection. The CLI joins the readers when generation,
validation, or perplexity evaluation exits.

Seven alternating comparisons used the same memory budgets, GGUF and
12 compute workers. The 128 generated IDs matched in every run.

| Input tokens | Mapped PP / TPS | Previous streaming PP / TPS | Queue PP / TPS | Highest queue RSS (MiB) |
|---:|---:|---:|---:|---:|
| 256 | 1342.1 / 463.03 | 1329.6 / 279.78 | 1331.6 / 398.72 | 105.12 |
| 3584 | 1028.1 / 319.07 | 1023.6 / 255.63 | 1027.0 / 293.83 | 207.81 |

Median paired decode throughput increased by 41.28% and 15.94% over
the previous streaming executable. Relative to the mapped executable,
paired PP changed by −0.90% and −0.09%, and decode throughput changed by
−14.12% and −6.54%. Total request time changed by +8.59% and +0.92%.
The original 111.25 and 216.89 MiB RSS ceilings still hold.

Across five passes of the eight ordinary text prompts, all three
executables generated the same 201 IDs per pass. Median aggregate
decode throughput was 507.35 TPS mapped, 283.91 TPS with the earlier
stream and 433.33 TPS with persistent readers. The median of each pass's
highest queue RSS was 99.20 MiB.

On the 3584-token input derived from ordinary text, the queue reached
294.90 TPS versus 318.77 TPS mapped and 256.23 TPS with the earlier
stream. The highest queue peak was 207.75 MiB across three passes.
On the 1536-token control with four workers, the queue reached 192.20
TPS versus 208.87 TPS mapped and 191.65 TPS with the earlier stream.
The highest queue peak was 177.11 MiB.

The time spent waiting for reads fell to 18.45–27.36 ms across the
three profiled short decodes and 7.60–22.10 ms across the two long
decodes. That was 6–8% and 2–5% of decode time respectively. The FFN
projections took the largest share of short decode, at about 45%.
Attention took the largest share of long decode, at about 36%.

The compared reader counts, split gate/up and down queues, additional
read stripes, wider softmax and cross-token prefetch stayed experimental.
Reusing the gate/up portion of a window before down completed changed
paired short decode throughput by +1.99% in seven pairs, with a lower
decode median on the long fixture in three pairs. The three-window,
four-reader queue is used for both workloads.

The rebuilt CLI matched 457 generated IDs and 2310 teacher-forced
top-1 IDs on M5 Max at perplexity 119.9085. All 262,144 bytes of the
final prefill logits matched for each of the 256- and 3584-token inputs.
Two interactive turns matched all 32 IDs and the output bytes. The
`MEM_FRAC` invocation matched all 128 IDs on the 256-token fixture.
Local CTest passed 176 tests with four platform skips. The ARMv8.0
suite and the synchronous fallback passed on Cortex-A53 QEMU, where
329 generated IDs and 2310 top-1 IDs matched at perplexity 119.8882.

The timing harness and binary, model and input hashes are in
`.model-cache/rss-q41-queue`. In the command below, `before` is the
saved mapped executable, `final` is the previous native streamer and
`native` is the executable with the persistent reader queue.

```sh
p=.model-cache/rss-q41-queue
cp ./c/moty "$p/native"
python3 "$p/bench.py" 256 12 7 before final:0.102 native:0.102
python3 "$p/bench.py" 3584 12 7 before final:0.203 native:0.203
python3 "$p/bench-natural.py" 5 before final:0.102 native:0.102
BENCH_TEXT_LONG=1 python3 "$p/bench.py" 3584 12 3 before final:0.203 native:0.203
```

### Lossless panel files in the native stream

With a native Pocket memory budget, `MOTY_Q41_PACKED=stream` opens the
same Group64 sidecars used by the resident panel loader. On Apple FMLAL
the streamed FFN projections read those files for single-token steps.
Multi-token steps read the original Q4_1 blocks for streamed matrices.
The tied head and the FFNs that stay resident can use the resident panel
loader. Each streamed panel transfers 10% fewer bytes than its original
Q4_1 matrix.

The loader builds missing sidecars and checks each header against the
source GGUF. Readers verify the payload checksum on first use in the
existing FFN window. When creation, validation or a read fails, the
projection uses the original GGUF blocks.
If one of gate or up fails, both use the original layout and continue
through the fused projection. The window reserves the raw size so the
fallback can reuse the same allocation. Sidecars are closed after the
reader threads join.

To test the panel path against the raw stream, use the timing command
above and add `MOTY_Q41_PACKED=stream` to the environment. That selection
is opt-in. The other `MOTY_Q41_PACKED` selections choose which resident
tensors use panels. `MMAP=0` uses the original blocks.

Seven alternating runs on M5 Max used 12 workers, 128 output tokens and
`MOTY_Q41_PANEL_DUAL=1` for the panel mode. All generated IDs matched.
PP and TPS in the table are medians. RSS is the highest observed peak.

| Input tokens | Raw stream PP → panel PP | Raw stream TPS → panel TPS | Peak RSS (MiB), raw → panel |
|---:|---:|---:|---:|
| 256 | 1335.9 → 1325.0 | 416.13 → 423.81 | 106.47 → 104.98 |
| 3584 | 1027.2 → 1016.0 | 304.30 → 315.51 | 206.47 → 205.98 |

The smaller tied-head table made room for one more resident FFN at each
budget. The FFN payload read per decode step fell from 106.64 to 92.29
MiB on the short fixture and from 98.44 to 84.90 MiB on the long fixture.
Paired TPS changes versus the raw stream were +1.70% and +3.76%.
Relative to the mapped executable, decode throughput changed by
−11.43% and −4.74% and total request time by +7.80% and +1.52%.
Control runs using the rebuilt binary's raw stream reached 415.10
and 305.22 TPS versus 401.13 and 300.94 TPS with the previous binary.

On the long input derived from text, the panel mode reached 304.01
TPS versus 299.53 TPS raw, at a maximum of 204.95 MiB. On the eight
short text prompts, five passes matched all 201 IDs with median
aggregate rates of 439.08 TPS raw and 413.81 TPS with panels.

Raising `MEM_GB` from 0.102 to 0.106 increased the short median from
403.33 to 405.60 TPS in three pairs, with a highest peak of 108.25 MiB.
Raising the long budget from 0.203 to 0.211 increased the median from
313.65 to 315.11 TPS in two pairs, with a highest peak of 212.33 MiB.
The timing command keeps the smaller budgets.

A prototype converted all four scale and minimum pairs in the 2+2
panel kernel with two NEON conversions. In the initial three short
pairs it reached 419.67 TPS versus 415.21 for the existing 2+2 kernel.
In two long pairs it reached 312.12 versus 314.23 TPS. The prototype
remains in the local timing harness.

At a fresh GGUF path with the source already cached, the mode built
85 sidecars. Time to first token was 272.62 ms on that run and
212.61–213.19 ms on two reopen runs. The raw control at the same path
measured 205.66 ms. The output IDs matched.

On Apple the panel mode matched 457 generated IDs and 2310 top-1 IDs
at perplexity 119.9085. Each fixture's final prefill logits matched
across all 262,144 bytes, and two interactive turns matched all 32 IDs
and output bytes. The regression test corrupts the gate payload and
checks the raw gate/up fallback, then alternates decode and prefill.
Local CTest passed 176 tests with four platform skips. ARMv8.0
functional validation in Cortex-A53 QEMU matched 329 generated IDs
and 2310 top-1 IDs at perplexity 119.8882.

Timing scripts, logs and hashes are in `.model-cache/rss-q41-panels`.
To rerun the comparison with the saved mapped and raw controls:

```sh
p=.model-cache/rss-q41-panels
python3 "$p/bench.py" 256 12 7 before native:0.102 stream4-cache-dual:0.102
python3 "$p/bench.py" 3584 12 7 before native:0.203 stream4-cache-dual:0.203
python3 "$p/bench-natural.py" 5 native:0.102 stream4-cache-dual:0.102
BENCH_TEXT_LONG=1 python3 "$p/bench.py" 3584 12 3 native:0.203 stream4-cache-dual:0.203
python3 "$p/first-open-final.py"
```

### Compact attention panels and pthread GQA

With a native Pocket budget on Apple FMLAL, `MOTY_Q41_PACKED=stream-all`
stores resident Q/K/V/O matrices as lossless Group64 panels. Multi-token
prefill reads the original Q4_1 blocks through one shared window. When
the reader queue is available, it fills that window for the next layer
while the current layer runs its FFN. Decode uses the resident panels.
The budget reserves the raw attention window and accounts for the actual
size of the resident matrices.

With `THREADPOOL=1`, `MOTY_Q41_GQA_TEAM=1` lets one pthread worker team
run score, softmax, value accumulation and merge. All threads, including
those assigned empty ranges, join each phase barrier. Each function keeps
the same FP32 reductions and per-position FMA order. The separate phase
dispatch handles nested calls and worker counts above the available CPU
count. This mode is opt-in for the pthread backend.

The split FFN prototype manages gate/up and down separately and uses the
same checked panel files. On the 256-token fixture at `MEM_GB=0.102`, the
three tested placements reached 390.23–416.64 TPS against 422.75 TPS for
the whole-layer queue. On 3584 tokens at `MEM_GB=0.203`, the placements
reached 283.47–308.21 TPS against 305.63 TPS across two pairs. Combining
split placement, compact attention and the pthread build reached
463.42 TPS versus 449.63 TPS without the split in a separate short
screen. It also reached 454.71 versus 444.27 aggregate TPS across three
passes of the eight short text prompts. The split scheduler remains in
the local timing harness pending long-context speed parity.

The final pthread build used `MOTY_POOL_SPIN_US=100`. The 256-token
fixture used `stream-all` and a 0.105 GiB budget. The 3584-token fixture
used `stream` and a 0.203 GiB budget. Both used `MOTY_Q41_PANEL_DUAL=1`,
12 compute workers and 128 generated tokens.

| Input tokens | Pairs | Mapped PP / TPS | Previous stream PP / TPS | Pthread PP / TPS | Highest pthread peak (MiB) |
|---:|---:|---:|---:|---:|---:|
| 256 | 7 | 1342.0 / 478.99 | 1333.5 / 417.74 | 1274.3 / 451.18 | 105.06 |
| 3584 | 3 | 1028.2 / 322.67 | 1009.6 / 299.46 | 1005.3 / 312.12 | 204.11 |

The short control used the raw FFN stream, and the long control used
the earlier FFN panel stream. Paired TPS improved by 8.01% and 5.60%
over those controls. Relative to the mapped executable, paired PP
changed by −5.34% and −2.09%, TPS by −5.74% and −3.27%, and total
request time by +6.15% and +1.59%. Both RSS peaks were below the
111.25 and 216.89 MiB ceilings. The 128 generated IDs matched in
every timed run.

On the short fixture, issuing attention reads through the queue
improved paired PP by 0.77% and TPS by 1.80% against the synchronous
implementation in the same build. On long inputs the pthread GQA
team reached 309.16 TPS in the three-pair comparison. Separate
phase dispatch reached 312.12 TPS. An earlier seven-pair run
returned 309.96 and 312.79 TPS respectively. This variation and
the absence of a consistent model-level gain keep the team opt-in.

On eight ordinary short requests, five passes matched all 201 IDs
per pass. Median aggregate decode was 518.37 TPS mapped, 442.31
TPS with the raw stream and 451.52 TPS with the current pthread
stream. The median of each pass's highest peak was 99.53 MiB for
the pthread stream.

On the 3584-token text-derived input, the earlier OpenMP panel
stream reached 314.29 TPS across two pairs. The pthread stream
reached 310.77 TPS with separate phase dispatch and 311.57 TPS
with the GQA team. Peak RSS was 204.08 and 204.34 MiB at most
for the two pthread modes. All generated IDs matched.

At a fresh GGUF path with the source already cached, `stream-all`
built 197 sidecars. The first token arrived after 307.83 ms, versus
220.54–222.28 ms on reopening and 207.42 ms for the raw control.
The peak on creation was 105.30 MiB. All output IDs matched.

The release build matched 457 generated IDs and 2310 top-1 IDs
on M5 Max, with perplexity 119.9085. Both final prefill logit
vectors matched bit for bit. Two interactive turns matched all 32
IDs and output bytes. The standalone pthread kernel suite also
checks GQA with one KV head and 12 workers, leaving five idle ranges
in softmax and merge. Local CTest passed 176 tests with four platform
skips. ARMv8.0 functional validation under Cortex-A53 QEMU matched
329 generated IDs and 2310 top-1 IDs at perplexity 119.8882.

The timing harness and hashes are in `.model-cache/rss-q41-split`.
The `currentpool` executable is a separate `THREADPOOL=1` build.

```sh
p=.model-cache/rss-q41-split
python3 "$p/bench.py" 256 12 7 before native:0.102 finalpool-w100-cacheall-dual:0.105 currentpool-w100-cacheall-dual:0.105
python3 "$p/bench.py" 3584 12 3 before stream4-cache-dual:0.203 currentpool-w100-cache-dual:0.203 currentpool-w100-gqa-cache-dual:0.203
python3 "$p/bench-natural.py" 5 before native:0.102 currentpool-w100-cacheall-dual:0.105
```

### Separate residency for FFN down projections

`MOTY_Q41_RESIDENT_PREFIX` caps the number of complete FFN layers
kept resident by native budgeted streaming. After reserving the read
windows, the engine uses the remaining budget for the earliest
`down` matrices. Gate and up stay together and share the existing
fused path. The synchronous and queued readers skip resident
projections, so the loaded down matrix remains available for both
prefill and decode. Whole-layer placement is the default. When all
FFN weights fit, the engine keeps all layers resident.

The operation profile used 128 generated tokens on M5 Max with 12
workers. Preparing Q4_1 activations accounted for 2.7% of short
decode and 1.9% of long decode, within the reported projection
times. Normalization, residual updates and sampling together took
under 2% on each input. The foreground FFN read wait took 10.5–11.2%
of short decode and 0.9–1.0% of long decode. Long GQA took
41.3–41.4%. An FFN prototype that kept the same pthread team for
gate/up and down changed short paired TPS by −1.29% across three
pairs. Issuing two FFN reads earlier, during the previous output head,
changed short paired TPS by −0.53% across four pairs.

In-place conversion from row-major keys to four-position tiles at
decode kept peak RSS within 0.03 MiB of the existing long stream.
Paired TPS changed by +0.35% on its own. With simultaneous changes
to GQA block order, merge and worker dispatch, the three-pair
changes ranged from −8.25% to +8.90%. Widening the V strip from
16 to 32 channels changed paired TPS by +0.61%. The wider strip
with lane FMAs across four positions changed it by −2.62%.
All screening runs matched the reference token IDs. These
scheduling and kernel variants remain local experiments.

The release measurements used a pthread build, 12 workers,
`MOTY_POOL_SPIN_US=100`, `MOTY_Q41_PANEL_DUAL=1` and 128
generated tokens. The two short rows use separate sets of seven
alternating pairs. The long rows share one seven-pair series.
The table shows run medians and the highest measured candidate RSS.

| Input tokens | Candidate | Mapped PP / TPS | Candidate PP / TPS | Paired PP / TPS change | Peak RSS (MiB) |
|---:|---|---:|---:|---:|---:|
| 256 | `stream-all`, 0.108 GiB, prefix 2 | 1340.4 / 476.59 | 1289.1 / 465.56 | −3.74% / −2.69% | 111.23 |
| 256 | `stream-all`, 0.107 GiB, prefix 2 | 1341.1 / 451.37 | 1280.4 / 454.84 | −4.29% / −0.04% | 110.25 |
| 3584 | `stream`, 0.203 GiB, whole layers | 1015.5 / 328.88 | 1004.7 / 317.46 | −1.91% / −3.68% | 204.12 |
| 3584 | `stream`, 0.211 GiB, prefix 4, chunk 384 | 1015.5 / 328.88 | 1010.5 / 312.52 | −0.40% / −6.15% | 214.48 |

The 0.107 GiB short configuration improved paired TPS by 3.56%
over the prior 0.105 GiB whole-layer stream. Compared with mapped
weights, its paired TTFT and request time increased by 5.93% and
2.65%. Five passes of eight ordinary prompts with the 0.108 GiB
placement reached 471.95 aggregate TPS, up from 446.60 for the
whole-layer stream and below the mapped 517.92. All 201 generated
IDs matched on every pass. The 0.109 GiB placement exceeded the
short ceiling once, at 111.28 MiB.

On a text-derived 3584-token input the prefix-4 placement reached
314.57 TPS at a maximum of 215.12 MiB. In three pairs, TPS changed
by −2.12% and PP by −1.63% against mapped. All 128 generated
IDs matched. The whole-layer stream remains the faster long option
on the fixed fixture. The selected short and long placements meet
the original RSS ceilings of 111.25 and 216.89 MiB. Speed parity
remains open across the fixed and ordinary workloads.

On Apple the new placement matched all 457 generated IDs and 2310
teacher-forced top-1 IDs, with perplexity 119.9085. The final prefill
logits matched bit for bit on both fixed inputs. Two interactive
turns matched all 32 IDs and output bytes. CTest passed 176 tests
with four platform skips. The budget regression covers a resident
prefix, additional resident down projections, bitwise prefill and
decode outputs, and reuse of the same down pointer across forwards.
The standalone ARMv8.0 suite passed under
`qemu-aarch64 -cpu cortex-a53`. The emulated model matched 329
generated IDs and 2310 top-1 IDs at perplexity 119.8882.

The paired measurements, profile and executable, model and fixture
hashes are in `.model-cache/rss-q41-compute`.

```sh
p=.model-cache/rss-q41-compute
python3 "$p/bench.py" 256 12 7 before currentpool-w100-cacheall-dual:0.105 finalpool-w100-cacheall-dual-prefix2:0.106 finalpool-w100-cacheall-dual-prefix2:0.107
python3 "$p/bench.py" 256 12 7 before currentpool-w100-cacheall-dual:0.105 finalpool-w100-cacheall-dual-prefix2:0.108 finalpool-w100-cacheall-dual-prefix2:0.109
python3 "$p/bench-natural.py" 5 before currentpool-w100-cacheall-dual:0.105 finalpool-w100-cacheall-dual-prefix2:0.108 finalpool-w100-cacheall-dual-prefix2:0.109
python3 "$p/bench.py" 3584 12 7 before currentpool-w100-cache-dual:0.203 finalpool-w100-cache-dual-chunk384-prefix4:0.211
BENCH_TEXT_LONG=1 python3 "$p/bench.py" 3584 12 3 before finalpool-w100-cache-dual-chunk384-prefix4:0.211
python3 "$p/quality.py"
python3 "$p/logit-proof.py"
python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Prefill from resident panels

`MOTY_Q41_PANEL_PREFILL=1` lets FMLAL use the raw-row prefill kernels
on resident Group64 panels. For each four-row panel, the worker
reconstructs the original Q4_1 bytes on its stack and reuses those
bytes for every input token. The workspace is at most 12,800 bytes
per worker for a pair of panels. The path applies to batches of at
least 16 tokens and input widths up to 2560. Wider inputs, smaller
batches and FP32 activations use the existing panel kernels.

The packed 2+2 `gate/up` kernel also loads four scale/minimum pairs
into SIMD lanes, two from each projection. It uses the same parameter
conversion as the four-row kernel and applies both FP32 updates in
the original order. It is selected by the existing
`MOTY_Q41_PANEL_DUAL=1` switch.

Profiling the 256-token stream on M5 Max measured 1.27–1.37 ms
waiting for and binding FFN weights per prefill, out of
197–199 ms. The attention window spent at most 0.04 ms in
foreground wait and binding. The gate/up and down kernels took
about 70% of prefill. Bypassing the full attention window and
expanding attention panels inside the workers remained within
the short ceiling in a four-pair screen, with a +0.43% paired
PP change against the previous stream. Fixed two-row tiles for
four or eight tokens reached 1206.0 and 957.3 PP, against
1280.6 for the previous stream in the three-pair screen.

The release comparison used 12 pthread workers, 128 generated
tokens, warm files and seven alternating pairs on each fixed input.
The short configuration combined panel prefill with
`MOTY_Q41_PP_ATTN=4` and kept two complete FFN layers resident.
The long configuration retained whole-layer placement.

| Input tokens | Placement budget (GiB) | Mapped PP / TPS | Candidate PP / TPS | Paired PP / TPS change | Highest peak RSS (MiB) |
|---:|---:|---:|---:|---:|---:|
| 256 | 0.106 | 1342.3 / 477.37 | 1298.1 / 453.53 | −1.61% / −4.46% | 109.17 |
| 3584 | 0.203 | 989.6 / 325.20 | 984.3 / 312.63 | −0.69% / −3.18% | 204.36 |

Against the previous stream, paired PP improved by 1.56% on the
short configuration and paired TTFT improved by 2.10%. Panel
prefill alone, with the usual short attention path, improved paired
PP by 1.33%. On the long fixture, paired PP improved by 2.16%
and paired TPS changed by −0.39% against the preceding stream.
The 0.107 GiB short control reached 111.33 MiB in the repeat
screen. The 0.106 GiB setting leaves more room under the
111.25 MiB ceiling.

On the 3584-token text fixture, the 0.203 GiB configuration
peaked at 204.36 MiB with paired changes of −0.31% PP and
−2.97% TPS against mapped. Five passes of the eight ordinary
short prompts reached 481.35 aggregate TPS at 0.112 GiB and a
maximum of 109.67 MiB. The 0.108 GiB control with the
previous executable reached 474.67 TPS, and mapped reached
513.75 TPS. Every pass matched all 201 generated IDs. The
0.112 GiB memory measurement applies to the eight ordinary
prompts. The fixed 256-token acceptance uses 0.106 GiB.

At 0.208 GiB with a 512-token automatic chunk, seven fixed-input
pairs gave +1.83% PP and −7.44% TPS against mapped, at a maximum
of 214.56 MiB. Three text-derived pairs gave +2.90% PP and
−15.71% TPS at a maximum of 213.98 MiB. Mapped decode itself
varied from 201.58 to 328.06 TPS in the fixed-input series.
The 512-token setting remains experimental. The release
comparison keeps the 256-token automatic chunk.

The release matched 457 generated IDs and all 2310 top-1 IDs
on M5 Max at perplexity 119.9085. The panel regression compares
single-token decode and 17- and 33-token batches bit for bit
with the raw layout. It uses distinct gate and up codes and
minima, and checks FP32 and FP16 activation selections.
Both fixed-input prefill logit vectors and two interactive
turns matched bit for bit. CTest passed 176 tests with four
platform skips. The ARMv8.0 suite passed under Cortex-A53 QEMU.
The emulated model matched 329 generated IDs and 2310 top-1
IDs at perplexity 119.8882.

The timing harness, profiles and hashes are in
`.model-cache/rss-q41-prefill`.

```sh
p=.model-cache/rss-q41-prefill
python3 "$p/bench.py" 256 12 7 before finalpool-w100-cacheall-dual-prefix2:0.106 release-prepack-w100-cacheall-dual-prefix2:0.106 release-prepack-attn4-w100-cacheall-dual-prefix2:0.106
python3 "$p/bench.py" 3584 12 7 before finalpool-w100-cache-dual:0.203 release-prepack-w100-cache-dual:0.203
BENCH_TEXT_LONG=1 python3 "$p/bench.py" 3584 12 3 before release-prepack-w100-cache-dual:0.203
python3 "$p/bench.py" 3584 12 7 before release-prepack-chunk512-w100-cache-dual:0.208
BENCH_TEXT_LONG=1 python3 "$p/bench.py" 3584 12 3 before release-prepack-chunk512-w100-cache-dual:0.208
python3 "$p/bench-natural.py" 5 before finalpool-w100-cacheall-dual-prefix2:0.108 release-prepack-w100-cacheall-dual-prefix2:0.108 release-prepack-w100-cacheall-dual-prefix2:0.112
python3 "$p/quality.py"
python3 "$p/logit-proof.py"
python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Long-context GQA in the pthread build

On Apple FMLAL with twelve workers and two KV heads, both worker
backends now select the three-head score tile, block-first scoring
and wide merge for row-major FP32 KV. The pthread build dispatches
the four phases separately by default. Each query head keeps its
original FP32 score reduction, full softmax and block-order value
merge. The kernels operate on the existing KV and attention scratch.

Using a first-layer Pocket KV capture and twelve pthread workers,
the median time for GQA at 3584 visible tokens was 34.36 µs with
the previous schedule and 32.63 µs with the three-head schedule.
The median change within the seven alternating series was −5.44%.
Four-position key tiles measured 34.31 µs with pair scoring and
33.70 µs with triple scoring. All schedules matched every context
float across eight visible lengths from 2048 through 3712. At
2047 positions the long path returned control to the caller.
The tiled-key experiments remain in the local harness.

On the full model, two instrumented runs measured each stage
over 128 generated tokens after 3584 input tokens.

| Stage | Previous pthread schedule (ms) | Three-head schedule (ms) |
|---|---:|---:|
| Score | 81.51 | 53.19 |
| Softmax | 21.73 | 22.86 |
| V accumulation | 62.10 | 62.62 |
| Merge | 5.29 | 4.78 |
| Total attention | 174.96 | 146.33 |

Seven release pairs on the fixed 3584-token input improved
paired TPS by 8.23% over the old schedule in the same binary,
with a +0.03% PP change. The release comparison uses the 0.203
GiB weight budget, native panel prefill, 12 workers and 128
generated tokens. Values in the table are run medians except
peak RSS, which is the highest observed candidate value.

| 3584-token input | Pairs | Mapped PP / TPS | Tuned PP / TPS | Paired PP / TPS change vs mapped | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Fixed IDs | 7 | 1028.1 / 329.36 | 1019.3 / 341.45 | −1.46% / +3.60% | 204.44 |
| Text-derived IDs | 5 | 1036.6 / 324.66 | 1020.8 / 342.17 | −1.04% / +5.83% | 204.36 |

All release pairs returned the same 128 generated IDs. The
256-token control peaked at 109.92 MiB in the tuned run. Its
visible length stays below the 2048-position threshold.

With the tuned schedule, the 0.208 GiB budget and
`MOTY_Q41_AUTO_CHUNK=512` also cleared paired prefill parity
on both long fixtures. The original RSS ceiling for either
long fixture is 216.89 MiB.

| 3584-token input | Pairs | Mapped PP / TPS | Tuned PP / TPS | Paired PP / TPS change vs mapped | Peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|
| Fixed IDs | 7 | 1037.8 / 323.92 | 1047.0 / 339.42 | +0.87% / +6.01% | 213.95 |
| Text-derived IDs | 5 | 1037.9 / 322.78 | 1045.6 / 338.93 | +0.64% / +4.97% | 214.72 |

Paired TTFT changed by −0.84% and −0.62%. Paired total request
time changed by −1.34% and −0.83%. Every run matched all 128
generated IDs. These figures use warm files. Short-context
throughput parity remains open.

On the short fixture, the respective medians for 25, 100,
200 and 400 µs of pthread spin were 419.05, 450.70, 444.26
and 444.90 TPS. Mapped reached 471.42 TPS. On eight ordinary
prompts, the same settings at 0.112 GiB gave 403.27, 477.28,
471.07 and 471.34 aggregate TPS, versus 512.29 mapped. All
201 IDs matched in each pass. The timings keep 100 µs for
the short streaming comparison.

The Apple quality suite matched all 457 generated IDs and
2310 top-1 IDs at perplexity 119.9085. Prefill logits
matched bit for bit on both fixed inputs. A separate
same-binary check compared all 127 subsequent decode logit
rows on the 3584-token fixture. All 8,323,072 FP32 values
matched bit for bit between the two GQA schedules. Two
interactive turns matched all 32 generated IDs and output
bytes. CTest passed 176 tests with four platform skips,
and the pthread and ARMv8.0 kernel suites passed.
Cortex-A53 QEMU matched 329 generated IDs and 2310 top-1
IDs at perplexity 119.8882.

The paired timings, profile, KV kernel check and hashes are in
`.model-cache/rss-q41-score`. The previous pthread schedule
can be selected in the same binary with
`MOTY_Q41_GQA_SCORE=0 MOTY_Q41_GQA_BLOCK_FIRST=0 MOTY_Q41_GQA_MERGE=0`.

```sh
p=.model-cache/rss-q41-score
python3 "$p/bench.py" 3584 12 7 before tuned-prepack-oldgqa-w100-cache-dual:0.203 tuned-prepack-w100-cache-dual:0.203
BENCH_TEXT_LONG=1 python3 "$p/bench.py" 3584 12 5 before tuned-prepack-w100-cache-dual:0.203
python3 "$p/bench.py" 3584 12 7 before tuned-prepack-chunk512-w100-cache-dual:0.208
BENCH_TEXT_LONG=1 python3 "$p/bench.py" 3584 12 5 before tuned-prepack-chunk512-w100-cache-dual:0.208
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/decode-logits.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Parallel checksum for Q4_1 panel caches

`MOTY_Q41_CACHE_HASH=parallel` reads and writes version 2 of the
lossless Q4_1 sidecar. It uses four independent 64-bit checksum
chains. The writer carries the next chain across write batches.
Disk identity, shape and payload length are checked against the
source GGUF. A checksum failure selects the original Q4_1 data.
If either gate or up fails, both run from the original layout.
Version 1 is the default and uses the original file suffix.
Version 2 uses `.moty-q41-v2-<offset>` and shares the existing
weight window and panel format.

An isolated 1,290,240-byte panel checksum took 175.75 µs in
version 1 and 44.50 µs in version 2 across seven alternating
series. In three cloned-GGUF pairs on M5 Max, TTFT was 297.01
vs 291.75 ms while creating the sidecars and 214.51 vs 210.05
ms on reopening. The raw GGUF was warm before creation. The
12-worker runs used the fixed 256-token input, 128 generated
tokens and a 0.106 GiB placement budget. All IDs matched.

| Warm workload | Pairs | Mapped TPS | Version 1 TPS | Version 2 TPS | Paired TPS change vs version 1 | Version 2 peak RSS (MiB) |
|---|---:|---:|---:|---:|---:|---:|
| 256 input tokens, 0.106 GiB | 7 | 472.96 | 447.66 | 451.42 | +0.46% | 109.17 |
| Eight ordinary prompts, 0.112 GiB | 7 | 510.88 | 472.15 | 489.17 | +3.82% | 109.67 |
| 3584 input tokens, 0.208 GiB, chunk 512 | 5 | 316.47 | 336.02 | 325.17 | +2.03% | 213.98 |

Rates in the table are run medians and paired changes are the
median of changes within each pair. On 256 inputs version 2
reduced paired TTFT by 2.53% against version 1. Paired PP and
TPS remained 2.60% and 2.77% below the mapped control. On
3584 inputs version 2 paired PP and TPS were 2.48% and 2.63%
above mapped. Each fixed run matched all 128 generated IDs.
The ordinary prompts matched all 201 IDs per pass.

All 197 pairs of sidecar payloads matched byte for byte. The
cache test covers incremental digest updates across uneven
write batches, corruption rejection in both versions and
fallback from a corrupt version 2 gate cache. On Apple the
release matched all 457 generated IDs and 2310 top-1 IDs at
perplexity 119.9085. Both fixed prefill logit vectors and the
output of two interactive turns matched byte for byte.
CTest passed 176 tests with four platform skips. The pthread
and ARMv8.0 kernel checks passed. In Cortex-A53 QEMU the
model matched 329 generated IDs and 2310 top-1 IDs at
perplexity 119.8882.

The short queue prototypes parked idle compute workers, allowed
the caller to take an unclaimed relevant read, or moved input
normalization ahead of the weight wait. Together they gave
+1.96% paired TPS against the unchanged queue across four
pairs. Parking on its own gave −2.37%. Waiting separately
for gate/up and down, recycling a window after gate/up when
down was resident, and a packed 4+4 kernel produced −0.46%,
−0.84% and +0.04% paired TPS respectively. With version 2,
five readers improved paired TPS by 1.52% on the fixed input
and 1.15% across the ordinary prompts against four readers.
The queue and 4+4 variants remain in the local harness.

Builds, timings and hashes are in `.model-cache/rss-q41-wait`.

```sh
p=.model-cache/rss-q41-wait
BENCH_MODEL_OVERRIDE="$p/v2/pocket.gguf" python3 "$p/bench.py" 256 12 7 before release-prepack-attn4-w100-cacheall-dual-prefix2:0.106 release-fast-prepack-attn4-w100-cacheall-dual-prefix2:0.106
BENCH_MODEL_OVERRIDE="$p/v2/pocket.gguf" python3 "$p/bench-natural.py" 7 before release-prepack-w100-cacheall-dual-prefix2:0.112 release-fast-prepack-w100-cacheall-dual-prefix2:0.112
BENCH_MODEL_OVERRIDE="$p/v2/pocket.gguf" python3 "$p/bench.py" 3584 12 5 before release-prepack-chunk512-w100-cache-dual:0.208 release-fast-prepack-chunk512-w100-cache-dual:0.208
python3 "$p/hash-probe.py"
python3 "$p/first-open.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Grouping resident down projections by stream window

With native budgeted streaming and `MOTY_Q41_RESIDENT_PREFIX`,
`MOTY_Q41_RESIDENT_DOWN=slot` fills the resident down set one
window at a time. Layers that rotate through the same window
receive their resident projections together. When the budget
covers every down matrix for that window, its raw down pages
stay untouched during prefill and decode. The allocator still
reserves full-sized raw windows for the synchronous fallback.
The default order visits consecutive layers.
After prefill at 256 tokens and 0.106 GiB, `mincore` counted
176 of 263 resident Darwin pages in the first slot, compared
with 263 of 263 in consecutive order. The other two slots
each held 263 pages in both modes. This accounts for 1.36 MiB
at the 16 KiB page size.

In a short decode profile, waiting for FFN reads took 34.10 ms
with the parallel checksum in one 300.7 ms decode. Binding the
completed reads took 0.38 ms and publishing subsequent reads
took 4.38 ms. Gate/up and down together took 134.30 ms.
Four-window screening, a counting semaphore and a queue with
atomic indices preserved the generated IDs. The confirmation
series measured −0.43% and +0.41% paired fixed-input TPS when
combining the last two queue variants with five readers, slot
placement and a 0.1075 GiB budget. Both remain local experiments.

The release used twelve pthread workers and the parallel checksum.
Each row uses seven alternating pairs against the mapped and
consecutive-layer controls. TPS is the run median and peak RSS
is the largest measured candidate value.

| Warm workload | Placement budget (GiB) | Placement | TPS | Paired TPS change vs consecutive layers | Peak RSS (MiB) |
|---|---:|---|---:|---:|---:|
| 256 input tokens | 0.106 | Consecutive layers | 437.67 | Reference | 109.17 |
| 256 input tokens | 0.106 | Slot order | 446.00 | +0.51% | 108.30 |
| 256 input tokens | 0.109 | Slot order | 452.49 | +3.08% | 110.48 |
| Eight ordinary prompts | 0.112 | Consecutive layers | 477.58 | Reference | 109.66 |
| Eight ordinary prompts | 0.112 | Slot order | 481.31 | +0.73% | 108.31 |
| Eight ordinary prompts | 0.1145 | Slot order | 482.77 | +1.47% | 109.53 |

Mapped gave 460.54 TPS on the fixed input and 494.08 TPS
across the ordinary prompts. The 0.109 GiB slot configuration
gave paired changes of −2.10% PP and −2.50% TPS against mapped.
All six streaming configurations stayed under the 111.25 MiB
short ceiling. Every fixed run matched all 128 IDs and every
ordinary pass matched all 201. The long control at 0.208 GiB
with a 512-token prefill chunk matched all 128 IDs in three
pairs and gave +3.20% paired PP and +5.65% paired TPS against
mapped, with a highest peak of 213.95 MiB.

The small-model regression checks both residence orders,
exact prefill and decode logits, and the stable address of
resident down weights while windows cycle. Apple matched all
457 generated IDs and 2310 top-1 IDs at perplexity 119.9085.
Both fixed prefill logit vectors and the outputs of two
interactive turns matched byte for byte. CTest passed 176 tests
with four platform skips. The standalone pthread and ARMv8.0
kernel tests passed. Cortex-A53 QEMU matched 329 generated IDs
and 2310 top-1 IDs at perplexity 119.8882.

Executables and logs are in `.model-cache/rss-q41-reserve`.

```sh
p=.model-cache/rss-q41-reserve
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
python3 "$p/bench.py" 256 12 7 before slotrelease-fast-prepack-attn4-w100-cacheall-dual-prefix2:0.106 slotrelease-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix2:0.106 slotrelease-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix2:0.109
python3 "$p/bench-natural.py" 7 before slotrelease-fast-prepack-w100-cacheall-dual-prefix2:0.112 slotrelease-bywindow-fast-prepack-w100-cacheall-dual-prefix2:0.112 slotrelease-bywindow-fast-prepack-w100-cacheall-dual-prefix2:0.1145
python3 "$p/bench.py" 3584 12 3 before slotrelease-fast-prepack-chunk512-w100-cache-dual:0.208
python3 "$p/window-pages.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Cross-token FFN reads and parallel prefill rows

`MOTY_Q41_READ_AHEAD=1/2/3` selects how many initial FFN
windows to prepare for the next single-token step. The
Apple reader queue reuses each selected window as soon as
its last layer finishes on the current token. The last
multi-token prefill step can start the same reads before
the output head. When the next call takes multiple tokens,
the engine waits for the staged reads and loads the raw
layout before using the window. Storage stays within the
existing window allocation.

`MOTY_Q41_PREFILL_PAR=1` processes normalization by token
for compatible QAT Q4_1 batches of at least 128 tokens with
FMLAL. With ordinary FP32 KV it also applies query/key
normalization and cached RoPE, and copies the key and value
rows in the same worker pass. The caller prepares the RoPE
table before dispatch. Workers use disjoint KV rows and
share the existing table. Tiled and int8 KV use the
separate attention and KV-store loops.

In the first short prefill screen, the serial control
reached 1307.4 PP. Parallel normalization reached 1323.9,
parallel QK/RoPE/KV reached 1334.6, and their combination
reached 1352.85 at the same 110.28 MiB median peak.
Preparing the next two windows on the ordinary text set
gave 467.73 aggregate TPS versus 450.96 in the initial
four-pass screen. An eight-row panel kernel gave mixed
results for the output head and down projection and
remains in the local harness.

The final validation used twelve pthread workers on M5 Max,
the same GGUF for every run and warm version 2 sidecars.
Fixed workloads generated 128 tokens, and the eight ordinary
prompts generated 201 in total. Rates are configuration
medians. Percentage changes are medians of paired changes
against the mapped executable. RSS is the highest candidate
peak measured by `/usr/bin/time -l`.

| Workload | Pairs | Mapped PP / TPS | Candidate PP / TPS | Paired PP / TPS change | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 256 IDs, parallel prefill and 3 staged windows | 7 | 1349.4 / 471.46 | 1372.4 / 466.66 | +1.70% / −2.11% | 110.42 MiB |
| Eight ordinary prompts, parallel prefill and 2 staged windows | 7 | — / 505.13 | — / 475.31 | — / −6.20% | 109.52 MiB |
| 3584 IDs, streaming with 512-token chunks | 3 | 994.5 / 323.07 | 1022.4 / 339.12 | +2.35% / +5.00% | 214.89 MiB |

On the fixed short workload paired TTFT fell 3.12%. The
previous streaming build gave 1321.3 PP and 456.87 TPS with
a 110.53 MiB peak. On the ordinary prompts the new candidate
gained 1.00% paired TPS against the previous streaming build.
The fixed workloads meet the RSS limits of 111.25 and 216.89 MiB
from the saved original baseline. The remaining short-context
decode losses prevent full acceptance.

The long control used the established streaming configuration
with both new options off. An earlier combined run gained
3.93% paired PP and lost 3.86% paired TPS on the fixed input,
while the text-derived input gained 3.99% PP and 4.48% TPS.
Both options remain opt-in.

Apple validation matched all 457 generated IDs and 2310
top-1 IDs at perplexity 119.9085. The full prefill logits
matched byte for byte on both fixed inputs. Two interactive
turns matched all 32 generated IDs and output bytes. The
unit cases cover staged-window reuse when returning to
prefill and exact KV contents after chunking with a nonzero
prefix. CTest passed 176 tests with four platform skips.
The standalone pthread and ARMv8.0 kernel checks passed.
Cortex-A53 QEMU matched all 329 generated IDs and 2310
top-1 IDs at perplexity 119.8882.

Executables, timing logs and input hashes are in
`.model-cache/rss-q41-ahead`.

```sh
p=.model-cache/rss-q41-ahead
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
python3 "$p/bench.py" 256 12 7 before slotrelease-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix2:0.109 checked-parprefill-lead3-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix2:0.109
python3 "$p/bench-natural.py" 7 before slotrelease-bywindow-fast-prepack-w100-cacheall-dual-prefix2:0.1145 checked-parprefill-lead2-bywindow-fast-prepack-w100-cacheall-dual-prefix2:0.1145
python3 "$p/bench.py" 3584 12 3 before checked-fast-prepack-chunk512-w100-cache-dual:0.208
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Cached FFN panels during prefill

`MOTY_Q41_STREAM_PREFILL=N` allows the bounded FFN reader
to use the same lossless Group64 panels for prefill batches
of at most `N` tokens. The reader verifies each payload on
first use and remembers the result for decode. A missing or
invalid panel follows the same raw GGUF fallback as decode.
Gate and up still use a common layout. A prefill call that
allows panels can consume its already staged batch when an
earlier decode step prepared that batch.

With `MOTY_Q41_PACKED=stream-all`,
`MOTY_Q41_PREFILL_ATTN=panel` also uses the resident attention
panels during prefill. This saves the allocation for reading
raw attention weights. The amount available for FFN residency
increases by the size of that window. The default attention
configuration retains the raw window.

The first five-pass screen on eight ordinary prompts reached
505.46 TPS and 1127.52 PP with both options, a 0.1165 GiB
budget and experimental worker scheduling. Paired changes
against mmap were −0.67% TPS and +2.80% PP. The largest peak
was 109.83 MiB. The earlier streaming configuration reached
487.21 TPS and 1036.27 PP in the same series.

The final build used twelve pthread workers and warm
sidecars. The fixed inputs generated 128 IDs per run,
and the ordinary prompt set generated 201 IDs per pass.
The table gives rate medians and medians of paired changes
against the mapped reference. Peak RSS is the highest
candidate peak among the runs.

| Workload | Pairs | Mapped PP / TPS | Candidate PP / TPS | Paired PP / TPS change | Peak RSS |
|---|---:|---:|---:|---:|---:|
| 256 IDs, 0.1105 GiB, panel FFN and three staged windows | 7 | 1337.0 / 475.73 | 1375.4 / 463.61 | +2.06% / −2.62% | 110.62 MiB |
| Eight ordinary prompts, 0.1165 GiB, panel FFN and attention and two staged windows | 7 | 1120.45 / 510.74 | 1107.18 / 510.31 | −0.71% / −0.16% | 109.84 MiB |
| 3584 IDs, 0.208 GiB, existing 512-token streaming configuration | 3 | 999.7 / 321.29 | 1019.9 / 334.21 | +2.54% / +3.90% | 214.17 MiB |

All generated IDs matched in the timing runs. In the
ordinary prompt series the previous streaming build measured
1032.81 PP and 495.26 TPS. Paired changes against mmap in
that build were −7.94% PP and −3.82% TPS. The fixed short
input still shows a decode penalty against mmap. The
long configuration used the existing path with both new
options off. Dynamic output-head assignment, reader QoS,
and software prefetch of packed weights each gave mixed
results and remain in the local harness.

One prototype reuses the raw-attention prefill window to
keep an additional down matrix resident during decode. In
the seven-pair confirmation it gained 1.03% paired TPS
against its streaming control, at a maximum of 110.78 MiB.
Paired TPS was still 3.98% below mmap in that series. All
128 IDs matched. The prototype remains in the local harness.

Apple validation matched all 457 generated IDs, 2310 top-1
IDs and perplexity 119.9085. Both fixed prefill logit vectors
matched byte for byte. Two interactive turns matched all
32 generated IDs and output bytes. The test fixture also
covers a packed prefill following staged decode. CTest
passed 176 tests with four platform skips. The pthread and
ARMv8.0 kernel suites passed. Cortex-A53 QEMU matched all
329 generated IDs and 2310 top-1 IDs at perplexity 119.8882.

The binaries, input hashes and timing logs are in
`.model-cache/rss-q41-prefetch`.

```sh
p=.model-cache/rss-q41-prefetch
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
s=parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix2
n=parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix2
python3 "$p/bench.py" 256 12 7 before "checked-lead3-$s:0.109" "final-packpp256-lead3-$s:0.1105" "final-packpp256-lead2-$s:0.1105" "final-packpp256-$s:0.1105"
python3 "$p/bench-natural.py" 7 before "checked-lead2-$n:0.1145" "final-packpp64-panelattn-lead2-$n:0.1165" "final-packpp64-panelattn-$n:0.1165"
python3 "$p/bench.py" 3584 12 3 before final-fast-prepack-chunk512-w100-cache-dual:0.208
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Separate readiness for gate/up and down

`MOTY_Q41_SPLIT_DOWN=1` lets Apple's bounded reader
publish gate/up and down readiness separately on
single-token steps. The caller starts gate/up as soon as
the pair is ready and waits for down at its first use.
The split reuses the existing weight windows. A prefill
call consumes any staged scalar reads using their
completion groups before it binds or replaces those
projections. Non-Apple builds retain the synchronous
path.

Five alternating profiles of the 256-input workload with
cached FFN prefill put 89.43 ms in gate/up and 48.74 ms in
down across 128 decode tokens. Foreground waits took
23.09 ms and queue publication took 6.83 ms. In a separate
read-completion trace, 8.01 ms of the wait came from down
finishing after both gate and up and after the caller
arrived. That trace counted 651 late batches.

In a three-pair instrumented comparison, the shared
wait took 24.78 ms. With separate readiness, gate/up
waited 20.38 ms and the later down wait took 3.98 ms
(stage medians). The ordinary prompts showed a clearer
throughput gain in the uninstrumented binary. Releasing
gate/up storage before down, changing the reader count
to three, five or six, and per-worker completion flags
gave mixed results and remain in the harness.

| Workload | Pairs | Streaming TPS | Split TPS | Paired split TPS change | Split PP / TPS change vs mmap | Split peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| 256 IDs, 12 workers | 7 | 437.98 | 445.40 | +1.20% | +2.85% / −3.44% | 110.58 MiB |
| Eight ordinary prompts, 12 workers | 7 | 470.59 | 484.63 | +1.87% | +0.68% / −2.85% | 109.80 MiB |
| 3584 IDs, 12 workers | 3 | 320.29 | 324.41 | +1.29% | +3.34% / +2.92% | 213.91 MiB |
| 1536 IDs, 4 workers | 3 | 213.79 | 212.04 | +0.26% | +2.10% / +2.20% | 178.19 MiB |

Every measured run matched the output IDs. Peak RSS on
the two primary inputs remains below half of the saved
initial peaks, 222.50 and 433.78 MiB. The short-input
throughput gap to the mapped reference remains open.
The split is opt-in.

Apple validation matched all 457 output IDs, 2310 top-1
IDs and perplexity 119.9085. Both final prefill logit
vectors matched byte for byte, as did the outputs of
two interactive turns with 32 generated IDs. The stream
fixture exercises a corrupt cache, staged scalar reads
and a return to both raw and packed prefill. CTest passed
176 tests with four platform skips. The pthread and
ARMv8.0 kernel checks passed. Cortex-A53 QEMU matched
329 generated IDs and 2310 top-1 IDs at perplexity 119.8882.

The tied head's 31.50 MiB panel contains 22937 distinct
scale/minimum pairs. A 16-bit index and one exact table
would save 1.66 MiB for this matrix after including the
table. The same count gives 1.40 MiB for attention and
5.61 MiB across FFN sidecars. These are layout estimates
from the model's actual sidecars.

Logs, binaries and input hashes are in
`.model-cache/rss-q41-profile-panel`.

```sh
p=.model-cache/rss-q41-profile-panel
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
s=packpp256-lead3-parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix2
n=packpp64-panelattn-lead2-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix2
l=fast-prepack-chunk512-w100-cache-dual
python3 "$p/bench.py" 256 12 7 before "release-$s:0.1105" "release-splitwait-$s:0.1105"
python3 "$p/bench-natural.py" 7 before "release-$n:0.1165" "release-splitwait-$n:0.1165"
python3 "$p/bench.py" 3584 12 3 before "release-$l:0.208" "release-splitwait-$l:0.208"
python3 "$p/bench.py" 1536 4 3 before "release-$l:0.178" "release-splitwait-$l:0.178"
python3 "$p/bench.py" 256 12 3 "releaseprof-$s:0.1105" "releaseprof-splitwait-$s:0.1105"
python3 "$p/panel-index-size.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Indexed tied head

`MOTY_Q41_INDEXED_HEAD=1` is an opt-in format for the
tied Pocket embedding and output projection on Apple
FMLAL when `MOTY_Q41_PACKED` selects the head. The cache
replaces each pair of FP16 scale and minimum values with
a 16-bit index and retains all four-bit codes. The table
covers the full index range. The sidecar has its own name,
source identity and payload checksum. The loader can fall
back to the existing panel and raw paths.

The single-token head computes the full vocabulary with
the existing FMLAL accumulation order. Embedding lookup
uses the same stored parameters. Batched or FP32 head
calls restore one panel at a time for the existing
kernel. Verification of the actual generated sidecar
matched all 229376 groups byte for byte against the
existing sidecar. Including the table, the payload fell
from 33030144 to 31457280 bytes, a saving of 1.50 MiB.
The memory budget accounts for the complete indexed file
payload before selecting resident FFN layers.

The earlier screen tested pre-expanded FP32 coefficients,
frequency-ordered indices and separate arrays of codes
and indices. Their decode results varied by workload.
The implementation retains the interleaved codes and
exact FP16 pairs.

| Warm workload, 12 workers | Pairs | Streaming TPS | Indexed TPS | Paired TPS change vs streaming | Indexed PP / TPS change vs mmap | Indexed peak RSS |
|---|---:|---:|---:|---:|---:|---:|
| 256 IDs, 0.1107 GiB, no complete resident FFNs | 7 | 453.11 | 456.14 | +0.90% | +1.57% / −2.53% | 110.34 MiB |
| Eight ordinary prompts, 0.115 GiB | 7 | 497.53 | 502.70 | +0.77% | −1.10% / −0.89% | 108.31 MiB |
| 3584 IDs, 0.207 GiB | 3 | 318.93 | 315.36 | +0.26% | +1.78% / −1.65% | 213.23 MiB |

Rates are medians and changes are medians of the
within-pair ratios. The indexed short allocation keeps
20 down projections resident. The streaming control
keeps two complete FFNs and twelve additional down
projections resident. On the ordinary prompts the
streaming control reached a peak of 110.97 MiB. All
timing runs matched the output IDs.

With the same resident placement, three short-profile
pairs measured 31.38 ms for the existing head and
33.05 ms for the indexed head across 128 output tokens.
The profiled median peak went from 110.47 to 109.00 MiB.
The indexed mode remains opt-in. The short decode gap
to mmap remains open.

A follow-up compared loading the four indices together
with explicit NEON lane loads. The lane-load source
compiled to the same object as the existing source, yet
those two modes measured 474.43 and 458.63 TPS in the
five-pair screen. The word-load variant measured 464.44
TPS. This spread includes ordinary timing variation.
The kernel keeps the existing loads.

Apple validation matched all 457 output IDs and 2310
top-1 IDs, with perplexity 119.9085. Both final prefill
logit vectors and the output of two interactive turns
matched byte for byte. The cache fixture checks every
synthetic weight, scalar and batched logits at both
activation precisions and rejection of a corrupt table.
CTest passed 176 tests with four platform skips. The
pthread and ARMv8.0 kernel suites passed. The Cortex-A53
QEMU model runs matched 329 generated IDs and 2310
top-1 IDs at perplexity 119.8882.

Executables, logs and input hashes are in
`.model-cache/rss-q41-head-index`.

```sh
p=.model-cache/rss-q41-head-index
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
s=splitwait-packpp256-lead3-parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual
n=splitwait-packpp64-panelattn-lead2-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix2
l=splitwait-fast-prepack-chunk512-w100-cache-dual
python3 "$p/bench.py" 256 12 7 before "release-$s-prefix2:0.1105" "final-headidx-$s-prefix1:0.1092" "final-headidx-$s-prefix0:0.1107"
python3 "$p/bench-natural.py" 7 before "release-$n:0.1165" "final-headidx-$n:0.115" "final-headidx-$n:0.1165"
python3 "$p/bench.py" 3584 12 3 before "release-$l:0.208" "final-headidx-$l:0.207"
python3 "$p/check-cache.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Four FFN read windows

`MOTY_Q41_WINDOWS=4` permits an extra read window on
Apple if all four windows fit in the placement budget.
The budget charges the window before selecting resident
layers. The default is three. `MOTY_Q41_READ_AHEAD=4`
can stage four windows for the next scalar token when the
allocation permits it.

The placement screen tested resident gate/up pairs at
the front, back and across the layer stack. Those
allocations exceeded the short RSS limit at the common
budget. Reversing or rotating the existing down-slot order
had mixed results. Two windows reached 415.41 TPS and four
reached 470.21 TPS in the five-pair screen. The follow-up
with five windows exceeded the RSS limit at 111.64 MiB.

Changing the packed GEMV reduction to vector pairwise
sums and unrolling the two halves preserved all generated
IDs. Both changes produced small and mixed throughput
differences against the four-window control. The NEON
kernels retain their existing implementation.

In three instrumented short pairs, the median combined
foreground wait for FFN weights fell from 23.96 to
18.64 ms across 128 output tokens. Five and six readers
did not improve on four readers in the short follow-up.

| Warm workload | Pairs | Windows / ahead | Paired PP / TPS change vs mmap | Peak RSS |
|---|---:|---|---:|---:|
| 256 IDs, 12 workers, 0.1107 GiB | 7 | 3 / 3 | +0.86% / −2.92% | 110.19 MiB |
| 256 IDs, 12 workers, 0.1107 GiB | 7 | 4 / 3 | +1.16% / +0.37% | 110.39 MiB |
| 256 IDs, 12 workers, 0.1107 GiB | 7 | 4 / 4 | +1.68% / +0.02% | 110.39 MiB |
| Eight ordinary prompts, 12 workers, 0.115 GiB | 7 | 4 / 4 | −0.96% / −0.33% | 108.55 MiB |
| Eight ordinary prompts, 12 workers, 0.117 GiB | 7 | 4 / 4 | +1.14% / +0.65% | 109.59 MiB |
| 3584 IDs, 12 workers, 0.208 GiB | 3 | 3 / 0 | +0.78% / +1.85% | 215.67 MiB |
| 3584 IDs, 12 workers, 0.208 GiB | 3 | 4 / 0 | +1.30% / −0.11% | 214.77 MiB |
| 1536 IDs, 4 workers, 0.178 GiB | 3 | 4 / 0 | +3.38% / +8.17% | 178.58 MiB |

Fixed inputs generated 128 tokens per run and the
ordinary requests generated 201 IDs per pass. Every
comparison matched the output IDs. The 0.117 GiB
ordinary result came from a separate budget comparison.
The long workload retains its three-window configuration.

The final audit reran the original saved executable,
the more recent mmap control and the selected streaming
configuration. The GGUF SHA-256 matched the file used
for the initial RSS baseline. Each workload used seven
alternating pairs after warmup.

| Workload | Streaming peak RSS | Paired PP / TPS change vs original | Paired PP / TPS change vs recent mmap |
|---|---:|---:|---:|
| 256 IDs, four windows | 110.44 MiB | +2.63% / −1.14% | +1.57% / −0.98% |
| Eight ordinary prompts, four windows | 109.61 MiB | +0.04% / −0.28% | +0.15% / +0.22% |
| 3584 IDs, three windows | 213.92 MiB | +0.46% / −0.25% | +0.09% / −1.38% |

Against the original executable, paired TTFT fell by
5.55% and 0.68% on the two fixed inputs. Total request
time fell by 1.75% and 0.68%. Both fixed inputs clear
the original RSS targets of 111.25 and 216.89 MiB.
The repeated decode medians still leave the strict
speed target open.

The four-window release matched 457 generated IDs and
2310 top-1 IDs on Apple, with perplexity 119.9085.
Both fixed prefill logit vectors and the output from
two interactive turns matched byte for byte. The C
fixture covers a four-window request under a two-window
budget and reusing four staged reads when switching
from decode to packed prefill. CTest passed 176 tests,
with four platform skips. The pthread and ARMv8.0
kernel checks passed. Cortex-A53 QEMU matched 329
generated IDs and 2310 top-1 IDs at perplexity 119.8882.

The local streamed lookahead prototype also tested
whether batching repeated continuations could close the
gap. Its verification build matched every batched logit
to a scalar step on both fixed inputs. The timing build
accepted 85 of 86 proposals after 256 IDs and 93 of 104
after 3584. At 0.1065 GiB the short workload reached
668.00 TPS at a peak of 109.72 MiB. At 0.208 GiB the
long workload reached 471.98 TPS at 214.28 MiB. These
are three-run medians on the repetitive fixtures. On
the eight ordinary prompts only four proposals were
made per pass and none was accepted. Aggregate medians
were 502.13 TPS without lookahead and 500.91 with it.
Streamed lookahead remains in the local harness.

Logs, executable hashes and fixtures are recorded under
`.model-cache/rss-q41-placement`.

```sh
p=.model-cache/rss-q41-placement
mkdir -p "$p/audit"
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
export BENCH_LOG_DIR="$p/audit"
s=headidx-splitwait-packpp256-parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix0
n=headidx-splitwait-packpp64-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix2
l=splitwait-fast-prepack-chunk512-w100-cache-dual
python3 "$p/bench.py" 256 12 7 original before "release-windows4-lead3-$s:0.1107" > "$p/audit/audit-256.log"
python3 "$p/bench-natural.py" 7 original before "release-windows4-lead4-$n:0.117" > "$p/audit/audit-natural.log"
python3 "$p/bench.py" 3584 12 7 original before "release-$l:0.208" > "$p/audit/audit-3584.log"
python3 "$p/audit-summary.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/quality.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/logit-proof.py"
OMP_WAIT_POLICY=PASSIVE python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### Residual updates and reader wakeups

The short experiments combined the residual addition
and first RMSNorm pass in NEON, keeping the existing
four accumulators and reduction order. Fusing the
attention update alone changed paired decode TPS by
−0.10%. Fusing both residual updates changed it by
+0.66% across seven short pairs. A second experiment
published the next FFN reads before the residual add.
It gave +1.94% paired TPS in the first short screen.
The production-equivalent follow-up against the saved
four-window executable measured −0.54% on the short
fixture, −1.07% on ordinary prompts and +1.26% on the
long fixture. Each workload used seven pairs. The
experiments remain in the local harness.

The logit checker captured every scalar decode step
after prefill. Both prototypes matched the four-window
baseline byte for byte at all 127 steps on each fixed
input and matched every generated ID. Skipping reader
wakeups when nobody was asleep gave +0.62% paired
short TPS but reached 111.38 MiB. Signaling one reader
for a single new task gave +0.55% paired short TPS.

Apple Clang's M5 target and profile-guided optimization
were also tested on the current streaming route. The
profile was trained on the two fixed inputs and the
eight ordinary prompts. Five short pairs produced the
following medians and maximum candidate RSS values.

| Build | Decode TPS | Peak RSS |
|---|---:|---:|
| M1 target, `-O2` | 446.64 | 111.73 MiB |
| M5 target, `-O2` | 445.99 | 110.50 MiB |
| M5 target, `-O3` | 435.49 | 110.55 MiB |
| M5 target, `-O2` with PGO | 436.96 | 110.09 MiB |

The original saved binary reached 467.56 TPS in the
same series. The generated IDs matched in every run.

The shared FFN dictionary was checked against the
current Group64 sidecars for all 84 matrices. It keeps
34547 exact parameter pairs and saves 6021120 bytes
across the packed payloads. In the initial five-pair
screen, indexing only the streamed gate/up matrices
gave 489.98 TPS versus 487.91 with the same executable
using regular panels. Indexing only down or all three
projections gave 477.33 or 478.71 TPS respectively.

In seven follow-up pairs, the regular, indexed gate/up
and indexed dual kernels reached 486.91, 486.46 and
485.43 TPS, at peaks of 110.39, 110.62 and 110.44 MiB.
On the ordinary prompts those modes reached 535.35,
536.40 and 529.82 TPS. The standard panel path already
outpaced the mmap control within this series. Dictionary
transport remains in the local harness.

A separate seven-pair run compared the saved release
directly with the prototype executable with its dictionary
disabled. The release reached 487.52 TPS at a peak of
110.23 MiB, for +3.45% paired TPS against the original.
The disabled prototype reached 488.81 TPS. This confirms
that the faster readings in this session also occur on
the existing route. The earlier audit with a small decode
deficit remains part of the performance record.

Build scripts, logs and logit hashes are in
`.model-cache/rss-q41-resid`. The checks can be rerun
against the saved model and sidecars.

```sh
p=.model-cache/rss-q41-resid
python3 "$p/dump.py"
python3 "$p/check-decode.py"
python3 "$p/verify-ffn-index.py"
python3 "$p/summarize.py" "$p"/{resid-256,early-256,release-256,release-3584,tune-256,ffnindex-256,dualdict-256}.log
python3 "$p/natural-paired.py" "$p/dualdict-natural.log" original
```

### Pthread team publication and decode scheduling

On M5 Max, the instrumented 256-input, 128-output run at 12 threads
issued 18034 parallel regions during decode. The three control runs
requested 419 to 637 wakeups each. The median time to publish jobs and
request wakeups was 5.82 ms. The median time spent waiting after the
caller's own share was 47.53 ms, which also includes useful work by
the other workers. No statically partitioned region had fewer work
items than threads.

Lazy spin timers, semaphores and a shared sleeping-worker mask did
not improve the full-model result in their paired runs. Dynamically
assigning rows in FFN reached 475.65 TPS versus 483.68 TPS with static
assignment. Applying dynamic assignment more broadly reached 472.73
TPS. Smaller teams also slowed decode. These scheduling variants
remain in the local benchmarks.

The team-size experiment exposed a race in the pthread runtime. An
idle worker could read the generation of a job that excluded it,
then read the team size of a succeeding job that included it. It
could then execute and report completion for the same job twice.
The publisher now stores the generation and team size in one atomic
word, and the worker partitions work using the size from that
snapshot. An early version with a plain atomic store missed a
wakeup in one QEMU Cortex-A53 run. The final version retains a
sequentially consistent read-modify-write before checking for
sleeping workers. The resize test alternates team sizes every
region and covers both static and dynamic assignment. The wakeup
test now configures zero spin time directly.

The standalone pthread stress test reproduced missed or duplicate
ranges with the old scheduler at spin settings of 100 and 1000 µs.
The fixed scheduler passed 50000 regions at each of 0, 100 and 1000 µs,
including repeated shutdown and restart. On M5 Max the model checks
matched all 457 generated IDs and all 2310 perplexity top-1 IDs.
Perplexity stayed at 119.9085. Both fixed prefill logit vectors and
the output of two interactive turns matched byte for byte.
Cortex-A53 QEMU matched 329 generated IDs and 2310 top-1 IDs at
perplexity 119.8882. The separate ARMv8.0 pthread stress binary
passed five runs of 50000 regions at each spin setting.

The budget screen used the first atomic-snapshot prototype, the
same GGUF and the saved original binary. Each reported change is
the median of per-pair changes. The highest observed RSS is shown
separately from the median.

| Input and profile | Pairs | PP change | Decode TPS change | Median peak RSS (MiB) | Highest peak RSS (MiB) |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256, budget 0.1107, scheduler comparison | 7 | +1.79% | -3.78% | 110.23 | 112.44 |
| 256, budget 0.1107, budget comparison | 7 | +2.75% | -0.33% | 110.25 | 110.39 |
| 256, budget 0.109 | 7 | +1.85% | -3.89% | 107.77 | 107.92 |
| 256, budget 0.1086 and panel attention | 7 | +0.37% | -2.39% | 108.03 | 108.66 |
| 3584, budget 0.208 | 3 | -0.65% | +1.36% | 213.98 | 214.00 |

The largest short-context RSS exceeded the original half-RSS target
of 111.25 MiB at budget 0.1107. Tightening the budget to 0.109
kept the highest reading 3.33 MiB below that target with a decode
throughput cost. At 0.1107, the old and fixed scheduler reached
464.48 and 464.53 median TPS in the same comparison. The seven paired
ordinary-prompt runs gave -0.96% PP and -1.76% TPS against the
original, with a highest peak of 109.61 MiB. The long-context run
met its RSS target of 216.89 MiB.

The final scheduler was retested in seven alternating pairs on
both fixed inputs and the eight ordinary prompts.

| Input and profile | Original / final max RSS (MiB) | Paired PP change | Paired TPS change | Paired TTFT change | Paired request time change |
| --- | ---: | ---: | ---: | ---: | ---: |
| 256, budget 0.1107 | 224.28 / 111.20 | +4.20% | -0.45% | -6.84% | -2.93% |
| 256, budget 0.109 | 224.28 / 109.39 | +2.92% | +0.47% | -5.42% | -2.87% |
| 3584, budget 0.208 | 434.03 / 214.00 | +0.39% | +10.39% | -0.60% | -0.98% |
| Eight ordinary prompts, budget 0.117 | 205.55 / 109.61 | -1.73% | -3.03% |  |  |

Both fixed inputs clear the original RSS limits in this run. The
short-context TPS result at 0.109 needs replication after the
earlier deficit at this budget. On the ordinary prompts this
profile reduces the peak by 46.7% with a decode throughput loss.

Five additional paired runs tightened the budget for ordinary
prompts. With no resident FFN prefix, four windows and three reads
ahead, budget 0.108 reached a highest peak of 100.95 MiB, down from
205.55 MiB. Paired PP and TPS changed by -0.83% and -0.31%. Keeping
a two-layer prefix at the same budget reached 102.22 MiB with
-0.13% PP and -2.83% TPS. All runs matched the 201 generated IDs.
Seven confirmation pairs at 0.108 with no resident FFN prefix
reached 100.97 MiB with +0.92% PP and -3.97% TPS. Increasing the
budget to 0.1095 and staging four reads narrowed the TPS difference
to -0.83% but raised the highest peak to 104.22 MiB. The stricter
speed condition remains open.

The local harness records the inputs, binary identities, warm runs
and generated IDs. The file `.model-cache/rss-q41-pool/sha256.txt`
identifies the final executables, GGUF and input fixtures. Run from
the repository root with the saved GGUF and warmed sidecars.

```sh
p=.model-cache/rss-q41-pool
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
export BENCH_LOG_DIR="$p/repeat"
mkdir -p "$BENCH_LOG_DIR"
s=windows4-lead3-headidx-splitwait-packpp256-parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix0
n=windows4-lead4-headidx-splitwait-packpp64-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix2
python3 "$p/bench.py" 256 12 7 original "poolfixed-$s:0.1107" "poolfixed-$s:0.109" > "$BENCH_LOG_DIR/bench-256.log"
python3 "$p/bench.py" 3584 12 7 original poolfixed-splitwait-fast-prepack-chunk512-w100-cache-dual:0.208 > "$BENCH_LOG_DIR/bench-3584.log"
python3 "$p/bench-natural.py" 7 original "poolfixed-$n:0.117" > "$BENCH_LOG_DIR/natural.log"
python3 "$p/bench-natural.py" 7 original poolfixed-windows4-lead3-headidx-splitwait-packpp64-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix0:0.108 > "$BENCH_LOG_DIR/natural-tight.log"
python3 "$p/summarize.py" "$BENCH_LOG_DIR"/bench-{256,3584}.log
python3 "$p/natural-paired.py" "$BENCH_LOG_DIR/natural-tight.log" original
python3 "$p/quality.py"
python3 "$p/logit-proof.py"
python3 "$p/interactive.py"
clang -Ic -O2 -DMOTY_THREADPOOL -pthread "$p/resize-stress.c" c/nn/par.c -o "$p/resize-fixed"
for spin in 0 100 1000; do "$p/resize-fixed" "$spin" 50000; done
```

### Shared workspace for short prefill

Pocket now uses the existing prefill workspace for every
multi-token chunk when the workspace is enabled. Attention and
FFN reuse the same allocation even on short prompts. The previous
64-token cutoff sent these prompts through separate heap allocations
for each projection. The existing workspace eligibility checks and
the scalar path still apply.

The 0.108 GiB profile on eight ordinary prompts used four read
windows, three reads ahead, panel attention and no resident FFN
prefix. Three instrumented repeats showed median gate/up and down
waits of about 16 ms per 64-token answer, about 12% of decode time
on the two 64-token answers. The queue experiments with targeted
signals, bounded gate/up priority and foreground help produced
mixed decode results. Batching SwiGLU across four or eight panels
and skipping unused environment lookups in packed GEMV also failed
to show a repeatable gain. None of those experiments is enabled.

In the ordinary prompt comparison after warmup, peak RSS reached
100.44 MiB at 0.108 GiB, versus 100.97 MiB with the old workspace
cutoff and 205.55 MiB with the original executable. The paired
change against the original was -0.79% PP and +4.40% TPS. A
separate seven-pair check reached 100.42 MiB with -1.14% PP and
+2.61% TPS. At 0.109 GiB and four reads ahead, the warmed series
reached 101.67 MiB with -0.79% PP and +3.60% TPS. Each series
compared the same 201 output IDs. The ordinary prompts still
show a small PP deficit in repeated runs.

A further warmed seven-pair screen compared grouped prefill
attention and an earlier panel-expansion threshold. Grouping four
queries improved PP by 2.81% against the new workspace at the
same budget, but decode remained slower than the original in that
series. The default workspace profile itself measured -0.69% PP
and -5.47% TPS, with a peak of 100.47 MiB. Grouping two queries
and enabling panel expansion from 12 tokens did not improve PP.
These screens leave the ordinary-prompt speed criterion open.

The two fixed inputs each generated 128 IDs at 12 threads in
seven alternating pairs. Changes are paired medians against the
original executable. RSS is the highest observation in each group.

| Input tokens | Weight budget (GiB) | Original / final peak RSS (MiB) | PP | TPS | TTFT | Request time |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 | 0.109 | 224.28 / 108.03 | +2.85% | +2.78% | -5.50% | -3.99% |
| 3584 | 0.208 | 434.03 / 213.97 | +3.37% | +3.26% | -3.52% | -3.37% |

Logits match byte for byte at 2, 3, 12, 15, 16, 19, 23, 63 and
64 input tokens against the previous streaming implementation.
The existing workspace test also compares the logits after a
three-token prefix, before the longer chunked prefill and the
incremental decode step. Apple validation matched all 457
generated IDs, 2310 top-1 IDs and perplexity 119.9085. The two
fixed prefill logit vectors matched byte for byte. Two interactive
turns matched all 32 IDs and output bytes. The ARMv8.0 check explicitly enabled the
workspace and matched all 329 generated IDs and 2310 top-1 IDs
under Cortex-A53 QEMU at perplexity 119.8882. The local CTest
suite passed 176 tests with four platform skips.

The commands below use the saved original, the previous streaming
build and a pthread build of the short-prefill workspace change.
The latter is named `smallwork` in this harness. Executable, GGUF
and fixture hashes are in `.model-cache/rss-q41-ordinary/sha256.txt`.

```sh
p=.model-cache/rss-q41-ordinary
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
export BENCH_LOG_DIR="$p/repeat" BENCH_WARMUP=1
mkdir -p "$BENCH_LOG_DIR"
n=windows4-lead3-headidx-splitwait-packpp64-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix0
s=windows4-lead3-headidx-splitwait-packpp256-parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix0
l=splitwait-fast-prepack-chunk512-w100-cache-dual
python3 "$p/bench-natural.py" 7 original "control-$n:0.108" "smallwork-$n:0.108" > "$BENCH_LOG_DIR/natural.log"
python3 "$p/bench.py" 256 12 7 original "control-$s:0.109" "smallwork-$s:0.109" > "$BENCH_LOG_DIR/short.log"
python3 "$p/bench.py" 3584 12 7 original "control-$l:0.208" "smallwork-$l:0.208" > "$BENCH_LOG_DIR/long.log"
python3 "$p/natural-paired.py" "$BENCH_LOG_DIR/natural.log" original
python3 "$p/summarize.py" "$BENCH_LOG_DIR"/{short,long}.log
python3 "$p/short-logits.py"
python3 "$p/quality.py"
python3 "$p/logit-proof.py"
python3 "$p/interactive.py"
python3 "$p/a53-quality.py"
```

### FFN read latency on ordinary prompts

In profiling builds, `ffn_read_ready` now measures the foreground
readiness wait and binding for a streamed layer, including synchronous
fallback. When `down` is read separately, its wait and binding are
reported under `ffn_read_down`. The latter interval is excluded from
`ffn_down` so the table distinguishes read latency from projection
time. Normal builds have no extra timing calls.

The local reader instrumentation measured ten 64-output requests
on M5 Max. The four-window control had a median request decode time
of 137.55 ms. The median gate/up readiness and separate down waits
were 8.84 and 2.58 ms. Per-request mean read service time was
99.17 µs at the median, with 53.64 µs elapsed from enqueue to dispatch.
All four readers were active at enqueue on 60.72% of the jobs, and
40.89% of reads took at least 100 µs. Five windows moved more time
into the queue, from 53.64 to 88.15 µs per job in these profiles.
It cut the median foreground waits to 6.38 and 2.10 ms, but
the extra window had no repeatable throughput benefit without
instrumentation.

Each timing screen used seven alternating pairs after warmup on
the eight ordinary prompts. Every candidate matched all 201 IDs.
Rate changes below compare against the four-window streaming
control in the same screen, at the same 0.108 GiB budget.

| Candidate | Paired TPS change | Peak RSS (MiB) |
| --- | ---: | ---: |
| Reader spin for 3, 8 or 20 µs | -0.39%, -0.95%, -0.31% | ≤100.45 |
| Five / six windows, four reads staged | -0.14% / -4.52% | 100.47 / 101.73 |
| Disable Darwin read-ahead on panel files | -0.82% | 100.44 |
| Process gate/up in one reader task | -7.78% | 100.47 |
| Combine gate/up in one read from a lossless file | -7.36% | 100.45 |
| Five / six readers | +2.03% / +1.68% | 100.47 / 100.47 |
| Five readers, confirmation | +0.00% | 100.48 |

Increasing the readers' Darwin QoS also failed to recover decode
parity and slowed some prefill comparisons. The bundled-cache
probe compared all 28 pairs against the existing panel files
byte for byte. The one-read variant and the single-task variant
both lost the parallel gate/up traffic and were slower than
four separate readers.

Another trial shared FP16 input loads between two four-row panels
in the indexed head, then extended the same approach to other
single-projection panels. The isolated head benchmark matched all
65,536 logits for each of 457 captured inputs. Its median across
the seven per-run medians was 250 µs, versus 248 µs for the
four-row kernel. The full-model head-only change was +1.52% TPS
against the streaming control in its screen and -3.13% against
the original executable. The trials remain in the local harness.

The Qwen engine object compiled without profiling matched the
previous workspace build byte for byte. In seven warmed paired
runs the rebuilt CLI kept all 201 generated IDs and peaked at
100.45 MiB, versus 205.55 MiB for the original executable.
Paired changes against the previous streaming build were
+0.53% PP and +0.18% TPS. Against the original they were -1.30%
and -3.56%. The ordinary-prompt speed condition remains open.

Varying the compute pool spin interval from 40 to 250 µs also
left that condition open. The best change against the 100 µs
control was +1.12% paired TPS at 75 µs, with -0.64% PP and a
peak of 100.45 MiB. In that same series, paired decode throughput
against the original fell by 7.57% at 100 µs and 7.72% at 75 µs.

The two read fields and the projection field appeared separately
in all 24 instrumented ordinary requests. Final validation matched
all 457 generated IDs and 2310 top-1 IDs on Apple at perplexity
119.9085. Cortex-A53 QEMU matched 329 generated IDs and 2310 top-1
IDs at perplexity 119.8882. The Apple short and fixed prefill logit
vectors matched bit for bit. CTest passed 176 tests with four
platform skips.

The saved builds and individual per-request measurements are in
`.model-cache/rss-q41-reader-idle`. To rerun the four-reader
comparison and collect the new operator timings:

```sh
p=.model-cache/rss-q41-reader-idle
python3 "$p/build-current.py"
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
export BENCH_LOG_DIR="$p/current-repeat" BENCH_WARMUP=1
mkdir -p "$BENCH_LOG_DIR"
n=windows4-lead3-headidx-splitwait-packpp64-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix0
python3 "$p/bench-natural.py" 7 original "control-$n:0.108" "current-$n:0.108"
python3 "$p/bench-natural.py" 3 "currentprof-$n:0.108" currentprof
python3 "$p/summarize-profile.py" "$p/profile"
```

### Packed FFN kernel scheduling and exact integer dot products

Each timing comparison used the original GGUF and the 0.108 GiB
ordinary-prompt streaming profile at 12 workers on M5 Max. Seven
alternating pairs followed a warm-up for each variant. All eight
prompts generated the same 201 IDs in every run. All candidates
in this screen stayed below 101 MiB peak RSS.

| Candidate | Paired rate change against the current streaming build |
| --- | ---: |
| Apply all bias FMAs before computing the four FMLAL dot products | +1.08% TPS |
| Interleave FMLAL accumulation for two output rows | +1.41% TPS |
| Prefetch gate/up two or four panels ahead | +1.62% / +1.29% TPS |
| Inline SwiGLU, then unroll Group64 loops by two or seven in separate builds | +0.37% / +0.14% / +0.98% TPS |
| Align packed gate/up, down, or both to output cache lines | -0.61% / -1.44% / -2.43% TPS |
| Alternate resident down windows, beginning with the odd windows | +1.87% TPS initially, +0.26% on repeat |
| Normalize FFN input before waiting for the stream | +1.35% PP, -1.15% TPS |
| Use one, four or seven compute workers only for short scalar GQA | +0.31% / -31.44% / -20.48% TPS |

The initial gate/up microbench matched all output digests for 2688
activation vectors across the 28 real Pocket layers. Its five
repeats gave the same 21 µs median at the timer's 1 µs resolution.
For multi-token inference, unpacking all four rows of one Group64
panel before moving to the next group improved ordinary paired PP
by 2.45% in the first screen and changed it by -0.15% in the follow-up.
The group-major path matched all 4,945,920 output floats byte for
byte in the 23-token kernel fixture, with or without prefetch.
The corresponding whole-model profile reached 100.45 MiB on
ordinary prompts and measured -0.51% PP and -2.47% TPS versus
the original executable in the follow-up. At 256 and 3584 tokens
its peaks were 109.05 and 214.91 MiB. The paired TPS differences
against the original were -3.78% and -0.72%, respectively.

The integer trial assigns a binary exponent to each of the four
existing FP32 accumulator lanes. A conservative bound requires
eight times the sum of absolute integer activations to fit below
`2^24`. Since every centered Q4_1 code has magnitude at most
eight, this bounds each partial sum and its final sum, preserving
the FMLAL dot result before the scale and bias updates. Two signed
bytes per activation and four exponents met the bound for 35,117
of 151,312 blocks. Three balanced signed bytes met it for 140,937
blocks (93.14%). These are from 5,404 captured FFN inputs on the
ordinary prompts. FMLAL handled every other block. For eligible
blocks, 563,748 direct comparisons using Pocket weight bytes
matched the four-lane FMLAL results bit for bit.

In the isolated single-core loop with prepared activations, FMLAL
took 1.509 ns per block at the median, versus 1.506 ns with SDOT
and a live weight transpose. Pretransposing the weights reduced
that SDOT time to 1.421 ns. Timing the full model with the live
transpose and activation preparation reached a peak of 100.45 MiB
and lost 4.30% paired TPS against the current streaming build.
The saved original reached 499.76 median TPS while the current
streaming build reached 485.78 in that series. The constrained
speed condition remains open. These candidates remain local.

```sh
p=.model-cache/rss-q41-kernel-schedule
python3 "$p/natural-paired.py" "$p/finalist/natural.log" original
python3 "$p/summarize.py" "$p/finalist"/{short,long}.log
python3 "$p/exact-dot-coverage.py"
cmp "$p/prefill-control.f32" "$p/prefill-expandgroups.f32"
set -- .model-cache/rss-q41-reader-idle/bundles/*
"$p/exact-dot" "$1" "$p"/ffn-input-*.bin
"$p/exact-dot-bench" "$1" "$p/ffn-input-3.bin"
python3 "$p/natural-paired.py" "$p/dotexact-compare/bench.log" original
```

### Reusable file windows for gate and down

`MOTY_Q41_FILE_WINDOWS=1` enables an Apple Silicon streaming
path for the packed `gate` and `down` projections. Each queue
slot reserves its own address range. Prefill and the first read
use writable scratch in that range. Once a packed tensor has
passed its checksum, the reader can replace that scratch with
a read-only file view at the same address for scalar decode.
It checks the current file length and touches one byte per
page before publishing the job. The same address returns to
writable scratch for raw fallback. The `up` projection uses
the buffered reader. The slot cannot be rebound while the
previous projection still uses it. [Apple mmap(2)](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/mmap.2.html)

The 0.1065 GiB ordinary-prompt profile keeps 14 down projections
resident and streams all 28 gate projections. Once verified, the
42 cold gate and down panels account for 51.68 MiB of payload
that can use file views instead of buffered copies per decode
step.

An isolated read-and-scan test on the warm files measured
18.73 GB/s for buffered reads and 18.34 GB/s for mapping and
unmapping each file. Keeping the views mapped achieved 91.02
GB/s with a 70.52 MiB peak. Applying `MADV_DONTNEED` reduced
throughput to 14.81 GB/s at the same measured peak. These
are measurements on the test host with these files. [Apple madvise(2)](https://developer.apple.com/library/archive/documentation/System/Conceptual/ManPages_iPhoneOS/man2/madvise.2.html)

The first whole-model trials created each file view at a new
address. Mapping every cold FFN projection reached a peak
of 395.69 MiB. Mapping only down and prefaulting from the
readers recovered throughput but still reached 166.50 MiB.
Keeping the address stable and sharing it with the writable
scratch brought the peak back under the memory ceiling. On
ordinary prompts, mapping gate and down reached 552.37 median
TPS at 100.42 MiB between the original at 525.00 TPS and the
down-only build at 508.11 TPS. The paired gain against the
original was 5.45% on that screen. Mapping all three projections
achieved 518.33 TPS at 99.89 MiB. Each comparison matched the
201 generated IDs.

The release build with the file-length check was retimed after
warm-up. PP, TPS, TTFT and request time changes are medians of
paired changes against the original binary. RSS is the highest
peak observed in each series.

| Workload | Pairs | Original / file-window peak RSS (MiB) | PP | TPS | TTFT | Request time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 256 IDs, 12 workers | 7 | 224.28 / 109.52 | +1.96% | +5.56% | -4.96% | -5.23% |
| 3584 IDs, 12 workers | 7 | 434.02 / 216.17 | -2.50% | +2.87% | +2.33% | +1.77% |
| Eight ordinary prompts, 12 workers, 0.1065 GiB | 7 | 205.55 / 100.33 | -1.63% | +4.74% |  |  |
| Eight ordinary prompts, 12 workers, 0.1078 GiB | 7 | 205.55 / 101.72 | -2.18% | +6.20% |  |  |
| 1536 IDs, 4-worker performance control | 3 | 299.14 / 178.27 | +2.02% | +4.80% | -2.15% | -2.57% |

The two fixed 12-worker inputs and ordinary prompts met their
original half-RSS ceilings with faster decode in these runs.
Prefill on the long input and the ordinary prompts remains
below the original, and the long request still takes longer
in total under the settings in this table. The file window route
is opt-in.

Reducing the ordinary-prompt read-ahead depth from three slots
to two or one kept the small prefill deficit in another seven
pairs. The three-slot profile reached -0.56% PP and +4.86% TPS
at a peak of 100.22 MiB. The two- and one-slot variants reached
+3.49% and +2.38% TPS respectively, with prefill changes of
-0.70% and -1.45%.

For 3584 inputs, enabling `MOTY_Q41_PREFILL_PAR=1` with
512-token chunks and the same 0.207 GiB budget brought prefill
back to within 0.10% of the original in seven further pairs.
Decode improved by 3.41%. TTFT and total request time fell by
0.09% and 0.56%. The largest file-window peak was 214.81 MiB,
against 434.06 MiB for the original in that series. The
384-token parallel profile used up to 211.09 MiB in the
four-pair screen. The ordinary-prompt prefill condition
remains open.

The Q4_1 budget fixture verifies that the queued gate and down
views become read-only, then checks raw fallback after a gate/up
layout mismatch and after shortening a down cache file. The
updated fixture passed in the regular test suite and in the
Apple pthread build. CTest passed 176 tests with four platform
skips. Apple validation matched 457 generated IDs, 2310 top-1
IDs, perplexity 119.9085, the short and fixed prefill logits,
and two interactive turns. ARMv8.0 in Cortex-A53 QEMU matched
329 generated IDs, 2310 top-1 IDs and perplexity 119.8882.

The local comparisons and reproduction scripts are under
`.model-cache/rss-q41-file-windows`. To time the saved original
against the new route on the same warm GGUF, run these commands
from the repository root.

```sh
p=.model-cache/rss-q41-file-windows
python3 "$p/build-current.py"
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
export BENCH_LOG_DIR="$p/repeat" BENCH_WARMUP=1
mkdir -p "$BENCH_LOG_DIR"
n=windows4-lead3-headidx-splitwait-packpp64-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix0
s=windows4-lead3-headidx-splitwait-packpp256-parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix0
l=splitwait-fast-prepack-chunk512-w100-cache-dual
python3 "$p/bench-natural.py" 7 original "current-filewin-$n-attn4:0.1065" > "$BENCH_LOG_DIR/natural.log"
python3 "$p/bench.py" 256 12 7 original "current-filewin-$s:0.108" > "$BENCH_LOG_DIR/short.log"
python3 "$p/bench.py" 3584 12 7 original "current-filewin-$l:0.207" > "$BENCH_LOG_DIR/long.log"
python3 "$p/bench.py" 3584 12 7 original "current-filewin-$l-parprefill:0.207" > "$BENCH_LOG_DIR/long-parallel.log"
python3 "$p/natural-paired.py" "$BENCH_LOG_DIR/natural.log" original
python3 "$p/summarize.py" "$BENCH_LOG_DIR"/{short,long}.log
python3 "$p/quality.py"
python3 "$p/logit-proof.py"
python3 "$p/a53-quality.py"
```

### Follow-up on prefill at half RSS

The four-worker control also fits below half its original peak
with a 0.145 GiB budget, four FFN windows, the indexed head,
resident attention panels and 512-token prefill chunks. After
warm-up, the seven alternating pairs on 1536 input IDs matched
all 128 generated IDs. Peak RSS fell from 300.17 to 147.16 MiB.
Median paired PP rose 7.28% and TPS rose 3.20%. TTFT fell 6.96%
and total request time fell 6.33%. A 384-token alternative
peaked at 142.25 MiB in the three-pair screen.

For the eight ordinary prompts, the instrumented low-RSS
build spent a median 69.4% of prefill in the two FFN projection
stages and 21.3% in Q/K/V and output projections. Median
gate/up readiness time rounded to 0.0% of this phase. Moving
the wait for down behind gate/up screened well in three pairs.
In seven further pairs, ordinary-prompt PP fell 1.52% and TPS
rose 5.32% versus the saved original. Starting the reads ahead
of embedding evaluation as well gave -0.63% PP and +6.18% TPS.
The unchanged file-window build reached -0.23% PP and +6.13%
TPS in the same comparison. The reader schedule was retained.

An additional seven-pair comparison measured first output and
request completion from process launch. At 0.1065 GiB the
file-window build reduced the maximum from 205.55 to
100.38 MiB. It matched all 201 generated IDs, and the paired
changes were -2.02% PP, +5.90% TPS, -13.75% TTFT and -9.58%
total time. Checksumming sidecars directly in a mapped prefill
window gave -1.23% PP, +5.08% TPS, -13.75% TTFT and -9.44%
total time at 100.62 MiB. The existing buffered verification
remains in use.

`MOTY_Q41_SHORT_PREFILL=1` enables two FMLAL changes for
batches of 8–32 tokens. Activation packing runs on the
calling thread for fewer than 24,576 activation values. An
eight-token panel tile accumulates one output row at a time
with eight FP32 vector accumulators. The tile reads the
original packed weights and applies each block in the same
order. The kernel uses the existing path for partial tiles
and token-interleaved activation layouts. The regular
packed expansion option can still take precedence.

The two changes together reached +2.06% PP, +5.86% TPS,
-14.86% TTFT and -9.94% total time at 100.38 MiB in seven
paired prototype runs on the ordinary prompts. Using the
packing change alone with a 65,536-value threshold reached
-0.06% PP. Eight-token tail and mixed-layout QKV checks
compare both short-prefill settings against the raw
projection in FP16 and FP32 modes.

These configurations met both the memory and speed targets on
M5 Max with warm version 2 sidecars. Each row uses seven
alternating pairs. RSS is the maximum across the series.
Changes in rates and times are paired medians.

| Workload | Budget (GiB) | Original / candidate peak (MiB) | PP | TPS | TTFT | Request time |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Eight ordinary prompts, 12 workers | 0.1065 | 205.55 / 100.42 | +0.58% | +6.66% | -15.06% | -10.47% |
| 256 IDs, 12 workers | 0.108 | 224.28 / 107.81 | +0.49% | +6.09% | -4.01% | -5.07% |
| 1536 IDs, 4 workers | 0.145 | 300.17 / 147.16 | +7.28% | +3.20% | -6.96% | -6.33% |
| 3584 IDs, 12 workers | 0.202 | 434.06 / 206.38 | +1.15% | +2.41% | -1.32% | -1.93% |
| 3584 IDs, 12 workers | 0.207 | 434.06 / 214.06 | +3.33% | +2.98% | -3.44% | -3.63% |

Every pair produced the same 128 IDs per fixed input and the
same 201 IDs across the ordinary prompts.

The release build matched 457 generated IDs and all 2310
top-1 IDs on M5 Max, with identical perplexity of 119.9085.
The first-token logits were bitwise identical on 15 short
lengths between 2 and 64 tokens and on the fixed 256- and
3584-token inputs, including both long budgets. Tests also
covered chunk overrides and fallbacks for FP32 activations,
int8 KV, buffered loading and disabled fused projections.
CTest passed 176 cases, with four platform skips. The
binary and GGUF hashes are under
`.model-cache/rss-q41-prefill-gap/final/sha256.txt`.
The ARMv8.0 build passed the Q4_1 and GQA test executable in
Cortex-A53 QEMU. It reproduced 329 generated IDs and all 2310
top-1 IDs with perplexity 119.8882. The Apple FMLAL results
use physical hardware. Cortex-A53 measurements here cover
correctness.

```sh
p=.model-cache/rss-q41-prefill-gap
export BENCH_MODEL_OVERRIDE=.model-cache/rss-q41-wait/v2/pocket.gguf
export BENCH_LOG_DIR="$p/repeat"
mkdir -p "$BENCH_LOG_DIR"
python3 "$p/build-current.py"
n=shortpp-filewin-windows4-lead3-headidx-splitwait-packpp64-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix0-attn4
s=shortpp-filewin-windows4-lead3-headidx-splitwait-packpp256-parprefill-bywindow-fast-prepack-attn4-w100-cacheall-dual-prefix0
b=filewin-windows4-lead3-headidx-splitwait-packpp384-panelattn-parprefill-bywindow-fast-prepack-w100-cacheall-dual-prefix0-attn4
l=shortpp-filewin-splitwait-fast-prepack-chunk512-w100-cache-dual-parprefill
python3 "$p/bench-natural-e2e.py" 7 original "current-$n:0.1065" > "$BENCH_LOG_DIR/natural.log"
python3 "$p/bench.py" 256 12 7 original "current-$s:0.108" > "$BENCH_LOG_DIR/short.log"
python3 "$p/bench.py" 1536 4 7 original "control-$b-chunk512:0.145" > "$BENCH_LOG_DIR/control.log"
python3 "$p/bench.py" 3584 12 7 original "current-$l:0.207" "current-shortpp-$b-chunk512:0.202" > "$BENCH_LOG_DIR/long.log"
python3 "$p/summarize.py" "$BENCH_LOG_DIR"/{short,control,long}.log
python3 "$p/short-logits.py"
python3 "$p/logit-proof-tight.py"
python3 "$p/quality.py"
python3 "$p/a53-quality.py"
```

## Token and perplexity checks

With greedy decoding, EOS enabled and a 64-token limit, native Q4_1 matched all 201 output tokens of the previous float32 path across eight prompts. The packed-copy loader (`MMAP=0`) matched the mapped loader on all eight prompts. The previous int8 path matched four complete sequences.

| Prompt | Native Q4_1 tokens | Native sequence matches int8 |
|---|---:|---|
| What is the capital of France? | 8 | Yes |
| 法国的首都是哪里？ | 7 | Yes |
| Name three colors. | 14 | Yes |
| Explain how a rainbow forms, step by step. | 64 | No |
| Write a haiku about a mountain lake. | 23 | No |
| What is 17 times 9? Answer with a number only. | 3 | Yes |
| Return a JSON object with keys city and country for Tokyo. | 18 | No |
| 请用三句话解释为什么天空是蓝色的。 | 64 | No |

Teacher-forced evaluation over 2,310 prediction positions used the same [reference passage](../../tools/ref/ppl_text_long.txt). The Hugging Face baseline ran the [official snapshot](https://huggingface.co/taobao-mnn/MNN-Pocket-0.3B/tree/48fae9c0be6ec505e80f36f69b8ba21097f9f822) in float32 with Transformers 4.57.1. Native Q4_1 matched the previous GGUF float32 top-1 ID at every position and the previous int8 top-1 ID at 2,092 positions.

| Weight mode | Perplexity | Top-1 vs. Hugging Face float32 |
|---|---:|---:|
| Hugging Face float32 | 114.8258 | 2,310 / 2,310 |
| Previous GGUF float32 | 119.8882 | 1,901 / 2,310 |
| Native Q4_1 | 119.8881 | 1,901 / 2,310 |
| Previous GGUF int8 | 118.2237 | 1,844 / 2,310 |

Reproduce the comparison with `pocket_snap` set to the official snapshot directory.

```sh
python tools/ref/hf_ppl.py "$pocket_snap" pocket-ref.json tools/ref/ppl_text_long.txt
for bits in 0 -1 8; do
  GGUF="$pocket_gguf" QBITS="$bits" THREADS=4 PPL=pocket-ref.json \
    PPL_OUT="pocket-$bits.top1" ./c/moty
  python tools/ref/cmp_ppl.py pocket-ref.json "pocket-$bits.top1"
done
cmp pocket-0.top1 pocket--1.top1
```
