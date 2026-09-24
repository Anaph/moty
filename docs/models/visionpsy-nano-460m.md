# VisionPsy-Nano-460M-Flash — the decoder in moty

The decoder of VisionPsy-Nano-460M-Flash is a plain `LlamaForCausalLM`
(32 layers, hidden 960, 15/5 heads, intermediate 2560, vocab 49218, tied
head). moty runs it with its Llama-family engine (`qwen`, `model_type`
`llama`) and injects the NPU image encoder's rows at the image token
(`EMBEDS=` / `moty_generate(..., embed_token)`), with no model-specific code.
The decoder's config has no `image_token_id`, so the placeholder id (49152)
is passed explicitly: `EMBEDS_TOKEN=49152`, `--image-token 49152` in the
tools, the last argument of `api/examples/moty_cycle.c`.

Evaluation set: agent B's 8 tiles (64 NPU encoder rows [64][960] each), the
real prompt (78 ids, 64 of them image placeholders), 64 greedy tokens. The
reference is HF f32 on the same rows.

## Exactness and quantization

| decoder | leading tokens = HF f32 (mean /64, full) | teacher-forced top-1 | size |
|---|---|---|---|
| moty f32, NPU rows | 64.0, 8/8 | 100.0 % | 1.45 GB |
| moty f32, HF encoder rows | 64.0, 8/8 | 100.0 % | 1.45 GB |
| int8 everywhere (Q8R4, RTN) | 33.2, 1/8 (board: 27.5, 1/8) | 96.9 % | 574 MB |
| int4 everywhere (Q4R4, RTN, head int4) | 4.2, 0/8 (board: 4.2) | 70.7 % | 393 MB |
| int4, head int8 | 4.2, 0/8 | 78.5 % | 416 MB |
| int4 MLP, attention + head int8 | 6.5, 0/8 | 87.1 % | 456 MB |
| int4 gate/up only, rest int8 | 11.5, 0/8 | 90.4 % | 495 MB |
| GPTQ int4, head int8 (text calibration) | 0.5, 0/8 | 73.4 % | 416 MB |
| GPTQ int4 gate/up only, rest int8 (text calibration) | 0.0, 0/8 | 83.6 % | 495 MB |

x86 moty unless marked "board" (the aarch64 kernels round differently: the
board's greedy answers match the reference as far as the table says, on the
board itself). `tools/ref/moty_vleval.py`, `tools/a53/board_lead.py`.

This decoder is far more sensitive to int4 than LFM2.5-VL's backbone. The
one-group sweep (`tools/ref/hf_vlq.py ... --image-token 49152 --greedy
'f32+4=<group>'`, everything else f32, teacher-forced top-1): head 0.846,
layers 28–31 0.863, `down_proj` 0.875, `v_proj` 0.881, `gate/up_proj`
0.895, layers 20–23 0.902, other 4-layer blocks 0.926–0.953, `q/k_proj`
0.926, `o_proj` 0.947. The damage is spread over every layer; no int8
subset of a few tensors rescues int4. With RTN, int8 everywhere is the
recipe to use.

GPTQ (`pack_r4.py --method gptq --calib tools/ref/calib_text.txt`, 2048
text tokens) is worse than RTN here (73.4 % vs 78.5 %, 83.6 % vs 90.4 %):
its answers stay on topic ("This is a black and white photograph ... a
cluttered indoor space") but leave the reference at the first word. GPTQ
calibrated on image sequences (`--calib-tiles`) needs live tiles other
than the 8 evaluation tiles; it has not been run.

## Speed on the board (RV1126B, 4× Cortex-A53, brownai running)

`tools/a53/vl_decoder_bench.sh`: `moty_cycle` (libmoty), open → 64 tokens →
close, 4 threads (decode 3), 2 reps × 2 cycles, tile t00.

| | moty int8 everywhere | moty int4 RTN | RKLLM w8a8 (agent B, board 132) |
|---|---|---|---|
| decoder load | 6.1–6.8 s | 2.2–5.8 s | 5.3 s unsigned / 11.1 s signed |
| prefill, 78 tokens | 2.36–2.79 s (28–33 tok/s) | 1.86–2.32 s (34–42 tok/s) | 2.2 s |
| decode | 8.6–9.6 tok/s | 13.0–15.0 tok/s | ≈3.3 tok/s |
| TTFT from open (decoder only) | 8.5–9.6 s | 4.1–8.1 s | ≈16.5 s per cycle |
| TTFT, model open | 2.4–2.8 s | 1.9–2.3 s | — |
| peak RSS | 466 MB | 300 MB | — |
| answer quality | 96.9 % teacher-forced | 70.7 % (not usable) | not measured here |

The load time depends on the page cache (1 GB board, a 574 MB file): the
int4 container's second open came mostly from cache. The RKLLM TTFT is B's
per-cycle figure; how much of it is the NPU encoder is B's to state. After
close and `moty_release_scratch()` RSS falls to 11–22 MB, threads to 1.

## Answer length and end of turn

Measured on x86 (the boards were unavailable), the 8 tiles, greedy.
VisionPsy writes long descriptions: with "Describe the image." HF f32 ends
its turn (`<|im_end|>`, id 2 = the config's EOS) only after 296 / 327 / 460
tokens (t10 / t00 / t08); any `max_new_tokens` of 32–64 cuts the answer in
mid-sentence. moty stops cleanly at the same EOS (int8, t10: 397 tokens,
last id 2, the marker not printed). The instruction decides the length
(`tools/ref/vl_answer_len.py`, prompts through the model's chat template;
"Describe the image." reproduces the node's 78 ids):

| instruction | prompt ids | HF f32 answer tokens | ends at EOS | moty int8: ends at EOS / identical to HF f32 |
|---|---|---|---|---|
| Describe the image. | 78 | ≥ 160 on every tile | 0/8 | — (see above) |
| Briefly describe the image. | 80 | 106, ≥ 160 on 7 tiles | 1/8 | — |
| Describe the image in one sentence. | 81 | 36–56 | 8/8 | 8/8 / 3/8 (25–57 tokens) |
| What is in the image? Answer in one short sentence. | 86 | 17–28 | 8/8 | 8/8 / 6/8 (17–25 tokens) |

The int8 answers that differ are rewordings of the same content, e.g. t00:
HF "A cluttered desk with various electronic devices, cables, and a box
labeled "GOTSCHLICH" visible in the background." — int8 "The image shows a
cluttered desk with various electronic devices, cables, and a power strip."
(the monitor/mouse this scene's answers mention come from the f32 model
itself).

Recommendation for a node that shows the latest answer: the short-question
contract with `max_new_tokens` 48 (the longest answer seen is 28 tokens;
the cap only guards against a runaway), or the one-sentence instruction
with 80. With the rates measured on board B (78-token prefill ≈ 2.4 s,
decode 8.6–9.6 tok/s) that is ≈ 4.3–5 s from the generate call to a
complete answer, against ≈ 9.5 s for 64 tokens of a truncated description
— an estimate from those rates, not an end-to-end board measurement.

## Decode: what did not pay

- Two-stage head (`head_topk`, sign-bit copy of the head + exact rows for
  the top K) extended to a Q8R4 head: the greedy output left the full
  head's at the first decode token on 8/8 tiles. Stage-1 recall of the
  true argmax on HF hidden states (512 positions): 54 % (K 256) – 60 %
  (K 512) with moty's 3-bit activations, 97 % at best (8-bit, K 4096) —
  the hidden states have strong outlier dimensions (median max/mean 14.5)
  and a sign-bit copy of this head cannot rank them. Not kept.
## VL-aware GPTQ and mixed layouts (46 calibration tiles)

Agent B's 46 calibration tiles (`from-b/visionpsy/calib`: 22 night IR +
24 day colour, one desk; none of the 8 evaluation tiles), calibration on
the f32 greedy answers of each tile with the image positions masked out
of the Hessians, tiles only:

    pack_r4.py --method gptq --calib "" --calib-tiles <calib dir> --calib-prompt <78 ids> \
        --image-token 49152 <decoder> <out> --variants "<out>=lm_head*;..."

Quality on x86 (the 8 evaluation tiles). "describe": the node's prompt,
64 tokens against HF f32 (leading tokens, teacher-forced top-1); "short
question": the recommended contract, answers ending at EOS (17–28 tokens),
teacher-forced top-1 and answers identical to HF f32. Speed on board B
(brownai running, 4 threads, decode on 3, 64 tokens, 2 repetitions).

| layout | MB/token | describe: teacher-forced / leading | short q.: teacher-forced / identical | decode tok/s (board) | RSS |
|---|---|---|---|---|---|
| int8 RTN (recommended) | 384.4 | 96.9 % / 33.2 | 98.8 % / 6/8 | 9.47–9.74 | 400 MB |
| int8 VL-GPTQ | 384.4 | 97.7 % / 38.6 | 96.9 % / 3/8 | (same format) | 400 MB |
| int4 gate/up, rest int8, VL-GPTQ | 305.8 | 94.3 % / 22.1 | 88.8 % / 1/8 | 11.00–11.29 | 324 MB |
| int4 MLP, attention + head int8, VL-GPTQ | 266.5 | 93.2 % / 21.0 | 84.5 % / 0/8 | 12.29–12.51 | 285 MB |
| int4, head int8, VL-GPTQ | 227.2 | 90.4 % / 13.9 | 85.1 % / 0/8 | 11.78–13.72 | 247 MB |
| int4 gate/up, RTN | 305.8 | 90.4 % / 11.5 | 85.1 % / 0/8 | — | — |
| int4 MLP, RTN | 266.5 | 87.1 % / 6.5 | 82.0 % / 0/8 | — | — |
| int4, head int8, RTN | 227.2 | 78.5 % / 4.2 | 82.0 % / 0/8 | — | — |

VL-aware calibration lifts every int4 layout by 4–12 points (describe),
text-only GPTQ had lowered them. It does not reach int8: the best int4
layout (gate/up) is 94.3 % on the describe prompt and 88.8 % on the short
contract, for +16–19 % decode speed. int8 stays the recommendation; the
int4 layouts are a quality trade to choose knowingly. GPTQ on int8 helps
the describe prompt (the calibration answers are descriptions) but not the
short contract.

## Decode threads and CPU (board B, int8, brownai running)

Prefill 78 tokens on 4 threads 2.35–2.8 s in every row; decode 64 tokens.
CPU per generate from getrusage (`moty_cycle`: "cpu X s over Y s"); the
decode phase's cores = (CPU − ~4 × prefill s) / decode s.

| THREADS_DECODE | pool spin 1000 µs | spin 0 | decode cores busy | tok/s per core |
|---|---|---|---|---|
| 1 | 5.17–5.66 tok/s | 5.62–5.63 | ~0.9 | ~5.9 |
| 2 | 8.15–8.61 | 8.15–8.46 | ~1.7 | ~4.9 |
| 3 | 9.55–9.66 | 8.88–9.00 | ~2.7 | ~3.5 |
| 4 | 7.59–7.85 | 7.52–8.50 | ~3.1 | ~2.6 |

Four decode threads are slower than three (the detector's core preempts
one). At 3 threads the int8 decode streams 384 MB × 9.66 ≈ 3.7 GB/s,
77–86 % of the board's measured 4.3–4.8 GB/s read roofline. For a board
that also runs the camera pipeline, `threads_decode` 2 gives 8.2–8.6 tok/s
(−12 %) on ~1.7 cores instead of ~2.7; `pool_spin_us` 0 saves a little CPU
at up to −7 % decode.

End to end, the short-question contract (86-id prompt, `max_new_tokens`
48, stop at EOS, decode on 3): 4.29–5.42 s from the generate call to the
complete answer on all 8 tiles (prefill 2.56–2.75 s, 17–25 tokens), against
~9 s for 64 tokens of the description prompt.


## One answer end to end, and lookahead decoding

Board B, brownai running, int8 container, the short-question contract
(86 ids with 64 image rows, stop at EOS, `max_new_tokens` 48), 4 threads
for the prefill; two passes over the 8 evaluation tiles (`la_check`, the
API as a node uses it). Per stage:

| stage | plain | with lookahead (`draft_k` 2) |
|---|---|---|
| open (container v2, mmap, page cache) | 0.17–0.20 s | same |
| prefill, 86 tokens | 2.52–2.92 s (one outlier 3.55 s) | same |
| decode 17–25 tokens, `threads_decode` 3 | 1.65–2.50 s | 1.04–2.01 s |
| decode, `threads_decode` 2 | 1.81–2.72 s | 1.38–2.41 s |

Lookahead (docs/api.md, `moty_sampling.draft`): the draft is the previous
answer on the same scene (here: the previous tile's answer, frames ~2.5 min
apart). Sums over 7 tiles (the first has no previous answer):

| `threads_decode` | plain decode | `draft_k` 2 | answers identical | `draft_k` 3 | identical | CPU per answer (plain → `draft_k` 2) |
|---|---|---|---|---|---|---|
| 3 | 13.60 / 13.82 s | 9.82 / 9.68 s (1.38–1.43x) | 7/7, 7/7 | 8.20 / 8.30 s | 6/7 | 105.2 → 94.4 s (−10 %) |
| 2 | 15.08 / 14.86 s | 12.20 / 12.14 s (1.22–1.24x) | 7/7, 7/7 | 10.64 / 10.66 s | 6/7 | 97.9 → 89.3 s (−9 %) |

`draft_k` 2 keeps the answers bit-identical (a 3-token forward runs the decode
kernels); `draft_k` 3 is faster but changed 1 of 7 answers (t02, the 4-token
GEMM tile's float order). Recommended for a camera node: pass the previous
answer as `draft` with `draft_k` 2; with `threads_decode` 2 (≈ 1.7 cores
during decode) the decode then runs at about the plain 3-thread speed.

Measured cost of one forward after the 86-token prompt (int8, 3 threads):
1 token 99–103 ms, 3 tokens 168–180 ms, 4 tokens 181–185 ms, 5 tokens
217–221 ms, 9 tokens 351–365 ms — the weights are read once per forward,
so a few extra positions cost little. A separate small draft model does not
apply here: it would not see the image rows.

What is left: the prefill is now the largest stage (≈ 55–60 % of an answer
after the open); its int8 GEMM runs near what the tile allows on the cores
the detector leaves free, and coarser activation scales cost quality
(docs/performance.md 5.15, 5.16).
