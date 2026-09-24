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
recipe to use. Calibrated int4 (GPTQ on image sequences) has not been
tried yet on this decoder.

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
