# LFM2.5-VL-450M on a Cortex-A53 board with an NPU

[`LiquidAI/LFM2.5-VL-450M`](https://huggingface.co/LiquidAI/LFM2.5-VL-450M) end to
end on a Rockchip RV1126B camera board (4× Cortex-A53, no dot product, 1 GB
RAM, its own vision workload running): the vision tower + projector on the
NPU (RKNN, fp16), the language model in moty on the CPU (int4 with int8 for
the sensitive tensors). This page is the recipe from the HF snapshot to what
runs on the board, the scripts that produced every number, and the results.
Background and more numbers: [performance.md §5.9, §5.10, §5.14](../performance.md).

| | |
|---|---|
| snapshot | `LiquidAI/LFM2.5-VL-450M` revision `fc6221ca597f3315e4f82fc2df606783267b34ba` (`model.safetensors` sha256 `2f6deb5dd43707de5cfe3c59470d3bccf4c3112a810a74570499f4728d412eea`) |
| NPU encoder | 512×512 RGB tile → 256 × 1024 f32 rows; fp16 `.rknn`, rknn-toolkit2 2.3.2, librknnrt 2.3.2 on the board |
| language model | moty `lfm2` engine; recommended container: round-to-nearest Q4R4 + Q8R4 for `lm_head` and layers 10–15 (286 MB/token) |
| result on the board | encode 1.5 s, prefill of the 272-token image prompt ~6.5 s, decode 11.5 tok/s, LM peak RSS ~400 MB; answers without the invented objects of the text-tuned int4 (below) |

Everything here takes paths and hosts as parameters; nothing proprietary is
in the repository (no Rockchip headers, libraries or model files — the
toolkit is installed from Rockchip's releases, `rknn_api.h` taken from it at
build time, see [`tools/rknn/README.md`](../../tools/rknn/README.md)).

## 1. Environments

- **host venv** (reference, export, quantization study): Python 3.11, torch
  2.14 (CPU), transformers 5.17, numpy 2.4, safetensors 0.8, onnx 1.23,
  Pillow.
- **rknn venv** (conversion only; the toolkit pins old versions, keep it
  separate): `rknn-toolkit2==2.3.2`, torch 2.4.0, numpy 1.26.4, onnx 1.16.1,
  `opencv-python-headless` 4.10 (the toolkit imports cv2; the non-headless
  wheel needs libGL).
- **cross compiler**: `aarch64-linux-gnu-gcc` 12 (static moty binaries);
  `rknn_encode` is linked dynamically (it `dlopen`s the board's librknnrt,
  glibc ≥ 2.34).

## 2. Vision encoder → ONNX → RKNN

    # host venv: the HF vision tower + projector as one graph for a 512x512 tile,
    # checked against HF get_image_features on a real image
    python tools/rknn/export_lfm2vl_encoder.py $SNAP enc.onnx --image photo.jpg
    #   wrapper vs HF get_image_features: (256, 1024), max|d| 6.90e-03, cos min 1.00000000
    # reference rows for the simulator check, and the tile as the NPU sees it
    python tools/ref/hf_vl_rows.py $SNAP rows photo.jpg          # rows/photo.f32
    python tools/rknn/prep_image.py photo.jpg tile.ppm            # bilinear 512x512, as the processor
    # rknn venv: fp16, normalisation (x/255-0.5)/0.5 folded in (mean = std = 127.5)
    python tools/rknn/convert_rknn.py enc.onnx lfm2vl_vision_fp16_rv1126b.rknn --target rv1126b \
        --check tile.ppm rows/photo.f32
    #   simulator vs reference: (256, 1024), cos min 0.9930 mean 0.99991, rel L2 0.0143

Checked from a clean checkout of this branch: the ONNX export is
byte-identical run to run; the `.rknn` bytes are not (the compiler embeds
build metadata — 53 k of 194 MB differ between two builds), but its output
on the board is: the rebuilt model gives rows with the same sha256 as the
earlier board run, which were bit-identical to an independent conversion
by another team.

On the board:

    RKNN_LIB=<path>/librknnrt.so ./rknn_encode lfm2vl_vision_fp16_rv1126b.rknn tile.ppm rows.f32 3
    # rknn_encode: init 888 ms | encode best 1531.9 ms mean 1536.3 ms (3 runs) | 262144 floats
    # MemAvailable: -425 MB while loaded, returned at process exit

Preprocessing on the board is only "resize the whole frame to 512×512, RGB,
uint8" — the normalisation is inside the `.rknn`. A camera frame is one
tile (HF would split a large image into 2–10 tiles; a fixed-shape NPU model
does not): feed HF the same 512×512 image when comparing.

Accuracy of the NPU rows (fp16 on the NPU) against HF f32: cosine mean
0.990, min 0.61 on the test tile. This alone changes answers: on the live
tiles below the f32 language model on NPU rows matches HF (on HF's rows)
for 7–64 of 64 tokens. Quantization of the language model is judged
against the f32 LM **on the same NPU rows**.

## 3. Language model container

    # host venv, from the HF snapshot (no calibration data needed)
    python tools/ref/pack_r4.py $SNAP lfm2vl-lm --method rtn --q8 "lm_head.weight,model.layers.1?.*"

`model.layers.1?.*` is layers 10–15 of 16 (the globs use moty's names;
`model.language_model.X` of the VL snapshot is `model.X`). The container
keeps the vision tensors untouched; moty reads only the language model.
Load and run (CLI, or the library API):

    SNAP=lfm2vl-lm QBITS=4 THREADS=4 THREADS_DECODE=3 TEMP=0 NGEN=64 \
      EMBEDS=rows.f32 PROMPT_IDS=prompt_ids.json ./lfm2

`prompt_ids.json` is the processor's expansion of the chat
(`<|image_start|>` + 256 × `<image>` + `<|image_end|>` + text, 272 ids for
"Describe the image."): `tools/ref/hf_vl_ref.py` writes it, together with
the HF f32 greedy answer and first logits:

    python tools/ref/hf_vl_ref.py $SNAP photo.jpg "Describe the image." ref_out 64

`tools/rknn/vl_pipeline.sh` runs encoder + LM on the board and reports the
time to first token.

### Layouts measured (33 live tiles)

Reference: the f32 LM on the same NPU rows. HF fake-quant with moty's int8
activations (`tools/ref/hf_vlq.py`); finalists confirmed in moty on x86
(`tools/ref/moty_vleval.py`); decode on the board (272-token prompt, 64
tokens, 4 threads / decode on 3, medians of 3).

| layout | MB/token | decode tok/s | KL | leading tokens = f32 (of 64) | moty: leading / teacher-forced | tiles inventing a computer set |
|---|---|---|---|---|---|---|
| f32 weights, int8 activations (ceiling) | — | — | 0.0003 | 59.2 | — | 0 |
| Q4R4, round-to-nearest | 199.4 | 15.35 | 0.052 | 20.4 | 21.5 / 93.3 % | 12 |
| GPTQ (text) + Q8R4 layers 0–1 (text recommendation) | 217.7 | 14.06 | 0.095 | 15.3 | 11.2 / 84.2 % | 24 |
| GPTQ (text + image sequences) | 199.4 | 15.35 | 0.044 | 28.6 | 24.0 / 92.9 % | 12 |
| RTN + Q8R4 head | 232.9 | 13.93 | 0.041 | 25.4 | — | 7 |
| RTN + Q8R4 layers 8–15 | 270.7 | 11.96 | 0.021 | 39.0 | — | 3 |
| **RTN + Q8R4 head + layers 10–15** | 286.4 | **11.49** | 0.015 | **42.5** | **42.0 / 96.4 %** | **0** |
| GPTQ (text + image) + Q8R4 head + layers 10–15 | 286.4 | 11.49 | 0.014 | 29.8 | 35.6 / 95.4 % | 21 |
| Q8R4 everywhere | 376.6 | 10.00 | 0.0005 | 55.0 | — | 3 |

Why this layout and not the text one: on image inputs the sensitive tensors
are the head, FFN w2 and the last layers (layers 0–1, the text recipe's
choice, barely matter), and GPTQ calibrated on text doubles the error of
plain rounding; every GPTQ variant, even calibrated on image sequences,
invents a monitor / keyboard / mouse on more tiles than its round-to-nearest
twin. Details: performance.md §5.14.

## 4. Evaluation (what produced the table)

    # the 33-tile study (hours on a desktop CPU); TILES = NPU rows of live tiles,
    # CALIB_TILES = HF rows of other images (hf_vl_rows.py --gray <images>), disjoint
    SNAP=$SNAP PROMPT_IDS=prompt_ids.json TILES=tiles,live CALIB_TILES=cal ./tools/ref/vl_eval.sh
    # moty itself on x86 for the finalists: HF f32 references once, then per container
    python tools/ref/moty_vleval.py refs $SNAP prompt_ids.json refs.json tiles/*.f32 live/*.f32
    python tools/ref/moty_vleval.py run prompt_ids.json refs.json c/lfm2 lfm2vl-lm 8
    #   lfm2vl-lm  tiles 33  lead mean  42.0/64 min 13 full 3  teacher-forced  96.4%

`vl_eval.sh` runs `hf_vlq.py` three times (round-to-nearest layouts,
text-calibrated GPTQ, VL-calibrated GPTQ) and counts invented objects
(`tools/ref/vl_halluc.py`). One-group sensitivity on image inputs:
`hf_vlq.py $SNAP prompt_ids.json tiles 'f32+4=layers\.14\.' 'f32+4=head' ...`.

## 5. Board runs

    # host: chunked, sha256-verified copies (tools/a53/xfer_verified.sh), static rssrun + lfm2
    # target: interleaved repetitions, memory sentinel, medians
    REPS=3 PROMPT_FILE=prompt.txt ./run_configs.sh \
      "vl-q4;./lfm2;SNAP=vl-q4 QBITS=4 THREADS=4 THREADS_DECODE=3 EMBEDS=rows.f32 PROMPT_IDS=prompt_ids.json" \
      "vl-rec;./lfm2;SNAP=lfm2vl-lm QBITS=4 THREADS=4 THREADS_DECODE=3 EMBEDS=rows.f32 PROMPT_IDS=prompt_ids.json" \
      > vl.log 2>&1
    python3 tools/a53/parse_runs.py vl.log
    # NPU sharing with the board's detector
    RKNN_LIB=... DET_MODEL=detector.rknn DET_BYTES=1566720 VL_MODEL=lfm2vl_vision_fp16_rv1126b.rknn \
      VL_IMAGE=tile.ppm ./tools/rknn/npu_contention.sh

Results on the board (brownai running):

| | |
|---|---|
| encoder | init 0.9–1.8 s, encode 1.52–1.53 s |
| LM load (container) | 3.1–3.8 s |
| prefill, 272-token image prompt | 37–47 tok/s (5.8–7.3 s) |
| decode, recommended layout | 11.5 tok/s (plain int4: 15.4) |
| time to first token | ≈ 13.5 s cold (encoder + LM load + prefill), ≈ 8 s with the LM loaded |
| memory | LM peak RSS 399 MB (plain int4 316 MB); encoder −425 MB MemAvailable while loaded — run it as its own step before the LM |
| NPU sharing | the detector (73 ms/frame alone) takes 125.5 ms/frame while the VL encoder runs (+72 %, frames queue); the encoder +2 % |

## 6. Known limits

- **One scene.** The 33 live tiles come from one fixed indoor camera in IR
  night mode over an hour (noise, exposure, the on-screen clock change);
  calibration used other images (brownai test images, colour and grey).
  Scene variety was checked only on 16 local HF-encoded tiles (same ranking).
- The "computer set" count is a word-set difference against the f32 answer;
  "screen" can be legitimate in this scene (on-screen clock).
- The NPU model has a fixed 512×512 input: one tile per frame, no HF-style
  multi-tile split.
- NPU contention: a detector that needs steady latency should schedule the
  VL encoder between its frames.
