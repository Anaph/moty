# rknn_encode — a Rockchip NPU vision encoder in front of a moty text model

Optional tool, not part of moty's build and not linked into any engine:
it runs a vision encoder + projector compiled to `.rknn` on the Rockchip
NPU and writes its output rows in the format moty's `EMBEDS=` takes (raw
little-endian f32 `[N][hidden]`). The text model then runs in moty with
`EMBEDS=` + `PROMPT_IDS=` (docs/performance.md §5.9–5.10).

## Build

`librknnrt.so` is loaded with `dlopen` at run time; only Rockchip's API
header is needed at build time. It is not included here (Rockchip's
license); take `rknn_api.h` from rknn-toolkit2
(`rknpu2/runtime/Linux/librknn_api/include/`) at the runtime version of
the board (`strings librknnrt.so | grep "librknnrt version"`).

    aarch64-linux-gnu-gcc -O2 -I$RKNN_INCLUDE rknn_encode.c -ldl -o rknn_encode

Dynamic glibc (for `dlopen`); the binary needs GLIBC_2.34.

## Run

    RKNN_LIB=/path/librknnrt.so ./rknn_encode model.rknn image.ppm embeds.f32 [reps]

- `image`: uint8 RGB at the model's input size, binary PPM (P6) or raw
  `H*W*3` bytes; the model's own normalisation is used (input UINT8 NHWC).
- prints the model's I/O, init and encode time (best/mean over `reps`),
  and MemAvailable before / with the model loaded / after.

## LFM2.5-VL-450M

Encoder: `lfm2vl_vision_fp16_rv1126b.rknn` (vision tower + projector,
fp16, input 512×512 RGB, output 256×1024; built with rknn-toolkit2 2.3.2).
The NPU model has a fixed 512×512 input, so it covers the processor's
single-tile case (one 512×512 tile → 256 image tokens), not HF's
multi-tile path for larger images. `prep_image.py` resizes an image to
512×512 (bilinear, as the HF processor's resize) and writes the PPM;
`tools/ref/hf_vl_ref.py` gives the matching prompt ids.

    python prep_image.py photo.jpg tile.ppm
    ./rknn_encode lfm2vl_vision_fp16_rv1126b.rknn tile.ppm img.f32
    SNAP=<LFM2.5-VL container> QBITS=4 EMBEDS=img.f32 PROMPT_IDS=ids.json ./lfm2

`vl_pipeline.sh` runs both steps on a board and reports the encoder time,
the time to first token, decode speed and memory.

## Making the encoder (LFM2.5-VL-450M)

| file | env | what |
|---|---|---|
| `export_lfm2vl_encoder.py <snap> <out.onnx> [--image img]` | host venv (torch, transformers) | the HF vision tower + projector as one ONNX graph for a 512×512 tile; `--image` checks it against HF `get_image_features` |
| `convert_rknn.py <onnx> <out.rknn> [--target rv1126b] [--check tile.ppm rows.f32]` | rknn venv | fp16 `.rknn`, normalisation folded in (mean = std = 127.5); `--check` compares the toolkit simulator with reference rows |
| `npu_contention.sh` | board | detector and VL encoder alone and at once, NPU load samples |

The rknn venv is separate from the host venv: `rknn-toolkit2==2.3.2` pins
torch 2.4.0, numpy 1.26.4 and onnx 1.16.1, and needs
`opencv-python-headless` (the default opencv wheel wants libGL). The full
recipe with the results: [docs/models/lfm2.5-vl-450m.md](../../docs/models/lfm2.5-vl-450m.md).
