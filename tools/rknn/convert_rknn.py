"""ONNX vision encoder -> .rknn (fp16, no quantization) for a Rockchip NPU,
with the image normalisation folded in: the board feeds uint8 RGB NHWC.

  convert_rknn.py <encoder.onnx> <out.rknn> [--target rv1126b]
                  [--mean 127.5 --std 127.5] [--check tile.rgb rows.f32]

--check runs the toolkit's simulator on a 512x512x3 uint8 RGB tile (raw or
the PPM of tools/rknn/prep_image.py) and
compares its output rows with reference rows (raw f32, e.g. HF's
get_image_features written by tools/ref/hf_vl_rows.py): cosine per row.
Environment: rknn-toolkit2 in its own venv (tools/rknn/README.md) — not
the host venv, the toolkit pins old torch/numpy/onnx."""
import argparse
import numpy as np
from rknn.api import RKNN

ap = argparse.ArgumentParser()
ap.add_argument("onnx"); ap.add_argument("out")
ap.add_argument("--target", default="rv1126b")
ap.add_argument("--mean", type=float, default=127.5); ap.add_argument("--std", type=float, default=127.5)
ap.add_argument("--check", nargs=2, metavar=("TILE_RGB", "ROWS_F32"))
a = ap.parse_args()

r = RKNN(verbose=False)
r.config(mean_values=[[a.mean] * 3], std_values=[[a.std] * 3], target_platform=a.target)
assert r.load_onnx(model=a.onnx) == 0, "load_onnx"
assert r.build(do_quantization=False) == 0, "build"
assert r.export_rknn(a.out) == 0, "export_rknn"
print("wrote", a.out)
if a.check:
    raw = open(a.check[0], "rb").read()                    # raw RGB or PPM: the pixels are the last bytes
    tile = np.frombuffer(raw[-512 * 512 * 3:], dtype=np.uint8).reshape(1, 512, 512, 3)
    assert r.init_runtime() == 0, "init_runtime (simulator)"
    out = np.asarray(r.inference(inputs=[tile], data_format=["nhwc"])[0], np.float64)
    ref = np.fromfile(a.check[1], dtype="<f4").astype(np.float64)
    out = out.reshape(-1, out.shape[-1])                 # [rows][hidden]
    ref = ref.reshape(out.shape)
    cos = (out * ref).sum(1) / np.linalg.norm(out, axis=1) / np.linalg.norm(ref, axis=1)
    print(f"simulator vs reference: {out.shape}, cos min {cos.min():.4f} mean {cos.mean():.5f}, "
          f"rel L2 {np.linalg.norm(out - ref) / np.linalg.norm(ref):.4f}")
r.release()
