"""Projected image rows (the LM's image-token inputs) from HF transformers
f32 for a vision-language snapshot, one 512x512 tile per image: the whole
frame resized to 512x512 (bilinear, aspect not kept — what a camera
pipeline with a fixed-shape NPU encoder does), so the processor makes a
single tile.

  hf_vl_rows.py <snapshot> <out_dir> [--gray] <image> ...

Writes <out_dir>/<image stem>[_gray].f32, raw little-endian f32 [N][hidden]
(moty EMBEDS=, hf_vlq.py tiles). --gray: luminance replicated to RGB (an
IR night frame), written in addition to the colour tile."""
import argparse, os
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("snap"); ap.add_argument("out"); ap.add_argument("--gray", action="store_true")
ap.add_argument("images", nargs="+")
a = ap.parse_args()
os.makedirs(a.out, exist_ok=True)
proc = AutoProcessor.from_pretrained(a.snap)
model = AutoModelForImageTextToText.from_pretrained(a.snap, dtype=torch.float32).eval()
for path in a.images:
    base = Image.open(path).convert("RGB").resize((512, 512), Image.BILINEAR)
    variants = [("", base)] + ([("_gray", base.convert("L").convert("RGB"))] if a.gray else [])
    for suf, img in variants:
        msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
        inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
        f = model.model.get_image_features(pixel_values=inp["pixel_values"], spatial_shapes=inp["spatial_shapes"],
                                           pixel_attention_mask=inp["pixel_attention_mask"], return_dict=True).pooler_output
        f = torch.cat(f, 0).float()
        out = os.path.join(a.out, os.path.splitext(os.path.basename(path))[0] + suf + ".f32")
        f.numpy().astype("<f4").tofile(out)
        print(out, tuple(f.shape), flush=True)
