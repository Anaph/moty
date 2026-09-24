"""Export the LFM2.5-VL vision tower + projector as one ONNX graph for a
fixed 512x512 tile (the NPU model: input [1,3,512,512] normalised as HF's
processor does, output [1,256,hidden] = the rows that replace the 256
image-token ids).

  export_lfm2vl_encoder.py <snapshot> <out.onnx> [--image img.png]

The wrapper calls the HF modules themselves (patch embedding, the position
embeddings resized to the 32x32 grid, encoder, post layernorm, projector),
so the graph computes exactly HF's get_image_features for a single 512x512
tile; with --image it checks that on a real image (resized to 512x512, the
same single-tile path) and prints the cosine against HF. Environment: the
host venv (torch + transformers; tools/rknn/README.md)."""
import argparse
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("snap"); ap.add_argument("out")
ap.add_argument("--image", help="check the wrapper against HF on this image (512x512 single tile)")
a = ap.parse_args()

vl = AutoModelForImageTextToText.from_pretrained(a.snap, dtype=torch.float32).eval()
vt = vl.model.vision_tower; vt = getattr(vt, "vision_model", vt); proj = vl.model.multi_modal_projector
emb = vt.embeddings
ps, G = emb.patch_size, 512 // emb.patch_size                  # 16 px patches, 32x32 grid
pe = emb.position_embedding.weight.reshape(emb.position_embedding_size, emb.position_embedding_size, -1)
pos = emb.resize_positional_embeddings(pe, torch.tensor([[G, G]]), G * G)[0]

class Enc(torch.nn.Module):
    def __init__(s):
        super().__init__()
        s.register_buffer("pos", pos.clone())
        s.pe = emb.patch_embedding; s.enc = vt.encoder; s.ln = vt.post_layernorm; s.proj = proj
    def forward(s, x):                                      # [1,3,512,512], (x/255 - 0.5) / 0.5
        p = x.reshape(1, 3, G, ps, G, ps).permute(0, 2, 4, 3, 5, 1).reshape(1, G * G, ps * ps * 3)
        h = s.pe(p) + s.pos
        h = s.enc(inputs_embeds=h, attention_mask=None).last_hidden_state
        h = s.ln(h)
        return s.proj(h.reshape(1, G, G, -1)).reshape(1, G * G // 4, -1)

enc = Enc().eval()
for p_ in enc.parameters(): p_.requires_grad_(False)       # traced as weights, not as constants needing grad

if a.image:
    img = Image.open(a.image).convert("RGB").resize((512, 512), Image.BILINEAR)
    x = torch.from_numpy((np.asarray(img, np.float32) / 255.0 - 0.5) / 0.5).permute(2, 0, 1)[None]
    proc = AutoProcessor.from_pretrained(a.snap)
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": "x"}]}]
    inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
    ref = vl.model.get_image_features(pixel_values=inp["pixel_values"], spatial_shapes=inp["spatial_shapes"],
                                      pixel_attention_mask=inp["pixel_attention_mask"], return_dict=True).pooler_output
    ref = torch.cat(ref, 0).double().numpy()
    y = enc(x)[0].double().numpy()
    cos = (y * ref).sum(1) / np.linalg.norm(y, axis=1) / np.linalg.norm(ref, axis=1)
    print(f"wrapper vs HF get_image_features: {y.shape}, max|d| {np.abs(y - ref).max():.2e}, cos min {cos.min():.8f}")
else:
    x = torch.zeros(1, 3, 512, 512)
torch.onnx.export(enc, x, a.out, input_names=["image"], output_names=["embeds"], opset_version=17, dynamo=False)
print("wrote", a.out)
