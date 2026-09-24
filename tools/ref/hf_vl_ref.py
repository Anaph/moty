"""HF transformers reference for a vision-language snapshot (LFM2-VL and
other *ForConditionalGeneration models whose language model takes the
projected image features at image-token positions).

  hf_vl_ref.py <snapshot> <image> <prompt> <out_dir> [ngen=32]

Writes to <out_dir>:
  prompt_ids.json   {"ids": [...]} the processor's expansion of the chat
                    (image placeholders included) -> moty PROMPT_IDS=
  ref.json          {"prompt_ids", "full_ids"}: greedy decode, f32 -> REF=
  embeds.f32        projected image features, raw little-endian f32
                    [N][hidden], N = number of image-token positions
                    -> moty EMBEDS= (EMBEDS_TOKEN = config image_token_id)
  logits.f32        f32 logits of the last prompt position (compare with
                    moty REF_LOGITS= through cmp_logits-style max |d|)
  answer.txt        the greedy answer text
  meta.json         shapes, image token id, token counts"""
import json, os, sys
import numpy as np, torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor
torch.set_grad_enabled(False)

snap, image, prompt, out = sys.argv[1:5]
ngen = int(sys.argv[5]) if len(sys.argv) > 5 else 32
os.makedirs(out, exist_ok=True)
proc = AutoProcessor.from_pretrained(snap)
model = AutoModelForImageTextToText.from_pretrained(snap, dtype=torch.float32).eval()
msgs = [{"role": "user", "content": [{"type": "image", "image": Image.open(image).convert("RGB")},
                                     {"type": "text", "text": prompt}]}]
inp = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt")
ids = inp["input_ids"]
img_tok = model.config.image_token_id
feats = model.model.get_image_features(pixel_values=inp["pixel_values"], spatial_shapes=inp["spatial_shapes"],
                                       pixel_attention_mask=inp["pixel_attention_mask"], return_dict=True).pooler_output
feats = torch.cat(feats, 0).float()
n_img = int((ids == img_tok).sum())
assert feats.shape[0] == n_img, (feats.shape, n_img)
feats.numpy().astype("<f4").tofile(os.path.join(out, "embeds.f32"))
lo = model(**inp).logits[0, -1].float()
lo.numpy().astype("<f4").tofile(os.path.join(out, "logits.f32"))
gen = model.generate(**inp, max_new_tokens=ngen, do_sample=False, num_beams=1)
full = gen[0].tolist(); pl = ids.shape[1]
answer = proc.tokenizer.decode(full[pl:], skip_special_tokens=True)
json.dump({"ids": ids[0].tolist()}, open(os.path.join(out, "prompt_ids.json"), "w"))
json.dump({"prompt_ids": ids[0].tolist(), "full_ids": full}, open(os.path.join(out, "ref.json"), "w"))
open(os.path.join(out, "answer.txt"), "w").write(answer + "\n")
json.dump({"image": os.path.basename(image), "prompt": prompt, "prompt_tokens": pl, "image_token_id": img_tok,
           "image_tokens": n_img, "embeds_shape": list(feats.shape), "embeds_dtype": "float32 little-endian",
           "generated": len(full) - pl}, open(os.path.join(out, "meta.json"), "w"), indent=1)
print("prompt tokens", pl, "image tokens", n_img, "embeds", tuple(feats.shape))
print("answer:", answer)
