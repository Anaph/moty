"""Teacher-forced HF reference on a fixed text.

  hf_ppl.py <snapshot> <out.json> [text=ppl_text.txt] [max_tokens]

out.json: {"ids", "argmax", "ppl"} — the ids feed the engines' PPL= mode
directly; cmp_ppl.py compares PPL_OUT argmax files against "argmax"."""
import json, sys, torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer
snap, out = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
ids = tok(open(sys.argv[3] if len(sys.argv) > 3 else "ppl_text.txt").read(), return_tensors="pt").input_ids
if len(sys.argv) > 4: ids = ids[:, :int(sys.argv[4])]
with torch.no_grad():
    lo = model(ids).logits[0].float()
lp = torch.log_softmax(lo, -1)
nll = -lp[:-1].gather(1, ids[0, 1:, None]).squeeze(1)
am = lo[:-1].argmax(-1).tolist()
json.dump({"ids": ids[0].tolist(), "argmax": am, "ppl": math.exp(nll.mean().item())}, open(out, "w"))
print("n", ids.shape[1], "ppl", math.exp(nll.mean().item()))
