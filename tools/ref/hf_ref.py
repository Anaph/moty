"""HF transformers f32 reference for moty validation.

  hf_ref.py <snapshot> <out.json> [ngen]

out.json: per prompt the prompt_ids, greedy full_ids, decoded text and the
last-prompt-position logits (cmp_logits.py). Next to it, <out>.case<N>.json
holds {prompt_ids, full_ids} for the engines' REF= greedy validation."""
import json, sys, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
snap, out, ngen = sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 32
tok = AutoTokenizer.from_pretrained(snap)
model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
prompts = [
    "The capital of France is",
    "def fibonacci(n):\n    \"\"\"Return the n-th Fibonacci number.\"\"\"\n",
    "Москва — столица России. Население города составляет",
    "Question: What is 17 * 23?\nAnswer:",
]
res = {"model": snap, "cases": []}
with torch.no_grad():
    for p in prompts:
        ids = tok(p, return_tensors="pt").input_ids
        o = model(ids)
        lo = o.logits[0, -1].float()
        top = torch.topk(lo, 20)
        g = model.generate(ids, max_new_tokens=ngen, do_sample=False, num_beams=1)
        res["cases"].append({"prompt": p, "prompt_ids": ids[0].tolist(), "full_ids": g[0].tolist(),
                             "text": tok.decode(g[0][ids.shape[1]:]),
                             "last_logits": lo.tolist(),
                             "top_ids": top.indices.tolist(), "top_vals": top.values.tolist()})
        print(repr(p), "->", repr(res["cases"][-1]["text"]), flush=True)
json.dump(res, open(out, "w"))
base = out[:-5] if out.endswith(".json") else out
for i, c in enumerate(res["cases"]):
    json.dump({"prompt_ids": c["prompt_ids"], "full_ids": c["full_ids"]}, open(f"{base}.case{i}.json", "w"))
