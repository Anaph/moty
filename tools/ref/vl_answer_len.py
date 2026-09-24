"""Answer length to end of turn for a vision-language decoder, per instruction:
HF f32 greedy on image rows until EOS (or --ngen), on every tile; writes the
prompt ids of each instruction (moty's PROMPT_IDS format) and the answers.

  vl_answer_len.py <snapshot> <tiles dir> --image-text '<|global_image|>' --image-rep '<|image|>' \\
      --image-token 49152 [--ngen 160] [--out answers.json] "Describe the image." "..."

The user turn is <image-text> + N x <image-rep> + instruction, through the
snapshot's own chat template (N = rows per tile); with the prompt that the
node sends this reproduces its ids exactly. Prints, per instruction: prompt
length, answer tokens per tile, how many answers ended at EOS, the first
tile's text. moty is compared on the same ids with PROMPT_IDS= (docs/models)."""
import argparse, glob, json, os, numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)

ap = argparse.ArgumentParser()
ap.add_argument("snap"); ap.add_argument("tiles"); ap.add_argument("instructions", nargs="+")
ap.add_argument("--image-text", default=""); ap.add_argument("--image-rep", required=True)
ap.add_argument("--image-token", type=int, required=True); ap.add_argument("--ngen", type=int, default=160)
ap.add_argument("--out", default="answers.json"); ap.add_argument("--prompt-dir", default=".")
a = ap.parse_args()
tok = AutoTokenizer.from_pretrained(a.snap); m = AutoModelForCausalLM.from_pretrained(a.snap, dtype=torch.float32).eval()
eos = m.generation_config.eos_token_id; eos = set(eos if isinstance(eos, list) else [eos])
emb = m.get_input_embeddings(); D = emb.weight.shape[1]
tiles = sorted(glob.glob(os.path.join(a.tiles, "*.f32")))
nrows = os.path.getsize(tiles[0]) // (4 * D)
out = {}
for k, text in enumerate(a.instructions):
    s = tok.apply_chat_template([{"role": "user", "content": a.image_text + a.image_rep * nrows + text}],
                                add_generation_prompt=True, tokenize=False)
    ids = tok(s, add_special_tokens=False)["input_ids"]
    json.dump({"ids": ids}, open(os.path.join(a.prompt_dir, f"prompt_{k}.json"), "w"))
    it = torch.tensor(ids); res = {}
    for f in tiles:
        rows = torch.from_numpy(np.fromfile(f, dtype="<f4").reshape(-1, D))
        e = emb(it.unsqueeze(0))[0].clone(); e[it == a.image_token] = rows
        o = m(inputs_embeds=e.unsqueeze(0), use_cache=True); gen = []
        for _ in range(a.ngen):
            t = int(o.logits[0, -1].argmax()); gen.append(t)
            if t in eos: break
            o = m(input_ids=torch.tensor([[t]]), past_key_values=o.past_key_values, use_cache=True)
        res[os.path.basename(f)] = gen
    out[text] = {"prompt": f"prompt_{k}.json", "answers": res}
    L = [len(g) for g in res.values()]; E = sum(g[-1] in eos for g in res.values())
    first = res[os.path.basename(tiles[0])]
    print(f"[{k}] {text!r}: prompt {len(ids)} ids | answer tokens {L} | EOS {E}/{len(L)} | "
          f"{tok.decode([t for t in first if t not in eos])!r}", flush=True)
json.dump(out, open(a.out, "w"))
