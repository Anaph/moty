"""moty on image inputs vs the HF f32 language model on the same rows.

  moty_vleval.py refs <snapshot> <prompt_ids.json> <out refs.json> <tile.f32> ...
      HF f32 greedy answers (64 tokens) of the VL snapshot on each tile's rows
  moty_vleval.py run  <prompt_ids.json> <refs.json> <moty lfm2 binary> <container> [threads]
      per tile: leading tokens of moty's greedy answer equal to the reference
      (TOKENS=1, IGNORE_EOS) and teacher-forced top-1 over the answer
      positions (PPL mode; 64, or fewer when a reference ended at EOS), the rows injected with EMBEDS=; one summary line,
      per-tile numbers in moty_<container>.json

Tiles: raw little-endian f32 [N][hidden] image rows (e.g. NPU encoder output).
Options (anywhere): --image-token N  the placeholder id (required for a plain
causal LM whose config names none; a vision-language snapshot has
image_token_id); --qbits Q  moty's QBITS for `run` (4 default, 8, 0 = f32:
the exactness check)."""
import json, os, subprocess, sys, tempfile
def opt(name, default):
    if name in sys.argv:
        i = sys.argv.index(name); v = sys.argv[i + 1]; del sys.argv[i:i + 2]; return v
    return default
IMG = opt("--image-token", None); QB = opt("--qbits", "4")
if len(sys.argv) < 6 or sys.argv[1] not in ("refs", "run"): sys.exit(__doc__)
mode = sys.argv[1]
if mode == "refs": snap, P, outp, tiles = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5:]
else: P, refp, binp, snap = sys.argv[2], sys.argv[3], sys.argv[4], sys.argv[5]
pids = json.load(open(P))["ids"]

if mode == "refs":
    import numpy as np, torch
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText
    torch.set_grad_enabled(False)
    try:                                          # a vision-language snapshot: its language model
        model = AutoModelForImageTextToText.from_pretrained(snap, dtype=torch.float32).eval()
        lm = model.model.language_model
    except (ValueError, KeyError):                # a plain causal LM (e.g. a split-off VLM decoder)
        model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
        lm = model.model
    head = model.lm_head; emb = lm.get_input_embeddings()
    it = int(IMG) if IMG is not None else model.config.image_token_id
    ids = torch.tensor(pids); out = {}
    for f in tiles:
        rows = torch.from_numpy(np.fromfile(f, dtype="<f4").reshape(-1, emb.weight.shape[1]))
        e = emb(ids.unsqueeze(0))[0].clone(); e[ids == it] = rows
        o = lm(inputs_embeds=e.unsqueeze(0), use_cache=True); ans = []
        for _ in range(64):
            t = int(head(o.last_hidden_state[:, -1]).argmax(-1)); ans.append(t)
            o = lm(inputs_embeds=emb(torch.tensor([[t]])), past_key_values=o.past_key_values, use_cache=True)
        out[f] = ans; print(os.path.basename(f), ans[:8], flush=True)
    json.dump(out, open(outp, "w"))
    sys.exit(0)

refs = json.load(open(refp))
th = sys.argv[6] if len(sys.argv) > 6 else "8"
env0 = dict(os.environ, SNAP=snap, QBITS=QB, THREADS=th, MOTY_NO_OMP_TUNE="1", TEMP="0")
if IMG is not None: env0["EMBEDS_TOKEN"] = IMG
res = {}
for f, ans in refs.items():
    env = dict(env0, EMBEDS=f, PROMPT_IDS=P, NGEN="64", IGNORE_EOS="1", TOKENS="1")
    err = subprocess.run([binp], env=env, capture_output=True, text=True, errors="replace").stderr
    toks = []
    for line in err.splitlines():                     # the dumped ids: a line of integers
        p = line.split()
        if len(p) >= 32 and all(x.lstrip("-").isdigit() for x in p): toks = [int(x) for x in p]
    lead = 0
    L = min(len(ans), 64)                             # a reference that ended at EOS is shorter
    while lead < min(len(toks), L) and toks[lead] == ans[lead]: lead += 1
    with tempfile.TemporaryDirectory() as d:
        pj = os.path.join(d, "p.json"); am = os.path.join(d, "am")
        json.dump({"ids": pids + ans, "argmax": [], "ppl": 0}, open(pj, "w"))
        subprocess.run([binp], env=dict(env0, EMBEDS=f, PPL=pj, PPL_OUT=am), capture_output=True, text=True, errors="replace")
        a = [int(x) for x in open(am).read().split()]
    k = len(pids) - 1
    tf = sum(a[k + i] == ans[i] for i in range(L))
    res[os.path.basename(f)] = {"lead": lead, "tf": tf, "L": L, "n": len(toks), "toks": toks}
lead = [r["lead"] for r in res.values()]; tf = [r["tf"] for r in res.values()]
Ls = [r["L"] for r in res.values()]; ref_len = f"/{Ls[0]}" if len(set(Ls)) == 1 else f" (refs {min(Ls)}-{max(Ls)} tokens)"
print(f"{os.path.basename(snap):22s} tiles {len(res)}  lead mean {sum(lead)/len(lead):5.1f}{ref_len} min {min(lead)} "
      f"full {sum(r['lead'] == r['L'] for r in res.values())}  "
      f"teacher-forced {sum(tf)/sum(Ls)*100:5.1f}%  (dumped {min(r['n'] for r in res.values())} tokens min)", flush=True)
json.dump(res, open(f"moty_{os.path.basename(os.path.normpath(snap))}.json", "w"), indent=1)
