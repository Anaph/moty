"""Quantization study for the language model of a vision-language snapshot
on image inputs (HF fake-quant, as hf_qstudy.py, whose schemes, GPTQ and
config syntax it reuses).

  hf_vlq.py <snapshot> <prompt_ids.json> <eval tiles> [--calib-tiles T]
            [--calib-text F] [--ngen 64] [--greedy] <config> ...

A tile is a raw little-endian f32 file [N][hidden] of projected image
rows (e.g. the NPU encoder's output); `tiles` are comma-separated files
or directories (every *.f32 inside). For each eval tile the reference is
the f32 language model's greedy answer (ngen tokens) on those rows; each
config is scored teacher-forced on prompt + reference answer: top-1
agreement and mean KL(f32 || q) over the answer positions, and with
--greedy by its own greedy answer: leading tokens equal to the reference.

Mixes as in hf_qstudy.py, plus `+4=<regex>|...`: sym4g32 (Q4R4) for the
matching tensors — `f32+4=<regex>` is the one-group sensitivity sweep.

Calibration (for :gptq/:imx configs): Linear inputs over the
--calib-text tokens (first 2048) and/or --calib-tiles sequences (prompt +
the f32 greedy answer on those rows, image rows at the image-token
positions); both given: one Hessian over all tokens. Only the language
model's Linear layers and the head are quantized; the token embedding is
int8 per row as in moty; image rows stay f32 (moty injects them as is)."""
import argparse, glob, json, math, os, re, sys
import numpy as np, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hf_qstudy import SCHEMES, BITS, GRID, q8row, gptq
torch.set_grad_enabled(False)

def tiles_of(spec):
    out = []
    for p in (spec or "").split(","):
        if not p: continue
        out += sorted(glob.glob(os.path.join(p, "*.f32"))) if os.path.isdir(p) else [p]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap"); ap.add_argument("prompt"); ap.add_argument("tiles")
    ap.add_argument("--calib-tiles", default=""); ap.add_argument("--calib-text", default="")
    ap.add_argument("--ngen", type=int, default=64); ap.add_argument("--greedy", action="store_true")
    ap.add_argument("--out", default="", help="json with per-tile results")
    ap.add_argument("--calib-img", default="all", choices=["all", "none"],
                    help="calibration statistics over the image-row positions of the tile sequences too (all) or not (none)")
    ap.add_argument("configs", nargs="+")
    a = ap.parse_args()
    model = AutoModelForImageTextToText.from_pretrained(a.snap, dtype=torch.float32).eval()
    lm = model.model.language_model; head = model.lm_head; emb = lm.get_input_embeddings()
    img_tok = model.config.image_token_id; D = emb.weight.shape[1]
    pj = json.load(open(a.prompt)); pids = torch.tensor(pj["ids"] if isinstance(pj, dict) else pj)
    npos = int((pids == img_tok).sum())
    if head.weight is emb.weight: head.weight = torch.nn.Parameter(emb.weight.detach().clone())
    lin = {n: m for n, m in lm.named_modules() if isinstance(m, torch.nn.Linear)}
    mods = dict(lin); mods["head"] = head
    orig = {n: m.weight.detach().clone() for n, m in mods.items()}; emb0 = emb.weight.detach().clone()

    def rows_of(f):
        r = torch.from_numpy(np.fromfile(f, dtype="<f4").reshape(-1, D))
        assert r.shape[0] == npos, (f, r.shape, npos)
        return r
    def embeds(ids, rows):
        e = emb(ids.unsqueeze(0))[0].clone()
        if rows is not None: e[ids == img_tok] = rows
        return e.unsqueeze(0)
    def logits(ids, rows):
        return head(lm(inputs_embeds=embeds(ids, rows)).last_hidden_state)[0]
    def greedy(rows, n):
        out = lm(inputs_embeds=embeds(pids, rows), use_cache=True); toks = []
        for _ in range(n):
            t = int(head(out.last_hidden_state[:, -1]).argmax(-1)); toks.append(t)
            out = lm(inputs_embeds=emb(torch.tensor([[t]])), past_key_values=out.past_key_values, use_cache=True)
        return toks

    ev = tiles_of(a.tiles); cal = tiles_of(a.calib_tiles)
    print(f"eval tiles {len(ev)}, calib tiles {len(cal)}, prompt {len(pids)} ids ({npos} image rows)", flush=True)
    refs = {}
    for f in ev + [c for c in cal if c not in ev]:
        rows = rows_of(f); ans = greedy(rows, a.ngen)
        seq = torch.cat([pids, torch.tensor(ans)])
        refs[f] = {"rows": rows, "ans": ans, "seq": seq,
                   "lp": torch.log_softmax(logits(seq, rows)[len(pids) - 1:-1], -1)}
    tok = AutoTokenizer.from_pretrained(a.snap)
    for f in ev: print(f"  ref {os.path.basename(f)}: {tok.decode(refs[f]['ans'])[:90]!r}", flush=True)

    stats = {}
    if any(":" in c for c in a.configs):
        acc = {}; keep = {"m": None}             # positions that enter the statistics
        def hook(name):
            def fn(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).double()
                if keep["m"] is not None: x = x[keep["m"]]
                s = acc.setdefault(name, {"n": 0, "x2": 0, "xa": 0, "H": 0, "X": []})
                s["n"] += x.shape[0]; s["x2"] = s["x2"] + (x * x).sum(0); s["xa"] = s["xa"] + x.abs().sum(0)
                s["H"] = s["H"] + x.T @ x; s["X"].append(x[::8].float())
            return fn
        hs = [m.register_forward_hook(hook(n)) for n, m in mods.items()]
        ntok = 0
        if a.calib_text:
            cids = tok(open(a.calib_text).read(), return_tensors="pt").input_ids[0, :2048]
            logits(cids, None); ntok += len(cids)
        for f in cal:
            seq = refs[f]["seq"]
            keep["m"] = None if a.calib_img == "all" else seq != img_tok
            logits(seq, refs[f]["rows"]); ntok += int(len(seq) if keep["m"] is None else keep["m"].sum())
        keep["m"] = None
        [h.remove() for h in hs]
        for n, s in acc.items():
            stats[n] = {"x2": (s["x2"] / s["n"]).float(), "xa": (s["xa"] / s["n"]).float(),
                        "H": (s["H"] / s["n"]).float(), "X": torch.cat(s["X"], 0)}
        print(f"calibration: {ntok} tokens ({'text ' if a.calib_text else ''}{len(cal)} tiles, image rows: {a.calib_img})", flush=True)

    act8 = {"on": False}
    def aq(mod, args):                   # moty's activation quantization: int8 per token, groups of 32
        if not act8["on"]: return None
        x = args[0]; sh = x.shape; g = x.reshape(-1, sh[-1] // 32, 32)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
        return (torch.round(g / s) * s).reshape(sh),
    for m in mods.values(): m.register_forward_pre_hook(aq)

    results = {}
    gcache = {}
    for cfg0 in a.configs:
        act8["on"] = cfg0.endswith("@a8"); cfg = cfg0[:-3] if act8["on"] else cfg0
        parts = cfg.split("+"); scheme, _, method = parts[0].partition(":"); method = method or "rtn"
        over = []
        for p in parts[1:]:
            k, _, pats = p.partition("=")
            over.append(({"8": "q8row", "5": "sym5g32", "Q": "sym8g32", "4": "sym4g32"}[k], [x for x in pats.split("|") if x]))
        nbytes = 0
        for n, m in mods.items():
            w = orig[n]; sch = scheme
            for s_, pats in over:
                if any((x == "head" and n == "head") or (x != "head" and re.search(x, n)) for x in pats): sch = s_
            qfn = SCHEMES[sch]
            if sch == "q8row": q = q8row(w); nbytes += w.numel() + w.shape[0] * 4
            elif method == "rtn" or sch == "f32": q = qfn(w); nbytes += w.numel() * BITS[sch] / 8
            elif method == "imx": q = qfn(w, stats[n]["x2"].unsqueeze(0), GRID); nbytes += w.numel() * BITS[sch] / 8
            elif method == "gptq":
                gs = int(re.search(r"g(\d+)", sch).group(1))
                key = (n, sch)
                if key not in gcache: gcache[key] = gptq(w, stats[n]["H"], lambda t: qfn(t), gs)
                q = gcache[key]; nbytes += w.numel() * BITS[sch] / 8
            else: raise SystemExit("unknown method " + method)
            m.weight.copy_(q)
        emb.weight.copy_(q8row(emb0))
        agree = kl = n_ans = 0; lead = []; per = {}
        for f in ev:
            r = refs[f]
            lp = torch.log_softmax(logits(r["seq"], r["rows"])[len(pids) - 1:-1], -1)
            agree += int((lp.argmax(-1) == torch.tensor(r["ans"])).sum()); n_ans += len(r["ans"])
            kl += float((r["lp"].exp() * (r["lp"] - lp)).sum(-1).sum())
            per[os.path.basename(f)] = {"tf": int((lp.argmax(-1) == torch.tensor(r["ans"])).sum())}
            if a.greedy:
                g = greedy(r["rows"], a.ngen); k = 0
                while k < len(g) and g[k] == r["ans"][k]: k += 1
                lead.append(k); per[os.path.basename(f)].update(lead=k, text=tok.decode(g))
        msg = f"{cfg0:48s} {nbytes/1e6:7.1f} MB/tok  top1 {agree/n_ans:.4f}  KL {kl/n_ans:.4f}"
        if a.greedy: msg += f"  lead {sum(lead)/len(lead):5.1f}/{a.ngen} (min {min(lead)}, full {sum(x == a.ngen for x in lead)}/{len(lead)})"
        print(msg, flush=True)
        results[cfg0] = {"mb_tok": nbytes / 1e6, "top1": agree / n_ans, "kl": kl / n_ans, "tiles": per}
        for n, m in mods.items(): m.weight.copy_(orig[n])
        emb.weight.copy_(emb0)
    if a.out: json.dump({"refs": {os.path.basename(f): tok.decode(refs[f]["ans"]) for f in ev}, "configs": results},
                        open(a.out, "w"), indent=1)

if __name__ == "__main__":
    main()
