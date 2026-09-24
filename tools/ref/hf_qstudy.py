"""Quantization study for one model (HF fake-quant): weight schemes,
calibration methods, mixed precision. Everything is simulated in f32 on
the first 2048 tokens of the evaluation text; the metrics are PPL, mean
KL(f32 || q) and top-1 agreement with f32.

  hf_qstudy.py <snapshot> <eval_text> <calib_text> <config> [<config> ...]

A config is `scheme[:method][+mix]`, applied to every nn.Linear and to the
(untied) lm_head; the input embedding is int8 per row (what moty keeps
resident).

  scheme  sym4g32 (Q4R4: symmetric, no-clip Q4_0 rule, f16 scale),
          sym4g16 / sym4g64 / sym4g128, asym4g32 (Q4_1: f16 scale + min),
          sym4g32i8 (per-group scales as int8 x one f32 per row),
          sym5g32, sym8g32 (Q8_0), q8row (int8, one scale per row), f32
  method  rtn (default), imx (activation-weighted scale search: minimise
          Σ E[x_j²] (w_j - q_j d)² per group over a grid of scales),
          awq (per-input-channel scaling s = E|x|^a, a searched per layer,
          quantize W·diag(s), use Q/diag(s) — the fold into the previous
          op is exact math, simulated here), gptq (Hessian error
          compensation, column order, 1 % damping, groups formed on the fly)
  +mix    `+8=<regex>|<regex>...`: matching Linear names (and "head")
          are q8row; `+5=<regex>|...`: sym5g32 with the same method, `+Q=`:
          sym8g32 (int8, group-32 f16 scales) with the same method;
          the rest the scheme
  @a8     suffix: also quantize every Linear input as moty does (int8 per
          token, groups of 32, amax/127)

Calibration: inputs of every Linear and of the head on the first 2048
tokens of <calib_text> (f32 model: non-sequential GPTQ)."""
import sys, math, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
torch.set_grad_enabled(False)
torch.manual_seed(0)

def _gview(w, gs):
    O, I = w.shape
    return w.reshape(O, I // gs, gs)

def q_sym(w, bits, gs, scale_fmt="f16", imp=None, grid=None):
    """symmetric, the Q4R4 no-clip rule generalised to `bits`: the largest
    magnitude maps to -2^(b-1), unless that clips the opposite sign.
    imp (per input channel, >= 0): search d over `grid` x the rule's d,
    minimising the importance-weighted squared error."""
    O, I = w.shape; g = _gview(w, gs)
    lo, hi = -(1 << (bits - 1)), (1 << (bits - 1)) - 1
    idx = g.abs().argmax(-1, keepdim=True); mx = torch.gather(g, -1, idx)
    opp = torch.where(torch.sign(g) == -torch.sign(mx), g.abs(), torch.zeros_like(g)).amax(-1, keepdim=True)
    d0 = -torch.sign(mx) * torch.maximum(mx.abs() / -lo, opp / hi)
    d0 = torch.where(d0 == 0, torch.full_like(d0, 1e-8), d0)
    def fmt(d):
        if scale_fmt == "f16": return d.half().float()
        if scale_fmt == "i8":     # int8 per group x one f32 per row
            r = d.abs().reshape(O, -1).amax(-1, keepdim=True).unsqueeze(-1) / 127
            return torch.round(d / r).clamp(-127, 127) * r
        return d
    def rt(d):
        d = fmt(d); d = torch.where(d == 0, torch.full_like(d, 1e-8), d)
        return torch.clamp(torch.round(g / d), lo, hi) * d
    if imp is None:
        return rt(d0).reshape(O, I)
    wimp = _gview(imp.expand(O, I), gs)
    best = rt(d0); be = (wimp * (g - best) ** 2).sum(-1, keepdim=True)
    for f in grid:
        q = rt(d0 * f); e = (wimp * (g - q) ** 2).sum(-1, keepdim=True)
        m = e < be; best = torch.where(m, q, best); be = torch.where(m, e, be)
    return best.reshape(O, I)

def q_asym(w, bits, gs, imp=None, grid=None):
    O, I = w.shape; g = _gview(w, gs)
    n = (1 << bits) - 1
    mn = g.amin(-1, keepdim=True); mxv = g.amax(-1, keepdim=True)
    def rt(lo_, hi_):
        s = ((hi_ - lo_) / n).half().float(); s = torch.where(s == 0, torch.full_like(s, 1e-8), s)
        m = lo_.half().float()
        return torch.clamp(torch.round((g - m) / s), 0, n) * s + m
    if imp is None: return rt(mn, mxv).reshape(O, I)
    wimp = _gview(imp.expand(O, I), gs)
    best = rt(mn, mxv); be = (wimp * (g - best) ** 2).sum(-1, keepdim=True)
    for f in grid:           # shrink the range towards its centre
        c = (mn + mxv) / 2; q = rt(c + (mn - c) * f, c + (mxv - c) * f)
        e = (wimp * (g - q) ** 2).sum(-1, keepdim=True)
        m = e < be; best = torch.where(m, q, best); be = torch.where(m, e, be)
    return best.reshape(O, I)

def q8row(w):
    s = w.abs().amax(-1, keepdim=True) / 127
    return torch.round(w / s.clamp_min(1e-12)).clamp(-127, 127) * s

SCHEMES = {
    "sym4g32": lambda w, imp=None, grid=None: q_sym(w, 4, 32, "f16", imp, grid),
    "sym4g16": lambda w, imp=None, grid=None: q_sym(w, 4, 16, "f16", imp, grid),
    "sym4g64": lambda w, imp=None, grid=None: q_sym(w, 4, 64, "f16", imp, grid),
    "sym4g128": lambda w, imp=None, grid=None: q_sym(w, 4, 128, "f16", imp, grid),
    "sym4g32i8": lambda w, imp=None, grid=None: q_sym(w, 4, 32, "i8", imp, grid),
    "asym4g32": lambda w, imp=None, grid=None: q_asym(w, 4, 32, imp, grid),
    "sym5g32": lambda w, imp=None, grid=None: q_sym(w, 5, 32, "f16", imp, grid),
    "sym8g32": lambda w, imp=None, grid=None: q_sym(w, 8, 32, "f16", imp, grid),
    "q8row": lambda w, imp=None, grid=None: q8row(w),
    "f32": lambda w, imp=None, grid=None: w.clone(),
}
BITS = {"sym4g32": 4.5, "sym4g16": 5.0, "sym4g64": 4.25, "sym4g128": 4.125, "sym4g32i8": 4.25,
        "asym4g32": 5.0, "sym5g32": 5.5, "sym8g32": 8.5, "q8row": 8.0, "f32": 32.0}
GRID = [0.80, 0.84, 0.88, 0.91, 0.94, 0.97, 1.03, 1.06, 1.10]

def gptq(w, H, qfn, gs, damp=0.01, blk=128):
    """GPTQ with groups of gs formed on the fly: each group's quantizer
    (scale) is fitted on the error-updated weights when the group starts."""
    W = w.clone().double(); O, I = W.shape
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += damp * torch.mean(torch.diag(H)) * torch.eye(I, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    Q = torch.zeros_like(W)
    for i1 in range(0, I, blk):
        i2 = min(i1 + blk, I); W1 = W[:, i1:i2].clone(); Q1 = torch.zeros_like(W1); E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % gs == 0:      # new group: quantize its current values to fix the scale
                gq = qfn(torch.cat([W1[:, i:], W[:, i2:]], 1)[:, :gs].float()).double()
                cur_g = col // gs
            q = gq[:, col - cur_g * gs]
            Q1[:, i] = q
            err = (W1[:, i] - q) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        Q[:, i1:i2] = Q1
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return Q.float()

def main():
    snap, etext, ctext = sys.argv[1:4]
    configs = sys.argv[4:]
    tok = AutoTokenizer.from_pretrained(snap)
    ids = tok(open(etext).read(), return_tensors="pt").input_ids[:, :2048]
    cids = tok(open(ctext).read(), return_tensors="pt").input_ids[:, :2048]
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
    lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
    emb = model.get_input_embeddings(); head = model.get_output_embeddings()
    if head.weight is emb.weight: head.weight = torch.nn.Parameter(emb.weight.detach().clone())
    mods = dict(lin); mods["head"] = head
    orig = {n: m.weight.detach().clone() for n, m in mods.items()}; emb0 = emb.weight.detach().clone()
    need_cal = any(":" in c for c in configs)
    stats = {}
    if need_cal:                  # calibration: E[x^2], E|x|, H = X^T X, a token subsample of X
        acc = {}
        def hook(name):
            def f(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).double()
                a = acc.setdefault(name, {"n": 0, "x2": 0, "xa": 0, "H": 0, "X": []})
                a["n"] += x.shape[0]; a["x2"] = a["x2"] + (x * x).sum(0); a["xa"] = a["xa"] + x.abs().sum(0)
                a["H"] = a["H"] + x.T @ x; a["X"].append(x[::8].float())
            return f
        hs = [m.register_forward_hook(hook(n)) for n, m in mods.items()]
        model(cids); [h.remove() for h in hs]
        for n, a in acc.items():
            stats[n] = {"x2": (a["x2"] / a["n"]).float(), "xa": (a["xa"] / a["n"]).float(),
                        "H": (a["H"] / a["n"]).float(), "X": torch.cat(a["X"], 0)}
        print(f"calibration: {cids.shape[1]} tokens of {ctext}", flush=True)
    act8 = {"on": False}
    def aq(mod, args):                # moty's activation quantization: int8 per token, groups of 32
        if not act8["on"]: return None
        x = args[0]; sh = x.shape; g = x.reshape(-1, sh[-1] // 32, 32)
        s = g.abs().amax(-1, keepdim=True).clamp_min(1e-30) / 127
        return (torch.round(g / s) * s).reshape(sh),
    for m in mods.values(): m.register_forward_pre_hook(aq)
    def run():
        return torch.log_softmax(model(ids).logits[0], -1)
    ref = run()
    def report(name, lp, bytes_tok):
        nll = -lp[:-1].gather(1, ids[0, 1:, None]).squeeze(1)
        kl = (ref.exp() * (ref - lp)).sum(-1).mean().item()
        top1 = (ref.argmax(-1) == lp.argmax(-1)).float().mean().item()
        print(f"{name:44s} {bytes_tok/1e6:7.1f} MB/tok  ppl {math.exp(nll.mean().item()):9.3f}  KL {kl:.4f}  top1 {top1:.4f}", flush=True)
    report("f32", ref, sum(w.numel() for w in orig.values()) * 4)
    gcache = {}
    for cfg in configs:
        act8["on"] = cfg.endswith("@a8"); cfg0 = cfg; cfg = cfg[:-3] if act8["on"] else cfg
        parts = cfg.split("+")
        scheme, _, method = parts[0].partition(":")
        method = method or "rtn"
        over = []                              # (scheme override, patterns)
        for p in parts[1:]:
            k, _, pats = p.partition("=")
            over.append(({"8": "q8row", "5": "sym5g32", "Q": "sym8g32"}[k], [x for x in pats.split("|") if x]))
        base_scheme = scheme; nbytes = 0
        for n, m in mods.items():
            w = orig[n]; scheme = base_scheme
            for sch, pats in over:
                if any((x == "head" and n == "head") or (x != "head" and re.search(x, n)) for x in pats): scheme = sch
            qfn = SCHEMES[scheme]
            if scheme == "q8row":
                q = q8row(w); nbytes += w.numel() * 8 / 8 + w.shape[0] * 4
            elif method == "rtn" or scheme == "f32":
                q = qfn(w); nbytes += w.numel() * BITS[scheme] / 8
            elif method == "imx":
                q = qfn(w, stats[n]["x2"].unsqueeze(0), GRID); nbytes += w.numel() * BITS[scheme] / 8
            elif method == "awq":
                X = stats[n]["X"]; y0 = X @ w.T; best = None
                for a in [0.0, 0.25, 0.5, 0.75]:
                    s = stats[n]["xa"].clamp_min(1e-6) ** a; s = s / s.mean()
                    qq = qfn(w * s.unsqueeze(0)) / s.unsqueeze(0)
                    e = ((X @ qq.T - y0) ** 2).mean().item()
                    if best is None or e < best[0]: best = (e, qq)
                q = best[1]; nbytes += w.numel() * BITS[scheme] / 8
            elif method == "gptq":
                gs = int(re.search(r"g(\d+)", scheme).group(1)) if "g" in scheme else w.shape[1]
                key = (n, scheme)                     # configs share GPTQ results per tensor and scheme
                if key not in gcache: gcache[key] = gptq(w, stats[n]["H"], lambda t: qfn(t), gs)
                q = gcache[key]; nbytes += w.numel() * BITS[scheme] / 8
            else:
                raise SystemExit("unknown method " + method)
            mods[n].weight.copy_(q)
        emb.weight.copy_(q8row(emb0))
        report(cfg0, run(), nbytes)
        for n, m in mods.items(): m.weight.copy_(orig[n])
        emb.weight.copy_(emb0)

if __name__ == "__main__":
    main()
