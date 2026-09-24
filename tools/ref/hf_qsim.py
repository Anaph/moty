"""HF fake-quant: how much does an int4 scale rule cost on a real model?

  hf_qsim.py <snapshot> <scheme[,scheme...]> [text=ppl_text.txt]

Every nn.Linear except lm_head is quantized to int4 in groups of 32 (a
"@64" suffix: groups of 64) and dequantized back; the model then runs in
f32 over the first 2048 tokens of the text. Schemes: f32 (baseline),
amax7 (symmetric amax/7), q40 (max |w| -> level -8), noclip (Q4R4's rule:
max -> -8 unless that clips the opposite sign), q40s (q40 + squared-error
scale search). Prints PPL, mean KL(f32 || q) and top-1 agreement."""
import sys, math, json, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
GS = 32
def q_g(w, mode, gs=None):
    """int4 fake-quant of a [O, I] weight in groups of gs (default GS)"""
    gs = gs or GS
    O, I = w.shape; g = w.reshape(O, I//gs, gs)
    if mode == "amax7":
        s = g.abs().amax(-1, keepdim=True) / 7; q = torch.clamp(torch.round(g / s.clamp_min(1e-12)), -8, 7)
        return (q * s).reshape(O, I)
    idx = g.abs().argmax(-1, keepdim=True); mx = torch.gather(g, -1, idx)
    if mode == "noclip":   # -8 on the max side, never clip the opposite sign
        opp = torch.where(torch.sign(g) == -torch.sign(mx), g.abs(), torch.zeros_like(g)).amax(-1, keepdim=True)
        dm = torch.maximum(mx.abs() / 8, opp / 7)
        d = (torch.sign(mx) * -dm).half().float()
        d = torch.where(d == 0, torch.ones_like(d), d)
        return (torch.clamp(torch.round(g / d), -8, 7) * d).reshape(O, I)
    best = None; beste = None
    facs = [1.0] if mode == "q40" else [1.0, 0.95, 0.9, 1.05, 0.85]
    for f in facs:
        d = (mx / -8 * f).half().float()
        q = torch.clamp(torch.round(g / torch.where(d == 0, torch.ones_like(d), d)), -8, 7) * d
        e = ((g - q) ** 2).sum(-1, keepdim=True)
        if best is None: best, beste = q, e
        else:
            m = e < beste; best = torch.where(m, q, best); beste = torch.where(m, e, beste)
    return best.reshape(O, I)
def q8_row(w):
    """int8 fake-quant with one scale per row (moty QBITS=8 weights)"""
    s = w.abs().amax(-1, keepdim=True) / 127
    return torch.round(w / s.clamp_min(1e-12)).clamp(-127, 127) * s

def main():
  global GS
  snap = sys.argv[1]; schemes = sys.argv[2].split(",")
  tok = AutoTokenizer.from_pretrained(snap)
  text = open(sys.argv[3] if len(sys.argv) > 3 else "ppl_text.txt").read()
  ids = tok(text, return_tensors="pt").input_ids[:, :2048]
  ref_lp = None
  for sc in schemes:
    GS = 64 if sc.endswith("@64") else 32
    sc0 = sc.split("@")[0]
    model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
    with torch.no_grad():
        if sc != "f32":
            for n, mod in model.named_modules():
                if isinstance(mod, torch.nn.Linear) and "lm_head" not in n:
                    mod.weight.copy_(q_g(mod.weight, sc0))
        lo = model(ids).logits[0]
    lp = torch.log_softmax(lo, -1); nll = -lp[:-1].gather(1, ids[0, 1:, None]).squeeze(1)
    if ref_lp is None: ref_lp = lp
    kl = (ref_lp.exp() * (ref_lp - lp)).sum(-1).mean().item()
    top1 = (ref_lp.argmax(-1) == lp.argmax(-1)).float().mean().item()
    print(sc, "n", ids.shape[1], "ppl", round(math.exp(nll.mean().item()), 3), "KL", round(kl, 4), "top1", round(top1, 4), flush=True)

if __name__ == "__main__":
    main()
