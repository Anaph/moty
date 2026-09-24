"""Per-tensor-group quantization sensitivity (HF fake-quant).

  hf_sens.py <snapshot> <text> [groups] [--int8 groups] [--q4 policy-groups]

The model runs in f32 over the first 2048 tokens of <text>; each group is
fake-quantized ALONE (everything else f32) and the PPL, mean KL(f32 || q)
and top-1 agreement with f32 are printed, so groups can be ranked by the
damage int4 (Q4R4's no-clip rule, groups of 32) does to them.

A group is a regex on nn.Linear names, or one of the specials
  head       the (tied) lm_head alone, int4 (untied copy)
  head8      the lm_head alone, int8 per row
  embed8     the input embedding alone, int8 per row (moty QBITS>0)
Default groups for LFM2: head, embed8, every Linear type, every layer.

--mix "a,b,c" evaluates one mixed point instead: every Linear int4 and
the head int4 (the moty int4 configuration), except the listed groups,
which are int8 per row (head8 / "head" in the list: head int8)."""
import sys, math, re, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
sys.path.insert(0, __file__.rsplit("/", 1)[0])
from hf_qsim import q_g, q8_row

snap, text = sys.argv[1], sys.argv[2]
args = sys.argv[3:]
mix = None
if args and args[0] == "--mix":
    mix = [g for g in args[1].split(",") if g] if len(args) > 1 else []
    args = []
tok = AutoTokenizer.from_pretrained(snap)
ids = tok(open(text).read(), return_tensors="pt").input_ids[:, :2048]
model = AutoModelForCausalLM.from_pretrained(snap, dtype=torch.float32).eval()
lin = {n: m for n, m in model.named_modules() if isinstance(m, torch.nn.Linear) and "lm_head" not in n}
orig = {n: m.weight.detach().clone() for n, m in lin.items()}
emb = model.get_input_embeddings(); emb0 = emb.weight.detach().clone()
head = model.get_output_embeddings()
if head.weight is emb.weight:                      # untie: head and embedding quantized separately
    head.weight = torch.nn.Parameter(emb0.clone())
head0 = head.weight.detach().clone()

def run():
    with torch.no_grad():
        return torch.log_softmax(model(ids).logits[0], -1)
def restore():
    with torch.no_grad():
        for n, m in lin.items(): m.weight.copy_(orig[n])
        emb.weight.copy_(emb0); head.weight.copy_(head0)
def report(name, lp, ref):
    nll = -lp[:-1].gather(1, ids[0, 1:, None]).squeeze(1)
    kl = (ref.exp() * (ref - lp)).sum(-1).mean().item()
    top1 = (ref.argmax(-1) == lp.argmax(-1)).float().mean().item()
    print(f"{name:34s} ppl {math.exp(nll.mean().item()):9.3f}  KL {kl:.4f}  top1 {top1:.4f}", flush=True)

ref = run(); report("f32", ref, ref)
with torch.no_grad():
    if mix is not None:
        keep8 = [g for g in mix if g not in ("head", "head8")]
        for n, m in lin.items():
            m.weight.copy_(q8_row(orig[n]) if any(re.search(g, n) for g in keep8) else q_g(orig[n], "noclip"))
        head.weight.copy_(q8_row(head0) if ("head" in mix or "head8" in mix) else q_g(head0, "noclip"))
        emb.weight.copy_(q8_row(emb0))
        report("mix int8[" + ",".join(mix) + "]", run(), ref)
        sys.exit(0)
    groups = args or (["head", "head8", "embed8",
                       r"conv\.in_proj", r"conv\.out_proj", r"q_proj", r"k_proj", r"v_proj",
                       r"self_attn\.out_proj", r"feed_forward\.w1", r"feed_forward\.w3", r"feed_forward\.w2"]
                      + [rf"layers\.{i}\." for i in range(len(model.model.layers))])
    for g in groups:
        restore()
        if g == "head": head.weight.copy_(q_g(head0, "noclip"))
        elif g == "head8": head.weight.copy_(q8_row(head0))
        elif g == "embed8": emb.weight.copy_(q8_row(emb0))
        else:
            hit = [n for n in lin if re.search(g, n)]
            if not hit: print(f"{g}: no match"); continue
            for n in hit: lin[n].weight.copy_(q_g(orig[n], "noclip"))
        report(g, run(), ref)
