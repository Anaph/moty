"""Write a moty pre-packed container (the SAVE_PACKED layout) with
calibrated int4 codes.

  pack_r4.py <snapshot> <out_dir> [--method rtn|gptq] [--calib text]
             [--q8 glob,glob,...] [--tokens N]

Every matrix moty packs for an LFM2 / Qwen / Llama snapshot (the Linear
weights of the layers and the tied or separate lm_head) is written as
Q4R4 — U8 "<name>" (4-row blocks, groups of 32, row r byte j = q[j] |
q[j+16] << 4) + F16 "<name>.s16" — or as Q8R4 (I8, 32 codes per row and
group) when its name matches a --q8 glob (the same globs as moty's
Q8_TENSORS, on engine-side names: "model.language_model.X" of a
vision-language snapshot is "model.X"). All other tensors and
config/tokenizer files are copied; for a vision-language snapshot only
the language model's matrices are packed (calibration is text-only).

--method rtn reproduces moty's own packer bit for bit (the no-clip rule
in f32, f16 scale, round-half-even). --method gptq (Frantar et al.) walks
the columns of each matrix and pushes every rounding error onto the
columns not yet quantized through the inverse Hessian of the layer's
calibration inputs (H = E[x x^T], 1 % damping, blocks of 128); a group's
f16 scale is fixed by the no-clip rule on its error-updated weights when
the walk reaches it. Calibration: the first --tokens (2048) tokens of
--calib through the f32 model, every Linear's input (non-sequential).

moty loads the result with SNAP=<out_dir> QBITS=4 (no policy needed)."""
import argparse, fnmatch, json, os, shutil, struct, sys
import numpy as np, torch
torch.set_grad_enabled(False)

def rule_scale(g, bits):
    """per-row-group no-clip scale (f16 bits as int16 view, f32 value)"""
    lo = 1 << (bits - 1)
    idx = g.abs().argmax(-1, keepdim=True)            # first max, as moty
    mx = torch.gather(g, -1, idx)
    other = (g > 0) != (mx > 0)
    opp = torch.where(other, g.abs(), torch.zeros_like(g)).amax(-1, keepdim=True)
    dm = torch.maximum(mx.abs() / float(lo), opp / float(lo - 1))
    h = torch.where(mx > 0, -dm, dm).half()
    return h, h.float()

def rtn_codes(g, dd, bits):
    lo = 1 << (bits - 1)
    inv = torch.where(dd != 0, 1.0 / dd, torch.zeros_like(dd))
    return torch.clamp(torch.round(g * inv), -lo, lo - 1)

def quantize(w, bits, method, H=None):
    """w [O, I] f32 -> codes [O, I] (int, signed) and f16 scales [O, I/32]"""
    O, I = w.shape; nb = I // 32
    if method == "rtn" or H is None:
        g = w.reshape(O, nb, 32)
        h, dd = rule_scale(g, bits)
        return rtn_codes(g, dd, bits).reshape(O, I), h.reshape(O, nb)
    W = w.clone().double()
    H = H.double().clone()
    dead = torch.diag(H) == 0
    H[dead, dead] = 1; W[:, dead] = 0
    H += 0.01 * torch.mean(torch.diag(H)) * torch.eye(I, dtype=H.dtype)
    Hinv = torch.linalg.cholesky(torch.cholesky_inverse(torch.linalg.cholesky(H)), upper=True)
    codes = torch.zeros(O, I); scales = torch.zeros(O, nb, dtype=torch.float16)
    lo = 1 << (bits - 1)
    blk = 128
    for i1 in range(0, I, blk):
        i2 = min(i1 + blk, I); W1 = W[:, i1:i2].clone(); E1 = torch.zeros_like(W1)
        Hi = Hinv[i1:i2, i1:i2]
        for i in range(i2 - i1):
            col = i1 + i
            if col % 32 == 0:                         # the group's scale from its current weights
                cur = W1[:, i:i + 32].float().reshape(O, 1, 32)
                h, dd = rule_scale(cur, bits)
                scales[:, col // 32] = h[:, 0, 0]; d = dd[:, 0, 0].double()
            inv = torch.where(d != 0, 1.0 / d, torch.zeros_like(d))
            q = torch.clamp(torch.round(W1[:, i] * inv), -lo, lo - 1)
            codes[:, col] = q.float()
            err = (W1[:, i] - q * d) / Hi[i, i]
            W1[:, i:] -= err.unsqueeze(1) * Hi[i, i:].unsqueeze(0)
            E1[:, i] = err
        W[:, i2:] -= E1 @ Hinv[i1:i2, i2:]
    return codes, scales

def pack(codes, scales, bits):
    """[O, I] signed codes + [O, nb] f16 -> (block bytes, scale stream) in moty's layout"""
    O, I = codes.shape; nb = I // 32; O4 = (O + 3) // 4
    c = torch.zeros(O4 * 4, I); c[:O] = codes
    s = torch.zeros(O4 * 4, nb, dtype=torch.float16); s[:O] = scales
    c = c.reshape(O4, 4, nb, 32).permute(0, 2, 1, 3)          # [block][group][row][32]
    if bits == 4:
        u = (c + 8).to(torch.uint8)
        b = u[..., :16] | (u[..., 16:] << 4)                   # [block][group][row][16]
        data = b.contiguous().numpy().tobytes()
    else:
        data = c.to(torch.int8).contiguous().numpy().tobytes()
    sc = s.reshape(O4, 4, nb).permute(0, 2, 1).contiguous()    # [block][group][row]
    return data, sc.numpy().tobytes()

def st_read(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]; hdr = json.loads(f.read(n)); base = 8 + n
    return hdr, base

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("snap"); ap.add_argument("out")
    ap.add_argument("--method", default="gptq", choices=["rtn", "gptq"])
    ap.add_argument("--calib"); ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--q8", default="")
    a = ap.parse_args()
    q8 = [p for p in a.q8.split(",") if p]
    files = sorted(f for f in os.listdir(a.snap) if f.endswith(".safetensors"))
    tensors = {}                                   # name -> (file, header entry, base)
    for fn in files:
        hdr, base = st_read(os.path.join(a.snap, fn))
        for k, v in hdr.items():
            if k != "__metadata__": tensors[k] = (os.path.join(a.snap, fn), v, base)
    from transformers import AutoModelForCausalLM, AutoModelForImageTextToText, AutoTokenizer
    try:
        model = AutoModelForCausalLM.from_pretrained(a.snap, dtype=torch.float32).eval()
    except ValueError:            # vision-language wrapper: its text model is what moty runs
        model = AutoModelForImageTextToText.from_pretrained(a.snap, dtype=torch.float32).eval()
    wrapped = any(k.startswith("model.language_model.") for k in tensors)
    lin = {n + ".weight": m for n, m in model.named_modules()
           if isinstance(m, torch.nn.Linear) and "lm_head" not in n and n + ".weight" in tensors
           and (not wrapped or n.startswith("model.language_model."))}
    tied = "lm_head.weight" not in tensors
    head_mod = model.get_output_embeddings()
    targets = dict(lin); targets["lm_head.weight"] = head_mod
    H = {}
    if a.method == "gptq":
        tok = AutoTokenizer.from_pretrained(a.snap)
        ids = tok(open(a.calib).read(), return_tensors="pt").input_ids[:, :a.tokens]
        acc = {}
        def hook(name):
            def f(mod, inp, out):
                x = inp[0].reshape(-1, inp[0].shape[-1]).double()
                s = acc.setdefault(name, [0, 0]); s[0] += x.shape[0]; s[1] = s[1] + x.T @ x
            return f
        hs = [m.register_forward_hook(hook(n)) for n, m in targets.items()]
        model(ids); [h.remove() for h in hs]
        H = {n: (s[1] / s[0]).float() for n, s in acc.items()}
        print(f"calibration: {ids.shape[1]} tokens of {a.calib}", flush=True)
    os.makedirs(a.out, exist_ok=True)
    out_t = []                                     # (name, dtype, n, bytes)
    def canon(name):              # moty's engine-side name (Q8_TENSORS globs match these)
        return "model." + name[len("model.language_model."):] if name.startswith("model.language_model.") else name
    def add_packed(name, w):
        bits = 8 if any(fnmatch.fnmatchcase(canon(name), p) for p in q8) else 4
        codes, scales = quantize(w.float(), bits, a.method, H.get(name))
        data, sc = pack(codes, scales, bits)
        out_t.append((name, "I8" if bits == 8 else "U8", len(data), data))
        out_t.append((name + ".s16", "F16", len(sc) // 2, sc))
        print(f"{name}: {'Q8R4' if bits == 8 else 'Q4R4'} {a.method}", flush=True)
    for name in tensors:
        if name in lin:
            add_packed(name, lin[name].weight)
        else:
            fn, v, base = tensors[name]
            o0, o1 = v["data_offsets"]
            with open(fn, "rb") as f: f.seek(base + o0); raw = f.read(o1 - o0)
            n = int(np.prod(v["shape"])) if v["shape"] else 1
            out_t.append((name, v["dtype"], n, raw))
    if "lm_head.weight" in tensors:                # separate head: pack it instead of copying
        out_t = [t for t in out_t if t[0] != "lm_head.weight"]
    add_packed("lm_head.weight", head_mod.weight)
    hdr = {"__metadata__": {"format": "moty-q4r4"}}; off = 0
    for name, dt, n, data in out_t:
        hdr[name] = {"dtype": dt, "shape": [n], "data_offsets": [off, off + len(data)]}; off += len(data)
    hb = json.dumps(hdr, separators=(",", ":")).encode()
    hb += b" " * ((8 - len(hb) % 8) % 8)
    with open(os.path.join(a.out, "model.safetensors"), "wb") as f:
        f.write(struct.pack("<Q", len(hb))); f.write(hb)
        for _, _, _, data in out_t: f.write(data)
    for aux in ("config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "special_tokens_map.json", "chat_template.jinja"):
        if os.path.exists(os.path.join(a.snap, aux)): shutil.copy(os.path.join(a.snap, aux), a.out)
    print("wrote", os.path.join(a.out, "model.safetensors"), off, "bytes of data")

if __name__ == "__main__":
    main()
