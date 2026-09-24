"""Leading tokens equal to the reference per tile, from vl_decoder_bench.sh output (IDS lines)."""
import json, sys
refs = json.load(open(sys.argv[2])); cur = None; res = {}
for l in open(sys.argv[1]):
    p = l.split()
    if l.startswith("== ids"): cur = (p[2], p[3])
    elif l.startswith("IDS") and cur:
        t = [int(x) for x in p[1:]]; a = {k.rsplit("/", 1)[-1]: v for k, v in refs.items()}[cur[1]]; k = 0
        while k < min(len(t), 64) and t[k] == a[k]: k += 1
        res.setdefault(cur[0], {})[cur[1]] = k
for c, r in res.items():
    v = list(r.values()); print(f"{c:12s} tiles {len(v)} lead mean {sum(v)/len(v):5.1f}/64 min {min(v)} full {sum(x == 64 for x in v)}  {v}")
