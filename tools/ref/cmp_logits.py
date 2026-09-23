import json, sys, numpy as np
ref = json.load(open(sys.argv[1])); case = int(sys.argv[2]); got = np.fromfile(sys.argv[3], dtype=np.float32)
r = np.array(ref["cases"][case]["last_logits"], dtype=np.float32)
k = 10
tr, tg = np.argsort(-r)[:k], np.argsort(-got)[:k]
print(f"case{case}: max|dlogit| {np.abs(r-got).max():.2e}  rel {np.abs(r-got).max()/np.abs(r).max():.2e}  "
      f"top1 {'OK' if tr[0]==tg[0] else 'DIFF'}  top{k} overlap {len(set(tr)&set(tg))}/{k}  top{k} order {'same' if (tr==tg).all() else 'differs'}")
