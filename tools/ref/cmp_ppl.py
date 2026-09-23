import json, sys
ref = json.load(open(sys.argv[1])); am = [int(x) for x in open(sys.argv[2]).read().split()]
ra = ref["argmax"][:len(am)]
agree = sum(a == b for a, b in zip(ra, am))
print(f"top1 agreement vs HF f32: {agree}/{len(am)} = {100*agree/len(am):.1f}%  (HF ppl {ref['ppl']:.3f})")
