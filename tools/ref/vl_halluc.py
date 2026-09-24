"""Invented objects in quantized answers (hf_vlq.py --greedy --out json).

  vl_halluc.py <out.json> ...

Two counts per config: tiles whose answer names an object noun (from a
fixed list) that the f32 reference answer for the same tile does not, and
the narrower "computer set" (monitor, keyboard, mouse, laptop, screen) —
the objects the int4 LM invented on the live tiles of docs/performance.md
§5.14. A word-set difference: a noun can be real content the reference did
not reach within its 64 tokens, so read the lists, not only the counts."""
import json, re, sys, collections
OBJ = set("""monitor monitors keyboard keyboards mouse laptop laptops computer computers screen screens phone phones
chair chairs lamp lamps book books person people man woman men women child cat cats dog dogs car cars truck window
windows plant plants cup cups mug bottle bottles printer printers speaker speakers camera cameras television tv
bag bags box boxes shelf shelves cabinet door clock clocks fan fans headphones microphone router modem tablet
drawer drawers bed sofa table tables paper papers pen pens tool tools toolbox bench vent wall floor ceiling
cable cables cord cords wire wires plug plugs strip outlet outlets socket sockets board boards device devices
light lights label labels sign sign container containers bin basket radio battery batteries""".split())
def words(t): return set(re.findall(r"[a-z]+", t.lower()))
for path in sys.argv[1:]:
    d = json.load(open(path)); refs = d["refs"]
    refs = {k: words(v) & OBJ for k, v in refs.items()}
    for cfg, r in d["configs"].items():
        if not r["tiles"] or "text" not in next(iter(r["tiles"].values())): continue
        inv = collections.Counter(); ntile = 0
        for tile, t in r["tiles"].items():
            new = (words(t["text"]) & OBJ) - refs.get(tile, set())
            if new: ntile += 1; inv.update(new)
        print(f"{cfg:52s} tiles with invented objects {ntile:2d}/{len(r['tiles'])}  {', '.join(f'{w} {n}' for w, n in inv.most_common(6))}")

# the failure agent B saw: a computer set that is not in the scene
PC = {"monitor", "monitors", "keyboard", "keyboards", "mouse", "laptop", "laptops", "screen", "screens"}
for path in sys.argv[1:]:
    d = json.load(open(path)); refs = {k: words(v) for k, v in d["refs"].items()}
    for cfg, r in d["configs"].items():
        if not r["tiles"] or "text" not in next(iter(r["tiles"].values())): continue
        n = sum(bool((words(t["text"]) & PC) - refs.get(tile, set())) for tile, t in r["tiles"].items())
        print(f"PC {cfg:49s} tiles inventing monitor/keyboard/mouse/laptop/screen {n:2d}/{len(r['tiles'])}")
