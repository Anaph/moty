"""run_configs.sh log(s) -> markdown table, medians over repetitions.

  parse_runs.py <log> [<log> ...]

moty rows: prefill / decode tok/s from the engine's summary line, peak RSS
from rssrun, load time from the banner. llama-bench rows: pp/tg per thread
count from its markdown output. Every run's values are listed in brackets
next to the median, so the spread stays visible."""
import re, sys, statistics as st, collections as C

rows = C.OrderedDict(); cur = None
for f in sys.argv[1:]:
    for line in open(f, encoding="utf-8", errors="replace"):
        m = re.match(r"== (.+?) rep=", line)
        if m:
            cur = rows.setdefault(m.group(1), C.defaultdict(list)); continue
        if cur is None: continue
        m = re.search(r"\(([\d.]+) tok/s\) \| decode \d+ tok in [\d.]+s \(([\d.]+) tok/s\)", line)
        if m: cur["prefill"].append(float(m.group(1))); cur["decode"].append(float(m.group(2)))
        m = re.search(r"peak_rss_mb=([\d.]+)", line)
        if m: cur["rss"].append(float(m.group(1)))
        m = re.search(r"\| load ([\d.]+)s", line)
        if m: cur["load"].append(float(m.group(1)))
        m = re.search(r"\|\s+(\d+)\s+\|\s+(pp|tg)\d+\s+\|\s+([\d.]+)", line)        # llama-bench md row
        if m: cur[("prefill" if m.group(2) == "pp" else "decode") + " t" + m.group(1)].append(float(m.group(3)))

def cell(v, nd):
    return f"**{st.median(v):.{nd}f}** ({'/'.join(f'{x:.{nd}f}' for x in v)})" if v else "—"
keys = [k for k in ("prefill", "decode") if any(r.get(k) for r in rows.values())]
extra = sorted({k for r in rows.values() for k in r if " t" in k})
cols = keys + extra
print("| config | " + " | ".join(c + " tok/s" for c in cols) + " | peak RSS MB | load s |")
print("|---" * (len(cols) + 3) + "|")
for lab, r in rows.items():
    rss = f"{st.median(r['rss']):.0f}" if r.get("rss") else "—"
    ld = f"{st.median(r['load']):.1f}" if r.get("load") else "—"
    print(f"| {lab} | " + " | ".join(cell(r.get(c, []), 1 if c.startswith("prefill") else 2) for c in cols) + f" | {rss} | {ld} |")
