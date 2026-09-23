"""Parse run_matrix.sh logs -> a markdown table, median over repetitions.

  parse_table.py <log>

A run starts with a line "== <moty|llama.cpp> <model> <config> THREADS=<n>
[rep=<k>] [| ...]"; it is followed by the moty banner ("load X.Xs"), the
gen_turn line ("prefill N tok in ... (X tok/s) | decode ...") or llama-bench
CSV rows, and rssrun's "peak_rss_mb=... wall_s=... status=...". Runs with the
same (engine, model, config, threads) are aggregated."""
import re, sys, statistics as st, csv, io
rows = {}
cur = None
for line in open(sys.argv[1], encoding="utf-8", errors="replace"):
    line = line.rstrip("\n")
    line = re.sub(r"\s*\|\s*MemAvailable.*$", "", line) if line.startswith("==") else line
    m = re.match(r"== (moty|llama\.cpp) (\S+) (.*?)(?: rep=\d+)?$", line)
    if m:
        eng, model, rest = m.groups()
        rest = re.sub(r"threads=", "THREADS=", rest).replace("t=", "THREADS=")
        th = re.search(r"THREADS=(\d+)", rest); th = th.group(1) if th else "?"
        cfg = re.sub(r"\s*THREADS=\d+", "", rest).strip()
        cur = rows.setdefault((eng, model, cfg, int(th)), {"pp": [], "tg": [], "rss": [], "load": [], "status": [], "wall": []})
        continue
    if cur is None: continue
    if (m := re.match(r"load ([0-9.]+)s", line)): cur["load"].append(float(m.group(1)))
    if (m := re.search(r"prefill \d+ tok in [0-9.]+s \(([0-9.]+) tok/s\) \| decode \d+ tok in [0-9.]+s \(([0-9.]+) tok/s\)", line)):
        cur["pp"].append(float(m.group(1))); cur["tg"].append(float(m.group(2)))
    if (m := re.search(r"peak_rss_mb=([0-9.]+) wall_s=([0-9.]+) status=(\d+)", line)):
        cur["rss"].append(float(m.group(1))); cur["status"].append(int(m.group(3))); cur["wall"].append(float(m.group(2)))
    if line.startswith('"'):
        v = next(csv.reader(io.StringIO(line)))
        npr, ngen, ts = int(v[-8]), int(v[-7]), float(v[-2])
        (cur["pp"] if npr > 0 else cur["tg"]).append(ts)
def med(x): return f"{st.median(x):.2f}" if x else "—"
print("| engine | model | config | threads | prefill tok/s | decode tok/s | peak RSS MB | load s | wall s | runs (pp values) |")
print("|---|---|---|---|---|---|---|---|---|---|")
for (eng, model, cfg, th), r in sorted(rows.items(), key=lambda k: (k[0][1], k[0][0], k[0][2], k[0][3])):
    bad = [s for s in r["status"] if s != 0]
    print(f"| {eng} | {model} | {cfg} | {th} | {med(r['pp'])} | {med(r['tg'])} | {max(r['rss']) if r['rss'] else '—'} | "
          f"{med(r['load'])} | {med(r['wall'])} | {len(r['pp'])} ({', '.join(f'{p:.1f}' for p in r['pp'])}){' FAILED status '+str(bad) if bad else ''} |")
