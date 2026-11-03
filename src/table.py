"""Mean over seeds of each run group: python table.py <outputs dir>  (run dirs are named <group>_s<seed>)"""
import glob, json, os, re, statistics as st, sys
from collections import defaultdict

g = defaultdict(list)
for d in sorted(glob.glob(os.path.join(sys.argv[1], "*_s[0-9]*"))):
    rs = [json.loads(l) for p in glob.glob(f"{d}/shard*.jsonl") for l in open(p)]
    if rs:
        g[re.sub(r"_s\d+$", "", os.path.basename(d))].append(rs)
print(f"{'run':28s} {'seeds':>5s} {'MATH500':>8s} {'GSM8K':>8s} {'IFEval':>8s}")
for k, runs in g.items():
    acc = [st.mean(100 * st.mean(r["correct"] for r in rs if r["src"] == b) for rs in runs) for b in ["math500", "gsm8k", "ifeval"]]
    print(f"{k:28s} {len(runs):5d} " + " ".join(f"{a:8.1f}" for a in acc))
