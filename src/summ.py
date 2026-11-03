"""summ.py <outdir> [<outdir> ...]: accuracy / cut share / mean tokens per source, plus accuracy on uncut samples."""
import glob, json, sys
from collections import defaultdict

for d in sys.argv[1:]:
    g = defaultdict(list)
    for p in sorted(glob.glob(f"{d}/shard*.jsonl")):
        for l in open(p):
            r = json.loads(l)
            g[r.get("src") or r["kind"]].append(r)
    res = {}
    for k, rs in sorted(g.items()):
        unc = [r for r in rs if not r["cut"]]
        res[k] = {"n": len(rs), "acc": round(100 * sum(r["correct"] for r in rs) / len(rs), 1),
                  "cut": round(100 * sum(r["cut"] for r in rs) / len(rs), 1),
                  "tok": round(sum(r["ntok"] for r in rs) / len(rs)),
                  "acc_uncut": round(100 * sum(r["correct"] for r in unc) / max(1, len(unc)), 1)}
    json.dump(res, open(f"{d}/summary.json", "w"), indent=1)
    print(d, json.dumps(res))
