"""Training rows from mined self-traces: kept if correct, thinking ended on its own (no cut, no forcing), <= maxtok.

  python sp_data.py <out.jsonl> <maxtok> <items dir> <mined out_*.jsonl ...>

The user message is rebuilt from the mined items (every *.jsonl in <items dir>).
"""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mathqa as Q

items = {}
for f in glob.glob(os.path.join(sys.argv[3], "*.jsonl")):
    for l in open(f):
        r = json.loads(l); items[r["id"]] = r
out, maxtok = [], int(sys.argv[2])
for p in sys.argv[4:]:
    for l in open(p):
        r = json.loads(l)
        if r["correct"] and not r["cut"] and not r.get("forced") and "</think>" in r["text"] and r["ntok"] <= maxtok:
            it = items[r["id"]]
            out.append({"user": it["problem"] + "\n" + Q.INSTRUCTION if it["kind"] == "math" else it["prompt"], "target": r["text"], "src": r["src"]})
open(sys.argv[1], "w").writelines(json.dumps(o) + "\n" for o in out)
print(sys.argv[1], len(out), {s: sum(o["src"] == s for o in out) for s in {o["src"] for o in out}})
