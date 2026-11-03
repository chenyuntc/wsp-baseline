"""Write a model copy whose unused embedding rows 248100.. hold the soft-prompt vectors (all other weights unchanged).

  python soft_model.py <base model dir> <vectors.safetensors> <out dir>
"""
import os, shutil, sys
from safetensors.torch import load_file, save_file

base, vec, out = sys.argv[1:4]
EMB, SOFT0 = "model.language_model.embed_tokens.weight", 248100
v = load_file(vec)["soft"]
os.makedirs(out, exist_ok=True)
for f in os.listdir(base):
    p = os.path.join(base, f)
    if not os.path.isfile(p):
        continue
    if not f.endswith(".safetensors"):
        shutil.copy(p, out)
        continue
    w = load_file(p)
    if EMB in w:
        w[EMB][SOFT0:SOFT0 + len(v)] = v.to(w[EMB].dtype)
    save_file(w, os.path.join(out, f), metadata={"format": "pt"})
print(f"{len(v)} soft vectors -> {out}")
