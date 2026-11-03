"""Pull the K trained vectors out of a train_sp.py model copy: python export_soft.py <model dir> <out.safetensors> [K]"""
import glob, sys
from safetensors.torch import load_file, save_file

K = int(sys.argv[3]) if len(sys.argv) > 3 else 32
w = {}
for f in glob.glob(f"{sys.argv[1]}/*.safetensors"):
    w.update(load_file(f))
save_file({"soft": w["model.language_model.embed_tokens.weight"][248100:248100 + K].contiguous()}, sys.argv[2])
