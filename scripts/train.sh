#!/bin/bash
# train.sh <sp9|pez9> (optional): retrain the soft prompt or the PEZ text prompt on one GPU (the base model stays frozen).
#   sp9  -> artifacts/sp9_retrained.safetensors and models/Qwen3.5-0.8B-sp9_retrained
#   pez9 -> configs/pez9_retrained.json (the learned text as the system prompt)
# PEZ is sensitive to the seed: a rerun gives a different string; most runs we tried beat p50c on math, not all.
set -e
cd "$(dirname "$0")/.."
B=models/Qwen3.5-0.8B; mkdir -p outputs/train
for f in d_long d_all; do [ -f data/train/$f.jsonl ] || gunzip -k data/train/$f.jsonl.gz; done
case $1 in
sp9)
  python src/train_sp.py $B outputs/train/m_sp9 data/train/d_long.jsonl --k 32 --epochs 2 --lr 3e-3 --seed 0 | tee outputs/train/sp9.log
  python src/export_soft.py outputs/train/m_sp9 artifacts/sp9_retrained.safetensors 32
  python src/soft_model.py $B artifacts/sp9_retrained.safetensors models/Qwen3.5-0.8B-sp9_retrained;;
pez9)
  python src/train_sp.py $B outputs/train/m_pez9 data/train/d_all.jsonl --k 32 --epochs 2 --lr 1e-2 --pez --pez_vocab 248077 --seed 11 | tee outputs/train/pez9.log
  python -c "import json; c=json.load(open('configs/pez9.json')); c['system']=json.load(open('outputs/train/m_pez9/pez_prompt.json'))['system']; json.dump(c, open('configs/pez9_retrained.json','w'), ensure_ascii=False, indent=1)";;
esac
