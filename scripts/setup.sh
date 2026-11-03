#!/bin/bash
# setup.sh: download Qwen3.5-0.8B, rebuild the author's p50c checkpoint from its released edit, build the sp9 model.
set -e
cd "$(dirname "$0")/.."
REV=2fc06364715b967f1860aea9cf38778875588b17   # the base revision the p50c edit was made on
mkdir -p models
[ -d models/Qwen3.5-0.8B ] || huggingface-cli download Qwen/Qwen3.5-0.8B --revision $REV --local-dir models/Qwen3.5-0.8B
[ -d third_party/lm_understand-logs ] || git clone https://github.com/RikiyaT/lm_understand-logs.git third_party/lm_understand-logs
A=third_party/lm_understand-logs/exp8_math
cp $A/code/mathqa.py src/   # the author's answer extraction / checker, so math grading is identical to theirs

# 1. author's method: p50c = base + the released edit (apply_edit.py checks the sha256 of every rebuilt tensor)
if [ ! -d models/Qwen3.5-0.8B-p50c ]; then
  mkdir -p models/Qwen3.5-0.8B-p50c
  for f in models/Qwen3.5-0.8B/*; do [[ $f == *.safetensors ]] || cp $f models/Qwen3.5-0.8B-p50c/; done
  F=$(basename models/Qwen3.5-0.8B/*.safetensors)
  python $A/code/apply_edit.py models/Qwen3.5-0.8B/$F $A/runs/p50c/edit models/Qwen3.5-0.8B-p50c/$F
fi
# 3/4. ours: base weights + 32 soft-prompt vectors in unused embedding rows 248100-248131
[ -d models/Qwen3.5-0.8B-sp9 ] || python src/soft_model.py models/Qwen3.5-0.8B artifacts/sp9.safetensors models/Qwen3.5-0.8B-sp9
