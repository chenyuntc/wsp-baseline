#!/bin/bash
# mine.sh (optional): regenerate the training traces instead of using data/train/*.jsonl.gz.
# The unedited model answers train-split problems (math: n6 prompt + 8k thinking budget; IF: 4k budget), one slice per GPU;
# sp_data.py keeps the correct, naturally finished traces. Sampling is not bit-exact across machines, so the rows differ
# slightly from the shipped ones.
set -e
cd "$(dirname "$0")/.."
B=models/Qwen3.5-0.8B; O=outputs/mine; mkdir -p $O
g=0
for s in math0 math1 math2 gsm0 gsm1 if0 if1; do
  c=configs/mine_math.json; [[ $s == if* ]] && c=configs/mine_if.json
  CUDA_VISIBLE_DEVICES=$g python src/gen.py $B data/mine/$s.jsonl $c $O/out_$s.jsonl > $O/log_$s.txt 2>&1 &
  g=$((g + 1))
done
wait
python src/sp_data.py data/train/d_long.jsonl 12000 data/mine $O/out_*.jsonl   # sp9 data (traces <= 12k tokens)
python src/sp_data.py data/train/d_all.jsonl 6000 data/mine $O/out_*.jsonl     # pez9 data (traces <= 6k tokens)
