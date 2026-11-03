#!/bin/bash
# eval.sh <model dir> <config.json> <out dir> [seed=0] [items=data/bench/test_all.jsonl]
# one vLLM worker per visible GPU (CUDA_VISIBLE_DEVICES, default all), each on a shard of the items
set -e
cd "$(dirname "$0")/.."
M=$1; C=$2; O=$3; S=${4:-0}; I=${5:-data/bench/test_all.jsonl}
G=(${CUDA_VISIBLE_DEVICES//,/ }); [ ${#G[@]} -gt 0 ] || G=($(seq 0 $(($(nvidia-smi -L | wc -l) - 1))))
mkdir -p $O
python -c "import json,sys; c=json.load(open('$C')); c['seed']=$S; json.dump(c, open('$O/config.json','w'), ensure_ascii=False)"
for i in "${!G[@]}"; do
  CUDA_VISIBLE_DEVICES=${G[$i]} python src/gen.py $M $I $O/config.json $O/shard$i.jsonl --shard $i/${#G[@]} > $O/log$i.txt 2>&1 &
done
wait
python src/summ.py $O
