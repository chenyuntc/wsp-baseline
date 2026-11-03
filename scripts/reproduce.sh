#!/bin/bash
# reproduce.sh <1|2|3|4|5|all> [seeds="0 1 2"]   -> outputs/<run>_s<seed>/, then the seed-averaged table
#   1  author's method: p50c (and the unedited model) under the author's think35 protocol
#   2  simplest baseline: the unedited model, no prompt, thinking budget b8a2 (and p50c with the same budget)
#   3  ours, best text prompt: pez9 system prompt (and the readable n6 prompt) on the unedited model
#   4  ours, best soft prompt: sp9 on the unedited model, think35 protocol
#   5  ours, best result: sp9 + thinking budget b8a2
set -e
cd "$(dirname "$0")/.."
B=models/Qwen3.5-0.8B; E=models/Qwen3.5-0.8B-p50c; SP=models/Qwen3.5-0.8B-sp9
SEEDS=${2:-"0 1 2"}
run() { for s in $SEEDS; do [ -f outputs/$1_s$s/summary.json ] || bash scripts/eval.sh $2 configs/$3.json outputs/$1_s$s $s; done; }
case $1 in 1|all) run p50c $E base; run unedited $B base;; esac
case $1 in 2|all) run unedited_b8a2 $B base_b8a2; run p50c_b8a2 $E base_b8a2;; esac
case $1 in 3|all) run unedited_pez9 $B pez9; run unedited_n6 $B n6;; esac
case $1 in 4|all) run unedited_sp9 $SP sp9;; esac
case $1 in 5|all) run unedited_sp9_b8a2 $SP sp9_b8a2;; esac
python src/table.py outputs
