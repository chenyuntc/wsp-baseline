# A think cap matches a weight edit on Qwen3.5-0.8B

[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) edits two MLP matrices of Qwen3.5-0.8B
(**p50c**) and reports large gains over the unedited model. That unedited baseline is weak. With thinking on,
72–93% of its samples hit the 32k-token limit before `</think>`, so they never give an answer.

**Change only the decoding: cap the thinking at 8k tokens, cap the answer at 2k, and ask for a `\boxed{}` if it is
missing** (budget forcing, as in [s1](https://arxiv.org/abs/2501.19393)). No weight change, no training, no learned or
tuned prompt. This beats p50c on all three benchmarks:

| model | MATH500 | GSM8K | IFEval |
|---|---:|---:|---:|
| unedited (author's protocol; our run) | 7.9 | 13.2 | 22.6 |
| p50c edit (author's protocol) | 50.3 | 67.4 | 61.9 |
| **unedited + think cap 8k** | **54.6** | **70.7** | **65.4** |

- Test sets: MATH500 ×4 samples, GSM8K ×2, IFEval ×2.
- p50c is averaged over 3 seeds. The think-cap row is averaged over 2 seeds; the third is still running.
- Seed-to-seed spread is up to about 2.5 points.
- The author reports 7.9 / 13.8 / 21.0 for the unedited model and 48.1 / 67.1 / 62.3 for p50c.
- Not sensitive to the cap: a 4k cap gives 53.6 / 70.2 / 66.3.
- With the same 8k cap, p50c scores 57.6 / 70.1 / 61.9. That is ahead only on MATH500, by 3 points.

## Protocols

**think35** is the author's protocol:
- Thinking on.
- Sampling: temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, seed 0.
- At most 32768 new tokens.

Grading:
- Math answers are graded with the author's `mathqa.py` (math-verify).
- IFEval uses prompt-level strict accuracy (lm_eval's checkers) on the text after `</think>`.
- A reply with no `</think>` scores 0 on IFEval. For math, the grader then reads the last `\boxed{}` anywhere in the
  text, so a cut-off reply can still be correct.

**think cap 8k** is think35 plus three fixed decoding rules (configs/runs call it `b8a2`). Sampling and grading are
the same. Two fixed strings are inserted during decoding; no system prompt is added.
1. Thinking is capped at 8192 tokens. If the cap is hit, the text "I am running out of time, so I will stop here and
   give my final answer now." and `</think>` are appended.
2. The answer is capped at 2048 tokens.
3. If a `\boxed{}` question still has no box, "Final answer: $\boxed{" is appended and the model gets 64 more tokens.

The presence-penalty history carries across these continuations, and the total stays under 32k tokens.

All three rules matter for math. Rules 2 and 3 are our additions to s1's budget forcing. On the dev split, rule 1 alone
gave 34.4 on MATH-train problems: after the forced `</think>`, the model often loops again in the answer. Rules 1–3
together gave 52.3. In the test run, about 60% of MATH500 samples needed rule 3. IFEval does not need rules 2–3
(rule 1 alone: 68.8 on the IF dev split). See [`configs/base_b8a2.json`](configs/base_b8a2.json) and
`src/gen.py`.

**p50c** is rebuilt by `scripts/setup.sh`:
- It uses the author's own `apply_edit.py` on their released edit (`exp8_math/runs/p50c/edit`).
- That script checks the sha256 of every rebuilt tensor.
- The edit is sparse: it changes about 65k entries in each of `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj`.

## Reproduce

Requires one 8-GPU node (we used B200s) and a Python env with `requirements.txt`.

```bash
pip install -r requirements.txt
bash scripts/setup.sh            # download Qwen3.5-0.8B, rebuild p50c (hash-checked)
bash scripts/reproduce.sh 1      # p50c and the unedited model, author's protocol
bash scripts/reproduce.sh 2      # unedited + think cap 8k (also p50c + think cap 8k)
```

- `reproduce.sh <exp> "0"` runs one seed instead of three. It prints the seed-averaged table at the end.
- Every run writes `outputs/<run>_s<seed>/`:
  - `shard*.jsonl`: every sample, with its full text, prediction, grade, token count and cut/forced flags.
  - `summary.json`: the run's scores.
- To evaluate a single configuration: `bash scripts/eval.sh <model dir> <config.json> <out dir> [seed]`.

## Limits

- The think cap changes the decoding, so this is not the author's exact protocol. The point is how little needs to
  change: three fixed rules, no weight change, no data and no search. The settings (8k, 2k) were chosen on dev splits.

## Prompts and soft prompts

We also tried prompts and soft prompts that leave the model's weights unchanged. Each used about 3k self-generated
training traces. [`readme_old.md`](readme_old.md) has the details.

| method | weights | decoding | MATH500 | GSM8K | IFEval |
|---|---|---|---:|---:|---:|
| pez9, a 33-token text system prompt found by PEZ | original | think35 | 56.4 | 71.5 | 41.2 |
| sp9, a 32-vector soft prompt, + think cap 8k | original + 32 unused embedding rows | think cap 8k | 62.9 | 73.2 | 66.4 |

They are reproduced with `reproduce.sh 3/4/5`.
