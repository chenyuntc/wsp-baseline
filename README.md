# A thinking budget matches a weight edit on Qwen3.5-0.8B

[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) edits two MLP matrices of Qwen3.5-0.8B
(the **p50c** edit) and reports large gains over the unedited model on MATH500, GSM8K and IFEval. This repo shows that
the unedited baseline in that comparison is weak.

The unedited baseline scores 7.9 / 13.2 / 22.6, and that mostly measures one failure. With thinking on, the model keeps
re-checking itself until it hits the 32k-token limit. 72–93% of its samples never reach `</think>`, so they never get
to an answer.

**Fix that failure and nothing else: cap the thinking at 8192 tokens, then make the model answer.** No prompt, no
training, no weight change. The unedited model then scores **54.6 / 70.7 / 65.4**, which beats p50c
(50.3 / 67.4 / 61.9) on all three benchmarks.

The cap is a standard test-time technique: budget forcing from
[s1 (Muennighoff et al. 2025)](https://arxiv.org/abs/2501.19393). Its one setting is not sensitive. A 4096-token cap
gives 53.6 / 70.2 / 66.3 (one seed).

## Results

Test sets:
- MATH500: 4 samples per problem, n = 2000.
- GSM8K: 2 samples per problem, n = 2638.
- IFEval: 2 samples per prompt, n = 1082.

Each number is accuracy over all samples, averaged over seeds. Seed-to-seed spread is about 1–2 points.

| # | model | weights | decoding | seeds | MATH500 | GSM8K | IFEval |
|---|-------|---------|----------|------:|--------:|------:|-------:|
|   | unedited | original | think35 | 1 | 7.9 | 13.2 | 22.6 |
| 1 | **author: p50c edit** | 2 tensors edited | think35 | 3 | 50.3 | 67.4 | 61.9 |
| 2 | **unedited + thinking budget b8a2** | original | b8a2 | 2 | **54.6** | **70.7** | **65.4** |
|   | control: p50c + b8a2 | 2 tensors edited | b8a2 | 3 | 57.6 | 70.1 | 61.9 |

The third seed for row 2 is still running.

Our rebuild of p50c scores 50.3 / 67.4 / 61.9. The author reports 48.1 / 67.1 / 62.3.

**Most of the reasoning is already in the unedited model.** In seed 0 of row 2:
- 94% of MATH500 samples hit the 8k cap.
- 51% of those capped samples are still correct once the model is made to answer.
- The 6% that stopped on their own are all correct.

So the gap between the unedited baseline (7.9) and p50c (50.3) is mostly about stopping, not about solving.

**The control row needs to be read alongside row 2.** With the same budget, p50c still leads on MATH500 (57.6 vs 54.6).
It gains nothing on GSM8K (70.1 vs 70.7) and loses on IFEval (61.9 vs 65.4). So the edit's reported gain mostly comes
from fixing the failure to stop. Whatever the edit adds beyond that is a few points of math.

## Protocols

**think35** is the author's protocol:
- Thinking on.
- Sampling: temperature 0.6, top_p 0.95, top_k 20, presence penalty 1.5, seed 0.
- At most 32768 new tokens.

Grading:
- Math answers are graded with the author's `mathqa.py` (math-verify).
- IFEval uses prompt-level strict accuracy (lm_eval's checkers) on the text after `</think>`.
- A reply with no `</think>` has no answer and scores 0.

**b8a2** is think35 plus a thinking budget. Sampling and grading are the same.
1. Thinking is capped at 8192 tokens. If the cap is hit, the text "I am running out of time, so I will stop here and
   give my final answer now." and `</think>` are appended.
2. The answer is capped at 2048 tokens.
3. If a `\boxed{}` question still has no box, "Final answer: $\boxed{" is appended and the model gets 64 more tokens.

The presence-penalty history carries across these continuations, and the total stays under 32k tokens. Steps 2 and 3
are small additions to s1's budget forcing. See [`configs/base_b8a2.json`](configs/base_b8a2.json) and
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
bash scripts/reproduce.sh 1      # row 1: author's p50c (+ unedited model)
bash scripts/reproduce.sh 2      # row 2: unedited model + b8a2 (+ p50c + b8a2 control)
```

- `reproduce.sh <exp> "0"` runs one seed instead of three. It prints the seed-averaged table at the end.
- Every run writes `outputs/<run>_s<seed>/`:
  - `shard*.jsonl`: every sample, with its full text, prediction, grade, token count and cut/forced flags.
  - `summary.json`: the run's scores.
- To evaluate a single configuration: `bash scripts/eval.sh <model dir> <config.json> <out dir> [seed]`.

## Limits

- b8a2 changes the decoding, so this is not the author's exact protocol. The point is how little needs to change:
  one setting, with no prompt, no data and no search.
- p50c with the same budget still leads on MATH500 by about 3 points.

## Prompts and soft prompts

We also tried prompts and soft prompts that leave the model's weights unchanged. Each used about 3k self-generated
training traces. [`readme_old.md`](readme_old.md) has the details.

| method | weights | decoding | MATH500 | GSM8K | IFEval |
|---|---|---|---:|---:|---:|
| pez9, a 33-token text system prompt found by PEZ | original | think35 | 56.4 | 71.5 | 41.2 |
| sp9, a 32-vector soft prompt, + b8a2 | original + 32 unused embedding rows | b8a2 | 62.9 | 73.2 | 66.4 |

They are reproduced with `reproduce.sh 3/4/5`.
