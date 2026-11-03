# No-edit baselines for Qwen3.5-0.8B weight editing

[RikiyaT/lm_understand-logs](https://github.com/RikiyaT/lm_understand-logs) edits two MLP matrices of Qwen3.5-0.8B
(the **p50c** edit) and reports large gains over the unedited model on MATH500, GSM8K and IFEval. This repo checks how
strong that comparison is. It runs the same model under the same protocol without changing any of its weights.

The unedited baseline (7.9 / 13.2 / 22.6) mostly measures one failure. With thinking on, the model keeps
re-checking itself until it hits the 32k-token limit. 72–93% of its samples never reach `</think>`, so they never get
to an answer. A better prompt, or simply stopping the thinking, recovers most of the gain that the edit gets credit for.

## Results

Test sets: MATH500 (4 samples per problem, n = 2000), GSM8K (2 per problem, n = 2638) and IFEval (2 per prompt, n = 1082).
Each number is accuracy over all samples, averaged over seeds 0/1/2. Seed-to-seed spread is about 1–2 points.

| # | method | weights | decoding | MATH500 | GSM8K | IFEval |
|---|--------|---------|----------|--------:|------:|-------:|
|   | unedited model, default prompt | original | think35 | 7.9 | 13.2 | 22.6 |
| 1 | **author: p50c edit** | 2 tensors edited | think35 | 50.3 | 67.4 | 61.9 |
| 2 | **ours: pez9 text prompt** | original | think35 | **56.4** | **71.5** | 41.2 |
|   | ours: n6 readable prompt (instruction + 16 examples) | original | think35 | 47.0 | 60.4 | 41.3 |
| 3 | **ours: sp9 soft prompt** | original + 32 unused embedding rows | think35 | SP9_M | SP9_G | SP9_I |
| 4 | **ours: sp9 + thinking budget b8a2** | original + 32 unused embedding rows | b8a2 | **62.9** | **73.2** | **66.4** |
|   | control: unedited, no prompt, b8a2 | original | b8a2 | 54.6 | 70.7 | 65.4 |
|   | control: p50c + b8a2 | 2 tensors edited | b8a2 | 57.6 | 70.1 | 61.9 |

Our rebuild of p50c scores 50.3 / 67.4 / 61.9. The author reports 48.1 / 67.1 / 62.3.

Main points:

- **(2) Only the system prompt changes, and math beats the edit.** MATH500 gains 6.1 points and GSM8K 4.1. The other
  PEZ prompts we tested on the test set scored 53.7–56.5 on MATH500 and 69.7–72.2 on GSM8K, all above p50c.
  IFEval does not improve: 38% of IFEval samples still think until the 32k limit and get no answer.
- **(4) Every benchmark beats the edit.** It also wins when p50c gets the same thinking budget.
- **Control: stopping the thinking alone beats p50c on all three.** This uses no prompt and no training.
  The reasoning is mostly already in the model. Of the MATH500 samples that had to be cut at 8k thinking tokens,
  51% are still correct once the model is made to answer.

## What each method changes

**Protocol think35** (the author's, used for everything): thinking on; temperature 0.6, top_p 0.95, top_k 20, presence
penalty 1.5; at most 32768 new tokens; seed 0. Math answers are graded with the author's `mathqa.py` (math-verify).
IFEval uses prompt-level strict accuracy (lm_eval's checkers) on the text after `</think>`. A reply with no `</think>`
has no answer and scores 0.

**1. p50c (author).** `scripts/setup.sh` rebuilds the checkpoint with the author's own `apply_edit.py` from their
released edit (`exp8_math/runs/p50c/edit`). That script checks the sha256 of every rebuilt tensor. The edit is
sparse: it changes about 65k entries in each of `layers.15.mlp.down_proj` and `layers.19.mlp.down_proj`.

**2. pez9 (text prompt).** The model and the decoding are unchanged. The only addition is this 33-token system prompt
([`configs/pez9.json`](configs/pez9.json)):

```
電影 può kW cash $\(board已赞过     <|vision_start|> Google<table</tool_response><|vision_end|><|endoftext|><think>
福州 ვ_labels_alokaUn峰此前
正是在李明 Kirk在不 Let<|vision_end|>
```

It was found with PEZ, a gradient search for a hard prompt.
- 32 trainable vectors sit in the system-prompt slot. Each forward pass uses the nearest real token embedding (by
  cosine) for each vector. The gradient flows straight through to the continuous vectors.
- The model is frozen. The loss is cross-entropy on the model's own correct, naturally finished reasoning traces for
  training-split problems (context distillation).
- Special tokens were allowed. The prompt contains 6 of them, and it relies on them: a search without special tokens
  found nothing useful.
- The prompt is brittle. Appending a normal sentence to it removes the gain.

**3. sp9 (soft prompt).** The same objective, but the 32 vectors stay continuous
([`artifacts/sp9.safetensors`](artifacts/sp9.safetensors), 32 × 1024).
- They are written into embedding rows 248100–248131. The tokenizer never produces these ids (it has 248077 of them;
  the embedding has 248320 rows). Every other weight is bit-identical to the base model.
- The config `{"system": "<SOFT>", "soft": 32}` puts those 32 ids in the system message.
- The embedding is tied to the output head, so ids ≥ 248077 are banned from sampling.
- Initialization: the token embeddings of "You are a careful but efficient problem solver. Keep your thinking brief
  and focused. Verify your answer at most once. As soon as you are confident, stop thinking and give the final answer."
- Training: Adam, lr 3e-3, 2 epochs, 3318 traces.

**4. b8a2 (thinking budget).** This changes the decoding, not the prompt.
- Thinking is capped at 8192 tokens. If the cap is hit, the text "I am running out of time, so I will stop here and
  give my final answer now." and `</think>` are appended.
- The answer is capped at 2048 tokens.
- If a `\boxed{}` question still has no box, "Final answer: $\boxed{" is appended and the model gets 64 more tokens.
- The presence-penalty history carries across these continuations. Total length stays under 32k.

**Training data** (methods 2 and 3, shipped in `data/train/`):
- The unedited model solved problems from the MATH and GSM8K train splits and from allenai/RLVR-IFeval.
  - Math problems used the n6 prompt plus an 8k thinking budget.
  - IF prompts used a 4k budget and no prompt.
- Kept: traces that are correct, end with `</think>` on their own, and are not forced.
- Totals: 3318 rows for sp9 (traces ≤ 12k tokens) and 3009 for pez9 (≤ 6k).
- None of these problems is an exact duplicate of a test or dev problem. Candidates were selected on held-out dev splits (`data/bench/*_devB.jsonl`).

## Reproduce

Requires one 8-GPU node (we used B200s) and a Python env with `requirements.txt`.

```bash
pip install -r requirements.txt
bash scripts/setup.sh            # download Qwen3.5-0.8B, rebuild p50c (hash-checked), build the sp9 model
bash scripts/reproduce.sh 1      # author's p50c (+ unedited model)
bash scripts/reproduce.sh 2      # pez9 text prompt (+ readable n6 prompt)
bash scripts/reproduce.sh 3      # sp9 soft prompt, think35
bash scripts/reproduce.sh 4      # sp9 + b8a2 (+ b8a2 controls on the unedited model and p50c)
bash scripts/reproduce.sh all    # everything; prints the seed-averaged table
```

`reproduce.sh <exp> "0"` runs one seed instead of three. Every run writes `outputs/<run>_s<seed>/`:
- `shard*.jsonl`: every sample, with its full text, prediction, grade, token count and cut/forced flags.
- `summary.json`: the run's scores.

To evaluate a single configuration: `bash scripts/eval.sh <model dir> <config.json> <out dir> [seed]`.

Optional, to retrain instead of using the shipped artifacts:

```bash
bash scripts/mine.sh             # regenerate the training traces (else data/train/*.jsonl.gz is used)
bash scripts/train.sh sp9        # -> models/Qwen3.5-0.8B-sp9_retrained
bash scripts/train.sh pez9       # -> configs/pez9_retrained.json
```

Retraining sp9 should land close to the shipped vectors. PEZ depends heavily on the seed. A rerun gives a different
string, and most strings we found beat p50c on math, though not all of them.

## Limits

- "Better prompt beats the edit" holds for math only. Text prompts did not beat p50c on IFEval. The win on all three
  benchmarks needs continuous vectors (sp9) plus a decoding change (b8a2). Neither is plain prompting, but both leave
  the model's weights unchanged.
- Prompts help p50c too. p50c with the n6 prompt scores 64.2 / 71.8 / 47.7 (one seed). A fair comparison holds the
  prompt and the decoding fixed across arms. Under matched b8a2 decoding, sp9 still leads.
- pez9 and sp9 both use about 3k self-generated traces on training-split problems. They are not zero-data prompt
  rewrites.

## Layout

```
src/gen.py           generation + grading (vLLM); soft prompts and thinking budgets are config options
src/pp_gpu.py        presence penalty on the GPU (same math as vLLM's; the built-in one is CPU-bound at 32k tokens)
src/train_sp.py      soft-prompt / PEZ training (frozen model, context distillation)
src/soft_model.py    write soft vectors into a copy of the base model
src/sp_data.py       mined traces -> training rows
src/if_functions.py  RLVR-IF checkers (from allenai/open-instruct)
src/mathqa.py        copied from the author's repo by setup.sh
configs/             one JSON per method (system prompt, soft prompt, budget, samples per benchmark)
data/bench/          test set (MATH500 + GSM8K test + IFEval) and the dev splits used for selection
data/mine/           train-split problems used for mining
data/train/          the mined training rows used for pez9 / sp9
artifacts/           sp9 vectors
```
