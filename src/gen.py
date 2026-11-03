"""One GPU worker: generate + grade a shard of items under a prompt config.

  python gen.py <model> <items.jsonl> <config.json> <out.jsonl> --shard i/k

Items: {"id", "kind": "math", "problem", "answer"} or {"id", "kind": "ifeval", "prompt", "instruction_id_list", "kwargs"}.
Config: {"system": str|null, "math_user": "...{problem}...", "ifeval_user": "...{prompt}...", "n": {"math": 4, "ifeval": 2}}.\nOptional (beyond prompt-only): "history": json of [{user, think, answer}] few-shot chat turns (+ "history_think": true),\n"budget": thinking-token cap with forced </think>, "force": text inserted before the forced </think>.
Protocol = author's think35: thinking on, T0.6 / top_p 0.95 / top_k 20 / presence 1.5, 32,768 new tokens, seed 0.
"""
import argparse, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mathqa as Q

SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20, "presence_penalty": 1.5, "max_tokens": 32768, "seed": 0}
# presence penalty is applied on the GPU by pp_gpu.PresencePenaltyGPU (same math; vLLM's built-in one is CPU-bound at 32k)
VLLM_SAMPLING = {**{k: v for k, v in SAMPLING.items() if k != "presence_penalty"}, "extra_args": {"pp": SAMPLING["presence_penalty"]}}
DEFAULT = {"system": None, "math_user": "{problem}\n" + Q.INSTRUCTION, "ifeval_user": "{prompt}",
           "n": {"math": 1, "ifeval": 1}}


def response(text, thinking):
    """the answer part: after </think>; a thinking-on reply without </think> has none, a thinking-off reply is all answer"""
    return text.split("</think>")[-1].strip() if "</think>" in text else ("" if thinking else text.strip())


def grade_ifeval(r, text, thinking=True):
    from lm_eval.tasks.ifeval import instructions_registry as IR
    resp = response(text, thinking)
    follow = []
    for iid, kw in zip(r["instruction_id_list"], r["kwargs"]):
        ins = IR.INSTRUCTION_DICT[iid](iid)
        ins.build_description(**{k: v for k, v in (kw or {}).items() if v is not None})
        args = ins.get_instruction_args()
        if args and "prompt" in args:
            ins.build_description(prompt=r["prompt"])
        follow.append(bool(resp) and bool(ins.check_following(resp)))
    return all(follow)


def grade_rlvr(r, text, thinking=True):
    import if_functions as F
    resp = response(text, thinking)
    gt = json.loads(r["gt"])
    f = F.IF_FUNCTIONS_MAP[gt.pop("func_name")]
    return bool(resp) and bool(f(resp, **{k: v for k, v in gt.items() if v is not None}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model"); ap.add_argument("items"); ap.add_argument("config"); ap.add_argument("out")
    ap.add_argument("--shard", default="0/1")
    a = ap.parse_args()
    cfg = {**DEFAULT, **json.load(open(a.config))}
    SAMPLING["seed"] = VLLM_SAMPLING["seed"] = cfg.get("seed", 0)   # resampling only (variance estimate); protocol seed is 0
    i, k = map(int, a.shard.split("/"))
    rows = [json.loads(l) for l in open(a.items)][i::k]
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__)) + ":" + os.environ.get("PYTHONPATH", "")
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", gpu_memory_utilization=0.85, max_model_len=SAMPLING["max_tokens"] + 24576,
              seed=0, limit_mm_per_prompt={"image": 0, "video": 0}, logits_processors=["pp_gpu:PresencePenaltyGPU"],
              async_scheduling=False)   # so each step's output token is known before the next step's penalty
    tok = llm.get_tokenizer()
    prompts = []
    for r in rows:
        user = cfg["math_user"].replace("{problem}", r["problem"]) if r["kind"] == "math" else cfg["ifeval_user"].replace("{prompt}", r["prompt"])  # ifeval and rlvr_if
        # optional few-shot history as earlier chat turns, rendered by hand (the template drops past <think> blocks)
        pre = f"<|im_start|>system\n{cfg['system']}<|im_end|>\n" if cfg["system"] else ""
        for h in json.load(open(cfg["history"])) if cfg.get("history") else []:
            th = f"<think>\n{h['think']}\n</think>\n\n" if cfg.get("history_think") else ""
            pre += f"<|im_start|>user\n{h['user']}<|im_end|>\n<|im_start|>assistant\n{th}{h['answer']}<|im_end|>\n"
        prompts.append(pre + tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False, add_generation_prompt=True,
                                                     enable_thinking=cfg.get("thinking", True)))
    K = cfg.get("soft")   # soft prompt (train_sp.py): the system text "<SOFT>" becomes K padding-row ids, which are never sampled
    if K:
        VLLM_SAMPLING["extra_args"]["ban"] = 248077
    def inp(s):
        if not K:
            return s
        x, y = s.split("<SOFT>", 1)
        return {"prompt_token_ids": tok.encode(x, add_special_tokens=False) + list(range(248100, 248100 + K)) + tok.encode(y, add_special_tokens=False)}
    # group by n (math / ifeval can have different sample counts); outs[j] = [[text, ntok, cut, forced], ...]
    # budget forcing (NOT prompt-only, cfg "budget"): thinking stops at </think> or `budget` tokens; if the budget ran out,
    # cfg "force" + "</think>" is appended and the model writes the answer with the rest of the 32k (penalty history kept)
    B = cfg.get("budget")
    outs = [None] * len(rows)
    ns = [cfg["n"].get(r.get("src"), cfg["n"].get(r["kind"], 1)) for r in rows]   # samples per src, else per kind
    p1 = {**VLLM_SAMPLING, "max_tokens": B, "stop": ["</think>"], "include_stop_str_in_output": True} if B else VLLM_SAMPLING
    for n in sorted(set(ns)):
        idx = [j for j in range(len(rows)) if ns[j] == n]
        res = llm.generate([inp(prompts[j]) for j in idx], SamplingParams(n=n, **p1))
        for j, o in zip(idx, res):
            outs[j] = [[x.text, list(x.token_ids), x.finish_reason == "length", False] for x in o.outputs]
    if B:
        force = cfg.get("force", "\n\nI am running out of time, so I will stop here and give my final answer now.\n")
        flat = [(j, x) for j in range(len(rows)) for x in outs[j]]
        for j, x in flat:
            if not x[0].endswith("</think>"):
                x[0] += force + "</think>"; x[1] += tok.encode(force + "</think>", add_special_tokens=False); x[3] = True
            x[0] += "\n\n"; x[1] += tok.encode("\n\n", add_special_tokens=False)
        def cont(todo, cap):   # continue each (j, x) by up to cap(x) tokens, presence-penalty history kept
            sps = [SamplingParams(n=1, **{**VLLM_SAMPLING, "max_tokens": cap(x), "extra_args": {**VLLM_SAMPLING["extra_args"], "pp_seed": x[1]}})
                   for _, x in todo]
            for (j, x), o in zip(todo, llm.generate([inp(prompts[j] + x[0]) for j, x in todo], sps)):
                x[0] += o.outputs[0].text; x[1] += list(o.outputs[0].token_ids); x[2] = o.outputs[0].finish_reason == "length"
        A = cfg.get("answer_budget")   # optional cap on the answer for prompts that ask for a \boxed answer
        boxq = [j for j in range(len(rows)) if "\\boxed" in prompts[j].split("<|im_start|>user")[-1]]
        cont(flat, lambda x: min(SAMPLING["max_tokens"] - len(x[1]), A) if A else SAMPLING["max_tokens"] - len(x[1]))
        if A:   # answer ran out without a box: append "Final answer: $\boxed{" and let the model fill it in
            todo = [(j, x) for j, x in flat if j in set(boxq) and "\\boxed" not in x[0].split("</think>")[-1]]
            for j, x in todo:
                t = "\n\nFinal answer: $\\boxed{"; x[0] += t; x[1] += tok.encode(t, add_special_tokens=False)
            cont(todo, lambda x: min(64, SAMPLING["max_tokens"] - len(x[1])))
    with open(a.out, "w") as f:
        for r, o in zip(rows, outs):
            for s_i, (text, ids, cut, forced) in enumerate(o):
                if r["kind"] == "math":
                    pred, _ = Q.extract(text)
                    good = Q.is_correct(pred, r["answer"])
                elif r["kind"] == "ifeval":
                    pred, good = None, grade_ifeval(r, text, cfg.get("thinking", True))
                else:
                    pred, good = None, grade_rlvr(r, text, cfg.get("thinking", True))
                f.write(json.dumps({"id": r["id"], "kind": r["kind"], "src": r.get("src"), "s": s_i, "correct": int(bool(good)),
                                    "pred": pred, "cut": cut, "ntok": len(ids), "forced": forced,
                                    "text": text}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
