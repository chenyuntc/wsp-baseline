"""Soft-prompt tuning (NOT prompt-only; base weights untouched): learn K input vectors placed as the system message.

  python train_sp.py <model> <out_model_dir> <data.jsonl> [--k 32] [--epochs 2] [--lr 3e-3] [--init "text"]

data.jsonl rows: {"user": the user message (problem + instruction), "target": the model's own reply after "<think>\n"}.
Loss = next-token CE on the target, with [system: K soft vectors] + user before it (context distillation of the
model's own short correct traces). The vectors are written into K unused padding rows of the embedding
(ids SOFT0..SOFT0+K-1, beyond the tokenizer); gen.py cfg {"soft": K} puts those ids in the system message and bans
them from being sampled (the embedding is tied to the output head).
"""
import argparse, json, math, os, random, shutil
import torch
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration

SOFT0 = 248100   # first padding row used (tokenizer has 248077 ids, embedding 248320 rows)
EMB = "model.language_model.embed_tokens.weight"

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("out"); ap.add_argument("data")
ap.add_argument("--k", type=int, default=32); ap.add_argument("--epochs", type=float, default=2); ap.add_argument("--lr", type=float, default=3e-3)
ap.add_argument("--accum", type=int, default=8); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--pez", action="store_true"); ap.add_argument("--pez_vocab", type=int, default=248044); ap.add_argument("--w_end", type=float, default=1); ap.add_argument("--init", default="You are a careful but efficient problem solver. Keep your thinking brief and focused. Verify your answer at most once. As soon as you are confident, stop thinking and give the final answer.")
a = ap.parse_args()

tok = AutoTokenizer.from_pretrained(a.model)
model = Qwen3_5ForConditionalGeneration.from_pretrained(a.model, dtype=torch.bfloat16).cuda()
model.requires_grad_(False); model.gradient_checkpointing_enable(); model.config.use_cache = False
E = model.get_input_embeddings()
init = tok.encode(a.init, add_special_tokens=False)
init = (init * (a.k // len(init) + 1))[:a.k]
P = torch.nn.Parameter(E.weight[torch.tensor(init, device="cuda")].float().clone())
opt = torch.optim.Adam([P], lr=a.lr)

rows = [json.loads(l) for l in open(a.data)]
pre = tok.encode("<|im_start|>system\n", add_special_tokens=False)
def example(r):
    q = tok.apply_chat_template([{"role": "user", "content": r["user"]}], tokenize=False, add_generation_prompt=True, enable_thinking=True)
    mid = tok.encode("<|im_end|>\n" + q, add_special_tokens=False)
    tgt = tok.encode(r["target"] + "<|im_end|>", add_special_tokens=False)
    return mid, tgt
data = [example(r) for r in rows]
END = tok.convert_tokens_to_ids("</think>")   # --w_end: extra loss weight on the </think> token (learn to stop thinking)
steps = math.ceil(len(data) * a.epochs / a.accum)
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / 10) * 0.5 * (1 + math.cos(math.pi * min(s, steps) / steps)))
print(f"{len(data)} examples, {steps} steps, mean target {sum(len(t) for _, t in data) / len(data):.0f} tok", flush=True)

V = E.weight[:a.pez_vocab]   # 248044: real tokens only; 248077: also special tokens
def nearest():
    with torch.no_grad():
        return (torch.nn.functional.normalize(P.to(V.dtype), dim=1) @ torch.nn.functional.normalize(V, dim=1).T).argmax(1)
rng = random.Random(a.seed)
order = []
while len(order) < steps * a.accum:
    o = list(range(len(data))); rng.shuffle(o); order += o
for s in range(steps):
    tot = ntok = 0
    for i in order[s * a.accum:(s + 1) * a.accum]:
        mid, tgt = data[i]
        ids = torch.tensor(pre + [0] * a.k + mid + tgt, device="cuda")
        x = E(ids)[None]
        Q = P
        if a.pez:   # PEZ: forward with the nearest real tokens, gradient straight through to P -> a hard (text) prompt
            Q = P + (E.weight[nearest()].float() - P).detach()
        x = torch.cat([x[:, :len(pre)], Q.to(x.dtype)[None], x[:, len(pre) + a.k:]], 1)
        h = model.model(inputs_embeds=x).last_hidden_state[0, -len(tgt) - 1:-1]   # positions that predict the target
        y = ids[-len(tgt):]
        loss = sum(torch.utils.checkpoint.checkpoint(lambda hh, yy: (torch.nn.functional.cross_entropy(model.lm_head(hh).float(), yy, reduction="none") * (1 + (a.w_end - 1) * (yy == END))).sum(),
                                                     h[c:c + 1024], y[c:c + 1024], use_reentrant=False) for c in range(0, len(y), 1024))
        (loss / (a.accum * 1000)).backward()
        tot += loss.item(); ntok += len(tgt)
    torch.nn.utils.clip_grad_norm_([P], 1.0)
    opt.step(); sched.step(); opt.zero_grad()
    if s % 10 == 0 or s == steps - 1:
        if a.pez:
            print(repr(tok.decode(nearest())), flush=True)
        print(f"step {s}/{steps} loss/tok {tot / ntok:.4f} |P| {P.norm(dim=1).mean():.3f} |E| {E.weight[:248077:997].float().norm(dim=1).mean():.3f}", flush=True)

# write the model copy: same files, embedding rows SOFT0.. = P
os.makedirs(a.out, exist_ok=True)
for f in os.listdir(a.model):
    if not f.endswith(".safetensors") and os.path.isfile(os.path.join(a.model, f)):
        shutil.copy(os.path.join(a.model, f), a.out)
for f in [f for f in os.listdir(a.model) if f.endswith(".safetensors")]:
    w = load_file(os.path.join(a.model, f))
    if EMB in w:
        w[EMB][SOFT0:SOFT0 + a.k] = P.detach().to(w[EMB].dtype).cpu()
    save_file(w, os.path.join(a.out, f), metadata={"format": "pt"})
if a.pez:
    json.dump({"system": tok.decode(nearest())}, open(os.path.join(a.out, "pez_prompt.json"), "w"), ensure_ascii=False)
json.dump({"k": a.k, "soft0": SOFT0, "data": a.data, "epochs": a.epochs, "lr": a.lr}, open(os.path.join(a.out, "soft.json"), "w"))
print("saved", a.out)
