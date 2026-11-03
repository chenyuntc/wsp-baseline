"""GPU presence penalty for vLLM v1 (same math as vLLM's built-in: logits -= p * [token appeared in the OUTPUT]).

vLLM 0.26's built-in penalty rebuilds a padded [batch, max_output_len] tensor on CPU every step, which is quadratic for
32k-token generations (GPU sat at 0 %). Here a [max_reqs, vocab] bool mask lives on the GPU and only each step's new
tokens are scattered in. Enabled per request by SamplingParams(extra_args={"pp": 1.5}) with presence_penalty=0.
"""
import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor
from vllm.v1.sample.logits_processor.interface import MoveDirectionality


class PresencePenaltyGPU(LogitsProcessor):
    def __init__(self, vllm_config, device, is_pin_memory):
        n = vllm_config.scheduler_config.max_num_seqs
        v = vllm_config.model_config.get_vocab_size()
        self.device = device
        self.mask = torch.zeros(n, v, dtype=torch.bool, device=device)
        self.pen = torch.zeros(n, dtype=torch.float32, device=device)
        self.state = {}   # row -> [penalty, output_tok_ids (live list), number of tokens already in the mask]

    def is_argmax_invariant(self):
        return False

    def update_state(self, bu: BatchUpdate | None):
        if bu:
            for i in bu.removed:
                if self.state.pop(i, None) is not None:
                    self.pen[i] = 0
            for i, params, _, out in bu.added:
                p = float((params.extra_args or {}).get("pp", 0.0))
                self.mask[i] = False
                self.pen[i] = p
                if p:
                    self.state[i] = [p, out, 0, (params.extra_args or {}).get("ban")]   # ban: ids >= this are never sampled
                    seed = (params.extra_args or {}).get("pp_seed")   # tokens generated earlier (budget-forcing continuation)
                    if seed:
                        self.mask[i, torch.tensor(seed, device=self.device)] = True
                else:
                    self.state.pop(i, None)
            for a, b, d in bu.moved:
                sa, sb = self.state.pop(a, None), self.state.pop(b, None)
                if d == MoveDirectionality.SWAP:
                    self.mask[[a, b]] = self.mask[[b, a]]
                    self.pen[[a, b]] = self.pen[[b, a]]
                    if sb is not None:
                        self.state[a] = sb
                else:
                    self.mask[b] = self.mask[a]
                    self.pen[b] = self.pen[a]
                    self.mask[a] = False
                    self.pen[a] = 0
                if sa is not None:
                    self.state[b] = sa
        rows, toks = [], []
        for i, s in self.state.items():
            out = s[1]
            if len(out) > s[2]:
                new = out[s[2]:]
                if -1 in new:            # async scheduling placeholder for a not-yet-known token: stop there
                    new = new[:new.index(-1)]
                rows += [i] * len(new)
                toks += new
                s[2] += len(new)
        if rows:
            self.mask[torch.tensor(rows, device=self.device), torch.tensor(toks, device=self.device)] = True

    def apply(self, logits):
        if not self.state:
            return logits
        b = logits.shape[0]
        logits -= self.pen[:b, None] * self.mask[:b]
        ban = [(i, s[3]) for i, s in self.state.items() if s[3] and i < b]
        if ban:   # soft-prompt rows of the tied embedding must not be sampled
            logits[torch.tensor([i for i, _ in ban], device=self.device), ban[0][1]:] = float("-inf")
        return logits
