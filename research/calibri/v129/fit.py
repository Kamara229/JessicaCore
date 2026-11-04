"""Matched TRAIN-only fit for the two physical placements with one frozen controller."""
import hashlib
import random
import torch
from torch.nn import functional as F
from v108.generation import build_chat
from v118.archive import pack as old_pack
from v119.native import build, install, restore
from .archive import pack
from .operators import installed
from .protocol import TRAIN, VALID, SEED, LR


def fit(model, tokenizer, policy, source, out, moved, native_dir):
    if {q for q, _ in TRAIN} & {q for q, _ in VALID}:
        raise ValueError("overlapping train/validation questions")
    torch.manual_seed(SEED)
    model.requires_grad_(False)
    fw, bw = build(native_dir)
    old = install(model, fw, bw)
    order = list(TRAIN)
    random.Random(SEED).shuffle(order)
    try:
        with installed(model, source, moved, training=True) as bridge:
            params = [p for p in model.parameters() if p.requires_grad]
            optimizer = torch.optim.Adam(params, lr=LR)
            for question, answer in order:
                bridge.control(policy, question)
                prompt = tokenizer.encode(build_chat(tokenizer, question), add_special_tokens=False)
                target = tokenizer.encode(answer, add_special_tokens=False) + [tokenizer.eos_token_id]
                seq = torch.tensor([prompt + target[:-1]])
                logits = model(input_ids=seq, use_cache=False).logits[0, len(prompt)-1:]
                loss = F.cross_entropy(logits, torch.tensor(target))
                if not torch.isfinite(loss):
                    raise ValueError("nonfinite supervised loss")
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1., error_if_nonfinite=True)
                optimizer.step()
            blocks, head = bridge.factors()
    finally:
        restore(model, old)
    if not all(u.any() and v.any() for u, v in blocks.values()):
        raise ValueError("some paid body factors were not trained")
    archive = pack(source.base, blocks, head) if moved else old_pack(source.base, head, blocks)
    if len(archive) > len(source.raw):
        raise ValueError("candidate exceeds the frozen v119 byte budget")
    dest = out / ("relocated.cb129" if moved else "control.cb118")
    dest.write_bytes(archive)
    return hashlib.sha256(archive).hexdigest()
